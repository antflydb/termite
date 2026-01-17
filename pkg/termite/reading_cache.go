// Copyright 2025 Antfly, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package termite

import (
	"context"
	"encoding/binary"
	"image"
	"image/jpeg"
	"sync/atomic"
	"time"

	"github.com/antflydb/termite/pkg/termite/lib/reading"
	"github.com/cespare/xxhash/v2"
	"github.com/jellydator/ttlcache/v3"
	"go.uber.org/zap"
	"golang.org/x/sync/singleflight"
)

// ReadingCacheTTL is the default TTL for cached reading results
const ReadingCacheTTL = 5 * time.Minute

// CachedReader wraps a reader with caching support
type CachedReader struct {
	reader  reading.Reader
	model   string
	cache   *ttlcache.Cache[string, []reading.Result]
	sfGroup *singleflight.Group
	logger  *zap.Logger

	// Metrics
	hits   atomic.Uint64
	misses atomic.Uint64
	sfHits atomic.Uint64
}

// NewCachedReader wraps a reader with caching
func NewCachedReader(
	reader reading.Reader,
	model string,
	cache *ttlcache.Cache[string, []reading.Result],
	logger *zap.Logger,
) *CachedReader {
	return &CachedReader{
		reader:  reader,
		model:   model,
		cache:   cache,
		sfGroup: &singleflight.Group{},
		logger:  logger,
	}
}

// Read extracts text from images with caching support
func (c *CachedReader) Read(ctx context.Context, images []image.Image, prompt string, maxTokens int) ([]reading.Result, error) {
	// Generate cache key from model + images + prompt + maxTokens
	key := c.cacheKey(images, prompt, maxTokens)

	// Check cache first
	if item := c.cache.Get(key); item != nil {
		c.hits.Add(1)
		RecordCacheHit("reading")
		c.logger.Debug("Reading cache hit",
			zap.String("model", c.model),
			zap.Int("num_images", len(images)))
		return item.Value(), nil
	}

	// Use singleflight to deduplicate concurrent identical requests
	result, err, shared := c.sfGroup.Do(key, func() (any, error) {
		c.misses.Add(1)
		RecordCacheMiss("reading")

		start := time.Now()
		results, err := c.reader.Read(ctx, images, prompt, maxTokens)
		if err != nil {
			return nil, err
		}

		// Record duration
		RecordRequestDuration("read", c.model, "200", time.Since(start).Seconds())

		// Store in cache
		c.cache.Set(key, results, ttlcache.DefaultTTL)

		c.logger.Debug("Reading completed and cached",
			zap.String("model", c.model),
			zap.Int("num_images", len(images)),
			zap.Duration("duration", time.Since(start)))

		return results, nil
	})

	if err != nil {
		return nil, err
	}

	if shared {
		c.sfHits.Add(1)
		c.logger.Debug("Singleflight hit for reading request",
			zap.String("model", c.model))
	}

	return result.([]reading.Result), nil
}

// cacheKey generates a unique cache key from model + images + prompt + maxTokens
func (c *CachedReader) cacheKey(images []image.Image, prompt string, maxTokens int) string {
	h := xxhash.New()

	// Include model name
	_, _ = h.WriteString(c.model)
	_, _ = h.WriteString("|")

	// Include prompt
	_, _ = h.WriteString("p:")
	_, _ = h.WriteString(prompt)
	_, _ = h.WriteString("|")

	// Include maxTokens
	_, _ = h.WriteString("t:")
	var tokenBuf [4]byte
	binary.BigEndian.PutUint32(tokenBuf[:], uint32(maxTokens))
	_, _ = h.Write(tokenBuf[:])
	_, _ = h.WriteString("|")

	// Hash each image
	for i, img := range images {
		_, _ = h.WriteString("i")
		// Use index to ensure order matters
		_, _ = h.Write([]byte{byte(i >> 8), byte(i)})
		_, _ = h.WriteString(":")

		// Hash image dimensions and pixel data hash
		bounds := img.Bounds()
		var dimBuf [16]byte
		binary.BigEndian.PutUint32(dimBuf[0:4], uint32(bounds.Min.X))
		binary.BigEndian.PutUint32(dimBuf[4:8], uint32(bounds.Min.Y))
		binary.BigEndian.PutUint32(dimBuf[8:12], uint32(bounds.Max.X))
		binary.BigEndian.PutUint32(dimBuf[12:16], uint32(bounds.Max.Y))
		_, _ = h.Write(dimBuf[:])

		// Hash image pixels by encoding to JPEG and hashing
		// This is more efficient than iterating all pixels
		imgHash := hashImage(img)
		var imgHashBuf [8]byte
		binary.BigEndian.PutUint64(imgHashBuf[:], imgHash)
		_, _ = h.Write(imgHashBuf[:])

		_, _ = h.WriteString("|")
	}

	// Convert uint64 hash to string key
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], h.Sum64())
	return string(buf[:])
}

// hashImage generates a hash for an image
func hashImage(img image.Image) uint64 {
	h := xxhash.New()

	// For efficiency, encode to JPEG and hash the bytes
	// This captures the visual content without iterating every pixel
	encoder := jpeg.Options{Quality: 50} // Lower quality is fine for hashing
	if err := jpeg.Encode(h, img, &encoder); err != nil {
		// Fallback: hash dimensions only
		bounds := img.Bounds()
		var buf [16]byte
		binary.BigEndian.PutUint32(buf[0:4], uint32(bounds.Dx()))
		binary.BigEndian.PutUint32(buf[4:8], uint32(bounds.Dy()))
		_, _ = h.Write(buf[:])
	}

	return h.Sum64()
}

// Close closes the underlying reader
func (c *CachedReader) Close() error {
	return c.reader.Close()
}

// Stats returns cache statistics for this reader
func (c *CachedReader) Stats() ReaderCacheStats {
	return ReaderCacheStats{
		Model:            c.model,
		Hits:             c.hits.Load(),
		Misses:           c.misses.Load(),
		SingleflightHits: c.sfHits.Load(),
	}
}

// ReaderCacheStats holds cache statistics for a reader
type ReaderCacheStats struct {
	Model            string `json:"model"`
	Hits             uint64 `json:"hits"`
	Misses           uint64 `json:"misses"`
	SingleflightHits uint64 `json:"singleflight_hits"`
}

// ReadingCache manages caching for multiple readers
type ReadingCache struct {
	cache  *ttlcache.Cache[string, []reading.Result]
	logger *zap.Logger
	cancel context.CancelFunc
}

// NewReadingCache creates a new reading cache
func NewReadingCache(logger *zap.Logger) *ReadingCache {
	cache := ttlcache.New(
		ttlcache.WithTTL[string, []reading.Result](ReadingCacheTTL),
	)
	go cache.Start()

	ctx, cancel := context.WithCancel(context.Background())
	rc := &ReadingCache{
		cache:  cache,
		logger: logger,
		cancel: cancel,
	}

	// Log cache stats periodically
	go rc.logStats(ctx)

	return rc
}

// WrapReader wraps a reader with caching
func (rc *ReadingCache) WrapReader(reader reading.Reader, model string) *CachedReader {
	return NewCachedReader(reader, model, rc.cache, rc.logger.Named(model))
}

// Close stops the cache
func (rc *ReadingCache) Close() {
	rc.cancel()
	rc.cache.Stop()
}

// logStats logs cache statistics periodically
func (rc *ReadingCache) logStats(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			metrics := rc.cache.Metrics()
			if metrics.Hits > 0 || metrics.Misses > 0 {
				hitRate := float64(0)
				total := metrics.Hits + metrics.Misses
				if total > 0 {
					hitRate = float64(metrics.Hits) / float64(total) * 100
				}
				rc.logger.Info("Reading cache stats",
					zap.Uint64("hits", metrics.Hits),
					zap.Uint64("misses", metrics.Misses),
					zap.Float64("hit_rate_pct", hitRate),
					zap.Int("items", rc.cache.Len()))
			}
		}
	}
}

// Stats returns global cache statistics
func (rc *ReadingCache) Stats() map[string]any {
	metrics := rc.cache.Metrics()
	return map[string]any{
		"hits":   metrics.Hits,
		"misses": metrics.Misses,
		"items":  rc.cache.Len(),
	}
}
