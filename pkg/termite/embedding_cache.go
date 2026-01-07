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
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/antflydb/antfly-go/libaf/ai"
	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/cespare/xxhash/v2"
	"github.com/jellydator/ttlcache/v3"
	"go.uber.org/zap"
	"golang.org/x/sync/singleflight"
)

// EmbeddingCacheTTL is the default TTL for cached embeddings
const EmbeddingCacheTTL = 2 * time.Minute

// CachedEmbedder wraps an embedder with caching support
type CachedEmbedder struct {
	embedder embeddings.Embedder
	model    string
	cache    *ttlcache.Cache[string, [][]float32]
	sfGroup  *singleflight.Group
	logger   *zap.Logger

	// Metrics
	hits   atomic.Uint64
	misses atomic.Uint64
	sfHits atomic.Uint64
}

// NewCachedEmbedder wraps an embedder with caching
func NewCachedEmbedder(
	embedder embeddings.Embedder,
	model string,
	cache *ttlcache.Cache[string, [][]float32],
	logger *zap.Logger,
) *CachedEmbedder {
	return &CachedEmbedder{
		embedder: embedder,
		model:    model,
		cache:    cache,
		sfGroup:  &singleflight.Group{},
		logger:   logger,
	}
}

// Capabilities returns the underlying embedder's capabilities
func (c *CachedEmbedder) Capabilities() embeddings.EmbedderCapabilities {
	return c.embedder.Capabilities()
}

// Embed generates embeddings with caching support
func (c *CachedEmbedder) Embed(ctx context.Context, contents [][]ai.ContentPart) ([][]float32, error) {
	// Generate cache key from model + content hash
	key := c.cacheKey(contents)

	// Check cache first
	if item := c.cache.Get(key); item != nil {
		c.hits.Add(1)
		RecordCacheHit("embedding")
		c.logger.Debug("Embedding cache hit",
			zap.String("model", c.model),
			zap.Int("num_embeddings", len(item.Value())))
		return item.Value(), nil
	}

	// Use singleflight to deduplicate concurrent identical requests
	result, err, shared := c.sfGroup.Do(key, func() (any, error) {
		c.misses.Add(1)
		RecordCacheMiss("embedding")

		start := time.Now()
		embeds, err := c.embedder.Embed(ctx, contents)
		if err != nil {
			return nil, err
		}

		// Record duration
		RecordRequestDuration("embed", c.model, "200", time.Since(start).Seconds())

		// Store in cache
		c.cache.Set(key, embeds, ttlcache.DefaultTTL)

		c.logger.Debug("Embedding generated and cached",
			zap.String("model", c.model),
			zap.Int("num_embeddings", len(embeds)),
			zap.Duration("duration", time.Since(start)))

		return embeds, nil
	})

	if err != nil {
		return nil, err
	}

	if shared {
		c.sfHits.Add(1)
		c.logger.Debug("Singleflight hit for embedding request",
			zap.String("model", c.model))
	}

	return result.([][]float32), nil
}

// cacheKey generates a unique cache key from model + content
func (c *CachedEmbedder) cacheKey(contents [][]ai.ContentPart) string {
	h := xxhash.New()

	// Include model name
	_, _ = h.WriteString(c.model)
	_, _ = h.WriteString("|")

	// Hash each content part
	for _, parts := range contents {
		for _, part := range parts {
			switch p := part.(type) {
			case ai.TextContent:
				_, _ = h.WriteString("t:")
				_, _ = h.WriteString(p.Text)
				c.logger.Debug("Cache key: text content",
					zap.String("text_prefix", truncateString(p.Text, 50)))
			case ai.BinaryContent:
				_, _ = h.WriteString("b:")
				_, _ = h.WriteString(p.MIMEType)
				_, _ = h.WriteString(":")
				// Use SHA256 for binary content (more collision-resistant)
				binHash := sha256.Sum256(p.Data)
				_, _ = h.Write(binHash[:])
				c.logger.Debug("Cache key: binary content",
					zap.String("mime_type", p.MIMEType),
					zap.Int("data_len", len(p.Data)),
					zap.String("sha256_prefix", fmt.Sprintf("%x", binHash[:8])))
			default:
				c.logger.Warn("Cache key: unknown content type",
					zap.String("type", fmt.Sprintf("%T", part)))
			}
			_, _ = h.WriteString("|")
		}
		_, _ = h.WriteString("||")
	}

	// Convert uint64 hash to hex string
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], h.Sum64())
	return string(buf[:])
}

// Close closes the underlying embedder
func (c *CachedEmbedder) Close() error {
	if closer, ok := c.embedder.(interface{ Close() error }); ok {
		return closer.Close()
	}
	return nil
}

// Stats returns cache statistics for this embedder
func (c *CachedEmbedder) Stats() EmbedderCacheStats {
	return EmbedderCacheStats{
		Model:            c.model,
		Hits:             c.hits.Load(),
		Misses:           c.misses.Load(),
		SingleflightHits: c.sfHits.Load(),
	}
}

// EmbedderCacheStats holds cache statistics for an embedder
type EmbedderCacheStats struct {
	Model            string `json:"model"`
	Hits             uint64 `json:"hits"`
	Misses           uint64 `json:"misses"`
	SingleflightHits uint64 `json:"singleflight_hits"`
}

// EmbeddingCache manages caching for multiple embedders
type EmbeddingCache struct {
	cache  *ttlcache.Cache[string, [][]float32]
	logger *zap.Logger
	cancel context.CancelFunc
}

// NewEmbeddingCache creates a new embedding cache
func NewEmbeddingCache(logger *zap.Logger) *EmbeddingCache {
	cache := ttlcache.New(
		ttlcache.WithTTL[string, [][]float32](EmbeddingCacheTTL),
	)
	go cache.Start()

	ctx, cancel := context.WithCancel(context.Background())
	ec := &EmbeddingCache{
		cache:  cache,
		logger: logger,
		cancel: cancel,
	}

	// Log cache stats periodically
	go ec.logStats(ctx)

	return ec
}

// WrapEmbedder wraps an embedder with caching
func (ec *EmbeddingCache) WrapEmbedder(embedder embeddings.Embedder, model string) *CachedEmbedder {
	return NewCachedEmbedder(embedder, model, ec.cache, ec.logger.Named(model))
}

// Close stops the cache
func (ec *EmbeddingCache) Close() {
	ec.cancel()
	ec.cache.Stop()
}

// logStats logs cache statistics periodically
func (ec *EmbeddingCache) logStats(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			metrics := ec.cache.Metrics()
			if metrics.Hits > 0 || metrics.Misses > 0 {
				hitRate := float64(0)
				total := metrics.Hits + metrics.Misses
				if total > 0 {
					hitRate = float64(metrics.Hits) / float64(total) * 100
				}
				ec.logger.Info("Embedding cache stats",
					zap.Uint64("hits", metrics.Hits),
					zap.Uint64("misses", metrics.Misses),
					zap.Float64("hit_rate_pct", hitRate),
					zap.Int("items", ec.cache.Len()))
			}
		}
	}
}

// Stats returns global cache statistics
func (ec *EmbeddingCache) Stats() map[string]any {
	metrics := ec.cache.Metrics()
	return map[string]any{
		"hits":   metrics.Hits,
		"misses": metrics.Misses,
		"items":  ec.cache.Len(),
	}
}

// truncateString returns the first n characters of s, or s if len(s) <= n
func truncateString(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
