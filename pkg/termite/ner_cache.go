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
	"sync/atomic"
	"time"

	"github.com/antflydb/termite/pkg/termite/lib/ner"
	"github.com/cespare/xxhash/v2"
	"github.com/jellydator/ttlcache/v3"
	"go.uber.org/zap"
	"golang.org/x/sync/singleflight"
)

// NERCacheTTL is the default TTL for cached NER results
const NERCacheTTL = 2 * time.Minute

// CachedNER wraps a NER model with caching support
type CachedNER struct {
	model   ner.Model
	name    string
	cache   *ttlcache.Cache[string, [][]ner.Entity]
	sfGroup *singleflight.Group
	logger  *zap.Logger

	// Metrics
	hits   atomic.Uint64
	misses atomic.Uint64
	sfHits atomic.Uint64
}

// NewCachedNER wraps a NER model with caching
func NewCachedNER(
	model ner.Model,
	name string,
	cache *ttlcache.Cache[string, [][]ner.Entity],
	logger *zap.Logger,
) *CachedNER {
	return &CachedNER{
		model:   model,
		name:    name,
		cache:   cache,
		sfGroup: &singleflight.Group{},
		logger:  logger,
	}
}

// Recognize extracts entities with caching support
func (c *CachedNER) Recognize(ctx context.Context, texts []string) ([][]ner.Entity, error) {
	// Generate cache key from model + texts hash
	key := c.cacheKey(texts)

	// Check cache first
	if item := c.cache.Get(key); item != nil {
		c.hits.Add(1)
		RecordCacheHit("ner")
		c.logger.Debug("NER cache hit",
			zap.String("model", c.name),
			zap.Int("num_texts", len(texts)))
		return item.Value(), nil
	}

	// Use singleflight to deduplicate concurrent identical requests
	result, err, shared := c.sfGroup.Do(key, func() (any, error) {
		c.misses.Add(1)
		RecordCacheMiss("ner")

		start := time.Now()
		entities, err := c.model.Recognize(ctx, texts)
		if err != nil {
			return nil, err
		}

		// Record duration
		RecordRequestDuration("ner", c.name, "200", time.Since(start).Seconds())

		// Store in cache
		c.cache.Set(key, entities, ttlcache.DefaultTTL)

		c.logger.Debug("NER completed and cached",
			zap.String("model", c.name),
			zap.Int("num_texts", len(texts)),
			zap.Duration("duration", time.Since(start)))

		return entities, nil
	})

	if err != nil {
		return nil, err
	}

	if shared {
		c.sfHits.Add(1)
		c.logger.Debug("Singleflight hit for NER request",
			zap.String("model", c.name))
	}

	return result.([][]ner.Entity), nil
}

// cacheKey generates a unique cache key from model + texts
func (c *CachedNER) cacheKey(texts []string) string {
	h := xxhash.New()

	// Include model name
	_, _ = h.WriteString(c.name)
	_, _ = h.WriteString("|")

	// Hash each text
	for i, text := range texts {
		_, _ = h.WriteString("t")
		// Use index to ensure order matters
		_, _ = h.Write([]byte{byte(i >> 8), byte(i)})
		_, _ = h.WriteString(":")
		_, _ = h.WriteString(text)
		_, _ = h.WriteString("|")
	}

	// Convert uint64 hash to string key
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], h.Sum64())
	return string(buf[:])
}

// Close closes the underlying model
func (c *CachedNER) Close() error {
	return c.model.Close()
}

// Stats returns cache statistics for this NER model
func (c *CachedNER) Stats() NERCacheStats {
	return NERCacheStats{
		Model:            c.name,
		Hits:             c.hits.Load(),
		Misses:           c.misses.Load(),
		SingleflightHits: c.sfHits.Load(),
	}
}

// NERCacheStats holds cache statistics for a NER model
type NERCacheStats struct {
	Model            string `json:"model"`
	Hits             uint64 `json:"hits"`
	Misses           uint64 `json:"misses"`
	SingleflightHits uint64 `json:"singleflight_hits"`
}

// NERCache manages caching for multiple NER models
type NERCache struct {
	cache  *ttlcache.Cache[string, [][]ner.Entity]
	logger *zap.Logger
	cancel context.CancelFunc
}

// NewNERCache creates a new NER cache
func NewNERCache(logger *zap.Logger) *NERCache {
	cache := ttlcache.New(
		ttlcache.WithTTL[string, [][]ner.Entity](NERCacheTTL),
	)
	go cache.Start()

	ctx, cancel := context.WithCancel(context.Background())
	nc := &NERCache{
		cache:  cache,
		logger: logger,
		cancel: cancel,
	}

	// Log cache stats periodically
	go nc.logStats(ctx)

	return nc
}

// WrapModel wraps a NER model with caching
func (nc *NERCache) WrapModel(model ner.Model, name string) *CachedNER {
	return NewCachedNER(model, name, nc.cache, nc.logger.Named(name))
}

// Close stops the cache
func (nc *NERCache) Close() {
	nc.cancel()
	nc.cache.Stop()
}

// logStats logs cache statistics periodically
func (nc *NERCache) logStats(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			metrics := nc.cache.Metrics()
			if metrics.Hits > 0 || metrics.Misses > 0 {
				hitRate := float64(0)
				total := metrics.Hits + metrics.Misses
				if total > 0 {
					hitRate = float64(metrics.Hits) / float64(total) * 100
				}
				nc.logger.Info("NER cache stats",
					zap.Uint64("hits", metrics.Hits),
					zap.Uint64("misses", metrics.Misses),
					zap.Float64("hit_rate_pct", hitRate),
					zap.Int("items", nc.cache.Len()))
			}
		}
	}
}

// Stats returns global cache statistics
func (nc *NERCache) Stats() map[string]any {
	metrics := nc.cache.Metrics()
	return map[string]any{
		"hits":   metrics.Hits,
		"misses": metrics.Misses,
		"items":  nc.cache.Len(),
	}
}
