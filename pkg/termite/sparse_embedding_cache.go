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
	"time"

	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/cespare/xxhash/v2"
	"github.com/jellydator/ttlcache/v3"
	"go.uber.org/zap"
	"golang.org/x/sync/singleflight"
)

// CachedSparseEmbedder wraps a SparseEmbedder with caching support.
type CachedSparseEmbedder struct {
	embedder embeddings.SparseEmbedder
	model    string
	cache    *ttlcache.Cache[string, []embeddings.SparseVector]
	sfGroup  *singleflight.Group
	logger   *zap.Logger
}

// SparseEmbed generates sparse embeddings with caching and singleflight deduplication.
func (c *CachedSparseEmbedder) SparseEmbed(ctx context.Context, texts []string) ([]embeddings.SparseVector, error) {
	key := c.cacheKey(texts)

	// Check cache
	if item := c.cache.Get(key); item != nil {
		RecordCacheHit("sparse_embedding")
		return item.Value(), nil
	}

	// Singleflight deduplication
	result, err, _ := c.sfGroup.Do(key, func() (any, error) {
		RecordCacheMiss("sparse_embedding")

		start := time.Now()
		vecs, err := c.embedder.SparseEmbed(ctx, texts)
		if err != nil {
			return nil, err
		}

		RecordRequestDuration("sparse_embed", c.model, "200", time.Since(start).Seconds())
		c.cache.Set(key, vecs, ttlcache.DefaultTTL)
		return vecs, nil
	})

	if err != nil {
		return nil, err
	}
	return result.([]embeddings.SparseVector), nil
}

func (c *CachedSparseEmbedder) cacheKey(texts []string) string {
	h := xxhash.New()
	_, _ = h.WriteString(c.model)
	_, _ = h.WriteString("|sparse|")
	for _, text := range texts {
		_, _ = h.WriteString(text)
		_, _ = h.WriteString("|")
	}
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], h.Sum64())
	return string(buf[:])
}

// SparseEmbeddingCache manages caching for sparse embedders.
type SparseEmbeddingCache struct {
	cache  *ttlcache.Cache[string, []embeddings.SparseVector]
	logger *zap.Logger
	cancel context.CancelFunc
}

// NewSparseEmbeddingCache creates a new sparse embedding cache.
func NewSparseEmbeddingCache(logger *zap.Logger) *SparseEmbeddingCache {
	cache := ttlcache.New(
		ttlcache.WithTTL[string, []embeddings.SparseVector](EmbeddingCacheTTL),
	)
	go cache.Start()

	_, cancel := context.WithCancel(context.Background())

	return &SparseEmbeddingCache{
		cache:  cache,
		logger: logger,
		cancel: cancel,
	}
}

// WrapSparseEmbedder wraps a sparse embedder with caching.
func (sc *SparseEmbeddingCache) WrapSparseEmbedder(embedder embeddings.SparseEmbedder, model string) *CachedSparseEmbedder {
	return &CachedSparseEmbedder{
		embedder: embedder,
		model:    model,
		cache:    sc.cache,
		sfGroup:  &singleflight.Group{},
		logger:   sc.logger.Named(model),
	}
}

// Close stops the cache.
func (sc *SparseEmbeddingCache) Close() {
	sc.cancel()
	sc.cache.Stop()
}
