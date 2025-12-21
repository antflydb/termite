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
	"fmt"
	"sync/atomic"
	"time"

	"github.com/antflydb/antfly-go/libaf/chunking"
	termchunking "github.com/antflydb/termite/pkg/termite/lib/chunking"
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	"github.com/cespare/xxhash/v2"
	"github.com/jellydator/ttlcache/v3"
	"go.uber.org/zap"
	"golang.org/x/sync/singleflight"
)

// CachedChunker provides in-memory caching for chunking operations with model registry
type CachedChunker struct {
	registry        *ChunkerRegistry
	fixedChunker    chunking.Chunker
	memCache        *ttlcache.Cache[uint64, ChunkResult]
	sfGroup         *singleflight.Group
	singleflightHit *atomic.Uint64
	logger          *zap.Logger
	cancel          context.CancelFunc
}

// ChunkResult stores chunking results with metadata
type ChunkResult struct {
	Chunks   []chunking.Chunk `json:"chunks"`
	Model    string           `json:"model"`
	CachedAt time.Time        `json:"cached_at"`
}

// NewCachedChunker creates a new cached chunker with model registry support
// If sessionManager is provided, it will be used to obtain sessions for model loading (required for ONNX Runtime)
func NewCachedChunker(
	modelsDir string,
	sessionManager *hugot.SessionManager,
	logger *zap.Logger,
) (*CachedChunker, error) {
	// Create memory cache with 2-minute TTL (same as embeddings)
	cache := ttlcache.New(
		ttlcache.WithTTL[uint64, ChunkResult](2 * time.Minute),
	)
	go cache.Start()

	// Create fixed chunker (always available as fallback)
	fixedChunker, err := termchunking.NewFixedChunker(termchunking.DefaultFixedChunkerConfig())
	if err != nil {
		cache.Stop()
		return nil, fmt.Errorf("failed to create fixed chunker: %w", err)
	}

	// Create model registry with session manager
	registry, err := NewChunkerRegistry(modelsDir, sessionManager, logger.Named("registry"))
	if err != nil {
		cache.Stop()
		_ = fixedChunker.Close()
		return nil, fmt.Errorf("failed to create chunker registry: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	singleflightHit := &atomic.Uint64{}
	singleflightHit.Store(0)

	cc := &CachedChunker{
		registry:        registry,
		fixedChunker:    fixedChunker,
		memCache:        cache,
		sfGroup:         &singleflight.Group{},
		singleflightHit: singleflightHit,
		logger:          logger,
		cancel:          cancel,
	}

	// Start cache stats logger
	go cc.logCacheStats(ctx)

	// Log available models
	models := registry.List()
	if len(models) > 0 {
		logger.Info("Loaded ONNX chunker models", zap.Strings("models", models))
	} else {
		logger.Info("No ONNX models loaded, using built-in fixed-bert-tokenizer model only")
	}

	return cc, nil
}

// chunkConfig is the internal config format for the public API
type chunkConfig struct {
	Model         string  `json:"model"`
	TargetTokens  int     `json:"target_tokens"`
	OverlapTokens int     `json:"overlap_tokens"`
	Separator     string  `json:"separator"`
	MaxChunks     int     `json:"max_chunks"`
	Threshold     float32 `json:"threshold"`
}

// Chunk performs chunking with two-tier caching
func (cc *CachedChunker) Chunk(ctx context.Context, text string, config chunkConfig) ([]chunking.Chunk, bool, error) {
	if text == "" {
		return nil, false, nil
	}

	// Compute cache key based on config and text hash
	cacheKey := cc.computeCacheKey(text, config)

	// Check memory cache
	if item := cc.memCache.Get(cacheKey); item != nil {
		cc.logger.Debug("Chunk cache hit (memory)",
			zap.Uint64("cache_key", cacheKey),
			zap.String("model", item.Value().Model),
			zap.Int("num_chunks", len(item.Value().Chunks)))
		return item.Value().Chunks, true, nil
	}

	// Cache miss: Use singleflight to deduplicate concurrent identical requests
	cc.logger.Debug("Chunk cache miss, performing chunking",
		zap.Uint64("cache_key", cacheKey),
		zap.Int("text_length", len(text)),
		zap.String("model", config.Model))

	v, err, shared := cc.sfGroup.Do(fmt.Sprintf("%d", cacheKey), func() (any, error) {
		// Double-check cache (another goroutine might have populated it)
		if item := cc.memCache.Get(cacheKey); item != nil {
			cc.logger.Debug("Chunk found in cache during singleflight")
			return item.Value(), nil
		}

		// Perform actual chunking
		chunks, model, err := cc.performChunking(ctx, text, config)
		if err != nil {
			return nil, err
		}

		result := ChunkResult{
			Chunks:   chunks,
			Model:    model,
			CachedAt: time.Now(),
		}

		// Store in memory cache
		cc.memCache.Set(cacheKey, result, ttlcache.DefaultTTL)

		cc.logger.Info("Chunking completed and cached",
			zap.Uint64("cache_key", cacheKey),
			zap.String("model", model),
			zap.Int("num_chunks", len(chunks)),
			zap.Int("text_length", len(text)))

		return result, nil
	})

	if shared {
		cc.singleflightHit.Add(1)
		cc.logger.Debug("Singleflight deduplication hit")
	}

	if err != nil {
		return nil, false, err
	}

	result := v.(ChunkResult)
	return result.Chunks, false, nil
}

// performChunking executes the actual chunking logic based on model
func (cc *CachedChunker) performChunking(ctx context.Context, text string, config chunkConfig) (chunks []chunking.Chunk, model string, err error) {
	model = config.Model

	// Build per-request options from config
	opts := cc.buildChunkOptions(config)

	// Check if it's a built-in fixed model
	isFixedModel := model == termchunking.ModelFixedBert || model == termchunking.ModelFixedBPE

	// Try to get ONNX model from registry first (if not a built-in fixed model)
	if !isFixedModel {
		if chunker, err := cc.registry.Get(model); err == nil {
			cc.logger.Debug("Using ONNX model from registry",
				zap.String("model", model))

			chunks, err = chunker.Chunk(ctx, text, opts)
			if err != nil {
				cc.logger.Warn("ONNX model failed, falling back to fixed-bert-tokenizer",
					zap.String("model", model),
					zap.Error(err))
				// Fall through to fixed chunker
			} else {
				return chunks, model, nil
			}
		} else {
			cc.logger.Debug("Model not found in registry, falling back to fixed-bert-tokenizer",
				zap.String("requested", model),
				zap.Error(err))
		}
	}

	// Use fixed chunker as fallback
	cc.logger.Debug("Using fixed chunker")
	chunks, err = cc.fixedChunker.Chunk(ctx, text, opts)
	model = termchunking.ModelFixedBert

	if err != nil {
		return nil, "", fmt.Errorf("chunking failed with model %s: %w", model, err)
	}

	return chunks, model, nil
}

// buildChunkOptions converts internal chunkConfig to the chunking.ChunkOptions type.
// Only sets non-zero values to allow chunker defaults to apply for unset options.
func (cc *CachedChunker) buildChunkOptions(config chunkConfig) chunking.ChunkOptions {
	var opts chunking.ChunkOptions
	if config.MaxChunks > 0 {
		opts.MaxChunks = config.MaxChunks
	}
	if config.Threshold > 0 {
		opts.Threshold = config.Threshold
	}
	if config.TargetTokens > 0 {
		opts.TargetTokens = config.TargetTokens
	}
	return opts
}

// computeCacheKey generates a cache key from text and config
func (cc *CachedChunker) computeCacheKey(text string, config chunkConfig) uint64 {
	// Create a deterministic key from config
	configStr := fmt.Sprintf("%s:%d:%d:%s:%d:%.3f",
		config.Model,
		config.TargetTokens,
		config.OverlapTokens,
		config.Separator,
		config.MaxChunks,
		config.Threshold)

	// Hash text separately (for large texts)
	textHash := sha256.Sum256([]byte(text))

	// Combine config and text hash
	combined := configStr + string(textHash[:])
	return xxhash.Sum64String(combined)
}

// logCacheStats periodically logs cache statistics
func (cc *CachedChunker) logCacheStats(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			metrics := cc.memCache.Metrics()
			hitRate := float64(0)
			if metrics.Hits+metrics.Misses > 0 {
				hitRate = float64(metrics.Hits) / float64(metrics.Hits+metrics.Misses) * 100
			}

			if cc.memCache.Len() == 0 {
				continue
			}

			cc.logger.Info("Chunking cache stats",
				zap.Int("size", cc.memCache.Len()),
				zap.Uint64("singleflight_hits", cc.singleflightHit.Load()),
				zap.Uint64("cache_hits", metrics.Hits),
				zap.Uint64("cache_misses", metrics.Misses),
				zap.String("hit_rate_percent", fmt.Sprintf("%.2f", hitRate)))

		case <-ctx.Done():
			cc.logger.Info("Stopping chunking cache stats logger")
			return
		}
	}
}

// ListModels returns all available chunker models and strategies
func (cc *CachedChunker) ListModels() []string {
	models := cc.registry.List()
	// Add built-in strategies
	all := append([]string{termchunking.ModelFixedBert, termchunking.ModelFixedBPE}, models...)
	return all
}

// Close releases resources
func (cc *CachedChunker) Close() error {
	cc.cancel()
	cc.memCache.Stop()

	if cc.registry != nil {
		if err := cc.registry.Close(); err != nil {
			cc.logger.Warn("Error closing chunker registry", zap.Error(err))
		}
	}

	if cc.fixedChunker != nil {
		if err := cc.fixedChunker.Close(); err != nil {
			cc.logger.Warn("Error closing fixed chunker", zap.Error(err))
		}
	}

	return nil
}
