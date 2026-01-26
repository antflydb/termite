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
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/antflydb/antfly-go/libaf/reranking"
	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
	termreranking "github.com/antflydb/termite/pkg/termite/lib/reranking"
	"github.com/jellydator/ttlcache/v3"
	"go.uber.org/zap"
)

// RerankerModelInfo holds metadata about a discovered reranker model (not loaded yet)
type RerankerModelInfo struct {
	Name         string
	Path         string
	OnnxFilename string
	PoolSize     int
}

// RerankerRegistry manages reranker models with lazy loading and TTL-based unloading
type RerankerRegistry struct {
	modelsDir      string
	sessionManager *backends.SessionManager
	logger         *zap.Logger

	// Model discovery (paths only, not loaded)
	discovered map[string]*RerankerModelInfo
	mu         sync.RWMutex

	// Loaded models with TTL cache
	cache *ttlcache.Cache[string, reranking.Model]

	// Reference counting to prevent eviction during active use
	refCounts   map[string]int
	refCountsMu sync.Mutex

	// Configuration
	keepAlive       time.Duration
	maxLoadedModels uint64
	poolSize        int
}

// RerankerConfig configures the reranker registry
type RerankerConfig struct {
	ModelsDir       string
	KeepAlive       time.Duration // How long to keep models loaded (0 = forever)
	MaxLoadedModels uint64        // Max models in memory (0 = unlimited)
	PoolSize        int           // Number of concurrent pipelines per model (0 = default)
}

// NewRerankerRegistry creates a new lazy-loading reranker registry
func NewRerankerRegistry(
	config RerankerConfig,
	sessionManager *backends.SessionManager,
	logger *zap.Logger,
) (*RerankerRegistry, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	keepAlive := config.KeepAlive
	if keepAlive == 0 {
		keepAlive = ttlcache.NoTTL // Never expire
	}

	poolSize := config.PoolSize
	if poolSize <= 0 {
		poolSize = min(runtime.NumCPU(), 4)
	}

	registry := &RerankerRegistry{
		modelsDir:       config.ModelsDir,
		sessionManager:  sessionManager,
		logger:          logger,
		discovered:      make(map[string]*RerankerModelInfo),
		refCounts:       make(map[string]int),
		keepAlive:       keepAlive,
		maxLoadedModels: config.MaxLoadedModels,
		poolSize:        poolSize,
	}

	// Configure TTL cache with LRU eviction
	cacheOpts := []ttlcache.Option[string, reranking.Model]{
		ttlcache.WithTTL[string, reranking.Model](keepAlive),
	}

	if config.MaxLoadedModels > 0 {
		cacheOpts = append(cacheOpts,
			ttlcache.WithCapacity[string, reranking.Model](config.MaxLoadedModels))
	}

	registry.cache = ttlcache.New(cacheOpts...)

	// Set up eviction callback to close unloaded models
	// Note: Only close on TTL expiration or capacity eviction, not on manual deletion
	// (manual deletion during Close() handles cleanup synchronously)
	registry.cache.OnEviction(func(ctx context.Context, reason ttlcache.EvictionReason, item *ttlcache.Item[string, reranking.Model]) {
		// Skip closing on manual deletion - Close() handles cleanup synchronously
		if reason == ttlcache.EvictionReasonDeleted {
			logger.Debug("Reranker model removed from cache (cleanup handled separately)",
				zap.String("model", item.Key()))
			return
		}

		reasonStr := "unknown"
		switch reason {
		case ttlcache.EvictionReasonExpired:
			reasonStr = "expired (keep-alive timeout)"
		case ttlcache.EvictionReasonCapacityReached:
			reasonStr = "capacity reached (LRU eviction)"
		}

		// Check if model is still in use (has active references)
		// Hold lock through check-and-action to prevent race with Release()
		registry.refCountsMu.Lock()
		refCount := registry.refCounts[item.Key()]
		if refCount > 0 {
			// Re-add while still holding lock to prevent race with Release()
			registry.cache.Set(item.Key(), item.Value(), registry.keepAlive)
			registry.refCountsMu.Unlock()
			logger.Warn("Preventing eviction of reranker model with active references",
				zap.String("model", item.Key()),
				zap.Int("refCount", refCount),
				zap.String("reason", reasonStr))
			return
		}
		registry.refCountsMu.Unlock()

		logger.Info("Evicting reranker model from cache",
			zap.String("model", item.Key()),
			zap.String("reason", reasonStr))
		if err := item.Value().Close(); err != nil {
			logger.Warn("Error closing evicted reranker model",
				zap.String("model", item.Key()),
				zap.Error(err))
		}
	})

	// Start cache cleanup goroutine
	go registry.cache.Start()

	// Discover models (but don't load them)
	if err := registry.discoverModels(); err != nil {
		registry.cache.Stop()
		return nil, err
	}

	logger.Info("Lazy reranker registry initialized",
		zap.Int("models_discovered", len(registry.discovered)),
		zap.Duration("keep_alive", keepAlive),
		zap.Uint64("max_loaded_models", config.MaxLoadedModels))

	return registry, nil
}

// discoverModels finds all reranker models in the models directory without loading them
func (r *RerankerRegistry) discoverModels() error {
	if r.modelsDir == "" {
		r.logger.Info("No reranker models directory configured")
		return nil
	}

	// Check if directory exists
	if _, err := os.Stat(r.modelsDir); os.IsNotExist(err) {
		r.logger.Warn("Reranker models directory does not exist",
			zap.String("dir", r.modelsDir))
		return nil
	}

	// Use discoverModelsInDir which handles owner/model structure
	discovered, err := discoverModelsInDir(r.modelsDir, modelregistry.ModelTypeReranker, r.logger)
	if err != nil {
		return fmt.Errorf("discovering reranker models: %w", err)
	}

	// Pool size for concurrent pipeline access
	poolSize := r.poolSize

	for _, dm := range discovered {
		modelPath := dm.Path
		registryFullName := dm.FullName()
		variants := dm.Variants

		// Skip if no model files exist
		if len(variants) == 0 {
			continue
		}

		// Log discovered variants
		variantIDs := make([]string, 0, len(variants))
		for v := range variants {
			if v == "" {
				variantIDs = append(variantIDs, "default")
			} else {
				variantIDs = append(variantIDs, v)
			}
		}
		r.logger.Info("Discovered reranker model (not loaded)",
			zap.String("name", registryFullName),
			zap.String("path", modelPath),
			zap.Strings("variants", variantIDs))

		// Store each variant for lazy loading
		for variantID, onnxFilename := range variants {
			// Determine registry name
			registryName := registryFullName
			if variantID != "" {
				registryName = registryFullName + "-" + variantID
			}

			r.discovered[registryName] = &RerankerModelInfo{
				Name:         registryName,
				Path:         modelPath,
				OnnxFilename: onnxFilename,
				PoolSize:     poolSize,
			}
		}
	}

	r.logger.Info("Reranker model discovery complete",
		zap.Int("models_discovered", len(r.discovered)),
		zap.Duration("keep_alive", r.keepAlive),
		zap.Uint64("max_loaded_models", r.maxLoadedModels))

	return nil
}

// Get returns a reranker by name, loading it if necessary.
// DEPRECATED: Use Acquire() instead for long-running operations to prevent
// the model from being evicted during use. Get() does not track usage and
// the returned reranker may be closed if the cache evicts it.
func (r *RerankerRegistry) Get(modelName string) (reranking.Model, error) {
	// Check cache first
	if item := r.cache.Get(modelName); item != nil {
		r.logger.Debug("Reranker cache hit", zap.String("model", modelName))
		return item.Value(), nil
	}

	// Check if model is discovered
	r.mu.RLock()
	info, ok := r.discovered[modelName]
	r.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("reranker model not found: %s", modelName)
	}

	// Load the model
	return r.loadModel(info)
}

// Acquire returns a reranker by name and increments its reference count.
// The caller MUST call Release() when done to allow the model to be evicted.
// This prevents the model from being closed while in use.
func (r *RerankerRegistry) Acquire(modelName string) (reranking.Model, error) {
	model, err := r.Get(modelName)
	if err != nil {
		return nil, err
	}

	r.refCountsMu.Lock()
	r.refCounts[modelName]++
	count := r.refCounts[modelName]
	r.refCountsMu.Unlock()

	r.logger.Debug("Acquired reranker model",
		zap.String("model", modelName),
		zap.Int("refCount", count))

	return model, nil
}

// Release decrements the reference count for a model.
// Must be called after Acquire() when the caller is done using the reranker.
func (r *RerankerRegistry) Release(modelName string) {
	r.refCountsMu.Lock()
	if r.refCounts[modelName] > 0 {
		r.refCounts[modelName]--
	}
	count := r.refCounts[modelName]
	r.refCountsMu.Unlock()

	r.logger.Debug("Released reranker model",
		zap.String("model", modelName),
		zap.Int("refCount", count))
}

// loadModel loads a reranker model from disk
func (r *RerankerRegistry) loadModel(info *RerankerModelInfo) (reranking.Model, error) {
	r.logger.Info("Loading reranker model on demand",
		zap.String("model", info.Name),
		zap.String("path", info.Path))

	// Load using pipeline-based reranker
	cfg := termreranking.PooledRerankerConfig{
		ModelPath:     info.Path,
		PoolSize:      info.PoolSize,
		ModelBackends: nil, // Use all available backends
		Logger:        r.logger.Named(info.Name),
	}
	model, backendUsed, err := termreranking.NewPooledReranker(cfg, r.sessionManager)
	if err != nil {
		return nil, fmt.Errorf("loading reranker model %s: %w", info.Name, err)
	}

	r.logger.Info("Successfully loaded reranker model",
		zap.String("name", info.Name),
		zap.String("backend", string(backendUsed)),
		zap.Int("poolSize", info.PoolSize))

	// Add to cache
	r.cache.Set(info.Name, model, r.keepAlive)

	return model, nil
}

// List returns all available reranker model names (discovered, not necessarily loaded)
func (r *RerankerRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.discovered))
	for name := range r.discovered {
		names = append(names, name)
	}
	return names
}

// ListLoaded returns only the currently loaded reranker model names
func (r *RerankerRegistry) ListLoaded() []string {
	keys := r.cache.Keys()
	return keys
}

// IsLoaded returns whether a model is currently loaded in memory
func (r *RerankerRegistry) IsLoaded(modelName string) bool {
	return r.cache.Has(modelName)
}

// Preload loads specified models at startup to avoid first-request latency
func (r *RerankerRegistry) Preload(modelNames []string) error {
	if len(modelNames) == 0 {
		return nil
	}

	r.logger.Info("Preloading reranker models", zap.Strings("models", modelNames))

	var loaded, failed int
	for _, name := range modelNames {
		if _, err := r.Get(name); err != nil {
			r.logger.Warn("Failed to preload reranker model",
				zap.String("model", name),
				zap.Error(err))
			failed++
		} else {
			r.logger.Info("Preloaded reranker model",
				zap.String("model", name))
			loaded++
		}
	}

	r.logger.Info("Reranker preloading complete",
		zap.Int("loaded", loaded),
		zap.Int("failed", failed))

	if failed > 0 && loaded == 0 {
		return fmt.Errorf("all %d reranker models failed to preload", failed)
	}

	return nil
}

// PreloadAll loads all discovered models (for eager loading mode)
func (r *RerankerRegistry) PreloadAll() error {
	return r.Preload(r.List())
}

// Close stops the cache and unloads all models
func (r *RerankerRegistry) Close() error {
	r.logger.Info("Closing lazy reranker registry")

	// Stop cache first to prevent new evictions
	r.cache.Stop()

	// Close all cached models synchronously (don't rely on async eviction callbacks)
	for _, key := range r.cache.Keys() {
		if item := r.cache.Get(key); item != nil {
			model := item.Value()
			r.logger.Debug("Closing cached reranker model",
				zap.String("model", key))
			if err := model.Close(); err != nil {
				r.logger.Warn("Error closing reranker model",
					zap.String("model", key),
					zap.Error(err))
			}
		}
	}

	// Clear the cache (eviction callbacks won't close since reason is EvictionReasonDeleted)
	r.cache.DeleteAll()

	return nil
}
