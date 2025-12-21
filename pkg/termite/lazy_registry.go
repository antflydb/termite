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
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/antflydb/antfly-go/libaf/embeddings"
	termembeddings "github.com/antflydb/termite/pkg/termite/lib/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	"github.com/jellydator/ttlcache/v3"
	"go.uber.org/zap"
)

// Default keep-alive duration (matches Ollama's 5-minute default)
const DefaultKeepAlive = 5 * time.Minute

// ModelInfo holds metadata about a discovered model (not loaded yet)
type ModelInfo struct {
	Name         string
	Path         string
	OnnxFilename string // e.g., "model.onnx", "model_f16.onnx", "model_i8.onnx"
	PoolSize     int
	ModelType    string   // "embedder", "chunker", "reranker"
	Variants     []string // Available variant IDs (e.g., ["f16", "i8"])
}

// LazyEmbedderRegistry manages embedding models with lazy loading and TTL-based unloading
type LazyEmbedderRegistry struct {
	modelsDir      string
	sessionManager *hugot.SessionManager
	logger         *zap.Logger

	// Model discovery (paths only, not loaded)
	discovered map[string]*ModelInfo
	mu         sync.RWMutex

	// Loaded models with TTL cache (for lazy models)
	cache *ttlcache.Cache[string, embeddings.Embedder]

	// Pinned models (never evicted, stored separately from cache)
	pinned   map[string]embeddings.Embedder
	pinnedMu sync.RWMutex

	// Configuration
	keepAlive       time.Duration
	maxLoadedModels uint64
}

// LazyEmbedderConfig configures the lazy embedder registry
type LazyEmbedderConfig struct {
	ModelsDir       string
	KeepAlive       time.Duration // How long to keep models loaded (0 = forever)
	MaxLoadedModels uint64        // Max models in memory (0 = unlimited)
}

// NewLazyEmbedderRegistry creates a new lazy-loading embedder registry
func NewLazyEmbedderRegistry(
	config LazyEmbedderConfig,
	sessionManager *hugot.SessionManager,
	logger *zap.Logger,
) (*LazyEmbedderRegistry, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	keepAlive := config.KeepAlive
	if keepAlive == 0 {
		keepAlive = ttlcache.NoTTL // Never expire
	}

	registry := &LazyEmbedderRegistry{
		modelsDir:       config.ModelsDir,
		sessionManager:  sessionManager,
		logger:          logger,
		discovered:      make(map[string]*ModelInfo),
		pinned:          make(map[string]embeddings.Embedder),
		keepAlive:       keepAlive,
		maxLoadedModels: config.MaxLoadedModels,
	}

	// Configure TTL cache with LRU eviction
	cacheOpts := []ttlcache.Option[string, embeddings.Embedder]{
		ttlcache.WithTTL[string, embeddings.Embedder](keepAlive),
	}

	// Add capacity limit for LRU eviction
	if config.MaxLoadedModels > 0 {
		cacheOpts = append(cacheOpts,
			ttlcache.WithCapacity[string, embeddings.Embedder](config.MaxLoadedModels))
	}

	registry.cache = ttlcache.New(cacheOpts...)

	// Set up eviction callback to close models
	registry.cache.OnEviction(func(ctx context.Context, reason ttlcache.EvictionReason, item *ttlcache.Item[string, embeddings.Embedder]) {
		modelName := item.Key()
		embedder := item.Value()

		// Check if model was moved to pinned (don't close in that case)
		registry.pinnedMu.RLock()
		isPinned := registry.pinned[modelName] == embedder
		registry.pinnedMu.RUnlock()

		if isPinned {
			logger.Debug("Model moved to pinned, skipping close",
				zap.String("model", modelName))
			return
		}

		reasonStr := "unknown"
		switch reason {
		case ttlcache.EvictionReasonExpired:
			reasonStr = "expired (keep-alive timeout)"
		case ttlcache.EvictionReasonCapacityReached:
			reasonStr = "capacity reached (LRU eviction)"
		case ttlcache.EvictionReasonDeleted:
			reasonStr = "manually deleted"
		}

		logger.Info("Unloading embedder model",
			zap.String("model", modelName),
			zap.String("reason", reasonStr))

		// Close the embedder to free resources
		if closer, ok := embedder.(interface{ Close() error }); ok {
			if err := closer.Close(); err != nil {
				logger.Warn("Error closing embedder",
					zap.String("model", modelName),
					zap.Error(err))
			}
		}
	})

	// Start the cache cleanup goroutine
	go registry.cache.Start()

	// Discover available models (but don't load them)
	if err := registry.discoverModels(); err != nil {
		registry.cache.Stop()
		return nil, err
	}

	return registry, nil
}

// discoverModels scans the models directory and records available models
func (r *LazyEmbedderRegistry) discoverModels() error {
	if r.modelsDir == "" {
		r.logger.Info("No embedder models directory configured")
		return nil
	}

	if _, err := os.Stat(r.modelsDir); os.IsNotExist(err) {
		r.logger.Warn("Embedder models directory does not exist",
			zap.String("dir", r.modelsDir))
		return nil
	}

	entries, err := os.ReadDir(r.modelsDir)
	if err != nil {
		return fmt.Errorf("reading models directory: %w", err)
	}

	poolSize := min(runtime.NumCPU(), 4)

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		modelName := entry.Name()
		modelPath := filepath.Join(r.modelsDir, modelName)

		// Discover all available variants using shared helper
		variants := discoverModelVariants(modelPath)
		if len(variants) == 0 {
			r.logger.Debug("Skipping directory without model files",
				zap.String("dir", modelName))
			continue
		}

		// Collect variant IDs for logging
		variantIDs := make([]string, 0, len(variants))
		for v := range variants {
			if v == "" {
				variantIDs = append(variantIDs, "default")
			} else {
				variantIDs = append(variantIDs, v)
			}
		}

		r.logger.Info("Discovered embedder model (not loaded)",
			zap.String("name", modelName),
			zap.String("path", modelPath),
			zap.Strings("variants", variantIDs))

		// Register each variant
		for variantID, onnxFilename := range variants {
			registryName := modelName
			if variantID != "" {
				registryName = modelName + "-" + variantID
			}

			r.discovered[registryName] = &ModelInfo{
				Name:         registryName,
				Path:         modelPath,
				OnnxFilename: onnxFilename,
				PoolSize:     poolSize,
				ModelType:    "embedder",
				Variants:     variantIDs,
			}
		}
	}

	r.logger.Info("Embedder model discovery complete",
		zap.Int("models_discovered", len(r.discovered)),
		zap.Duration("keep_alive", r.keepAlive),
		zap.Uint64("max_loaded_models", r.maxLoadedModels))

	return nil
}

// Get returns an embedder by model name, loading it if necessary
func (r *LazyEmbedderRegistry) Get(modelName string) (embeddings.Embedder, error) {
	// Check if model is pinned (never evicted)
	r.pinnedMu.RLock()
	if embedder, ok := r.pinned[modelName]; ok {
		r.pinnedMu.RUnlock()
		r.logger.Debug("Embedder pinned hit",
			zap.String("model", modelName))
		return embedder, nil
	}
	r.pinnedMu.RUnlock()

	// Check if already loaded in cache
	if item := r.cache.Get(modelName); item != nil {
		r.logger.Debug("Embedder cache hit",
			zap.String("model", modelName))
		return item.Value(), nil
	}

	// Check if model is known
	r.mu.RLock()
	info, known := r.discovered[modelName]
	r.mu.RUnlock()

	if !known {
		return nil, fmt.Errorf("embedder model not found: %s", modelName)
	}

	// Load the model (with synchronization to prevent double-loading)
	return r.loadModel(info)
}

// loadModel loads a model on demand
func (r *LazyEmbedderRegistry) loadModel(info *ModelInfo) (embeddings.Embedder, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Double-check cache after acquiring lock
	if item := r.cache.Get(info.Name); item != nil {
		return item.Value(), nil
	}

	r.logger.Info("Loading embedder model on demand",
		zap.String("model", info.Name),
		zap.String("path", info.Path),
		zap.String("onnx_filename", info.OnnxFilename),
		zap.Int("pool_size", info.PoolSize))

	embedder, backendUsed, err := termembeddings.NewPooledHugotEmbedderWithSessionManager(
		info.Path,
		info.OnnxFilename,
		info.PoolSize,
		r.sessionManager,
		nil, // modelBackends - use default priority
		r.logger.Named(info.Name),
	)
	if err != nil {
		r.logger.Error("Failed to load embedder model",
			zap.String("model", info.Name),
			zap.Error(err))
		return nil, fmt.Errorf("loading embedder model %s: %w", info.Name, err)
	}

	// Store in cache with TTL
	r.cache.Set(info.Name, embedder, ttlcache.DefaultTTL)

	r.logger.Info("Successfully loaded embedder model",
		zap.String("model", info.Name),
		zap.String("backend", string(backendUsed)),
		zap.Duration("keep_alive", r.keepAlive))

	return embedder, nil
}

// Touch refreshes the TTL for a model (call after each use to implement Ollama-style keep-alive)
func (r *LazyEmbedderRegistry) Touch(modelName string) {
	if item := r.cache.Get(modelName); item != nil {
		// Get refreshes TTL automatically
		r.logger.Debug("Refreshed model keep-alive",
			zap.String("model", modelName))
	}
}

// List returns all available (discovered) model names
func (r *LazyEmbedderRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.discovered))
	for name := range r.discovered {
		names = append(names, name)
	}
	return names
}

// ListLoaded returns currently loaded model names (from cache and pinned)
func (r *LazyEmbedderRegistry) ListLoaded() []string {
	// Get cache keys
	keys := r.cache.Keys()

	// Add pinned models
	r.pinnedMu.RLock()
	pinnedNames := make([]string, 0, len(r.pinned))
	for name := range r.pinned {
		pinnedNames = append(pinnedNames, name)
	}
	r.pinnedMu.RUnlock()

	// Combine (pinned first, then cache)
	names := make([]string, 0, len(keys)+len(pinnedNames))
	names = append(names, pinnedNames...)
	names = append(names, keys...)
	return names
}

// IsLoaded checks if a model is currently loaded (in cache or pinned)
func (r *LazyEmbedderRegistry) IsLoaded(modelName string) bool {
	r.pinnedMu.RLock()
	isPinned := r.pinned[modelName] != nil
	r.pinnedMu.RUnlock()
	return isPinned || r.cache.Has(modelName)
}

// Unload explicitly unloads a model (triggers eviction callback)
// Note: Pinned models cannot be unloaded via this method.
func (r *LazyEmbedderRegistry) Unload(modelName string) {
	r.pinnedMu.RLock()
	isPinned := r.pinned[modelName] != nil
	r.pinnedMu.RUnlock()

	if isPinned {
		r.logger.Debug("Cannot unload pinned model",
			zap.String("model", modelName))
		return
	}
	r.cache.Delete(modelName)
}

// Pin marks a model as pinned (never evicted). If the model is already loaded
// in the cache, it is moved to the pinned map. If not loaded, it will be loaded
// first. Pinned models survive TTL expiration and LRU eviction.
func (r *LazyEmbedderRegistry) Pin(modelName string) error {
	// Check if already pinned
	r.pinnedMu.RLock()
	if r.pinned[modelName] != nil {
		r.pinnedMu.RUnlock()
		r.logger.Debug("Model already pinned",
			zap.String("model", modelName))
		return nil
	}
	r.pinnedMu.RUnlock()

	// Get the model (may load it if not already loaded)
	embedder, err := r.Get(modelName)
	if err != nil {
		return fmt.Errorf("pin model %s: %w", modelName, err)
	}

	// Move from cache to pinned map
	r.pinnedMu.Lock()
	r.pinned[modelName] = embedder
	r.pinnedMu.Unlock()

	// Remove from cache (without triggering close callback - we moved it)
	// We use DeleteAll pattern with a filter, but simpler is to just delete
	// and the eviction callback checks if it's now in pinned
	r.cache.Delete(modelName)

	r.logger.Info("Pinned model (will not be evicted)",
		zap.String("model", modelName))

	return nil
}

// IsPinned returns true if a model is pinned (never evicted)
func (r *LazyEmbedderRegistry) IsPinned(modelName string) bool {
	r.pinnedMu.RLock()
	defer r.pinnedMu.RUnlock()
	return r.pinned[modelName] != nil
}

// Preload loads specified models at startup to avoid first-request latency
func (r *LazyEmbedderRegistry) Preload(modelNames []string) error {
	if len(modelNames) == 0 {
		return nil
	}

	r.logger.Info("Preloading models", zap.Strings("models", modelNames))

	var loaded, failed int
	for _, name := range modelNames {
		if _, err := r.Get(name); err != nil {
			r.logger.Warn("Failed to preload model",
				zap.String("model", name),
				zap.Error(err))
			failed++
		} else {
			r.logger.Info("Preloaded model",
				zap.String("model", name))
			loaded++
		}
	}

	r.logger.Info("Preloading complete",
		zap.Int("loaded", loaded),
		zap.Int("failed", failed))

	if failed > 0 && loaded == 0 {
		return fmt.Errorf("all %d models failed to preload", failed)
	}

	return nil
}

// Close stops the cache and unloads all models (including pinned)
func (r *LazyEmbedderRegistry) Close() error {
	r.logger.Info("Closing lazy embedder registry")

	// Stop cache and delete all cached models
	r.cache.Stop()
	r.cache.DeleteAll()

	// Close all pinned models
	r.pinnedMu.Lock()
	for name, embedder := range r.pinned {
		r.logger.Debug("Closing pinned model",
			zap.String("model", name))
		if closer, ok := embedder.(interface{ Close() error }); ok {
			if err := closer.Close(); err != nil {
				r.logger.Warn("Error closing pinned embedder",
					zap.String("model", name),
					zap.Error(err))
			}
		}
	}
	r.pinned = make(map[string]embeddings.Embedder)
	r.pinnedMu.Unlock()

	return nil
}

// Stats returns cache statistics
func (r *LazyEmbedderRegistry) Stats() map[string]any {
	metrics := r.cache.Metrics()

	r.pinnedMu.RLock()
	pinnedCount := len(r.pinned)
	pinnedNames := make([]string, 0, pinnedCount)
	for name := range r.pinned {
		pinnedNames = append(pinnedNames, name)
	}
	r.pinnedMu.RUnlock()

	return map[string]any{
		"discovered":    len(r.discovered),
		"loaded":        r.cache.Len() + pinnedCount,
		"pinned":        pinnedCount,
		"pinned_models": pinnedNames,
		"cached":        r.cache.Len(),
		"hits":          metrics.Hits,
		"misses":        metrics.Misses,
		"keep_alive":    r.keepAlive.String(),
		"max_loaded":    r.maxLoadedModels,
		"loaded_models": r.ListLoaded(),
	}
}
