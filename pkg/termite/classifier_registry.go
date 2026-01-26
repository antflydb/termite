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

	"github.com/antflydb/termite/pkg/termite/lib/classification"
	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
	"github.com/jellydator/ttlcache/v3"
	"go.uber.org/zap"
)

// ClassifierModelInfo holds metadata about a discovered classifier model (not loaded yet)
type ClassifierModelInfo struct {
	Name         string
	Path         string
	OnnxFilename string
	PoolSize     int
}

// loadedClassifier wraps a loaded classifier
type loadedClassifier struct {
	classifier classification.Classifier
	config     classification.Config
}

// ClassifierRegistry manages zero-shot classification models with lazy loading and TTL-based unloading
type ClassifierRegistry struct {
	modelsDir      string
	sessionManager *backends.SessionManager
	logger         *zap.Logger

	// Model discovery (paths only, not loaded)
	discovered map[string]*ClassifierModelInfo
	mu         sync.RWMutex

	// Loaded models with TTL cache
	cache *ttlcache.Cache[string, *loadedClassifier]

	// Reference counting to prevent eviction during active use
	refCounts   map[string]int
	refCountsMu sync.Mutex

	// Configuration
	keepAlive       time.Duration
	maxLoadedModels uint64
	poolSize        int
}

// ClassifierConfig configures the classifier registry
type ClassifierConfig struct {
	ModelsDir       string
	KeepAlive       time.Duration // How long to keep models loaded (0 = forever)
	MaxLoadedModels uint64        // Max models in memory (0 = unlimited)
	PoolSize        int           // Number of concurrent pipelines per model (0 = default)
}

// NewClassifierRegistry creates a new lazy-loading classifier registry
func NewClassifierRegistry(
	config ClassifierConfig,
	sessionManager *backends.SessionManager,
	logger *zap.Logger,
) (*ClassifierRegistry, error) {
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

	registry := &ClassifierRegistry{
		modelsDir:       config.ModelsDir,
		sessionManager:  sessionManager,
		logger:          logger,
		discovered:      make(map[string]*ClassifierModelInfo),
		refCounts:       make(map[string]int),
		keepAlive:       keepAlive,
		maxLoadedModels: config.MaxLoadedModels,
		poolSize:        poolSize,
	}

	// Configure TTL cache with LRU eviction
	cacheOpts := []ttlcache.Option[string, *loadedClassifier]{
		ttlcache.WithTTL[string, *loadedClassifier](keepAlive),
	}

	if config.MaxLoadedModels > 0 {
		cacheOpts = append(cacheOpts,
			ttlcache.WithCapacity[string, *loadedClassifier](config.MaxLoadedModels))
	}

	registry.cache = ttlcache.New(cacheOpts...)

	// Set up eviction callback to close unloaded models
	registry.cache.OnEviction(func(ctx context.Context, reason ttlcache.EvictionReason, item *ttlcache.Item[string, *loadedClassifier]) {
		// Skip closing on manual deletion - Close() handles cleanup synchronously
		if reason == ttlcache.EvictionReasonDeleted {
			logger.Debug("Classifier model removed from cache (cleanup handled separately)",
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
			logger.Warn("Preventing eviction of classifier model with active references",
				zap.String("model", item.Key()),
				zap.Int("refCount", refCount),
				zap.String("reason", reasonStr))
			return
		}
		registry.refCountsMu.Unlock()

		logger.Info("Evicting classifier model from cache",
			zap.String("model", item.Key()),
			zap.String("reason", reasonStr))
		if err := item.Value().classifier.Close(); err != nil {
			logger.Warn("Error closing evicted classifier model",
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

	logger.Info("Lazy classifier registry initialized",
		zap.Int("models_discovered", len(registry.discovered)),
		zap.Duration("keep_alive", keepAlive),
		zap.Uint64("max_loaded_models", config.MaxLoadedModels))

	return registry, nil
}

// discoverModels finds all classifier models in the models directory without loading them
func (r *ClassifierRegistry) discoverModels() error {
	if r.modelsDir == "" {
		r.logger.Info("No classifier models directory configured")
		return nil
	}

	// Check if directory exists
	if _, err := os.Stat(r.modelsDir); os.IsNotExist(err) {
		r.logger.Warn("Classifier models directory does not exist",
			zap.String("dir", r.modelsDir))
		return nil
	}

	// Use discoverModelsInDir which handles owner/model structure
	discovered, err := discoverModelsInDir(r.modelsDir, modelregistry.ModelTypeClassifier, r.logger)
	if err != nil {
		return fmt.Errorf("discovering classifier models: %w", err)
	}

	// Pool size for concurrent pipeline access
	poolSize := r.poolSize

	for _, dm := range discovered {
		modelPath := dm.Path
		registryFullName := dm.FullName()

		// Check if this is a valid classifier model (has zsc_config.json or is NLI model)
		if !classification.IsClassifierModel(modelPath) {
			r.logger.Debug("Skipping non-classifier model",
				zap.String("dir", registryFullName))
			continue
		}

		// Discover all available model variants
		variants := dm.Variants
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
		r.logger.Info("Discovered classifier model (not loaded)",
			zap.String("name", registryFullName),
			zap.String("path", modelPath),
			zap.Strings("variants", variantIDs))

		// Store each variant for lazy loading
		for variantID, onnxFilename := range variants {
			// Determine registry name
			registryName := registryFullName
			if variantID != "" {
				registryName = registryFullName + ":" + variantID
			}

			r.discovered[registryName] = &ClassifierModelInfo{
				Name:         registryName,
				Path:         modelPath,
				OnnxFilename: onnxFilename,
				PoolSize:     poolSize,
			}
		}
	}

	r.logger.Info("Classifier model discovery complete",
		zap.Int("models_discovered", len(r.discovered)),
		zap.Duration("keep_alive", r.keepAlive),
		zap.Uint64("max_loaded_models", r.maxLoadedModels))

	return nil
}

// Get returns a classifier model by name, loading it if necessary.
// DEPRECATED: Use Acquire() instead for long-running operations to prevent
// the model from being evicted during use. Get() does not track usage and
// the returned classifier may be closed if the cache evicts it.
func (r *ClassifierRegistry) Get(modelName string) (classification.Classifier, error) {
	loaded, err := r.getLoaded(modelName)
	if err != nil {
		return nil, err
	}
	return loaded.classifier, nil
}

// Acquire returns a classifier by name and increments its reference count.
// The caller MUST call Release() when done to allow the model to be evicted.
// This prevents the model from being closed while in use.
func (r *ClassifierRegistry) Acquire(modelName string) (*loadedClassifier, error) {
	loaded, err := r.getLoaded(modelName)
	if err != nil {
		return nil, err
	}

	r.refCountsMu.Lock()
	r.refCounts[modelName]++
	count := r.refCounts[modelName]
	r.refCountsMu.Unlock()

	r.logger.Debug("Acquired classifier model",
		zap.String("model", modelName),
		zap.Int("refCount", count))

	return loaded, nil
}

// Release decrements the reference count for a model.
// Must be called after Acquire() when the caller is done using the classifier.
func (r *ClassifierRegistry) Release(modelName string) {
	r.refCountsMu.Lock()
	if r.refCounts[modelName] > 0 {
		r.refCounts[modelName]--
	}
	count := r.refCounts[modelName]
	r.refCountsMu.Unlock()

	r.logger.Debug("Released classifier model",
		zap.String("model", modelName),
		zap.Int("refCount", count))
}

// getLoaded gets or loads a model from cache
func (r *ClassifierRegistry) getLoaded(modelName string) (*loadedClassifier, error) {
	// Check cache first
	if item := r.cache.Get(modelName); item != nil {
		r.logger.Debug("Classifier cache hit", zap.String("model", modelName))
		return item.Value(), nil
	}

	// Check if model is discovered
	r.mu.RLock()
	info, ok := r.discovered[modelName]
	r.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("classifier model not found: %s", modelName)
	}

	// Load the model
	return r.loadModel(info)
}

// loadModel loads a classifier model from disk
func (r *ClassifierRegistry) loadModel(info *ClassifierModelInfo) (*loadedClassifier, error) {
	r.logger.Info("Loading classifier model on demand",
		zap.String("model", info.Name),
		zap.String("path", info.Path))

	// Load using pipeline-based classifier
	cfg := classification.PooledClassifierConfig{
		ModelPath:     info.Path,
		PoolSize:      info.PoolSize,
		ModelBackends: nil, // Use all available backends
		Logger:        r.logger.Named(info.Name),
	}
	model, backendUsed, err := classification.NewPooledClassifier(cfg, r.sessionManager)
	if err != nil {
		return nil, fmt.Errorf("loading classifier model %s: %w", info.Name, err)
	}

	r.logger.Info("Successfully loaded classifier model",
		zap.String("name", info.Name),
		zap.String("backend", string(backendUsed)),
		zap.Int("poolSize", info.PoolSize))

	loaded := &loadedClassifier{
		classifier: model,
		config:     model.Config(),
	}

	// Add to cache
	r.cache.Set(info.Name, loaded, r.keepAlive)

	return loaded, nil
}

// List returns all available classifier model names (discovered, not necessarily loaded)
func (r *ClassifierRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.discovered))
	for name := range r.discovered {
		names = append(names, name)
	}
	return names
}

// ListLoaded returns only the currently loaded classifier model names
func (r *ClassifierRegistry) ListLoaded() []string {
	return r.cache.Keys()
}

// IsLoaded returns whether a model is currently loaded in memory
func (r *ClassifierRegistry) IsLoaded(modelName string) bool {
	return r.cache.Has(modelName)
}

// Preload loads specified models at startup to avoid first-request latency
func (r *ClassifierRegistry) Preload(modelNames []string) error {
	if len(modelNames) == 0 {
		return nil
	}

	r.logger.Info("Preloading classifier models", zap.Strings("models", modelNames))

	var loaded, failed int
	for _, name := range modelNames {
		if _, err := r.Get(name); err != nil {
			r.logger.Warn("Failed to preload classifier model",
				zap.String("model", name),
				zap.Error(err))
			failed++
		} else {
			loaded++
		}
	}

	r.logger.Info("Classifier model preloading complete",
		zap.Int("loaded", loaded),
		zap.Int("failed", failed))

	return nil
}

// Close closes all loaded models and stops the cache
func (r *ClassifierRegistry) Close() error {
	r.logger.Info("Closing classifier registry")

	// Close all loaded models
	for _, name := range r.cache.Keys() {
		if item := r.cache.Get(name); item != nil {
			if err := item.Value().classifier.Close(); err != nil {
				r.logger.Warn("Error closing classifier model",
					zap.String("model", name),
					zap.Error(err))
			}
			r.cache.Delete(name)
		}
	}

	// Stop the cache cleanup goroutine
	r.cache.Stop()

	return nil
}
