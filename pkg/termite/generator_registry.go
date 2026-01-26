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
	"sync"
	"time"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/generation"
	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
	"github.com/jellydator/ttlcache/v3"
	"go.uber.org/zap"
)

// GeneratorModelInfo holds metadata about a discovered generator model (not loaded yet)
type GeneratorModelInfo struct {
	Name      string
	Path      string // Path to base variant
	ModelType string
	Variants  map[string]string // variant name -> path (e.g., "i4" -> "/path/to/model/i4")
}

// GeneratorRegistry manages generator models with lazy loading and TTL-based unloading
type GeneratorRegistry struct {
	modelsDir      string
	sessionManager *backends.SessionManager
	logger         *zap.Logger

	// Model discovery (paths only, not loaded)
	discovered map[string]*GeneratorModelInfo
	mu         sync.RWMutex

	// Loaded models with TTL cache
	cache *ttlcache.Cache[string, generation.Generator]

	// Reference counting to prevent eviction during active use
	refCounts   map[string]int
	refCountsMu sync.Mutex

	// Configuration
	keepAlive       time.Duration
	maxLoadedModels uint64
}

// GeneratorConfig configures the generator registry
type GeneratorConfig struct {
	ModelsDir       string
	KeepAlive       time.Duration // How long to keep models loaded (0 = forever)
	MaxLoadedModels uint64        // Max models in memory (0 = unlimited)
}

// NewGeneratorRegistry creates a new lazy-loading generator registry
func NewGeneratorRegistry(
	config GeneratorConfig,
	sessionManager *backends.SessionManager,
	logger *zap.Logger,
) (*GeneratorRegistry, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	keepAlive := config.KeepAlive
	if keepAlive == 0 {
		keepAlive = ttlcache.NoTTL // Never expire
	}

	registry := &GeneratorRegistry{
		modelsDir:       config.ModelsDir,
		sessionManager:  sessionManager,
		logger:          logger,
		discovered:      make(map[string]*GeneratorModelInfo),
		refCounts:       make(map[string]int),
		keepAlive:       keepAlive,
		maxLoadedModels: config.MaxLoadedModels,
	}

	// Configure TTL cache with LRU eviction
	cacheOpts := []ttlcache.Option[string, generation.Generator]{
		ttlcache.WithTTL[string, generation.Generator](keepAlive),
	}

	if config.MaxLoadedModels > 0 {
		cacheOpts = append(cacheOpts,
			ttlcache.WithCapacity[string, generation.Generator](config.MaxLoadedModels))
	}

	registry.cache = ttlcache.New(cacheOpts...)

	// Set up eviction callback to close unloaded models
	// Note: Only close on TTL expiration or capacity eviction, not on manual deletion
	// (manual deletion during Close() handles cleanup synchronously)
	registry.cache.OnEviction(func(ctx context.Context, reason ttlcache.EvictionReason, item *ttlcache.Item[string, generation.Generator]) {
		// Skip closing on manual deletion - Close() handles cleanup synchronously
		if reason == ttlcache.EvictionReasonDeleted {
			logger.Debug("Generator model removed from cache (cleanup handled separately)",
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
			// Model is still in use - re-added to cache to prevent closing
			logger.Warn("Preventing eviction of generator model with active references",
				zap.String("model", item.Key()),
				zap.Int("refCount", refCount),
				zap.String("reason", reasonStr))
			return
		}
		registry.refCountsMu.Unlock()

		logger.Info("Evicting generator model from cache",
			zap.String("model", item.Key()),
			zap.String("reason", reasonStr))
		if closer, ok := item.Value().(interface{ Close() error }); ok {
			if err := closer.Close(); err != nil {
				logger.Warn("Error closing evicted generator model",
					zap.String("model", item.Key()),
					zap.Error(err))
			}
		}
	})

	// Start cache cleanup goroutine
	go registry.cache.Start()

	// Discover models (but don't load them)
	if err := registry.discoverModels(); err != nil {
		registry.cache.Stop()
		return nil, err
	}

	logger.Info("Lazy generator registry initialized",
		zap.Int("models_discovered", len(registry.discovered)),
		zap.Duration("keep_alive", keepAlive),
		zap.Uint64("max_loaded_models", config.MaxLoadedModels))

	return registry, nil
}

// discoverModels finds all generator models in the models directory without loading them
func (r *GeneratorRegistry) discoverModels() error {
	if r.modelsDir == "" {
		r.logger.Info("No generator models directory configured")
		return nil
	}

	// Check if directory exists
	if _, err := os.Stat(r.modelsDir); os.IsNotExist(err) {
		r.logger.Warn("Generator models directory does not exist",
			zap.String("dir", r.modelsDir))
		return nil
	}

	// Use discoverModelsInDir which handles owner/model structure
	discovered, err := discoverModelsInDir(r.modelsDir, modelregistry.ModelTypeGenerator, r.logger)
	if err != nil {
		return fmt.Errorf("discovering generator models: %w", err)
	}

	// Known variant subdirectory names for generators
	knownVariants := []string{"i4", "i4-cuda", "i4-dml"}

	for _, dm := range discovered {
		modelPath := dm.Path
		registryFullName := dm.FullName()

		// Check for genai_config.json (preferred) or model.onnx in root or onnx/ subdirectory
		if !isValidGeneratorModel(modelPath) {
			// Also check onnx/ subdirectory
			onnxSubpath := filepath.Join(modelPath, "onnx")
			if isValidGeneratorModel(onnxSubpath) {
				modelPath = onnxSubpath
			} else {
				r.logger.Debug("Skipping directory - not a valid generator model",
					zap.String("dir", registryFullName))
				continue
			}
		}

		// Discover available variant subdirectories
		variants := make(map[string]string)
		for _, variantName := range knownVariants {
			variantPath := filepath.Join(dm.Path, variantName)
			if isValidGeneratorModel(variantPath) {
				variants[variantName] = variantPath
				r.logger.Debug("Found generator variant",
					zap.String("model", registryFullName),
					zap.String("variant", variantName),
					zap.String("path", variantPath))
			}
		}

		r.logger.Info("Discovered generator model (not loaded)",
			zap.String("name", registryFullName),
			zap.String("path", modelPath),
			zap.Int("variants", len(variants)))

		r.discovered[registryFullName] = &GeneratorModelInfo{
			Name:     registryFullName,
			Path:     modelPath,
			Variants: variants,
		}
	}

	r.logger.Info("Generator model discovery complete",
		zap.Int("models_discovered", len(r.discovered)),
		zap.Duration("keep_alive", r.keepAlive),
		zap.Uint64("max_loaded_models", r.maxLoadedModels))

	return nil
}

// Get returns a generator by name, loading it if necessary.
// DEPRECATED: Use Acquire() instead for long-running operations to prevent
// the model from being evicted during use. Get() does not track usage and
// the returned generator may be closed if the cache evicts it.
func (r *GeneratorRegistry) Get(modelName string) (generation.Generator, error) {
	// Check cache first
	if item := r.cache.Get(modelName); item != nil {
		r.logger.Debug("Generator cache hit", zap.String("model", modelName))
		return item.Value(), nil
	}

	// Check if model is discovered
	r.mu.RLock()
	info, ok := r.discovered[modelName]
	r.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("generator model not found: %s", modelName)
	}

	// Load the model
	return r.loadModel(info)
}

// Acquire returns a generator by name and increments its reference count.
// The caller MUST call Release() when done to allow the model to be evicted.
// This prevents the model from being closed while in use.
func (r *GeneratorRegistry) Acquire(modelName string) (generation.Generator, error) {
	gen, err := r.Get(modelName)
	if err != nil {
		return nil, err
	}

	r.refCountsMu.Lock()
	r.refCounts[modelName]++
	count := r.refCounts[modelName]
	r.refCountsMu.Unlock()

	r.logger.Debug("Acquired generator model",
		zap.String("model", modelName),
		zap.Int("refCount", count))

	return gen, nil
}

// Release decrements the reference count for a model.
// Must be called after Acquire() when the caller is done using the generator.
func (r *GeneratorRegistry) Release(modelName string) {
	r.refCountsMu.Lock()
	if r.refCounts[modelName] > 0 {
		r.refCounts[modelName]--
	}
	count := r.refCounts[modelName]
	r.refCountsMu.Unlock()

	r.logger.Debug("Released generator model",
		zap.String("model", modelName),
		zap.Int("refCount", count))
}

// AcquireWithVariant returns a generator by name with a specific variant
// and increments its reference count.
// The caller MUST call ReleaseWithVariant() when done.
func (r *GeneratorRegistry) AcquireWithVariant(modelName, variant string) (generation.Generator, error) {
	gen, err := r.GetWithVariant(modelName, variant)
	if err != nil {
		return nil, err
	}

	// Build cache key including variant
	cacheKey := modelName
	if variant != "" {
		cacheKey = modelName + ":" + variant
	}

	r.refCountsMu.Lock()
	r.refCounts[cacheKey]++
	count := r.refCounts[cacheKey]
	r.refCountsMu.Unlock()

	r.logger.Debug("Acquired generator model with variant",
		zap.String("model", cacheKey),
		zap.Int("refCount", count))

	return gen, nil
}

// ReleaseWithVariant decrements the reference count for a model variant.
func (r *GeneratorRegistry) ReleaseWithVariant(modelName, variant string) {
	// Build cache key including variant
	cacheKey := modelName
	if variant != "" {
		cacheKey = modelName + ":" + variant
	}

	r.refCountsMu.Lock()
	if r.refCounts[cacheKey] > 0 {
		r.refCounts[cacheKey]--
	}
	count := r.refCounts[cacheKey]
	r.refCountsMu.Unlock()

	r.logger.Debug("Released generator model with variant",
		zap.String("model", cacheKey),
		zap.Int("refCount", count))
}

// loadModel loads a generator model from disk (base variant)
func (r *GeneratorRegistry) loadModel(info *GeneratorModelInfo) (generation.Generator, error) {
	return r.loadModelFromPath(info.Name, info.Path)
}

// loadModelFromPath loads a generator model from a specific path
func (r *GeneratorRegistry) loadModelFromPath(cacheKey, modelPath string) (generation.Generator, error) {
	r.logger.Info("Loading generator model on demand",
		zap.String("cacheKey", cacheKey),
		zap.String("path", modelPath))

	// Try to generate genai_config.json if needed
	if err := generateGenaiConfig(modelPath, r.logger); err != nil {
		r.logger.Warn("Failed to generate genai_config.json",
			zap.String("cacheKey", cacheKey),
			zap.Error(err))
	}

	// Load the generator model
	// This will try the pipeline-based approach first, then fall back to ortgenai
	model, backendUsed, loadErr := generation.LoadGenerator(
		modelPath,
		1, // Use single pipeline, registry manages caching
		r.logger.Named(cacheKey),
		r.sessionManager,
		[]string{"onnx"}, // Generative models currently only support ONNX
	)

	if loadErr != nil {
		return nil, fmt.Errorf("loading generator model %s: %w", cacheKey, loadErr)
	}

	r.logger.Info("Successfully loaded generator model",
		zap.String("cacheKey", cacheKey),
		zap.String("backend", string(backendUsed)))

	// Add to cache
	r.cache.Set(cacheKey, model, r.keepAlive)

	return model, nil
}

// GetWithVariant returns a generator by name with a specific variant, loading it if necessary.
// If variant is empty, the base model is loaded.
func (r *GeneratorRegistry) GetWithVariant(modelName, variant string) (generation.Generator, error) {
	// Build cache key including variant
	cacheKey := modelName
	if variant != "" {
		cacheKey = modelName + ":" + variant
	}

	// Check cache first
	if item := r.cache.Get(cacheKey); item != nil {
		r.logger.Debug("Generator cache hit", zap.String("model", cacheKey))
		return item.Value(), nil
	}

	// Check if model is discovered
	r.mu.RLock()
	info, ok := r.discovered[modelName]
	r.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("generator model not found: %s", modelName)
	}

	// Determine path based on variant
	modelPath := info.Path
	if variant != "" {
		variantPath, ok := info.Variants[variant]
		if !ok {
			return nil, fmt.Errorf("variant %q not found for model %s (available: %v)",
				variant, modelName, r.ListVariants(modelName))
		}
		modelPath = variantPath
	}

	// Load the model with the specific path
	return r.loadModelFromPath(cacheKey, modelPath)
}

// ListVariants returns the available variant names for a model
func (r *GeneratorRegistry) ListVariants(modelName string) []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	info, ok := r.discovered[modelName]
	if !ok {
		return nil
	}

	variants := make([]string, 0, len(info.Variants))
	for name := range info.Variants {
		variants = append(variants, name)
	}
	return variants
}

// List returns all available generator model names (discovered, not necessarily loaded)
func (r *GeneratorRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.discovered))
	for name := range r.discovered {
		names = append(names, name)
	}
	return names
}

// ListLoaded returns only the currently loaded generator model names
func (r *GeneratorRegistry) ListLoaded() []string {
	keys := r.cache.Keys()
	return keys
}

// IsLoaded returns whether a model is currently loaded in memory
func (r *GeneratorRegistry) IsLoaded(modelName string) bool {
	return r.cache.Has(modelName)
}

// Preload loads specified models at startup to avoid first-request latency
func (r *GeneratorRegistry) Preload(modelNames []string) error {
	if len(modelNames) == 0 {
		return nil
	}

	r.logger.Info("Preloading generator models", zap.Strings("models", modelNames))

	var loaded, failed int
	for _, name := range modelNames {
		if _, err := r.Get(name); err != nil {
			r.logger.Warn("Failed to preload generator model",
				zap.String("model", name),
				zap.Error(err))
			failed++
		} else {
			r.logger.Info("Preloaded generator model",
				zap.String("model", name))
			loaded++
		}
	}

	r.logger.Info("Generator preloading complete",
		zap.Int("loaded", loaded),
		zap.Int("failed", failed))

	if failed > 0 && loaded == 0 {
		return fmt.Errorf("all %d generator models failed to preload", failed)
	}

	return nil
}

// PreloadAll loads all discovered models (for eager loading mode)
func (r *GeneratorRegistry) PreloadAll() error {
	return r.Preload(r.List())
}

// Close stops the cache and unloads all models
func (r *GeneratorRegistry) Close() error {
	r.logger.Info("Closing lazy generator registry")

	// Stop cache first to prevent new evictions
	r.cache.Stop()

	// Close all cached models synchronously (don't rely on async eviction callbacks)
	for _, key := range r.cache.Keys() {
		if item := r.cache.Get(key); item != nil {
			model := item.Value()
			r.logger.Debug("Closing cached generator model",
				zap.String("model", key))
			if closer, ok := model.(interface{ Close() error }); ok {
				if err := closer.Close(); err != nil {
					r.logger.Warn("Error closing generator model",
						zap.String("model", key),
						zap.Error(err))
				}
			}
		}
	}

	// Clear the cache (eviction callbacks won't close since reason is EvictionReasonDeleted)
	r.cache.DeleteAll()

	return nil
}
