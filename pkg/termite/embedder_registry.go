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
	"strings"
	"sync"
	"time"

	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/backends"
	termembeddings "github.com/antflydb/termite/pkg/termite/lib/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
	"github.com/jellydator/ttlcache/v3"
	"go.uber.org/zap"
)

// Default keep-alive duration (matches Ollama's 5-minute default)
const DefaultKeepAlive = 5 * time.Minute

// ModelInfo holds metadata about a discovered model (not loaded yet)
type ModelInfo struct {
	Name             string
	Path             string
	OnnxFilename     string // e.g., "model.onnx", "model_f16.onnx", "model_i8.onnx"
	PoolSize         int
	ModelType        string   // "embedder", "chunker", "reranker"
	Variants         []string // Available variant IDs (e.g., ["f16", "i8"])
	RequiredBackends []string // If set, only use these backends (e.g., ["onnx"] for models with XLA-incompatible ops)
}

// modelsRequiringONNX lists model name patterns that require the ONNX backend.
// These models use ONNX ops that aren't supported by the XLA/GoMLX backend.
// This is a fallback for models without manifest backend specifications.
var modelsRequiringONNX = []string{
	"nomic-ai/nomic-embed-text-v1.5", // Uses dynamic Range op in rotary embeddings
}

// getRequiredBackends returns the required backends for a model.
// Priority: manifest.Backends > hardcoded patterns > nil (all backends)
func getRequiredBackends(modelName string, manifest *modelregistry.ModelManifest) []string {
	// First check manifest if present
	if manifest != nil && len(manifest.Backends) > 0 {
		return manifest.Backends
	}

	// Fall back to hardcoded patterns
	for _, pattern := range modelsRequiringONNX {
		if strings.Contains(modelName, pattern) {
			return []string{"onnx"}
		}
	}
	return nil
}

// EmbedderRegistry manages embedding models with lazy loading and TTL-based unloading
type EmbedderRegistry struct {
	modelsDir      string
	sessionManager *backends.SessionManager
	logger         *zap.Logger

	// Model discovery (paths only, not loaded)
	discovered map[string]*ModelInfo
	mu         sync.RWMutex

	// Loaded models with TTL cache (for lazy models)
	cache *ttlcache.Cache[string, embeddings.Embedder]

	// Reference counting to prevent eviction during active use
	refCounts   map[string]int
	refCountsMu sync.Mutex

	// Pinned models (never evicted, stored separately from cache)
	pinned   map[string]embeddings.Embedder
	pinnedMu sync.RWMutex

	// Configuration
	keepAlive       time.Duration
	maxLoadedModels uint64
	poolSize        int
}

// EmbedderConfig configures the embedder registry
type EmbedderConfig struct {
	ModelsDir       string
	KeepAlive       time.Duration // How long to keep models loaded (0 = forever)
	MaxLoadedModels uint64        // Max models in memory (0 = unlimited)
	PoolSize        int           // Number of concurrent pipelines per model (0 = default)
}

// NewEmbedderRegistry creates a new lazy-loading embedder registry
func NewEmbedderRegistry(
	config EmbedderConfig,
	sessionManager *backends.SessionManager,
	logger *zap.Logger,
) (*EmbedderRegistry, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	keepAlive := config.KeepAlive
	if keepAlive == 0 {
		keepAlive = ttlcache.NoTTL // Never expire
	}

	registry := &EmbedderRegistry{
		modelsDir:       config.ModelsDir,
		sessionManager:  sessionManager,
		logger:          logger,
		discovered:      make(map[string]*ModelInfo),
		refCounts:       make(map[string]int),
		pinned:          make(map[string]embeddings.Embedder),
		keepAlive:       keepAlive,
		maxLoadedModels: config.MaxLoadedModels,
		poolSize:        config.PoolSize,
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
	// Note: Only close on TTL expiration or capacity eviction, not on manual deletion
	// (manual deletion during Close() handles cleanup synchronously)
	registry.cache.OnEviction(func(ctx context.Context, reason ttlcache.EvictionReason, item *ttlcache.Item[string, embeddings.Embedder]) {
		modelName := item.Key()
		embedder := item.Value()

		// Skip closing on manual deletion - Close() handles cleanup synchronously.
		// Don't log here since ttlcache runs eviction callbacks in goroutines,
		// which can cause panics if the logger (e.g., test logger) is closed.
		if reason == ttlcache.EvictionReasonDeleted {
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
		refCount := registry.refCounts[modelName]
		if refCount > 0 {
			// Re-add while still holding lock to prevent race with Release()
			registry.cache.Set(modelName, embedder, registry.keepAlive)
			registry.refCountsMu.Unlock()
			logger.Warn("Preventing eviction of embedder model with active references",
				zap.String("model", modelName),
				zap.Int("refCount", refCount),
				zap.String("reason", reasonStr))
			return
		}
		registry.refCountsMu.Unlock()

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
// Supports owner/model-name directory structure (e.g., embedders/BAAI/bge-small-en-v1.5/)
func (r *EmbedderRegistry) discoverModels() error {
	if r.modelsDir == "" {
		r.logger.Info("No embedder models directory configured")
		return nil
	}

	if _, err := os.Stat(r.modelsDir); os.IsNotExist(err) {
		r.logger.Warn("Embedder models directory does not exist",
			zap.String("dir", r.modelsDir))
		return nil
	}

	// Use discoverModelsInDir which handles owner/model structure
	discovered, err := discoverModelsInDir(r.modelsDir, "embedder", r.logger)
	if err != nil {
		return fmt.Errorf("discovering embedder models: %w", err)
	}

	poolSize := r.poolSize
	if poolSize <= 0 {
		poolSize = min(runtime.NumCPU(), 4)
	}

	for _, dm := range discovered {
		modelPath := dm.Path
		registryFullName := dm.FullName()
		variants := dm.Variants

		// Check if this is a multimodal (CLIP-style) model
		hasMultimodalStd, hasMultimodalQt := isMultimodalModel(modelPath)
		if hasMultimodalStd || hasMultimodalQt {
			r.logger.Info("Discovered multimodal embedder model (not loaded)",
				zap.String("name", registryFullName),
				zap.String("path", modelPath),
				zap.Bool("has_standard", hasMultimodalStd),
				zap.Bool("has_quantized", hasMultimodalQt))

			// Register standard precision multimodal model
			if hasMultimodalStd {
				r.discovered[registryFullName] = &ModelInfo{
					Name:             registryFullName,
					Path:             modelPath,
					OnnxFilename:     "", // CLIP uses multiple files, not a single ONNX
					PoolSize:         poolSize,
					ModelType:        "clip",
					Variants:         []string{"default"},
					RequiredBackends: getRequiredBackends(registryFullName, dm.Manifest),
				}
			}

			// Register quantized multimodal model with suffix
			if hasMultimodalQt {
				quantizedName := registryFullName + "-i8-qt"
				r.discovered[quantizedName] = &ModelInfo{
					Name:             quantizedName,
					Path:             modelPath,
					OnnxFilename:     "", // CLIP uses multiple files, not a single ONNX
					PoolSize:         poolSize,
					ModelType:        "clip-quantized",
					Variants:         []string{"quantized"},
					RequiredBackends: getRequiredBackends(registryFullName, dm.Manifest),
				}
			}
			continue // Skip standard embedder handling
		}

		if len(variants) == 0 {
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
			zap.String("name", registryFullName),
			zap.String("path", modelPath),
			zap.Strings("variants", variantIDs))

		// Register each variant
		for variantID, onnxFilename := range variants {
			registryName := registryFullName
			if variantID != "" {
				registryName = registryFullName + "-" + variantID
			}

			r.discovered[registryName] = &ModelInfo{
				Name:             registryName,
				Path:             modelPath,
				OnnxFilename:     onnxFilename,
				PoolSize:         poolSize,
				ModelType:        "embedder",
				Variants:         variantIDs,
				RequiredBackends: getRequiredBackends(registryFullName, dm.Manifest),
			}
		}
	}

	r.logger.Info("Embedder model discovery complete",
		zap.Int("models_discovered", len(r.discovered)),
		zap.Duration("keep_alive", r.keepAlive),
		zap.Uint64("max_loaded_models", r.maxLoadedModels))

	return nil
}

// Get returns an embedder by model name, loading it if necessary.
// DEPRECATED: Use Acquire() instead for long-running operations to prevent
// the model from being evicted during use. Get() does not track usage and
// the returned embedder may be closed if the cache evicts it.
func (r *EmbedderRegistry) Get(modelName string) (embeddings.Embedder, error) {
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

// Acquire returns an embedder by model name and increments its reference count.
// The caller MUST call Release() when done to allow the model to be evicted.
// This prevents the model from being closed while in use.
func (r *EmbedderRegistry) Acquire(modelName string) (embeddings.Embedder, error) {
	embedder, err := r.Get(modelName)
	if err != nil {
		return nil, err
	}

	r.refCountsMu.Lock()
	r.refCounts[modelName]++
	count := r.refCounts[modelName]
	r.refCountsMu.Unlock()

	r.logger.Debug("Acquired embedder model",
		zap.String("model", modelName),
		zap.Int("refCount", count))

	return embedder, nil
}

// Release decrements the reference count for a model.
// Must be called after Acquire() when the caller is done using the embedder.
func (r *EmbedderRegistry) Release(modelName string) {
	r.refCountsMu.Lock()
	if r.refCounts[modelName] > 0 {
		r.refCounts[modelName]--
	}
	count := r.refCounts[modelName]
	r.refCountsMu.Unlock()

	r.logger.Debug("Released embedder model",
		zap.String("model", modelName),
		zap.Int("refCount", count))
}

// loadModel loads a model on demand
func (r *EmbedderRegistry) loadModel(info *ModelInfo) (embeddings.Embedder, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Double-check cache after acquiring lock
	if item := r.cache.Get(info.Name); item != nil {
		return item.Value(), nil
	}

	r.logger.Info("Loading embedder model on demand",
		zap.String("model", info.Name),
		zap.String("path", info.Path),
		zap.String("model_type", info.ModelType),
		zap.String("onnx_filename", info.OnnxFilename),
		zap.Int("pool_size", info.PoolSize))

	var embedder embeddings.Embedder
	var backendUsed backends.BackendType
	var err error

	// Handle different model types
	switch info.ModelType {
	case "clip":
		// Load standard precision CLIP multimodal model
		embedder, backendUsed, err = termembeddings.NewCLIPEmbedder(
			info.Path,
			false, // not quantized
			r.sessionManager,
			nil, // modelBackends - use default priority
			r.logger.Named(info.Name),
		)
	case "clip-quantized":
		// Load quantized CLIP multimodal model
		embedder, backendUsed, err = termembeddings.NewCLIPEmbedder(
			info.Path,
			true, // quantized
			r.sessionManager,
			nil, // modelBackends - use default priority
			r.logger.Named(info.Name),
		)
	default:
		// Standard pooled embedder using pipeline
		cfg := termembeddings.PooledEmbedderConfig{
			ModelPath:     info.Path,
			PoolSize:      info.PoolSize,
			Normalize:     true,                  // Enable L2 normalization for unit-length embeddings
			ModelBackends: info.RequiredBackends, // nil = all backends, or specific backends for compatibility
			Logger:        r.logger.Named(info.Name),
		}
		embedder, backendUsed, err = termembeddings.NewPooledEmbedder(cfg, r.sessionManager)
	}

	if err != nil {
		r.logger.Error("Failed to load embedder model",
			zap.String("model", info.Name),
			zap.String("model_type", info.ModelType),
			zap.Error(err))
		return nil, fmt.Errorf("loading embedder model %s: %w", info.Name, err)
	}

	// Store in cache with TTL
	r.cache.Set(info.Name, embedder, ttlcache.DefaultTTL)

	r.logger.Info("Successfully loaded embedder model",
		zap.String("model", info.Name),
		zap.String("model_type", info.ModelType),
		zap.String("backend", string(backendUsed)),
		zap.Duration("keep_alive", r.keepAlive))

	return embedder, nil
}

// Touch refreshes the TTL for a model (call after each use to implement Ollama-style keep-alive)
func (r *EmbedderRegistry) Touch(modelName string) {
	if item := r.cache.Get(modelName); item != nil {
		// Get refreshes TTL automatically
		r.logger.Debug("Refreshed model keep-alive",
			zap.String("model", modelName))
	}
}

// List returns all available (discovered) model names
func (r *EmbedderRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.discovered))
	for name := range r.discovered {
		names = append(names, name)
	}
	return names
}

// ListLoaded returns currently loaded model names (from cache and pinned)
func (r *EmbedderRegistry) ListLoaded() []string {
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
func (r *EmbedderRegistry) IsLoaded(modelName string) bool {
	r.pinnedMu.RLock()
	isPinned := r.pinned[modelName] != nil
	r.pinnedMu.RUnlock()
	return isPinned || r.cache.Has(modelName)
}

// Unload explicitly unloads a model (triggers eviction callback)
// Note: Pinned models cannot be unloaded via this method.
func (r *EmbedderRegistry) Unload(modelName string) {
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
func (r *EmbedderRegistry) Pin(modelName string) error {
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
func (r *EmbedderRegistry) IsPinned(modelName string) bool {
	r.pinnedMu.RLock()
	defer r.pinnedMu.RUnlock()
	return r.pinned[modelName] != nil
}

// Preload loads specified models at startup to avoid first-request latency
func (r *EmbedderRegistry) Preload(modelNames []string) error {
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
func (r *EmbedderRegistry) Close() error {
	r.logger.Info("Closing lazy embedder registry")

	// Stop cache first to prevent new evictions
	r.cache.Stop()

	// Close all cached models synchronously (don't rely on async eviction callbacks)
	for _, key := range r.cache.Keys() {
		if item := r.cache.Get(key); item != nil {
			embedder := item.Value()
			r.logger.Debug("Closing cached embedder",
				zap.String("model", key))
			if closer, ok := embedder.(interface{ Close() error }); ok {
				if err := closer.Close(); err != nil {
					r.logger.Warn("Error closing embedder",
						zap.String("model", key),
						zap.Error(err))
				}
			}
		}
	}

	// Clear the cache (eviction callbacks won't close since reason is EvictionReasonDeleted)
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
func (r *EmbedderRegistry) Stats() map[string]any {
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
