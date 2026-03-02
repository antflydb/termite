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

	"slices"

	"github.com/antflydb/antfly-go/libaf/chunking"
	"github.com/antflydb/termite/pkg/termite/lib/backends"
	termchunking "github.com/antflydb/termite/pkg/termite/lib/chunking"
	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
	"github.com/jellydator/ttlcache/v3"
	"go.uber.org/zap"
)

// ChunkerModelInfo holds metadata about a discovered chunker model (not loaded yet)
type ChunkerModelInfo struct {
	Name         string
	Path         string
	OnnxFilename string                   // Text chunkers (variant ONNX file)
	PoolSize     int                      // Pipeline pool size (text chunkers)
	Capabilities []string                 // From manifest (e.g., ["audio"])
	Backends     []string                 // Required backends from manifest
	SessionOpts  []backends.SessionOption // For GoMLX backend support
}

// mediaCapabilities are the capabilities that qualify a chunker model as a media chunker.
var mediaCapabilities = []modelregistry.Capability{
	modelregistry.CapabilityAudio,
}

// ChunkerRegistry manages chunker models with lazy loading and TTL-based unloading.
// It handles both text chunkers and media chunkers (audio, etc.) in a single registry
// with capability-based dispatch, following the EmbedderRegistry pattern.
type ChunkerRegistry struct {
	modelsDir      string
	sessionManager *backends.SessionManager
	logger         *zap.Logger

	// Model discovery (paths only, not loaded)
	discovered map[string]*ChunkerModelInfo
	mu         sync.RWMutex

	// Loaded text chunker models with TTL cache
	cache *ttlcache.Cache[string, chunking.Chunker]

	// Loaded media chunker models with TTL cache (separate Go type, like embedder sparse cache)
	mediaCache *ttlcache.Cache[string, termchunking.CloseableMediaChunker]

	// Reference counting to prevent eviction during active use
	// Shared across text and media caches (model names are unique)
	refCounts   map[string]int
	refCountsMu sync.Mutex

	// Configuration
	keepAlive       time.Duration
	mediaKeepAlive  time.Duration // Separate TTL for media models (may differ from text)
	maxLoadedModels uint64
	poolSize        int
}

// ChunkerConfig configures the lazy chunker registry
type ChunkerConfig struct {
	ModelsDir       string
	KeepAlive       time.Duration // How long to keep text models loaded (0 = forever)
	MediaKeepAlive  time.Duration // How long to keep media models loaded (0 = use KeepAlive)
	MaxLoadedModels uint64        // Max models in memory (0 = unlimited)
	PoolSize        int           // Number of concurrent pipelines per model (0 = default)
}

// NewChunkerRegistry creates a new lazy-loading chunker registry
func NewChunkerRegistry(
	config ChunkerConfig,
	sessionManager *backends.SessionManager,
	logger *zap.Logger,
) (*ChunkerRegistry, error) {
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

	mediaKeepAlive := config.MediaKeepAlive
	if mediaKeepAlive == 0 {
		mediaKeepAlive = keepAlive
	}

	registry := &ChunkerRegistry{
		modelsDir:       config.ModelsDir,
		sessionManager:  sessionManager,
		logger:          logger,
		discovered:      make(map[string]*ChunkerModelInfo),
		refCounts:       make(map[string]int),
		keepAlive:       keepAlive,
		mediaKeepAlive:  mediaKeepAlive,
		maxLoadedModels: config.MaxLoadedModels,
		poolSize:        poolSize,
	}

	// Configure text chunker TTL cache with LRU eviction
	cacheOpts := []ttlcache.Option[string, chunking.Chunker]{
		ttlcache.WithTTL[string, chunking.Chunker](keepAlive),
	}

	if config.MaxLoadedModels > 0 {
		cacheOpts = append(cacheOpts,
			ttlcache.WithCapacity[string, chunking.Chunker](config.MaxLoadedModels))
	}

	registry.cache = ttlcache.New(cacheOpts...)

	// Configure media chunker TTL cache with capacity (mirrors text cache pattern)
	mediaCacheOpts := []ttlcache.Option[string, termchunking.CloseableMediaChunker]{
		ttlcache.WithTTL[string, termchunking.CloseableMediaChunker](mediaKeepAlive),
	}
	if config.MaxLoadedModels > 0 {
		mediaCacheOpts = append(mediaCacheOpts,
			ttlcache.WithCapacity[string, termchunking.CloseableMediaChunker](config.MaxLoadedModels))
	}
	registry.mediaCache = ttlcache.New(mediaCacheOpts...)

	// Set up eviction callback to close unloaded models
	// Note: Only close on TTL expiration or capacity eviction, not on manual deletion
	// (manual deletion during Close() handles cleanup synchronously)
	registry.cache.OnEviction(func(ctx context.Context, reason ttlcache.EvictionReason, item *ttlcache.Item[string, chunking.Chunker]) {
		// Skip closing on manual deletion - Close() handles cleanup synchronously
		if reason == ttlcache.EvictionReasonDeleted {
			logger.Debug("Chunker model removed from cache (cleanup handled separately)",
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
			logger.Warn("Preventing eviction of chunker model with active references",
				zap.String("model", item.Key()),
				zap.Int("refCount", refCount),
				zap.String("reason", reasonStr))
			return
		}
		registry.refCountsMu.Unlock()

		logger.Info("Evicting chunker model from cache",
			zap.String("model", item.Key()),
			zap.String("reason", reasonStr))
		if err := item.Value().Close(); err != nil {
			logger.Warn("Error closing evicted chunker model",
				zap.String("model", item.Key()),
				zap.Error(err))
		}
	})

	// Set up media cache eviction callback
	registry.mediaCache.OnEviction(func(ctx context.Context, reason ttlcache.EvictionReason, item *ttlcache.Item[string, termchunking.CloseableMediaChunker]) {
		if reason == ttlcache.EvictionReasonDeleted {
			return
		}

		registry.refCountsMu.Lock()
		refCount := registry.refCounts[item.Key()]
		if refCount > 0 {
			registry.mediaCache.Set(item.Key(), item.Value(), registry.mediaKeepAlive)
			registry.refCountsMu.Unlock()
			logger.Warn("Preventing eviction of media chunker model with active references",
				zap.String("model", item.Key()),
				zap.Int("refCount", refCount))
			return
		}
		registry.refCountsMu.Unlock()

		logger.Info("Evicting media chunker model from cache",
			zap.String("model", item.Key()))
		if err := item.Value().Close(); err != nil {
			logger.Warn("Error closing evicted media chunker model",
				zap.String("model", item.Key()),
				zap.Error(err))
		}
	})

	// Start cache cleanup goroutines
	go registry.cache.Start()
	go registry.mediaCache.Start()

	// Discover models (but don't load them)
	if err := registry.discoverModels(); err != nil {
		registry.cache.Stop()
		registry.mediaCache.Stop()
		return nil, err
	}

	logger.Info("Lazy chunker registry initialized",
		zap.Int("models_discovered", len(registry.discovered)),
		zap.Duration("keep_alive", keepAlive),
		zap.Uint64("max_loaded_models", config.MaxLoadedModels))

	return registry, nil
}

// discoverModels finds all chunker models in the models directory without loading them
func (r *ChunkerRegistry) discoverModels() error {
	if r.modelsDir == "" {
		r.logger.Info("No chunker models directory configured")
		return nil
	}

	// Check if directory exists
	if _, err := os.Stat(r.modelsDir); os.IsNotExist(err) {
		r.logger.Warn("Chunker models directory does not exist",
			zap.String("dir", r.modelsDir))
		return nil
	}

	// Use discoverModelsInDir which handles owner/model structure
	discovered, err := discoverModelsInDir(r.modelsDir, modelregistry.ModelTypeChunker, r.logger)
	if err != nil {
		return fmt.Errorf("discovering chunker models: %w", err)
	}

	// Pool size for concurrent pipeline access
	poolSize := r.poolSize

	// Collect log entries to emit outside the lock
	type discoveryLog struct {
		name         string
		path         string
		capabilities []string // non-nil for media models
		variants     []string // non-nil for text models
	}
	var logEntries []discoveryLog

	r.mu.Lock()
	for _, dm := range discovered {
		modelPath := dm.Path
		registryFullName := dm.FullName()
		variants := dm.Variants

		// Check if this model has media capabilities (e.g., audio)
		hasMediaCap := false
		if dm.Manifest != nil {
			for _, cap := range mediaCapabilities {
				if dm.Manifest.HasCapability(cap) {
					hasMediaCap = true
					break
				}
			}
		}

		// Media-capable models: register with capabilities and session options
		if hasMediaCap && dm.Manifest != nil {
			if _, exists := r.discovered[registryFullName]; exists {
				continue
			}

			// Build session options from manifest
			var sessionOpts []backends.SessionOption
			if dm.Manifest.SessionOptions != nil {
				if len(dm.Manifest.SessionOptions.InputConstants) > 0 {
					sessionOpts = append(sessionOpts, backends.WithInputConstants(dm.Manifest.SessionOptions.InputConstants))
				}
				if len(dm.Manifest.SessionOptions.DynamicAxes) > 0 {
					overrides := make([]backends.DynamicAxisOverride, len(dm.Manifest.SessionOptions.DynamicAxes))
					for i, da := range dm.Manifest.SessionOptions.DynamicAxes {
						overrides[i] = backends.DynamicAxisOverride{
							InputName: da.InputName,
							Axis:      da.Axis,
							ParamName: da.ParamName,
						}
					}
					sessionOpts = append(sessionOpts, backends.WithDynamicAxes(overrides))
				}
			}

			r.discovered[registryFullName] = &ChunkerModelInfo{
				Name:         registryFullName,
				Path:         modelPath,
				Capabilities: dm.Manifest.Capabilities,
				Backends:     dm.Manifest.Backends,
				SessionOpts:  sessionOpts,
			}

			logEntries = append(logEntries, discoveryLog{
				name:         registryFullName,
				path:         modelPath,
				capabilities: dm.Manifest.Capabilities,
			})
			continue
		}

		// Text chunker models: register each variant
		if len(variants) == 0 {
			continue
		}

		anyNew := false
		for variantID, onnxFilename := range variants {
			registryName := registryFullName
			if variantID != "" {
				registryName = registryFullName + "-" + variantID
			}

			if _, exists := r.discovered[registryName]; exists {
				continue
			}

			r.discovered[registryName] = &ChunkerModelInfo{
				Name:         registryName,
				Path:         modelPath,
				OnnxFilename: onnxFilename,
				PoolSize:     poolSize,
			}
			anyNew = true
		}

		if anyNew {
			variantIDs := make([]string, 0, len(variants))
			for v := range variants {
				if v == "" {
					variantIDs = append(variantIDs, "default")
				} else {
					variantIDs = append(variantIDs, v)
				}
			}
			logEntries = append(logEntries, discoveryLog{
				name:     registryFullName,
				path:     modelPath,
				variants: variantIDs,
			})
		}
	}
	discoveredCount := len(r.discovered)
	r.mu.Unlock()

	// Log discovered models outside the lock to avoid I/O under mutex
	for _, entry := range logEntries {
		if entry.capabilities != nil {
			r.logger.Info("Discovered media chunker model (not loaded)",
				zap.String("name", entry.name),
				zap.String("path", entry.path),
				zap.Strings("capabilities", entry.capabilities))
		} else {
			r.logger.Info("Discovered text chunker model (not loaded)",
				zap.String("name", entry.name),
				zap.String("path", entry.path),
				zap.Strings("variants", entry.variants))
		}
	}

	r.logger.Info("Chunker model discovery complete",
		zap.Int("models_discovered", discoveredCount),
		zap.Duration("keep_alive", r.keepAlive),
		zap.Uint64("max_loaded_models", r.maxLoadedModels))

	return nil
}

// Get returns a chunker by name, loading it if necessary.
// DEPRECATED: Use Acquire() instead for long-running operations to prevent
// the model from being evicted during use. Get() does not track usage and
// the returned chunker may be closed if the cache evicts it.
func (r *ChunkerRegistry) Get(modelName string) (chunking.Chunker, error) {
	// Check cache first
	if item := r.cache.Get(modelName); item != nil {
		r.logger.Debug("Chunker cache hit", zap.String("model", modelName))
		return item.Value(), nil
	}

	// Check if model is discovered
	r.mu.RLock()
	info, ok := r.discovered[modelName]
	r.mu.RUnlock()

	if !ok {
		// Model not yet discovered — rescan disk for newly pulled models
		if err := r.discoverModels(); err != nil {
			r.logger.Debug("Chunker re-discovery failed", zap.Error(err))
		}
		r.mu.RLock()
		info, ok = r.discovered[modelName]
		r.mu.RUnlock()
		if !ok {
			return nil, fmt.Errorf("chunker model not found: %s", modelName)
		}
	}

	// Load the model
	return r.loadModel(info)
}

// Acquire returns a chunker by name and increments its reference count.
// The caller MUST call Release() when done to allow the model to be evicted.
// This prevents the model from being closed while in use.
func (r *ChunkerRegistry) Acquire(modelName string) (chunking.Chunker, error) {
	// Check if model is discovered
	r.mu.RLock()
	info, ok := r.discovered[modelName]
	r.mu.RUnlock()

	if !ok {
		if err := r.discoverModels(); err != nil {
			r.logger.Debug("Chunker re-discovery failed", zap.Error(err))
		}
		r.mu.RLock()
		info, ok = r.discovered[modelName]
		r.mu.RUnlock()
		if !ok {
			return nil, fmt.Errorf("chunker model not found: %s", modelName)
		}
	}

	// loadModelWithRef handles cache lookup, loading, and refcount increment
	// atomically under r.mu.Lock to prevent TOCTOU between cache hit and eviction.
	return r.loadModelWithRef(info)
}

// Release decrements the reference count for a model.
// Must be called after Acquire() when the caller is done using the chunker.
func (r *ChunkerRegistry) Release(modelName string) {
	r.refCountsMu.Lock()
	if r.refCounts[modelName] > 0 {
		r.refCounts[modelName]--
	}
	count := r.refCounts[modelName]
	r.refCountsMu.Unlock()

	r.logger.Debug("Released chunker model",
		zap.String("model", modelName),
		zap.Int("refCount", count))
}

// AcquireMedia returns a media chunker by name and increments its reference count.
// Only valid for models with media capabilities (e.g., "audio").
// The caller MUST call Release() when done to allow the model to be evicted.
func (r *ChunkerRegistry) AcquireMedia(modelName string) (termchunking.MediaChunker, error) {
	// Check if model is discovered and has media capabilities
	r.mu.RLock()
	info, ok := r.discovered[modelName]
	r.mu.RUnlock()

	if !ok {
		if err := r.discoverModels(); err != nil {
			r.logger.Debug("Chunker re-discovery failed", zap.Error(err))
		}
		r.mu.RLock()
		info, ok = r.discovered[modelName]
		r.mu.RUnlock()
		if !ok {
			return nil, fmt.Errorf("chunker model not found: %s", modelName)
		}
	}

	if !r.isMediaModel(info) {
		return nil, fmt.Errorf("model %s does not have media capabilities", modelName)
	}

	// loadMediaModel handles cache lookup, loading, and refcount increment atomically
	// under r.mu.Lock to prevent TOCTOU between cache hit and eviction.
	return r.loadMediaModel(info)
}

// isMediaModel checks if a model has any media capability.
func (r *ChunkerRegistry) isMediaModel(info *ChunkerModelInfo) bool {
	for _, cap := range mediaCapabilities {
		if slices.Contains(info.Capabilities, string(cap)) {
			return true
		}
	}
	return false
}

// loadMediaModel loads a media chunker model from disk using the lib/chunking factory.
// It atomically increments the refcount under the lock to prevent TOCTOU races
// between cache lookup and TTL eviction.
func (r *ChunkerRegistry) loadMediaModel(info *ChunkerModelInfo) (termchunking.CloseableMediaChunker, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Double-check cache after acquiring lock.
	// Increment refcount under the same lock to prevent eviction between
	// the cache hit and the caller using the chunker.
	if item := r.mediaCache.Get(info.Name); item != nil {
		r.refCountsMu.Lock()
		r.refCounts[info.Name]++
		r.refCountsMu.Unlock()
		return item.Value(), nil
	}

	r.logger.Info("Loading media chunker model on demand",
		zap.String("model", info.Name),
		zap.String("path", info.Path))

	cfg := termchunking.MediaChunkerConfig{
		ModelPath:     info.Path,
		Capabilities:  info.Capabilities,
		ModelBackends: info.Backends,
		SessionOpts:   info.SessionOpts,
		Logger:        r.logger.Named(info.Name),
	}

	chunker, backendUsed, err := termchunking.NewMediaChunkerFromModel(cfg, r.sessionManager)
	if err != nil {
		return nil, fmt.Errorf("loading media chunker model %s: %w", info.Name, err)
	}

	r.logger.Info("Successfully loaded media chunker model",
		zap.String("name", info.Name),
		zap.String("backend", string(backendUsed)))

	r.mediaCache.Set(info.Name, chunker, r.mediaKeepAlive)

	// Increment refcount for the newly loaded model
	r.refCountsMu.Lock()
	r.refCounts[info.Name]++
	r.refCountsMu.Unlock()

	return chunker, nil
}

// loadModel loads a text chunker model from disk.
// Uses a double-checked lock to prevent concurrent duplicate loads.
func (r *ChunkerRegistry) loadModel(info *ChunkerModelInfo) (chunking.Chunker, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.loadModelLocked(info)
}

// loadModelWithRef loads a text chunker model and atomically increments its refcount.
// This prevents TOCTOU between cache hit and TTL eviction.
func (r *ChunkerRegistry) loadModelWithRef(info *ChunkerModelInfo) (chunking.Chunker, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	chunker, err := r.loadModelLocked(info)
	if err != nil {
		return nil, err
	}

	r.refCountsMu.Lock()
	r.refCounts[info.Name]++
	r.refCountsMu.Unlock()

	return chunker, nil
}

// loadModelLocked loads a text chunker model from disk. Caller must hold r.mu.
func (r *ChunkerRegistry) loadModelLocked(info *ChunkerModelInfo) (chunking.Chunker, error) {
	// Double-check cache after acquiring lock to prevent concurrent duplicate loads
	if item := r.cache.Get(info.Name); item != nil {
		return item.Value(), nil
	}

	r.logger.Info("Loading chunker model on demand",
		zap.String("model", info.Name),
		zap.String("path", info.Path))

	// Load using pipeline-based chunker
	cfg := termchunking.PooledChunkerConfig{
		ModelPath:     info.Path,
		PoolSize:      info.PoolSize,
		ChunkerConfig: termchunking.DefaultChunkerConfig(),
		ModelBackends: nil, // Use all available backends
		Logger:        r.logger.Named(info.Name),
	}
	chunker, backendUsed, err := termchunking.NewPooledChunker(cfg, r.sessionManager)
	if err != nil {
		return nil, fmt.Errorf("loading chunker model %s: %w", info.Name, err)
	}

	r.logger.Info("Successfully loaded chunker model",
		zap.String("name", info.Name),
		zap.String("backend", string(backendUsed)),
		zap.Int("poolSize", info.PoolSize))

	// Add to cache
	r.cache.Set(info.Name, chunker, r.keepAlive)

	return chunker, nil
}

// List returns all available chunker model names (discovered, not necessarily loaded).
// Re-scans the models directory to pick up newly pulled models.
func (r *ChunkerRegistry) List() []string {
	_ = r.discoverModels()

	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.discovered))
	for name := range r.discovered {
		names = append(names, name)
	}
	return names
}

// ListWithCapabilities returns a map of model names to their capabilities.
// Re-scans the models directory to pick up newly pulled models.
func (r *ChunkerRegistry) ListWithCapabilities() map[string][]string {
	_ = r.discoverModels()

	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string][]string, len(r.discovered))
	for name, info := range r.discovered {
		result[name] = info.Capabilities
	}
	return result
}

// HasCapability checks if a model has a specific capability (e.g., audio).
func (r *ChunkerRegistry) HasCapability(modelName string, capability modelregistry.Capability) bool {
	r.mu.RLock()
	info, known := r.discovered[modelName]
	r.mu.RUnlock()

	if !known {
		if err := r.discoverModels(); err != nil {
			r.logger.Debug("Chunker re-discovery failed", zap.Error(err))
		}
		r.mu.RLock()
		info, known = r.discovered[modelName]
		r.mu.RUnlock()
		if !known {
			return false
		}
	}

	return slices.Contains(info.Capabilities, string(capability))
}

// ListLoaded returns only the currently loaded chunker model names
func (r *ChunkerRegistry) ListLoaded() []string {
	keys := r.cache.Keys()
	return keys
}

// IsLoaded returns whether a model is currently loaded in memory
func (r *ChunkerRegistry) IsLoaded(modelName string) bool {
	return r.cache.Has(modelName)
}

// Preload loads specified models at startup to avoid first-request latency
func (r *ChunkerRegistry) Preload(modelNames []string) error {
	if len(modelNames) == 0 {
		return nil
	}

	r.logger.Info("Preloading chunker models", zap.Strings("models", modelNames))

	var loaded, failed int
	for _, name := range modelNames {
		if _, err := r.Acquire(name); err != nil {
			r.logger.Warn("Failed to preload chunker model",
				zap.String("model", name),
				zap.Error(err))
			failed++
		} else {
			r.Release(name)
			r.logger.Info("Preloaded chunker model",
				zap.String("model", name))
			loaded++
		}
	}

	r.logger.Info("Chunker preloading complete",
		zap.Int("loaded", loaded),
		zap.Int("failed", failed))

	if failed > 0 && loaded == 0 {
		return fmt.Errorf("all %d chunker models failed to preload", failed)
	}

	return nil
}

// PreloadAll loads all discovered models (for eager loading mode)
func (r *ChunkerRegistry) PreloadAll() error {
	return r.Preload(r.List())
}

// Close stops the caches and unloads all models
func (r *ChunkerRegistry) Close() error {
	r.logger.Info("Closing chunker registry")

	// Stop caches first to prevent new evictions
	r.cache.Stop()
	r.mediaCache.Stop()

	// Close all cached text models synchronously
	for _, key := range r.cache.Keys() {
		if item := r.cache.Get(key); item != nil {
			r.logger.Debug("Closing cached text chunker model",
				zap.String("model", key))
			if err := item.Value().Close(); err != nil {
				r.logger.Warn("Error closing text chunker model",
					zap.String("model", key),
					zap.Error(err))
			}
		}
	}

	// Close all cached media models synchronously
	for _, key := range r.mediaCache.Keys() {
		if item := r.mediaCache.Get(key); item != nil {
			r.logger.Debug("Closing cached media chunker model",
				zap.String("model", key))
			if err := item.Value().Close(); err != nil {
				r.logger.Warn("Error closing media chunker model",
					zap.String("model", key),
					zap.Error(err))
			}
		}
	}

	// Clear both caches
	r.cache.DeleteAll()
	r.mediaCache.DeleteAll()

	return nil
}
