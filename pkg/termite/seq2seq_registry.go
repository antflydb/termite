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
	"sync"
	"time"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
	"github.com/antflydb/termite/pkg/termite/lib/seq2seq"
	"github.com/jellydator/ttlcache/v3"
	"go.uber.org/zap"
)

// Seq2SeqModelInfo holds metadata about a discovered Seq2Seq model (not loaded yet)
type Seq2SeqModelInfo struct {
	Name string
	Path string
}

// Seq2SeqRegistry manages Seq2Seq models with lazy loading and TTL-based unloading
type Seq2SeqRegistry struct {
	modelsDir      string
	sessionManager *backends.SessionManager
	logger         *zap.Logger

	// Model discovery (paths only, not loaded)
	discovered map[string]*Seq2SeqModelInfo
	mu         sync.RWMutex

	// Loaded models with TTL cache
	cache *ttlcache.Cache[string, seq2seq.Model]

	// Reference counting to prevent eviction during active use
	refCounts   map[string]int
	refCountsMu sync.Mutex

	// Configuration
	keepAlive       time.Duration
	maxLoadedModels uint64
}

// Seq2SeqConfig configures the Seq2Seq registry
type Seq2SeqConfig struct {
	ModelsDir       string
	KeepAlive       time.Duration // How long to keep models loaded (0 = forever)
	MaxLoadedModels uint64        // Max models in memory (0 = unlimited)
}

// NewSeq2SeqRegistry creates a new lazy-loading Seq2Seq registry
func NewSeq2SeqRegistry(
	config Seq2SeqConfig,
	sessionManager *backends.SessionManager,
	logger *zap.Logger,
) (*Seq2SeqRegistry, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	keepAlive := config.KeepAlive
	if keepAlive == 0 {
		keepAlive = ttlcache.NoTTL // Never expire
	}

	registry := &Seq2SeqRegistry{
		modelsDir:       config.ModelsDir,
		sessionManager:  sessionManager,
		logger:          logger,
		discovered:      make(map[string]*Seq2SeqModelInfo),
		refCounts:       make(map[string]int),
		keepAlive:       keepAlive,
		maxLoadedModels: config.MaxLoadedModels,
	}

	// Configure TTL cache with LRU eviction
	cacheOpts := []ttlcache.Option[string, seq2seq.Model]{
		ttlcache.WithTTL[string, seq2seq.Model](keepAlive),
	}

	if config.MaxLoadedModels > 0 {
		cacheOpts = append(cacheOpts,
			ttlcache.WithCapacity[string, seq2seq.Model](config.MaxLoadedModels))
	}

	registry.cache = ttlcache.New(cacheOpts...)

	// Set up eviction callback to close unloaded models
	// Note: Only close on TTL expiration or capacity eviction, not on manual deletion
	// (manual deletion during Close() handles cleanup synchronously)
	registry.cache.OnEviction(func(ctx context.Context, reason ttlcache.EvictionReason, item *ttlcache.Item[string, seq2seq.Model]) {
		// Skip closing on manual deletion - Close() handles cleanup synchronously
		if reason == ttlcache.EvictionReasonDeleted {
			logger.Debug("Seq2Seq model removed from cache (cleanup handled separately)",
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
			logger.Warn("Preventing eviction of Seq2Seq model with active references",
				zap.String("model", item.Key()),
				zap.Int("refCount", refCount),
				zap.String("reason", reasonStr))
			return
		}
		registry.refCountsMu.Unlock()

		logger.Info("Evicting Seq2Seq model from cache",
			zap.String("model", item.Key()),
			zap.String("reason", reasonStr))
		if err := item.Value().Close(); err != nil {
			logger.Warn("Error closing evicted Seq2Seq model",
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

	logger.Info("Lazy Seq2Seq registry initialized",
		zap.Int("models_discovered", len(registry.discovered)),
		zap.Duration("keep_alive", keepAlive),
		zap.Uint64("max_loaded_models", config.MaxLoadedModels))

	return registry, nil
}

// discoverModels finds all Seq2Seq models in the models directory without loading them
func (r *Seq2SeqRegistry) discoverModels() error {
	if r.modelsDir == "" {
		r.logger.Info("No Seq2Seq models directory configured")
		return nil
	}

	// Check if directory exists
	if _, err := os.Stat(r.modelsDir); os.IsNotExist(err) {
		r.logger.Warn("Seq2Seq models directory does not exist",
			zap.String("dir", r.modelsDir))
		return nil
	}

	// Use discoverModelsInDir which handles owner/model structure
	discovered, err := discoverModelsInDir(r.modelsDir, modelregistry.ModelTypeRewriter, r.logger)
	if err != nil {
		return fmt.Errorf("discovering Seq2Seq models: %w", err)
	}

	for _, dm := range discovered {
		modelPath := dm.Path
		registryFullName := dm.FullName()

		// Check if this is a Seq2Seq model (has encoder.onnx, decoder-init.onnx, decoder.onnx)
		if !seq2seq.IsSeq2SeqModel(modelPath) {
			r.logger.Debug("Skipping directory - not a Seq2Seq model",
				zap.String("dir", registryFullName))
			continue
		}

		r.logger.Info("Discovered Seq2Seq model (not loaded)",
			zap.String("name", registryFullName),
			zap.String("path", modelPath))

		r.discovered[registryFullName] = &Seq2SeqModelInfo{
			Name: registryFullName,
			Path: modelPath,
		}
	}

	r.logger.Info("Seq2Seq model discovery complete",
		zap.Int("models_discovered", len(r.discovered)),
		zap.Duration("keep_alive", r.keepAlive),
		zap.Uint64("max_loaded_models", r.maxLoadedModels))

	return nil
}

// Get returns a Seq2Seq model by name, loading it if necessary.
// DEPRECATED: Use Acquire() instead for long-running operations to prevent
// the model from being evicted during use. Get() does not track usage and
// the returned model may be closed if the cache evicts it.
func (r *Seq2SeqRegistry) Get(modelName string) (seq2seq.Model, error) {
	// Check cache first
	if item := r.cache.Get(modelName); item != nil {
		r.logger.Debug("Seq2Seq cache hit", zap.String("model", modelName))
		return item.Value(), nil
	}

	// Check if model is discovered
	r.mu.RLock()
	info, ok := r.discovered[modelName]
	r.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("Seq2Seq model not found: %s", modelName)
	}

	// Load the model
	return r.loadModel(info)
}

// Acquire returns a Seq2Seq model by name and increments its reference count.
// The caller MUST call Release() when done to allow the model to be evicted.
// This prevents the model from being closed while in use.
func (r *Seq2SeqRegistry) Acquire(modelName string) (seq2seq.Model, error) {
	model, err := r.Get(modelName)
	if err != nil {
		return nil, err
	}

	r.refCountsMu.Lock()
	r.refCounts[modelName]++
	count := r.refCounts[modelName]
	r.refCountsMu.Unlock()

	r.logger.Debug("Acquired Seq2Seq model",
		zap.String("model", modelName),
		zap.Int("refCount", count))

	return model, nil
}

// Release decrements the reference count for a model.
// Must be called after Acquire() when the caller is done using the model.
func (r *Seq2SeqRegistry) Release(modelName string) {
	r.refCountsMu.Lock()
	if r.refCounts[modelName] > 0 {
		r.refCounts[modelName]--
	}
	count := r.refCounts[modelName]
	r.refCountsMu.Unlock()

	r.logger.Debug("Released Seq2Seq model",
		zap.String("model", modelName),
		zap.Int("refCount", count))
}

// GetQuestionGenerator returns a Seq2Seq model as a QuestionGenerator by name
func (r *Seq2SeqRegistry) GetQuestionGenerator(modelName string) (seq2seq.QuestionGenerator, error) {
	model, err := r.Get(modelName)
	if err != nil {
		return nil, err
	}

	qg, ok := model.(seq2seq.QuestionGenerator)
	if !ok {
		return nil, fmt.Errorf("model %s does not support question generation", modelName)
	}
	return qg, nil
}

// loadModel loads a Seq2Seq model from disk
func (r *Seq2SeqRegistry) loadModel(info *Seq2SeqModelInfo) (seq2seq.Model, error) {
	r.logger.Info("Loading Seq2Seq model on demand",
		zap.String("model", info.Name),
		zap.String("path", info.Path))

	// Load the Seq2Seq model using PooledSeq2Seq
	// Restrict to ONNX backend only - CoreML cannot handle dynamic batch sizes > 1
	// and large seq2seq models (like PEGASUS) exceed CoreML's model size limits.
	cfg := seq2seq.PooledSeq2SeqConfig{
		ModelPath:     info.Path,
		PoolSize:      1, // Registry manages pooling at a higher level
		ModelBackends: []string{"onnx"},
		Logger:        r.logger.Named(info.Name),
	}
	model, backendUsed, err := seq2seq.NewPooledSeq2Seq(cfg, r.sessionManager)
	if err != nil {
		return nil, fmt.Errorf("loading Seq2Seq model %s: %w", info.Name, err)
	}

	config := model.Config()
	r.logger.Info("Successfully loaded Seq2Seq model",
		zap.String("name", info.Name),
		zap.String("task", config.Task),
		zap.Int("max_length", config.MaxLength),
		zap.String("backend", string(backendUsed)))

	// Add to cache
	r.cache.Set(info.Name, model, r.keepAlive)

	return model, nil
}

// List returns all available Seq2Seq model names (discovered, not necessarily loaded)
func (r *Seq2SeqRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.discovered))
	for name := range r.discovered {
		names = append(names, name)
	}
	return names
}

// ListLoaded returns only the currently loaded Seq2Seq model names
func (r *Seq2SeqRegistry) ListLoaded() []string {
	return r.cache.Keys()
}

// IsLoaded returns whether a model is currently loaded in memory
func (r *Seq2SeqRegistry) IsLoaded(modelName string) bool {
	return r.cache.Has(modelName)
}

// Preload loads specified models at startup to avoid first-request latency
func (r *Seq2SeqRegistry) Preload(modelNames []string) error {
	if len(modelNames) == 0 {
		return nil
	}

	r.logger.Info("Preloading Seq2Seq models", zap.Strings("models", modelNames))

	var loaded, failed int
	for _, name := range modelNames {
		if _, err := r.Get(name); err != nil {
			r.logger.Warn("Failed to preload Seq2Seq model",
				zap.String("model", name),
				zap.Error(err))
			failed++
		} else {
			r.logger.Info("Preloaded Seq2Seq model",
				zap.String("model", name))
			loaded++
		}
	}

	r.logger.Info("Seq2Seq preloading complete",
		zap.Int("loaded", loaded),
		zap.Int("failed", failed))

	if failed > 0 && loaded == 0 {
		return fmt.Errorf("all %d Seq2Seq models failed to preload", failed)
	}

	return nil
}

// PreloadAll loads all discovered models (for eager loading mode)
func (r *Seq2SeqRegistry) PreloadAll() error {
	return r.Preload(r.List())
}

// Close stops the cache and unloads all models
func (r *Seq2SeqRegistry) Close() error {
	r.logger.Info("Closing lazy Seq2Seq registry")

	// Stop cache first to prevent new evictions
	r.cache.Stop()

	// Close all cached models synchronously (don't rely on async eviction callbacks)
	for _, key := range r.cache.Keys() {
		if item := r.cache.Get(key); item != nil {
			model := item.Value()
			r.logger.Debug("Closing cached Seq2Seq model",
				zap.String("model", key))
			if err := model.Close(); err != nil {
				r.logger.Warn("Error closing Seq2Seq model",
					zap.String("model", key),
					zap.Error(err))
			}
		}
	}

	// Clear the cache (eviction callbacks may still fire but models are already closed)
	r.cache.DeleteAll()

	return nil
}
