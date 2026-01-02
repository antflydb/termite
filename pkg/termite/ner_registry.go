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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/antflydb/termite/pkg/termite/lib/gliner"
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
	"github.com/antflydb/termite/pkg/termite/lib/ner"
	"github.com/antflydb/termite/pkg/termite/lib/rebel"
	"github.com/jellydator/ttlcache/v3"
	"go.uber.org/zap"
)

// NERModelType indicates the type of NER model
type NERModelType int

const (
	NERModelTypeStandard NERModelType = iota
	NERModelTypeGLiNER
	NERModelTypeREBEL
)

// NERModelInfo holds metadata about a discovered NER model (not loaded yet)
type NERModelInfo struct {
	Name         string
	Path         string
	OnnxFilename string
	PoolSize     int
	ModelType    NERModelType
	Quantized    bool
	Capabilities []string
}

// loadedNERModel wraps both Model and optional Recognizer interfaces
type loadedNERModel struct {
	model       ner.Model
	recognizer  ner.Recognizer // May be nil for standard NER models
	modelType   NERModelType
	capabilities []string
}

// NERRegistry manages NER models with lazy loading and TTL-based unloading
type NERRegistry struct {
	modelsDir      string
	sessionManager *hugot.SessionManager
	logger         *zap.Logger

	// Model discovery (paths only, not loaded)
	discovered map[string]*NERModelInfo
	mu         sync.RWMutex

	// Loaded models with TTL cache
	cache *ttlcache.Cache[string, *loadedNERModel]

	// Configuration
	keepAlive       time.Duration
	maxLoadedModels uint64
	poolSize        int
}

// NERConfig configures the NER registry
type NERConfig struct {
	ModelsDir       string
	KeepAlive       time.Duration // How long to keep models loaded (0 = forever)
	MaxLoadedModels uint64        // Max models in memory (0 = unlimited)
	PoolSize        int           // Number of concurrent pipelines per model (0 = default)
}

// NewNERRegistry creates a new lazy-loading NER registry
func NewNERRegistry(
	config NERConfig,
	sessionManager *hugot.SessionManager,
	logger *zap.Logger,
) (*NERRegistry, error) {
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

	registry := &NERRegistry{
		modelsDir:       config.ModelsDir,
		sessionManager:  sessionManager,
		logger:          logger,
		discovered:      make(map[string]*NERModelInfo),
		keepAlive:       keepAlive,
		maxLoadedModels: config.MaxLoadedModels,
		poolSize:        poolSize,
	}

	// Configure TTL cache with LRU eviction
	cacheOpts := []ttlcache.Option[string, *loadedNERModel]{
		ttlcache.WithTTL[string, *loadedNERModel](keepAlive),
	}

	if config.MaxLoadedModels > 0 {
		cacheOpts = append(cacheOpts,
			ttlcache.WithCapacity[string, *loadedNERModel](config.MaxLoadedModels))
	}

	registry.cache = ttlcache.New(cacheOpts...)

	// Set up eviction callback to close unloaded models
	// Note: Only close on TTL expiration or capacity eviction, not on manual deletion
	// (manual deletion during Close() handles cleanup synchronously)
	registry.cache.OnEviction(func(ctx context.Context, reason ttlcache.EvictionReason, item *ttlcache.Item[string, *loadedNERModel]) {
		// Skip closing on manual deletion - Close() handles cleanup synchronously
		if reason == ttlcache.EvictionReasonDeleted {
			logger.Debug("NER model removed from cache (cleanup handled separately)",
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
		logger.Info("Evicting NER model from cache",
			zap.String("model", item.Key()),
			zap.String("reason", reasonStr))
		if err := item.Value().model.Close(); err != nil {
			logger.Warn("Error closing evicted NER model",
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

	logger.Info("Lazy NER registry initialized",
		zap.Int("models_discovered", len(registry.discovered)),
		zap.Duration("keep_alive", keepAlive),
		zap.Uint64("max_loaded_models", config.MaxLoadedModels))

	return registry, nil
}

// discoverModels finds all NER models in the models directory without loading them
func (r *NERRegistry) discoverModels() error {
	if r.modelsDir == "" {
		r.logger.Info("No NER models directory configured")
		return nil
	}

	// Check if directory exists
	if _, err := os.Stat(r.modelsDir); os.IsNotExist(err) {
		r.logger.Warn("NER models directory does not exist",
			zap.String("dir", r.modelsDir))
		return nil
	}

	// Use discoverModelsInDir which handles owner/model structure
	discovered, err := discoverModelsInDir(r.modelsDir, modelregistry.ModelTypeRecognizer, r.logger)
	if err != nil {
		return fmt.Errorf("discovering NER models: %w", err)
	}

	// Pool size for concurrent pipeline access
	poolSize := r.poolSize

	for _, dm := range discovered {
		modelPath := dm.Path
		registryFullName := dm.FullName()

		// Check model type: GLiNER, REBEL, or traditional NER
		isGLiNER := gliner.IsGLiNERModel(modelPath)
		isREBEL := rebel.IsREBELModel(modelPath)

		if isREBEL {
			r.logger.Info("Discovered REBEL model (not loaded)",
				zap.String("name", registryFullName),
				zap.String("path", modelPath))

			// REBEL models have 'relations' capability
			caps := []string{modelregistry.CapabilityRelations}
			// Check manifest for additional capabilities
			manifestPath := filepath.Join(modelPath, "manifest.json")
			if data, err := os.ReadFile(manifestPath); err == nil {
				var manifest modelregistry.ModelManifest
				if err := json.Unmarshal(data, &manifest); err == nil && len(manifest.Capabilities) > 0 {
					caps = manifest.Capabilities
				}
			}

			r.discovered[registryFullName] = &NERModelInfo{
				Name:         registryFullName,
				Path:         modelPath,
				PoolSize:     poolSize,
				ModelType:    NERModelTypeREBEL,
				Capabilities: caps,
			}
		} else if isGLiNER {
			r.logger.Info("Discovered GLiNER model (not loaded)",
				zap.String("name", registryFullName),
				zap.String("path", modelPath))

			// Try quantized first, then non-quantized
			quantized := false
			if _, err := os.Stat(filepath.Join(modelPath, "model_quantized.onnx")); err == nil {
				quantized = true
			} else if _, err := os.Stat(filepath.Join(modelPath, "model.onnx")); err != nil {
				r.logger.Debug("Skipping GLiNER directory without model files",
					zap.String("dir", registryFullName))
				continue
			}

			// Load capabilities from manifest if available
			caps := []string{modelregistry.CapabilityLabels, modelregistry.CapabilityZeroshot}
			manifestPath := filepath.Join(modelPath, "manifest.json")
			if data, err := os.ReadFile(manifestPath); err == nil {
				var manifest modelregistry.ModelManifest
				if err := json.Unmarshal(data, &manifest); err == nil && len(manifest.Capabilities) > 0 {
					caps = manifest.Capabilities
				}
			}

			r.discovered[registryFullName] = &NERModelInfo{
				Name:         registryFullName,
				Path:         modelPath,
				PoolSize:     poolSize,
				ModelType:    NERModelTypeGLiNER,
				Quantized:    quantized,
				Capabilities: caps,
			}
		} else {
			// Discover all available model variants for regular NER models
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
			r.logger.Info("Discovered NER model (not loaded)",
				zap.String("name", registryFullName),
				zap.String("path", modelPath),
				zap.Strings("variants", variantIDs))

			// Load capabilities from manifest if available
			caps := []string{modelregistry.CapabilityLabels}
			manifestPath := filepath.Join(modelPath, "manifest.json")
			if data, err := os.ReadFile(manifestPath); err == nil {
				var manifest modelregistry.ModelManifest
				if err := json.Unmarshal(data, &manifest); err == nil && len(manifest.Capabilities) > 0 {
					caps = manifest.Capabilities
				}
			}

			// Store each variant for lazy loading
			for variantID, onnxFilename := range variants {
				// Determine registry name
				registryName := registryFullName
				if variantID != "" {
					registryName = registryFullName + "-" + variantID
				}

				r.discovered[registryName] = &NERModelInfo{
					Name:         registryName,
					Path:         modelPath,
					OnnxFilename: onnxFilename,
					PoolSize:     poolSize,
					ModelType:    NERModelTypeStandard,
					Capabilities: caps,
				}
			}
		}
	}

	r.logger.Info("NER model discovery complete",
		zap.Int("models_discovered", len(r.discovered)),
		zap.Duration("keep_alive", r.keepAlive),
		zap.Uint64("max_loaded_models", r.maxLoadedModels))

	return nil
}

// Get returns a NER model by name, loading it if necessary
func (r *NERRegistry) Get(modelName string) (ner.Model, error) {
	loaded, err := r.getLoaded(modelName)
	if err != nil {
		return nil, err
	}
	return loaded.model, nil
}

// GetRecognizer returns a zero-shot capable Recognizer by name (e.g., GLiNER)
func (r *NERRegistry) GetRecognizer(modelName string) (ner.Recognizer, error) {
	loaded, err := r.getLoaded(modelName)
	if err != nil {
		return nil, err
	}
	if loaded.recognizer == nil {
		return nil, fmt.Errorf("model %s is not a Recognizer", modelName)
	}
	return loaded.recognizer, nil
}

// getLoaded gets or loads a model from cache
func (r *NERRegistry) getLoaded(modelName string) (*loadedNERModel, error) {
	// Check cache first
	if item := r.cache.Get(modelName); item != nil {
		r.logger.Debug("NER cache hit", zap.String("model", modelName))
		return item.Value(), nil
	}

	// Check if model is discovered
	r.mu.RLock()
	info, ok := r.discovered[modelName]
	r.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("NER model not found: %s", modelName)
	}

	// Load the model
	return r.loadModel(info)
}

// loadModel loads a NER model from disk
func (r *NERRegistry) loadModel(info *NERModelInfo) (*loadedNERModel, error) {
	r.logger.Info("Loading NER model on demand",
		zap.String("model", info.Name),
		zap.String("path", info.Path),
		zap.Int("modelType", int(info.ModelType)))

	var loaded *loadedNERModel

	switch info.ModelType {
	case NERModelTypeREBEL:
		model, backendUsed, err := rebel.NewHugotREBELWithSessionManager(
			info.Path, r.sessionManager, r.logger.Named(info.Name))
		if err != nil {
			return nil, fmt.Errorf("loading REBEL model %s: %w", info.Name, err)
		}
		r.logger.Info("Successfully loaded REBEL model",
			zap.String("name", info.Name),
			zap.String("backend", string(backendUsed)),
			zap.Strings("capabilities", info.Capabilities))
		loaded = &loadedNERModel{
			model:        model,
			recognizer:   model,
			modelType:    NERModelTypeREBEL,
			capabilities: info.Capabilities,
		}

	case NERModelTypeGLiNER:
		model, backendUsed, err := gliner.NewHugotGLiNERWithSessionManager(
			info.Path, info.Quantized, r.sessionManager, nil, r.logger.Named(info.Name))
		if err != nil {
			return nil, fmt.Errorf("loading GLiNER model %s: %w", info.Name, err)
		}
		// Update capabilities based on loaded model
		caps := info.Capabilities
		if model.SupportsRelationExtraction() && !containsCapability(caps, modelregistry.CapabilityRelations) {
			caps = append(caps, modelregistry.CapabilityRelations)
		}
		if model.SupportsQA() && !containsCapability(caps, modelregistry.CapabilityAnswers) {
			caps = append(caps, modelregistry.CapabilityAnswers)
		}
		r.logger.Info("Successfully loaded GLiNER model",
			zap.String("name", info.Name),
			zap.Bool("quantized", info.Quantized),
			zap.String("backend", string(backendUsed)),
			zap.Strings("default_labels", model.Labels()),
			zap.Strings("capabilities", caps))
		loaded = &loadedNERModel{
			model:        model,
			recognizer:   model,
			modelType:    NERModelTypeGLiNER,
			capabilities: caps,
		}

	default: // NERModelTypeStandard
		model, backendUsed, err := ner.NewPooledHugotNERWithSessionManager(
			info.Path, info.OnnxFilename, info.PoolSize, r.sessionManager, nil, r.logger.Named(info.Name))
		if err != nil {
			return nil, fmt.Errorf("loading NER model %s: %w", info.Name, err)
		}
		r.logger.Info("Successfully loaded NER model",
			zap.String("name", info.Name),
			zap.String("onnxFile", info.OnnxFilename),
			zap.String("backend", string(backendUsed)),
			zap.Int("poolSize", info.PoolSize),
			zap.Strings("capabilities", info.Capabilities))
		loaded = &loadedNERModel{
			model:        model,
			recognizer:   nil, // Standard NER models don't implement Recognizer
			modelType:    NERModelTypeStandard,
			capabilities: info.Capabilities,
		}
	}

	// Add to cache
	r.cache.Set(info.Name, loaded, r.keepAlive)

	return loaded, nil
}

// containsCapability checks if a capability is in the list
func containsCapability(caps []string, cap string) bool {
	for _, c := range caps {
		if c == cap {
			return true
		}
	}
	return false
}

// IsRecognizer returns true if the model is a zero-shot capable Recognizer
func (r *NERRegistry) IsRecognizer(modelName string) bool {
	r.mu.RLock()
	info, ok := r.discovered[modelName]
	r.mu.RUnlock()

	if !ok {
		return false
	}
	return info.ModelType == NERModelTypeGLiNER || info.ModelType == NERModelTypeREBEL
}

// List returns all available NER model names (discovered, not necessarily loaded)
func (r *NERRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.discovered))
	for name := range r.discovered {
		names = append(names, name)
	}
	return names
}

// ListRecognizers returns all available zero-shot Recognizer model names
func (r *NERRegistry) ListRecognizers() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0)
	for name, info := range r.discovered {
		if info.ModelType == NERModelTypeGLiNER || info.ModelType == NERModelTypeREBEL {
			names = append(names, name)
		}
	}
	return names
}

// ListLoaded returns only the currently loaded NER model names
func (r *NERRegistry) ListLoaded() []string {
	return r.cache.Keys()
}

// IsLoaded returns whether a model is currently loaded in memory
func (r *NERRegistry) IsLoaded(modelName string) bool {
	return r.cache.Has(modelName)
}

// GetCapabilities returns the capabilities for a specific model.
// Returns nil if the model is not found.
func (r *NERRegistry) GetCapabilities(modelName string) []string {
	r.mu.RLock()
	info, ok := r.discovered[modelName]
	r.mu.RUnlock()

	if !ok {
		return nil
	}
	return info.Capabilities
}

// HasCapability checks if a model has a specific capability.
func (r *NERRegistry) HasCapability(modelName, capability string) bool {
	caps := r.GetCapabilities(modelName)
	return containsCapability(caps, capability)
}

// ListWithCapabilities returns a map of model name to capabilities for all models.
func (r *NERRegistry) ListWithCapabilities() map[string][]string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string][]string, len(r.discovered))
	for name, info := range r.discovered {
		// Make a copy of the slice to avoid sharing internal state
		capsCopy := make([]string, len(info.Capabilities))
		copy(capsCopy, info.Capabilities)
		result[name] = capsCopy
	}
	return result
}

// Preload loads specified models at startup to avoid first-request latency
func (r *NERRegistry) Preload(modelNames []string) error {
	if len(modelNames) == 0 {
		return nil
	}

	r.logger.Info("Preloading NER models", zap.Strings("models", modelNames))

	var loaded, failed int
	for _, name := range modelNames {
		if _, err := r.Get(name); err != nil {
			r.logger.Warn("Failed to preload NER model",
				zap.String("model", name),
				zap.Error(err))
			failed++
		} else {
			r.logger.Info("Preloaded NER model",
				zap.String("model", name))
			loaded++
		}
	}

	r.logger.Info("NER preloading complete",
		zap.Int("loaded", loaded),
		zap.Int("failed", failed))

	if failed > 0 && loaded == 0 {
		return fmt.Errorf("all %d NER models failed to preload", failed)
	}

	return nil
}

// PreloadAll loads all discovered models (for eager loading mode)
func (r *NERRegistry) PreloadAll() error {
	return r.Preload(r.List())
}

// Close stops the cache and unloads all models
func (r *NERRegistry) Close() error {
	r.logger.Info("Closing lazy NER registry")

	// Stop cache first to prevent new evictions
	r.cache.Stop()

	// Close all cached models synchronously (don't rely on async eviction callbacks)
	for _, key := range r.cache.Keys() {
		if item := r.cache.Get(key); item != nil {
			loaded := item.Value()
			r.logger.Debug("Closing cached NER model",
				zap.String("model", key))
			if err := loaded.model.Close(); err != nil {
				r.logger.Warn("Error closing NER model",
					zap.String("model", key),
					zap.Error(err))
			}
		}
	}

	// Clear the cache (eviction callbacks won't close since reason is EvictionReasonDeleted)
	r.cache.DeleteAll()

	return nil
}
