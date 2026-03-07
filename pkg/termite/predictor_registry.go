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

	"github.com/antflydb/termite/pkg/termite/lib/tabular"
	"github.com/jellydator/ttlcache/v3"
	"go.uber.org/zap"
)

// PredictorModelInfo holds metadata about a discovered predictor model (not loaded yet).
type PredictorModelInfo struct {
	Name string
	Path string
}

// PredictorConfig configures the predictor registry.
type PredictorConfig struct {
	ModelsDir       string
	KeepAlive       time.Duration
	MaxLoadedModels uint64
}

// PredictorRegistry manages tabular predictor models with lazy loading and TTL-based unloading.
type PredictorRegistry struct {
	modelsDir string
	logger    *zap.Logger

	discovered map[string]*PredictorModelInfo
	mu         sync.RWMutex

	cache *ttlcache.Cache[string, tabular.Predictor]
	refs  refTracker

	keepAlive       time.Duration
	maxLoadedModels uint64
}

// NewPredictorRegistry creates a new lazy-loading predictor registry.
func NewPredictorRegistry(config PredictorConfig, logger *zap.Logger) (*PredictorRegistry, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	keepAlive := config.KeepAlive
	if keepAlive == 0 {
		keepAlive = ttlcache.NoTTL
	}

	registry := &PredictorRegistry{
		modelsDir:       config.ModelsDir,
		logger:          logger,
		discovered:      make(map[string]*PredictorModelInfo),
		refs:            newRefTracker(),
		keepAlive:       keepAlive,
		maxLoadedModels: config.MaxLoadedModels,
	}

	cacheOpts := []ttlcache.Option[string, tabular.Predictor]{
		ttlcache.WithTTL[string, tabular.Predictor](keepAlive),
	}
	if config.MaxLoadedModels > 0 {
		cacheOpts = append(cacheOpts,
			ttlcache.WithCapacity[string, tabular.Predictor](config.MaxLoadedModels))
	}

	registry.cache = ttlcache.New(cacheOpts...)

	registry.cache.OnEviction(func(ctx context.Context, reason ttlcache.EvictionReason, item *ttlcache.Item[string, tabular.Predictor]) {
		modelName := item.Key()
		predictor := item.Value()

		if reason == ttlcache.EvictionReasonDeleted {
			return
		}

		reasonStr := evictionReasonString(reason)

		if registry.refs.deferCloseIfInUse(modelName, func() error {
			return predictor.Close()
		}) {
			logger.Warn("Predictor model evicted while in use, deferring close",
				zap.String("model", modelName),
				zap.String("reason", reasonStr))
			return
		}

		logger.Info("Unloading predictor model",
			zap.String("model", modelName),
			zap.String("reason", reasonStr))

		if err := predictor.Close(); err != nil {
			logger.Warn("Error closing predictor",
				zap.String("model", modelName),
				zap.Error(err))
		}
	})

	go registry.cache.Start()

	if err := registry.discoverModels(); err != nil {
		registry.cache.Stop()
		return nil, err
	}

	return registry, nil
}

// discoverModels scans the models directory for tabular predictor models.
// A predictor model directory must contain a tabular_model.json file.
func (r *PredictorRegistry) discoverModels() error {
	if r.modelsDir == "" {
		r.logger.Info("No predictor models directory configured")
		return nil
	}

	if _, err := os.Stat(r.modelsDir); os.IsNotExist(err) {
		r.logger.Warn("Predictor models directory does not exist",
			zap.String("dir", r.modelsDir))
		return nil
	}

	// Use the shared model discovery helper
	discovered, err := discoverModelsInDir(r.modelsDir, "predictor", r.logger)
	if err != nil {
		return fmt.Errorf("discovering predictor models: %w", err)
	}

	r.mu.Lock()
	for _, dm := range discovered {
		fullName := dm.FullName()
		if _, exists := r.discovered[fullName]; exists {
			continue
		}

		// Verify tabular_model.json exists
		if !fileExistsRegistry(dm.Path + "/" + tabular.TabularModelFilename) {
			r.logger.Debug("Skipping model without tabular_model.json",
				zap.String("name", fullName),
				zap.String("path", dm.Path))
			continue
		}

		r.discovered[fullName] = &PredictorModelInfo{
			Name: fullName,
			Path: dm.Path,
		}

		r.logger.Info("Discovered predictor model (not loaded)",
			zap.String("name", fullName),
			zap.String("path", dm.Path))
	}
	discoveredCount := len(r.discovered)
	r.mu.Unlock()

	r.logger.Info("Predictor model discovery complete",
		zap.Int("models_discovered", discoveredCount),
		zap.Duration("keep_alive", r.keepAlive))

	return nil
}

// Acquire returns a predictor by model name and increments its reference count.
func (r *PredictorRegistry) Acquire(modelName string) (tabular.Predictor, error) {
	r.refs.incRef(modelName)

	pred, err := r.get(modelName)
	if err != nil {
		r.refs.rollbackRef(modelName)
		return nil, err
	}

	r.logger.Debug("Acquired predictor model",
		zap.String("model", modelName))

	return pred, nil
}

// Release decrements the reference count for a model.
func (r *PredictorRegistry) Release(modelName string) {
	count, orphans := r.refs.releaseRef(modelName)

	r.logger.Debug("Released predictor model",
		zap.String("model", modelName),
		zap.Int("refCount", count))

	closeOrphans(r.logger, "predictor", modelName, orphans)
}

func (r *PredictorRegistry) get(modelName string) (tabular.Predictor, error) {
	if item := r.cache.Get(modelName); item != nil {
		return item.Value(), nil
	}

	r.mu.RLock()
	info, known := r.discovered[modelName]
	r.mu.RUnlock()

	if !known {
		if err := r.discoverModels(); err != nil {
			r.logger.Debug("Predictor re-discovery failed", zap.Error(err))
		}
		r.mu.RLock()
		info, known = r.discovered[modelName]
		r.mu.RUnlock()
		if !known {
			return nil, fmt.Errorf("predictor model not found: %s", modelName)
		}
	}

	return r.loadModel(info)
}

func (r *PredictorRegistry) loadModel(info *PredictorModelInfo) (tabular.Predictor, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if item := r.cache.Get(info.Name); item != nil {
		return item.Value(), nil
	}

	r.logger.Info("Loading predictor model on demand",
		zap.String("model", info.Name),
		zap.String("path", info.Path))

	predictor, err := tabular.LoadPredictor(info.Path)
	if err != nil {
		r.logger.Error("Failed to load predictor model",
			zap.String("model", info.Name),
			zap.Error(err))
		return nil, fmt.Errorf("loading predictor model %s: %w", info.Name, err)
	}

	r.cache.Set(info.Name, predictor, ttlcache.DefaultTTL)

	r.logger.Info("Successfully loaded predictor model",
		zap.String("model", info.Name),
		zap.String("task", string(predictor.ModelMetadata().Task)),
		zap.Int("num_features", predictor.NumFeatures()),
		zap.Duration("keep_alive", r.keepAlive))

	return predictor, nil
}

// List returns all available predictor model names.
func (r *PredictorRegistry) List() []string {
	_ = r.discoverModels()

	r.mu.RLock()
	names := make([]string, 0, len(r.discovered))
	for name := range r.discovered {
		names = append(names, name)
	}
	r.mu.RUnlock()

	return names
}

// PreloadAll loads all discovered models at startup.
func (r *PredictorRegistry) PreloadAll() error {
	names := r.List()
	if len(names) == 0 {
		return nil
	}

	r.logger.Info("Preloading predictor models", zap.Strings("models", names))

	var failed int
	for _, name := range names {
		if _, err := r.get(name); err != nil {
			r.logger.Warn("Failed to preload predictor model",
				zap.String("model", name),
				zap.Error(err))
			failed++
		}
	}

	if failed > 0 && failed == len(names) {
		return fmt.Errorf("all %d predictor models failed to preload", failed)
	}
	return nil
}

// Close stops the cache and unloads all models.
func (r *PredictorRegistry) Close() error {
	r.logger.Info("Closing predictor registry")

	r.cache.Stop()

	for _, key := range r.cache.Keys() {
		if item := r.cache.Get(key); item != nil {
			predictor := item.Value()
			r.logger.Debug("Closing cached predictor",
				zap.String("model", key))
			if err := predictor.Close(); err != nil {
				r.logger.Warn("Error closing predictor",
					zap.String("model", key),
					zap.Error(err))
			}
		}
	}

	r.cache.DeleteAll()

	logDrainErrors(r.logger, "predictor", r.refs.drainOrphans())

	return nil
}
