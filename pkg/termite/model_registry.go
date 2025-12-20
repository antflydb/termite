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
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/antflydb/antfly-go/libaf/chunking"
	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/antfly-go/libaf/reranking"
	termchunking "github.com/antflydb/termite/pkg/termite/lib/chunking"
	termembeddings "github.com/antflydb/termite/pkg/termite/lib/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
	termreranking "github.com/antflydb/termite/pkg/termite/lib/reranking"
	"go.uber.org/zap"
)

// discoverModelVariants scans a model directory and returns a map of variant ID to ONNX filename.
// The default FP32 model (model.onnx) is returned with an empty string key.
func discoverModelVariants(modelPath string) map[string]string {
	variants := make(map[string]string)

	// Check for standard FP32 model
	if _, err := os.Stat(filepath.Join(modelPath, "model.onnx")); err == nil {
		variants[""] = "model.onnx" // Empty key = default/FP32
	}

	// Check for all known variant files
	for variantID, filename := range modelregistry.VariantFilenames {
		if _, err := os.Stat(filepath.Join(modelPath, filename)); err == nil {
			variants[variantID] = filename
		}
	}

	return variants
}

// ChunkerRegistry manages multiple chunker models loaded from a directory
type ChunkerRegistry struct {
	models map[string]chunking.Chunker // model name -> chunker instance
	mu     sync.RWMutex
	logger *zap.Logger
}

// NewChunkerRegistry creates a registry and discovers models in the given directory
// Directory structure: modelsDir/model_name/model.onnx
// If sessionManager is provided, it will be used to obtain sessions for model loading (required for ONNX Runtime)
func NewChunkerRegistry(modelsDir string, sessionManager *hugot.SessionManager, logger *zap.Logger) (*ChunkerRegistry, error) {
	registry := &ChunkerRegistry{
		models: make(map[string]chunking.Chunker),
		logger: logger,
	}

	if modelsDir == "" {
		logger.Info("No chunker models directory configured, only built-in fixed tokenizer models available")
		return registry, nil
	}

	// Check if directory exists
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		logger.Warn("Chunker models directory does not exist, only built-in fixed tokenizer models available",
			zap.String("dir", modelsDir))
		return registry, nil
	}

	// Scan directory for model subdirectories
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return nil, fmt.Errorf("reading models directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		modelName := entry.Name()
		modelPath := filepath.Join(modelsDir, modelName)

		// Discover all available model variants
		variants := discoverModelVariants(modelPath)

		// Skip if no model files exist
		if len(variants) == 0 {
			logger.Debug("Skipping directory without model files",
				zap.String("dir", modelName))
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
		logger.Info("Discovered chunker model directory",
			zap.String("name", modelName),
			zap.String("path", modelPath),
			zap.Strings("variants", variantIDs))

		// Pool size for concurrent pipeline access
		// Cap at 4 to avoid excessive memory usage (each pipeline loads full model)
		poolSize := min(runtime.NumCPU(), 4)

		// Load each variant
		for variantID, onnxFilename := range variants {
			// Determine registry name
			registryName := modelName
			if variantID != "" {
				registryName = modelName + "-" + variantID
			}

			// Create chunker config for this model with sensible defaults
			config := termchunking.DefaultHugotChunkerConfig()

			// Pass model path, ONNX filename, and session manager to pooled chunker
			chunker, backendUsed, err := termchunking.NewPooledHugotChunkerWithSessionManager(config, modelPath, onnxFilename, poolSize, sessionManager, nil, logger.Named(registryName))
			if err != nil {
				logger.Warn("Failed to load chunker model variant",
					zap.String("name", registryName),
					zap.String("onnxFile", onnxFilename),
					zap.Error(err))
			} else {
				registry.models[registryName] = chunker
				logger.Info("Successfully loaded chunker model",
					zap.String("name", registryName),
					zap.String("onnxFile", onnxFilename),
					zap.String("backend", string(backendUsed)),
					zap.Int("poolSize", poolSize))
			}
		}
	}

	logger.Info("Chunker registry initialized",
		zap.Int("models_loaded", len(registry.models)))

	return registry, nil
}

// Get returns a chunker by model name
func (r *ChunkerRegistry) Get(modelName string) (chunking.Chunker, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	chunker, ok := r.models[modelName]
	if !ok {
		return nil, fmt.Errorf("chunker model not found: %s", modelName)
	}
	return chunker, nil
}

// List returns all available model names
func (r *ChunkerRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.models))
	for name := range r.models {
		names = append(names, name)
	}
	return names
}

// Close closes all loaded models
func (r *ChunkerRegistry) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	for name, chunker := range r.models {
		if err := chunker.Close(); err != nil {
			r.logger.Warn("Error closing chunker model",
				zap.String("name", name),
				zap.Error(err))
		}
	}
	return nil
}

// RerankerRegistry manages multiple reranker models loaded from a directory
type RerankerRegistry struct {
	models map[string]reranking.Model // model name -> reranker instance
	mu     sync.RWMutex
	logger *zap.Logger
}

// NewRerankerRegistry creates a registry and discovers models in the given directory
// If sessionManager is provided, it will be used to obtain sessions for model loading (required for ONNX Runtime)
func NewRerankerRegistry(modelsDir string, sessionManager *hugot.SessionManager, logger *zap.Logger) (*RerankerRegistry, error) {
	registry := &RerankerRegistry{
		models: make(map[string]reranking.Model),
		logger: logger,
	}

	if modelsDir == "" {
		logger.Info("No reranker models directory configured")
		return registry, nil
	}

	// Check if directory exists
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		logger.Warn("Reranker models directory does not exist",
			zap.String("dir", modelsDir))
		return registry, nil
	}

	// Scan directory for model subdirectories
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return nil, fmt.Errorf("reading models directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		modelName := entry.Name()
		modelPath := filepath.Join(modelsDir, modelName)

		// Discover all available model variants
		variants := discoverModelVariants(modelPath)

		// Skip if no model files exist
		if len(variants) == 0 {
			logger.Debug("Skipping directory without model files",
				zap.String("dir", modelName))
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
		logger.Info("Discovered reranker model directory",
			zap.String("name", modelName),
			zap.String("path", modelPath),
			zap.Strings("variants", variantIDs))

		// Pool size for concurrent pipeline access
		// Cap at 4 to avoid excessive memory usage (each pipeline loads full model)
		poolSize := min(runtime.NumCPU(), 4)

		// Load each variant
		for variantID, onnxFilename := range variants {
			// Determine registry name
			registryName := modelName
			if variantID != "" {
				registryName = modelName + "-" + variantID
			}

			// Pass model path, ONNX filename, and session manager to pooled reranker
			model, backendUsed, err := termreranking.NewPooledHugotRerankerWithSessionManager(modelPath, onnxFilename, poolSize, sessionManager, nil, logger.Named(registryName))
			if err != nil {
				logger.Warn("Failed to load reranker model variant",
					zap.String("name", registryName),
					zap.String("onnxFile", onnxFilename),
					zap.Error(err))
			} else {
				registry.models[registryName] = model
				logger.Info("Successfully loaded reranker model",
					zap.String("name", registryName),
					zap.String("onnxFile", onnxFilename),
					zap.String("backend", string(backendUsed)),
					zap.Int("poolSize", poolSize))
			}
		}
	}

	logger.Info("Reranker registry initialized",
		zap.Int("models_loaded", len(registry.models)))

	return registry, nil
}

// Get returns a reranker by model name
func (r *RerankerRegistry) Get(modelName string) (reranking.Model, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	model, ok := r.models[modelName]
	if !ok {
		return nil, fmt.Errorf("reranker model not found: %s", modelName)
	}
	return model, nil
}

// List returns all available model names
func (r *RerankerRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.models))
	for name := range r.models {
		names = append(names, name)
	}
	return names
}

// Close closes all loaded models
func (r *RerankerRegistry) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	for name, model := range r.models {
		if err := model.Close(); err != nil {
			r.logger.Warn("Error closing reranker model",
				zap.String("name", name),
				zap.Error(err))
		}
	}
	return nil
}

// EmbedderRegistry manages multiple embedder models loaded from a directory
type EmbedderRegistry struct {
	models map[string]embeddings.Embedder // model name -> embedder instance
	mu     sync.RWMutex
	logger *zap.Logger
}

// NewEmbedderRegistry creates a registry and discovers models in the given directory
// If sessionManager is provided, it will be used to obtain sessions for model loading (required for ONNX Runtime)
func NewEmbedderRegistry(modelsDir string, sessionManager *hugot.SessionManager, logger *zap.Logger) (*EmbedderRegistry, error) {
	registry := &EmbedderRegistry{
		models: make(map[string]embeddings.Embedder),
		logger: logger,
	}

	if modelsDir == "" {
		logger.Info("No embedder models directory configured")
		return registry, nil
	}

	// Check if directory exists
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		logger.Warn("Embedder models directory does not exist",
			zap.String("dir", modelsDir))
		return registry, nil
	}

	// Scan directory for model subdirectories
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return nil, fmt.Errorf("reading models directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		modelName := entry.Name()
		modelPath := filepath.Join(modelsDir, modelName)

		// Discover all available model variants
		variants := discoverModelVariants(modelPath)

		// Skip if no model files exist
		if len(variants) == 0 {
			logger.Debug("Skipping directory without model files",
				zap.String("dir", modelName))
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
		logger.Info("Discovered embedder model directory",
			zap.String("name", modelName),
			zap.String("path", modelPath),
			zap.Strings("variants", variantIDs))

		// Pool size for concurrent pipeline access
		// Cap at 4 to avoid excessive memory usage (each pipeline loads full model)
		poolSize := min(runtime.NumCPU(), 4)

		// Load each variant
		for variantID, onnxFilename := range variants {
			// Determine registry name
			registryName := modelName
			if variantID != "" {
				registryName = modelName + "-" + variantID
			}

			// Pass model path, ONNX filename, and session manager to pooled embedder
			model, backendUsed, err := termembeddings.NewPooledHugotEmbedderWithSessionManager(modelPath, onnxFilename, poolSize, sessionManager, nil, logger.Named(registryName))
			if err != nil {
				logger.Warn("Failed to load embedder model variant",
					zap.String("name", registryName),
					zap.String("onnxFile", onnxFilename),
					zap.Error(err))
			} else {
				registry.models[registryName] = model
				logger.Info("Successfully loaded embedder model",
					zap.String("name", registryName),
					zap.String("onnxFile", onnxFilename),
					zap.String("backend", string(backendUsed)),
					zap.Int("poolSize", poolSize))
			}
		}
	}

	logger.Info("Embedder registry initialized",
		zap.Int("models_loaded", len(registry.models)))

	return registry, nil
}

// Get returns an embedder by model name
func (r *EmbedderRegistry) Get(modelName string) (embeddings.Embedder, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	model, ok := r.models[modelName]
	if !ok {
		return nil, fmt.Errorf("embedder model not found: %s", modelName)
	}
	return model, nil
}

// List returns all available model names
func (r *EmbedderRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.models))
	for name := range r.models {
		names = append(names, name)
	}
	return names
}

// Close closes all loaded models
func (r *EmbedderRegistry) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	for name, model := range r.models {
		var err error
		switch emb := model.(type) {
		case *termembeddings.HugotEmbedder:
			err = emb.Close()
		case *termembeddings.PooledHugotEmbedder:
			err = emb.Close()
		}
		if err != nil {
			r.logger.Warn("Error closing embedder model",
				zap.String("name", name),
				zap.Error(err))
		}
	}
	return nil
}
