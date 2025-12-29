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
	"github.com/antflydb/termite/pkg/termite/lib/gliner"
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
	"github.com/antflydb/termite/pkg/termite/lib/ner"
	termreranking "github.com/antflydb/termite/pkg/termite/lib/reranking"
	"github.com/antflydb/termite/pkg/termite/lib/seq2seq"
	"go.uber.org/zap"
)

// discoverModelVariants scans a model directory and returns a map of variant ID to ONNX filename.
// The default FP32 model (model.onnx) is returned with an empty string key.
func discoverModelVariants(modelPath string) map[string]string {
	variants := make(map[string]string)
	usedFilenames := make(map[string]bool)

	// Check for standard FP32 model
	if _, err := os.Stat(filepath.Join(modelPath, "model.onnx")); err == nil {
		variants[""] = "model.onnx" // Empty key = default/FP32
		usedFilenames["model.onnx"] = true
	}

	// Check for all known variant files, but skip if filename already used
	for variantID, filename := range modelregistry.VariantFilenames {
		if usedFilenames[filename] {
			continue // Skip duplicates (e.g., f32 uses same model.onnx as default)
		}
		if _, err := os.Stat(filepath.Join(modelPath, filename)); err == nil {
			variants[variantID] = filename
			usedFilenames[filename] = true
		}
	}

	return variants
}

// isMultimodalModel checks if a model directory contains CLIP-style multimodal model files.
// These models have visual_model.onnx + text_model.onnx instead of a single model.onnx.
func isMultimodalModel(modelPath string) (hasStandard, hasQuantized bool) {
	visualPath := filepath.Join(modelPath, "visual_model.onnx")
	textPath := filepath.Join(modelPath, "text_model.onnx")
	visualQuantizedPath := filepath.Join(modelPath, "visual_model_quantized.onnx")
	textQuantizedPath := filepath.Join(modelPath, "text_model_quantized.onnx")

	hasStandard = fileExistsRegistry(visualPath) && fileExistsRegistry(textPath)
	hasQuantized = fileExistsRegistry(visualQuantizedPath) && fileExistsRegistry(textQuantizedPath)
	return
}

// fileExistsRegistry checks if a file exists
func fileExistsRegistry(path string) bool {
	_, err := os.Stat(path)
	return err == nil
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

		// Check if this is a multimodal (CLIP-style) model
		hasMultimodalStd, hasMultimodalQt := isMultimodalModel(modelPath)
		if hasMultimodalStd || hasMultimodalQt {
			logger.Info("Discovered multimodal embedder model directory",
				zap.String("name", modelName),
				zap.String("path", modelPath),
				zap.Bool("has_standard", hasMultimodalStd),
				zap.Bool("has_quantized", hasMultimodalQt))

			// Load standard precision multimodal model
			if hasMultimodalStd {
				model, backendUsed, err := termembeddings.NewHugotCLIPEmbedderWithSessionManager(modelPath, false, sessionManager, nil, logger.Named(modelName))
				if err != nil {
					logger.Warn("Failed to load multimodal embedder model",
						zap.String("name", modelName),
						zap.Error(err))
				} else {
					registry.models[modelName] = model
					logger.Info("Successfully loaded multimodal embedder model",
						zap.String("name", modelName),
						zap.String("type", "clip"),
						zap.String("backend", string(backendUsed)))
				}
			}

			// Load quantized multimodal model with suffix
			if hasMultimodalQt {
				quantizedName := modelName + "-i8-qt"
				model, backendUsed, err := termembeddings.NewHugotCLIPEmbedderWithSessionManager(modelPath, true, sessionManager, nil, logger.Named(quantizedName))
				if err != nil {
					logger.Warn("Failed to load quantized multimodal embedder model",
						zap.String("name", quantizedName),
						zap.Error(err))
				} else {
					registry.models[quantizedName] = model
					logger.Info("Successfully loaded quantized multimodal embedder model",
						zap.String("name", quantizedName),
						zap.String("type", "clip"),
						zap.String("backend", string(backendUsed)))
				}
			}
			continue // Skip standard embedder loading for multimodal models
		}

		// Discover all available model variants (standard embedders)
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
		case *termembeddings.PooledHugotEmbedder:
			err = emb.Close()
		case *termembeddings.HugotCLIPEmbedder:
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

// NERRegistry manages multiple NER (Named Entity Recognition) models loaded from a directory
type NERRegistry struct {
	models      map[string]ner.Model      // model name -> NER model instance
	recognizers map[string]ner.Recognizer // model name -> zero-shot capable Recognizer (e.g., GLiNER)
	mu          sync.RWMutex
	logger      *zap.Logger
}

// NewNERRegistry creates a registry and discovers NER models in the given directory
// If sessionManager is provided, all models will use it for backend selection (required for ONNX Runtime)
func NewNERRegistry(modelsDir string, sessionManager *hugot.SessionManager, logger *zap.Logger) (*NERRegistry, error) {
	registry := &NERRegistry{
		models:      make(map[string]ner.Model),
		recognizers: make(map[string]ner.Recognizer),
		logger:      logger,
	}

	if modelsDir == "" {
		logger.Info("No NER models directory configured")
		return registry, nil
	}

	// Check if directory exists
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		logger.Warn("NER models directory does not exist",
			zap.String("dir", modelsDir))
		return registry, nil
	}

	// Scan directory for model subdirectories
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return nil, fmt.Errorf("reading NER models directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		modelName := entry.Name()
		modelPath := filepath.Join(modelsDir, modelName)

		// Check if this is a GLiNER model
		isGLiNER := gliner.IsGLiNERModel(modelPath)

		if isGLiNER {
			// Load GLiNER model (no variants, uses model.onnx or model_quantized.onnx)
			logger.Info("Discovered GLiNER model directory",
				zap.String("name", modelName),
				zap.String("path", modelPath))

			// Try quantized first, then non-quantized
			quantized := false
			if _, err := os.Stat(filepath.Join(modelPath, "model_quantized.onnx")); err == nil {
				quantized = true
			} else if _, err := os.Stat(filepath.Join(modelPath, "model.onnx")); err != nil {
				logger.Debug("Skipping GLiNER directory without model files",
					zap.String("dir", modelName))
				continue
			}

			model, backendUsed, err := gliner.NewHugotGLiNERWithSessionManager(modelPath, quantized, sessionManager, nil, logger.Named(modelName))
			if err != nil {
				logger.Warn("Failed to load GLiNER model",
					zap.String("name", modelName),
					zap.Bool("quantized", quantized),
					zap.Error(err))
			} else {
				registry.recognizers[modelName] = model
				registry.models[modelName] = model // Also register as regular Model for compatibility
				logger.Info("Successfully loaded GLiNER model",
					zap.String("name", modelName),
					zap.Bool("quantized", quantized),
					zap.String("backend", string(backendUsed)),
					zap.Strings("default_labels", model.Labels()))
			}
		} else {
			// Discover all available model variants for regular NER models
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
			logger.Info("Discovered NER model directory",
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

				// Pass model path, ONNX filename, and session manager to pooled NER model
				model, backendUsed, err := ner.NewPooledHugotNERWithSessionManager(modelPath, onnxFilename, poolSize, sessionManager, nil, logger.Named(registryName))
				if err != nil {
					logger.Warn("Failed to load NER model variant",
						zap.String("name", registryName),
						zap.String("onnxFile", onnxFilename),
						zap.Error(err))
				} else {
					registry.models[registryName] = model
					logger.Info("Successfully loaded NER model",
						zap.String("name", registryName),
						zap.String("onnxFile", onnxFilename),
						zap.String("backend", string(backendUsed)),
						zap.Int("poolSize", poolSize))
				}
			}
		}
	}

	logger.Info("NER registry initialized",
		zap.Int("models_loaded", len(registry.models)),
		zap.Int("recognizers", len(registry.recognizers)))

	return registry, nil
}

// Get returns a NER model by name
func (r *NERRegistry) Get(modelName string) (ner.Model, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	model, ok := r.models[modelName]
	if !ok {
		return nil, fmt.Errorf("NER model not found: %s", modelName)
	}
	return model, nil
}

// GetRecognizer returns a zero-shot capable Recognizer by name (e.g., GLiNER)
func (r *NERRegistry) GetRecognizer(modelName string) (ner.Recognizer, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	model, ok := r.recognizers[modelName]
	if !ok {
		return nil, fmt.Errorf("Recognizer not found: %s", modelName)
	}
	return model, nil
}

// IsRecognizer returns true if the model is a zero-shot capable Recognizer
func (r *NERRegistry) IsRecognizer(modelName string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	_, ok := r.recognizers[modelName]
	return ok
}

// List returns all available NER model names
func (r *NERRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.models))
	for name := range r.models {
		names = append(names, name)
	}
	return names
}

// ListRecognizers returns all available zero-shot Recognizer model names
func (r *NERRegistry) ListRecognizers() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.recognizers))
	for name := range r.recognizers {
		names = append(names, name)
	}
	return names
}

// Close closes all loaded NER models
func (r *NERRegistry) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	for name, model := range r.models {
		if err := model.Close(); err != nil {
			r.logger.Warn("Error closing NER model",
				zap.String("name", name),
				zap.Error(err))
		}
	}
	return nil
}

// Seq2SeqRegistry manages multiple Seq2Seq text generation models loaded from a directory
type Seq2SeqRegistry struct {
	models map[string]seq2seq.Model // model name -> Seq2Seq model instance
	mu     sync.RWMutex
	logger *zap.Logger
}

// NewSeq2SeqRegistry creates a registry and discovers Seq2Seq models in the given directory
// Seq2Seq models have encoder.onnx, decoder-init.onnx, and decoder.onnx files
// If sessionManager is provided, all models will use it for backend selection
func NewSeq2SeqRegistry(modelsDir string, sessionManager *hugot.SessionManager, logger *zap.Logger) (*Seq2SeqRegistry, error) {
	registry := &Seq2SeqRegistry{
		models: make(map[string]seq2seq.Model),
		logger: logger,
	}

	if modelsDir == "" {
		logger.Info("No Seq2Seq models directory configured")
		return registry, nil
	}

	// Check if directory exists
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		logger.Warn("Seq2Seq models directory does not exist",
			zap.String("dir", modelsDir))
		return registry, nil
	}

	// Scan directory for model subdirectories
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return nil, fmt.Errorf("reading Seq2Seq models directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		modelName := entry.Name()
		modelPath := filepath.Join(modelsDir, modelName)

		// Check if this is a Seq2Seq model (has encoder.onnx, decoder-init.onnx, decoder.onnx)
		if !seq2seq.IsSeq2SeqModel(modelPath) {
			logger.Debug("Skipping directory - not a Seq2Seq model",
				zap.String("dir", modelName))
			continue
		}

		logger.Info("Discovered Seq2Seq model directory",
			zap.String("name", modelName),
			zap.String("path", modelPath))

		// Load the Seq2Seq model
		model, backendUsed, err := seq2seq.NewHugotSeq2SeqWithSessionManager(modelPath, sessionManager, nil, logger.Named(modelName))
		if err != nil {
			logger.Warn("Failed to load Seq2Seq model",
				zap.String("name", modelName),
				zap.Error(err))
		} else {
			registry.models[modelName] = model
			config := model.Config()
			logger.Info("Successfully loaded Seq2Seq model",
				zap.String("name", modelName),
				zap.String("task", config.Task),
				zap.Int("max_length", config.MaxLength),
				zap.String("backend", string(backendUsed)))
		}
	}

	logger.Info("Seq2Seq registry initialized",
		zap.Int("models_loaded", len(registry.models)))

	return registry, nil
}

// Get returns a Seq2Seq model by name
func (r *Seq2SeqRegistry) Get(modelName string) (seq2seq.Model, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	model, ok := r.models[modelName]
	if !ok {
		return nil, fmt.Errorf("Seq2Seq model not found: %s", modelName)
	}
	return model, nil
}

// GetQuestionGenerator returns a Seq2Seq model as a QuestionGenerator by name
func (r *Seq2SeqRegistry) GetQuestionGenerator(modelName string) (seq2seq.QuestionGenerator, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	model, ok := r.models[modelName]
	if !ok {
		return nil, fmt.Errorf("Seq2Seq model not found: %s", modelName)
	}

	qg, ok := model.(seq2seq.QuestionGenerator)
	if !ok {
		return nil, fmt.Errorf("model %s does not support question generation", modelName)
	}
	return qg, nil
}

// List returns all available Seq2Seq model names
func (r *Seq2SeqRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.models))
	for name := range r.models {
		names = append(names, name)
	}
	return names
}

// Close closes all loaded Seq2Seq models
func (r *Seq2SeqRegistry) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	for name, model := range r.models {
		if err := model.Close(); err != nil {
			r.logger.Warn("Error closing Seq2Seq model",
				zap.String("name", name),
				zap.Error(err))
		}
	}
	return nil
}
