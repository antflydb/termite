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
	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
	"github.com/antflydb/termite/pkg/termite/lib/ner"
	termreranking "github.com/antflydb/termite/pkg/termite/lib/reranking"
	"github.com/antflydb/termite/pkg/termite/lib/seq2seq"
	khugot "github.com/knights-analytics/hugot"
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

// ChunkerRegistry manages multiple chunker models loaded from a directory
type ChunkerRegistry struct {
	models map[string]chunking.Chunker // model name -> chunker instance
	mu     sync.RWMutex
	logger *zap.Logger
}

// NewChunkerRegistry creates a registry and discovers models in the given directory
// Directory structure: modelsDir/model_name/model.onnx
// If sharedSession is provided, all models will share the same Hugot session (required for ONNX Runtime)
func NewChunkerRegistry(modelsDir string, sharedSession *khugot.Session, logger *zap.Logger) (*ChunkerRegistry, error) {
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

			// Pass model path, ONNX filename, and shared session to pooled chunker
			chunker, err := termchunking.NewPooledHugotChunkerWithSession(config, modelPath, onnxFilename, poolSize, sharedSession, logger.Named(registryName))
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
// If sharedSession is provided, all models will share the same Hugot session (required for ONNX Runtime)
func NewRerankerRegistry(modelsDir string, sharedSession *khugot.Session, logger *zap.Logger) (*RerankerRegistry, error) {
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

			// Pass model path, ONNX filename, and shared session to pooled reranker
			model, err := termreranking.NewPooledHugotRerankerWithSession(modelPath, onnxFilename, poolSize, sharedSession, logger.Named(registryName))
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
// If sharedSession is provided, all models will share the same Hugot session (required for ONNX Runtime)
func NewEmbedderRegistry(modelsDir string, sharedSession *khugot.Session, logger *zap.Logger) (*EmbedderRegistry, error) {
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

			// Pass model path, ONNX filename, and shared session to pooled embedder
			model, err := termembeddings.NewPooledHugotEmbedderWithSession(modelPath, onnxFilename, poolSize, sharedSession, logger.Named(registryName))
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

// NERRegistry manages multiple NER (Named Entity Recognition) models loaded from a directory
type NERRegistry struct {
	models       map[string]ner.Model       // model name -> NER model instance
	glinerModels map[string]ner.GLiNERModel // model name -> GLiNER model instance
	mu           sync.RWMutex
	logger       *zap.Logger
}

// NewNERRegistry creates a registry and discovers NER models in the given directory
// If sharedSession is provided, all models will share the same Hugot session (required for ONNX Runtime)
func NewNERRegistry(modelsDir string, sharedSession *khugot.Session, logger *zap.Logger) (*NERRegistry, error) {
	registry := &NERRegistry{
		models:       make(map[string]ner.Model),
		glinerModels: make(map[string]ner.GLiNERModel),
		logger:       logger,
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
		isGLiNER := ner.IsGLiNERModel(modelPath)

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

			model, err := ner.NewHugotGLiNERWithSession(modelPath, quantized, sharedSession, logger.Named(modelName))
			if err != nil {
				logger.Warn("Failed to load GLiNER model",
					zap.String("name", modelName),
					zap.Bool("quantized", quantized),
					zap.Error(err))
			} else {
				registry.glinerModels[modelName] = model
				registry.models[modelName] = model // Also register as regular Model for compatibility
				logger.Info("Successfully loaded GLiNER model",
					zap.String("name", modelName),
					zap.Bool("quantized", quantized),
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

				// Pass model path, ONNX filename, and shared session to pooled NER model
				model, err := ner.NewPooledHugotNERWithSession(modelPath, onnxFilename, poolSize, sharedSession, logger.Named(registryName))
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
						zap.Int("poolSize", poolSize))
				}
			}
		}
	}

	logger.Info("NER registry initialized",
		zap.Int("models_loaded", len(registry.models)),
		zap.Int("gliner_models", len(registry.glinerModels)))

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

// GetGLiNER returns a GLiNER model by name (for zero-shot NER with custom labels)
func (r *NERRegistry) GetGLiNER(modelName string) (ner.GLiNERModel, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	model, ok := r.glinerModels[modelName]
	if !ok {
		return nil, fmt.Errorf("GLiNER model not found: %s", modelName)
	}
	return model, nil
}

// IsGLiNER returns true if the model is a GLiNER model
func (r *NERRegistry) IsGLiNER(modelName string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	_, ok := r.glinerModels[modelName]
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

// ListGLiNER returns all available GLiNER model names
func (r *NERRegistry) ListGLiNER() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.glinerModels))
	for name := range r.glinerModels {
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
// If sharedSession is provided, all models will share the same Hugot session
func NewSeq2SeqRegistry(modelsDir string, sharedSession *khugot.Session, logger *zap.Logger) (*Seq2SeqRegistry, error) {
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
		model, err := seq2seq.NewHugotSeq2SeqWithSession(modelPath, sharedSession, logger.Named(modelName))
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
				zap.Int("max_length", config.MaxLength))
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
