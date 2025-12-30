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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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
	"github.com/antflydb/termite/pkg/termite/lib/rebel"
	termreranking "github.com/antflydb/termite/pkg/termite/lib/reranking"
	"github.com/antflydb/termite/pkg/termite/lib/seq2seq"
	"github.com/antflydb/termite/pkg/termite/lib/generation"
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

// DiscoveredModel represents a model found during directory scanning
type DiscoveredModel struct {
	// Ref is the parsed model reference (owner/name)
	Ref modelregistry.ModelRef
	// Path is the absolute path to the model directory
	Path string
	// Manifest is the loaded manifest (nil if not found/generated)
	Manifest *modelregistry.ModelManifest
	// Variants maps variant ID to ONNX filename
	Variants map[string]string
}

// RegistryName returns the short name for registry lookups
func (d *DiscoveredModel) RegistryName() string {
	return d.Ref.Name
}

// FullName returns the full owner/name
func (d *DiscoveredModel) FullName() string {
	return d.Ref.FullName()
}

// discoverModelsInDir scans a directory for models using the owner/model structure.
// It supports both the new owner/model structure and the legacy flat structure.
//
// New structure: modelsDir/owner/model-name/model.onnx
// Legacy structure: modelsDir/model-name/model.onnx
func discoverModelsInDir(modelsDir string, modelType modelregistry.ModelType, logger *zap.Logger) ([]DiscoveredModel, error) {
	if modelsDir == "" {
		return nil, nil
	}

	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		return nil, nil
	}

	var discovered []DiscoveredModel

	// First level: could be owner directories or model directories (legacy)
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return nil, fmt.Errorf("reading models directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		firstLevelPath := filepath.Join(modelsDir, entry.Name())

		// Check if this is an owner directory (has subdirectories with models)
		// or a model directory itself (legacy flat structure)
		subEntries, err := os.ReadDir(firstLevelPath)
		if err != nil {
			logger.Debug("Failed to read directory", zap.String("path", firstLevelPath), zap.Error(err))
			continue
		}

		// Check if this looks like a model directory (has model.onnx or manifest)
		isModelDir := hasModelFiles(firstLevelPath)

		if isModelDir {
			// Legacy flat structure: modelsDir/model-name/
			model := discoverSingleModel(firstLevelPath, "", entry.Name(), modelType, logger)
			if model != nil {
				discovered = append(discovered, *model)
			}
		} else {
			// New owner structure: modelsDir/owner/model-name/
			owner := entry.Name()
			for _, subEntry := range subEntries {
				if !subEntry.IsDir() {
					continue
				}
				modelPath := filepath.Join(firstLevelPath, subEntry.Name())
				model := discoverSingleModel(modelPath, owner, subEntry.Name(), modelType, logger)
				if model != nil {
					discovered = append(discovered, *model)
				}
			}
		}
	}

	return discovered, nil
}

// hasModelFiles checks if a directory contains model files
func hasModelFiles(path string) bool {
	// Check for common model indicators
	indicators := []string{
		"model.onnx",
		"model_manifest.json",
		"visual_model.onnx", // CLIP
		"encoder.onnx",      // seq2seq
		"genai_config.json", // generator
	}
	for _, indicator := range indicators {
		if fileExistsRegistry(filepath.Join(path, indicator)) {
			return true
		}
	}
	return false
}

// discoverSingleModel discovers a model from a directory
func discoverSingleModel(modelPath, owner, name string, modelType modelregistry.ModelType, logger *zap.Logger) *DiscoveredModel {
	// Try to load manifest first
	manifest, err := modelregistry.LoadManifestFromDir(modelPath)
	if err != nil {
		// Manifest not found or invalid - discover from files
		variants := discoverModelVariants(modelPath)
		if len(variants) == 0 {
			// Check for multimodal or seq2seq models
			hasStd, hasQt := isMultimodalModel(modelPath)
			if !hasStd && !hasQt && !isSeq2SeqModelDir(modelPath) && !isGeneratorModelDir(modelPath) {
				return nil // No model files found
			}
		}

		// Create a basic manifest from discovery
		manifest = &modelregistry.ModelManifest{
			SchemaVersion: modelregistry.CurrentSchemaVersion,
			Name:          name,
			Owner:         owner,
			Type:          modelType,
		}
		if owner != "" {
			manifest.Source = owner + "/" + name
		} else {
			manifest.Source = name
		}
	} else {
		// Override owner/name from directory structure if manifest is legacy
		if manifest.Owner == "" && owner != "" {
			manifest.Owner = owner
			manifest.Source = owner + "/" + manifest.Name
		}
	}

	variants := discoverModelVariants(modelPath)

	return &DiscoveredModel{
		Ref: modelregistry.ModelRef{
			Owner: owner,
			Name:  name,
		},
		Path:     modelPath,
		Manifest: manifest,
		Variants: variants,
	}
}

// isSeq2SeqModelDir checks if a directory contains a seq2seq model
func isSeq2SeqModelDir(path string) bool {
	return fileExistsRegistry(filepath.Join(path, "encoder.onnx")) &&
		fileExistsRegistry(filepath.Join(path, "decoder.onnx"))
}

// isGeneratorModelDir checks if a directory contains a generator model
func isGeneratorModelDir(path string) bool {
	return fileExistsRegistry(filepath.Join(path, "genai_config.json")) ||
		(fileExistsRegistry(filepath.Join(path, "config.json")) &&
			fileExistsRegistry(filepath.Join(path, "model.onnx")))
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
	models       map[string]ner.Model      // model name -> NER model instance
	recognizers  map[string]ner.Recognizer // model name -> zero-shot capable Recognizer (e.g., GLiNER, REBEL)
	capabilities map[string][]string       // model name -> list of capabilities (labels, zeroshot, relations, answers)
	mu           sync.RWMutex
	logger       *zap.Logger
}

// loadRecognizerCapabilities loads capabilities from manifest.json or auto-detects them.
// Priority: manifest.json > auto-detection based on model type
func loadRecognizerCapabilities(modelPath string, isGLiNER bool, glinerModel *gliner.HugotGLiNER) []string {
	// Try to load from manifest.json first
	manifestPath := filepath.Join(modelPath, "manifest.json")
	if data, err := os.ReadFile(manifestPath); err == nil {
		var manifest modelregistry.ModelManifest
		if err := json.Unmarshal(data, &manifest); err == nil && len(manifest.Capabilities) > 0 {
			return manifest.Capabilities
		}
	}

	// Auto-detect based on model type
	if isGLiNER && glinerModel != nil {
		caps := []string{modelregistry.CapabilityLabels, modelregistry.CapabilityZeroshot}
		// Check if model supports relations (multitask models)
		if glinerModel.SupportsRelationExtraction() {
			caps = append(caps, modelregistry.CapabilityRelations)
		}
		// Check if model supports QA (multitask models)
		if glinerModel.SupportsQA() {
			caps = append(caps, modelregistry.CapabilityAnswers)
		}
		return caps
	}

	// Default for traditional NER models
	return []string{modelregistry.CapabilityLabels}
}

// NewNERRegistry creates a registry and discovers NER models in the given directory
// If sessionManager is provided, all models will use it for backend selection (required for ONNX Runtime)
func NewNERRegistry(modelsDir string, sessionManager *hugot.SessionManager, logger *zap.Logger) (*NERRegistry, error) {
	registry := &NERRegistry{
		models:       make(map[string]ner.Model),
		recognizers:  make(map[string]ner.Recognizer),
		capabilities: make(map[string][]string),
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

		// Check model type: GLiNER, REBEL, or traditional NER
		isGLiNER := gliner.IsGLiNERModel(modelPath)
		isREBEL := rebel.IsREBELModel(modelPath)

		if isREBEL {
			// Load REBEL relation extraction model
			logger.Info("Discovered REBEL model directory",
				zap.String("name", modelName),
				zap.String("path", modelPath))

			model, backendUsed, err := rebel.NewHugotREBELWithSessionManager(modelPath, sessionManager, logger.Named(modelName))
			if err != nil {
				logger.Warn("Failed to load REBEL model",
					zap.String("name", modelName),
					zap.Error(err))
			} else {
				// REBEL implements ner.Recognizer with relation extraction capability
				registry.recognizers[modelName] = model
				registry.models[modelName] = model
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
				registry.capabilities[modelName] = caps
				logger.Info("Successfully loaded REBEL model",
					zap.String("name", modelName),
					zap.String("backend", string(backendUsed)),
					zap.Strings("capabilities", caps))
			}
		} else if isGLiNER {
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
				// Load capabilities from manifest or auto-detect
				caps := loadRecognizerCapabilities(modelPath, true, model)
				registry.capabilities[modelName] = caps
				logger.Info("Successfully loaded GLiNER model",
					zap.String("name", modelName),
					zap.Bool("quantized", quantized),
					zap.String("backend", string(backendUsed)),
					zap.Strings("default_labels", model.Labels()),
					zap.Strings("capabilities", caps))
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
					// Load capabilities from manifest or use default (labels only for traditional NER)
					caps := loadRecognizerCapabilities(modelPath, false, nil)
					registry.capabilities[registryName] = caps
					logger.Info("Successfully loaded NER model",
						zap.String("name", registryName),
						zap.String("onnxFile", onnxFilename),
						zap.String("backend", string(backendUsed)),
						zap.Int("poolSize", poolSize),
						zap.Strings("capabilities", caps))
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

// GetCapabilities returns the capabilities for a specific model.
// Returns nil if the model is not found.
func (r *NERRegistry) GetCapabilities(modelName string) []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if caps, ok := r.capabilities[modelName]; ok {
		return caps
	}
	return nil
}

// HasCapability checks if a model has a specific capability.
func (r *NERRegistry) HasCapability(modelName, capability string) bool {
	caps := r.GetCapabilities(modelName)
	for _, c := range caps {
		if c == capability {
			return true
		}
	}
	return false
}

// ListWithCapabilities returns a map of model name to capabilities for all models.
func (r *NERRegistry) ListWithCapabilities() map[string][]string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string][]string, len(r.capabilities))
	for name, caps := range r.capabilities {
		// Make a copy of the slice to avoid sharing internal state
		capsCopy := make([]string, len(caps))
		copy(capsCopy, caps)
		result[name] = capsCopy
	}
	return result
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

// GeneratorRegistry manages multiple generator (LLM) models loaded from a directory
type GeneratorRegistry struct {
	models map[string]generation.Generator // model name -> generator instance
	mu     sync.RWMutex
	logger *zap.Logger
}

// generateGenaiConfig creates a genai_config.json file from a HuggingFace config.json.
// This enables ONNX Runtime GenAI to load standard HuggingFace ONNX models.
// Returns nil if successful, error otherwise.
func generateGenaiConfig(modelPath string, logger *zap.Logger) error {
	genaiConfigPath := filepath.Join(modelPath, "genai_config.json")

	// Skip if genai_config.json already exists
	if _, err := os.Stat(genaiConfigPath); err == nil {
		return nil
	}

	// Read HuggingFace config.json
	configPath := filepath.Join(modelPath, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("reading config.json: %w", err)
	}

	var hfConfig map[string]any
	if err := json.Unmarshal(configData, &hfConfig); err != nil {
		return fmt.Errorf("parsing config.json: %w", err)
	}

	// Determine model type from HuggingFace config
	modelType := "gpt2" // default fallback
	if mt, ok := hfConfig["model_type"].(string); ok {
		// Map HuggingFace model types to GenAI types
		switch mt {
		case "gemma", "gemma2", "gemma3_text":
			modelType = "gemma"
		case "llama":
			modelType = "llama"
		case "mistral":
			modelType = "mistral"
		case "phi", "phi3":
			modelType = "phi"
		case "qwen2":
			modelType = "qwen2"
		case "gpt2":
			modelType = "gpt2"
		default:
			// Try to infer from architectures
			if archs, ok := hfConfig["architectures"].([]any); ok && len(archs) > 0 {
				archStr := fmt.Sprintf("%v", archs[0])
				archLower := strings.ToLower(archStr)
				if strings.Contains(archLower, "gemma") {
					modelType = "gemma"
				} else if strings.Contains(archLower, "llama") {
					modelType = "llama"
				} else if strings.Contains(archLower, "mistral") {
					modelType = "mistral"
				}
			}
		}
	}

	logger.Info("Generating genai_config.json",
		zap.String("modelPath", modelPath),
		zap.String("modelType", modelType))

	// Extract relevant config values with defaults
	vocabSize := 32000
	if vs, ok := hfConfig["vocab_size"].(float64); ok {
		vocabSize = int(vs)
	}

	hiddenSize := 2048
	if hs, ok := hfConfig["hidden_size"].(float64); ok {
		hiddenSize = int(hs)
	}

	numHiddenLayers := 16
	if nhl, ok := hfConfig["num_hidden_layers"].(float64); ok {
		numHiddenLayers = int(nhl)
	}

	numAttentionHeads := 8
	if nah, ok := hfConfig["num_attention_heads"].(float64); ok {
		numAttentionHeads = int(nah)
	}

	numKeyValueHeads := numAttentionHeads
	if nkvh, ok := hfConfig["num_key_value_heads"].(float64); ok {
		numKeyValueHeads = int(nkvh)
	}

	intermediateSize := hiddenSize * 4
	if is, ok := hfConfig["intermediate_size"].(float64); ok {
		intermediateSize = int(is)
	}

	headDim := hiddenSize / numAttentionHeads
	if hd, ok := hfConfig["head_dim"].(float64); ok {
		headDim = int(hd)
	}

	// Build genai_config.json
	genaiConfig := map[string]any{
		"model": map[string]any{
			"bos_token_id": 2,
			"context_length": func() int {
				if cl, ok := hfConfig["max_position_embeddings"].(float64); ok {
					return int(cl)
				}
				return 8192
			}(),
			"decoder": map[string]any{
				"session_options": map[string]any{},
				"filename":        "model.onnx",
				"head_dim":        headDim,
				"hidden_size":     hiddenSize,
				"inputs": map[string]string{
					"input_ids":      "input_ids",
					"attention_mask": "attention_mask",
					"position_ids":   "position_ids",
				},
				"num_attention_heads": numAttentionHeads,
				"num_hidden_layers":   numHiddenLayers,
				"num_key_value_heads": numKeyValueHeads,
				"outputs": map[string]string{
					"logits": "logits",
				},
			},
			"eos_token_id":      1,
			"pad_token_id":      0,
			"type":              modelType,
			"vocab_size":        vocabSize,
			"intermediate_size": intermediateSize,
		},
		"search": map[string]any{
			"diversity_penalty": 0.0,
			"do_sample":         false,
			"early_stopping":    true,
			"length_penalty":    1.0,
			"max_length":        2048,
			"min_length":        0,
			"no_repeat_ngram_size": 0,
			"num_beams":         1,
			"num_return_sequences": 1,
			"past_present_share_buffer": false,
			"repetition_penalty": 1.0,
			"temperature":        1.0,
			"top_k":              1,
			"top_p":              1.0,
		},
	}

	// Write the file
	genaiData, err := json.MarshalIndent(genaiConfig, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling genai_config: %w", err)
	}

	if err := os.WriteFile(genaiConfigPath, genaiData, 0644); err != nil {
		return fmt.Errorf("writing genai_config.json: %w", err)
	}

	logger.Info("Generated genai_config.json successfully",
		zap.String("path", genaiConfigPath))

	return nil
}

// isValidGeneratorModel checks if a model directory contains a valid generator model.
// Supports both ONNX Runtime GenAI format (genai_config.json) and HuggingFace ONNX format.
func isValidGeneratorModel(modelPath string) bool {
	// Check for ONNX Runtime GenAI format
	if _, err := os.Stat(filepath.Join(modelPath, "genai_config.json")); err == nil {
		return true
	}

	// Check for standard HuggingFace ONNX format (config.json + model.onnx)
	hasConfig := false
	hasModel := false

	if _, err := os.Stat(filepath.Join(modelPath, "config.json")); err == nil {
		hasConfig = true
	}

	// Check for model.onnx in root or onnx/ subdirectory
	if _, err := os.Stat(filepath.Join(modelPath, "model.onnx")); err == nil {
		hasModel = true
	} else if _, err := os.Stat(filepath.Join(modelPath, "onnx", "model.onnx")); err == nil {
		hasModel = true
	}

	return hasConfig && hasModel
}

// NewGeneratorRegistry creates a registry and discovers models in the given directory
// Directory structure: modelsDir/model_name/model.onnx (or model_name/onnx/model.onnx)
// If sessionManager is provided, it will be used to obtain sessions for model loading
func NewGeneratorRegistry(modelsDir string, sessionManager *hugot.SessionManager, logger *zap.Logger) (*GeneratorRegistry, error) {
	registry := &GeneratorRegistry{
		models: make(map[string]generation.Generator),
		logger: logger,
	}

	if modelsDir == "" {
		logger.Info("No generator models directory configured")
		return registry, nil
	}

	// Check if directory exists
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		logger.Warn("Generator models directory does not exist",
			zap.String("dir", modelsDir))
		return registry, nil
	}

	// Scan directory for model subdirectories
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return nil, fmt.Errorf("reading generator models directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		modelName := entry.Name()
		modelPath := filepath.Join(modelsDir, modelName)

		// Check for genai_config.json (preferred) or model.onnx in root or onnx/ subdirectory
		hasValidModel := false

		// First check for genai_config.json (onnxruntime-genai native format)
		genaiConfigPath := filepath.Join(modelPath, "genai_config.json")
		if _, err := os.Stat(genaiConfigPath); err == nil {
			hasValidModel = true
		}

		// Fall back to checking for model.onnx
		if !hasValidModel {
			for _, subpath := range []string{"", "onnx"} {
				checkPath := filepath.Join(modelPath, subpath, "model.onnx")
				if _, err := os.Stat(checkPath); err == nil {
					hasValidModel = true
					if subpath != "" {
						modelPath = filepath.Join(modelPath, subpath)
					}
					break
				}
			}
		}

		if !hasValidModel {
			logger.Debug("Skipping directory - no genai_config.json or model.onnx found",
				zap.String("dir", modelName))
			continue
		}

		logger.Info("Discovered generator model directory",
			zap.String("name", modelName),
			zap.String("path", modelPath))

		// Try to generate genai_config.json if needed
		if err := generateGenaiConfig(modelPath, logger); err != nil {
			logger.Warn("Failed to generate genai_config.json",
				zap.String("name", modelName),
				zap.Error(err))
		}

		// Load the generator model
		var model generation.Generator
		var backendUsed hugot.BackendType
		var err error

		if sessionManager != nil {
			model, backendUsed, err = generation.NewHugotGeneratorWithSessionManager(modelPath, sessionManager, nil, logger.Named(modelName))
		} else {
			model, err = generation.NewHugotGeneratorWithSession(modelPath, nil, logger.Named(modelName))
		}

		if err != nil {
			logger.Warn("Failed to load generator model",
				zap.String("name", modelName),
				zap.Error(err))
		} else {
			registry.models[modelName] = model
			logger.Info("Successfully loaded generator model",
				zap.String("name", modelName),
				zap.String("backend", string(backendUsed)))
		}
	}

	logger.Info("Generator registry initialized",
		zap.Int("models_loaded", len(registry.models)))

	return registry, nil
}

// Get returns a generator by name
func (r *GeneratorRegistry) Get(modelName string) (generation.Generator, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	model, ok := r.models[modelName]
	if !ok {
		return nil, fmt.Errorf("generator model not found: %s", modelName)
	}
	return model, nil
}

// List returns all available generator model names
func (r *GeneratorRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.models))
	for name := range r.models {
		names = append(names, name)
	}
	return names
}

// Close closes all loaded generator models
func (r *GeneratorRegistry) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	for name, model := range r.models {
		if err := model.Close(); err != nil {
			r.logger.Warn("Error closing generator model",
				zap.String("name", name),
				zap.Error(err))
		}
	}
	return nil
}
