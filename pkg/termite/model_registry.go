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
	"github.com/antflydb/termite/pkg/termite/lib/generation"
	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
	termreranking "github.com/antflydb/termite/pkg/termite/lib/reranking"
	khugot "github.com/knights-analytics/hugot"
	"go.uber.org/zap"
)

// discoverModelVariants scans a model directory and returns a map of variant ID to ONNX filename.
// The default FP32 model (model.onnx) is returned with an empty string key.
// It checks both the root directory and an "onnx/" subdirectory (common for onnx-community models).
func discoverModelVariants(modelPath string) map[string]string {
	variants := make(map[string]string)

	// Directories to check for ONNX files (root and onnx/ subdirectory)
	searchDirs := []string{"", "onnx"}

	for _, subdir := range searchDirs {
		searchPath := modelPath
		prefix := ""
		if subdir != "" {
			searchPath = filepath.Join(modelPath, subdir)
			prefix = subdir + "/"
		}

		// Check for standard FP32 model
		if _, err := os.Stat(filepath.Join(searchPath, "model.onnx")); err == nil {
			if _, exists := variants[""]; !exists {
				variants[""] = prefix + "model.onnx" // Empty key = default/FP32
			}
		}

		// Check for all known variant files
		for variantID, filename := range modelregistry.VariantFilenames {
			if _, err := os.Stat(filepath.Join(searchPath, filename)); err == nil {
				if _, exists := variants[variantID]; !exists {
					variants[variantID] = prefix + filename
				}
			}
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
				if arch, ok := archs[0].(string); ok {
					switch {
					case strings.Contains(arch, "Gemma"):
						modelType = "gemma"
					case strings.Contains(arch, "Llama"):
						modelType = "llama"
					case strings.Contains(arch, "Mistral"):
						modelType = "mistral"
					case strings.Contains(arch, "Phi"):
						modelType = "phi"
					case strings.Contains(arch, "Qwen"):
						modelType = "qwen2"
					}
				}
			}
		}
	}

	// Extract model parameters with defaults
	getInt := func(key string, defaultVal int) int {
		if v, ok := hfConfig[key].(float64); ok {
			return int(v)
		}
		return defaultVal
	}

	// Find the ONNX model file
	onnxFilename := "model.onnx"
	if _, err := os.Stat(filepath.Join(modelPath, "model.onnx")); os.IsNotExist(err) {
		if _, err := os.Stat(filepath.Join(modelPath, "onnx", "model.onnx")); err == nil {
			onnxFilename = "onnx/model.onnx"
		}
	}

	// Build genai_config structure
	genaiConfig := map[string]any{
		"model": map[string]any{
			"bos_token_id":   getInt("bos_token_id", 1),
			"eos_token_id":   getInt("eos_token_id", 2),
			"pad_token_id":   getInt("pad_token_id", 0),
			"vocab_size":     getInt("vocab_size", 32000),
			"context_length": getInt("max_position_embeddings", 2048),
			"type":           modelType,
			"decoder": map[string]any{
				"session_options": map[string]any{
					"provider_options": []any{},
				},
				"filename":            onnxFilename,
				"head_size":           getInt("head_dim", 64),
				"hidden_size":         getInt("hidden_size", 768),
				"num_attention_heads": getInt("num_attention_heads", 12),
				"num_key_value_heads": getInt("num_key_value_heads", getInt("num_attention_heads", 12)),
				"num_hidden_layers":   getInt("num_hidden_layers", 12),
				"inputs": map[string]string{
					"attention_mask":  "attention_mask",
					"input_ids":       "input_ids",
					"past_key_names":  "past_key_values.%d.key",
					"past_value_names": "past_key_values.%d.value",
				},
				"outputs": map[string]string{
					"logits":              "logits",
					"present_key_names":   "present.%d.key",
					"present_value_names": "present.%d.value",
				},
			},
		},
		"search": map[string]any{
			"diversity_penalty":        0.0,
			"do_sample":                true,
			"early_stopping":           true,
			"length_penalty":           1.0,
			"max_length":               2048,
			"min_length":               0,
			"no_repeat_ngram_size":     0,
			"num_beams":                1,
			"num_return_sequences":     1,
			"past_present_share_buffer": false,
			"repetition_penalty":       1.0,
			"top_k":                    50,
			"top_p":                    0.95,
		},
	}

	// Try to read generation_config.json for sampling parameters
	genConfigPath := filepath.Join(modelPath, "generation_config.json")
	if genConfigData, err := os.ReadFile(genConfigPath); err == nil {
		var genConfig map[string]any
		if err := json.Unmarshal(genConfigData, &genConfig); err == nil {
			search := genaiConfig["search"].(map[string]any)
			if v, ok := genConfig["do_sample"].(bool); ok {
				search["do_sample"] = v
			}
			if v, ok := genConfig["top_k"].(float64); ok {
				search["top_k"] = int(v)
			}
			if v, ok := genConfig["top_p"].(float64); ok {
				search["top_p"] = v
			}
			if v, ok := genConfig["temperature"].(float64); ok {
				search["temperature"] = v
			}
		}
	}

	// Write genai_config.json
	output, err := json.MarshalIndent(genaiConfig, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling genai_config: %w", err)
	}

	if err := os.WriteFile(genaiConfigPath, output, 0644); err != nil {
		return fmt.Errorf("writing genai_config.json: %w", err)
	}

	logger.Info("Generated genai_config.json from HuggingFace config",
		zap.String("path", genaiConfigPath),
		zap.String("modelType", modelType))

	return nil
}


// isValidGeneratorModel checks if a directory contains a valid generator model.
// Returns true if the directory has either:
// - genai_config.json (ONNX Runtime GenAI format)
// - config.json with model.onnx (standard HuggingFace ONNX format)
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
// Directory structure: modelsDir/model_name/
// Supports both:
// - ONNX Runtime GenAI format (genai_config.json)
// - Standard HuggingFace ONNX format (config.json + model.onnx)
// If sharedSession is provided, all models will share the same Hugot session (required for ONNX Runtime)
func NewGeneratorRegistry(modelsDir string, sharedSession *khugot.Session, logger *zap.Logger) (*GeneratorRegistry, error) {
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
		return nil, fmt.Errorf("reading models directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		modelName := entry.Name()
		modelPath := filepath.Join(modelsDir, modelName)

		// Check if this is a valid generator model
		if !isValidGeneratorModel(modelPath) {
			logger.Debug("Skipping directory without valid generator model files",
				zap.String("dir", modelName))
			continue
		}

		logger.Info("Discovered generator model directory",
			zap.String("name", modelName),
			zap.String("path", modelPath))

		// Auto-generate genai_config.json from HuggingFace config if needed
		if err := generateGenaiConfig(modelPath, logger); err != nil {
			logger.Warn("Failed to generate genai_config.json",
				zap.String("name", modelName),
				zap.Error(err))
			continue
		}

		// Pool size for concurrent pipeline access
		// Cap at 4 to avoid excessive memory usage (each pipeline loads full model)
		poolSize := min(runtime.NumCPU(), 4)

		// Create pooled generator
		model, err := generation.NewPooledHugotGeneratorWithSession(modelPath, poolSize, sharedSession, logger.Named(modelName))
		if err != nil {
			logger.Warn("Failed to load generator model",
				zap.String("name", modelName),
				zap.Error(err))
		} else {
			registry.models[modelName] = model
			logger.Info("Successfully loaded generator model",
				zap.String("name", modelName),
				zap.Int("poolSize", poolSize))
		}
	}

	logger.Info("Generator registry initialized",
		zap.Int("models_loaded", len(registry.models)))

	return registry, nil
}

// Get returns a generator by model name
func (r *GeneratorRegistry) Get(modelName string) (generation.Generator, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	model, ok := r.models[modelName]
	if !ok {
		return nil, fmt.Errorf("generator model not found: %s", modelName)
	}
	return model, nil
}

// List returns all available model names
func (r *GeneratorRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.models))
	for name := range r.models {
		names = append(names, name)
	}
	return names
}

// Close closes all loaded models
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
