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
	"strings"

	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
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
//
// Structure: modelsDir/owner/model-name/model.onnx
func discoverModelsInDir(modelsDir string, modelType modelregistry.ModelType, logger *zap.Logger) ([]DiscoveredModel, error) {
	if modelsDir == "" {
		return nil, nil
	}

	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		return nil, nil
	}

	var discovered []DiscoveredModel

	// First level: owner directories
	ownerEntries, err := os.ReadDir(modelsDir)
	if err != nil {
		return nil, fmt.Errorf("reading models directory: %w", err)
	}

	for _, ownerEntry := range ownerEntries {
		if !ownerEntry.IsDir() {
			continue
		}

		owner := ownerEntry.Name()
		ownerPath := filepath.Join(modelsDir, owner)

		// Second level: model directories within owner
		modelEntries, err := os.ReadDir(ownerPath)
		if err != nil {
			logger.Debug("Failed to read owner directory", zap.String("path", ownerPath), zap.Error(err))
			continue
		}

		for _, modelEntry := range modelEntries {
			if !modelEntry.IsDir() {
				continue
			}
			modelPath := filepath.Join(ownerPath, modelEntry.Name())
			model := discoverSingleModel(modelPath, owner, modelEntry.Name(), modelType, logger)
			if model != nil {
				discovered = append(discovered, *model)
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
			"diversity_penalty":         0.0,
			"do_sample":                 false,
			"early_stopping":            true,
			"length_penalty":            1.0,
			"max_length":                2048,
			"min_length":                0,
			"no_repeat_ngram_size":      0,
			"num_beams":                 1,
			"num_return_sequences":      1,
			"past_present_share_buffer": false,
			"repetition_penalty":        1.0,
			"temperature":               1.0,
			"top_k":                     1,
			"top_p":                     1.0,
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
