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

	"go.uber.org/zap"
)

// zapLogf adapts a zap.Logger to the logf signature used by modelregistry.DiscoverModelsInDir.
func zapLogf(logger *zap.Logger) func(string, ...any) {
	return func(msg string, args ...any) {
		logger.Debug(fmt.Sprintf(msg, args...))
	}
}

// generateGenaiConfig creates a genai_config.json file from a HuggingFace config.json.
// It also ensures chat_template.jinja exists for chat-based generation.
// This enables ONNX Runtime GenAI to load standard HuggingFace ONNX models.
// Returns nil if successful, error otherwise.
func generateGenaiConfig(modelPath string, logger *zap.Logger) error {
	genaiConfigPath := filepath.Join(modelPath, "genai_config.json")
	chatTemplateJinjaPath := filepath.Join(modelPath, "chat_template.jinja")

	// Ensure chat_template.jinja exists (ortgenai loads it from this file)
	if _, err := os.Stat(chatTemplateJinjaPath); os.IsNotExist(err) {
		// Try to create it from tokenizer_config.json
		chatTemplate := getChatTemplateFromTokenizer(modelPath, logger)
		if chatTemplate != "" {
			if err := os.WriteFile(chatTemplateJinjaPath, []byte(chatTemplate), 0644); err != nil {
				logger.Warn("Failed to write chat_template.jinja",
					zap.String("path", chatTemplateJinjaPath),
					zap.Error(err))
			} else {
				logger.Info("Created chat_template.jinja from tokenizer_config.json",
					zap.String("path", chatTemplateJinjaPath))
			}
		}
	}

	// Skip genai_config.json generation if it already exists
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

	// Build genai_config.json (chat_template is loaded from chat_template.jinja file, not from JSON)
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

// getChatTemplateFromTokenizer tries to get a chat template from tokenizer_config.json.
// Returns empty string if no template is found.
func getChatTemplateFromTokenizer(modelPath string, logger *zap.Logger) string {
	tokenizerConfigPath := filepath.Join(modelPath, "tokenizer_config.json")
	tokenizerData, err := os.ReadFile(tokenizerConfigPath)
	if err != nil {
		logger.Debug("No tokenizer_config.json found",
			zap.String("modelPath", modelPath))
		return ""
	}

	var tokenizerConfig map[string]any
	if err := json.Unmarshal(tokenizerData, &tokenizerConfig); err != nil {
		logger.Debug("Failed to parse tokenizer_config.json",
			zap.String("modelPath", modelPath),
			zap.Error(err))
		return ""
	}

	if ct, ok := tokenizerConfig["chat_template"].(string); ok && ct != "" {
		logger.Debug("Found chat_template in tokenizer_config.json",
			zap.String("modelPath", modelPath))
		return ct
	}

	logger.Debug("No chat_template in tokenizer_config.json",
		zap.String("modelPath", modelPath))
	return ""
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
