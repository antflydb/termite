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

//go:build !(onnx && ORT)

package seq2seq

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"

	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	"go.uber.org/zap"
)

// T5Gemma2Generator is a stub when built without ONNX support.
// To enable T5Gemma-2 generation, build with: CGO_ENABLED=1 go build -tags="onnx,ORT"
type T5Gemma2Generator struct{}

// T5Gemma2GeneratorConfig holds configuration for T5Gemma-2 generation
type T5Gemma2GeneratorConfig struct {
	ModelID      string  `json:"model_id"`
	ModelType    string  `json:"model_type"`
	Task         string  `json:"task"`
	MaxNewTokens int     `json:"max_new_tokens"`
	NumBeams     int     `json:"num_beams"`
	DoSample     bool    `json:"do_sample"`
	Temperature  float32 `json:"temperature"`
	TopP         float32 `json:"top_p"`
	ImageSize    int     `json:"image_size"`
	HiddenSize   int     `json:"hidden_size"`
}

// MultimodalInput represents input that can contain both text and images
type MultimodalInput struct {
	Text   string
	Images [][]byte
}

// NewT5Gemma2Generator returns an error when ONNX Runtime is not available.
func NewT5Gemma2Generator(modelPath string, logger *zap.Logger) (*T5Gemma2Generator, error) {
	return nil, errors.New("T5Gemma-2 generator requires ONNX Runtime: build with -tags=\"onnx,ORT\"")
}

// NewT5Gemma2GeneratorWithSession returns an error when ONNX Runtime is not available.
func NewT5Gemma2GeneratorWithSession(modelPath string, sharedSession interface{}, logger *zap.Logger) (*T5Gemma2Generator, error) {
	return nil, errors.New("T5Gemma-2 generator requires ONNX Runtime: build with -tags=\"onnx,ORT\"")
}

// NewT5Gemma2GeneratorWithSessionManager returns an error when ONNX Runtime is not available.
func NewT5Gemma2GeneratorWithSessionManager(modelPath string, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*T5Gemma2Generator, hugot.BackendType, error) {
	return nil, "", errors.New("T5Gemma-2 generator requires ONNX Runtime: build with -tags=\"onnx,ORT\"")
}

// Generate returns an error for the stub.
func (g *T5Gemma2Generator) Generate(ctx context.Context, inputs []string) (*GeneratedOutput, error) {
	return nil, errors.New("T5Gemma-2 generator requires ONNX Runtime: build with -tags=\"onnx,ORT\"")
}

// GenerateMultimodal returns an error for the stub.
func (g *T5Gemma2Generator) GenerateMultimodal(ctx context.Context, inputs []MultimodalInput) (*GeneratedOutput, error) {
	return nil, errors.New("T5Gemma-2 generator requires ONNX Runtime: build with -tags=\"onnx,ORT\"")
}

// ProcessImage returns an error for the stub.
func (g *T5Gemma2Generator) ProcessImage(imageData []byte) ([]float32, error) {
	return nil, errors.New("T5Gemma-2 generator requires ONNX Runtime: build with -tags=\"onnx,ORT\"")
}

// HasVision returns false for the stub.
func (g *T5Gemma2Generator) HasVision() bool {
	return false
}

// Config returns an empty configuration for the stub.
func (g *T5Gemma2Generator) Config() T5Gemma2GeneratorConfig {
	return T5Gemma2GeneratorConfig{}
}

// Close is a no-op for the stub.
func (g *T5Gemma2Generator) Close() error {
	return nil
}

// IsT5Gemma2GeneratorModel checks if a model directory contains T5Gemma-2 generator files.
func IsT5Gemma2GeneratorModel(modelPath string) bool {
	// Must have standard seq2seq files
	requiredFiles := []string{"encoder.onnx", "decoder.onnx", "decoder-init.onnx"}
	for _, file := range requiredFiles {
		filePath := filepath.Join(modelPath, file)
		if _, err := os.Stat(filePath); err != nil {
			return false
		}
	}

	// Check config for model_type: "t5gemma2"
	configPath := filepath.Join(modelPath, "config.json")
	if data, err := os.ReadFile(configPath); err == nil {
		var config struct {
			ModelType string `json:"model_type"`
		}
		if err := json.Unmarshal(data, &config); err == nil {
			return config.ModelType == "t5gemma2"
		}
	}

	// Also check t5gemma2_config.json
	t5gemma2ConfigPath := filepath.Join(modelPath, "t5gemma2_config.json")
	if _, err := os.Stat(t5gemma2ConfigPath); err == nil {
		return true
	}

	return false
}

// DecodeBase64Image decodes a base64-encoded image (data URI or raw base64).
func DecodeBase64Image(input string) ([]byte, error) {
	if strings.HasPrefix(input, "data:") {
		parts := strings.SplitN(input, ",", 2)
		if len(parts) != 2 {
			return nil, errors.New("invalid data URI format")
		}
		input = parts[1]
	}

	return base64.StdEncoding.DecodeString(input)
}
