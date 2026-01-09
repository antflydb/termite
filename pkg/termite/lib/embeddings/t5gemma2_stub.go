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

package embeddings

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"

	"github.com/antflydb/antfly-go/libaf/ai"
	libafembed "github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	"go.uber.org/zap"
)

// T5Gemma2Embedder is a stub when built without ONNX support.
// To enable T5Gemma-2 multimodal embeddings, build with: CGO_ENABLED=1 go build -tags="onnx,ORT"
type T5Gemma2Embedder struct{}

// NewT5Gemma2Embedder returns an error when T5Gemma-2 support is disabled.
func NewT5Gemma2Embedder(modelPath string, logger *zap.Logger) (*T5Gemma2Embedder, error) {
	return nil, errors.New("T5Gemma-2 embedder not available: build with -tags=\"onnx,ORT\" to enable")
}

// NewT5Gemma2EmbedderWithSession returns an error when T5Gemma-2 support is disabled.
func NewT5Gemma2EmbedderWithSession(modelPath string, sharedSession interface{}, logger *zap.Logger) (*T5Gemma2Embedder, error) {
	return nil, errors.New("T5Gemma-2 embedder not available: build with -tags=\"onnx,ORT\" to enable")
}

// NewT5Gemma2EmbedderWithSessionManager returns an error when T5Gemma-2 support is disabled.
func NewT5Gemma2EmbedderWithSessionManager(modelPath string, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*T5Gemma2Embedder, hugot.BackendType, error) {
	return nil, "", errors.New("T5Gemma-2 embedder not available: build with -tags=\"onnx,ORT\" to enable")
}

// Capabilities returns empty capabilities for the stub.
func (t *T5Gemma2Embedder) Capabilities() libafembed.EmbedderCapabilities {
	return libafembed.EmbedderCapabilities{}
}

// Embed returns an error for the stub since it cannot be used.
func (t *T5Gemma2Embedder) Embed(ctx context.Context, contents [][]ai.ContentPart) ([][]float32, error) {
	return nil, errors.New("T5Gemma-2 embedder not available: build with -tags=\"onnx,ORT\" to enable")
}

// Close is a no-op for the stub.
func (t *T5Gemma2Embedder) Close() error {
	return nil
}

// IsT5Gemma2Model checks if a model directory contains T5Gemma-2 model files
func IsT5Gemma2Model(modelPath string) bool {
	encoderPath := filepath.Join(modelPath, "encoder.onnx")
	visionPath := filepath.Join(modelPath, "vision_encoder.onnx")

	// Must have both encoder and vision encoder
	if _, err := os.Stat(encoderPath); err != nil {
		return false
	}
	if _, err := os.Stat(visionPath); err != nil {
		return false
	}

	// Check config for model_type
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
