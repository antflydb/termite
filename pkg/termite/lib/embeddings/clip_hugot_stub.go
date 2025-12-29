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
	"errors"

	"github.com/antflydb/antfly-go/libaf/ai"
	libafembed "github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"go.uber.org/zap"
)

// HugotCLIPEmbedder is a stub when built without ONNX support.
// To enable CLIP multimodal embeddings, build with: CGO_ENABLED=1 go build -tags="onnx,ORT"
type HugotCLIPEmbedder struct{}

// NewHugotCLIPEmbedder returns an error when CLIP support is disabled.
func NewHugotCLIPEmbedder(modelPath string, quantized bool, logger *zap.Logger) (*HugotCLIPEmbedder, error) {
	return nil, errors.New("HugotCLIP embedder not available: build with -tags=\"onnx,ORT\" to enable")
}

// NewHugotCLIPEmbedderWithSession returns an error when CLIP support is disabled.
func NewHugotCLIPEmbedderWithSession(modelPath string, quantized bool, sharedSession *khugot.Session, logger *zap.Logger) (*HugotCLIPEmbedder, error) {
	return nil, errors.New("HugotCLIP embedder not available: build with -tags=\"onnx,ORT\" to enable")
}

// NewHugotCLIPEmbedderWithSessionManager returns an error when CLIP support is disabled.
func NewHugotCLIPEmbedderWithSessionManager(modelPath string, quantized bool, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*HugotCLIPEmbedder, hugot.BackendType, error) {
	return nil, "", errors.New("HugotCLIP embedder not available: build with -tags=\"onnx,ORT\" to enable")
}

// Capabilities returns empty capabilities for the stub.
func (c *HugotCLIPEmbedder) Capabilities() libafembed.EmbedderCapabilities {
	return libafembed.EmbedderCapabilities{}
}

// Embed returns an error for the stub since it cannot be used.
func (c *HugotCLIPEmbedder) Embed(ctx context.Context, contents [][]ai.ContentPart) ([][]float32, error) {
	return nil, errors.New("HugotCLIP embedder not available: build with -tags=\"onnx,ORT\" to enable")
}

// Close is a no-op for the stub.
func (c *HugotCLIPEmbedder) Close() error {
	return nil
}
