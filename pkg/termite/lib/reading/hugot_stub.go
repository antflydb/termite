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

package reading

import (
	"context"
	"errors"
	"image"

	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"go.uber.org/zap"
)

// PooledHugotReader is a stub when built without ONNX support.
// To enable Vision2Seq reader models (TrOCR, Donut, Florence-2),
// build with: CGO_ENABLED=1 go build -tags="onnx,ORT"
type PooledHugotReader struct{}

// NewPooledHugotReader returns an error when Vision2Seq support is disabled.
func NewPooledHugotReader(modelPath string, poolSize int, logger *zap.Logger) (*PooledHugotReader, error) {
	return nil, errors.New("PooledHugotReader not available: build with -tags=\"onnx,ORT\" to enable Vision2Seq models")
}

// NewPooledHugotReaderWithSession returns an error when Vision2Seq support is disabled.
func NewPooledHugotReaderWithSession(modelPath string, poolSize int, sharedSession *khugot.Session, logger *zap.Logger) (*PooledHugotReader, error) {
	return nil, errors.New("PooledHugotReader not available: build with -tags=\"onnx,ORT\" to enable Vision2Seq models")
}

// NewPooledHugotReaderWithSessionManager returns an error when Vision2Seq support is disabled.
func NewPooledHugotReaderWithSessionManager(modelPath string, poolSize int, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*PooledHugotReader, hugot.BackendType, error) {
	return nil, "", errors.New("PooledHugotReader not available: build with -tags=\"onnx,ORT\" to enable Vision2Seq models")
}

// Read returns an error for the stub since it cannot be used.
func (r *PooledHugotReader) Read(ctx context.Context, images []image.Image, prompt string, maxTokens int) ([]Result, error) {
	return nil, errors.New("PooledHugotReader not available: build with -tags=\"onnx,ORT\" to enable Vision2Seq models")
}

// Close is a no-op for the stub.
func (r *PooledHugotReader) Close() error {
	return nil
}

// ModelType returns an empty model type for the stub.
func (r *PooledHugotReader) ModelType() ModelType {
	return ModelTypeGeneric
}
