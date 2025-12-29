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

	"github.com/antflydb/antfly-go/libaf/reranking"
	termreranking "github.com/antflydb/termite/pkg/termite/lib/reranking"
	"go.uber.org/zap"
)

// TermiteReranker wraps a reranking Model for lifecycle management in Termite service
type TermiteReranker struct {
	model  reranking.Model
	logger *zap.Logger
}

// NewTermiteReranker creates a new Termite reranker with the provided model.
// If model is nil, returns nil (reranking disabled).
func NewTermiteReranker(model reranking.Model, logger *zap.Logger) (*TermiteReranker, error) {
	if model == nil {
		logger.Info("Reranking disabled: no model provided")
		return nil, nil
	}

	logger.Info("Termite reranker initialized successfully")

	return &TermiteReranker{
		model:  model,
		logger: logger,
	}, nil
}

// NewHugotModel creates a Hugot ONNX reranking model.
// If modelPath is empty, returns nil (reranking disabled).
// This is a convenience function for the common case of using Hugot.
func NewHugotModel(modelPath string, logger *zap.Logger) (reranking.Model, error) {
	if modelPath == "" {
		logger.Info("Reranking disabled: reranker_models_dir not set in config")
		return nil, nil
	}

	// Check if model path exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("reranker model path does not exist: %s", modelPath)
	}

	logger.Info("Initializing Hugot reranker",
		zap.String("modelPath", modelPath),
	)

	// Prefer quantized model if available, otherwise use standard model
	onnxFilename := "model.onnx"
	quantizedPath := modelPath + "/model_i8.onnx"
	if _, err := os.Stat(quantizedPath); err == nil {
		onnxFilename = "model_i8.onnx"
		logger.Info("Found quantized model, will use it")
	}

	// Create Hugot reranker (using pooled variant with poolSize=1)
	hugot, err := termreranking.NewPooledHugotReranker(modelPath, onnxFilename, 1, logger)
	if err != nil {
		return nil, fmt.Errorf("creating hugot reranker: %w", err)
	}

	logger.Info("Hugot reranker model loaded successfully")
	return hugot, nil
}

// GetModel returns the underlying reranking model
func (tr *TermiteReranker) GetModel() reranking.Model {
	return tr.model
}

// Close releases reranker resources
func (tr *TermiteReranker) Close() error {
	if tr.model != nil {
		tr.logger.Info("Closing reranker model")
		return tr.model.Close()
	}
	return nil
}
