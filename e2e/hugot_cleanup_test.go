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

//go:build onnx && ORT

package e2e

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/antflydb/termite/pkg/termite/lib/embeddings"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// TestPooledHugotEmbedder_CleanupOnClose verifies that pipelines are properly
// removed from the session when Close() is called.
//
// This test demonstrates the bug where pipelines remain in the session after
// Close(), causing "pipeline has already been initialised" errors when a new
// embedder tries to create pipelines with the same names.
func TestPooledHugotEmbedder_CleanupOnClose(t *testing.T) {
	// Ensure model is available
	ensureHuggingFaceModel(t, "bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", ModelTypeEmbedder)

	modelsDir := getTestModelsDir()
	modelPath := filepath.Join(modelsDir, "embedders", "BAAI", "bge-small-en-v1.5")

	// Verify model exists
	onnxFile := filepath.Join(modelPath, "model.onnx")
	if _, err := os.Stat(onnxFile); os.IsNotExist(err) {
		t.Skipf("Model not found at %s, skipping test", modelPath)
	}

	// Get ONNX library path
	libPath := os.Getenv("ONNXRUNTIME_ROOT")
	if libPath == "" {
		t.Skip("ONNXRUNTIME_ROOT not set")
	}
	libPath = filepath.Join(libPath, "linux-amd64", "lib")

	logger := zap.NewNop()
	onnxFilename := "model.onnx"
	poolSize := 2

	// Create shared session
	session, err := khugot.NewORTSession(options.WithOnnxLibraryPath(libPath))
	require.NoError(t, err, "Failed to create session")
	defer session.Destroy()

	// Verify session starts empty
	initialStats := session.GetStatistics()
	require.Equal(t, 0, len(initialStats), "session should start with no pipelines")

	// Create embedder with shared session
	embedder, err := embeddings.NewPooledHugotEmbedderWithSession(
		modelPath, onnxFilename, poolSize, session, logger,
	)
	require.NoError(t, err, "Failed to create first embedder")

	// Verify pipelines were registered
	afterCreate := session.GetStatistics()
	require.Equal(t, poolSize, len(afterCreate), "should have %d pipelines after creation", poolSize)
	t.Logf("After creating embedder: %d pipelines registered", len(afterCreate))
	for name := range afterCreate {
		t.Logf("  - %s", name)
	}

	// Close embedder - this should remove pipelines from the session
	err = embedder.Close()
	require.NoError(t, err, "Failed to close embedder")

	// Verify pipelines were removed (THIS IS THE BUG - they won't be removed)
	afterClose := session.GetStatistics()
	t.Logf("After closing embedder: %d pipelines still registered", len(afterClose))
	for name := range afterClose {
		t.Logf("  - %s (LEAK!)", name)
	}
	require.Equal(t, 0, len(afterClose), "pipelines should be removed after Close()")

	// Create second embedder - should work if pipelines were properly cleaned up
	embedder2, err := embeddings.NewPooledHugotEmbedderWithSession(
		modelPath, onnxFilename, poolSize, session, logger,
	)
	require.NoError(t, err, "should be able to create new embedder with same names after Close()")
	defer embedder2.Close()

	// Verify second embedder's pipelines were registered
	afterSecondCreate := session.GetStatistics()
	require.Equal(t, poolSize, len(afterSecondCreate), "should have %d pipelines after second creation", poolSize)
	t.Logf("After creating second embedder: %d pipelines registered", len(afterSecondCreate))
}
