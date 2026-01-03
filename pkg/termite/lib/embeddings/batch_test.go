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

package embeddings

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

// GPUModeSetter is an interface for backends that support GPU mode configuration
type GPUModeSetter interface {
	SetGPUMode(mode hugot.GPUMode)
}

// TestBatchSizeWithDifferentBackends documents and validates the CoreML EP batch limitation.
//
// This test demonstrates that the ONNX Runtime CoreML Execution Provider cannot handle
// batch sizes > 1 for embedding models, while pure ONNX Runtime CPU handles batching fine.
//
// Expected results:
//   - CoreML EP (ALL): Only batch_1 passes, all larger batches fail with error code -1
//   - CoreML EP (CPUAndGPU): Testing if GPU-only mode handles batching
//   - Pure ONNX Runtime CPU: All batch sizes pass
//
// This test directly calls the hugot pipeline to bypass the batching fix in Embed().
func TestBatchSizeWithDifferentBackends(t *testing.T) {
	// Find model path
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Model not found, skipping batch test")
	}

	batchSizes := []int{1, 2, 4, 8, 16}

	// Test with CoreML EP using ALL compute units (Neural Engine > GPU > CPU)
	// Expected: Only batch_1 works, others fail
	t.Run("CoreML_ALL", func(t *testing.T) {
		testBatchSizesWithConfig(t, modelPath, batchSizes, "ALL", true)
	})

	// Test with CoreML EP using CPUAndGPU (skip Neural Engine)
	// Testing: Does GPU-only mode handle batching?
	t.Run("CoreML_CPUAndGPU", func(t *testing.T) {
		testBatchSizesWithConfig(t, modelPath, batchSizes, "CPUAndGPU", false) // false = we don't know yet
	})

	// Test with CoreML EP using CPUOnly
	// Testing: Does CPU-via-CoreML handle batching?
	t.Run("CoreML_CPUOnly", func(t *testing.T) {
		testBatchSizesWithConfig(t, modelPath, batchSizes, "CPUOnly", false)
	})

	// Test WITHOUT CoreML - pure ONNX Runtime CPU
	// Expected: All batch sizes work
	t.Run("PureONNX_CPU", func(t *testing.T) {
		testBatchSizesWithConfig(t, modelPath, batchSizes, "none", false)
	})
}

// testBatchSizesWithConfig tests batch sizes with different CoreML configurations.
// mlComputeUnits can be: "ALL", "CPUAndGPU", "CPUOnly", "CPUAndNeuralEngine", or "none" (pure ONNX)
func testBatchSizesWithConfig(t *testing.T, modelPath string, batchSizes []int, mlComputeUnits string, expectBatchFailure bool) {
	var session *khugot.Session
	var err error

	libPath := getOnnxLibPath()
	if libPath == "" {
		t.Skip("Could not find ONNX Runtime library")
	}

	if mlComputeUnits == "none" {
		// Pure ONNX Runtime without CoreML
		t.Logf("Using pure ONNX Runtime CPU (no CoreML)")
		session, err = khugot.NewORTSession(options.WithOnnxLibraryPath(libPath))
	} else {
		// CoreML with specific compute units
		t.Logf("Using CoreML with MLComputeUnits=%s", mlComputeUnits)
		coremlConfig := map[string]string{
			"MLComputeUnits": mlComputeUnits,
		}
		session, err = khugot.NewORTSession(
			options.WithOnnxLibraryPath(libPath),
			options.WithCoreML(coremlConfig),
		)
	}

	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	defer session.Destroy()

	// Create pipeline
	pipelineConfig := khugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         fmt.Sprintf("batch-test-%s", mlComputeUnits),
		OnnxFilename: "model.onnx",
		Options: []backends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
			pipelines.WithNormalization(),
		},
	}

	pipeline, err := khugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		t.Fatalf("Failed to create pipeline: %v", err)
	}

	for _, batchSize := range batchSizes {
		t.Run(fmt.Sprintf("batch_%d", batchSize), func(t *testing.T) {
			texts := make([]string, batchSize)
			for i := 0; i < batchSize; i++ {
				texts[i] = fmt.Sprintf("This is test sentence number %d for batch testing.", i+1)
			}

			output, err := pipeline.RunPipeline(texts)

			expectFailure := expectBatchFailure && batchSize > 1

			if err != nil {
				if expectFailure {
					t.Logf("EXPECTED FAILURE: %v", err)
				} else {
					t.Logf("FAILURE: %v", err)
				}
			} else {
				t.Logf("SUCCESS: generated %d embeddings, dim=%d", len(output.Embeddings), len(output.Embeddings[0]))
			}
		})
	}
}

func getOnnxLibPath() string {
	// Check ONNXRUNTIME_ROOT first
	if root := os.Getenv("ONNXRUNTIME_ROOT"); root != "" {
		platform := runtime.GOOS + "-" + runtime.GOARCH
		libName := "libonnxruntime.dylib"
		if runtime.GOOS == "linux" {
			libName = "libonnxruntime.so"
		}

		// Try platform-specific path
		platformPath := filepath.Join(root, platform, "lib", libName)
		if _, err := os.Stat(platformPath); err == nil {
			return filepath.Dir(platformPath)
		}

		// Try direct lib path
		directPath := filepath.Join(root, "lib", libName)
		if _, err := os.Stat(directPath); err == nil {
			return filepath.Dir(directPath)
		}
	}

	// Check DYLD_LIBRARY_PATH on macOS
	if dyldPath := os.Getenv("DYLD_LIBRARY_PATH"); dyldPath != "" {
		for _, dir := range filepath.SplitList(dyldPath) {
			libPath := filepath.Join(dir, "libonnxruntime.dylib")
			if _, err := os.Stat(libPath); err == nil {
				return dir
			}
		}
	}

	return ""
}

func findModelPath(t *testing.T) string {
	t.Helper()

	// Check common locations
	paths := []string{
		filepath.Join(os.Getenv("HOME"), ".cache/termite/models/BAAI--bge-small-en-v1.5"),
		"./models/BAAI--bge-small-en-v1.5",
		"../../../../../models/embedders/BAAI/bge-small-en-v1.5",
		"../../../../../../models/embedders/BAAI/bge-small-en-v1.5",
	}

	for _, p := range paths {
		modelFile := filepath.Join(p, "model.onnx")
		if _, err := os.Stat(modelFile); err == nil {
			t.Logf("Found model at: %s", p)
			return p
		}
	}

	// Try to find it
	home := os.Getenv("HOME")
	cacheDir := filepath.Join(home, ".cache/termite/models")
	entries, err := os.ReadDir(cacheDir)
	if err == nil {
		for _, e := range entries {
			if e.IsDir() && (e.Name() == "BAAI--bge-small-en-v1.5" || e.Name() == "bge-small-en-v1.5") {
				return filepath.Join(cacheDir, e.Name())
			}
		}
	}

	return ""
}
