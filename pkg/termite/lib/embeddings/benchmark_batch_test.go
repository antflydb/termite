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
	"testing"
	"time"

	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

// BenchmarkCoreMLvsONNXBatching compares:
// - CoreML with 100 individual requests (batch size 1)
// - Pure ONNX CPU with a single batch of 100
//
// Run with: go test -tags="onnx,ORT" -run=BenchmarkCoreMLvsONNXBatching -v ./pkg/termite/lib/embeddings/
func TestBenchmarkCoreMLvsONNXBatching(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Model not found, skipping benchmark")
	}

	libPath := getOnnxLibPath()
	if libPath == "" {
		t.Skip("Could not find ONNX Runtime library")
	}

	const numTexts = 100

	// Generate test texts
	texts := make([]string, numTexts)
	for i := 0; i < numTexts; i++ {
		texts[i] = fmt.Sprintf("This is test sentence number %d for benchmarking embedding generation performance.", i+1)
	}

	// Benchmark CoreML with individual requests
	t.Run("CoreML_100_Individual", func(t *testing.T) {
		session, err := khugot.NewORTSession(
			options.WithOnnxLibraryPath(libPath),
			options.WithCoreML(map[string]string{"MLComputeUnits": "ALL"}),
		)
		if err != nil {
			t.Fatalf("Failed to create CoreML session: %v", err)
		}
		defer session.Destroy()

		pipeline, err := khugot.NewPipeline(session, khugot.FeatureExtractionConfig{
			ModelPath:    modelPath,
			Name:         "benchmark-coreml",
			OnnxFilename: "model.onnx",
			Options: []backends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
				pipelines.WithNormalization(),
			},
		})
		if err != nil {
			t.Fatalf("Failed to create pipeline: %v", err)
		}

		// Warmup
		for i := 0; i < 5; i++ {
			_, _ = pipeline.RunPipeline([]string{texts[0]})
		}

		// Benchmark: 100 individual requests
		start := time.Now()
		for i := 0; i < numTexts; i++ {
			output, err := pipeline.RunPipeline([]string{texts[i]})
			if err != nil {
				t.Fatalf("CoreML inference failed at %d: %v", i, err)
			}
			if len(output.Embeddings) != 1 {
				t.Fatalf("Expected 1 embedding, got %d", len(output.Embeddings))
			}
		}
		elapsed := time.Since(start)

		t.Logf("CoreML (100 individual requests): %v total, %v per request", elapsed, elapsed/numTexts)
	})

	// Benchmark Pure ONNX CPU with single batch
	t.Run("PureONNX_Batch100", func(t *testing.T) {
		session, err := khugot.NewORTSession(
			options.WithOnnxLibraryPath(libPath),
			// No CoreML - pure ONNX Runtime CPU
		)
		if err != nil {
			t.Fatalf("Failed to create ONNX session: %v", err)
		}
		defer session.Destroy()

		pipeline, err := khugot.NewPipeline(session, khugot.FeatureExtractionConfig{
			ModelPath:    modelPath,
			Name:         "benchmark-onnx",
			OnnxFilename: "model.onnx",
			Options: []backends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
				pipelines.WithNormalization(),
			},
		})
		if err != nil {
			t.Fatalf("Failed to create pipeline: %v", err)
		}

		// Warmup
		for i := 0; i < 5; i++ {
			_, _ = pipeline.RunPipeline([]string{texts[0]})
		}

		// Benchmark: single batch of 100
		start := time.Now()
		output, err := pipeline.RunPipeline(texts)
		if err != nil {
			t.Fatalf("ONNX batch inference failed: %v", err)
		}
		if len(output.Embeddings) != numTexts {
			t.Fatalf("Expected %d embeddings, got %d", numTexts, len(output.Embeddings))
		}
		elapsed := time.Since(start)

		t.Logf("Pure ONNX CPU (batch of 100): %v total, %v per text", elapsed, elapsed/numTexts)
	})

	// Also test ONNX with smaller batches for comparison
	for _, batchSize := range []int{10, 25, 50} {
		t.Run(fmt.Sprintf("PureONNX_Batches_%d", batchSize), func(t *testing.T) {
			session, err := khugot.NewORTSession(
				options.WithOnnxLibraryPath(libPath),
			)
			if err != nil {
				t.Fatalf("Failed to create ONNX session: %v", err)
			}
			defer session.Destroy()

			pipeline, err := khugot.NewPipeline(session, khugot.FeatureExtractionConfig{
				ModelPath:    modelPath,
				Name:         fmt.Sprintf("benchmark-onnx-%d", batchSize),
				OnnxFilename: "model.onnx",
				Options: []backends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
					pipelines.WithNormalization(),
				},
			})
			if err != nil {
				t.Fatalf("Failed to create pipeline: %v", err)
			}

			// Warmup
			_, _ = pipeline.RunPipeline([]string{texts[0]})

			// Benchmark: multiple batches
			start := time.Now()
			totalEmbeddings := 0
			for i := 0; i < numTexts; i += batchSize {
				end := i + batchSize
				if end > numTexts {
					end = numTexts
				}
				batch := texts[i:end]
				output, err := pipeline.RunPipeline(batch)
				if err != nil {
					t.Fatalf("ONNX batch inference failed: %v", err)
				}
				totalEmbeddings += len(output.Embeddings)
			}
			elapsed := time.Since(start)

			t.Logf("Pure ONNX CPU (batches of %d): %v total, %v per text", batchSize, elapsed, elapsed/numTexts)
		})
	}
}

// TestBenchmarkSingleRequest compares single-request latency between CoreML and ONNX CPU
func TestBenchmarkSingleRequest(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Model not found, skipping benchmark")
	}

	libPath := getOnnxLibPath()
	if libPath == "" {
		t.Skip("Could not find ONNX Runtime library")
	}

	text := "This is a single test sentence for measuring latency."
	const iterations = 100

	t.Run("CoreML_Single", func(t *testing.T) {
		session, err := khugot.NewORTSession(
			options.WithOnnxLibraryPath(libPath),
			options.WithCoreML(map[string]string{"MLComputeUnits": "ALL"}),
		)
		if err != nil {
			t.Fatalf("Failed to create CoreML session: %v", err)
		}
		defer session.Destroy()

		pipeline, err := khugot.NewPipeline(session, khugot.FeatureExtractionConfig{
			ModelPath:    modelPath,
			Name:         "latency-coreml",
			OnnxFilename: "model.onnx",
			Options: []backends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
				pipelines.WithNormalization(),
			},
		})
		if err != nil {
			t.Fatalf("Failed to create pipeline: %v", err)
		}

		// Warmup
		for i := 0; i < 10; i++ {
			_, _ = pipeline.RunPipeline([]string{text})
		}

		// Measure
		start := time.Now()
		for i := 0; i < iterations; i++ {
			_, err := pipeline.RunPipeline([]string{text})
			if err != nil {
				t.Fatalf("Failed: %v", err)
			}
		}
		elapsed := time.Since(start)

		t.Logf("CoreML single request: %v avg latency (%d iterations)", elapsed/iterations, iterations)
	})

	t.Run("PureONNX_Single", func(t *testing.T) {
		session, err := khugot.NewORTSession(
			options.WithOnnxLibraryPath(libPath),
		)
		if err != nil {
			t.Fatalf("Failed to create ONNX session: %v", err)
		}
		defer session.Destroy()

		pipeline, err := khugot.NewPipeline(session, khugot.FeatureExtractionConfig{
			ModelPath:    modelPath,
			Name:         "latency-onnx",
			OnnxFilename: "model.onnx",
			Options: []backends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
				pipelines.WithNormalization(),
			},
		})
		if err != nil {
			t.Fatalf("Failed to create pipeline: %v", err)
		}

		// Warmup
		for i := 0; i < 10; i++ {
			_, _ = pipeline.RunPipeline([]string{text})
		}

		// Measure
		start := time.Now()
		for i := 0; i < iterations; i++ {
			_, err := pipeline.RunPipeline([]string{text})
			if err != nil {
				t.Fatalf("Failed: %v", err)
			}
		}
		elapsed := time.Since(start)

		t.Logf("Pure ONNX CPU single request: %v avg latency (%d iterations)", elapsed/iterations, iterations)
	})
}
