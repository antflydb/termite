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

package e2e

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/backends"
	embeddingsLib "github.com/antflydb/termite/pkg/termite/lib/embeddings"
	_ "github.com/gomlx/gomlx/backends/simplego/highway"
	"go.uber.org/zap"
)

// TestCompareBackendEmbeddings compares embedding generation timing between
// CoreML and Go backends.
func TestCompareBackendEmbeddings(t *testing.T) {
	// Ensure an embedder model is downloaded
	// modelPath := ensureRegistryModel(t, "BAAI/bge-small-en-v1.5", ModelTypeEmbedder)
	modelPath := ensureHuggingFaceModel(t, "Snowflake/snowflake-arctic-embed-l-v2.0", "Snowflake/snowflake-arctic-embed-l-v2.0", ModelTypeEmbedder)

	t.Logf("Using model: %s", modelPath)

	// Test texts
	texts := []string{
		"This is a test sentence for embedding generation.",
		"Machine learning models can generate dense vector representations.",
		"Semantic search uses embeddings to find similar documents.",
		"The quick brown fox jumps over the lazy dog.",
		"Artificial intelligence is transforming many industries.",
	}

	ctx := context.Background()
	logger := zap.NewNop()

	// Get available backends
	availableBackends := backends.ListAvailable()
	t.Logf("Available backends: %d", len(availableBackends))
	for _, b := range availableBackends {
		t.Logf("  - %s (%s)", b.Name(), b.Type())
	}

	results := make(map[backends.BackendType]time.Duration)

	for _, backend := range availableBackends {
		backendType := backend.Type()
		t.Run(string(backendType), func(t *testing.T) {
			// Create session manager with specific backend priority
			sessionManager := backends.NewSessionManager()
			sessionManager.SetPriority([]backends.BackendSpec{
				{Backend: backendType, Device: backends.DeviceAuto},
			})

			cfg := embeddingsLib.PooledEmbedderConfig{
				ModelPath: modelPath,
				PoolSize:  1,
				Normalize: true,
				Logger:    logger,
			}

			embedder, usedBackend, err := embeddingsLib.NewPooledEmbedder(cfg, sessionManager)
			if err != nil {
				t.Skipf("Cannot create embedder with %s backend: %v", backendType, err)
			}
			defer embedder.Close()

			t.Logf("Created embedder with backend: %s", usedBackend)

			// Warmup
			_, err = embeddings.EmbedText(ctx, embedder, texts[:1])
			if err != nil {
				t.Fatalf("Warmup failed: %v", err)
			}

			// Time embedding generation with per-iteration breakdown
			const iterations = 10
			iterTimes := make([]time.Duration, iterations)
			totalStart := time.Now()
			for i := 0; i < iterations; i++ {
				iterStart := time.Now()
				_, err = embeddings.EmbedText(ctx, embedder, texts)
				if err != nil {
					t.Fatalf("Embed failed: %v", err)
				}
				iterTimes[i] = time.Since(iterStart)
			}
			elapsed := time.Since(totalStart)
			avgPerBatch := elapsed / iterations

			results[backendType] = avgPerBatch
			t.Logf("Backend %s: %v avg per batch (%d texts)", backendType, avgPerBatch, len(texts))
			for i, d := range iterTimes {
				t.Logf("  iter %d: %v", i, d)
			}
		})
	}

	// Print comparison summary
	t.Log("\n=== Backend Comparison Summary ===")
	for backend, duration := range results {
		t.Logf("  %s: %v per batch", backend, duration)
	}

	// Calculate speedup if we have both
	if goTime, ok := results[backends.BackendGo]; ok {
		if coremlTime, ok := results[backends.BackendCoreML]; ok {
			speedup := float64(goTime) / float64(coremlTime)
			t.Logf("\nCoreML speedup vs Go: %.2fx", speedup)
		}
	}
}

// BenchmarkBackendComparison runs a proper benchmark comparing backends.
func BenchmarkBackendComparison(b *testing.B) {
	// Look for a model already downloaded by TestCompareBackendEmbeddings
	modelPath := ""
	patterns := []string{
		filepath.Join(testModelsDir, "embedders", "*", "*", "model.onnx"),
		os.ExpandEnv("$HOME/.termite/models/embedders/*/*/model.onnx"),
	}
	for _, pattern := range patterns {
		matches, _ := filepath.Glob(pattern)
		if len(matches) > 0 {
			modelPath = filepath.Dir(matches[0])
			break
		}
	}
	if path := os.Getenv("EMBEDDING_MODEL_PATH"); path != "" {
		modelPath = path
	}
	if modelPath == "" {
		b.Skip("No embedder model found — run TestCompareBackendEmbeddings first to download one")
	}

	texts := []string{
		"This is a test sentence for embedding generation.",
		"Machine learning models can generate dense vector representations.",
		"Semantic search uses embeddings to find similar documents.",
		"The quick brown fox jumps over the lazy dog.",
		"Artificial intelligence is transforming many industries.",
	}

	ctx := context.Background()
	logger := zap.NewNop()

	for _, backend := range backends.ListAvailable() {
		backendType := backend.Type()
		b.Run(string(backendType), func(b *testing.B) {
			sessionManager := backends.NewSessionManager()
			sessionManager.SetPriority([]backends.BackendSpec{
				{Backend: backendType, Device: backends.DeviceAuto},
			})

			cfg := embeddingsLib.PooledEmbedderConfig{
				ModelPath: modelPath,
				PoolSize:  1,
				Normalize: true,
				Logger:    logger,
			}

			embedder, _, err := embeddingsLib.NewPooledEmbedder(cfg, sessionManager)
			if err != nil {
				b.Skipf("Cannot create embedder: %v", err)
			}
			defer embedder.Close()

			// Warmup
			_, _ = embeddings.EmbedText(ctx, embedder, texts[:1])

			b.ResetTimer()
			for b.Loop() {
				_, err = embeddings.EmbedText(ctx, embedder, texts)
				if err != nil {
					b.Fatalf("Embed failed: %v", err)
				}
			}

			docsPerSec := float64(b.N*len(texts)) / b.Elapsed().Seconds()
			b.ReportMetric(docsPerSec, "docs/sec")
		})
	}
}
