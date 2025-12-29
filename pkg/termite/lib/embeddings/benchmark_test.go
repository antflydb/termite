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

// Package embeddings provides benchmarks comparing Hugot (ONNX) embedding performance.
//
// # Usage
//
// Run all benchmarks:
//
//	HUGOT_MODEL_PATH=./models/bge-small-en-v1.5 go test -bench=. -benchmem
//
// Run Hugot benchmarks only:
//
//	HUGOT_MODEL_PATH=./models/bge-small-en-v1.5 go test -bench=Hugot -benchmem
//
// Run throughput analysis:
//
//	HUGOT_MODEL_PATH=./models/bge-small-en-v1.5 go test -bench=Throughput -benchmem
//
// # Environment Variables
//
//   - HUGOT_MODEL_PATH: Path to ONNX model directory for Hugot benchmarks
//
// # Benchmark Categories
//
//   - BenchmarkHugotEmbedder_*: Tests Hugot ONNX embedder with various text lengths
//   - BenchmarkLatency_*: Single-text embedding latency measurements
//   - BenchmarkThroughput_*: Batch throughput analysis across different sizes
//   - BenchmarkWarmup: Cold vs warm start performance analysis
package embeddings

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

const (
	// Benchmark configuration
	benchSmallBatchSize  = 10
	benchMediumBatchSize = 50
	benchLargeBatchSize  = 100

	// Sample texts of varying lengths
	benchShortText  = "Quick search query"
	benchMediumText = "This is a medium length document that contains several sentences with various topics to embed."
	benchLongText   = "This is a longer document that simulates a more realistic use case with multiple paragraphs. " +
		"It contains detailed information about various topics that need to be embedded for semantic search. " +
		"The embedding system should handle this efficiently while maintaining good quality representations. " +
		"Performance characteristics may vary based on the length and complexity of the input text."
)

// generateBenchmarkTexts creates a slice of test texts for benchmarking
func generateBenchmarkTexts(count int, textType string) []string {
	texts := make([]string, count)
	var baseText string

	switch textType {
	case "short":
		baseText = benchShortText
	case "medium":
		baseText = benchMediumText
	case "long":
		baseText = benchLongText
	default:
		baseText = benchMediumText
	}

	for i := range texts {
		texts[i] = fmt.Sprintf("%s [doc %d]", baseText, i)
	}
	return texts
}

// BenchmarkHugotEmbedder_ShortTexts benchmarks Hugot with short texts
func BenchmarkHugotEmbedder_ShortTexts(b *testing.B) {
	modelPath := os.Getenv("HUGOT_MODEL_PATH")
	if modelPath == "" {
		b.Skip("HUGOT_MODEL_PATH not set, skipping Hugot benchmark")
	}

	logger := zap.NewNop()
	embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", 1, logger)
	require.NoError(b, err)
	defer func() { _ = embedder.Close() }()

	texts := generateBenchmarkTexts(benchSmallBatchSize, "short")
	ctx := context.Background()

	b.ResetTimer()
	for b.Loop() {
		_, err := embeddings.EmbedText(ctx, embedder, texts)
		if err != nil {
			b.Fatalf("Embed failed: %v", err)
		}
	}
}

// BenchmarkHugotEmbedder_MediumTexts benchmarks Hugot with medium-length texts
func BenchmarkHugotEmbedder_MediumTexts(b *testing.B) {
	modelPath := os.Getenv("HUGOT_MODEL_PATH")
	if modelPath == "" {
		b.Skip("HUGOT_MODEL_PATH not set, skipping Hugot benchmark")
	}

	logger := zap.NewNop()
	embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", 1, logger)
	require.NoError(b, err)
	defer func() { _ = embedder.Close() }()

	texts := generateBenchmarkTexts(benchMediumBatchSize, "medium")
	ctx := context.Background()

	b.ResetTimer()
	for b.Loop() {
		_, err := embeddings.EmbedText(ctx, embedder, texts)
		if err != nil {
			b.Fatalf("Embed failed: %v", err)
		}
	}
}

// BenchmarkHugotEmbedder_LongTexts benchmarks Hugot with long texts
func BenchmarkHugotEmbedder_LongTexts(b *testing.B) {
	modelPath := os.Getenv("HUGOT_MODEL_PATH")
	if modelPath == "" {
		b.Skip("HUGOT_MODEL_PATH not set, skipping Hugot benchmark")
	}

	logger := zap.NewNop()
	embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", 1, logger)
	require.NoError(b, err)
	defer func() { _ = embedder.Close() }()

	texts := generateBenchmarkTexts(benchLargeBatchSize, "long")
	ctx := context.Background()

	b.ResetTimer()
	for b.Loop() {
		_, err := embeddings.EmbedText(ctx, embedder, texts)
		if err != nil {
			b.Fatalf("Embed failed: %v", err)
		}
	}
}

// BenchmarkHugotEmbedder_Quantized benchmarks quantized vs non-quantized models
func BenchmarkHugotEmbedder_Quantized(b *testing.B) {
	modelPath := os.Getenv("HUGOT_MODEL_PATH")
	if modelPath == "" {
		b.Skip("HUGOT_MODEL_PATH not set, skipping Hugot benchmark")
	}

	texts := generateBenchmarkTexts(benchMediumBatchSize, "medium")
	ctx := context.Background()

	b.Run("Standard", func(b *testing.B) {
		logger := zap.NewNop()
		embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", 1, logger)
		require.NoError(b, err)
		defer func() { _ = embedder.Close() }()

		b.ResetTimer()
		for b.Loop() {
			_, err := embeddings.EmbedText(ctx, embedder, texts)
			if err != nil {
				b.Fatalf("Embed failed: %v", err)
			}
		}
	})

	b.Run("Quantized", func(b *testing.B) {
		logger := zap.NewNop()
		embedder, err := NewPooledHugotEmbedder(modelPath, "model_i8.onnx", 1, logger)
		if err != nil {
			b.Skipf("Quantized model not available: %v", err)
			return
		}
		defer func() { _ = embedder.Close() }()

		b.ResetTimer()
		for b.Loop() {
			_, err := embeddings.EmbedText(ctx, embedder, texts)
			if err != nil {
				b.Fatalf("Embed failed: %v", err)
			}
		}
	})
}

// BenchmarkLatency_SingleText measures single-text embedding latency
func BenchmarkLatency_SingleText(b *testing.B) {
	modelPath := os.Getenv("HUGOT_MODEL_PATH")
	if modelPath == "" {
		b.Skip("HUGOT_MODEL_PATH not set, skipping latency benchmark")
	}

	singleText := []string{benchMediumText}
	ctx := context.Background()

	b.Run("Hugot", func(b *testing.B) {
		logger := zap.NewNop()
		embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", 1, logger)
		require.NoError(b, err)
		defer func() { _ = embedder.Close() }()

		b.ResetTimer()
		for b.Loop() {
			_, err := embeddings.EmbedText(ctx, embedder, singleText)
			if err != nil {
				b.Fatalf("Embed failed: %v", err)
			}
		}
	})
}

// BenchmarkThroughput_BatchSizes compares throughput across different batch sizes
func BenchmarkThroughput_BatchSizes(b *testing.B) {
	modelPath := os.Getenv("HUGOT_MODEL_PATH")
	if modelPath == "" {
		b.Skip("HUGOT_MODEL_PATH not set, skipping throughput benchmark")
	}

	ctx := context.Background()
	batchSizes := []int{1, 10, 50, 100}

	for _, batchSize := range batchSizes {
		texts := generateBenchmarkTexts(batchSize, "medium")

		b.Run(fmt.Sprintf("Hugot/BatchSize_%d", batchSize), func(b *testing.B) {
			logger := zap.NewNop()
			embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", 1, logger)
			require.NoError(b, err)
			defer func() { _ = embedder.Close() }()

			b.ResetTimer()
			for b.Loop() {
				_, err := embeddings.EmbedText(ctx, embedder, texts)
				if err != nil {
					b.Fatalf("Embed failed: %v", err)
				}
			}

			// Report throughput
			docsPerSec := float64(b.N*batchSize) / b.Elapsed().Seconds()
			b.ReportMetric(docsPerSec, "docs/sec")
		})
	}
}

// BenchmarkWarmup tests cold vs warm start performance
func BenchmarkWarmup(b *testing.B) {
	modelPath := os.Getenv("HUGOT_MODEL_PATH")
	if modelPath == "" {
		b.Skip("HUGOT_MODEL_PATH not set, skipping warmup benchmark")
	}

	texts := generateBenchmarkTexts(benchSmallBatchSize, "medium")
	ctx := context.Background()

	b.Run("ColdStart", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			b.StopTimer()
			logger := zap.NewNop()
			embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", 1, logger)
			require.NoError(b, err)
			b.StartTimer()

			_, err = embeddings.EmbedText(ctx, embedder, texts)
			if err != nil {
				b.Fatalf("Embed failed: %v", err)
			}

			b.StopTimer()
			_ = embedder.Close()
			b.StartTimer()
		}
	})

	b.Run("WarmStart", func(b *testing.B) {
		logger := zap.NewNop()
		embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", 1, logger)
		require.NoError(b, err)
		defer func() { _ = embedder.Close() }()

		// Warmup
		_, _ = embeddings.EmbedText(ctx, embedder, texts)
		time.Sleep(100 * time.Millisecond)

		b.ResetTimer()
		for b.Loop() {
			_, err := embeddings.EmbedText(ctx, embedder, texts)
			if err != nil {
				b.Fatalf("Embed failed: %v", err)
			}
		}
	})
}
