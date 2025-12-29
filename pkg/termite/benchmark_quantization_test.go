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
	"context"
	"testing"

	termreranking "github.com/antflydb/termite/pkg/termite/lib/reranking"
	"go.uber.org/zap"
)

// BenchmarkRerankerQuantization benchmarks all three quantization approaches
func BenchmarkRerankerQuantization(b *testing.B) {

	// Sample query and documents
	query := "What is machine learning?"
	documents := []string{
		"Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
		"The weather today is sunny with a chance of rain in the afternoon.",
		"Deep learning uses neural networks with multiple layers to learn hierarchical representations.",
		"Cooking pasta requires boiling water and adding salt.",
		"Supervised learning algorithms learn from labeled training data to make predictions.",
		"The stock market fluctuates based on various economic factors.",
		"Natural language processing enables computers to understand and generate human language.",
		"Gardening is a relaxing hobby that connects people with nature.",
		"Reinforcement learning involves agents learning through trial and error with rewards.",
		"Classical music has been popular for centuries across many cultures.",
	}

	b.Run("NonQuantized", func(b *testing.B) {
		reranker, err := termreranking.NewPooledHugotReranker(
			"../../models/rerankers/mxbai-rerank-base-v1",
			"model.onnx",
			1,
			zap.NewNop(),
		)
		if err != nil {
			b.Fatalf("Failed to create non-quantized reranker: %v", err)
		}
		defer func() { _ = reranker.Close() }()

		b.ResetTimer()
		for b.Loop() {
			_, err := reranker.Rerank(context.Background(), query, documents)
			if err != nil {
				b.Fatalf("Rerank failed: %v", err)
			}
		}
	})

	b.Run("DynamicQuantized", func(b *testing.B) {
		reranker, err := termreranking.NewPooledHugotReranker(
			"../../models/rerankers/mxbai-rerank-base-v1",
			"model_i8.onnx",
			1,
			zap.NewNop(),
		)
		if err != nil {
			b.Fatalf("Failed to create dynamically quantized reranker: %v", err)
		}
		defer func() { _ = reranker.Close() }()

		b.ResetTimer()
		for b.Loop() {
			_, err := reranker.Rerank(context.Background(), query, documents)
			if err != nil {
				b.Fatalf("Rerank failed: %v", err)
			}
		}
	})

	b.Run("FP16", func(b *testing.B) {
		reranker, err := termreranking.NewPooledHugotReranker(
			"../../models/rerankers/reranker_onnx_fp16",
			"model_f16.onnx",
			1,
			zap.NewNop(),
		)
		if err != nil {
			b.Fatalf("Failed to create FP16 reranker: %v", err)
		}
		defer func() { _ = reranker.Close() }()

		b.ResetTimer()
		for b.Loop() {
			_, err := reranker.Rerank(context.Background(), query, documents)
			if err != nil {
				b.Fatalf("Rerank failed: %v", err)
			}
		}
	})
}
