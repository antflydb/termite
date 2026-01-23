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
	"path/filepath"
	"testing"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	termreranking "github.com/antflydb/termite/pkg/termite/lib/reranking"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// BenchmarkStaticQuantizedReranker benchmarks the static quantized model with Q/DQ fusion
func BenchmarkStaticQuantizedReranker(b *testing.B) {
	logger := zap.NewNop()

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
		modelPath := filepath.Join("..", "..", "models", "rerankers", "mxbai-rerank-base-v1")
		sessionManager := backends.NewSessionManager()
		cfg := termreranking.PooledRerankerConfig{
			ModelPath: modelPath,
			PoolSize:  1,
			Logger:    logger,
		}
		model, _, err := termreranking.NewPooledReranker(cfg, sessionManager)
		if err != nil {
			b.Skipf("Model not available: %v", err)
		}
		defer func() { _ = model.Close() }()

		b.ResetTimer()
		for b.Loop() {
			_, err := model.Rerank(context.Background(), query, documents)
			require.NoError(b, err)
		}
	})

	b.Run("StaticQuantized_WithFusion", func(b *testing.B) {
		modelPath := filepath.Join("..", "..", "models", "rerankers", "reranker_onnx_static")
		sessionManager := backends.NewSessionManager()
		cfg := termreranking.PooledRerankerConfig{
			ModelPath: modelPath,
			PoolSize:  1,
			Logger:    logger,
		}
		model, _, err := termreranking.NewPooledReranker(cfg, sessionManager)
		if err != nil {
			b.Skipf("Model not available: %v", err)
		}
		defer func() { _ = model.Close() }()

		b.ResetTimer()
		for b.Loop() {
			_, err := model.Rerank(context.Background(), query, documents)
			require.NoError(b, err)
		}
	})
}
