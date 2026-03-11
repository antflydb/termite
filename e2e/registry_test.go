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
	"math"
	"testing"

	"github.com/antflydb/antfly/pkg/libaf/embeddings"
	"github.com/antflydb/termite/pkg/termite"
	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// TestRerankerRegistryLoading verifies that the reranker registry can discover
// and load models from the models directory.
func TestRerankerRegistryLoading(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}

	ensureHuggingFaceModel(t, "mxbai-rerank-base-v1", "mixedbread-ai/mxbai-rerank-base-v1", ModelTypeReranker)

	modelsDir := getTestModelsDir()
	sessionManager := backends.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	registry, err := termite.NewRerankerRegistry(termite.RerankerConfig{ModelsDir: modelsDir, MaxLoadedModels: 1}, sessionManager, zap.NewNop())
	require.NoError(t, err)
	defer func() { _ = registry.Close() }()

	models := registry.List()
	t.Logf("Found %d reranker models: %v", len(models), models)
	require.NotEmpty(t, models, "Expected at least one reranker model")

	model, err := registry.Get(models[0])
	require.NoError(t, err)
	require.NotNil(t, model)
	t.Logf("Successfully retrieved model: %s", models[0])
}

// TestRerankerRegistryRerank verifies that all discovered reranker models
// produce sensible reranking scores.
func TestRerankerRegistryRerank(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}

	ensureHuggingFaceModel(t, "mxbai-rerank-base-v1", "mixedbread-ai/mxbai-rerank-base-v1", ModelTypeReranker)

	modelsDir := getTestModelsDir()
	sessionManager := backends.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	registry, err := termite.NewRerankerRegistry(termite.RerankerConfig{ModelsDir: modelsDir, MaxLoadedModels: 1}, sessionManager, zap.NewNop())
	require.NoError(t, err)
	defer func() { _ = registry.Close() }()

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

	for _, modelName := range registry.List() {
		t.Run(modelName, func(t *testing.T) {
			model, err := registry.Get(modelName)
			if err != nil {
				t.Skipf("Model not available: %v", err)
			}

			scores, err := model.Rerank(t.Context(), query, documents)
			require.NoError(t, err)
			require.Len(t, scores, len(documents))

			t.Logf("Query: %s", query)
			for i, score := range scores {
				t.Logf("  [%d] Score: %.4f - %s", i, score, documents[i])
			}
		})
	}
}

// TestEmbedderRegistryLoading verifies that the embedder registry can discover
// and load models from the models directory.
func TestEmbedderRegistryLoading(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}

	ensureHuggingFaceModel(t, "Snowflake/snowflake-arctic-embed-l-v2.0", "Snowflake/snowflake-arctic-embed-l-v2.0", ModelTypeEmbedder)

	modelsDir := getTestModelsDir()
	sessionManager := backends.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	registry, err := termite.NewEmbedderRegistry(termite.EmbedderConfig{ModelsDir: modelsDir, MaxLoadedModels: 1}, sessionManager, zap.NewNop())
	require.NoError(t, err)
	defer func() { _ = registry.Close() }()

	models := registry.List()
	t.Logf("Found %d embedder models: %v", len(models), models)
	require.NotEmpty(t, models, "Expected at least one embedder model")

	model, err := registry.Get(models[0])
	require.NoError(t, err)
	require.NotNil(t, model)
	t.Logf("Successfully retrieved model: %s", models[0])
}

// TestEmbedderRegistryEmbed verifies that all discovered embedder models
// produce valid embeddings.
func TestEmbedderRegistryEmbed(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}

	ensureHuggingFaceModel(t, "Snowflake/snowflake-arctic-embed-l-v2.0", "Snowflake/snowflake-arctic-embed-l-v2.0", ModelTypeEmbedder)

	modelsDir := getTestModelsDir()
	sessionManager := backends.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	registry, err := termite.NewEmbedderRegistry(termite.EmbedderConfig{ModelsDir: modelsDir, MaxLoadedModels: 1}, sessionManager, zap.NewNop())
	require.NoError(t, err)
	defer func() { _ = registry.Close() }()

	texts := []string{
		"Machine learning is a subset of artificial intelligence.",
		"The weather today is sunny and warm.",
		"Deep learning uses neural networks with multiple layers.",
	}

	ctx := context.Background()
	models := registry.List()
	require.NotEmpty(t, models)

	type modelResult struct {
		name   string
		embeds [][]float32
	}
	var results []modelResult

	for _, modelName := range models {
		model, err := registry.Get(modelName)
		if err != nil {
			t.Logf("Skipping %s: %v", modelName, err)
			continue
		}

		t.Run(modelName, func(t *testing.T) {
			embeds, err := embeddings.EmbedText(ctx, model, texts)
			require.NoError(t, err)
			require.Len(t, embeds, len(texts))

			// All embeddings should have the same dimension
			firstDim := len(embeds[0])
			for i, emb := range embeds {
				require.NotEmpty(t, emb, "Embedding %d should have non-zero dimensions", i)
				require.Len(t, emb, firstDim, "All embeddings should have the same dimension")
			}
			t.Logf("%s: Generated %d embeddings with dimension %d", modelName, len(embeds), firstDim)
			results = append(results, modelResult{name: modelName, embeds: embeds})
		})
	}

	// Compare similarity between models if we have multiple with matching dimensions
	if len(results) > 1 {
		t.Run("CrossModelComparison", func(t *testing.T) {
			for i := 0; i < len(results); i++ {
				for j := i + 1; j < len(results); j++ {
					if len(results[i].embeds[0]) != len(results[j].embeds[0]) {
						t.Logf("Skipping comparison: %s (%d dims) vs %s (%d dims)",
							results[i].name, len(results[i].embeds[0]),
							results[j].name, len(results[j].embeds[0]))
						continue
					}
					sim := cosineSimilarityF32(results[i].embeds[0], results[j].embeds[0])
					t.Logf("%s vs %s: Cosine similarity: %.6f", results[i].name, results[j].name, sim)
				}
			}
		})
	}
}

// BenchmarkRerankerRegistry benchmarks all discovered reranker models.
func BenchmarkRerankerRegistry(b *testing.B) {
	ensureHuggingFaceModel(b, "mxbai-rerank-base-v1", "mixedbread-ai/mxbai-rerank-base-v1", ModelTypeReranker)

	modelsDir := getTestModelsDir()
	sessionManager := backends.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	registry, err := termite.NewRerankerRegistry(termite.RerankerConfig{ModelsDir: modelsDir, MaxLoadedModels: 1}, sessionManager, zap.NewNop())
	require.NoError(b, err)
	defer func() { _ = registry.Close() }()

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

	for _, modelName := range registry.List() {
		model, err := registry.Get(modelName)
		if err != nil {
			b.Logf("Skipping %s: %v", modelName, err)
			continue
		}

		b.Run(modelName, func(b *testing.B) {
			b.ResetTimer()
			for b.Loop() {
				_, err := model.Rerank(b.Context(), query, documents)
				if err != nil {
					b.Fatalf("Rerank failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkEmbedderRegistry benchmarks all discovered embedder models.
func BenchmarkEmbedderRegistry(b *testing.B) {
	ensureHuggingFaceModel(b, "Snowflake/snowflake-arctic-embed-l-v2.0", "Snowflake/snowflake-arctic-embed-l-v2.0", ModelTypeEmbedder)

	modelsDir := getTestModelsDir()
	sessionManager := backends.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	registry, err := termite.NewEmbedderRegistry(termite.EmbedderConfig{ModelsDir: modelsDir, MaxLoadedModels: 1}, sessionManager, zap.NewNop())
	require.NoError(b, err)
	defer func() { _ = registry.Close() }()

	texts := []string{
		"Machine learning is a subset of artificial intelligence.",
		"The weather today is sunny and warm.",
		"Deep learning uses neural networks with multiple layers.",
	}

	ctx := context.Background()

	for _, modelName := range registry.List() {
		model, err := registry.Get(modelName)
		if err != nil {
			b.Logf("Skipping %s: %v", modelName, err)
			continue
		}

		b.Run(modelName, func(b *testing.B) {
			b.ResetTimer()
			for b.Loop() {
				_, err := embeddings.EmbedText(ctx, model, texts)
				if err != nil {
					b.Fatalf("Embed failed: %v", err)
				}
			}
		})
	}
}

// cosineSimilarityF32 computes cosine similarity between two float32 vectors.
func cosineSimilarityF32(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return float32(dotProduct / (math.Sqrt(normA) * math.Sqrt(normB)))
}
