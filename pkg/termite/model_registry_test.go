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
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/antfly-go/libaf/reranking"
	"github.com/antflydb/termite/pkg/termite/lib/backends"
	termreranking "github.com/antflydb/termite/pkg/termite/lib/reranking"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// skipIfNoModels skips the test if the models directory doesn't exist or is empty
func skipIfNoModels(t testing.TB, modelsDir string) {
	t.Helper()
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		t.Skipf("Skipping: models directory not found: %s", modelsDir)
	}
	if len(entries) == 0 {
		t.Skipf("Skipping: no models found in %s", modelsDir)
	}
}

// newRerankerModel creates a reranker model using the new pipeline-based API
func newRerankerModel(modelPath string, sessionManager *backends.SessionManager, logger *zap.Logger) (reranking.Model, error) {
	cfg := termreranking.PooledRerankerConfig{
		ModelPath:     modelPath,
		PoolSize:      1,
		ModelBackends: nil,
		Logger:        logger,
	}
	model, _, err := termreranking.NewPooledReranker(cfg, sessionManager)
	return model, err
}

func TestRerankerRegistryLoading(t *testing.T) {
	// Get path to models directory
	modelsDir := filepath.Join("..", "..", "models", "rerankers")
	skipIfNoModels(t, modelsDir)
	t.Logf("Looking for models in: %s", modelsDir)

	// Create logger for debugging
	logger := zap.NewExample()
	defer func() { _ = logger.Sync() }()

	// Create session manager
	sessionManager := backends.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	// Create registry
	registry, err := NewRerankerRegistry(RerankerConfig{ModelsDir: modelsDir}, sessionManager, logger)
	require.NoError(t, err)
	require.NotNil(t, registry)
	defer func() { _ = registry.Close() }()

	// List models
	models := registry.List()
	t.Logf("Found %d models: %v", len(models), models)

	// Verify that we have at least one model
	require.NotEmpty(t, models, "Expected at least one model to be loaded")

	// Try to get the first available model
	firstModel := models[0]
	model, err := registry.Get(firstModel)
	if err != nil {
		t.Logf("Failed to get %s: %v", firstModel, err)
		t.Logf("Available models: %v", models)
		t.Fatalf("Model %s not loaded", firstModel)
	}
	require.NotNil(t, model)
	t.Logf("Successfully retrieved model: %s", firstModel)
}

func TestCompareQuantizedVsNonQuantized(t *testing.T) {
	modelsDir := filepath.Join("..", "..", "models", "rerankers")
	skipIfNoModels(t, modelsDir)
	logger := zap.NewExample()
	defer func() { _ = logger.Sync() }()

	// Create session manager
	sessionManager := backends.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	// Create registry
	registry, err := NewRerankerRegistry(RerankerConfig{ModelsDir: modelsDir}, sessionManager, logger)
	require.NoError(t, err)
	require.NotNil(t, registry)
	defer func() { _ = registry.Close() }()

	// Test documents
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

	models := registry.List()
	require.NotEmpty(t, models, "At least one model should be available")

	// Test each available model
	for _, modelName := range models {
		t.Run(modelName, func(t *testing.T) {
			model, err := registry.Get(modelName)
			if err != nil {
				t.Skipf("Model not available: %v", err)
			}

			scores, err := model.Rerank(t.Context(), query, documents)
			require.NoError(t, err)
			require.Len(t, scores, len(documents))

			t.Logf("\n%s Results:", modelName)
			t.Logf("Query: %s", query)
			for i, score := range scores {
				t.Logf("  [%d] Score: %.4f - %s", i, score, documents[i])
			}
		})
	}
}

func TestCompareAllRerankerModels(t *testing.T) {
	modelsDir := filepath.Join("..", "..", "models", "rerankers")
	skipIfNoModels(t, modelsDir)
	logger := zap.NewExample()
	defer func() { _ = logger.Sync() }()

	// Create session manager
	sessionManager := backends.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	// Create registry
	registry, err := NewRerankerRegistry(RerankerConfig{ModelsDir: modelsDir}, sessionManager, logger)
	require.NoError(t, err)
	require.NotNil(t, registry)
	defer func() { _ = registry.Close() }()

	// Test documents - a mix of relevant and irrelevant content
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

	type modelResult struct {
		name   string
		scores []float32
	}

	var results []modelResult

	// Test all available models
	for _, modelName := range registry.List() {
		model, err := registry.Get(modelName)
		if err != nil {
			t.Logf("Skipping model %s: %v", modelName, err)
			continue
		}
		scores, err := model.Rerank(context.Background(), query, documents)
		if err != nil {
			t.Logf("Rerank failed for %s: %v", modelName, err)
			continue
		}
		require.Len(t, scores, len(documents))
		results = append(results, modelResult{name: modelName, scores: scores})
	}

	require.NotEmpty(t, results, "At least one model should be available for testing")

	// Print results for each model
	t.Logf("\n=== Reranking Comparison ===")
	t.Logf("Query: %s\n", query)

	for _, result := range results {
		t.Logf("\nModel: %s", result.name)
		for i, score := range result.scores {
			t.Logf("  [%d] Score: %.6f - %s", i, score, documents[i])
		}
	}

	// Compare rankings between models
	if len(results) > 1 {
		t.Logf("\n=== Ranking Correlations ===")
		for i := 0; i < len(results); i++ {
			for j := i + 1; j < len(results); j++ {
				correlation := rankCorrelation(results[i].scores, results[j].scores)
				t.Logf("%s vs %s: %.4f", results[i].name, results[j].name, correlation)
			}
		}
	}
}

// rankCorrelation computes Spearman's rank correlation coefficient
func rankCorrelation(scores1, scores2 []float32) float32 {
	if len(scores1) != len(scores2) {
		return 0
	}

	n := len(scores1)
	rank1 := getRanks(scores1)
	rank2 := getRanks(scores2)

	var sumDiffSquared float32
	for i := range n {
		diff := float32(rank1[i] - rank2[i])
		sumDiffSquared += diff * diff
	}

	// Spearman's rho = 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
	return 1 - (6*sumDiffSquared)/float32(n*(n*n-1))
}

// getRanks returns the rank of each element (1-indexed, higher score = lower rank number)
func getRanks(scores []float32) []int {
	n := len(scores)
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	// Sort indices by scores (descending)
	for i := range n {
		for j := i + 1; j < n; j++ {
			if scores[indices[j]] > scores[indices[i]] {
				indices[i], indices[j] = indices[j], indices[i]
			}
		}
	}

	ranks := make([]int, n)
	for rank, idx := range indices {
		ranks[idx] = rank + 1
	}
	return ranks
}

func BenchmarkRerankerQuantizedVsNonQuantized(b *testing.B) {
	modelsDir := filepath.Join("..", "..", "models", "rerankers")
	skipIfNoModels(b, modelsDir)
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

	// Create session manager
	sessionManager := backends.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	// Create registry
	registry, err := NewRerankerRegistry(RerankerConfig{ModelsDir: modelsDir}, sessionManager, logger)
	require.NoError(b, err)
	require.NotNil(b, registry)
	defer func() { _ = registry.Close() }()

	// Benchmark all available models
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
				require.NoError(b, err)
			}
		})
	}
}

func BenchmarkAllRerankerModels(b *testing.B) {
	modelsDir := filepath.Join("..", "..", "models", "rerankers")
	skipIfNoModels(b, modelsDir)
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

	// Create session manager
	sessionManager := backends.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	// Create registry once for all benchmarks
	registry, err := NewRerankerRegistry(RerankerConfig{ModelsDir: modelsDir}, sessionManager, logger)
	require.NoError(b, err)
	require.NotNil(b, registry)
	defer func() { _ = registry.Close() }()

	// Benchmark all available models
	for _, modelName := range registry.List() {
		model, err := registry.Get(modelName)
		if err != nil {
			b.Logf("Skipping %s: %v", modelName, err)
			continue
		}

		b.Run(modelName, func(b *testing.B) {
			b.ResetTimer()
			for b.Loop() {
				_, err := model.Rerank(context.Background(), query, documents)
				require.NoError(b, err)
			}
		})
	}
}

// Embedder Registry Tests

func TestEmbedderRegistryLoading(t *testing.T) {
	// Get path to models directory
	modelsDir := filepath.Join("..", "..", "models", "embedders")
	skipIfNoModels(t, modelsDir)
	t.Logf("Looking for embedder models in: %s", modelsDir)

	// Create logger for debugging
	logger := zap.NewExample()
	defer func() { _ = logger.Sync() }()

	// Create session manager
	sessionManager := backends.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	// Create registry
	registry, err := NewEmbedderRegistry(EmbedderConfig{ModelsDir: modelsDir}, sessionManager, logger)
	require.NoError(t, err)
	require.NotNil(t, registry)
	defer func() { _ = registry.Close() }()

	// List models
	models := registry.List()
	t.Logf("Found %d embedder models: %v", len(models), models)

	// Verify that we have at least one model
	require.NotEmpty(t, models, "Expected at least one embedder model to be loaded")

	// Try to get the first available model
	firstModel := models[0]
	model, err := registry.Get(firstModel)
	if err != nil {
		t.Logf("Failed to get %s: %v", firstModel, err)
		t.Logf("Available models: %v", models)
		t.Fatalf("Model %s not loaded", firstModel)
	}
	require.NotNil(t, model)
	t.Logf("Successfully retrieved model: %s", firstModel)
}

func TestEmbedderModelEmbedding(t *testing.T) {
	modelsDir := filepath.Join("..", "..", "models", "embedders")
	skipIfNoModels(t, modelsDir)
	logger := zap.NewExample()
	defer func() { _ = logger.Sync() }()

	// Create session manager
	sessionManager := backends.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	// Create registry
	registry, err := NewEmbedderRegistry(EmbedderConfig{ModelsDir: modelsDir}, sessionManager, logger)
	require.NoError(t, err)
	require.NotNil(t, registry)
	defer func() { _ = registry.Close() }()

	models := registry.List()
	require.NotEmpty(t, models, "Expected at least one embedder model")

	// Get the first available model
	firstModel := models[0]
	model, err := registry.Get(firstModel)
	require.NoError(t, err, "Failed to get model %s", firstModel)
	require.NotNil(t, model)

	// Test input texts
	texts := []string{
		"Machine learning is a subset of artificial intelligence.",
		"The weather today is sunny and warm.",
		"Deep learning uses neural networks with multiple layers.",
	}

	ctx := context.Background()

	// Generate embeddings
	embeds, err := embeddings.EmbedText(ctx, model, texts)
	require.NoError(t, err)
	require.NotNil(t, embeds)
	require.Len(t, embeds, len(texts), "Should return one embedding per input text")

	// Verify embeddings have expected dimensions
	for i, embedding := range embeds {
		require.NotEmpty(t, embedding, "Embedding %d should have non-zero dimensions", i)
		t.Logf("Text %d: %d dimensions", i, len(embedding))
	}

	// All embeddings should have the same dimension
	firstDim := len(embeds[0])
	for i, embedding := range embeds {
		require.Len(t, embedding, firstDim, "All embeddings should have the same dimension (embedding %d)", i)
	}

	t.Logf("Successfully generated %d embeddings with dimension %d using %s", len(embeds), firstDim, firstModel)
}

func TestEmbedderQuantizedVsNonQuantized(t *testing.T) {
	modelsDir := filepath.Join("..", "..", "models", "embedders")
	skipIfNoModels(t, modelsDir)
	logger := zap.NewExample()
	defer func() { _ = logger.Sync() }()

	// Create session manager
	sessionManager := backends.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	// Create registry
	registry, err := NewEmbedderRegistry(EmbedderConfig{ModelsDir: modelsDir}, sessionManager, logger)
	require.NoError(t, err)
	require.NotNil(t, registry)
	defer func() { _ = registry.Close() }()

	// Test texts
	texts := []string{
		"Machine learning is a subset of artificial intelligence.",
		"The weather today is sunny and warm.",
	}

	ctx := context.Background()
	models := registry.List()
	require.NotEmpty(t, models, "Expected at least one embedder model")

	type modelResult struct {
		name   string
		embeds [][]float32
	}

	var results []modelResult

	// Test all available models
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
			t.Logf("%s: Generated %d embeddings with dimension %d", modelName, len(embeds), len(embeds[0]))
			results = append(results, modelResult{name: modelName, embeds: embeds})
		})
	}

	// Compare similarity between models if we have multiple
	if len(results) > 1 {
		t.Run("Compare", func(t *testing.T) {
			for i := 0; i < len(results); i++ {
				for j := i + 1; j < len(results); j++ {
					// Only compare if dimensions match
					if len(results[i].embeds[0]) != len(results[j].embeds[0]) {
						t.Logf("Skipping comparison: %s (%d dims) vs %s (%d dims)",
							results[i].name, len(results[i].embeds[0]),
							results[j].name, len(results[j].embeds[0]))
						continue
					}
					similarity := cosineSimilarity(results[i].embeds[0], results[j].embeds[0])
					t.Logf("%s vs %s: Cosine similarity: %.6f", results[i].name, results[j].name, similarity)
				}
			}
		})
	}
}

func BenchmarkEmbedderQuantizedVsNonQuantized(b *testing.B) {
	modelsDir := filepath.Join("..", "..", "models", "embedders")
	skipIfNoModels(b, modelsDir)
	logger := zap.NewNop()

	texts := []string{
		"Machine learning is a subset of artificial intelligence.",
		"The weather today is sunny and warm.",
		"Deep learning uses neural networks with multiple layers.",
	}

	ctx := context.Background()

	// Create session manager
	sessionManager := backends.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	registry, err := NewEmbedderRegistry(EmbedderConfig{ModelsDir: modelsDir}, sessionManager, logger)
	require.NoError(b, err)
	require.NotNil(b, registry)
	defer func() { _ = registry.Close() }()

	// Benchmark all available models
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
				require.NoError(b, err)
			}
		})
	}
}

// cosineSimilarity computes cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float32 {
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
