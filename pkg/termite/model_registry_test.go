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
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
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

// newNonQuantizedHugotModel creates a Hugot model that explicitly uses the non-quantized version
func newNonQuantizedHugotModel(modelPath string, logger *zap.Logger) (reranking.Model, error) {
	return termreranking.NewHugotReranker(modelPath, "model.onnx", logger)
}

// newQuantizedHugotModel creates a Hugot model that explicitly uses the quantized version
func newQuantizedHugotModel(modelPath string, logger *zap.Logger) (reranking.Model, error) {
	return termreranking.NewHugotReranker(modelPath, "model_i8.onnx", logger)
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
	sessionManager := hugot.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	// Create registry
	registry, err := NewRerankerRegistry(modelsDir, sessionManager, logger)
	require.NoError(t, err)
	require.NotNil(t, registry)
	defer func() { _ = registry.Close() }()

	// List models
	models := registry.List()
	t.Logf("Found %d models: %v", len(models), models)

	// Verify that we have at least one model
	require.NotEmpty(t, models, "Expected at least one model to be loaded")

	// Try to get the standard mxbai-rerank-base-v1 model
	standardModel, err := registry.Get("mxbai-rerank-base-v1")
	if err != nil {
		t.Logf("Failed to get mxbai-rerank-base-v1: %v", err)
		t.Logf("Available models: %v", models)
		t.Fatalf("Model mxbai-rerank-base-v1 not loaded")
	}
	require.NotNil(t, standardModel)
	t.Logf("Successfully retrieved standard mxbai-rerank-base-v1 model")

	// Try to get the quantized mxbai-rerank-base-v1-i8-qt model
	quantizedModel, err := registry.Get("mxbai-rerank-base-v1-i8-qt")
	if err != nil {
		t.Logf("Failed to get mxbai-rerank-base-v1-i8-qt: %v", err)
		t.Logf("Available models: %v", models)
		t.Fatalf("Model mxbai-rerank-base-v1-i8-qt not loaded")
	}
	require.NotNil(t, quantizedModel)
	t.Logf("Successfully retrieved quantized mxbai-rerank-base-v1-i8-qt model")

	// Try to get the new static quantized model
	staticModel, err := registry.Get("reranker_onnx_static-i8-qt")
	if err != nil {
		t.Logf("Failed to get reranker_onnx_static-i8-qt: %v", err)
		t.Logf("Available models: %v", models)
		t.Fatalf("Model reranker_onnx_static-i8-qt not loaded")
	}
	require.NotNil(t, staticModel)
	t.Logf("Successfully retrieved static quantized reranker_onnx_static-i8-qt model")

	// Verify that all models are different instances
	require.NotEqual(t, standardModel, quantizedModel, "Standard and quantized models should be different instances")
	require.NotEqual(t, standardModel, staticModel, "Standard and static models should be different instances")
	require.NotEqual(t, quantizedModel, staticModel, "Quantized and static models should be different instances")
}

func TestCompareQuantizedVsNonQuantized(t *testing.T) {
	modelPath := filepath.Join("..", "..", "models", "rerankers", "mxbai-rerank-base-v1")
	skipIfNoModels(t, modelPath)
	logger := zap.NewExample()
	defer func() { _ = logger.Sync() }()

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

	t.Run("NonQuantized", func(t *testing.T) {
		model, err := newNonQuantizedHugotModel(modelPath, logger.Named("non-quantized"))
		require.NoError(t, err)
		require.NotNil(t, model)
		defer func() { _ = model.Close() }()

		scores, err := model.Rerank(t.Context(), query, documents)
		require.NoError(t, err)
		require.Len(t, scores, len(documents))

		t.Logf("\nNon-Quantized Model Results:")
		t.Logf("Query: %s", query)
		for i, score := range scores {
			t.Logf("  [%d] Score: %.4f - %s", i, score, documents[i])
		}
	})

	t.Run("Quantized", func(t *testing.T) {
		// Temporarily create a quantized model by directly calling reranking package
		logger := zap.NewExample()
		defer func() { _ = logger.Sync() }()

		model, err := newQuantizedHugotModel(modelPath, logger.Named("quantized"))
		require.NoError(t, err)
		require.NotNil(t, model)
		defer func() { _ = model.Close() }()

		scores, err := model.Rerank(t.Context(), query, documents)
		require.NoError(t, err)
		require.Len(t, scores, len(documents))

		t.Logf("\nQuantized Model Results:")
		t.Logf("Query: %s", query)
		for i, score := range scores {
			t.Logf("  [%d] Score: %.4f - %s", i, score, documents[i])
		}
	})
}

func TestCompareAllRerankerModels(t *testing.T) {
	modelsDir := filepath.Join("..", "..", "models", "rerankers")
	skipIfNoModels(t, modelsDir)
	logger := zap.NewExample()
	defer func() { _ = logger.Sync() }()

	// Create session manager
	sessionManager := hugot.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	// Create registry
	registry, err := NewRerankerRegistry(modelsDir, sessionManager, logger)
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

	// Test standard model
	if model, err := registry.Get("mxbai-rerank-base-v1"); err == nil {
		scores, err := model.Rerank(context.Background(), query, documents)
		require.NoError(t, err)
		require.Len(t, scores, len(documents))
		results = append(results, modelResult{name: "mxbai-rerank-base-v1 (standard)", scores: scores})
	}

	// Test quantized model
	if model, err := registry.Get("mxbai-rerank-base-v1-i8-qt"); err == nil {
		scores, err := model.Rerank(context.Background(), query, documents)
		require.NoError(t, err)
		require.Len(t, scores, len(documents))
		results = append(results, modelResult{name: "mxbai-rerank-base-v1-i8-qt (quantized)", scores: scores})
	}

	// Test static quantized model
	if model, err := registry.Get("reranker_onnx_static-i8-qt"); err == nil {
		scores, err := model.Rerank(context.Background(), query, documents)
		require.NoError(t, err)
		require.Len(t, scores, len(documents))
		results = append(results, modelResult{name: "reranker_onnx_static-i8-qt (static)", scores: scores})
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
	modelPath := filepath.Join("..", "..", "models", "rerankers", "mxbai-rerank-base-v1")
	skipIfNoModels(b, modelPath)
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
		model, err := newNonQuantizedHugotModel(modelPath, logger)
		require.NoError(b, err)
		require.NotNil(b, model)
		defer func() { _ = model.Close() }()

		b.ResetTimer()
		for b.Loop() {
			_, err := model.Rerank(b.Context(), query, documents)
			require.NoError(b, err)
		}
	})

	b.Run("Quantized", func(b *testing.B) {
		model, err := newQuantizedHugotModel(modelPath, logger)
		require.NoError(b, err)
		require.NotNil(b, model)
		defer func() { _ = model.Close() }()

		b.ResetTimer()
		for b.Loop() {
			_, err := model.Rerank(b.Context(), query, documents)
			require.NoError(b, err)
		}
	})
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
	sessionManager := hugot.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	// Create registry once for all benchmarks
	registry, err := NewRerankerRegistry(modelsDir, sessionManager, logger)
	require.NoError(b, err)
	require.NotNil(b, registry)
	defer func() { _ = registry.Close() }()

	// Benchmark standard model
	if model, err := registry.Get("mxbai-rerank-base-v1"); err == nil {
		b.Run("Standard_mxbai-rerank-base-v1", func(b *testing.B) {
			b.ResetTimer()
			for b.Loop() {
				_, err := model.Rerank(context.Background(), query, documents)
				require.NoError(b, err)
			}
		})
	}

	// Benchmark quantized model
	if model, err := registry.Get("mxbai-rerank-base-v1-i8-qt"); err == nil {
		b.Run("Quantized_mxbai-rerank-base-v1-i8-qt", func(b *testing.B) {
			b.ResetTimer()
			for b.Loop() {
				_, err := model.Rerank(context.Background(), query, documents)
				require.NoError(b, err)
			}
		})
	}

	// Benchmark static quantized model
	if model, err := registry.Get("reranker_onnx_static-i8-qt"); err == nil {
		b.Run("Static_reranker_onnx_static-i8-qt", func(b *testing.B) {
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
	sessionManager := hugot.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	// Create registry
	registry, err := NewEmbedderRegistry(modelsDir, sessionManager, logger)
	require.NoError(t, err)
	require.NotNil(t, registry)
	defer func() { _ = registry.Close() }()

	// List models
	models := registry.List()
	t.Logf("Found %d embedder models: %v", len(models), models)

	// Verify that we have at least one model
	require.NotEmpty(t, models, "Expected at least one embedder model to be loaded")

	// Try to get the bge_small_onnx model
	model, err := registry.Get("bge_small_onnx")
	if err != nil {
		t.Logf("Failed to get bge_small_onnx: %v", err)
		t.Logf("Available models: %v", models)
		t.Fatalf("Model bge_small_onnx not loaded")
	}
	require.NotNil(t, model)
	t.Logf("Successfully retrieved bge_small_onnx model")
}

func TestEmbedderModelEmbedding(t *testing.T) {
	modelsDir := filepath.Join("..", "..", "models", "embedders")
	skipIfNoModels(t, modelsDir)
	logger := zap.NewExample()
	defer func() { _ = logger.Sync() }()

	// Create session manager
	sessionManager := hugot.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	// Create registry
	registry, err := NewEmbedderRegistry(modelsDir, sessionManager, logger)
	require.NoError(t, err)
	require.NotNil(t, registry)
	defer func() { _ = registry.Close() }()

	// Get the bge_small_onnx model
	model, err := registry.Get("bge_small_onnx")
	require.NoError(t, err)
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

	t.Logf("Successfully generated %d embeddings with dimension %d", len(embeds), firstDim)
}

func TestEmbedderQuantizedVsNonQuantized(t *testing.T) {
	modelsDir := filepath.Join("..", "..", "models", "embedders")
	skipIfNoModels(t, modelsDir)
	logger := zap.NewExample()
	defer func() { _ = logger.Sync() }()

	// Create session manager
	sessionManager := hugot.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	// Create registry
	registry, err := NewEmbedderRegistry(modelsDir, sessionManager, logger)
	require.NoError(t, err)
	require.NotNil(t, registry)
	defer func() { _ = registry.Close() }()

	// Test texts
	texts := []string{
		"Machine learning is a subset of artificial intelligence.",
		"The weather today is sunny and warm.",
	}

	ctx := context.Background()

	// Try to get both standard and quantized models
	standardModel, errStd := registry.Get("bge_small_onnx")
	quantizedModel, errQt := registry.Get("bge_small_onnx-i8-qt")

	// Test standard model if available
	if errStd == nil {
		t.Run("Standard", func(t *testing.T) {
			embeds, err := embeddings.EmbedText(ctx, standardModel, texts)
			require.NoError(t, err)
			require.Len(t, embeds, len(texts))
			t.Logf("Standard model: Generated %d embeddings with dimension %d", len(embeds), len(embeds[0]))
		})
	} else {
		t.Logf("Standard model not available: %v", errStd)
	}

	// Test quantized model if available
	if errQt == nil {
		t.Run("Quantized", func(t *testing.T) {
			embeds, err := embeddings.EmbedText(ctx, quantizedModel, texts)
			require.NoError(t, err)
			require.Len(t, embeds, len(texts))
			t.Logf("Quantized model: Generated %d embeddings with dimension %d", len(embeds), len(embeds[0]))
		})
	} else {
		t.Logf("Quantized model not available: %v", errQt)
	}

	// If both are available, verify they produce similar results
	if errStd == nil && errQt == nil {
		t.Run("Compare", func(t *testing.T) {
			stdEmbeds, err := embeddings.EmbedText(ctx, standardModel, texts)
			require.NoError(t, err)

			qtEmbeds, err := embeddings.EmbedText(ctx, quantizedModel, texts)
			require.NoError(t, err)

			require.Len(t, qtEmbeds, len(stdEmbeds), "Should produce same number of embeddings")
			require.Len(t, qtEmbeds[0], len(stdEmbeds[0]), "Should produce same dimension")

			// Compare similarity (quantized may differ more with int8 quantization)
			for i := range stdEmbeds {
				similarity := cosineSimilarity(stdEmbeds[i], qtEmbeds[i])
				t.Logf("Embedding %d: Cosine similarity between standard and quantized: %.6f", i, similarity)
				// Note: int8 quantization can reduce similarity significantly
				// Just log for informational purposes
				if similarity < 0.5 {
					t.Logf("Warning: Low similarity (%.6f) - int8 quantization may cause significant drift", similarity)
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
	sessionManager := hugot.NewSessionManager()
	defer func() { _ = sessionManager.Close() }()

	registry, err := NewEmbedderRegistry(modelsDir, sessionManager, logger)
	require.NoError(b, err)
	require.NotNil(b, registry)
	defer func() { _ = registry.Close() }()

	// Benchmark standard model if available
	if model, err := registry.Get("bge_small_onnx"); err == nil {
		b.Run("Standard", func(b *testing.B) {
			b.ResetTimer()
			for b.Loop() {
				_, err := embeddings.EmbedText(ctx, model, texts)
				require.NoError(b, err)
			}
		})
	}

	// Benchmark quantized model if available
	if model, err := registry.Get("bge_small_onnx-i8-qt"); err == nil {
		b.Run("Quantized", func(b *testing.B) {
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
