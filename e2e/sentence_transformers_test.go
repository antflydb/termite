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
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/termite"
	"go.uber.org/zap/zaptest"
)

// Sentence-transformers model specifications
const (
	// all-MiniLM-L6-v2: Fast, lightweight model with 384 dimensions
	// Downloaded from HuggingFace: sentence-transformers/all-MiniLM-L6-v2
	miniLMHFRepo       = "sentence-transformers/all-MiniLM-L6-v2"
	miniLMLocalName    = "all-MiniLM-L6-v2"
	miniLMModelName    = "sentence-transformers/all-MiniLM-L6-v2"
	miniLMEmbeddingDim = 384
	miniLMMaxSeqLen    = 256

	// all-mpnet-base-v2: Higher quality model with 768 dimensions
	// Downloaded from HuggingFace: sentence-transformers/all-mpnet-base-v2
	mpnetHFRepo       = "sentence-transformers/all-mpnet-base-v2"
	mpnetLocalName    = "all-mpnet-base-v2"
	mpnetModelName    = "sentence-transformers/all-mpnet-base-v2"
	mpnetEmbeddingDim = 768
	mpnetMaxSeqLen    = 384
)

// TestSentenceTransformersMiniLME2E tests the all-MiniLM-L6-v2 embedding model.
//
// all-MiniLM-L6-v2 is a popular sentence-transformers model that:
// - Maps sentences to 384-dimensional dense vectors
// - Supports up to 256 tokens
// - Uses mean pooling over transformer outputs
// - Is optimized for semantic similarity and search
func TestSentenceTransformersMiniLME2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Ensure model is downloaded from HuggingFace (lazy download)
	ensureHuggingFaceModel(t, miniLMLocalName, miniLMHFRepo, ModelTypeEmbedder)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)

	// Use shared models directory from test harness
	modelsDir := getTestModelsDir()
	t.Logf("Using models directory: %s", modelsDir)

	// Find an available port
	port := findAvailablePort(t)
	serverURL := fmt.Sprintf("http://localhost:%d", port)
	t.Logf("Starting server on %s", serverURL)

	// Start termite server
	config := termite.Config{
		ApiUrl:    serverURL,
		ModelsDir: modelsDir,
	}

	serverCtx, serverCancel := context.WithCancel(ctx)
	defer serverCancel()

	readyC := make(chan struct{})
	serverDone := make(chan struct{})

	go func() {
		defer close(serverDone)
		termite.RunAsTermite(serverCtx, logger, config, readyC)
	}()

	// Wait for server to be ready
	select {
	case <-readyC:
		t.Log("Server is ready")
	case <-time.After(60 * time.Second):
		t.Fatal("Timeout waiting for server to be ready")
	}

	// Create client
	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Run test cases
	t.Run("ListModels", func(t *testing.T) {
		testListModelsContains(t, ctx, termiteClient, miniLMModelName)
	})

	t.Run("TextEmbedding", func(t *testing.T) {
		testTextEmbeddingWithDim(t, ctx, termiteClient, miniLMModelName, miniLMEmbeddingDim)
	})

	t.Run("SemanticSimilarity", func(t *testing.T) {
		testSemanticSimilarity(t, ctx, termiteClient, miniLMModelName)
	})

	t.Run("BatchEmbedding", func(t *testing.T) {
		testBatchEmbedding(t, ctx, termiteClient, miniLMModelName, miniLMEmbeddingDim)
	})

	// Graceful shutdown
	t.Log("Shutting down server...")
	serverCancel()

	select {
	case <-serverDone:
		t.Log("Server shutdown complete")
	case <-time.After(30 * time.Second):
		t.Error("Timeout waiting for server shutdown")
	}
}

// TestSentenceTransformersMPNetE2E tests the all-mpnet-base-v2 embedding model.
//
// all-mpnet-base-v2 is a high-quality sentence-transformers model that:
// - Maps sentences to 768-dimensional dense vectors
// - Supports up to 384 tokens
// - Uses mean pooling over MPNet outputs
// - Offers better quality than MiniLM at the cost of speed
func TestSentenceTransformersMPNetE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Ensure model is downloaded from HuggingFace (lazy download)
	ensureHuggingFaceModel(t, mpnetLocalName, mpnetHFRepo, ModelTypeEmbedder)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)

	// Use shared models directory from test harness
	modelsDir := getTestModelsDir()
	t.Logf("Using models directory: %s", modelsDir)

	// Find an available port
	port := findAvailablePort(t)
	serverURL := fmt.Sprintf("http://localhost:%d", port)
	t.Logf("Starting server on %s", serverURL)

	// Start termite server
	config := termite.Config{
		ApiUrl:    serverURL,
		ModelsDir: modelsDir,
	}

	serverCtx, serverCancel := context.WithCancel(ctx)
	defer serverCancel()

	readyC := make(chan struct{})
	serverDone := make(chan struct{})

	go func() {
		defer close(serverDone)
		termite.RunAsTermite(serverCtx, logger, config, readyC)
	}()

	// Wait for server to be ready
	select {
	case <-readyC:
		t.Log("Server is ready")
	case <-time.After(60 * time.Second):
		t.Fatal("Timeout waiting for server to be ready")
	}

	// Create client
	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Run test cases
	t.Run("ListModels", func(t *testing.T) {
		testListModelsContains(t, ctx, termiteClient, mpnetModelName)
	})

	t.Run("TextEmbedding", func(t *testing.T) {
		testTextEmbeddingWithDim(t, ctx, termiteClient, mpnetModelName, mpnetEmbeddingDim)
	})

	t.Run("SemanticSimilarity", func(t *testing.T) {
		testSemanticSimilarity(t, ctx, termiteClient, mpnetModelName)
	})

	t.Run("BatchEmbedding", func(t *testing.T) {
		testBatchEmbedding(t, ctx, termiteClient, mpnetModelName, mpnetEmbeddingDim)
	})

	// Graceful shutdown
	t.Log("Shutting down server...")
	serverCancel()

	select {
	case <-serverDone:
		t.Log("Server shutdown complete")
	case <-time.After(30 * time.Second):
		t.Error("Timeout waiting for server shutdown")
	}
}

// testListModelsContains verifies the specified model appears in the models list
func testListModelsContains(t *testing.T, ctx context.Context, c *client.TermiteClient, modelName string) {
	t.Helper()

	models, err := c.ListModels(ctx)
	if err != nil {
		t.Fatalf("ListModels failed: %v", err)
	}

	// Check that model is in the embedders list
	found := false
	for _, name := range models.Embedders {
		if name == modelName {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("Model %s not found in embedders list: %v", modelName, models.Embedders)
	} else {
		t.Logf("Found model in embedders: %s", modelName)
	}
}

// testTextEmbeddingWithDim tests embedding text strings and verifies dimensions
func testTextEmbeddingWithDim(t *testing.T, ctx context.Context, c *client.TermiteClient, modelName string, expectedDim int) {
	t.Helper()

	texts := []string{
		"The quick brown fox jumps over the lazy dog.",
		"Machine learning is a subset of artificial intelligence.",
		"Natural language processing enables computers to understand text.",
	}

	embeddings, err := c.Embed(ctx, modelName, texts)
	if err != nil {
		t.Fatalf("Text embedding failed: %v", err)
	}

	if len(embeddings) != len(texts) {
		t.Errorf("Expected %d embeddings, got %d", len(texts), len(embeddings))
	}

	for i, emb := range embeddings {
		if len(emb) != expectedDim {
			t.Errorf("Embedding %d: expected dimension %d, got %d", i, expectedDim, len(emb))
		}

		// Verify embedding is normalized (L2 norm should be ~1.0)
		norm := l2Norm(emb)
		if math.Abs(norm-1.0) > 0.01 {
			t.Errorf("Embedding %d: L2 norm %.4f is not close to 1.0", i, norm)
		}

		t.Logf("Text embedding %d: dim=%d, norm=%.4f, first3=[%.4f, %.4f, %.4f]",
			i, len(emb), norm, emb[0], emb[1], emb[2])
	}
}

// testSemanticSimilarity verifies that semantically similar texts have higher similarity scores
func testSemanticSimilarity(t *testing.T, ctx context.Context, c *client.TermiteClient, modelName string) {
	t.Helper()

	// Sentence pairs: (anchor, similar, dissimilar)
	anchor := "A man is eating pizza."
	similar := "A person is consuming a slice of pizza."
	dissimilar := "The stock market crashed yesterday."

	texts := []string{anchor, similar, dissimilar}
	embeddings, err := c.Embed(ctx, modelName, texts)
	if err != nil {
		t.Fatalf("Embedding failed: %v", err)
	}

	anchorEmb := embeddings[0]
	similarEmb := embeddings[1]
	dissimilarEmb := embeddings[2]

	// Calculate cosine similarities
	simToSimilar := cosineSim(anchorEmb, similarEmb)
	simToDissimilar := cosineSim(anchorEmb, dissimilarEmb)

	t.Logf("Similarity scores:")
	t.Logf("  Anchor <-> Similar: %.4f", simToSimilar)
	t.Logf("  Anchor <-> Dissimilar: %.4f", simToDissimilar)

	// The similar sentence should have higher similarity than the dissimilar one
	if simToSimilar <= simToDissimilar {
		t.Errorf("Expected similar sentence to have higher similarity (%.4f) than dissimilar sentence (%.4f)",
			simToSimilar, simToDissimilar)
	}

	// Similar sentences should have reasonably high similarity (> 0.5)
	if simToSimilar < 0.5 {
		t.Logf("Warning: Similarity between similar sentences is low: %.4f", simToSimilar)
	}
}

// testBatchEmbedding tests embedding multiple texts in a single batch
func testBatchEmbedding(t *testing.T, ctx context.Context, c *client.TermiteClient, modelName string, expectedDim int) {
	t.Helper()

	// Create a batch of 10 texts
	texts := []string{
		"The weather is sunny today.",
		"I enjoy reading books.",
		"Programming is fun and challenging.",
		"Music helps me concentrate.",
		"Coffee is my favorite drink.",
		"The mountain view is breathtaking.",
		"Learning new skills is important.",
		"Exercise improves mental health.",
		"Technology changes rapidly.",
		"Family time is precious.",
	}

	embeddings, err := c.Embed(ctx, modelName, texts)
	if err != nil {
		t.Fatalf("Batch embedding failed: %v", err)
	}

	if len(embeddings) != len(texts) {
		t.Errorf("Expected %d embeddings, got %d", len(texts), len(embeddings))
	}

	// Verify all embeddings have correct dimension
	for i, emb := range embeddings {
		if len(emb) != expectedDim {
			t.Errorf("Embedding %d: expected dimension %d, got %d", i, expectedDim, len(emb))
		}
	}

	t.Logf("Successfully embedded batch of %d texts with dimension %d", len(texts), expectedDim)
}

// l2Norm calculates the L2 norm of a vector
func l2Norm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

// cosineSim calculates cosine similarity between two vectors
func cosineSim(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
	}

	// Since embeddings are normalized, we can just return the dot product
	// (cosine = dot(a,b) / (|a| * |b|), and |a| = |b| = 1 for normalized vectors)
	return dotProduct
}
