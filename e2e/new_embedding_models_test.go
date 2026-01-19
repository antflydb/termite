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

package e2e

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/termite"
	"go.uber.org/zap/zaptest"
)

// New embedding model specifications
const (
	// nomic-embed-text-v1.5: Matryoshka embeddings with task prefixes
	// Downloaded from HuggingFace: nomic-ai/nomic-embed-text-v1.5
	nomicHFRepo       = "nomic-ai/nomic-embed-text-v1.5"
	nomicLocalName    = "nomic-embed-text-v1.5"
	nomicModelName    = "nomic-ai/nomic-embed-text-v1.5"
	nomicEmbeddingDim = 768
	nomicMaxSeqLen    = 8192

	// bge-m3: Multilingual embeddings supporting 100+ languages
	// Downloaded from HuggingFace: BAAI/bge-m3
	bgeM3HFRepo       = "BAAI/bge-m3"
	bgeM3LocalName    = "bge-m3"
	bgeM3ModelName    = "BAAI/bge-m3"
	bgeM3EmbeddingDim = 1024
	bgeM3MaxSeqLen    = 8192

	// gte-Qwen2-1.5B-instruct: Instruction-following embeddings
	// Downloaded from HuggingFace: Alibaba-NLP/gte-Qwen2-1.5B-instruct
	gteQwenHFRepo       = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
	gteQwenLocalName    = "gte-Qwen2-1.5B-instruct"
	gteQwenModelName    = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
	gteQwenEmbeddingDim = 1536
	gteQwenMaxSeqLen    = 32768

	// snowflake-arctic-embed-l-v2.0: Retrieval-optimized with Matryoshka
	// Downloaded from HuggingFace: Snowflake/snowflake-arctic-embed-l-v2.0
	arcticHFRepo       = "Snowflake/snowflake-arctic-embed-l-v2.0"
	arcticLocalName    = "snowflake-arctic-embed-l-v2.0"
	arcticModelName    = "Snowflake/snowflake-arctic-embed-l-v2.0"
	arcticEmbeddingDim = 1024
	arcticMaxSeqLen    = 8192

	// stella_en_1.5B_v5: Premium English embeddings
	// Downloaded from HuggingFace: dunzhang/stella_en_1.5B_v5
	stellaHFRepo       = "dunzhang/stella_en_1.5B_v5"
	stellaLocalName    = "stella_en_1.5B_v5"
	stellaModelName    = "dunzhang/stella_en_1.5B_v5"
	stellaEmbeddingDim = 1024
	stellaMaxSeqLen    = 8192

	// gemma-embedding-308m: Compact multilingual embeddings
	// Downloaded from HuggingFace: google/gemma-embedding-308m
	gemmaEmbedHFRepo       = "google/gemma-embedding-308m"
	gemmaEmbedLocalName    = "gemma-embedding-308m"
	gemmaEmbedModelName    = "google/gemma-embedding-308m"
	gemmaEmbedEmbeddingDim = 2048
	gemmaEmbedMaxSeqLen    = 8192
)

// TestNomicEmbedTextV15E2E tests the nomic-embed-text-v1.5 embedding model.
//
// nomic-embed-text-v1.5 is a high-quality embedding model that:
// - Maps sentences to 768-dimensional dense vectors (Matryoshka: 64-768)
// - Supports up to 8192 tokens (long context)
// - Uses task prefixes for optimal retrieval (search_query, search_document)
// - Outperforms OpenAI text-embedding-3-small
func TestNomicEmbedTextV15E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Ensure model is downloaded from HuggingFace (lazy download)
	ensureHuggingFaceModel(t, nomicLocalName, nomicHFRepo, ModelTypeEmbedder)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)
	modelsDir := getTestModelsDir()
	t.Logf("Using models directory: %s", modelsDir)

	port := findAvailablePort(t)
	serverURL := fmt.Sprintf("http://localhost:%d", port)
	t.Logf("Starting server on %s", serverURL)

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

	select {
	case <-readyC:
		t.Log("Server is ready")
	case <-time.After(60 * time.Second):
		t.Fatal("Timeout waiting for server to be ready")
	}

	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	t.Run("ListModels", func(t *testing.T) {
		testListModelsContains(t, ctx, termiteClient, nomicModelName)
	})

	t.Run("TextEmbedding", func(t *testing.T) {
		testTextEmbeddingWithDim(t, ctx, termiteClient, nomicModelName, nomicEmbeddingDim)
	})

	t.Run("SemanticSimilarity", func(t *testing.T) {
		testSemanticSimilarity(t, ctx, termiteClient, nomicModelName)
	})

	t.Run("BatchEmbedding", func(t *testing.T) {
		testBatchEmbedding(t, ctx, termiteClient, nomicModelName, nomicEmbeddingDim)
	})

	t.Log("Shutting down server...")
	serverCancel()

	select {
	case <-serverDone:
		t.Log("Server shutdown complete")
	case <-time.After(30 * time.Second):
		t.Error("Timeout waiting for server shutdown")
	}
}

// TestBGEM3MultilingualE2E tests the BAAI/bge-m3 multilingual embedding model.
//
// bge-m3 is a state-of-the-art multilingual model that:
// - Maps sentences to 1024-dimensional dense vectors
// - Supports 100+ languages
// - Supports up to 8192 tokens
// - Achieves best-in-class results on cross-lingual retrieval benchmarks
func TestBGEM3MultilingualE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	ensureHuggingFaceModel(t, bgeM3LocalName, bgeM3HFRepo, ModelTypeEmbedder)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)
	modelsDir := getTestModelsDir()
	t.Logf("Using models directory: %s", modelsDir)

	port := findAvailablePort(t)
	serverURL := fmt.Sprintf("http://localhost:%d", port)
	t.Logf("Starting server on %s", serverURL)

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

	select {
	case <-readyC:
		t.Log("Server is ready")
	case <-time.After(60 * time.Second):
		t.Fatal("Timeout waiting for server to be ready")
	}

	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	t.Run("ListModels", func(t *testing.T) {
		testListModelsContains(t, ctx, termiteClient, bgeM3ModelName)
	})

	t.Run("TextEmbedding", func(t *testing.T) {
		testTextEmbeddingWithDim(t, ctx, termiteClient, bgeM3ModelName, bgeM3EmbeddingDim)
	})

	t.Run("SemanticSimilarity", func(t *testing.T) {
		testSemanticSimilarity(t, ctx, termiteClient, bgeM3ModelName)
	})

	t.Run("MultilingualSimilarity", func(t *testing.T) {
		testMultilingualSimilarity(t, ctx, termiteClient, bgeM3ModelName)
	})

	t.Run("BatchEmbedding", func(t *testing.T) {
		testBatchEmbedding(t, ctx, termiteClient, bgeM3ModelName, bgeM3EmbeddingDim)
	})

	t.Log("Shutting down server...")
	serverCancel()

	select {
	case <-serverDone:
		t.Log("Server shutdown complete")
	case <-time.After(30 * time.Second):
		t.Error("Timeout waiting for server shutdown")
	}
}

// TestGTEQwen2InstructE2E tests the Alibaba-NLP/gte-Qwen2-1.5B-instruct model.
//
// gte-Qwen2-1.5B-instruct is an instruction-following embedding model that:
// - Maps sentences to 1536-dimensional dense vectors
// - Supports up to 32768 tokens (very long context)
// - Uses instruction-following for better task-specific embeddings
// - Best for long document retrieval
//
// Note: This test is skipped by default due to large model size (~6GB).
// Run with: go test -run TestGTEQwen2InstructE2E -tags="onnx,ORT" ./e2e/
func TestGTEQwen2InstructE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Skip by default due to large model size
	if os.Getenv("RUN_LARGE_MODEL_TESTS") != "true" {
		t.Skip("Skipping large model test. Set RUN_LARGE_MODEL_TESTS=true to run.")
	}

	ensureHuggingFaceModel(t, gteQwenLocalName, gteQwenHFRepo, ModelTypeEmbedder)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)
	modelsDir := getTestModelsDir()

	port := findAvailablePort(t)
	serverURL := fmt.Sprintf("http://localhost:%d", port)

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

	select {
	case <-readyC:
		t.Log("Server is ready")
	case <-time.After(120 * time.Second):
		t.Fatal("Timeout waiting for server to be ready")
	}

	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	t.Run("TextEmbedding", func(t *testing.T) {
		testTextEmbeddingWithDim(t, ctx, termiteClient, gteQwenModelName, gteQwenEmbeddingDim)
	})

	t.Run("SemanticSimilarity", func(t *testing.T) {
		testSemanticSimilarity(t, ctx, termiteClient, gteQwenModelName)
	})

	serverCancel()
	<-serverDone
}

// TestSnowflakeArcticEmbedE2E tests the Snowflake/snowflake-arctic-embed-l-v2.0 model.
//
// snowflake-arctic-embed-l-v2.0 is a retrieval-optimized model that:
// - Maps sentences to 1024-dimensional dense vectors (Matryoshka: 256-1024)
// - Supports up to 8192 tokens
// - Optimized specifically for retrieval tasks
func TestSnowflakeArcticEmbedE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	ensureHuggingFaceModel(t, arcticLocalName, arcticHFRepo, ModelTypeEmbedder)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)
	modelsDir := getTestModelsDir()

	port := findAvailablePort(t)
	serverURL := fmt.Sprintf("http://localhost:%d", port)

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

	select {
	case <-readyC:
		t.Log("Server is ready")
	case <-time.After(60 * time.Second):
		t.Fatal("Timeout waiting for server to be ready")
	}

	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	t.Run("ListModels", func(t *testing.T) {
		testListModelsContains(t, ctx, termiteClient, arcticModelName)
	})

	t.Run("TextEmbedding", func(t *testing.T) {
		testTextEmbeddingWithDim(t, ctx, termiteClient, arcticModelName, arcticEmbeddingDim)
	})

	t.Run("SemanticSimilarity", func(t *testing.T) {
		testSemanticSimilarity(t, ctx, termiteClient, arcticModelName)
	})

	t.Run("BatchEmbedding", func(t *testing.T) {
		testBatchEmbedding(t, ctx, termiteClient, arcticModelName, arcticEmbeddingDim)
	})

	serverCancel()
	<-serverDone
}

// TestStellaEnglishE2E tests the dunzhang/stella_en_1.5B_v5 model.
//
// stella_en_1.5B_v5 is a premium English embedding model that:
// - Maps sentences to 1024-dimensional dense vectors (Matryoshka support)
// - Supports up to 8192 tokens
// - Top-tier MTEB scores for English
//
// Note: This test is skipped by default due to large model size (~6GB).
func TestStellaEnglishE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Skip by default due to large model size
	if os.Getenv("RUN_LARGE_MODEL_TESTS") != "true" {
		t.Skip("Skipping large model test. Set RUN_LARGE_MODEL_TESTS=true to run.")
	}

	ensureHuggingFaceModel(t, stellaLocalName, stellaHFRepo, ModelTypeEmbedder)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)
	modelsDir := getTestModelsDir()

	port := findAvailablePort(t)
	serverURL := fmt.Sprintf("http://localhost:%d", port)

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

	select {
	case <-readyC:
		t.Log("Server is ready")
	case <-time.After(120 * time.Second):
		t.Fatal("Timeout waiting for server to be ready")
	}

	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	t.Run("TextEmbedding", func(t *testing.T) {
		testTextEmbeddingWithDim(t, ctx, termiteClient, stellaModelName, stellaEmbeddingDim)
	})

	t.Run("SemanticSimilarity", func(t *testing.T) {
		testSemanticSimilarity(t, ctx, termiteClient, stellaModelName)
	})

	serverCancel()
	<-serverDone
}

// TestGemmaEmbedding308ME2E tests the google/gemma-embedding-308m model.
//
// gemma-embedding-308m is a compact multilingual model that:
// - Maps sentences to 2048-dimensional dense vectors (Matryoshka: 128-2048)
// - Supports 100+ languages
// - Optimized for edge deployment (<200MB with quantization)
// - #1 on MTEB multilingual leaderboard for models under 500M params
func TestGemmaEmbedding308ME2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	ensureHuggingFaceModel(t, gemmaEmbedLocalName, gemmaEmbedHFRepo, ModelTypeEmbedder)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)
	modelsDir := getTestModelsDir()

	port := findAvailablePort(t)
	serverURL := fmt.Sprintf("http://localhost:%d", port)

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

	select {
	case <-readyC:
		t.Log("Server is ready")
	case <-time.After(60 * time.Second):
		t.Fatal("Timeout waiting for server to be ready")
	}

	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	t.Run("ListModels", func(t *testing.T) {
		testListModelsContains(t, ctx, termiteClient, gemmaEmbedModelName)
	})

	t.Run("TextEmbedding", func(t *testing.T) {
		testTextEmbeddingWithDim(t, ctx, termiteClient, gemmaEmbedModelName, gemmaEmbedEmbeddingDim)
	})

	t.Run("SemanticSimilarity", func(t *testing.T) {
		testSemanticSimilarity(t, ctx, termiteClient, gemmaEmbedModelName)
	})

	t.Run("MultilingualSimilarity", func(t *testing.T) {
		testMultilingualSimilarity(t, ctx, termiteClient, gemmaEmbedModelName)
	})

	t.Run("BatchEmbedding", func(t *testing.T) {
		testBatchEmbedding(t, ctx, termiteClient, gemmaEmbedModelName, gemmaEmbedEmbeddingDim)
	})

	serverCancel()
	<-serverDone
}

// testMultilingualSimilarity verifies that the model can find similarity across languages.
// Tests that semantically equivalent sentences in different languages have high similarity.
func testMultilingualSimilarity(t *testing.T, ctx context.Context, c *client.TermiteClient, modelName string) {
	t.Helper()

	// Same concept in multiple languages
	englishText := "Machine learning is a subset of artificial intelligence."
	chineseText := "机器学习是人工智能的一个子集。"
	frenchText := "L'apprentissage automatique est un sous-ensemble de l'intelligence artificielle."
	germanText := "Maschinelles Lernen ist eine Teilmenge der künstlichen Intelligenz."

	// Unrelated text for comparison
	unrelatedText := "The weather is sunny and warm today."

	texts := []string{englishText, chineseText, frenchText, germanText, unrelatedText}
	embeddings, err := c.Embed(ctx, modelName, texts)
	if err != nil {
		t.Fatalf("Embedding failed: %v", err)
	}

	englishEmb := embeddings[0]
	chineseEmb := embeddings[1]
	frenchEmb := embeddings[2]
	germanEmb := embeddings[3]
	unrelatedEmb := embeddings[4]

	// Calculate cross-lingual similarities
	enZhSim := cosineSimilarity(englishEmb, chineseEmb)
	enFrSim := cosineSimilarity(englishEmb, frenchEmb)
	enDeSim := cosineSimilarity(englishEmb, germanEmb)
	enUnrelatedSim := cosineSimilarity(englishEmb, unrelatedEmb)

	t.Logf("Cross-lingual similarity scores:")
	t.Logf("  English <-> Chinese: %.4f", enZhSim)
	t.Logf("  English <-> French: %.4f", enFrSim)
	t.Logf("  English <-> German: %.4f", enDeSim)
	t.Logf("  English <-> Unrelated: %.4f", enUnrelatedSim)

	// Cross-lingual similarities should be higher than unrelated text
	if enZhSim <= enUnrelatedSim {
		t.Errorf("Expected English-Chinese similarity (%.4f) > unrelated (%.4f)", enZhSim, enUnrelatedSim)
	}
	if enFrSim <= enUnrelatedSim {
		t.Errorf("Expected English-French similarity (%.4f) > unrelated (%.4f)", enFrSim, enUnrelatedSim)
	}
	if enDeSim <= enUnrelatedSim {
		t.Errorf("Expected English-German similarity (%.4f) > unrelated (%.4f)", enDeSim, enUnrelatedSim)
	}

	// Cross-lingual pairs should have reasonable similarity (> 0.5 for good multilingual models)
	minExpectedSim := 0.5
	if enZhSim < minExpectedSim {
		t.Logf("Warning: English-Chinese similarity %.4f is below expected %.4f", enZhSim, minExpectedSim)
	}
}

// cosineSimilarity calculates cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float64 {
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

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
