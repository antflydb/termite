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
	"os"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/termite"
	"go.uber.org/zap/zaptest"
)

// embeddingModelSpec defines the configuration for testing an embedding model.
type embeddingModelSpec struct {
	// Model identification
	name        string // Human-readable name for test output
	hfRepo      string // HuggingFace repository (e.g., "nomic-ai/nomic-embed-text-v1.5")
	localName   string // Local directory name after download
	modelName   string // Full model name used in API calls

	// Model properties
	embeddingDim int // Expected embedding dimension

	// Test configuration
	isLargeModel  bool // If true, requires RUN_LARGE_MODEL_TESTS=true
	isMultilingual bool // If true, run multilingual similarity tests

	// Timeouts (optional - defaults provided)
	testTimeout  time.Duration // Overall test timeout (default: 5m)
	readyTimeout time.Duration // Server ready timeout (default: 60s)
}

// embeddingModels defines all embedding models to test.
var embeddingModels = []embeddingModelSpec{
	{
		name:           "nomic-embed-text-v1.5",
		hfRepo:         "nomic-ai/nomic-embed-text-v1.5",
		localName:      "nomic-embed-text-v1.5",
		modelName:      "nomic-ai/nomic-embed-text-v1.5",
		embeddingDim:   768,
		isMultilingual: false,
	},
	{
		name:           "bge-m3",
		hfRepo:         "BAAI/bge-m3",
		localName:      "bge-m3",
		modelName:      "BAAI/bge-m3",
		embeddingDim:   1024,
		isMultilingual: true,
	},
	{
		name:           "gte-Qwen2-1.5B-instruct",
		hfRepo:         "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
		localName:      "gte-Qwen2-1.5B-instruct",
		modelName:      "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
		embeddingDim:   1536,
		isLargeModel:   true,
		isMultilingual: false,
		testTimeout:    10 * time.Minute,
		readyTimeout:   120 * time.Second,
	},
	{
		name:           "snowflake-arctic-embed-l-v2.0",
		hfRepo:         "Snowflake/snowflake-arctic-embed-l-v2.0",
		localName:      "snowflake-arctic-embed-l-v2.0",
		modelName:      "Snowflake/snowflake-arctic-embed-l-v2.0",
		embeddingDim:   1024,
		isMultilingual: false,
	},
	{
		name:           "stella_en_1.5B_v5",
		hfRepo:         "dunzhang/stella_en_1.5B_v5",
		localName:      "stella_en_1.5B_v5",
		modelName:      "dunzhang/stella_en_1.5B_v5",
		embeddingDim:   1024,
		isLargeModel:   true,
		isMultilingual: false,
		testTimeout:    10 * time.Minute,
		readyTimeout:   120 * time.Second,
	},
	{
		name:           "embeddinggemma-300m-ONNX",
		hfRepo:         "onnx-community/embeddinggemma-300m-ONNX",
		localName:      "embeddinggemma-300m-ONNX",
		modelName:      "onnx-community/embeddinggemma-300m-ONNX",
		embeddingDim:   768,
		isMultilingual: true,
	},
}

// TestEmbeddingModels runs E2E tests for all configured embedding models.
// Each model is tested as a subtest, allowing individual model tests to be run with:
//
//	go test -run TestEmbeddingModels/nomic-embed-text-v1.5 -tags="onnx,ORT" ./e2e/
//	go test -run TestEmbeddingModels/bge-m3 -tags="onnx,ORT" ./e2e/
func TestEmbeddingModels(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E tests in short mode")
	}

	for _, model := range embeddingModels {
		model := model // capture range variable
		t.Run(model.name, func(t *testing.T) {
			runEmbeddingModelTest(t, model)
		})
	}
}

// runEmbeddingModelTest executes the full E2E test suite for a single embedding model.
func runEmbeddingModelTest(t *testing.T, spec embeddingModelSpec) {
	// Skip large models unless explicitly enabled
	if spec.isLargeModel && os.Getenv("RUN_LARGE_MODEL_TESTS") != "true" {
		t.Skip("Skipping large model test. Set RUN_LARGE_MODEL_TESTS=true to run.")
	}

	// Apply default timeouts
	testTimeout := spec.testTimeout
	if testTimeout == 0 {
		testTimeout = 5 * time.Minute
	}
	readyTimeout := spec.readyTimeout
	if readyTimeout == 0 {
		readyTimeout = 60 * time.Second
	}

	// Ensure model is downloaded
	ensureHuggingFaceModel(t, spec.localName, spec.hfRepo, ModelTypeEmbedder)

	// Setup context and server
	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	termiteClient, serverCancel, serverDone := startTestServer(t, ctx, readyTimeout)
	defer func() {
		serverCancel()
		<-serverDone
	}()

	// Run test suite
	t.Run("ListModels", func(t *testing.T) {
		testListModelsContains(t, ctx, termiteClient, spec.modelName)
	})

	t.Run("TextEmbedding", func(t *testing.T) {
		testTextEmbeddingWithDim(t, ctx, termiteClient, spec.modelName, spec.embeddingDim)
	})

	t.Run("SemanticSimilarity", func(t *testing.T) {
		testSemanticSimilarity(t, ctx, termiteClient, spec.modelName)
	})

	if spec.isMultilingual {
		t.Run("MultilingualSimilarity", func(t *testing.T) {
			testMultilingualSimilarity(t, ctx, termiteClient, spec.modelName)
		})
	}

	t.Run("BatchEmbedding", func(t *testing.T) {
		testBatchEmbedding(t, ctx, termiteClient, spec.modelName, spec.embeddingDim)
	})
}

// startTestServer starts a Termite server for testing and returns a client,
// cancel function, and done channel. The caller should defer cleanup:
//
//	client, cancel, done := startTestServer(t, ctx, readyTimeout)
//	defer func() { cancel(); <-done }()
func startTestServer(t *testing.T, ctx context.Context, readyTimeout time.Duration) (*client.TermiteClient, context.CancelFunc, <-chan struct{}) {
	t.Helper()

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

	readyC := make(chan struct{})
	serverDone := make(chan struct{})

	go func() {
		defer close(serverDone)
		termite.RunAsTermite(serverCtx, logger, config, readyC)
	}()

	select {
	case <-readyC:
		t.Log("Server is ready")
	case <-time.After(readyTimeout):
		serverCancel()
		t.Fatalf("Timeout waiting for server to be ready after %v", readyTimeout)
	}

	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		serverCancel()
		t.Fatalf("Failed to create client: %v", err)
	}

	return termiteClient, serverCancel, serverDone
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
