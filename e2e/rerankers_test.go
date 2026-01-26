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
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
)

// rerankerModelSpec defines the configuration for testing a reranker model.
type rerankerModelSpec struct {
	name       string // Human-readable name for test output
	hfRepo     string // HuggingFace repository
	localName  string // Local directory name after download
	modelName  string // Full model name used in API calls
	isLarge    bool   // If true, requires RUN_LARGE_MODEL_TESTS=true
}

// rerankerModels defines all reranker models to test.
var rerankerModels = []rerankerModelSpec{
	{
		name:      "mxbai-rerank-base-v1",
		hfRepo:    "mixedbread-ai/mxbai-rerank-base-v1",
		localName: "mxbai-rerank-base-v1",
		modelName: "mixedbread-ai/mxbai-rerank-base-v1",
	},
}

// TestRerankerModels runs E2E tests for all configured reranker models.
// Each model is tested as a subtest, allowing individual model tests to be run with:
//
//	go test -run TestRerankerModels/mxbai-rerank-base-v1 -tags="onnx,ORT" ./e2e/
//	go test -run TestRerankerModels/mxbai-rerank-base-v1 -tags="xla,XLA" ./e2e/
func TestRerankerModels(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E tests in short mode")
	}

	for _, model := range rerankerModels {
		model := model // capture range variable
		t.Run(model.name, func(t *testing.T) {
			if model.isLarge && os.Getenv("RUN_LARGE_MODEL_TESTS") != "true" {
				t.Skip("Skipping large model test; set RUN_LARGE_MODEL_TESTS=true to run")
			}
			runRerankerModelTest(t, model)
		})
	}
}

func runRerankerModelTest(t *testing.T, spec rerankerModelSpec) {
	// Ensure model is downloaded
	ensureHuggingFaceModel(t, spec.localName, spec.hfRepo, ModelTypeReranker)

	// Setup context and server
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	// Use the shared startTestServer which configures all model types
	termiteClient, serverCancel, serverDone := startTestServer(t, ctx, 60*time.Second)
	defer func() {
		serverCancel()
		<-serverDone
	}()

	// Run test suite
	t.Run("Rerank", func(t *testing.T) {
		testReranking(t, ctx, termiteClient, spec.modelName)
	})
}

func testReranking(t *testing.T, ctx context.Context, c *client.TermiteClient, modelName string) {
	query := "What is machine learning?"
	documents := []string{
		"Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
		"The weather today is sunny with a chance of rain.",
		"Deep learning uses neural networks to learn representations.",
		"Cooking pasta requires boiling water.",
	}

	t.Logf("Testing reranker: %s", modelName)
	scores, err := c.Rerank(ctx, modelName, query, documents)
	if err != nil {
		t.Fatalf("Rerank failed: %v", err)
	}

	if len(scores) != len(documents) {
		t.Fatalf("Expected %d scores, got %d", len(documents), len(scores))
	}

	t.Logf("Query: %s", query)
	for i, score := range scores {
		t.Logf("  [%d] Score: %.4f - %s", i, score, documents[i])
	}

	// Verify ML-related documents score higher than unrelated ones
	// Index 0: ML definition (should be high)
	// Index 1: Weather (should be low)
	// Index 2: Deep learning (should be moderate-high)
	// Index 3: Cooking (should be low)
	if scores[0] <= scores[1] {
		t.Errorf("Expected ML document (%.4f) to score higher than weather (%.4f)", scores[0], scores[1])
	}
	if scores[0] <= scores[3] {
		t.Errorf("Expected ML document (%.4f) to score higher than cooking (%.4f)", scores[0], scores[3])
	}
	if scores[2] <= scores[1] {
		t.Errorf("Expected deep learning document (%.4f) to score higher than weather (%.4f)", scores[2], scores[1])
	}

	t.Logf("Reranker %s test passed!", modelName)
}
