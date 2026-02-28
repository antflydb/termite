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
	"path/filepath"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
)

// sparseModelSpec defines the configuration for testing a sparse embedding model.
type sparseModelSpec struct {
	name      string // Human-readable name for test output
	localName string // Local directory path under models/embedders/ (e.g., "naver/splade-cocondenser-ensembledistil")
	modelName string // Full model name used in API calls

	// Test configuration
	testTimeout  time.Duration // Overall test timeout (default: 5m)
	readyTimeout time.Duration // Server ready timeout (default: 60s)
}

// sparseModels defines all sparse embedding models to test.
var sparseModels = []sparseModelSpec{
	{
		name:      "splade-cocondenser-ensembledistil",
		localName: "naver/splade-cocondenser-ensembledistil",
		modelName: "naver/splade-cocondenser-ensembledistil",
	},
}

// TestSparseEmbeddingModels runs E2E tests for all configured sparse embedding models.
// These tests require the model to be pre-exported to ONNX format using:
//
//	cd termite && ./scripts/export_model_to_registry.py embedder naver/splade-cocondenser-ensembledistil --capabilities sparse
//
// Each model is tested as a subtest:
//
//	go test -run TestSparseEmbeddingModels/splade-cocondenser-ensembledistil -tags="onnx,ORT" ./e2e/
func TestSparseEmbeddingModels(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E tests in short mode")
	}

	for _, model := range sparseModels {
		model := model // capture range variable
		t.Run(model.name, func(t *testing.T) {
			runSparseModelTest(t, model)
		})
	}
}

// runSparseModelTest executes the full E2E test suite for a single sparse embedding model.
func runSparseModelTest(t *testing.T, spec sparseModelSpec) {
	// Check if the model has been exported locally
	modelDir := filepath.Join(getTestModelsDir(), "embedders", spec.localName)
	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		t.Skipf("Sparse model not found at %s. Export it first:\n"+
			"  cd termite && ./scripts/export_model_to_registry.py embedder %s --capabilities sparse",
			modelDir, spec.modelName)
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
		testSparseListModels(t, ctx, termiteClient, spec.modelName)
	})

	t.Run("SparseEmbedding", func(t *testing.T) {
		testSparseEmbedding(t, ctx, termiteClient, spec.modelName)
	})

	t.Run("SparseSemanticSimilarity", func(t *testing.T) {
		testSparseSemanticSimilarity(t, ctx, termiteClient, spec.modelName)
	})

	t.Run("SparseBatch", func(t *testing.T) {
		testSparseBatch(t, ctx, termiteClient, spec.modelName)
	})
}

// testSparseListModels verifies the sparse model appears in the embedders list.
func testSparseListModels(t *testing.T, ctx context.Context, c *client.TermiteClient, modelName string) {
	t.Helper()

	models, err := c.ListModels(ctx)
	if err != nil {
		t.Fatalf("ListModels failed: %v", err)
	}

	if _, found := models.Embedders[modelName]; !found {
		t.Errorf("Model %s not found in embedders list", modelName)
	} else {
		t.Logf("Found sparse model in embedders: %s", modelName)
	}
}

// testSparseEmbedding tests sparse embedding of text strings and validates the output structure.
func testSparseEmbedding(t *testing.T, ctx context.Context, c *client.TermiteClient, modelName string) {
	t.Helper()

	texts := []string{
		"The quick brown fox jumps over the lazy dog.",
		"Machine learning is a subset of artificial intelligence.",
		"Natural language processing enables computers to understand text.",
	}

	vectors, err := c.SparseEmbed(ctx, modelName, texts)
	if err != nil {
		t.Fatalf("SparseEmbed failed: %v", err)
	}

	if len(vectors) != len(texts) {
		t.Fatalf("Expected %d sparse vectors, got %d", len(texts), len(vectors))
	}

	for i, vec := range vectors {
		// Must have some non-zero entries
		if len(vec.Indices) == 0 {
			t.Errorf("Vector %d: expected non-empty indices", i)
			continue
		}

		// Indices and values must have same length
		if len(vec.Indices) != len(vec.Values) {
			t.Errorf("Vector %d: indices length %d != values length %d",
				i, len(vec.Indices), len(vec.Values))
			continue
		}

		// Indices must be sorted ascending
		for j := 1; j < len(vec.Indices); j++ {
			if vec.Indices[j] <= vec.Indices[j-1] {
				t.Errorf("Vector %d: indices not sorted ascending at position %d: %v",
					i, j, vec.Indices[:min(10, len(vec.Indices))])
				break
			}
		}

		// All values must be positive (softplus output)
		for j, v := range vec.Values {
			if v <= 0 {
				t.Errorf("Vector %d: non-positive value %f at index %d", i, v, vec.Indices[j])
				break
			}
		}

		// Indices must be within reasonable vocab range (BERT vocab is ~30522)
		maxIndex := vec.Indices[len(vec.Indices)-1]
		if maxIndex > 100000 {
			t.Errorf("Vector %d: max index %d exceeds expected vocab range", i, maxIndex)
		}

		t.Logf("Vector %d: %d non-zero entries, max_index=%d, max_value=%.4f, min_value=%.4f",
			i, len(vec.Indices), maxIndex, vec.Values[0], vec.Values[len(vec.Values)-1])
	}
}

// testSparseSemanticSimilarity verifies that semantically similar texts produce more overlapping
// sparse representations than dissimilar texts, using sparse dot product.
func testSparseSemanticSimilarity(t *testing.T, ctx context.Context, c *client.TermiteClient, modelName string) {
	t.Helper()

	anchor := "A man is eating pizza."
	similar := "A person is consuming a slice of pizza."
	dissimilar := "The stock market crashed yesterday."

	texts := []string{anchor, similar, dissimilar}
	vectors, err := c.SparseEmbed(ctx, modelName, texts)
	if err != nil {
		t.Fatalf("SparseEmbed failed: %v", err)
	}

	anchorVec := vectors[0]
	similarVec := vectors[1]
	dissimilarVec := vectors[2]

	simToSimilar := sparseDotProduct(anchorVec, similarVec)
	simToDissimilar := sparseDotProduct(anchorVec, dissimilarVec)

	t.Logf("Sparse similarity scores:")
	t.Logf("  Anchor <-> Similar: %.4f", simToSimilar)
	t.Logf("  Anchor <-> Dissimilar: %.4f", simToDissimilar)

	if simToSimilar <= simToDissimilar {
		t.Errorf("Expected similar text to have higher sparse similarity (%.4f) than dissimilar text (%.4f)",
			simToSimilar, simToDissimilar)
	}
}

// testSparseBatch tests sparse embedding of multiple texts in a single batch.
func testSparseBatch(t *testing.T, ctx context.Context, c *client.TermiteClient, modelName string) {
	t.Helper()

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

	vectors, err := c.SparseEmbed(ctx, modelName, texts)
	if err != nil {
		t.Fatalf("Sparse batch embedding failed: %v", err)
	}

	if len(vectors) != len(texts) {
		t.Errorf("Expected %d sparse vectors, got %d", len(texts), len(vectors))
	}

	// Verify all vectors are non-empty and well-formed
	for i, vec := range vectors {
		if len(vec.Indices) == 0 {
			t.Errorf("Vector %d: expected non-empty indices", i)
		}
		if len(vec.Indices) != len(vec.Values) {
			t.Errorf("Vector %d: indices length %d != values length %d",
				i, len(vec.Indices), len(vec.Values))
		}
	}

	t.Logf("Successfully embedded batch of %d texts as sparse vectors", len(texts))
}

// sparseDotProduct calculates the dot product between two sparse vectors.
// Both vectors must have sorted indices (ascending).
func sparseDotProduct(a, b client.SparseVector) float64 {
	var sum float64
	i, j := 0, 0
	for i < len(a.Indices) && j < len(b.Indices) {
		if a.Indices[i] == b.Indices[j] {
			sum += float64(a.Values[i]) * float64(b.Values[j])
			i++
			j++
		} else if a.Indices[i] < b.Indices[j] {
			i++
		} else {
			j++
		}
	}
	return sum
}
