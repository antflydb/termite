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
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/termite"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

const (
	// GLiNER model name (pulled from HuggingFace in TestMain)
	glinerModelName = "gliner_small-v2.1"
)

// TestGLiNERE2E tests the GLiNER (entity recognition) pipeline:
// 1. Starts termite server with GLiNER model
// 2. Tests entity recognition with default labels
// 3. Tests entity recognition with custom labels (zero-shot NER)
func TestGLiNERE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
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
	case <-time.After(120 * time.Second):
		t.Fatal("Timeout waiting for server to be ready")
	}

	// Create client
	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Run test cases
	t.Run("ListModels", func(t *testing.T) {
		testListModelsGLiNER(t, ctx, termiteClient)
	})

	t.Run("RecognizeEntities", func(t *testing.T) {
		testRecognizeEntities(t, ctx, termiteClient)
	})

	t.Run("RecognizeWithCustomLabels", func(t *testing.T) {
		testRecognizeWithCustomLabels(t, ctx, termiteClient)
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

// testListModelsGLiNER verifies the GLiNER model appears in the models list
func testListModelsGLiNER(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	require.NoError(t, err, "ListModels failed")

	// Check that GLiNER model is in the recognizers or extractors list
	foundRecognizer := false
	for _, name := range models.Recognizers {
		if name == glinerModelName {
			foundRecognizer = true
			break
		}
	}
	for _, name := range models.Extractors {
		if name == glinerModelName {
			foundRecognizer = true
			break
		}
	}

	if !foundRecognizer {
		t.Errorf("GLiNER model %s not found in recognizers: %v or extractors: %v",
			glinerModelName, models.Recognizers, models.Extractors)
	} else {
		t.Logf("Found GLiNER model in recognizers/extractors")
	}
}

// testRecognizeEntities tests entity recognition without custom labels
func testRecognizeEntities(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	texts := []string{
		"John Smith works at Google in New York.",
		"Apple Inc. was founded by Steve Jobs.",
	}

	resp, err := c.Recognize(ctx, glinerModelName, texts, nil)
	require.NoError(t, err, "Recognize failed")

	assert.Equal(t, glinerModelName, resp.Model)
	assert.Len(t, resp.Entities, len(texts), "Should have entities for each input text")

	// Log the entities found
	for i, textEntities := range resp.Entities {
		t.Logf("Text %d entities:", i)
		for _, entity := range textEntities {
			t.Logf("  - %q (%s) at [%d:%d] score=%.2f",
				entity.Text, entity.Label, entity.Start, entity.End, entity.Score)
		}
	}

	// First text should have entities (John Smith, Google, New York)
	assert.NotEmpty(t, resp.Entities[0], "First text should have entities")
}

// testRecognizeWithCustomLabels tests GLiNER's zero-shot capability with custom labels
func testRecognizeWithCustomLabels(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	texts := []string{
		"The iPhone 15 Pro is a great smartphone released in September 2023.",
		"Tesla Model Y is an electric vehicle manufactured by Tesla Inc.",
	}

	// Use custom labels for zero-shot NER
	labels := []string{"product", "company", "date", "vehicle"}

	resp, err := c.Recognize(ctx, glinerModelName, texts, labels)
	require.NoError(t, err, "Recognize with custom labels failed")

	assert.Equal(t, glinerModelName, resp.Model)
	assert.Len(t, resp.Entities, len(texts), "Should have entities for each input text")

	// Log the entities found
	for i, textEntities := range resp.Entities {
		t.Logf("Text %d entities (custom labels):", i)
		for _, entity := range textEntities {
			t.Logf("  - %q (%s) at [%d:%d] score=%.2f",
				entity.Text, entity.Label, entity.Start, entity.End, entity.Score)
		}
	}

	// Should find product and company entities
	assert.NotEmpty(t, resp.Entities[0], "First text should have product/company entities")
	assert.NotEmpty(t, resp.Entities[1], "Second text should have vehicle/company entities")

	// Verify custom labels are used
	for _, textEntities := range resp.Entities {
		for _, entity := range textEntities {
			found := false
			for _, label := range labels {
				if entity.Label == label {
					found = true
					break
				}
			}
			assert.True(t, found, "Entity label %q should be one of %v", entity.Label, labels)
		}
	}
}
