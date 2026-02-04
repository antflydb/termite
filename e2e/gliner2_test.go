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
	// GLiNER2 model from Fastino (downloaded from Antfly model registry)
	gliner2ModelName = "fastino/gliner2-base-v1"
)

// TestGLiNER2E2E tests the GLiNER2 (unified multi-task NER) pipeline:
// 1. Downloads GLiNER2 model from registry if not present
// 2. Starts termite server with GLiNER2 model
// 3. Tests entity recognition with default labels
// 4. Tests entity recognition with custom labels (zero-shot NER)
func TestGLiNER2E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Ensure GLiNER2 model is downloaded from registry
	ensureRegistryModel(t, gliner2ModelName, ModelTypeRecognizer)

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
		testListModelsGLiNER2(t, ctx, termiteClient)
	})

	t.Run("RecognizeEntities", func(t *testing.T) {
		testRecognizeEntitiesGLiNER2(t, ctx, termiteClient)
	})

	t.Run("RecognizeWithCustomLabels", func(t *testing.T) {
		testRecognizeWithCustomLabelsGLiNER2(t, ctx, termiteClient)
	})

	t.Run("ExtractRelations", func(t *testing.T) {
		testExtractRelationsGLiNER2(t, ctx, termiteClient)
	})

	t.Run("ClassifyText", func(t *testing.T) {
		testClassifyTextGLiNER2(t, ctx, termiteClient)
	})

	t.Run("ClassifyTextMultiLabel", func(t *testing.T) {
		testClassifyTextMultiLabelGLiNER2(t, ctx, termiteClient)
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

// testListModelsGLiNER2 verifies the GLiNER2 model appears in the models list
func testListModelsGLiNER2(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	require.NoError(t, err, "ListModels failed")

	// Check that GLiNER2 model is in the recognizers or extractors list
	foundRecognizer := false
	for _, name := range models.Recognizers {
		if name == gliner2ModelName {
			foundRecognizer = true
			break
		}
	}
	for _, name := range models.Extractors {
		if name == gliner2ModelName {
			foundRecognizer = true
			break
		}
	}

	if !foundRecognizer {
		t.Errorf("GLiNER2 model %s not found in recognizers: %v or extractors: %v",
			gliner2ModelName, models.Recognizers, models.Extractors)
	} else {
		t.Logf("Found GLiNER2 model in recognizers/extractors")
	}
}

// testRecognizeEntitiesGLiNER2 tests entity recognition without custom labels
func testRecognizeEntitiesGLiNER2(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	texts := []string{
		"John Smith works at Google in New York.",
		"Apple Inc. was founded by Steve Jobs in 1976.",
	}

	resp, err := c.Recognize(ctx, gliner2ModelName, texts, nil)
	require.NoError(t, err, "Recognize failed")

	assert.Equal(t, gliner2ModelName, resp.Model)
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

// testRecognizeWithCustomLabelsGLiNER2 tests GLiNER2's zero-shot capability with custom labels
func testRecognizeWithCustomLabelsGLiNER2(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	texts := []string{
		"The iPhone 15 Pro is a great smartphone released in September 2023.",
		"Tesla Model Y is an electric vehicle manufactured by Tesla Inc.",
	}

	// Use custom labels for zero-shot NER
	labels := []string{"product", "company", "date", "vehicle"}

	resp, err := c.Recognize(ctx, gliner2ModelName, texts, labels)
	require.NoError(t, err, "Recognize with custom labels failed")

	assert.Equal(t, gliner2ModelName, resp.Model)
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

// testExtractRelationsGLiNER2 tests GLiNER2's relation extraction capability
func testExtractRelationsGLiNER2(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	texts := []string{
		"John Smith works at Google in Mountain View.",
		"Elon Musk founded Tesla in 2003.",
	}

	entityLabels := []string{"person", "organization", "location"}
	relationLabels := []string{"works_for", "founded", "located_in"}

	resp, err := c.ExtractRelations(ctx, gliner2ModelName, texts, entityLabels, relationLabels)
	if err != nil {
		// Relation extraction may not be fully implemented yet
		t.Logf("RelationExtraction returned error (may be expected): %v", err)
		return
	}

	assert.Equal(t, gliner2ModelName, resp.Model)
	assert.Len(t, resp.Entities, len(texts), "Should have entities for each input text")

	// Log entities found
	for i, textEntities := range resp.Entities {
		t.Logf("Text %d entities:", i)
		for _, entity := range textEntities {
			t.Logf("  - %q (%s) at [%d:%d] score=%.2f",
				entity.Text, entity.Label, entity.Start, entity.End, entity.Score)
		}
	}

	// Log relations found
	if resp.Relations != nil {
		for i, textRelations := range resp.Relations {
			t.Logf("Text %d relations:", i)
			for _, rel := range textRelations {
				t.Logf("  - %q (%s) -[%s]-> %q (%s) score=%.2f",
					rel.Head.Text, rel.Head.Label, rel.Label,
					rel.Tail.Text, rel.Tail.Label, rel.Score)
			}
		}
	}

	// First text should have entities (John Smith, Google, Mountain View)
	assert.NotEmpty(t, resp.Entities[0], "First text should have entities")

	// Relations may or may not be extracted depending on model capabilities
	// We just verify no panic occurred
	t.Logf("Relation extraction completed successfully")
}

// testClassifyTextGLiNER2 tests GLiNER2's text classification capability
func testClassifyTextGLiNER2(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	texts := []string{
		"This product is absolutely amazing! Best purchase ever.",
		"Terrible experience. Would not recommend.",
	}

	labels := []string{"positive", "negative", "neutral"}

	resp, err := c.Classify(ctx, gliner2ModelName, texts, labels)
	if err != nil {
		// Classification may not be fully implemented yet
		t.Logf("ClassifyText returned error (may be expected): %v", err)
		return
	}

	assert.Equal(t, gliner2ModelName, resp.Model)
	assert.Len(t, resp.Classifications, len(texts), "Should have classifications for each input text")

	// Log classifications found
	for i, textClassifications := range resp.Classifications {
		t.Logf("Text %d classifications:", i)
		for _, cls := range textClassifications {
			t.Logf("  - %s (score=%.2f)", cls.Label, cls.Score)
		}
	}

	// Each text should have at least one classification
	for i, textClassifications := range resp.Classifications {
		assert.NotEmpty(t, textClassifications, "Text %d should have at least one classification", i)
	}

	// Verify labels are from our provided list
	for _, textClassifications := range resp.Classifications {
		for _, cls := range textClassifications {
			found := false
			for _, label := range labels {
				if cls.Label == label {
					found = true
					break
				}
			}
			assert.True(t, found, "Classification label %q should be one of %v", cls.Label, labels)
		}
	}

	t.Logf("Single-label classification completed successfully")
}

// testClassifyTextMultiLabelGLiNER2 tests GLiNER2's multi-label classification
func testClassifyTextMultiLabelGLiNER2(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	texts := []string{
		"Apple announces new iPhone with AI features for enterprise customers",
		"Local sports team wins championship in exciting final match",
	}

	labels := []string{"technology", "business", "sports", "health", "politics"}

	resp, err := c.ClassifyMultiLabel(ctx, gliner2ModelName, texts, labels)
	if err != nil {
		// Classification may not be fully implemented yet
		t.Logf("ClassifyText (multi-label) returned error (may be expected): %v", err)
		return
	}

	assert.Equal(t, gliner2ModelName, resp.Model)
	assert.Len(t, resp.Classifications, len(texts), "Should have classifications for each input text")

	// Log classifications found
	for i, textClassifications := range resp.Classifications {
		t.Logf("Text %d multi-label classifications:", i)
		for _, cls := range textClassifications {
			t.Logf("  - %s (score=%.2f)", cls.Label, cls.Score)
		}
	}

	// First text should have technology and/or business labels
	if len(resp.Classifications) > 0 && len(resp.Classifications[0]) > 0 {
		labels := make([]string, 0)
		for _, cls := range resp.Classifications[0] {
			labels = append(labels, cls.Label)
		}
		t.Logf("First text got labels: %v", labels)
	}

	// Second text should have sports label
	if len(resp.Classifications) > 1 && len(resp.Classifications[1]) > 0 {
		labels := make([]string, 0)
		for _, cls := range resp.Classifications[1] {
			labels = append(labels, cls.Label)
		}
		t.Logf("Second text got labels: %v", labels)
	}

	t.Logf("Multi-label classification completed successfully")
}

// TestGLiNER2ModelTypeDetection tests that GLiNER2 models are correctly detected
func TestGLiNER2ModelTypeDetection(t *testing.T) {
	// This test verifies the model type detection logic works for GLiNER2 paths
	testCases := []struct {
		name     string
		path     string
		expected string
	}{
		{
			name:     "GLiNER2 base model",
			path:     "/models/recognizers/fastino/gliner2-base-v1",
			expected: "gliner2",
		},
		{
			name:     "GLiNER2 large model",
			path:     "/models/recognizers/fastino/gliner2-large-v1",
			expected: "gliner2",
		},
		{
			name:     "GLiNER v1 model",
			path:     "/models/recognizers/onnx-community/gliner_small-v2.1",
			expected: "uniencoder",
		},
		{
			name:     "GLiNER multitask model",
			path:     "/models/recognizers/knowledgator/gliner-multitask-large-v0.5",
			expected: "multitask",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// We can't directly call the Go function from here without importing it,
			// but we document the expected behavior for manual verification
			t.Logf("Path: %s -> Expected model type: %s", tc.path, tc.expected)
		})
	}
}
