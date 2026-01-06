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
	// REBEL model name (expected to be in models/recognizers/)
	rebelModelName = "Babelscape/rebel-large"
)

// TestREBELE2E tests the REBEL (relation extraction) pipeline:
// 1. Downloads REBEL model if not present (lazy download)
// 2. Starts termite server with REBEL model
// 3. Tests relation extraction via /api/recognize with relations capability
func TestREBELE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Ensure REBEL model is downloaded (lazy download)
	ensureRegistryModel(t, rebelModelName, ModelTypeRecognizer)

	modelsDir := getTestModelsDir()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)

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
		testListModelsREBEL(t, ctx, termiteClient)
	})

	t.Run("ExtractRelations", func(t *testing.T) {
		testExtractRelations(t, ctx, termiteClient)
	})

	t.Run("ExtractRelationsMultiple", func(t *testing.T) {
		testExtractRelationsMultiple(t, ctx, termiteClient)
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

// testListModelsREBEL verifies the REBEL model appears in the models list
func testListModelsREBEL(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	require.NoError(t, err, "ListModels failed")

	// Check that REBEL model is in the recognizers list
	foundRecognizer := false
	for _, name := range models.Recognizers {
		if name == rebelModelName {
			foundRecognizer = true
			break
		}
	}

	if !foundRecognizer {
		t.Errorf("REBEL model %s not found in recognizers: %v", rebelModelName, models.Recognizers)
	} else {
		t.Logf("Found REBEL model in recognizers")
	}

	// Check capabilities if model info is available
	if models.RecognizerInfo != nil {
		if info, ok := models.RecognizerInfo[rebelModelName]; ok {
			hasRelations := false
			for _, cap := range info.Capabilities {
				if cap == "relations" {
					hasRelations = true
					break
				}
			}
			assert.True(t, hasRelations, "REBEL model should have 'relations' capability")
			t.Logf("REBEL model capabilities: %v", info.Capabilities)
		}
	}
}

// testExtractRelations tests basic relation extraction
func testExtractRelations(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	texts := []string{
		"Barack Obama was born in Hawaii and served as the 44th President of the United States.",
	}

	resp, err := c.ExtractRelations(ctx, rebelModelName, texts, nil, nil)
	require.NoError(t, err, "ExtractRelations failed")

	assert.Equal(t, rebelModelName, resp.Model)
	assert.Len(t, resp.Entities, len(texts), "Should have entities for each input text")

	// Log entities found
	for i, textEntities := range resp.Entities {
		t.Logf("Text %d entities:", i)
		for _, entity := range textEntities {
			t.Logf("  Entity: %q (%s) at [%d:%d] score=%.2f",
				entity.Text, entity.Label, entity.Start, entity.End, entity.Score)
		}
	}

	// Check for relations
	if resp.Relations != nil && len(resp.Relations) > 0 {
		for i, textRelations := range resp.Relations {
			t.Logf("Text %d relations:", i)
			for _, rel := range textRelations {
				t.Logf("  Relation: %q -[%s]-> %q (score=%.2f)",
					rel.Head.Text, rel.Label, rel.Tail.Text, rel.Score)
			}
		}
		assert.NotEmpty(t, resp.Relations[0], "First text should have relations")
	} else {
		t.Log("No relations extracted (model may return entities only for this text)")
	}
}

// testExtractRelationsMultiple tests relation extraction with multiple texts
func testExtractRelationsMultiple(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	texts := []string{
		"Steve Jobs founded Apple Inc in Cupertino, California.",
		"Elon Musk is the CEO of Tesla and SpaceX.",
		"Albert Einstein was born in Ulm, Germany and later moved to the United States.",
	}

	resp, err := c.ExtractRelations(ctx, rebelModelName, texts, nil, nil)
	require.NoError(t, err, "ExtractRelations failed")

	assert.Equal(t, rebelModelName, resp.Model)
	assert.Len(t, resp.Entities, len(texts), "Should have entities for each input text")

	// Log all results
	for i := range texts {
		t.Logf("=== Text %d: %q ===", i, texts[i])

		if len(resp.Entities[i]) > 0 {
			t.Logf("  Entities:")
			for _, entity := range resp.Entities[i] {
				t.Logf("    - %q (%s) score=%.2f", entity.Text, entity.Label, entity.Score)
			}
		}

		if resp.Relations != nil && i < len(resp.Relations) && len(resp.Relations[i]) > 0 {
			t.Logf("  Relations:")
			for _, rel := range resp.Relations[i] {
				t.Logf("    - %q -[%s]-> %q score=%.2f",
					rel.Head.Text, rel.Label, rel.Tail.Text, rel.Score)
			}
		}
	}

	// At least one text should have extracted entities
	hasEntities := false
	for _, textEntities := range resp.Entities {
		if len(textEntities) > 0 {
			hasEntities = true
			break
		}
	}
	assert.True(t, hasEntities, "At least one text should have extracted entities")
}
