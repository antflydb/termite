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
	"path/filepath"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/termite"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

const (
	// REBEL model name (expected in models_dir/relators/)
	rebelModelName = "rebel-large"
)

// findRebelModel looks for the REBEL model in possible locations
func findRebelModel() (modelsDir string, modelName string, found bool) {
	// Check test models directory first
	testDir := getTestModelsDir()
	relatorsDir := filepath.Join(testDir, "relators")

	// Check for rebel-large in relators/
	if _, err := os.Stat(filepath.Join(relatorsDir, "rebel-large")); err == nil {
		return testDir, "rebel-large", true
	}

	// Check ~/.termite/models for relators
	homeDir, err := os.UserHomeDir()
	if err == nil {
		termiteModels := filepath.Join(homeDir, ".termite", "models")

		// Check relators/ subdirectory
		if _, err := os.Stat(filepath.Join(termiteModels, "relators", "rebel-large")); err == nil {
			return termiteModels, "rebel-large", true
		}

		// Check rel/ subdirectory (legacy location)
		if _, err := os.Stat(filepath.Join(termiteModels, "rel", "rebel-large")); err == nil {
			// Need to move/symlink to relators/ or use rel/ as models dir
			// For now, use the parent and assume the server can find it
			return termiteModels, "rebel-large", true
		}

		// Check if there's any model in rel/ directory
		relDir := filepath.Join(termiteModels, "rel")
		if entries, err := os.ReadDir(relDir); err == nil && len(entries) > 0 {
			// Use the first model found in rel/
			for _, entry := range entries {
				if entry.IsDir() {
					// Return parent dir - server needs relators/ subdirectory
					// We'll need to handle this in test setup
					return termiteModels, entry.Name(), true
				}
			}
		}
	}

	return "", "", false
}

// TestREBELE2E tests the REBEL (relation extraction) pipeline:
// 1. Starts termite server with REBEL model
// 2. Tests relation extraction from text
// 3. Validates extracted triplets (subject, relation, object)
func TestREBELE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Find REBEL model
	modelsDir, modelName, found := findRebelModel()
	if !found {
		t.Skip("REBEL model not found. Set TERMITE_MODELS_DIR or place model in ~/.termite/models/relators/")
	}

	t.Logf("Using REBEL model: %s from %s", modelName, modelsDir)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)

	// Check if we need to create a symlink for legacy 'rel' directory
	homeDir, _ := os.UserHomeDir()
	legacyRelDir := filepath.Join(homeDir, ".termite", "models", "rel")
	relatorsDir := filepath.Join(modelsDir, "relators")

	// If model is in rel/ but not relators/, create relators/ symlink or copy
	if _, err := os.Stat(legacyRelDir); err == nil {
		if _, err := os.Stat(relatorsDir); os.IsNotExist(err) {
			// Create relators directory and symlink the model
			if err := os.MkdirAll(relatorsDir, 0755); err != nil {
				t.Fatalf("Failed to create relators directory: %v", err)
			}
			// Symlink each model from rel/ to relators/
			entries, _ := os.ReadDir(legacyRelDir)
			for _, entry := range entries {
				if entry.IsDir() {
					src := filepath.Join(legacyRelDir, entry.Name())
					dst := filepath.Join(relatorsDir, entry.Name())
					if _, err := os.Stat(dst); os.IsNotExist(err) {
						if err := os.Symlink(src, dst); err != nil {
							t.Logf("Warning: Could not symlink %s to %s: %v", src, dst, err)
						} else {
							t.Logf("Created symlink: %s -> %s", dst, src)
						}
					}
				}
			}
		}
	}

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
		testListModelsREBEL(t, ctx, termiteClient, modelName)
	})

	t.Run("ExtractRelations", func(t *testing.T) {
		testExtractRelations(t, ctx, termiteClient, modelName)
	})

	t.Run("ExtractRelationsMultipleTexts", func(t *testing.T) {
		testExtractRelationsMultiple(t, ctx, termiteClient, modelName)
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
func testListModelsREBEL(t *testing.T, ctx context.Context, c *client.TermiteClient, modelName string) {
	t.Helper()

	models, err := c.ListModels(ctx)
	require.NoError(t, err, "ListModels failed")

	// Check that REBEL model is in the relators list
	foundRelator := false
	for _, name := range models.Relators {
		t.Logf("Found relator model: %s", name)
		if name == modelName {
			foundRelator = true
		}
	}

	if !foundRelator {
		t.Errorf("REBEL model %s not found in relators: %v", modelName, models.Relators)
	} else {
		t.Logf("Found REBEL model %s in relators list", modelName)
	}
}

// testExtractRelations tests relation extraction from a single text
func testExtractRelations(t *testing.T, ctx context.Context, c *client.TermiteClient, modelName string) {
	t.Helper()

	texts := []string{
		"Albert Einstein was born in Ulm and developed the theory of relativity.",
	}

	resp, err := c.Relate(ctx, modelName, texts)
	require.NoError(t, err, "Relate failed")

	assert.Equal(t, modelName, resp.Model)
	assert.Len(t, resp.Relations, len(texts), "Should have relations for each input text")

	// Log the relations found
	for i, textRelations := range resp.Relations {
		t.Logf("Text %d relations:", i)
		for _, rel := range textRelations {
			t.Logf("  - (%s) -[%s]-> (%s) score=%.2f",
				rel.Subject, rel.Relation, rel.Object, rel.Score)
		}
	}

	// Should find at least some relations
	assert.NotEmpty(t, resp.Relations[0], "Should extract at least one relation from text about Einstein")

	// Validate relation structure
	for _, textRelations := range resp.Relations {
		for _, rel := range textRelations {
			assert.NotEmpty(t, rel.Subject, "Relation subject should not be empty")
			assert.NotEmpty(t, rel.Relation, "Relation type should not be empty")
			assert.NotEmpty(t, rel.Object, "Relation object should not be empty")
		}
	}
}

// testExtractRelationsMultiple tests relation extraction from multiple texts
func testExtractRelationsMultiple(t *testing.T, ctx context.Context, c *client.TermiteClient, modelName string) {
	t.Helper()

	texts := []string{
		"Apple Inc. was founded by Steve Jobs in Cupertino.",
		"The Eiffel Tower is located in Paris, France.",
		"Barack Obama was the 44th president of the United States.",
	}

	resp, err := c.Relate(ctx, modelName, texts)
	require.NoError(t, err, "Relate failed for multiple texts")

	assert.Equal(t, modelName, resp.Model)
	assert.Len(t, resp.Relations, len(texts), "Should have relations array for each input text")

	// Log all extracted relations
	totalRelations := 0
	for i, textRelations := range resp.Relations {
		t.Logf("Text %d: %q", i, texts[i])
		for _, rel := range textRelations {
			t.Logf("  -> (%s) -[%s]-> (%s)", rel.Subject, rel.Relation, rel.Object)
			totalRelations++
		}
	}

	t.Logf("Total relations extracted: %d from %d texts", totalRelations, len(texts))

	// Each text should have at least one relation
	for i, textRelations := range resp.Relations {
		assert.NotEmpty(t, textRelations, "Text %d should have at least one relation", i)
	}
}
