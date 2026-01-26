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
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/termite"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

const (
	// Paraphraser model name (pulled from model registry in TestMain)
	// Using PEGASUS fine-tuned for paraphrasing
	paraphraserModelName = "tuner007/pegasus_paraphrase"
)

// TestParaphraseE2E tests the PEGASUS paraphraser pipeline:
// 1. Downloads Paraphraser model if not present (lazy download)
// 2. Starts termite server with Paraphraser model
// 3. Tests paraphrasing functionality with multiple outputs
func TestParaphraseE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Ensure paraphraser model is downloaded (lazy download)
	ensureRegistryModel(t, paraphraserModelName, ModelTypeRewriter)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
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

	// Wait for server to be ready (PEGASUS is larger, may take longer)
	select {
	case <-readyC:
		t.Log("Server is ready")
	case <-time.After(180 * time.Second):
		t.Fatal("Timeout waiting for server to be ready")
	}

	// Create client
	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Run test cases
	t.Run("ListModels", func(t *testing.T) {
		testListModelsParaphraser(t, ctx, termiteClient)
	})

	t.Run("ParaphraseText", func(t *testing.T) {
		testParaphraseText(t, ctx, termiteClient)
	})

	t.Run("ParaphraseMultiple", func(t *testing.T) {
		testParaphraseMultiple(t, ctx, termiteClient)
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

// testListModelsParaphraser verifies the Paraphraser model appears in the models list
func testListModelsParaphraser(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	require.NoError(t, err, "ListModels failed")

	// Check that Paraphraser model is in the rewriters list
	foundParaphraser := false
	for _, name := range models.Rewriters {
		if name == paraphraserModelName {
			foundParaphraser = true
			break
		}
	}

	if !foundParaphraser {
		t.Errorf("Paraphraser model %s not found in rewriters: %v",
			paraphraserModelName, models.Rewriters)
	} else {
		t.Logf("Found Paraphraser model in rewriters: %v", models.Rewriters)
	}
}

// testParaphraseText tests basic paraphrasing functionality
func testParaphraseText(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	// PEGASUS paraphrase models take raw text as input
	inputs := []string{
		"The ultimate test of your knowledge is your capacity to convey it to another.",
	}

	resp, err := c.RewriteText(ctx, paraphraserModelName, inputs)
	require.NoError(t, err, "RewriteText failed")

	assert.Equal(t, paraphraserModelName, resp.Model)
	assert.Len(t, resp.Texts, len(inputs), "Should have generated text for each input")

	// Log the paraphrased texts
	for i, generated := range resp.Texts {
		t.Logf("Input %d paraphrases:", i)
		for j, text := range generated {
			t.Logf("  Variant %d: %q", j, text)
		}
	}

	// Should have non-empty generated text
	for i, generated := range resp.Texts {
		assert.NotEmpty(t, generated, "Input %d should have generated paraphrases", i)
		if len(generated) > 0 {
			assert.NotEmpty(t, generated[0], "Input %d variant 0 should have text", i)
			// The paraphrase should be different from the input
			assert.NotEqual(t, inputs[i], generated[0], "Paraphrase should differ from input")
		}
	}
}

// testParaphraseMultiple tests paraphrasing multiple inputs at once
func testParaphraseMultiple(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	inputs := []string{
		"Machine learning is a subset of artificial intelligence.",
		"The quick brown fox jumps over the lazy dog.",
		"Climate change is one of the greatest challenges of our time.",
	}

	resp, err := c.RewriteText(ctx, paraphraserModelName, inputs)
	require.NoError(t, err, "RewriteText failed for multiple inputs")

	assert.Equal(t, paraphraserModelName, resp.Model)
	assert.Len(t, resp.Texts, len(inputs), "Should have generated text for each input")

	// Log and validate each paraphrase
	for i, generated := range resp.Texts {
		t.Logf("Input %d: %q", i, inputs[i])
		assert.NotEmpty(t, generated, "Input %d should have generated paraphrases", i)

		for j, text := range generated {
			t.Logf("  Variant %d: %q", j, text)
			assert.NotEmpty(t, text, "Variant %d should not be empty", j)
		}
	}
}
