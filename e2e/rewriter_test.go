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
	// Rewriter model name (pulled from HuggingFace in TestMain)
	// Using a small T5-based model for question generation
	rewriterModelName = "lmqg/flan-t5-small-squad-qg"
)

// TestRewriteE2E tests the Rewriter (Seq2Seq text rewriting) pipeline:
// 1. Downloads Rewriter model if not present (lazy download)
// 2. Starts termite server with Rewriter model
// 3. Tests text rewriting (question generation from context)
func TestRewriteE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Ensure rewriter model is downloaded (lazy download)
	ensureRegistryModel(t, rewriterModelName, ModelTypeRewriter)

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
		testListModelsRewriter(t, ctx, termiteClient)
	})

	t.Run("RewriteText", func(t *testing.T) {
		testRewriteText(t, ctx, termiteClient)
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

// testListModelsRewriter verifies the Rewriter model appears in the models list
func testListModelsRewriter(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	require.NoError(t, err, "ListModels failed")

	// Check that Rewriter model is in the rewriters list
	foundRewriter := false
	for _, name := range models.Rewriters {
		if name == rewriterModelName {
			foundRewriter = true
			break
		}
	}

	if !foundRewriter {
		t.Errorf("Rewriter model %s not found in rewriters: %v",
			rewriterModelName, models.Rewriters)
	} else {
		t.Logf("Found Rewriter model in rewriters: %v", models.Rewriters)
	}
}

// testRewriteText tests text rewriting (question generation from context)
func testRewriteText(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	// LMQG models use a specific format: "generate question: <hl> answer <hl> context"
	inputs := []string{
		"generate question: <hl> Beyonce <hl> Beyonce starred as Etta James in Cadillac Records.",
		"generate question: <hl> 1955 <hl> Steve Jobs was born in 1955 in San Francisco.",
	}

	resp, err := c.RewriteText(ctx, rewriterModelName, inputs)
	require.NoError(t, err, "RewriteText failed")

	assert.Equal(t, rewriterModelName, resp.Model)
	assert.Len(t, resp.Texts, len(inputs), "Should have generated text for each input")

	// Log the rewritten texts
	for i, generated := range resp.Texts {
		t.Logf("Input %d generated texts:", i)
		for j, text := range generated {
			t.Logf("  Beam %d: %q", j, text)
		}
	}

	// Should have non-empty generated text
	for i, generated := range resp.Texts {
		assert.NotEmpty(t, generated, "Input %d should have generated text", i)
		if len(generated) > 0 {
			assert.NotEmpty(t, generated[0], "Input %d beam 0 should have text", i)
		}
	}
}
