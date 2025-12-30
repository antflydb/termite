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
	// Questionator model name (pulled from HuggingFace in TestMain)
	// Using a small T5-based model for question generation
	questionatorModelName = "flan-t5-small-squad-qg"
)

// TestQuestionateE2E tests the Questionator (Seq2Seq question generation) pipeline:
// 1. Starts termite server with Questionator model
// 2. Tests question generation from context
func TestQuestionateE2E(t *testing.T) {
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
		testListModelsQuestionator(t, ctx, termiteClient)
	})

	t.Run("GenerateQuestions", func(t *testing.T) {
		testGenerateQuestions(t, ctx, termiteClient)
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

// testListModelsQuestionator verifies the question generation model appears in the rewriters list
func testListModelsQuestionator(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	require.NoError(t, err, "ListModels failed")

	// Check that question generation model is in the rewriters list
	foundQuestionator := false
	for _, name := range models.Rewriters {
		if name == questionatorModelName {
			foundQuestionator = true
			break
		}
	}

	if !foundQuestionator {
		t.Errorf("Question generation model %s not found in rewriters: %v",
			questionatorModelName, models.Rewriters)
	} else {
		t.Logf("Found question generation model in rewriters: %v", models.Rewriters)
	}
}

// testGenerateQuestions tests question generation from context using the rewriter API
func testGenerateQuestions(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	// LMQG models use a specific format: "generate question: <hl> answer <hl> context"
	inputs := []string{
		"generate question: <hl> Beyonce <hl> Beyonce starred as Etta James in Cadillac Records.",
		"generate question: <hl> 1955 <hl> Steve Jobs was born in 1955 in San Francisco.",
	}

	resp, err := c.RewriteText(ctx, questionatorModelName, inputs)
	require.NoError(t, err, "RewriteText failed")

	assert.Equal(t, questionatorModelName, resp.Model)
	assert.Len(t, resp.Texts, len(inputs), "Should have generated text for each input")

	// Log the generated questions
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
