//go:build onnx && ORT

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
	"strconv"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/termite"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

// T5Gemma-2 model name using owner/model format
const t5Gemma2ModelName = "google/t5gemma-2-270m"

// T5Gemma-2 hidden size (640). The actual embedding dimension may vary
// based on sequence length as the encoder returns token-level embeddings.
const t5Gemma2HiddenSize = 640

// getT5Gemma2ModelsDir returns the models directory, preferring ~/.termite/models
// if TERMITE_MODELS_DIR is not set and the model exists there.
func getT5Gemma2ModelsDir(t *testing.T) string {
	// First check TERMITE_MODELS_DIR
	if dir := os.Getenv("TERMITE_MODELS_DIR"); dir != "" {
		return dir
	}

	// Check test models dir from harness
	testDir := getTestModelsDir()
	if testDir != "" {
		return testDir
	}

	// Fall back to ~/.termite/models
	homeDir, err := os.UserHomeDir()
	if err != nil {
		t.Logf("Could not get home directory: %v", err)
		return ""
	}
	return filepath.Join(homeDir, ".termite", "models")
}

// TestT5Gemma2EmbedderE2E tests the T5Gemma-2 embedder end-to-end.
// This test requires the model to be downloaded to the models directory.
func TestT5Gemma2EmbedderE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Check if model exists
	modelsDir := getT5Gemma2ModelsDir(t)
	if modelsDir == "" {
		t.Skip("No models directory available")
	}

	embeddersDir := filepath.Join(modelsDir, "embedders")
	modelPath := filepath.Join(embeddersDir, "google", "t5gemma-2-270m")
	if !fileExists(filepath.Join(modelPath, "encoder.onnx")) {
		t.Skipf("T5Gemma-2 model not found at %s. Skipping E2E test.", modelPath)
	}

	t.Logf("Using models directory: %s", modelsDir)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)
	port := findAvailablePort(t)
	serverURL := "http://localhost:" + itoa(port)

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

	select {
	case <-readyC:
		t.Log("Server is ready")
	case <-time.After(120 * time.Second):
		t.Fatal("Timeout waiting for server to be ready")
	}

	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	require.NoError(t, err, "Creating client failed")

	// Run sub-tests
	t.Run("ListModels", func(t *testing.T) {
		testT5Gemma2ListModelsEmbedder(t, ctx, termiteClient)
	})

	t.Run("EmbedText", func(t *testing.T) {
		testT5Gemma2EmbedText(t, ctx, termiteClient)
	})

	t.Run("EmbedMultipleTexts", func(t *testing.T) {
		testT5Gemma2EmbedMultipleTexts(t, ctx, termiteClient)
	})

	t.Run("EmbeddingDimensions", func(t *testing.T) {
		testT5Gemma2EmbeddingDimensions(t, ctx, termiteClient)
	})

	// Graceful shutdown
	serverCancel()
	select {
	case <-serverDone:
		t.Log("Server shutdown complete")
	case <-time.After(30 * time.Second):
		t.Log("Server shutdown timeout (may still be cleaning up)")
	}
}

func testT5Gemma2ListModelsEmbedder(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	require.NoError(t, err, "ListModels failed")

	// T5Gemma-2 should appear in embedders list
	found := false
	for _, name := range models.Embedders {
		if name == t5Gemma2ModelName {
			found = true
			break
		}
	}

	require.True(t, found, "T5Gemma-2 should be in embedders list. Available: %v", models.Embedders)
	t.Logf("Found T5Gemma-2 in embedders: %s", t5Gemma2ModelName)
}

func testT5Gemma2EmbedText(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	inputs := []string{
		"The quick brown fox jumps over the lazy dog.",
	}

	embeddings, err := c.Embed(ctx, t5Gemma2ModelName, inputs)
	require.NoError(t, err, "T5Gemma-2 embedding failed")
	require.Len(t, embeddings, 1, "Should return one embedding")
	require.NotEmpty(t, embeddings[0], "Embedding should not be empty")

	t.Logf("Single text embedding: %d dimensions", len(embeddings[0]))
}

func testT5Gemma2EmbedMultipleTexts(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	inputs := []string{
		"Machine learning is transforming artificial intelligence.",
		"Neural networks can learn complex patterns from data.",
		"Natural language processing enables computers to understand human language.",
	}

	embeddings, err := c.Embed(ctx, t5Gemma2ModelName, inputs)
	require.NoError(t, err, "T5Gemma-2 batch embedding failed")
	require.Len(t, embeddings, len(inputs), "Should return embedding for each input")

	for i, emb := range embeddings {
		require.NotEmpty(t, emb, "Embedding %d should not be empty", i)
		t.Logf("Embedding %d: %d dimensions", i, len(emb))
	}
}

func testT5Gemma2EmbeddingDimensions(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	inputs := []string{"Test embedding dimensions."}

	embeddings, err := c.Embed(ctx, t5Gemma2ModelName, inputs)
	require.NoError(t, err, "Embedding failed")
	require.Len(t, embeddings, 1, "Should return one embedding")

	// T5Gemma-2 returns token-level embeddings (seq_len * hidden_size).
	// The dimension should be a multiple of the hidden size (640).
	embDim := len(embeddings[0])
	assert.Greater(t, embDim, 0, "Embedding dimension should be positive")
	assert.Equal(t, 0, embDim%t5Gemma2HiddenSize,
		"T5Gemma-2 embedding dimension (%d) should be a multiple of hidden size (%d)",
		embDim, t5Gemma2HiddenSize)

	seqLen := embDim / t5Gemma2HiddenSize
	t.Logf("Verified embedding dimension: %d (seq_len=%d × hidden=%d)", embDim, seqLen, t5Gemma2HiddenSize)
}

// TestT5Gemma2GeneratorE2E tests the T5Gemma-2 seq2seq generator end-to-end.
func TestT5Gemma2GeneratorE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Check if model exists in rewriters directory (seq2seq models)
	modelsDir := getT5Gemma2ModelsDir(t)
	if modelsDir == "" {
		t.Skip("No models directory available")
	}

	rewritersDir := filepath.Join(modelsDir, "rewriters")
	modelPath := filepath.Join(rewritersDir, "google", "t5gemma-2-270m")
	if !fileExists(filepath.Join(modelPath, "encoder.onnx")) {
		t.Skipf("T5Gemma-2 generator model not found at %s. Skipping E2E test.", modelPath)
	}

	t.Logf("Using models directory: %s", modelsDir)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)
	port := findAvailablePort(t)
	serverURL := "http://localhost:" + itoa(port)

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

	select {
	case <-readyC:
		t.Log("Server is ready")
	case <-time.After(120 * time.Second):
		t.Fatal("Timeout waiting for server to be ready")
	}

	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	require.NoError(t, err, "Creating client failed")

	t.Run("ListModels", func(t *testing.T) {
		testT5Gemma2ListModelsRewriter(t, ctx, termiteClient)
	})

	t.Run("RewriteText", func(t *testing.T) {
		testT5Gemma2RewriteText(t, ctx, termiteClient)
	})

	t.Run("RewriteWithSummarizePrompt", func(t *testing.T) {
		testT5Gemma2RewriteSummarize(t, ctx, termiteClient)
	})

	t.Run("RewriteMultipleInputs", func(t *testing.T) {
		testT5Gemma2RewriteMultiple(t, ctx, termiteClient)
	})

	serverCancel()
	select {
	case <-serverDone:
		t.Log("Server shutdown complete")
	case <-time.After(30 * time.Second):
		t.Log("Server shutdown timeout")
	}
}

func testT5Gemma2ListModelsRewriter(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	require.NoError(t, err, "ListModels failed")

	// T5Gemma-2 should appear in rewriters list
	found := false
	for _, name := range models.Rewriters {
		if name == t5Gemma2ModelName {
			found = true
			break
		}
	}

	require.True(t, found, "T5Gemma-2 should be in rewriters list. Available: %v", models.Rewriters)
	t.Logf("Found T5Gemma-2 in rewriters: %s", t5Gemma2ModelName)
}

func testT5Gemma2RewriteText(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	inputs := []string{
		"The T5Gemma-2 model is a multimodal encoder-decoder architecture.",
	}

	resp, err := c.RewriteText(ctx, t5Gemma2ModelName, inputs)
	require.NoError(t, err, "T5Gemma-2 rewrite failed")
	require.NotNil(t, resp, "Response should not be nil")
	require.Equal(t, t5Gemma2ModelName, resp.Model, "Response model should match")
	require.NotEmpty(t, resp.Texts, "Should return generated text")

	for i, texts := range resp.Texts {
		require.NotEmpty(t, texts, "Output %d should have generated sequences", i)
		t.Logf("Input %d generated: %q", i, texts[0])
	}
}

func testT5Gemma2RewriteSummarize(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	// T5-style models typically work with task prefixes
	inputs := []string{
		"Summarize: The quick brown fox jumps over the lazy dog. This is a classic pangram that contains every letter of the English alphabet at least once. It has been used for testing typewriters and fonts since the late 19th century.",
	}

	resp, err := c.RewriteText(ctx, t5Gemma2ModelName, inputs)
	require.NoError(t, err, "T5Gemma-2 summarize failed")
	require.NotNil(t, resp, "Response should not be nil")
	require.NotEmpty(t, resp.Texts, "Should return summarized text")
	require.NotEmpty(t, resp.Texts[0], "First output should have generated text")

	t.Logf("Summarization result: %q", resp.Texts[0][0])
}

func testT5Gemma2RewriteMultiple(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	inputs := []string{
		"Hello world",
		"Translate this to French",
		"What is machine learning",
	}

	resp, err := c.RewriteText(ctx, t5Gemma2ModelName, inputs)
	require.NoError(t, err, "T5Gemma-2 batch rewrite failed")
	require.NotNil(t, resp, "Response should not be nil")
	require.Len(t, resp.Texts, len(inputs), "Should return output for each input")

	for i, texts := range resp.Texts {
		require.NotEmpty(t, texts, "Output %d should have generated text", i)
		t.Logf("Input %d: %q -> %q", i, inputs[i], texts[0])
	}
}

// Helper functions
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func itoa(i int) string {
	return strconv.Itoa(i)
}
