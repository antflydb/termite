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
	"strings"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/termite"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

const (
	// Semantic chunker model from Antfly registry
	// Uses mbert-based architecture for multilingual sentence boundary detection
	// Registry format: owner/model-name (with hyphens)
	chunkerRegistryName = "mirth/chonky-mmbert-small-multilingual-1"
	chunkerModelName    = "mirth/chonky-mmbert-small-multilingual-1"
)

// TestChunkerE2E tests the semantic chunking pipeline:
// 1. Downloads chonky-mmbert model if not present (lazy download from Antfly registry)
// 2. Starts termite server with chunker model
// 3. Tests basic chunking with semantic model
// 4. Tests fixed-size chunking as fallback
// 5. Tests chunk boundary detection (semantic boundaries)
func TestChunkerE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Ensure chunker model is downloaded from Antfly registry (lazy download)
	ensureRegistryModel(t, chunkerRegistryName, ModelTypeChunker)

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
		testListModelsChunker(t, ctx, termiteClient)
	})

	t.Run("SemanticChunking", func(t *testing.T) {
		testSemanticChunking(t, ctx, termiteClient)
	})

	t.Run("FixedChunking", func(t *testing.T) {
		testFixedChunking(t, ctx, termiteClient)
	})

	t.Run("ChunkBoundaries", func(t *testing.T) {
		testChunkBoundaries(t, ctx, termiteClient)
	})

	t.Run("LongDocument", func(t *testing.T) {
		testLongDocumentChunking(t, ctx, termiteClient)
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

// testListModelsChunker verifies the chunker model appears in the models list
func testListModelsChunker(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	require.NoError(t, err, "ListModels failed")

	// Check that chunker model is in the chunkers list
	foundChunker := false
	for _, name := range models.Chunkers {
		if name == chunkerModelName {
			foundChunker = true
			break
		}
	}

	// Also check that fixed chunkers are always available
	hasFixed := false
	for _, name := range models.Chunkers {
		if strings.HasPrefix(name, "fixed") {
			hasFixed = true
			break
		}
	}

	if !foundChunker {
		t.Errorf("Chunker model %s not found in chunkers: %v", chunkerModelName, models.Chunkers)
	} else {
		t.Logf("Found chunker model: %s", chunkerModelName)
	}

	assert.True(t, hasFixed, "Fixed chunker should always be available, got: %v", models.Chunkers)
	t.Logf("Available chunkers: %v", models.Chunkers)
}

// testSemanticChunking tests chunking with the neural model
func testSemanticChunking(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	// Multi-sentence paragraph that should be split semantically
	text := `Machine learning is a subset of artificial intelligence. It enables computers to learn from data.
Deep learning uses neural networks with many layers. These networks can recognize complex patterns.
Natural language processing helps computers understand human language. It powers chatbots and translation systems.`

	chunks, err := c.Chunk(ctx, text, client.ChunkConfig{
		Model:        chunkerModelName,
		TargetTokens: 50, // Small target to encourage multiple chunks
	})
	require.NoError(t, err, "Semantic chunking failed")

	// Should produce multiple chunks for this multi-sentence text
	assert.NotEmpty(t, chunks, "Should produce at least one chunk")

	// Log the chunks
	t.Logf("Semantic chunking produced %d chunks:", len(chunks))
	for i, chunk := range chunks {
		preview := chunk.Text
		if len(preview) > 80 {
			preview = preview[:80] + "..."
		}
		t.Logf("  Chunk %d [%d:%d]: %q", i, chunk.StartChar, chunk.EndChar, preview)
	}

	// Verify chunk properties
	for i, chunk := range chunks {
		assert.NotEmpty(t, chunk.Text, "Chunk %d should have text", i)
		assert.GreaterOrEqual(t, chunk.EndChar, chunk.StartChar, "Chunk %d end should be >= start", i)

		// Verify chunk text matches the original text slice
		if int(chunk.EndChar) <= len(text) && int(chunk.StartChar) <= int(chunk.EndChar) {
			expected := text[chunk.StartChar:chunk.EndChar]
			assert.Equal(t, expected, chunk.Text, "Chunk %d text should match source slice", i)
		}
	}
}

// testFixedChunking tests chunking with the fixed-size fallback
func testFixedChunking(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	text := `This is a test of fixed-size chunking. It splits text based on token counts rather than semantic boundaries.
The fixed chunker is always available as a fallback when neural models are not loaded.
It uses a BERT tokenizer to count tokens and ensures consistent chunk sizes.`

	chunks, err := c.Chunk(ctx, text, client.ChunkConfig{
		Model:        "fixed", // Use fixed chunker
		TargetTokens: 30,      // Small target for multiple chunks
	})
	require.NoError(t, err, "Fixed chunking failed")

	assert.NotEmpty(t, chunks, "Fixed chunking should produce chunks")

	t.Logf("Fixed chunking produced %d chunks:", len(chunks))
	for i, chunk := range chunks {
		preview := chunk.Text
		if len(preview) > 80 {
			preview = preview[:80] + "..."
		}
		t.Logf("  Chunk %d [%d:%d]: %q", i, chunk.StartChar, chunk.EndChar, preview)
	}

	// Fixed chunking should produce relatively uniform chunk sizes
	for i, chunk := range chunks {
		assert.NotEmpty(t, chunk.Text, "Chunk %d should have text", i)
	}
}

// testChunkBoundaries tests that semantic chunking respects sentence boundaries
func testChunkBoundaries(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	// Text with clear sentence boundaries
	text := `First sentence ends here. Second sentence starts here and continues. Third sentence is also present. Fourth sentence concludes the paragraph.`

	chunks, err := c.Chunk(ctx, text, client.ChunkConfig{
		Model:        chunkerModelName,
		TargetTokens: 20, // Very small to force splits
		Threshold:    0.3,
	})
	require.NoError(t, err, "Chunk boundary test failed")

	t.Logf("Boundary test produced %d chunks:", len(chunks))
	for i, chunk := range chunks {
		t.Logf("  Chunk %d: %q", i, chunk.Text)
	}

	// Chunks should generally end at sentence boundaries (period + space or end of text)
	for i, chunk := range chunks {
		trimmed := strings.TrimSpace(chunk.Text)
		if len(trimmed) > 0 && i < len(chunks)-1 {
			// Non-final chunks should ideally end with punctuation
			lastChar := trimmed[len(trimmed)-1]
			if lastChar != '.' && lastChar != '!' && lastChar != '?' {
				t.Logf("  Note: Chunk %d doesn't end with sentence punctuation: %q", i, trimmed)
			}
		}
	}
}

// testLongDocumentChunking tests chunking of a longer document
func testLongDocumentChunking(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	// Generate a longer document with multiple paragraphs
	paragraphs := []string{
		"Artificial intelligence has transformed many industries over the past decade. From healthcare to finance, AI systems are making decisions that were once exclusively human. Machine learning algorithms can now diagnose diseases, predict stock prices, and drive autonomous vehicles.",
		"The development of large language models represents a significant breakthrough in natural language processing. These models can understand context, generate coherent text, and even write code. Companies are racing to build ever-larger and more capable AI systems.",
		"However, the rapid advancement of AI also raises important ethical questions. Concerns about job displacement, algorithmic bias, and privacy are increasingly prominent in public discourse. Researchers and policymakers are working to establish guidelines for responsible AI development.",
		"Looking ahead, the future of AI seems both promising and uncertain. While the technology offers tremendous potential benefits, it also poses risks that society must carefully manage. The key challenge is to harness AI's power while minimizing its potential harms.",
	}
	text := strings.Join(paragraphs, "\n\n")

	chunks, err := c.Chunk(ctx, text, client.ChunkConfig{
		Model:        chunkerModelName,
		TargetTokens: 100,
		MaxChunks:    20,
	})
	require.NoError(t, err, "Long document chunking failed")

	assert.NotEmpty(t, chunks, "Should produce chunks for long document")
	assert.LessOrEqual(t, len(chunks), 20, "Should respect MaxChunks limit")

	t.Logf("Long document (%d chars) produced %d chunks:", len(text), len(chunks))
	totalChars := 0
	for i, chunk := range chunks {
		totalChars += len(chunk.Text)
		preview := chunk.Text
		if len(preview) > 60 {
			preview = preview[:60] + "..."
		}
		t.Logf("  Chunk %d (%d chars): %q", i, len(chunk.Text), preview)
	}

	// Verify we're not losing significant content
	// (some overlap or whitespace differences are acceptable)
	coverage := float64(totalChars) / float64(len(text))
	t.Logf("Content coverage: %.1f%%", coverage*100)
	assert.Greater(t, coverage, 0.8, "Chunks should cover most of the original text")
}
