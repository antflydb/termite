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
	"image/color"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/termite"
	"go.uber.org/zap/zaptest"
)

const (
	// CLIPCLAP model name in the registry
	clipclapModelName = "antflydb/clipclap"

	// Expected embedding dimension for CLIPCLAP (same as CLIP/CLAP: 512)
	clipclapEmbeddingDim = 512
)

// TestCLIPCLAPE2E tests the full CLIPCLAP unified multimodal embedding pipeline:
// 1. Downloads CLIPCLAP model if not present (lazy download)
// 2. Starts termite server with CLIPCLAP model
// 3. Tests text, image, AND audio embedding
// 4. Verifies all three modalities produce 512-dim embeddings in the same space
// 5. Checks cross-modal similarity between all pairs
func TestCLIPCLAPE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Ensure CLIPCLAP model is downloaded
	ensureRegistryModel(t, clipclapModelName, ModelTypeEmbedder)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
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
	case <-time.After(60 * time.Second):
		t.Fatal("Timeout waiting for server to be ready")
	}

	// Create client
	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Run test cases
	t.Run("ListModels", func(t *testing.T) {
		testListModelsCLIPCLAP(t, ctx, termiteClient)
	})

	t.Run("TextEmbedding", func(t *testing.T) {
		testTextEmbeddingCLIPCLAP(t, ctx, termiteClient)
	})

	t.Run("ImageEmbedding", func(t *testing.T) {
		testImageEmbeddingCLIPCLAP(t, ctx, serverURL)
	})

	t.Run("AudioEmbedding", func(t *testing.T) {
		testAudioEmbeddingCLIPCLAP(t, ctx, serverURL)
	})

	t.Run("CrossModalSimilarity", func(t *testing.T) {
		testCrossModalSimilarityCLIPCLAP(t, ctx, termiteClient, serverURL)
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

// testListModelsCLIPCLAP verifies the CLIPCLAP model appears in the models list
func testListModelsCLIPCLAP(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	if err != nil {
		t.Fatalf("ListModels failed: %v", err)
	}

	found := false
	for _, name := range models.Embedders {
		if name == clipclapModelName {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("CLIPCLAP model %s not found in embedders list: %v", clipclapModelName, models.Embedders)
	} else {
		t.Logf("Found CLIPCLAP model in embedders: %v", models.Embedders)
	}
}

// testTextEmbeddingCLIPCLAP tests embedding text strings with the unified model
func testTextEmbeddingCLIPCLAP(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	texts := []string{
		"a photo of a cat",
		"a dog barking loudly",
		"piano music playing softly",
	}

	embeddings, err := c.Embed(ctx, clipclapModelName, texts)
	if err != nil {
		t.Fatalf("Text embedding failed: %v", err)
	}

	if len(embeddings) != len(texts) {
		t.Errorf("Expected %d embeddings, got %d", len(texts), len(embeddings))
	}

	for i, emb := range embeddings {
		if len(emb) != clipclapEmbeddingDim {
			t.Errorf("Embedding %d: expected dimension %d, got %d", i, clipclapEmbeddingDim, len(emb))
		}
		t.Logf("Text embedding %d (%q): dim=%d, first3=[%.4f, %.4f, %.4f]",
			i, texts[i], len(emb), emb[0], emb[1], emb[2])
	}
}

// testImageEmbeddingCLIPCLAP tests embedding an image via the unified model
func testImageEmbeddingCLIPCLAP(t *testing.T, ctx context.Context, serverURL string) {
	t.Helper()

	// Create a test image (100x100 red square)
	imageData := createTestImage(t, 100, 100, color.RGBA{255, 0, 0, 255})

	embedding := embedImage(t, ctx, serverURL, clipclapModelName, imageData)

	if len(embedding) != clipclapEmbeddingDim {
		t.Errorf("Expected embedding dimension %d, got %d", clipclapEmbeddingDim, len(embedding))
	}

	t.Logf("Image embedding: dim=%d, first3=[%.4f, %.4f, %.4f]",
		len(embedding), embedding[0], embedding[1], embedding[2])
}

// testAudioEmbeddingCLIPCLAP tests embedding audio via the unified model
func testAudioEmbeddingCLIPCLAP(t *testing.T, ctx context.Context, serverURL string) {
	t.Helper()

	// Create a test audio file (1 second of 440Hz sine wave)
	audioData := createTestAudio(t, 48000, 1.0, 440.0)

	embedding := embedAudio(t, ctx, serverURL, clipclapModelName, audioData)

	if len(embedding) != clipclapEmbeddingDim {
		t.Errorf("Expected embedding dimension %d, got %d", clipclapEmbeddingDim, len(embedding))
	}

	t.Logf("Audio embedding: dim=%d, first3=[%.4f, %.4f, %.4f]",
		len(embedding), embedding[0], embedding[1], embedding[2])
}

// testCrossModalSimilarityCLIPCLAP verifies all three modalities produce embeddings
// in the same space (same dimension) and computes cross-modal similarity.
func testCrossModalSimilarityCLIPCLAP(t *testing.T, ctx context.Context, c *client.TermiteClient, serverURL string) {
	t.Helper()

	// Get text embedding
	textEmbeddings, err := c.Embed(ctx, clipclapModelName, []string{"a red square"})
	if err != nil {
		t.Fatalf("Text embedding failed: %v", err)
	}
	textEmb := textEmbeddings[0]

	// Get image embedding for a red square
	imageData := createTestImage(t, 100, 100, color.RGBA{255, 0, 0, 255})
	imageEmb := embedImage(t, ctx, serverURL, clipclapModelName, imageData)

	// Get audio embedding for a sine wave
	audioData := createTestAudio(t, 48000, 1.0, 440.0)
	audioEmb := embedAudio(t, ctx, serverURL, clipclapModelName, audioData)

	// Verify all dimensions match
	if len(textEmb) != clipclapEmbeddingDim {
		t.Errorf("Text embedding dimension: expected %d, got %d", clipclapEmbeddingDim, len(textEmb))
	}
	if len(imageEmb) != clipclapEmbeddingDim {
		t.Errorf("Image embedding dimension: expected %d, got %d", clipclapEmbeddingDim, len(imageEmb))
	}
	if len(audioEmb) != clipclapEmbeddingDim {
		t.Errorf("Audio embedding dimension: expected %d, got %d", clipclapEmbeddingDim, len(audioEmb))
	}

	if len(textEmb) != len(imageEmb) || len(textEmb) != len(audioEmb) {
		t.Errorf("Cross-modal dimension mismatch: text=%d, image=%d, audio=%d",
			len(textEmb), len(imageEmb), len(audioEmb))
	} else {
		t.Logf("All three modalities have matching dimension: %d", len(textEmb))
	}

	// Compute all pairwise similarities
	textImageSim := cosineSimilarity(textEmb, imageEmb)
	textAudioSim := cosineSimilarity(textEmb, audioEmb)
	imageAudioSim := cosineSimilarity(imageEmb, audioEmb)

	t.Logf("Cross-modal cosine similarities:")
	t.Logf("  text ↔ image: %.4f", textImageSim)
	t.Logf("  text ↔ audio: %.4f", textAudioSim)
	t.Logf("  image ↔ audio: %.4f", imageAudioSim)

	// Verify embeddings are not identical (different modalities should produce different embeddings)
	if textImageSim > 0.99 {
		t.Errorf("Text and image embeddings are suspiciously similar (%.4f)", textImageSim)
	}
	if textAudioSim > 0.99 {
		t.Errorf("Text and audio embeddings are suspiciously similar (%.4f)", textAudioSim)
	}
}
