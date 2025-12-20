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
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/client/oapi"
	"github.com/antflydb/termite/pkg/termite"
	"go.uber.org/zap/zaptest"
)

const (
	// CLIP model name in the registry
	clipModelName = "clip-vit-base-patch32"

	// Expected embedding dimension for CLIP ViT-B/32
	clipEmbeddingDim = 512
)

// TestCLIPMultimodalE2E tests the full CLIP multimodal embedding pipeline:
// 1. Starts termite server with CLIP model (downloaded by TestMain)
// 2. Tests text and image embedding
// 3. Verifies cross-modal embedding dimensions match
func TestCLIPMultimodalE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

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

	// 4. Start termite server
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

	// 5. Create client
	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Run test cases
	t.Run("ListModels", func(t *testing.T) {
		testListModels(t, ctx, termiteClient)
	})

	t.Run("TextEmbedding", func(t *testing.T) {
		testTextEmbedding(t, ctx, termiteClient)
	})

	t.Run("ImageEmbedding", func(t *testing.T) {
		testImageEmbedding(t, ctx, termiteClient, serverURL)
	})

	t.Run("CrossModalSimilarity", func(t *testing.T) {
		testCrossModalSimilarity(t, ctx, termiteClient, serverURL)
	})

	// 6. Graceful shutdown
	t.Log("Shutting down server...")
	serverCancel()

	select {
	case <-serverDone:
		t.Log("Server shutdown complete")
	case <-time.After(30 * time.Second):
		t.Error("Timeout waiting for server shutdown")
	}
}

// testListModels verifies the CLIP model appears in the models list
func testListModels(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	if err != nil {
		t.Fatalf("ListModels failed: %v", err)
	}

	// Check that CLIP model is in the embedders list
	found := false
	for _, name := range models.Embedders {
		if name == clipModelName {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("CLIP model %s not found in embedders list: %v", clipModelName, models.Embedders)
	} else {
		t.Logf("Found CLIP model in embedders: %v", models.Embedders)
	}
}

// testTextEmbedding tests embedding text strings
func testTextEmbedding(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	texts := []string{
		"a photo of a cat",
		"a photo of a dog",
		"machine learning is interesting",
	}

	embeddings, err := c.Embed(ctx, clipModelName, texts)
	if err != nil {
		t.Fatalf("Text embedding failed: %v", err)
	}

	if len(embeddings) != len(texts) {
		t.Errorf("Expected %d embeddings, got %d", len(texts), len(embeddings))
	}

	for i, emb := range embeddings {
		if len(emb) != clipEmbeddingDim {
			t.Errorf("Embedding %d: expected dimension %d, got %d", i, clipEmbeddingDim, len(emb))
		}
		t.Logf("Text embedding %d: dim=%d, first3=[%.4f, %.4f, %.4f]",
			i, len(emb), emb[0], emb[1], emb[2])
	}
}

// testImageEmbedding tests embedding an image via multimodal ContentPart
func testImageEmbedding(t *testing.T, ctx context.Context, c *client.TermiteClient, serverURL string) {
	t.Helper()

	// Create a test image (100x100 red square)
	imageData := createTestImage(t, 100, 100, color.RGBA{255, 0, 0, 255})

	// Build multimodal embed request
	embedding := embedImage(t, ctx, serverURL, clipModelName, imageData)

	if len(embedding) != clipEmbeddingDim {
		t.Errorf("Expected embedding dimension %d, got %d", clipEmbeddingDim, len(embedding))
	}

	t.Logf("Image embedding: dim=%d, first3=[%.4f, %.4f, %.4f]",
		len(embedding), embedding[0], embedding[1], embedding[2])
}

// testCrossModalSimilarity verifies text and image embeddings have the same dimension
func testCrossModalSimilarity(t *testing.T, ctx context.Context, c *client.TermiteClient, serverURL string) {
	t.Helper()

	// Get text embedding
	textEmbeddings, err := c.Embed(ctx, clipModelName, []string{"a red square"})
	if err != nil {
		t.Fatalf("Text embedding failed: %v", err)
	}
	textEmb := textEmbeddings[0]

	// Get image embedding for a red square
	imageData := createTestImage(t, 100, 100, color.RGBA{255, 0, 0, 255})
	imageEmb := embedImage(t, ctx, serverURL, clipModelName, imageData)

	// Verify same dimensions
	if len(textEmb) != len(imageEmb) {
		t.Errorf("Cross-modal dimension mismatch: text=%d, image=%d", len(textEmb), len(imageEmb))
	} else {
		t.Logf("Cross-modal embeddings have matching dimension: %d", len(textEmb))
	}

	// Compute cosine similarity (optional, just for logging)
	similarity := cosineSimilarity(textEmb, imageEmb)
	t.Logf("Cosine similarity between 'a red square' and red square image: %.4f", similarity)
}

// embedImage sends an image embedding request using the oapi client directly
func embedImage(t *testing.T, ctx context.Context, serverURL, model string, imageData []byte) []float32 {
	t.Helper()

	// Build data URI
	base64Image := base64.StdEncoding.EncodeToString(imageData)
	dataURI := "data:image/png;base64," + base64Image

	// Build ContentPart for image
	var contentPart oapi.ContentPart
	err := contentPart.FromImageURLContentPart(oapi.ImageURLContentPart{
		Type: oapi.ImageURLContentPartTypeImageUrl,
		ImageUrl: oapi.ImageURL{
			Url: dataURI,
		},
	})
	if err != nil {
		t.Fatalf("Failed to create ContentPart: %v", err)
	}

	// Build request with multimodal input
	var inputUnion oapi.EmbedRequest_Input
	if err := inputUnion.FromEmbedRequestInput2([]oapi.ContentPart{contentPart}); err != nil {
		t.Fatalf("Failed to build input union: %v", err)
	}

	req := oapi.EmbedRequest{
		Model: model,
		Input: inputUnion,
	}

	// Create oapi client directly
	apiURL := serverURL + "/api"
	oapiClient, err := oapi.NewClientWithResponses(apiURL)
	if err != nil {
		t.Fatalf("Failed to create oapi client: %v", err)
	}

	// Send request
	resp, err := oapiClient.GenerateEmbeddingsWithResponse(ctx, req, func(ctx context.Context, req *http.Request) error {
		req.Header.Set("Accept", "application/json")
		return nil
	})
	if err != nil {
		t.Fatalf("Image embedding request failed: %v", err)
	}

	if resp.JSON400 != nil {
		t.Fatalf("Bad request: %s", resp.JSON400.Error)
	}
	if resp.JSON404 != nil {
		t.Fatalf("Model not found: %s", resp.JSON404.Error)
	}
	if resp.JSON500 != nil {
		t.Fatalf("Server error: %s", resp.JSON500.Error)
	}
	if resp.JSON200 == nil {
		t.Fatalf("Unexpected response: status=%d, body=%s", resp.StatusCode(), string(resp.Body))
	}

	if len(resp.JSON200.Embeddings) == 0 {
		t.Fatal("No embeddings returned")
	}

	return resp.JSON200.Embeddings[0]
}

// createTestImage creates a PNG image with the specified dimensions and color
func createTestImage(t *testing.T, width, height int, c color.Color) []byte {
	t.Helper()

	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, c)
		}
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("Failed to encode PNG: %v", err)
	}

	return buf.Bytes()
}

// findAvailablePort finds an available TCP port
func findAvailablePort(t *testing.T) int {
	t.Helper()

	listener, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Failed to find available port: %v", err)
	}
	defer listener.Close()

	return listener.Addr().(*net.TCPAddr).Port
}

// cosineSimilarity computes cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (sqrt(normA) * sqrt(normB))
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	// Newton's method
	z := x / 2
	for i := 0; i < 10; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return z
}
