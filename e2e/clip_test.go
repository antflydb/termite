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
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/client/oapi"
	"github.com/antflydb/termite/pkg/termite"
	"go.uber.org/zap/zaptest"
)

const (
	// CLIP model name in the registry
	clipModelName = "openai/clip-vit-base-patch32"

	// Expected embedding dimension for CLIP ViT-B/32
	clipEmbeddingDim = 512
)

// TestCLIPE2E tests the full CLIP multimodal embedding pipeline:
// 1. Downloads CLIP model if not present (lazy download)
// 2. Starts termite server with CLIP model
// 3. Tests text and image embedding
// 4. Verifies cross-modal embedding dimensions match
func TestCLIPE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Ensure CLIP model is downloaded (lazy download)
	ensureRegistryModel(t, clipModelName, ModelTypeEmbedder)

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
		testListModelsCLIP(t, ctx, termiteClient)
	})

	t.Run("TextEmbedding", func(t *testing.T) {
		testTextEmbeddingCLIP(t, ctx, termiteClient)
	})

	t.Run("ImageEmbedding", func(t *testing.T) {
		testImageEmbeddingCLIP(t, ctx, termiteClient, serverURL)
	})

	t.Run("CrossModalSimilarity", func(t *testing.T) {
		testCrossModalSimilarityCLIP(t, ctx, termiteClient, serverURL)
	})

	t.Run("DifferentImagesProduceDifferentEmbeddings", func(t *testing.T) {
		testDifferentImagesCLIP(t, ctx, serverURL)
	})

	t.Run("CrossModalRetrieval", func(t *testing.T) {
		testCrossModalRetrievalCLIP(t, ctx, termiteClient, serverURL)
	})

	t.Run("MixedModalityBatch", func(t *testing.T) {
		testMixedModalityBatchCLIP(t, ctx, serverURL)
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

// testListModelsCLIP verifies the CLIP model appears in the models list
func testListModelsCLIP(t *testing.T, ctx context.Context, c *client.TermiteClient) {
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

// testTextEmbeddingCLIP tests embedding text strings
func testTextEmbeddingCLIP(t *testing.T, ctx context.Context, c *client.TermiteClient) {
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

// testImageEmbeddingCLIP tests embedding an image via multimodal ContentPart
func testImageEmbeddingCLIP(t *testing.T, ctx context.Context, c *client.TermiteClient, serverURL string) {
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

// testCrossModalSimilarityCLIP verifies text and image embeddings have the same dimension
func testCrossModalSimilarityCLIP(t *testing.T, ctx context.Context, c *client.TermiteClient, serverURL string) {
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

// testDifferentImagesCLIP verifies that different images produce different embeddings
func testDifferentImagesCLIP(t *testing.T, ctx context.Context, serverURL string) {
	t.Helper()

	// Create a synthetic red square image
	redSquareData := createTestImage(t, 100, 100, color.RGBA{255, 0, 0, 255})
	redSquareEmb := embedImage(t, ctx, serverURL, clipModelName, redSquareData)

	// Create a synthetic blue square image
	blueSquareData := createTestImage(t, 100, 100, color.RGBA{0, 0, 255, 255})
	blueSquareEmb := embedImage(t, ctx, serverURL, clipModelName, blueSquareData)

	// Load the real flower image if available
	flowerPath := filepath.Join("testdata", "flower.jpg")
	flowerData, err := os.ReadFile(flowerPath)
	if err != nil {
		t.Logf("Skipping flower.jpg test (file not found at %s): %v", flowerPath, err)
	}

	var flowerEmb []float32
	if flowerData != nil {
		flowerEmb = embedImageWithMimeType(t, ctx, serverURL, clipModelName, flowerData, "image/jpeg")
	}

	// Log the embeddings
	t.Logf("Red square embedding: dim=%d, first3=[%.4f, %.4f, %.4f]",
		len(redSquareEmb), redSquareEmb[0], redSquareEmb[1], redSquareEmb[2])
	t.Logf("Blue square embedding: dim=%d, first3=[%.4f, %.4f, %.4f]",
		len(blueSquareEmb), blueSquareEmb[0], blueSquareEmb[1], blueSquareEmb[2])
	if flowerEmb != nil {
		t.Logf("Flower embedding: dim=%d, first3=[%.4f, %.4f, %.4f]",
			len(flowerEmb), flowerEmb[0], flowerEmb[1], flowerEmb[2])
	}

	// Verify embeddings are different
	redBlueSim := cosineSimilarity(redSquareEmb, blueSquareEmb)
	t.Logf("Cosine similarity (red square vs blue square): %.4f", redBlueSim)

	// Red and blue squares should be somewhat similar (both simple colored squares) but not identical
	if redBlueSim > 0.99 {
		t.Errorf("Red and blue square embeddings are too similar (%.4f), expected different embeddings", redBlueSim)
	}

	// Verify the embeddings are not exactly the same
	sameCount := 0
	for i := range redSquareEmb {
		if redSquareEmb[i] == blueSquareEmb[i] {
			sameCount++
		}
	}
	if sameCount == len(redSquareEmb) {
		t.Error("Red and blue square embeddings are identical - model may not be processing images correctly")
	}

	if flowerEmb != nil {
		redFlowerSim := cosineSimilarity(redSquareEmb, flowerEmb)
		blueFlowerSim := cosineSimilarity(blueSquareEmb, flowerEmb)
		t.Logf("Cosine similarity (red square vs flower): %.4f", redFlowerSim)
		t.Logf("Cosine similarity (blue square vs flower): %.4f", blueFlowerSim)

		// Flower should be quite different from simple colored squares
		if redFlowerSim > 0.95 {
			t.Errorf("Red square and flower embeddings are too similar (%.4f)", redFlowerSim)
		}

		// Verify flower embedding is not identical to squares
		flowerSameAsRed := 0
		flowerSameAsBlue := 0
		for i := range flowerEmb {
			if flowerEmb[i] == redSquareEmb[i] {
				flowerSameAsRed++
			}
			if flowerEmb[i] == blueSquareEmb[i] {
				flowerSameAsBlue++
			}
		}
		if flowerSameAsRed == len(flowerEmb) {
			t.Error("Flower and red square embeddings are identical")
		}
		if flowerSameAsBlue == len(flowerEmb) {
			t.Error("Flower and blue square embeddings are identical")
		}
	}
}

// embedImage sends an image embedding request using the oapi client directly (assumes PNG)
func embedImage(t *testing.T, ctx context.Context, serverURL, model string, imageData []byte) []float32 {
	return embedImageWithMimeType(t, ctx, serverURL, model, imageData, "image/png")
}

// embedImageWithMimeType sends an image embedding request with a specific MIME type
func embedImageWithMimeType(t *testing.T, ctx context.Context, serverURL, model string, imageData []byte, mimeType string) []float32 {
	t.Helper()

	// Build data URI
	base64Image := base64.StdEncoding.EncodeToString(imageData)
	dataURI := fmt.Sprintf("data:%s;base64,%s", mimeType, base64Image)

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

// embedMultimodal sends a multimodal embed request with multiple content parts
// and returns all resulting embeddings. Useful for testing mixed-modality batches.
func embedMultimodal(t *testing.T, ctx context.Context, serverURL, model string, parts []oapi.ContentPart) [][]float32 {
	t.Helper()

	var inputUnion oapi.EmbedRequest_Input
	if err := inputUnion.FromEmbedRequestInput2(parts); err != nil {
		t.Fatalf("Failed to build input union: %v", err)
	}

	req := oapi.EmbedRequest{
		Model: model,
		Input: inputUnion,
	}

	apiURL := serverURL + "/api"
	oapiClient, err := oapi.NewClientWithResponses(apiURL)
	if err != nil {
		t.Fatalf("Failed to create oapi client: %v", err)
	}

	resp, err := oapiClient.GenerateEmbeddingsWithResponse(ctx, req, func(ctx context.Context, req *http.Request) error {
		req.Header.Set("Accept", "application/json")
		return nil
	})
	if err != nil {
		t.Fatalf("Multimodal embedding request failed: %v", err)
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

	return resp.JSON200.Embeddings
}

// makeTextContentPart creates a text ContentPart for multimodal embed requests.
func makeTextContentPart(t *testing.T, text string) oapi.ContentPart {
	t.Helper()
	var cp oapi.ContentPart
	if err := cp.FromTextContentPart(oapi.TextContentPart{
		Type: oapi.TextContentPartTypeText,
		Text: text,
	}); err != nil {
		t.Fatalf("Failed to create text ContentPart: %v", err)
	}
	return cp
}

// makeImageContentPart creates an image ContentPart from raw PNG data.
func makeImageContentPart(t *testing.T, imageData []byte) oapi.ContentPart {
	t.Helper()
	base64Data := base64.StdEncoding.EncodeToString(imageData)
	dataURI := fmt.Sprintf("data:image/png;base64,%s", base64Data)
	var cp oapi.ContentPart
	if err := cp.FromImageURLContentPart(oapi.ImageURLContentPart{
		Type:     oapi.ImageURLContentPartTypeImageUrl,
		ImageUrl: oapi.ImageURL{Url: dataURI},
	}); err != nil {
		t.Fatalf("Failed to create image ContentPart: %v", err)
	}
	return cp
}

// testCrossModalRetrievalCLIP verifies that text queries retrieve the semantically
// correct image from a set of candidates based on cosine similarity ranking.
// Uses real photographs (cat, car, flower) from testdata/ for reliable cross-modal
// alignment — CLIP was trained on natural images and struggles with synthetic data.
func testCrossModalRetrievalCLIP(t *testing.T, ctx context.Context, c *client.TermiteClient, serverURL string) {
	t.Helper()

	// Load two semantically distinct real test images (cat vs car).
	// Pairwise comparisons are more robust than multi-way ranking because
	// CLIP's cross-modal similarities for small images are near zero with
	// tight margins — a third candidate can act as a confounder.
	catData, err := os.ReadFile(filepath.Join("testdata", "cat.jpg"))
	if err != nil {
		t.Skipf("Skipping cross-modal retrieval: cat.jpg not found: %v", err)
	}
	carData, err := os.ReadFile(filepath.Join("testdata", "car.jpg"))
	if err != nil {
		t.Skipf("Skipping cross-modal retrieval: car.jpg not found: %v", err)
	}

	catEmb := embedImageWithMimeType(t, ctx, serverURL, clipModelName, catData, "image/jpeg")
	carEmb := embedImageWithMimeType(t, ctx, serverURL, clipModelName, carData, "image/jpeg")

	textEmbs, err := c.Embed(ctx, clipModelName, []string{
		"a photo of a cat",
		"a photo of a car",
	})
	if err != nil {
		t.Fatalf("Text embedding failed: %v", err)
	}
	catTextEmb := textEmbs[0]
	carTextEmb := textEmbs[1]

	// "a photo of a cat" should be closer to the cat image than to the car image
	simCatTextCat := cosineSimilarity(catTextEmb, catEmb)
	simCatTextCar := cosineSimilarity(catTextEmb, carEmb)
	t.Logf("  sim(\"a photo of a cat\", cat) = %.4f", simCatTextCat)
	t.Logf("  sim(\"a photo of a cat\", car) = %.4f", simCatTextCar)

	if simCatTextCat <= simCatTextCar {
		t.Errorf("Retrieval failed: \"a photo of a cat\" closer to car (%.4f) than cat (%.4f)",
			simCatTextCar, simCatTextCat)
	} else {
		t.Logf("Retrieval OK: \"a photo of a cat\" -> cat (margin=%.4f)", simCatTextCat-simCatTextCar)
	}

	// "a photo of a car" should be closer to the car image than to the cat image
	simCarTextCar := cosineSimilarity(carTextEmb, carEmb)
	simCarTextCat := cosineSimilarity(carTextEmb, catEmb)
	t.Logf("  sim(\"a photo of a car\", car) = %.4f", simCarTextCar)
	t.Logf("  sim(\"a photo of a car\", cat) = %.4f", simCarTextCat)

	if simCarTextCar <= simCarTextCat {
		t.Errorf("Retrieval failed: \"a photo of a car\" closer to cat (%.4f) than car (%.4f)",
			simCarTextCat, simCarTextCar)
	} else {
		t.Logf("Retrieval OK: \"a photo of a car\" -> car (margin=%.4f)", simCarTextCar-simCarTextCat)
	}
}

// testMixedModalityBatchCLIP verifies that a single embed request containing
// both text and image content parts returns correct, distinct embeddings.
func testMixedModalityBatchCLIP(t *testing.T, ctx context.Context, serverURL string) {
	t.Helper()

	imageData := createTestImage(t, 100, 100, color.RGBA{255, 0, 0, 255})

	parts := []oapi.ContentPart{
		makeTextContentPart(t, "a photo of a cat"),
		makeImageContentPart(t, imageData),
	}

	embeddings := embedMultimodal(t, ctx, serverURL, clipModelName, parts)

	if len(embeddings) != 2 {
		t.Fatalf("Expected 2 embeddings from mixed batch, got %d", len(embeddings))
	}

	for i, emb := range embeddings {
		if len(emb) != clipEmbeddingDim {
			t.Errorf("Embedding %d: expected dim %d, got %d", i, clipEmbeddingDim, len(emb))
		}
	}

	// Text and image embeddings should not be identical
	sim := cosineSimilarity(embeddings[0], embeddings[1])
	t.Logf("Mixed batch: text-image similarity = %.4f", sim)
	if sim > 0.99 {
		t.Error("Mixed batch text and image embeddings are suspiciously identical")
	}
}
