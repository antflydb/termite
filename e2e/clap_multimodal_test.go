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
	"encoding/binary"
	"fmt"
	"math"
	"net/http"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/client/oapi"
	"github.com/antflydb/termite/pkg/termite"
	"go.uber.org/zap/zaptest"
)

const (
	// CLAP model name - using the smaller unfused version for faster tests
	clapModelName = "Xenova/clap-htsat-unfused"
	clapModelRepo = "Xenova/clap-htsat-unfused"

	// Expected embedding dimension for CLAP
	clapEmbeddingDim = 512
)

// TestCLAPMultimodalE2E tests the full CLAP multimodal embedding pipeline:
// 1. Downloads CLAP model if not present (lazy download)
// 2. Starts termite server with CLAP model
// 3. Tests text and audio embedding
// 4. Verifies cross-modal embedding dimensions match
func TestCLAPMultimodalE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Ensure CLAP model is downloaded from HuggingFace
	ensureHuggingFaceModel(t, clapModelName, clapModelRepo, ModelTypeEmbedder)

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
		testCLAPListModels(t, ctx, termiteClient)
	})

	t.Run("TextEmbedding", func(t *testing.T) {
		testCLAPTextEmbedding(t, ctx, termiteClient)
	})

	t.Run("AudioEmbedding", func(t *testing.T) {
		testCLAPAudioEmbedding(t, ctx, serverURL)
	})

	t.Run("CrossModalSimilarity", func(t *testing.T) {
		testCLAPCrossModalSimilarity(t, ctx, termiteClient, serverURL)
	})

	t.Run("DifferentAudiosProduceDifferentEmbeddings", func(t *testing.T) {
		testDifferentAudiosProduceDifferentEmbeddings(t, ctx, serverURL)
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

// testCLAPListModels verifies the CLAP model appears in the models list
func testCLAPListModels(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	if err != nil {
		t.Fatalf("ListModels failed: %v", err)
	}

	// Check that CLAP model is in the embedders list
	found := false
	for _, name := range models.Embedders {
		if name == clapModelName {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("CLAP model %s not found in embedders list: %v", clapModelName, models.Embedders)
	} else {
		t.Logf("Found CLAP model in embedders: %v", models.Embedders)
	}
}

// testCLAPTextEmbedding tests embedding text strings with CLAP
func testCLAPTextEmbedding(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	texts := []string{
		"a dog barking",
		"piano music playing",
		"birds chirping in the forest",
	}

	embeddings, err := c.Embed(ctx, clapModelName, texts)
	if err != nil {
		t.Fatalf("Text embedding failed: %v", err)
	}

	if len(embeddings) != len(texts) {
		t.Errorf("Expected %d embeddings, got %d", len(texts), len(embeddings))
	}

	for i, emb := range embeddings {
		if len(emb) != clapEmbeddingDim {
			t.Errorf("Embedding %d: expected dimension %d, got %d", i, clapEmbeddingDim, len(emb))
		}
		t.Logf("Text embedding %d: dim=%d, first3=[%.4f, %.4f, %.4f]",
			i, len(emb), emb[0], emb[1], emb[2])
	}
}

// testCLAPAudioEmbedding tests embedding audio via multimodal ContentPart
func testCLAPAudioEmbedding(t *testing.T, ctx context.Context, serverURL string) {
	t.Helper()

	// Create a test audio file (1 second of 440Hz sine wave)
	audioData := createTestAudio(t, 48000, 1.0, 440.0)

	// Embed the audio
	embedding := embedAudio(t, ctx, serverURL, clapModelName, audioData)

	if len(embedding) != clapEmbeddingDim {
		t.Errorf("Expected embedding dimension %d, got %d", clapEmbeddingDim, len(embedding))
	}

	t.Logf("Audio embedding: dim=%d, first3=[%.4f, %.4f, %.4f]",
		len(embedding), embedding[0], embedding[1], embedding[2])
}

// testCLAPCrossModalSimilarity verifies text and audio embeddings have the same dimension
func testCLAPCrossModalSimilarity(t *testing.T, ctx context.Context, c *client.TermiteClient, serverURL string) {
	t.Helper()

	// Get text embedding for a description
	textEmbeddings, err := c.Embed(ctx, clapModelName, []string{"a sine wave tone"})
	if err != nil {
		t.Fatalf("Text embedding failed: %v", err)
	}
	textEmb := textEmbeddings[0]

	// Get audio embedding for a sine wave
	audioData := createTestAudio(t, 48000, 1.0, 440.0)
	audioEmb := embedAudio(t, ctx, serverURL, clapModelName, audioData)

	// Verify same dimensions
	if len(textEmb) != len(audioEmb) {
		t.Errorf("Cross-modal dimension mismatch: text=%d, audio=%d", len(textEmb), len(audioEmb))
	} else {
		t.Logf("Cross-modal embeddings have matching dimension: %d", len(textEmb))
	}

	// Compute cosine similarity
	similarity := cosineSimilarity(textEmb, audioEmb)
	t.Logf("Cosine similarity between 'a sine wave tone' and 440Hz sine wave: %.4f", similarity)
}

// testDifferentAudiosProduceDifferentEmbeddings verifies that different audio produces different embeddings
func testDifferentAudiosProduceDifferentEmbeddings(t *testing.T, ctx context.Context, serverURL string) {
	t.Helper()

	// Create a low frequency sine wave (220Hz)
	lowFreqData := createTestAudio(t, 48000, 1.0, 220.0)
	lowFreqEmb := embedAudio(t, ctx, serverURL, clapModelName, lowFreqData)

	// Create a high frequency sine wave (880Hz)
	highFreqData := createTestAudio(t, 48000, 1.0, 880.0)
	highFreqEmb := embedAudio(t, ctx, serverURL, clapModelName, highFreqData)

	// Create silence
	silenceData := createTestAudio(t, 48000, 1.0, 0.0)
	silenceEmb := embedAudio(t, ctx, serverURL, clapModelName, silenceData)

	// Log the embeddings
	t.Logf("Low freq (220Hz) embedding: dim=%d, first3=[%.4f, %.4f, %.4f]",
		len(lowFreqEmb), lowFreqEmb[0], lowFreqEmb[1], lowFreqEmb[2])
	t.Logf("High freq (880Hz) embedding: dim=%d, first3=[%.4f, %.4f, %.4f]",
		len(highFreqEmb), highFreqEmb[0], highFreqEmb[1], highFreqEmb[2])
	t.Logf("Silence embedding: dim=%d, first3=[%.4f, %.4f, %.4f]",
		len(silenceEmb), silenceEmb[0], silenceEmb[1], silenceEmb[2])

	// Compute similarities
	lowHighSim := cosineSimilarity(lowFreqEmb, highFreqEmb)
	lowSilenceSim := cosineSimilarity(lowFreqEmb, silenceEmb)
	highSilenceSim := cosineSimilarity(highFreqEmb, silenceEmb)

	t.Logf("Cosine similarity (low freq vs high freq): %.4f", lowHighSim)
	t.Logf("Cosine similarity (low freq vs silence): %.4f", lowSilenceSim)
	t.Logf("Cosine similarity (high freq vs silence): %.4f", highSilenceSim)

	// Low and high frequency tones should be somewhat similar (both are tones) but not identical.
	// Note: Pure sine waves at different frequencies are very similar in CLAP's embedding space
	// since they're both perceived as "tones". We use a lenient threshold.
	if lowHighSim > 0.995 {
		t.Errorf("Low and high frequency embeddings are too similar (%.4f), expected different embeddings", lowHighSim)
	}

	// Tones should be different from silence
	if lowSilenceSim > 0.95 {
		t.Errorf("Low frequency and silence embeddings are too similar (%.4f)", lowSilenceSim)
	}

	// Verify the embeddings are not exactly the same
	sameCount := 0
	for i := range lowFreqEmb {
		if lowFreqEmb[i] == highFreqEmb[i] {
			sameCount++
		}
	}
	if sameCount == len(lowFreqEmb) {
		t.Error("Low and high frequency embeddings are identical - model may not be processing audio correctly")
	}
}

// embedAudio sends an audio embedding request using the oapi client directly
func embedAudio(t *testing.T, ctx context.Context, serverURL, model string, audioData []byte) []float32 {
	t.Helper()

	// Build data URI for audio/wav
	base64Audio := base64.StdEncoding.EncodeToString(audioData)
	dataURI := fmt.Sprintf("data:audio/wav;base64,%s", base64Audio)

	// Build ContentPart for audio (using ImageURLContentPart since the API uses the same structure)
	// Note: The MIME type in the data URI tells the server this is audio
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
		t.Fatalf("Audio embedding request failed: %v", err)
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

// createTestAudio creates a WAV file with a sine wave at the specified frequency.
// If frequency is 0, creates silence.
// sampleRate is typically 48000 for CLAP.
// duration is in seconds.
func createTestAudio(t *testing.T, sampleRate int, duration float64, frequency float64) []byte {
	t.Helper()

	numSamples := int(float64(sampleRate) * duration)
	samples := make([]int16, numSamples)

	if frequency > 0 {
		for i := 0; i < numSamples; i++ {
			// Generate sine wave
			sample := math.Sin(2.0 * math.Pi * frequency * float64(i) / float64(sampleRate))
			// Scale to 16-bit range with some headroom
			samples[i] = int16(sample * 32000)
		}
	}
	// If frequency is 0, samples remain as zeros (silence)

	// Create WAV file
	var buf bytes.Buffer

	// RIFF header
	buf.WriteString("RIFF")
	dataSize := numSamples * 2 // 16-bit samples = 2 bytes each
	fileSize := uint32(36 + dataSize)
	binary.Write(&buf, binary.LittleEndian, fileSize)
	buf.WriteString("WAVE")

	// fmt chunk
	buf.WriteString("fmt ")
	binary.Write(&buf, binary.LittleEndian, uint32(16))        // chunk size
	binary.Write(&buf, binary.LittleEndian, uint16(1))         // audio format (PCM)
	binary.Write(&buf, binary.LittleEndian, uint16(1))         // num channels (mono)
	binary.Write(&buf, binary.LittleEndian, uint32(sampleRate)) // sample rate
	byteRate := uint32(sampleRate * 2)                          // bytes per second
	binary.Write(&buf, binary.LittleEndian, byteRate)
	binary.Write(&buf, binary.LittleEndian, uint16(2)) // block align
	binary.Write(&buf, binary.LittleEndian, uint16(16)) // bits per sample

	// data chunk
	buf.WriteString("data")
	binary.Write(&buf, binary.LittleEndian, uint32(dataSize))
	for _, sample := range samples {
		binary.Write(&buf, binary.LittleEndian, sample)
	}

	return buf.Bytes()
}
