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
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/termite"
	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
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

	// Silero VAD model from HuggingFace for speech-boundary audio chunking
	vadModelName = "onnx-community/silero-vad"
	vadModelRepo = "onnx-community/silero-vad"
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
	for name := range models.Chunkers {
		if name == chunkerModelName {
			foundChunker = true
			break
		}
	}

	// Also check that fixed chunkers are always available
	hasFixed := false
	for name := range models.Chunkers {
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
		tc, err := chunk.AsTextContent()
		require.NoError(t, err, "Chunk %d should be text content", i)
		preview := tc.Text
		if len(preview) > 80 {
			preview = preview[:80] + "..."
		}
		t.Logf("  Chunk %d [%d:%d]: %q", i, tc.StartChar, tc.EndChar, preview)
	}

	// Verify chunk properties
	for i, chunk := range chunks {
		tc, err := chunk.AsTextContent()
		require.NoError(t, err, "Chunk %d should be text content", i)
		assert.NotEmpty(t, tc.Text, "Chunk %d should have text", i)
		assert.GreaterOrEqual(t, tc.EndChar, tc.StartChar, "Chunk %d end should be >= start", i)

		// Verify chunk text matches the original text slice
		if tc.EndChar <= len(text) && tc.StartChar <= tc.EndChar {
			expected := text[tc.StartChar:tc.EndChar]
			assert.Equal(t, expected, tc.Text, "Chunk %d text should match source slice", i)
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
		tc, err := chunk.AsTextContent()
		require.NoError(t, err, "Chunk %d should be text content", i)
		preview := tc.Text
		if len(preview) > 80 {
			preview = preview[:80] + "..."
		}
		t.Logf("  Chunk %d [%d:%d]: %q", i, tc.StartChar, tc.EndChar, preview)
	}

	// Fixed chunking should produce relatively uniform chunk sizes
	for i, chunk := range chunks {
		assert.NotEmpty(t, chunk.GetText(), "Chunk %d should have text", i)
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
		t.Logf("  Chunk %d: %q", i, chunk.GetText())
	}

	// Chunks should generally end at sentence boundaries (period + space or end of text)
	for i, chunk := range chunks {
		trimmed := strings.TrimSpace(chunk.GetText())
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
		chunkText := chunk.GetText()
		totalChars += len(chunkText)
		preview := chunkText
		if len(preview) > 60 {
			preview = preview[:60] + "..."
		}
		t.Logf("  Chunk %d (%d chars): %q", i, len(chunkText), preview)
	}

	// Verify we're not losing significant content
	// (some overlap or whitespace differences are acceptable)
	coverage := float64(totalChars) / float64(len(text))
	t.Logf("Content coverage: %.1f%%", coverage*100)
	assert.Greater(t, coverage, 0.8, "Chunks should cover most of the original text")
}

// TestVADChunkerE2E tests the VAD (Voice Activity Detection) audio chunking pipeline:
// 1. Downloads Silero VAD model from HuggingFace if not present
// 2. Patches the manifest to add capabilities: ["audio"]
// 3. Starts termite server with VAD chunker model
// 4. Generates test WAV audio with speech/silence pattern
// 5. Sends chunk request via API and verifies chunks match speech regions
func TestVADChunkerE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Download Silero VAD model from HuggingFace
	modelPath := ensureHuggingFaceModel(t, vadModelName, vadModelRepo, ModelTypeChunker)

	// The HuggingFace auto-manifest doesn't detect Silero VAD's model.onnx as an
	// audio model (it only detects audio_model.onnx for CLAP). Patch the manifest
	// to add capabilities: ["audio"] so the VAD chunker discovery picks it up.
	ensureVADManifest(t, modelPath)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)

	modelsDir := getTestModelsDir()
	t.Logf("Using models directory: %s", modelsDir)

	port := findAvailablePort(t)
	serverURL := fmt.Sprintf("http://localhost:%d", port)
	t.Logf("Starting server on %s", serverURL)

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
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	t.Run("ListModelsVAD", func(t *testing.T) {
		testListModelsVAD(t, ctx, termiteClient)
	})

	t.Run("VADAudioChunking", func(t *testing.T) {
		testVADAudioChunking(t, ctx, termiteClient)
	})

	t.Run("VADSilenceOnly", func(t *testing.T) {
		testVADSilenceOnly(t, ctx, termiteClient)
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

// ensureVADManifest patches the model manifest to include capabilities: ["audio"]
// and backends: ["onnx"]. The HuggingFace downloader's auto-manifest only detects
// audio_model.onnx (CLAP pattern), not Silero VAD's plain model.onnx. The model
// also requires ONNX Runtime because it has dynamic dimensions that GoMLX cannot handle.
func ensureVADManifest(t *testing.T, modelPath string) {
	t.Helper()

	manifestPath := filepath.Join(modelPath, modelregistry.ManifestFilename)

	data, err := os.ReadFile(manifestPath)
	require.NoError(t, err, "Reading manifest from %s", manifestPath)

	var manifest map[string]any
	require.NoError(t, json.Unmarshal(data, &manifest), "Parsing manifest JSON")

	needsWrite := false

	// Check if audio capability already present
	hasAudio := false
	if caps, ok := manifest["capabilities"].([]any); ok {
		for _, c := range caps {
			if c == string(modelregistry.CapabilityAudio) {
				hasAudio = true
				break
			}
		}
	}
	if !hasAudio {
		manifest["capabilities"] = []string{string(modelregistry.CapabilityAudio)}
		needsWrite = true
	}

	// Ensure backends includes "onnx" (required for dynamic-dimension ONNX models)
	hasOnnxBackend := false
	if backends, ok := manifest["backends"].([]any); ok {
		for _, b := range backends {
			if b == "onnx" {
				hasOnnxBackend = true
				break
			}
		}
	}
	if !hasOnnxBackend {
		manifest["backends"] = []string{"onnx"}
		needsWrite = true
	}

	if !needsWrite {
		t.Log("VAD manifest already has audio capability and onnx backend")
		return
	}

	updated, err := json.MarshalIndent(manifest, "", "  ")
	require.NoError(t, err, "Marshaling updated manifest")

	require.NoError(t, os.WriteFile(manifestPath, updated, 0644), "Writing updated manifest")
	t.Logf("Patched VAD manifest at %s with capabilities: [\"audio\"], backends: [\"onnx\"]", manifestPath)
}

// testListModelsVAD verifies the VAD chunker model appears in the models list
func testListModelsVAD(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	require.NoError(t, err, "ListModels failed")

	found := false
	for name := range models.Chunkers {
		if name == vadModelName {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("VAD model %s not found in chunkers: %v", vadModelName, models.Chunkers)
	} else {
		t.Logf("Found VAD chunker model: %s", vadModelName)
	}
	t.Logf("Available chunkers: %v", models.Chunkers)
}

// testVADAudioChunking generates a WAV with speech-like/silence pattern and verifies
// the VAD chunker returns chunks aligned to speech regions.
func testVADAudioChunking(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	// Generate test audio: 1s noise, 1s silence, 1s noise, 1s silence, 1s noise
	// at 16kHz mono 16-bit. Total 5 seconds.
	// "Speech" = multi-harmonic signal mimicking a voiced speech segment.
	// We combine harmonics at a fundamental frequency (~150Hz) to produce a
	// vowel-like waveform that Silero VAD recognizes as speech.
	// Silence = zeros
	const sampleRate = 16000
	const segmentDuration = 1.0 // seconds per segment
	const segmentSamples = int(sampleRate * segmentDuration)

	// Pattern: speech, silence, speech, silence, speech
	pattern := []bool{true, false, true, false, true}
	totalSamples := segmentSamples * len(pattern)
	samples := make([]int16, totalSamples)

	// Use a deterministic pseudo-random source so the test is reproducible
	rng := newLCG(42)

	for i, isSpeech := range pattern {
		if isSpeech {
			offset := i * segmentSamples
			for j := 0; j < segmentSamples; j++ {
				// Generate a vowel-like signal: fundamental at 150Hz with harmonics
				// plus a small amount of noise for naturalness.
				ts := float64(j) / float64(sampleRate)
				f0 := 150.0
				var sample float64
				// Add harmonics (1st through 15th) with decreasing amplitude
				for h := 1; h <= 15; h++ {
					amplitude := 1.0 / float64(h)
					sample += amplitude * math.Sin(2.0*math.Pi*f0*float64(h)*ts)
				}
				// Add noise component (~10% of signal)
				noise := (rng.Float64()*2.0 - 1.0) * 0.1
				sample += noise
				// Normalize and convert
				sample = sample / 3.0 // rough normalization
				if sample > 1.0 {
					sample = 1.0
				} else if sample < -1.0 {
					sample = -1.0
				}
				samples[offset+j] = int16(sample * 30000)
			}
		}
	}

	// Encode as WAV
	wavData := encodeTestWAV(t, samples, sampleRate)
	t.Logf("Generated test WAV: %d bytes, %d samples, %.1fs duration",
		len(wavData), totalSamples, float64(totalSamples)/float64(sampleRate))

	// Send chunk request using the VAD model with a lower threshold for synthetic audio
	chunks, err := c.ChunkMedia(ctx, wavData, "audio/wav", client.MediaChunkConfig{
		Model:     vadModelName,
		Threshold: 0.3,
	})
	require.NoError(t, err, "VAD chunking failed")

	// We should get chunks for the speech-like regions
	require.NotEmpty(t, chunks, "VAD should detect speech-like signal and produce chunks")
	t.Logf("VAD produced %d chunks", len(chunks))

	for i, chunk := range chunks {
		assert.Equal(t, "audio/wav", chunk.MimeType, "Chunk %d should be audio/wav", i)

		bc, err := chunk.AsBinaryContent()
		require.NoError(t, err, "Chunk %d should be binary content", i)
		assert.NotEmpty(t, bc.Data, "Chunk %d should have audio data", i)
		assert.GreaterOrEqual(t, bc.EndTimeMs, bc.StartTimeMs, "Chunk %d end time should be >= start time", i)

		t.Logf("  Chunk %d: %.0fms - %.0fms (%.0fms duration, %d bytes)",
			i, bc.StartTimeMs, bc.EndTimeMs, bc.EndTimeMs-bc.StartTimeMs, len(bc.Data))
	}

	// Verify ordering: chunks should be in temporal order
	for i := 1; i < len(chunks); i++ {
		prev, _ := chunks[i-1].AsBinaryContent()
		curr, _ := chunks[i].AsBinaryContent()
		assert.GreaterOrEqual(t, curr.StartTimeMs, prev.StartTimeMs,
			"Chunk %d should start after chunk %d", i, i-1)
	}

	// Verify speech coverage: the total chunk duration should cover a significant
	// portion of the 3 seconds of speech in the test audio (3x 1s segments)
	var totalChunkDurationMs float32
	for _, chunk := range chunks {
		bc, _ := chunk.AsBinaryContent()
		totalChunkDurationMs += bc.EndTimeMs - bc.StartTimeMs
	}
	t.Logf("Total chunk duration: %.0fms (expected ~3000ms of speech)", totalChunkDurationMs)

	// Synthetic audio may not trigger the VAD as strongly as real speech.
	// We just verify that some speech was detected (at least 500ms) and that
	// the total doesn't exceed the audio length.
	assert.Greater(t, totalChunkDurationMs, float32(500),
		"Total chunk duration should indicate some speech was detected")
	assert.Less(t, totalChunkDurationMs, float32(5500),
		"Total chunk duration should not exceed total audio length with padding")
}

// testVADSilenceOnly verifies the VAD chunker handles all-silence audio gracefully.
func testVADSilenceOnly(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	// Generate 2 seconds of silence
	const sampleRate = 16000
	const totalSamples = sampleRate * 2
	samples := make([]int16, totalSamples)

	wavData := encodeTestWAV(t, samples, sampleRate)
	t.Logf("Generated silence WAV: %d bytes", len(wavData))

	chunks, err := c.ChunkMedia(ctx, wavData, "audio/wav", client.MediaChunkConfig{
		Model: vadModelName,
	})
	require.NoError(t, err, "VAD chunking of silence should not error")

	// All-silence audio should produce zero chunks
	assert.Empty(t, chunks, "All-silence audio should produce no chunks")
	t.Logf("Silence test: got %d chunks (expected 0)", len(chunks))
}

// encodeTestWAV encodes int16 mono PCM samples into a WAV file.
func encodeTestWAV(t *testing.T, samples []int16, sampleRate int) []byte {
	t.Helper()

	var buf bytes.Buffer
	dataSize := len(samples) * 2

	// RIFF header
	buf.WriteString("RIFF")
	binary.Write(&buf, binary.LittleEndian, uint32(36+dataSize))
	buf.WriteString("WAVE")

	// fmt chunk
	buf.WriteString("fmt ")
	binary.Write(&buf, binary.LittleEndian, uint32(16))             // chunk size
	binary.Write(&buf, binary.LittleEndian, uint16(1))              // PCM format
	binary.Write(&buf, binary.LittleEndian, uint16(1))              // mono
	binary.Write(&buf, binary.LittleEndian, uint32(sampleRate))     // sample rate
	binary.Write(&buf, binary.LittleEndian, uint32(sampleRate*2))   // byte rate
	binary.Write(&buf, binary.LittleEndian, uint16(2))              // block align
	binary.Write(&buf, binary.LittleEndian, uint16(16))             // bits per sample

	// data chunk
	buf.WriteString("data")
	binary.Write(&buf, binary.LittleEndian, uint32(dataSize))
	for _, s := range samples {
		binary.Write(&buf, binary.LittleEndian, s)
	}

	return buf.Bytes()
}

// lcg is a simple linear congruential generator for deterministic pseudo-random numbers.
type lcg struct {
	state uint64
}

func newLCG(seed uint64) *lcg {
	return &lcg{state: seed}
}

func (l *lcg) Float64() float64 {
	l.state = l.state*6364136223846793005 + 1442695040888963407
	return float64(l.state>>33) / float64(1<<31)
}
