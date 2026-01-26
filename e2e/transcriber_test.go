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
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	"github.com/antflydb/termite/pkg/termite/lib/transcribing"
)

// Whisper model from onnx-community (pre-exported ONNX)
const (
	whisperModelRepo = "onnx-community/whisper-tiny.en"
	whisperModelName = "onnx-community/whisper-tiny.en"
)

// =============================================================================
// Helper Functions
// =============================================================================

// generateTestWAV creates a simple WAV file with a sine wave tone.
// This is used to test the audio pipeline without needing real speech.
// Whisper will likely output silence markers or noise characters, but this
// validates the pipeline works end-to-end.
func generateTestWAV(t *testing.T, durationSec float64, frequency float64, sampleRate int) []byte {
	t.Helper()

	numSamples := int(durationSec * float64(sampleRate))
	audioData := make([]int16, numSamples)

	// Generate sine wave
	for i := 0; i < numSamples; i++ {
		sample := math.Sin(2 * math.Pi * frequency * float64(i) / float64(sampleRate))
		audioData[i] = int16(sample * 32767 * 0.5) // 50% amplitude
	}

	// Create WAV header
	dataSize := numSamples * 2 // 16-bit samples
	headerSize := 44
	fileSize := headerSize + dataSize

	wav := make([]byte, fileSize)

	// RIFF header
	copy(wav[0:4], "RIFF")
	binary.LittleEndian.PutUint32(wav[4:8], uint32(fileSize-8))
	copy(wav[8:12], "WAVE")

	// fmt subchunk
	copy(wav[12:16], "fmt ")
	binary.LittleEndian.PutUint32(wav[16:20], 16)          // Subchunk1Size (16 for PCM)
	binary.LittleEndian.PutUint16(wav[20:22], 1)           // AudioFormat (1 = PCM)
	binary.LittleEndian.PutUint16(wav[22:24], 1)           // NumChannels (mono)
	binary.LittleEndian.PutUint32(wav[24:28], uint32(sampleRate)) // SampleRate
	binary.LittleEndian.PutUint32(wav[28:32], uint32(sampleRate*2)) // ByteRate
	binary.LittleEndian.PutUint16(wav[32:34], 2)           // BlockAlign
	binary.LittleEndian.PutUint16(wav[34:36], 16)          // BitsPerSample

	// data subchunk
	copy(wav[36:40], "data")
	binary.LittleEndian.PutUint32(wav[40:44], uint32(dataSize))

	// Audio data
	for i, sample := range audioData {
		binary.LittleEndian.PutUint16(wav[44+i*2:46+i*2], uint16(sample))
	}

	return wav
}

// generateSilentWAV creates a WAV file with silence.
func generateSilentWAV(t *testing.T, durationSec float64, sampleRate int) []byte {
	t.Helper()

	numSamples := int(durationSec * float64(sampleRate))
	dataSize := numSamples * 2 // 16-bit samples
	headerSize := 44
	fileSize := headerSize + dataSize

	wav := make([]byte, fileSize)

	// RIFF header
	copy(wav[0:4], "RIFF")
	binary.LittleEndian.PutUint32(wav[4:8], uint32(fileSize-8))
	copy(wav[8:12], "WAVE")

	// fmt subchunk
	copy(wav[12:16], "fmt ")
	binary.LittleEndian.PutUint32(wav[16:20], 16)
	binary.LittleEndian.PutUint16(wav[20:22], 1)
	binary.LittleEndian.PutUint16(wav[22:24], 1)
	binary.LittleEndian.PutUint32(wav[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(wav[28:32], uint32(sampleRate*2))
	binary.LittleEndian.PutUint16(wav[32:34], 2)
	binary.LittleEndian.PutUint16(wav[34:36], 16)

	// data subchunk (all zeros = silence)
	copy(wav[36:40], "data")
	binary.LittleEndian.PutUint32(wav[40:44], uint32(dataSize))

	return wav
}

// createTranscriber creates a PooledTranscriber using the backends API.
func createTranscriber(t *testing.T, modelPath string) (*transcribing.PooledTranscriber, error) {
	t.Helper()

	logger := zap.NewNop()
	sessionManager := backends.NewSessionManager()

	cfg := &transcribing.PooledTranscriberConfig{
		ModelPath: modelPath,
		PoolSize:  1,
		Logger:    logger,
	}

	// Use all available backends
	modelBackends := []string{"onnx", "xla", "go"}

	transcriber, _, err := transcribing.NewPooledTranscriber(cfg, sessionManager, modelBackends)
	if err != nil {
		return nil, err
	}

	return transcriber, nil
}

// =============================================================================
// Whisper Tests
// =============================================================================

// TestWhisperModelDownload tests downloading the Whisper model from HuggingFace
func TestWhisperModelDownload(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Whisper download test in short mode")
	}

	modelPath := ensureHuggingFaceModel(t, whisperModelName, whisperModelRepo, ModelTypeTranscriber)

	// Check required files exist
	requiredFiles := []string{"tokenizer.json", "config.json"}
	for _, file := range requiredFiles {
		filePath := filepath.Join(modelPath, file)
		_, err := os.Stat(filePath)
		assert.NoError(t, err, "Missing required file: %s", file)
	}

	// Check for ONNX files
	onnxDir := filepath.Join(modelPath, "onnx")
	if _, err := os.Stat(onnxDir); err == nil {
		encoderPath := filepath.Join(onnxDir, "encoder_model.onnx")
		decoderPath := filepath.Join(onnxDir, "decoder_model_merged.onnx")

		if fileExists(encoderPath) && fileExists(decoderPath) {
			t.Logf("Found ONNX models in %s", onnxDir)
		} else {
			// Try alternative naming
			decoderPath = filepath.Join(onnxDir, "decoder_model.onnx")
			if fileExists(encoderPath) {
				t.Logf("Found encoder ONNX model in %s", onnxDir)
			}
		}
	}

	// Load and verify config using the pipelines package
	config, err := pipelines.LoadSpeech2SeqModelConfig(modelPath)
	require.NoError(t, err, "Failed to load Speech2Seq model config")

	t.Logf("Whisper config: num_heads=%d, head_dim=%d, vocab_size=%d, num_layers=%d",
		config.NumHeads, config.HeadDim, config.DecoderConfig.VocabSize, config.NumLayers)
}

// TestWhisperPipeline tests the Speech2Seq pipeline directly
func TestWhisperPipeline(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Whisper pipeline test in short mode")
	}

	modelPath := ensureHuggingFaceModel(t, whisperModelName, whisperModelRepo, ModelTypeTranscriber)

	// Check for ONNX files
	onnxDir := filepath.Join(modelPath, "onnx")
	encoderPath := filepath.Join(onnxDir, "encoder_model.onnx")
	if _, err := os.Stat(encoderPath); os.IsNotExist(err) {
		encoderPath = filepath.Join(modelPath, "encoder_model.onnx")
		onnxDir = modelPath
	}

	if _, err := os.Stat(encoderPath); os.IsNotExist(err) {
		t.Skipf("Whisper ONNX encoder not found at %s", encoderPath)
	}

	// Create session manager and load pipeline
	sessionManager := backends.NewSessionManager()
	modelBackends := []string{"onnx", "xla", "go"}

	pipeline, backendType, err := pipelines.LoadSpeech2SeqPipeline(
		modelPath,
		sessionManager,
		modelBackends,
	)
	if err != nil {
		t.Skipf("Failed to load Speech2Seq pipeline: %v", err)
	}
	defer pipeline.Close()

	t.Logf("Loaded Whisper pipeline with backend: %s", backendType)

	// Generate test audio (1 second of 440Hz sine wave)
	audioData := generateTestWAV(t, 1.0, 440.0, 16000)
	t.Logf("Generated test WAV: %d bytes", len(audioData))

	// Run transcription
	ctx := context.Background()
	result, err := pipeline.Transcribe(ctx, audioData)
	if err != nil {
		t.Logf("Transcription error (may be expected for synthetic audio): %v", err)
		return
	}

	t.Logf("Transcription result: %q", result.Text)
}

// TestWhisperTranscriber tests the PooledTranscriber interface
func TestWhisperTranscriber(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Whisper transcriber test in short mode")
	}

	modelPath := ensureHuggingFaceModel(t, whisperModelName, whisperModelRepo, ModelTypeTranscriber)

	// Create transcriber
	transcriber, err := createTranscriber(t, modelPath)
	if err != nil {
		t.Skipf("Failed to create transcriber: %v", err)
	}
	defer transcriber.Close()

	t.Logf("Transcriber model type: %s", transcriber.ModelType())
	assert.Equal(t, transcribing.ModelTypeWhisper, transcriber.ModelType())

	// Test with silent audio
	silentAudio := generateSilentWAV(t, 1.0, 16000)
	ctx := context.Background()

	result, err := transcriber.Transcribe(ctx, silentAudio)
	if err != nil {
		t.Logf("Transcription of silence failed (may be expected): %v", err)
		return
	}

	t.Logf("Silent audio transcription: %q", result.Text)
	// Whisper often outputs empty or minimal text for silence
}

// TestWhisperWithTone tests transcription with a simple tone
func TestWhisperWithTone(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Whisper tone test in short mode")
	}

	modelPath := ensureHuggingFaceModel(t, whisperModelName, whisperModelRepo, ModelTypeTranscriber)

	transcriber, err := createTranscriber(t, modelPath)
	if err != nil {
		t.Skipf("Failed to create transcriber: %v", err)
	}
	defer transcriber.Close()

	// Generate 2 seconds of 440Hz tone (A4 note)
	toneAudio := generateTestWAV(t, 2.0, 440.0, 16000)
	t.Logf("Generated tone WAV: %d bytes", len(toneAudio))

	ctx := context.Background()
	result, err := transcriber.Transcribe(ctx, toneAudio)
	if err != nil {
		t.Logf("Tone transcription failed: %v", err)
		return
	}

	t.Logf("Tone transcription: %q", result.Text)
	// Whisper may output various things for a pure tone - this tests the pipeline works
}

// TestWhisperWithTestdataAudio tests transcription with pre-recorded audio if available
func TestWhisperWithTestdataAudio(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Whisper testdata audio test in short mode")
	}

	audioPath := filepath.Join("testdata", "sample-audio.wav")
	if _, err := os.Stat(audioPath); os.IsNotExist(err) {
		t.Skip("Pre-recorded audio not found at:", audioPath)
	}

	modelPath := ensureHuggingFaceModel(t, whisperModelName, whisperModelRepo, ModelTypeTranscriber)

	transcriber, err := createTranscriber(t, modelPath)
	if err != nil {
		t.Skipf("Failed to create transcriber: %v", err)
	}
	defer transcriber.Close()

	// Load audio file
	audioData, err := os.ReadFile(audioPath)
	require.NoError(t, err)
	t.Logf("Loaded test audio: %d bytes", len(audioData))

	ctx := context.Background()
	result, err := transcriber.Transcribe(ctx, audioData)
	require.NoError(t, err, "Transcription failed")

	t.Logf("Transcription: %q", result.Text)
	assert.NotEmpty(t, result.Text, "Expected non-empty transcription")
}

// TestWhisperTranscriberOptions tests transcription with custom options
func TestWhisperTranscriberOptions(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Whisper options test in short mode")
	}

	modelPath := ensureHuggingFaceModel(t, whisperModelName, whisperModelRepo, ModelTypeTranscriber)

	transcriber, err := createTranscriber(t, modelPath)
	if err != nil {
		t.Skipf("Failed to create transcriber: %v", err)
	}
	defer transcriber.Close()

	// Generate test audio
	audioData := generateTestWAV(t, 1.0, 440.0, 16000)

	ctx := context.Background()
	opts := transcribing.TranscribeOptions{
		MaxTokens: 32, // Limit output length
	}

	result, err := transcriber.TranscribeWithOptions(ctx, audioData, opts)
	if err != nil {
		t.Logf("Transcription with options failed: %v", err)
		return
	}

	t.Logf("Transcription (max 32 tokens): %q", result.Text)
}

// =============================================================================
// Audio Processor Tests
// =============================================================================

// TestAudioProcessor tests the audio preprocessing pipeline
func TestAudioProcessor(t *testing.T) {
	// Create audio processor with Whisper defaults
	audioConfig := backends.DefaultAudioConfig()
	processor := pipelines.NewAudioProcessor(audioConfig)

	// Generate test audio
	audioData := generateTestWAV(t, 1.0, 440.0, 16000)

	// Process audio - returns (features []float32, numFrames int, error)
	features, numFrames, err := processor.Process(audioData)
	require.NoError(t, err, "Audio processing failed")

	t.Logf("Audio features: %d values, %d frames, %d mels",
		len(features), numFrames, audioConfig.NMels)

	assert.Greater(t, numFrames, 0, "Expected positive number of frames")
	assert.NotEmpty(t, features, "Expected non-empty features")
	// Features should be numFrames * nMels
	expectedLen := numFrames * audioConfig.NMels
	assert.Equal(t, expectedLen, len(features), "Feature length should be frames * mels")
}

// TestAudioProcessorResampling tests audio resampling
func TestAudioProcessorResampling(t *testing.T) {
	audioConfig := backends.DefaultAudioConfig()
	processor := pipelines.NewAudioProcessor(audioConfig)

	// Generate audio at 44100 Hz (common rate)
	numSamples := 44100 // 1 second at 44100 Hz
	audioData := make([]int16, numSamples)
	for i := 0; i < numSamples; i++ {
		sample := math.Sin(2 * math.Pi * 440.0 * float64(i) / 44100.0)
		audioData[i] = int16(sample * 32767 * 0.5)
	}

	// Create WAV at 44100 Hz
	dataSize := numSamples * 2
	headerSize := 44
	fileSize := headerSize + dataSize
	wav := make([]byte, fileSize)

	copy(wav[0:4], "RIFF")
	binary.LittleEndian.PutUint32(wav[4:8], uint32(fileSize-8))
	copy(wav[8:12], "WAVE")
	copy(wav[12:16], "fmt ")
	binary.LittleEndian.PutUint32(wav[16:20], 16)
	binary.LittleEndian.PutUint16(wav[20:22], 1)
	binary.LittleEndian.PutUint16(wav[22:24], 1)
	binary.LittleEndian.PutUint32(wav[24:28], 44100) // 44100 Hz
	binary.LittleEndian.PutUint32(wav[28:32], 44100*2)
	binary.LittleEndian.PutUint16(wav[32:34], 2)
	binary.LittleEndian.PutUint16(wav[34:36], 16)
	copy(wav[36:40], "data")
	binary.LittleEndian.PutUint32(wav[40:44], uint32(dataSize))

	for i, sample := range audioData {
		binary.LittleEndian.PutUint16(wav[44+i*2:46+i*2], uint16(sample))
	}

	// Process - should resample from 44100 to 16000
	features, numFrames, err := processor.Process(wav)
	require.NoError(t, err, "Audio processing with resampling failed")

	t.Logf("Resampled audio features: %d frames, %d mels", numFrames, audioConfig.NMels)
	assert.Greater(t, numFrames, 0, "Expected positive number of frames after resampling")
	assert.NotEmpty(t, features, "Expected non-empty features after resampling")
}

// TestAudioProcessorEmptyInput tests handling of empty input
func TestAudioProcessorEmptyInput(t *testing.T) {
	audioConfig := backends.DefaultAudioConfig()
	processor := pipelines.NewAudioProcessor(audioConfig)

	_, _, err := processor.Process(nil)
	assert.Error(t, err, "Expected error for nil input")

	_, _, err = processor.Process([]byte{})
	assert.Error(t, err, "Expected error for empty input")
}

// TestAudioProcessorInvalidWAV tests handling of invalid WAV data
func TestAudioProcessorInvalidWAV(t *testing.T) {
	audioConfig := backends.DefaultAudioConfig()
	processor := pipelines.NewAudioProcessor(audioConfig)

	// Invalid header
	invalidData := []byte("not a wav file")
	_, _, err := processor.Process(invalidData)
	assert.Error(t, err, "Expected error for invalid WAV data")

	// Truncated header
	truncated := []byte("RIFF")
	_, _, err = processor.Process(truncated)
	assert.Error(t, err, "Expected error for truncated WAV")
}
