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

package transcribing

import (
	"context"
	"fmt"
	"runtime"
	"strings"
	"sync/atomic"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// ModelType represents the type of Speech2Seq model for output parsing.
type ModelType string

const (
	// ModelTypeWhisper is OpenAI's Whisper model (openai/whisper-*)
	ModelTypeWhisper ModelType = "whisper"
	// ModelTypeWav2Vec2 is Facebook's Wav2Vec 2.0 model (facebook/wav2vec2-*)
	ModelTypeWav2Vec2 ModelType = "wav2vec2"
	// ModelTypeHubert is Facebook's HuBERT model (facebook/hubert-*)
	ModelTypeHubert ModelType = "hubert"
	// ModelTypeGeneric is used when the model type is unknown
	ModelTypeGeneric ModelType = "generic"
)

// Result contains the output from transcribing audio.
type Result struct {
	// Text is the transcribed text from the audio
	Text string

	// Language is the detected language (if available)
	Language string

	// Confidence is the confidence score (if available)
	Confidence float32
}

// TranscribeOptions provides advanced options for transcription.
type TranscribeOptions struct {
	// Language forces a specific language (optional, model-dependent)
	Language string

	// MaxTokens overrides the maximum number of tokens to generate (optional)
	MaxTokens int
}

// Transcriber provides speech-to-text transcription for audio.
// It wraps Speech2Seq models (Whisper, Wav2Vec2, HuBERT) to transcribe audio to text.
type Transcriber interface {
	// Transcribe converts audio data to text.
	// The audioData should be raw audio bytes (WAV, MP3, etc.).
	//
	// Returns a Result containing the transcribed text.
	Transcribe(ctx context.Context, audioData []byte) (*Result, error)

	// TranscribeWithOptions converts audio data to text with advanced options.
	TranscribeWithOptions(ctx context.Context, audioData []byte, opts TranscribeOptions) (*Result, error)

	// Close releases model resources.
	Close() error
}

// Ensure PooledTranscriber implements the Transcriber interface
var _ Transcriber = (*PooledTranscriber)(nil)

// PooledTranscriber manages multiple Speech2Seq pipelines for concurrent transcription.
// Each request acquires a pipeline slot via semaphore, enabling true parallelism.
type PooledTranscriber struct {
	pipelines    []*pipelines.Speech2SeqPipeline
	sem          *semaphore.Weighted
	nextPipeline atomic.Uint64
	logger       *zap.Logger
	poolSize     int
	modelType    ModelType
	modelPath    string
}

// PooledTranscriberConfig holds configuration for creating a PooledTranscriber.
type PooledTranscriberConfig struct {
	// ModelPath is the path to the Speech2Seq model.
	ModelPath string

	// PoolSize is the number of concurrent pipelines (0 = auto-detect from CPU count).
	PoolSize int

	// GenerationConfig holds text generation parameters. If nil, uses defaults.
	GenerationConfig *backends.GenerationConfig

	// AudioConfig holds audio preprocessing parameters. If nil, uses model's default.
	AudioConfig *backends.AudioConfig

	// Logger for logging. If nil, uses a no-op logger.
	Logger *zap.Logger
}

// NewPooledTranscriber creates a new pooled transcriber from the given configuration.
// sessionManager is used to load the speech2seq model.
func NewPooledTranscriber(
	cfg *PooledTranscriberConfig,
	sessionManager *backends.SessionManager,
	modelBackends []string,
) (*PooledTranscriber, backends.BackendType, error) {
	if cfg == nil {
		return nil, "", fmt.Errorf("config is required")
	}

	logger := cfg.Logger
	if logger == nil {
		logger = zap.NewNop()
	}

	poolSize := cfg.PoolSize
	if poolSize <= 0 {
		poolSize = min(runtime.NumCPU(), 4)
	}

	// Detect model type from path
	modelType := detectModelType(cfg.ModelPath)
	logger.Info("Detected transcriber model type",
		zap.String("path", cfg.ModelPath),
		zap.String("type", string(modelType)))

	// Build pipeline options
	var opts []pipelines.Speech2SeqPipelineOption
	if cfg.AudioConfig != nil {
		opts = append(opts, pipelines.WithSpeech2SeqAudioConfig(cfg.AudioConfig))
	}
	if cfg.GenerationConfig != nil {
		opts = append(opts, pipelines.WithSpeech2SeqGenerationConfig(cfg.GenerationConfig))
	}

	// Create pooled pipelines using LoadSpeech2SeqPipeline
	pipelineSlice := make([]*pipelines.Speech2SeqPipeline, poolSize)
	var backendType backends.BackendType

	for i := 0; i < poolSize; i++ {
		pipeline, bt, err := pipelines.LoadSpeech2SeqPipeline(
			cfg.ModelPath,
			sessionManager,
			modelBackends,
			opts...,
		)
		if err != nil {
			// Clean up already-created pipelines
			for j := 0; j < i; j++ {
				if pipelineSlice[j] != nil {
					_ = pipelineSlice[j].Close()
				}
			}
			return nil, "", fmt.Errorf("loading Speech2Seq pipeline %d: %w", i, err)
		}
		pipelineSlice[i] = pipeline
		backendType = bt
	}

	transcriber := &PooledTranscriber{
		pipelines: pipelineSlice,
		sem:       semaphore.NewWeighted(int64(poolSize)),
		logger:    logger,
		poolSize:  poolSize,
		modelType: modelType,
		modelPath: cfg.ModelPath,
	}

	logger.Info("Created pooled transcriber",
		zap.Int("poolSize", poolSize),
		zap.String("backend", string(backendType)),
		zap.String("modelType", string(modelType)))

	return transcriber, backendType, nil
}

// detectModelType determines the model type from the model path.
func detectModelType(modelPath string) ModelType {
	pathLower := strings.ToLower(modelPath)

	if strings.Contains(pathLower, "whisper") {
		return ModelTypeWhisper
	}
	if strings.Contains(pathLower, "wav2vec2") || strings.Contains(pathLower, "wav2vec-2") {
		return ModelTypeWav2Vec2
	}
	if strings.Contains(pathLower, "hubert") {
		return ModelTypeHubert
	}

	return ModelTypeGeneric
}

// Transcribe converts audio data to text using the Speech2Seq model.
func (t *PooledTranscriber) Transcribe(ctx context.Context, audioData []byte) (*Result, error) {
	return t.TranscribeWithOptions(ctx, audioData, TranscribeOptions{})
}

// TranscribeWithOptions converts audio data to text with advanced options.
func (t *PooledTranscriber) TranscribeWithOptions(ctx context.Context, audioData []byte, opts TranscribeOptions) (*Result, error) {
	if len(audioData) == 0 {
		return nil, fmt.Errorf("no audio data provided")
	}

	// Acquire a pipeline slot
	if err := t.sem.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}
	defer t.sem.Release(1)

	// Get the next pipeline in round-robin fashion
	idx := t.nextPipeline.Add(1) - 1
	pipeline := t.pipelines[idx%uint64(t.poolSize)]

	// Temporarily override max tokens if specified
	originalMaxTokens := pipeline.GenerationConfig.MaxNewTokens
	if opts.MaxTokens > 0 {
		pipeline.GenerationConfig.MaxNewTokens = opts.MaxTokens
	}
	defer func() {
		pipeline.GenerationConfig.MaxNewTokens = originalMaxTokens
	}()

	// Run transcription
	output, err := pipeline.Transcribe(ctx, audioData)
	if err != nil {
		return nil, fmt.Errorf("running Speech2Seq inference: %w", err)
	}

	result := t.parseOutput(output)

	t.logger.Debug("Transcription completed",
		zap.Int("audioBytes", len(audioData)),
		zap.Int("textLen", len(result.Text)),
		zap.String("language", result.Language))

	return result, nil
}

// parseOutput converts the raw pipeline output to a Result.
func (t *PooledTranscriber) parseOutput(output *pipelines.Speech2SeqResult) *Result {
	result := &Result{
		Text: strings.TrimSpace(output.Text),
	}

	// Model-specific output parsing
	switch t.modelType {
	case ModelTypeWhisper:
		// Whisper may include language tokens at the start
		result.Text = cleanWhisperOutput(result.Text)
		// Could extract language from special tokens if present

	case ModelTypeWav2Vec2, ModelTypeHubert:
		// These models typically output clean text
		result.Text = strings.TrimSpace(result.Text)

	case ModelTypeGeneric:
		// Generic cleanup
		result.Text = strings.TrimSpace(result.Text)
	}

	return result
}

// cleanWhisperOutput removes Whisper-specific tokens from output.
func cleanWhisperOutput(text string) string {
	// Remove common Whisper special tokens
	text = strings.TrimSpace(text)

	// Remove language tags like <|en|>
	for strings.HasPrefix(text, "<|") {
		endIdx := strings.Index(text, "|>")
		if endIdx == -1 {
			break
		}
		text = strings.TrimSpace(text[endIdx+2:])
	}

	// Remove timestamp tokens like <|0.00|>
	for strings.Contains(text, "<|") {
		startIdx := strings.Index(text, "<|")
		endIdx := strings.Index(text[startIdx:], "|>")
		if endIdx == -1 {
			break
		}
		text = text[:startIdx] + text[startIdx+endIdx+2:]
	}

	return strings.TrimSpace(text)
}

// Close releases all pipeline resources.
func (t *PooledTranscriber) Close() error {
	t.logger.Info("Closing pooled transcriber", zap.Int("poolSize", t.poolSize))

	var errs []error

	// Close all pipelines
	for i, pipeline := range t.pipelines {
		if pipeline != nil {
			if err := pipeline.Close(); err != nil {
				t.logger.Warn("Error closing pipeline",
					zap.Int("index", i),
					zap.Error(err))
				errs = append(errs, err)
			}
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing transcriber: %v", errs)
	}
	return nil
}

// ModelType returns the detected model type.
func (t *PooledTranscriber) ModelType() ModelType {
	return t.modelType
}
