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

package seq2seq

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/util/imageutil"
	"go.uber.org/zap"
	_ "golang.org/x/image/webp"
)

// Ensure T5Gemma2Generator implements the Model and Decoder interfaces
var _ Model = (*T5Gemma2Generator)(nil)
var _ Decoder = (*T5Gemma2Generator)(nil)

// T5Gemma2Generator implements multimodal seq2seq text generation using T5Gemma-2.
// It supports text-to-text and image+text-to-text generation.
//
// T5Gemma-2 is a multimodal encoder-decoder model from Google's Gemma 3 family,
// trained with UL2 objective. It can generate text conditioned on both text and images.
//
// Build with: CGO_ENABLED=1 go build -tags="onnx,ORT"
type T5Gemma2Generator struct {
	session         *khugot.Session
	seq2seqPipeline *pipelines.Seq2SeqPipeline
	visionPipeline  *pipelines.FeatureExtractionPipeline
	logger          *zap.Logger
	sessionShared   bool
	config          T5Gemma2GeneratorConfig
	modelPath       string

	// Lazy loading for vision encoder
	hasVisionFile bool       // true if vision_encoder.onnx exists
	visionOnce    sync.Once  // ensures vision pipeline loaded only once
	visionLoadErr error      // captures any error during lazy load
}

// T5Gemma2GeneratorConfig holds configuration for T5Gemma-2 generation
type T5Gemma2GeneratorConfig struct {
	// ModelID is the original HuggingFace model ID
	ModelID string `json:"model_id"`
	// ModelType should be "t5gemma2"
	ModelType string `json:"model_type"`
	// Task indicates the model's intended use
	Task string `json:"task"`
	// MaxNewTokens is the maximum number of tokens to generate
	MaxNewTokens int `json:"max_new_tokens"`
	// NumBeams is the number of beams for beam search (1 = greedy)
	NumBeams int `json:"num_beams"`
	// DoSample enables sampling instead of greedy/beam search
	DoSample bool `json:"do_sample"`
	// Temperature controls randomness (used when DoSample=true)
	Temperature float32 `json:"temperature"`
	// TopP is the nucleus sampling probability (used when DoSample=true)
	TopP float32 `json:"top_p"`
	// RepetitionPenalty penalizes repeated tokens (1.0 = no penalty, >1.0 = penalize)
	RepetitionPenalty float32 `json:"repetition_penalty"`
	// ImageSize for vision encoder preprocessing
	ImageSize int `json:"image_size"`
	// HiddenSize of the model
	HiddenSize int `json:"hidden_size"`
}

// MultimodalInput represents input that can contain both text and images
type MultimodalInput struct {
	// Text is the input text prompt
	Text string
	// Images contains base64-encoded or raw image data
	Images [][]byte
}

// NewT5Gemma2Generator creates a new T5Gemma-2 generator.
// The directory should contain:
//   - encoder.onnx (text + multimodal encoder)
//   - decoder.onnx (with past_key_values)
//   - decoder-init.onnx (without past_key_values)
//   - vision_encoder.onnx (SigLIP vision encoder)
//   - config.json or t5gemma2_config.json
//   - tokenizer.json
//
// Build with -tags="onnx,ORT" to enable this generator.
func NewT5Gemma2Generator(modelPath string, logger *zap.Logger) (*T5Gemma2Generator, error) {
	return NewT5Gemma2GeneratorWithSession(modelPath, nil, logger)
}

// NewT5Gemma2GeneratorWithSession creates a new T5Gemma-2 generator with an optional shared session.
func NewT5Gemma2GeneratorWithSession(modelPath string, sharedSession *khugot.Session, logger *zap.Logger) (*T5Gemma2Generator, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Initializing T5Gemma-2 generator",
		zap.String("modelPath", modelPath),
		zap.String("backend", hugot.BackendName()))

	// Load configuration
	config, err := loadT5Gemma2GeneratorConfig(modelPath)
	if err != nil {
		return nil, fmt.Errorf("loading T5Gemma-2 config: %w", err)
	}

	// Verify required files exist
	requiredFiles := []string{"encoder.onnx", "decoder.onnx", "decoder-init.onnx"}
	for _, file := range requiredFiles {
		filePath := filepath.Join(modelPath, file)
		if _, err := os.Stat(filePath); err != nil {
			return nil, fmt.Errorf("required file not found: %s", filePath)
		}
	}

	// Check for optional vision encoder
	visionPath := filepath.Join(modelPath, "vision_encoder.onnx")
	hasVision := false
	if _, err := os.Stat(visionPath); err == nil {
		hasVision = true
	}

	// Create or reuse session
	session, err := hugot.NewSessionOrUseExisting(sharedSession)
	if err != nil {
		return nil, fmt.Errorf("creating hugot session: %w", err)
	}
	sessionShared := (sharedSession != nil)

	// Create Seq2Seq pipeline for text generation
	pipelineName := fmt.Sprintf("t5gemma2-seq2seq:%s", filepath.Base(modelPath))
	pipelineOptions := []khugot.Seq2SeqOption{
		pipelines.WithSeq2SeqMaxTokens(config.MaxNewTokens),
		// T5Gemma2 models produce a garbage first token due to decoder_start_token_id
		// being None in the model config. Skip it in output.
		pipelines.WithSkipFirstToken(true),
	}

	// Apply repetition penalty if configured (default is 1.2)
	if config.RepetitionPenalty > 0 {
		pipelineOptions = append(pipelineOptions,
			pipelines.WithRepetitionPenalty(config.RepetitionPenalty))
	}

	if config.DoSample && config.TopP > 0 && config.Temperature > 0 {
		pipelineOptions = append(pipelineOptions,
			pipelines.WithSampling(config.TopP, config.Temperature))
	}

	pipelineConfig := khugot.Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      pipelineName,
		Options:   pipelineOptions,
	}

	seq2seqPipeline, err := khugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		if !sessionShared {
			session.Destroy()
		}
		return nil, fmt.Errorf("creating Seq2Seq pipeline: %w", err)
	}

	// Vision pipeline is loaded lazily on first image request to save memory.
	// Only log whether the file exists, don't load it yet.
	if hasVision {
		logger.Info("T5Gemma-2 generator initialized (vision available, will load on first use)",
			zap.String("task", config.Task),
			zap.Int("max_new_tokens", config.MaxNewTokens),
			zap.Int("imageSize", config.ImageSize))
	} else {
		logger.Info("T5Gemma-2 generator initialized (text-only mode)",
			zap.String("task", config.Task),
			zap.Int("max_new_tokens", config.MaxNewTokens))
	}

	return &T5Gemma2Generator{
		session:         session,
		seq2seqPipeline: seq2seqPipeline,
		visionPipeline:  nil, // Loaded lazily
		logger:          logger,
		sessionShared:   sessionShared,
		config:          config,
		modelPath:       modelPath,
		hasVisionFile:   hasVision,
	}, nil
}

// NewT5Gemma2GeneratorWithSessionManager creates a new T5Gemma-2 generator using a SessionManager.
func NewT5Gemma2GeneratorWithSessionManager(modelPath string, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*T5Gemma2Generator, hugot.BackendType, error) {
	if sessionManager == nil {
		gen, err := NewT5Gemma2GeneratorWithSession(modelPath, nil, logger)
		if err != nil {
			return nil, "", err
		}
		return gen, hugot.BackendType(""), nil
	}

	// T5Gemma-2 requires ONNX Runtime
	if modelBackends == nil {
		modelBackends = []string{"onnx"}
	}
	session, backendUsed, err := sessionManager.GetSessionForModel(modelBackends)
	if err != nil {
		return nil, "", fmt.Errorf("getting session from manager: %w", err)
	}

	gen, err := NewT5Gemma2GeneratorWithSession(modelPath, session, logger)
	if err != nil {
		return nil, "", err
	}

	gen.sessionShared = true
	return gen, backendUsed, nil
}

// Generate runs text generation on the given text inputs.
func (g *T5Gemma2Generator) Generate(ctx context.Context, inputs []string) (*GeneratedOutput, error) {
	if len(inputs) == 0 {
		return &GeneratedOutput{
			Texts:  [][]string{},
			Tokens: [][][]uint32{},
		}, nil
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	g.logger.Debug("Starting T5Gemma-2 generation",
		zap.Int("num_inputs", len(inputs)))

	output, err := g.seq2seqPipeline.RunPipeline(inputs)
	if err != nil {
		g.logger.Error("T5Gemma-2 generation failed", zap.Error(err))
		return nil, fmt.Errorf("running T5Gemma-2 pipeline: %w", err)
	}

	g.logger.Debug("T5Gemma-2 generation completed",
		zap.Int("num_inputs", len(inputs)),
		zap.Int("total_outputs", len(output.GeneratedTexts)))

	return &GeneratedOutput{
		Texts:  output.GeneratedTexts,
		Tokens: output.GeneratedTokens,
	}, nil
}

// GenerateMultimodal generates text from multimodal inputs (text + images).
//
// CURRENT LIMITATION: This method validates and processes images through the vision
// encoder but does not yet inject the image embeddings into the text generation.
// The seq2seq pipeline currently only processes the text with <image> placeholders.
// Full multimodal generation (where image embeddings condition the output) requires
// modifications to the hugot Seq2SeqPipeline to accept external embeddings.
//
// Use ProcessImage() to get image embeddings for custom multimodal workflows.
func (g *T5Gemma2Generator) GenerateMultimodal(ctx context.Context, inputs []MultimodalInput) (*GeneratedOutput, error) {
	if len(inputs) == 0 {
		return &GeneratedOutput{
			Texts:  [][]string{},
			Tokens: [][][]uint32{},
		}, nil
	}

	// Load vision pipeline lazily when we have images to process
	hasImages := false
	for _, input := range inputs {
		if len(input.Images) > 0 {
			hasImages = true
			break
		}
	}

	if hasImages {
		if err := g.loadVisionPipeline(); err != nil {
			return nil, fmt.Errorf("multimodal generation: %w", err)
		}
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Validate and process images through the vision pipeline
	// This ensures images are valid even though embeddings aren't used yet
	for i, input := range inputs {
		for j, imgData := range input.Images {
			if _, err := g.ProcessImage(imgData); err != nil {
				return nil, fmt.Errorf("processing image %d for input %d: %w", j, i, err)
			}
		}
	}

	// Process text inputs with image placeholders
	// Note: Image embeddings are validated above but not injected into generation yet
	textInputs := make([]string, len(inputs))
	for i, input := range inputs {
		// Prepend image tokens for each image
		prefix := ""
		for range input.Images {
			prefix += "<image> "
		}
		textInputs[i] = prefix + input.Text
	}

	return g.Generate(ctx, textInputs)
}

// ProcessImage encodes an image through the vision encoder.
// Returns the image embeddings that can be used for conditioning.
// The vision encoder is loaded lazily on first call.
func (g *T5Gemma2Generator) ProcessImage(imageData []byte) ([]float32, error) {
	if err := g.loadVisionPipeline(); err != nil {
		return nil, err
	}

	// Decode image
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return nil, fmt.Errorf("decoding image: %w", err)
	}

	// Run through vision pipeline
	output, err := g.visionPipeline.RunWithImages([]image.Image{img})
	if err != nil {
		return nil, fmt.Errorf("running vision pipeline: %w", err)
	}

	if len(output.Embeddings) == 0 || len(output.Embeddings[0]) == 0 {
		return nil, errors.New("no embedding returned from vision pipeline")
	}

	return output.Embeddings[0], nil
}

// HasVision returns true if this generator supports image input.
// This checks if the vision encoder file exists, not if it's currently loaded.
func (g *T5Gemma2Generator) HasVision() bool {
	return g.hasVisionFile
}

// IsVisionLoaded returns true if the vision encoder is currently loaded in memory.
func (g *T5Gemma2Generator) IsVisionLoaded() bool {
	return g.visionPipeline != nil
}

// loadVisionPipeline lazily loads the vision encoder pipeline.
// Thread-safe: uses sync.Once to ensure only one load attempt.
func (g *T5Gemma2Generator) loadVisionPipeline() error {
	g.visionOnce.Do(func() {
		if !g.hasVisionFile {
			g.visionLoadErr = errors.New("vision encoder not available: vision_encoder.onnx not found")
			return
		}

		g.logger.Info("Lazily loading vision encoder pipeline",
			zap.String("modelPath", g.modelPath))

		imageSize := g.config.ImageSize
		if imageSize == 0 {
			imageSize = 896 // Default for T5Gemma-2 (SigLIP)
		}

		visionPipelineName := fmt.Sprintf("t5gemma2-vision:%s", filepath.Base(g.modelPath))
		visionConfig := khugot.FeatureExtractionConfig{
			ModelPath:    g.modelPath,
			Name:         visionPipelineName,
			OnnxFilename: "vision_encoder.onnx",
			Options: []backends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
				pipelines.WithImageMode(),
				pipelines.WithPreprocessSteps[*pipelines.FeatureExtractionPipeline](
					imageutil.ResizeStep(imageSize),
					imageutil.CenterCropStep(imageSize, imageSize),
				),
				pipelines.WithNormalizationSteps[*pipelines.FeatureExtractionPipeline](
					imageutil.RescaleStep(),
					imageutil.CLIPPixelNormalizationStep(),
				),
				pipelines.WithNCHWFormat[*pipelines.FeatureExtractionPipeline](),
			},
		}

		pipeline, err := khugot.NewPipeline(g.session, visionConfig)
		if err != nil {
			g.visionLoadErr = fmt.Errorf("loading vision pipeline: %w", err)
			return
		}

		g.visionPipeline = pipeline
		g.logger.Info("Vision encoder pipeline loaded successfully",
			zap.Int("imageSize", imageSize))
	})
	return g.visionLoadErr
}

// Config returns the generator configuration.
func (g *T5Gemma2Generator) Config() T5Gemma2GeneratorConfig {
	return g.config
}

// HiddenSize returns the expected embedding dimension for the model.
// This is used to validate embeddings passed to DecodeFromEmbeddings.
func (g *T5Gemma2Generator) HiddenSize() int {
	return g.config.HiddenSize
}

// DecodeFromEmbeddings generates text from pre-computed encoder hidden states.
// This enables embedding-to-text generation workflows where custom embeddings
// (from external sources, manipulated vectors, or vision encoders) are used
// directly for text generation without running the encoder.
//
// The embeddings are used as encoder_hidden_states in the decoder's cross-attention
// mechanism, allowing the decoder to generate text conditioned on these embeddings.
//
// Example use cases:
//   - Embedding inversion: Reconstruct text from embeddings
//   - Custom embedding injection: Use manipulated or external embeddings
//   - Cross-modal generation: Convert vision embeddings to text descriptions
func (g *T5Gemma2Generator) DecodeFromEmbeddings(
	ctx context.Context,
	input *DecoderInput,
	opts DecodeOptions,
) (*GeneratedOutput, error) {
	if len(input.EncoderHiddenStates) == 0 {
		return nil, errors.New("encoder hidden states are required")
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Validate embedding dimensions
	expectedDim := g.config.HiddenSize
	for i, emb := range input.EncoderHiddenStates {
		if len(emb) != expectedDim {
			return nil, fmt.Errorf("embedding %d dimension mismatch: got %d, expected %d",
				i, len(emb), expectedDim)
		}
	}

	g.logger.Debug("Starting T5Gemma-2 decode from embeddings",
		zap.Int("num_embeddings", len(input.EncoderHiddenStates)),
		zap.Int("hidden_size", expectedDim),
		zap.Int("max_tokens", opts.MaxTokens))

	// Apply default options if not specified
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = g.config.MaxNewTokens
	}
	if opts.Temperature <= 0 {
		opts.Temperature = g.config.Temperature
	}
	if opts.TopP <= 0 {
		opts.TopP = g.config.TopP
	}
	if opts.RepetitionPenalty <= 0 {
		opts.RepetitionPenalty = g.config.RepetitionPenalty
	}

	// Run generation using the decoder with custom encoder hidden states
	output, err := g.runDecoderWithEmbeddings(ctx, input, opts)
	if err != nil {
		g.logger.Error("T5Gemma-2 decode from embeddings failed", zap.Error(err))
		return nil, fmt.Errorf("decoding from embeddings: %w", err)
	}

	g.logger.Debug("T5Gemma-2 decode from embeddings completed",
		zap.Int("num_outputs", len(output.Texts)))

	return output, nil
}

// Close releases resources.
func (g *T5Gemma2Generator) Close() error {
	var errs []error

	if g.seq2seqPipeline != nil {
		if err := g.seq2seqPipeline.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("destroying seq2seq pipeline: %w", err))
		}
	}

	// Note: FeatureExtractionPipeline doesn't have a Destroy method,
	// it's cleaned up when the session is destroyed

	if g.session != nil && !g.sessionShared {
		g.logger.Info("Destroying Hugot session (owned by this T5Gemma-2 generator)")
		g.session.Destroy()
	}

	return errors.Join(errs...)
}

// loadT5Gemma2GeneratorConfig loads T5Gemma-2 generation config from model directory
func loadT5Gemma2GeneratorConfig(modelPath string) (T5Gemma2GeneratorConfig, error) {
	config := T5Gemma2GeneratorConfig{
		MaxNewTokens:      256,
		NumBeams:          1,
		DoSample:          false,
		Temperature:       1.0,
		TopP:              1.0,
		RepetitionPenalty: 1.2, // Default penalty to avoid degenerate repetition
		ImageSize:         896,
		HiddenSize:        640,
	}

	// Try t5gemma2_config.json first, then config.json
	configPaths := []string{
		filepath.Join(modelPath, "t5gemma2_config.json"),
		filepath.Join(modelPath, "config.json"),
	}

	for _, path := range configPaths {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}

		// Parse config with nested generation_config
		var rawConfig struct {
			ModelID   string `json:"model_id"`
			ModelType string `json:"model_type"`
			Task      string `json:"task"`
			HiddenSize int `json:"hidden_size"`
			GenerationConfig struct {
				MaxNewTokens      int     `json:"max_new_tokens"`
				NumBeams          int     `json:"num_beams"`
				DoSample          bool    `json:"do_sample"`
				Temperature       float32 `json:"temperature"`
				TopP              float32 `json:"top_p"`
				RepetitionPenalty float32 `json:"repetition_penalty"`
			} `json:"generation_config"`
			Encoder struct {
				VisionConfig struct {
					ImageSize int `json:"image_size"`
				} `json:"vision_config"`
			} `json:"encoder"`
			Decoder struct {
				HiddenSize int `json:"hidden_size"`
			} `json:"decoder"`
		}

		if err := json.Unmarshal(data, &rawConfig); err != nil {
			continue
		}

		config.ModelID = rawConfig.ModelID
		config.ModelType = rawConfig.ModelType
		config.Task = rawConfig.Task

		if rawConfig.GenerationConfig.MaxNewTokens > 0 {
			config.MaxNewTokens = rawConfig.GenerationConfig.MaxNewTokens
		}
		if rawConfig.GenerationConfig.NumBeams > 0 {
			config.NumBeams = rawConfig.GenerationConfig.NumBeams
		}
		config.DoSample = rawConfig.GenerationConfig.DoSample
		if rawConfig.GenerationConfig.Temperature > 0 {
			config.Temperature = rawConfig.GenerationConfig.Temperature
		}
		if rawConfig.GenerationConfig.TopP > 0 {
			config.TopP = rawConfig.GenerationConfig.TopP
		}
		if rawConfig.GenerationConfig.RepetitionPenalty > 0 {
			config.RepetitionPenalty = rawConfig.GenerationConfig.RepetitionPenalty
		}

		if rawConfig.Encoder.VisionConfig.ImageSize > 0 {
			config.ImageSize = rawConfig.Encoder.VisionConfig.ImageSize
		}
		if rawConfig.HiddenSize > 0 {
			config.HiddenSize = rawConfig.HiddenSize
		} else if rawConfig.Decoder.HiddenSize > 0 {
			config.HiddenSize = rawConfig.Decoder.HiddenSize
		}

		return config, nil
	}

	return config, nil
}

// IsT5Gemma2GeneratorModel checks if a model directory contains T5Gemma-2 generator files.
func IsT5Gemma2GeneratorModel(modelPath string) bool {
	// Must have standard seq2seq files
	requiredFiles := []string{"encoder.onnx", "decoder.onnx", "decoder-init.onnx"}
	for _, file := range requiredFiles {
		filePath := filepath.Join(modelPath, file)
		if _, err := os.Stat(filePath); err != nil {
			return false
		}
	}

	// Check config for model_type: "t5gemma2"
	configPath := filepath.Join(modelPath, "config.json")
	if data, err := os.ReadFile(configPath); err == nil {
		var config struct {
			ModelType string `json:"model_type"`
		}
		if err := json.Unmarshal(data, &config); err == nil {
			return config.ModelType == "t5gemma2"
		}
	}

	// Also check t5gemma2_config.json
	t5gemma2ConfigPath := filepath.Join(modelPath, "t5gemma2_config.json")
	if _, err := os.Stat(t5gemma2ConfigPath); err == nil {
		return true
	}

	return false
}

// DecodeBase64Image decodes a base64-encoded image (data URI or raw base64).
func DecodeBase64Image(input string) ([]byte, error) {
	// Handle data URI format: data:image/png;base64,...
	if strings.HasPrefix(input, "data:") {
		parts := strings.SplitN(input, ",", 2)
		if len(parts) != 2 {
			return nil, errors.New("invalid data URI format")
		}
		input = parts[1]
	}

	return base64.StdEncoding.DecodeString(input)
}
