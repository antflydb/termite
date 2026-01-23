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

package pipelines

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"os"
	"path/filepath"

	"github.com/gomlx/go-huggingface/tokenizers"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
)

// =============================================================================
// Configuration Types
// =============================================================================

// Vision2SeqModelConfig holds parsed configuration for a Vision2Seq model.
// This is loaded from config.json and preprocessor_config.json.
type Vision2SeqModelConfig struct {
	// Path to the model directory
	ModelPath string

	// Paths to ONNX files (if present)
	EncoderPath string
	DecoderPath string

	// Decoder configuration
	DecoderConfig *backends.DecoderConfig

	// Image preprocessing configuration
	ImageConfig *backends.ImageConfig

	// Architecture details for KV-cache
	NumLayers  int
	NumHeads   int
	HeadDim    int
	HiddenSize int
}

// Vision2SeqConfig holds configuration for creating a Vision2SeqPipeline.
type Vision2SeqConfig struct {
	// ImageConfig for preprocessing. If nil, uses model's default.
	ImageConfig *backends.ImageConfig

	// GenerationConfig for text generation. If nil, uses defaults.
	GenerationConfig *backends.GenerationConfig
}

// Vision2SeqResult is an alias for EncoderDecoderResult for backwards compatibility.
type Vision2SeqResult = EncoderDecoderResult

// =============================================================================
// Configuration Loading
// =============================================================================

// LoadVision2SeqModelConfig loads and parses configuration for a Vision2Seq model.
// This is backend-agnostic and can be used by both ONNX and GoMLX backends.
func LoadVision2SeqModelConfig(modelPath string) (*Vision2SeqModelConfig, error) {
	// Find encoder ONNX file
	encoderPath := FindONNXFile(modelPath, []string{
		"encoder.onnx",
		"encoder_model.onnx",
		"vision_encoder.onnx",
	})

	// Find decoder ONNX file
	decoderPath := FindONNXFile(modelPath, []string{
		"decoder_model_merged.onnx", // Preferred: merged decoder with KV-cache
		"decoder_with_past.onnx",
		"decoder.onnx",
		"decoder_model.onnx",
	})

	// Load model configuration from config.json
	rawConfig, err := loadRawVision2SeqConfig(modelPath)
	if err != nil {
		return nil, fmt.Errorf("loading model config: %w", err)
	}

	// Load preprocessor config if available
	preprocConfig := loadPreprocessorConfig(modelPath)

	// Build decoder config
	decoderConfig := buildDecoderConfig(rawConfig)

	// Build image config
	imageConfig := buildImageConfig(rawConfig, preprocConfig)

	// Determine architecture
	numLayers := FirstNonZero(rawConfig.DecoderLayers, rawConfig.NumDecoderLayers, 6)
	numHeads := FirstNonZero(rawConfig.DecoderAttentionHeads, rawConfig.NumAttentionHeads, rawConfig.DecoderNumHeads, 8)
	hiddenSize := FirstNonZero(rawConfig.DecoderHiddenSize, rawConfig.HiddenSize, 768)
	headDim := hiddenSize / numHeads

	return &Vision2SeqModelConfig{
		ModelPath:     modelPath,
		EncoderPath:   encoderPath,
		DecoderPath:   decoderPath,
		DecoderConfig: decoderConfig,
		ImageConfig:   imageConfig,
		NumLayers:     numLayers,
		NumHeads:      numHeads,
		HeadDim:       headDim,
		HiddenSize:    hiddenSize,
	}, nil
}

// IsVision2SeqModel checks if a model path contains a Vision2Seq model.
func IsVision2SeqModel(path string) bool {
	encoderPath := FindONNXFile(path, []string{
		"encoder.onnx",
		"encoder_model.onnx",
		"vision_encoder.onnx",
	})
	decoderPath := FindONNXFile(path, []string{
		"decoder_model_merged.onnx",
		"decoder_with_past.onnx",
		"decoder.onnx",
		"decoder_model.onnx",
	})
	return encoderPath != "" && decoderPath != ""
}

// =============================================================================
// Raw Config Structs and Parsing Helpers
// =============================================================================

// rawVision2SeqConfig represents the model's config.json structure.
type rawVision2SeqConfig struct {
	// Decoder config
	VocabSize           int   `json:"vocab_size"`
	DecoderStartTokenID int32 `json:"decoder_start_token_id"`
	EOSTokenID          any   `json:"eos_token_id"` // Can be int or []int
	BOSTokenID          int32 `json:"bos_token_id"`
	PadTokenID          int32 `json:"pad_token_id"`
	MaxLength           int   `json:"max_length"`

	// Decoder architecture
	DecoderLayers         int `json:"decoder_layers"`
	DecoderAttentionHeads int `json:"decoder_attention_heads"`
	DecoderHiddenSize     int `json:"d_model"`
	NumDecoderLayers      int `json:"num_decoder_layers"`
	NumAttentionHeads     int `json:"num_attention_heads"`
	DecoderNumHeads       int `json:"decoder_num_heads"`
	HiddenSize            int `json:"hidden_size"`

	// Image config (from config.json)
	ImageSize    any       `json:"image_size"`
	DoCenterCrop bool      `json:"do_center_crop"`
	ImageMean    []float32 `json:"image_mean"`
	ImageStd     []float32 `json:"image_std"`
	Size         any       `json:"size"`

	// Encoder config (for vision models)
	EncoderConfig *struct {
		ImageSize int `json:"image_size"`
	} `json:"encoder"`

	// Vision config (for some models)
	VisionConfig *struct {
		ImageSize int `json:"image_size"`
	} `json:"vision_config"`
}

// rawPreprocessorConfig represents preprocessor_config.json
type rawPreprocessorConfig struct {
	ImageMean     []float32 `json:"image_mean"`
	ImageStd      []float32 `json:"image_std"`
	DoNormalize   bool      `json:"do_normalize"`
	DoCenterCrop  bool      `json:"do_center_crop"`
	RescaleFactor float32   `json:"rescale_factor"`
	Size          any       `json:"size"`
	CropSize      any       `json:"crop_size"`
}

// loadRawVision2SeqConfig loads the model configuration from config.json.
func loadRawVision2SeqConfig(path string) (*rawVision2SeqConfig, error) {
	configPath := filepath.Join(path, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("reading config.json: %w", err)
	}

	var config rawVision2SeqConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("parsing config.json: %w", err)
	}

	return &config, nil
}

// loadPreprocessorConfig loads preprocessor_config.json if it exists.
func loadPreprocessorConfig(path string) *rawPreprocessorConfig {
	configPath := filepath.Join(path, "preprocessor_config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil
	}

	var config rawPreprocessorConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil
	}

	return &config
}

// buildDecoderConfig creates a DecoderConfig from the raw config.
func buildDecoderConfig(cfg *rawVision2SeqConfig) *backends.DecoderConfig {
	// Handle eos_token_id which can be int or []int
	var eosTokenID int32
	switch v := cfg.EOSTokenID.(type) {
	case float64:
		eosTokenID = int32(v)
	case []interface{}:
		if len(v) > 0 {
			if f, ok := v[0].(float64); ok {
				eosTokenID = int32(f)
			}
		}
	}

	maxLength := cfg.MaxLength
	if maxLength == 0 {
		maxLength = 512
	}

	numHeads := FirstNonZero(cfg.DecoderAttentionHeads, cfg.NumAttentionHeads, cfg.DecoderNumHeads, 8)
	hiddenSize := FirstNonZero(cfg.DecoderHiddenSize, cfg.HiddenSize, 768)

	return &backends.DecoderConfig{
		VocabSize:           cfg.VocabSize,
		MaxLength:           maxLength,
		EOSTokenID:          eosTokenID,
		BOSTokenID:          cfg.BOSTokenID,
		PadTokenID:          cfg.PadTokenID,
		DecoderStartTokenID: cfg.DecoderStartTokenID,
		NumLayers:           FirstNonZero(cfg.DecoderLayers, cfg.NumDecoderLayers, 6),
		NumHeads:            numHeads,
		HeadDim:             hiddenSize / numHeads,
	}
}

// buildImageConfig creates an ImageConfig from the raw configs.
func buildImageConfig(cfg *rawVision2SeqConfig, preproc *rawPreprocessorConfig) *backends.ImageConfig {
	// Merge preprocessor config if available
	var imageMean, imageStd []float32
	var doCenterCrop bool
	var rescaleFactor float32
	var sizeField any

	if preproc != nil {
		imageMean = preproc.ImageMean
		imageStd = preproc.ImageStd
		doCenterCrop = preproc.DoCenterCrop
		rescaleFactor = preproc.RescaleFactor
		sizeField = preproc.Size
	}

	// Fall back to config.json values
	if len(imageMean) == 0 {
		imageMean = cfg.ImageMean
	}
	if len(imageStd) == 0 {
		imageStd = cfg.ImageStd
	}
	if !doCenterCrop {
		doCenterCrop = cfg.DoCenterCrop
	}
	if sizeField == nil {
		sizeField = cfg.Size
	}

	// Determine image size
	imageSize := 224 // default
	if size := extractImageSize(cfg.ImageSize); size > 0 {
		imageSize = size
	} else if size := extractImageSize(sizeField); size > 0 {
		imageSize = size
	} else if cfg.EncoderConfig != nil && cfg.EncoderConfig.ImageSize > 0 {
		imageSize = cfg.EncoderConfig.ImageSize
	} else if cfg.VisionConfig != nil && cfg.VisionConfig.ImageSize > 0 {
		imageSize = cfg.VisionConfig.ImageSize
	}

	// Default normalization values (ImageNet)
	mean := [3]float32{0.485, 0.456, 0.406}
	std := [3]float32{0.229, 0.224, 0.225}
	if len(imageMean) == 3 {
		copy(mean[:], imageMean)
	}
	if len(imageStd) == 3 {
		copy(std[:], imageStd)
	}

	if rescaleFactor == 0 {
		rescaleFactor = 1.0 / 255.0
	}

	return &backends.ImageConfig{
		Width:         imageSize,
		Height:        imageSize,
		Channels:      3,
		Mean:          mean,
		Std:           std,
		RescaleFactor: rescaleFactor,
		DoCenterCrop:  doCenterCrop,
	}
}

// extractImageSize extracts an integer size from various JSON formats.
func extractImageSize(v any) int {
	switch val := v.(type) {
	case float64:
		return int(val)
	case int:
		return val
	case map[string]interface{}:
		// Handle {height: N, width: N} or {shortest_edge: N}
		if h, ok := val["height"].(float64); ok {
			return int(h)
		}
		if se, ok := val["shortest_edge"].(float64); ok {
			return int(se)
		}
	}
	return 0
}

// =============================================================================
// Model Management (Vision Encoder + Decoder)
// =============================================================================

// Ensure vision2SeqModel implements backends.Model
var _ backends.Model = (*vision2SeqModel)(nil)

// vision2SeqModel implements backends.Model for image-to-text tasks.
// It uses separate encoder and decoder sessions (TrOCR, Donut, etc.).
type vision2SeqModel struct {
	config *Vision2SeqModelConfig

	encoderSession backends.Session
	decoderSession backends.Session

	backendType backends.BackendType
}

// NewVision2SeqModel creates a Model from encoder and decoder sessions.
func NewVision2SeqModel(
	config *Vision2SeqModelConfig,
	encoderSession backends.Session,
	decoderSession backends.Session,
	backendType backends.BackendType,
) backends.Model {
	return &vision2SeqModel{
		config:         config,
		encoderSession: encoderSession,
		decoderSession: decoderSession,
		backendType:    backendType,
	}
}

// LoadVision2SeqModel loads a vision2seq Model using the given session factory.
// It automatically discovers encoder and decoder ONNX files and creates sessions.
func LoadVision2SeqModel(modelPath string, factory backends.SessionFactory, opts ...backends.SessionOption) (backends.Model, error) {
	// Load configuration
	config, err := LoadVision2SeqModelConfig(modelPath)
	if err != nil {
		return nil, fmt.Errorf("loading model config: %w", err)
	}

	if config.EncoderPath == "" {
		return nil, fmt.Errorf("encoder ONNX file not found in %s", modelPath)
	}
	if config.DecoderPath == "" {
		return nil, fmt.Errorf("decoder ONNX file not found in %s", modelPath)
	}

	// Create encoder session
	encoderSession, err := factory.CreateSession(config.EncoderPath, opts...)
	if err != nil {
		return nil, fmt.Errorf("creating encoder session: %w", err)
	}

	// Create decoder session
	decoderSession, err := factory.CreateSession(config.DecoderPath, opts...)
	if err != nil {
		encoderSession.Close()
		return nil, fmt.Errorf("creating decoder session: %w", err)
	}

	return &vision2SeqModel{
		config:         config,
		encoderSession: encoderSession,
		decoderSession: decoderSession,
		backendType:    factory.Backend(),
	}, nil
}

// Forward runs encoder or decoder based on inputs.
// - If ImagePixels is set (and EncoderOutput is nil): runs vision encoder, returns EncoderOutput
// - If EncoderOutput is set: runs decoder step, returns Logits and PastKeyValues
func (m *vision2SeqModel) Forward(ctx context.Context, inputs *backends.ModelInputs) (*backends.ModelOutput, error) {
	if inputs == nil {
		return nil, fmt.Errorf("nil inputs")
	}

	// If encoder output provided, run decoder
	if inputs.EncoderOutput != nil {
		return m.runDecoder(ctx, inputs)
	}

	// Otherwise run encoder on image
	if inputs.ImagePixels == nil || len(inputs.ImagePixels) == 0 {
		return nil, fmt.Errorf("no image pixels or encoder output provided")
	}

	return m.runEncoder(ctx, inputs)
}

// runEncoder encodes preprocessed image pixels into encoder hidden states.
func (m *vision2SeqModel) runEncoder(ctx context.Context, inputs *backends.ModelInputs) (*backends.ModelOutput, error) {
	// Create input tensor [batch, channels, height, width]
	input := backends.NamedTensor{
		Name:  m.getEncoderInputName(),
		Shape: []int64{int64(inputs.ImageBatch), int64(inputs.ImageChannels), int64(inputs.ImageHeight), int64(inputs.ImageWidth)},
		Data:  inputs.ImagePixels,
	}

	// Run encoder
	outputs, err := m.encoderSession.Run([]backends.NamedTensor{input})
	if err != nil {
		return nil, fmt.Errorf("running encoder: %w", err)
	}

	if len(outputs) == 0 {
		return nil, fmt.Errorf("no encoder output")
	}

	// Extract encoder output (typically "last_hidden_state" or first output)
	output := outputs[0]
	if len(output.Shape) < 3 {
		return nil, fmt.Errorf("unexpected encoder output shape: %v", output.Shape)
	}

	// Extract data
	hiddenStates, ok := output.Data.([]float32)
	if !ok {
		return nil, fmt.Errorf("encoder output is not float32")
	}

	encoderOutput := &backends.EncoderOutput{
		HiddenStates: hiddenStates,
		Shape:        [3]int{int(output.Shape[0]), int(output.Shape[1]), int(output.Shape[2])},
	}

	return &backends.ModelOutput{
		EncoderOutput: encoderOutput,
	}, nil
}

// getEncoderInputName returns the expected input name for the encoder.
func (m *vision2SeqModel) getEncoderInputName() string {
	// Check session input info for the actual name
	inputInfo := m.encoderSession.InputInfo()
	if len(inputInfo) > 0 {
		return inputInfo[0].Name
	}
	// Default name for vision encoders
	return "pixel_values"
}

// runDecoder performs one step of autoregressive decoding.
func (m *vision2SeqModel) runDecoder(ctx context.Context, inputs *backends.ModelInputs) (*backends.ModelOutput, error) {
	inputIDs := inputs.InputIDs
	encoderOutput := inputs.EncoderOutput
	pastKeyValues := inputs.PastKeyValues

	batchSize := len(inputIDs)
	if batchSize == 0 {
		return nil, fmt.Errorf("empty input")
	}

	seqLen := len(inputIDs[0])

	// Flatten input IDs to int64 for most models
	flatInputIDs := make([]int64, batchSize*seqLen)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			flatInputIDs[i*seqLen+j] = int64(inputIDs[i][j])
		}
	}

	// Build decoder inputs
	tensorInputs, err := m.buildDecoderInputs(flatInputIDs, batchSize, seqLen, encoderOutput, pastKeyValues)
	if err != nil {
		return nil, fmt.Errorf("building decoder inputs: %w", err)
	}

	// Run decoder
	outputs, err := m.decoderSession.Run(tensorInputs)
	if err != nil {
		return nil, fmt.Errorf("running decoder: %w", err)
	}

	if len(outputs) == 0 {
		return nil, fmt.Errorf("no decoder output")
	}

	// Extract logits (first output)
	logitsOutput := outputs[0]
	logitsData, ok := logitsOutput.Data.([]float32)
	if !ok {
		return nil, fmt.Errorf("logits tensor is not float32")
	}

	logitsShape := logitsOutput.Shape

	// Reshape logits to [batch, vocab_size] (taking last position if sequence)
	vocabSize := int(logitsShape[len(logitsShape)-1])
	logits := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		logits[i] = make([]float32, vocabSize)
		// Take logits from last position
		startIdx := i*seqLen*vocabSize + (seqLen-1)*vocabSize
		copy(logits[i], logitsData[startIdx:startIdx+vocabSize])
	}

	// Extract updated KV-cache from outputs (if present)
	newKVCache := m.extractKVCache(outputs, batchSize, pastKeyValues)

	return &backends.ModelOutput{
		Logits:        logits,
		PastKeyValues: newKVCache,
	}, nil
}

// buildDecoderInputs creates the input tensors for the decoder.
func (m *vision2SeqModel) buildDecoderInputs(inputIDs []int64, batchSize, seqLen int, encoderOutput *backends.EncoderOutput, pastKV *backends.KVCache) ([]backends.NamedTensor, error) {
	var inputs []backends.NamedTensor

	// Get decoder input names from session
	inputInfo := m.decoderSession.InputInfo()
	inputNames := make(map[string]bool)
	for _, info := range inputInfo {
		inputNames[info.Name] = true
	}

	// Add input_ids
	inputs = append(inputs, backends.NamedTensor{
		Name:  m.getDecoderInputIDsName(inputNames),
		Shape: []int64{int64(batchSize), int64(seqLen)},
		Data:  inputIDs,
	})

	// Add encoder hidden states
	if inputNames["encoder_hidden_states"] || inputNames["encoder_outputs"] {
		name := "encoder_hidden_states"
		if inputNames["encoder_outputs"] {
			name = "encoder_outputs"
		}
		inputs = append(inputs, backends.NamedTensor{
			Name:  name,
			Shape: []int64{int64(encoderOutput.Shape[0]), int64(encoderOutput.Shape[1]), int64(encoderOutput.Shape[2])},
			Data:  encoderOutput.HiddenStates,
		})
	}

	// Add encoder attention mask if needed
	if inputNames["encoder_attention_mask"] {
		encSeqLen := encoderOutput.Shape[1]
		mask := make([]int64, batchSize*encSeqLen)
		for i := range mask {
			mask[i] = 1
		}
		inputs = append(inputs, backends.NamedTensor{
			Name:  "encoder_attention_mask",
			Shape: []int64{int64(batchSize), int64(encSeqLen)},
			Data:  mask,
		})
	}

	// Add use_cache_branch if needed
	if inputNames["use_cache_branch"] {
		useCacheBool := []bool{pastKV != nil}
		inputs = append(inputs, backends.NamedTensor{
			Name:  "use_cache_branch",
			Shape: []int64{1},
			Data:  useCacheBool,
		})
	}

	// Add past_key_values inputs if needed
	for _, info := range inputInfo {
		if IsPastKeyValueInput(info.Name) {
			tensor := m.createPastKVTensor(info.Name, pastKV, batchSize)
			inputs = append(inputs, tensor)
		}
	}

	return inputs, nil
}

// getDecoderInputIDsName returns the name for decoder input IDs.
func (m *vision2SeqModel) getDecoderInputIDsName(inputNames map[string]bool) string {
	return GetDecoderInputIDsName(inputNames)
}

// createPastKVTensor creates a tensor for past key/value cache.
func (m *vision2SeqModel) createPastKVTensor(name string, pastKV *backends.KVCache, batchSize int) backends.NamedTensor {
	// If no past KV, create empty tensor
	if pastKV == nil || pastKV.SeqLen == 0 {
		// Create zero-sized tensor for first step
		// Shape: [batch, num_heads, 0, head_dim]
		return backends.NamedTensor{
			Name:  name,
			Shape: []int64{int64(batchSize), int64(m.config.NumHeads), 0, int64(m.config.HeadDim)},
			Data:  []float32{},
		}
	}

	// TODO: Extract the appropriate slice from pastKV based on layer index in name
	// For now, return empty - full implementation would parse layer index from name
	// and extract the corresponding KV tensors
	return backends.NamedTensor{
		Name:  name,
		Shape: []int64{int64(batchSize), int64(m.config.NumHeads), 0, int64(m.config.HeadDim)},
		Data:  []float32{},
	}
}

// extractKVCache extracts the KV-cache from decoder outputs.
func (m *vision2SeqModel) extractKVCache(outputs []backends.NamedTensor, batchSize int, pastKV *backends.KVCache) *backends.KVCache {
	// For now, return nil - full implementation would extract present_key_values
	// from the output tensors and build a KVCache struct
	// This is a simplified implementation that doesn't use KV-cache
	return nil
}

// DecoderConfig returns configuration needed for generation.
func (m *vision2SeqModel) DecoderConfig() *backends.DecoderConfig {
	return m.config.DecoderConfig
}

// ImageConfig returns configuration for image preprocessing.
func (m *vision2SeqModel) ImageConfig() *backends.ImageConfig {
	return m.config.ImageConfig
}

// Close releases resources associated with the model.
func (m *vision2SeqModel) Close() error {
	var errs []error

	if m.encoderSession != nil {
		if err := m.encoderSession.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing encoder: %w", err))
		}
		m.encoderSession = nil
	}

	if m.decoderSession != nil {
		if err := m.decoderSession.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing decoder: %w", err))
		}
		m.decoderSession = nil
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing model: %v", errs)
	}
	return nil
}

// Name returns the model name for logging and debugging.
func (m *vision2SeqModel) Name() string {
	return m.config.ModelPath
}

// Backend returns the backend type this model uses.
func (m *vision2SeqModel) Backend() backends.BackendType {
	return m.backendType
}

// =============================================================================
// Pipeline
// =============================================================================

// Vision2SeqPipeline handles image-to-text tasks like OCR and document understanding.
// It combines image preprocessing, vision encoding, and autoregressive text generation.
type Vision2SeqPipeline struct {
	// EncoderDecoderPipeline provides shared generation functionality.
	*EncoderDecoderPipeline

	// ImageProcessor handles image preprocessing.
	ImageProcessor *ImageProcessor
}

// NewVision2SeqPipeline creates a new Vision2SeqPipeline.
func NewVision2SeqPipeline(
	model backends.Model,
	tokenizer tokenizers.Tokenizer,
	config *Vision2SeqConfig,
) *Vision2SeqPipeline {
	if config == nil {
		config = &Vision2SeqConfig{}
	}

	// Resolve image config
	imageConfig := ResolveImageConfig(model, config.ImageConfig)

	// Create base encoder-decoder pipeline
	base := NewEncoderDecoderPipeline(model, tokenizer, config.GenerationConfig)

	return &Vision2SeqPipeline{
		EncoderDecoderPipeline: base,
		ImageProcessor:         NewImageProcessor(imageConfig),
	}
}

// Run processes an image and generates text.
func (p *Vision2SeqPipeline) Run(ctx context.Context, img image.Image) (*Vision2SeqResult, error) {
	return p.RunWithPrompt(ctx, img, "")
}

// RunWithPrompt processes an image with an optional text prompt.
// The prompt can be used for task-conditioned generation (e.g., Florence-2).
func (p *Vision2SeqPipeline) RunWithPrompt(ctx context.Context, img image.Image, prompt string) (*Vision2SeqResult, error) {
	// Encode image
	encoderOutput, err := p.encodeImage(ctx, img)
	if err != nil {
		return nil, err
	}

	// Get start tokens
	startTokens := p.GetStartTokens(prompt)

	// Generate using shared base
	return p.GenerateFromEncoderOutput(ctx, encoderOutput, startTokens)
}

// RunBytes processes an image from bytes and generates text.
func (p *Vision2SeqPipeline) RunBytes(ctx context.Context, data []byte) (*Vision2SeqResult, error) {
	// Encode image from bytes
	encoderOutput, err := p.encodeImageBytes(ctx, data)
	if err != nil {
		return nil, err
	}

	// Get start tokens (no prompt)
	startTokens := p.GetStartTokens("")

	// Generate using shared base
	return p.GenerateFromEncoderOutput(ctx, encoderOutput, startTokens)
}

// RunBatch processes multiple images and generates text for each.
func (p *Vision2SeqPipeline) RunBatch(ctx context.Context, images []image.Image) ([]*Vision2SeqResult, error) {
	results := make([]*Vision2SeqResult, len(images))

	// Process images one at a time for now
	// TODO: Batch processing with proper batched encoder/decoder
	for i, img := range images {
		result, err := p.Run(ctx, img)
		if err != nil {
			return nil, fmt.Errorf("processing image %d: %w", i, err)
		}
		results[i] = result
	}

	return results, nil
}

// RunWithStreaming processes an image and streams generated tokens.
// The callback is called for each generated token. Return false to stop generation.
func (p *Vision2SeqPipeline) RunWithStreaming(
	ctx context.Context,
	img image.Image,
	prompt string,
	callback func(token int32, text string) bool,
) (*Vision2SeqResult, error) {
	// Encode image
	encoderOutput, err := p.encodeImage(ctx, img)
	if err != nil {
		return nil, err
	}

	// Get start tokens
	startTokens := p.GetStartTokens(prompt)

	// Generate with streaming using shared base
	return p.GenerateFromEncoderOutputStreaming(ctx, encoderOutput, startTokens, callback)
}

// encodeImage preprocesses and encodes an image.
func (p *Vision2SeqPipeline) encodeImage(ctx context.Context, img image.Image) (*backends.EncoderOutput, error) {
	// Preprocess image
	pixels, err := p.ImageProcessor.Process(img)
	if err != nil {
		return nil, fmt.Errorf("preprocessing image: %w", err)
	}

	return p.encodePixels(ctx, pixels)
}

// encodeImageBytes preprocesses and encodes an image from bytes.
func (p *Vision2SeqPipeline) encodeImageBytes(ctx context.Context, data []byte) (*backends.EncoderOutput, error) {
	// Preprocess image from bytes
	pixels, err := p.ImageProcessor.ProcessBytes(data)
	if err != nil {
		return nil, fmt.Errorf("preprocessing image: %w", err)
	}

	return p.encodePixels(ctx, pixels)
}

// encodePixels runs the encoder on preprocessed image pixels.
func (p *Vision2SeqPipeline) encodePixels(ctx context.Context, pixels []float32) (*backends.EncoderOutput, error) {
	cfg := p.ImageProcessor.Config
	batchSize := 1

	// Encode image using Forward with image inputs
	encodeOutput, err := p.Model.Forward(ctx, &backends.ModelInputs{
		ImagePixels:   pixels,
		ImageBatch:    batchSize,
		ImageChannels: cfg.Channels,
		ImageHeight:   cfg.Height,
		ImageWidth:    cfg.Width,
	})
	if err != nil {
		return nil, fmt.Errorf("encoding image: %w", err)
	}

	return encodeOutput.EncoderOutput, nil
}

// =============================================================================
// Loader
// =============================================================================

// LoadVision2SeqPipeline loads a complete Vision2Seq pipeline from a model path.
// It automatically discovers encoder/decoder ONNX files, loads the tokenizer,
// and creates the pipeline.
func LoadVision2SeqPipeline(modelPath string, factory backends.SessionFactory, opts ...backends.SessionOption) (*Vision2SeqPipeline, error) {
	// Load the model
	model, err := LoadVision2SeqModel(modelPath, factory, opts...)
	if err != nil {
		return nil, fmt.Errorf("loading vision2seq model: %w", err)
	}

	// Load tokenizer
	tokenizer, err := LoadTokenizer(modelPath)
	if err != nil {
		model.Close()
		return nil, fmt.Errorf("loading tokenizer: %w", err)
	}

	// Create pipeline with default config
	pipeline := NewVision2SeqPipeline(model, tokenizer, nil)

	return pipeline, nil
}
