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
	"os"
	"path/filepath"
	"strings"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/tokenizers"
)

// =============================================================================
// Moondream2 Model Detection
// =============================================================================

// IsMoondream2Model checks if a model path contains a Moondream2 model.
// Moondream2 is detected by the presence of vision_encoder.onnx, projection.onnx,
// and decoder_model.onnx (the 3-session architecture with SigLIP + Phi-2).
func IsMoondream2Model(path string) bool {
	visionEncoder := FindONNXFile(path, []string{"vision_encoder.onnx"})
	projection := FindONNXFile(path, []string{"projection.onnx"})
	decoder := FindONNXFile(path, []string{"decoder_model.onnx", "decoder.onnx"})

	// Also check the model name contains "moondream" to distinguish from Florence-2
	// which also has vision_encoder.onnx but uses a different architecture
	pathLower := strings.ToLower(filepath.Base(path))
	isMoondreamName := strings.Contains(pathLower, "moondream")

	return visionEncoder != "" && projection != "" && decoder != "" && isMoondreamName
}

// =============================================================================
// Moondream2 Model Configuration
// =============================================================================

// Moondream2ModelConfig holds parsed configuration for a Moondream2 model.
type Moondream2ModelConfig struct {
	// Path to the model directory
	ModelPath string

	// Paths to ONNX files
	VisionEncoderPath string
	ProjectionPath    string
	DecoderPath       string

	// DecoderConfig holds decoder configuration
	DecoderConfig *backends.DecoderConfig

	// ImageConfig holds image preprocessing configuration
	ImageConfig *backends.ImageConfig

	// Architecture details
	NumLayers    int
	NumHeads     int
	HeadDim      int
	HiddenSize   int
	VisionHidden int // Vision encoder hidden size (may differ from text)
}

// LoadMoondream2ModelConfig loads configuration for a Moondream2 model.
func LoadMoondream2ModelConfig(modelPath string) (*Moondream2ModelConfig, error) {
	// Find ONNX files
	visionEncoderPath := FindONNXFile(modelPath, []string{"vision_encoder.onnx"})
	projectionPath := FindONNXFile(modelPath, []string{"projection.onnx"})
	decoderPath := FindONNXFile(modelPath, []string{"decoder_model.onnx", "decoder.onnx"})

	// Load config.json
	configPath := filepath.Join(modelPath, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("reading config.json: %w", err)
	}

	var rawConfig map[string]any
	if err := json.Unmarshal(configData, &rawConfig); err != nil {
		return nil, fmt.Errorf("parsing config.json: %w", err)
	}

	config := &Moondream2ModelConfig{
		ModelPath:         modelPath,
		VisionEncoderPath: visionEncoderPath,
		ProjectionPath:    projectionPath,
		DecoderPath:       decoderPath,
	}

	// Extract architecture details (Moondream2 uses Phi-2 decoder)
	config.HiddenSize = getIntFromMoondreamConfig(rawConfig, "hidden_size", 2048)
	config.NumLayers = getIntFromMoondreamConfig(rawConfig, "num_hidden_layers", 24)
	config.NumHeads = getIntFromMoondreamConfig(rawConfig, "num_attention_heads", 32)
	config.HeadDim = config.HiddenSize / config.NumHeads
	config.VisionHidden = getIntFromMoondreamConfig(rawConfig, "vision_hidden_size", 1152)

	// Build decoder config (Moondream2 uses GPT-2 tokenizer)
	config.DecoderConfig = &backends.DecoderConfig{
		VocabSize:           getIntFromMoondreamConfig(rawConfig, "vocab_size", 51200),
		MaxLength:           getIntFromMoondreamConfig(rawConfig, "max_length", 2048),
		BOSTokenID:          int32(getIntFromMoondreamConfig(rawConfig, "bos_token_id", 50256)),
		EOSTokenID:          int32(getIntFromMoondreamConfig(rawConfig, "eos_token_id", 50256)),
		PadTokenID:          int32(getIntFromMoondreamConfig(rawConfig, "pad_token_id", 50256)),
		DecoderStartTokenID: int32(getIntFromMoondreamConfig(rawConfig, "bos_token_id", 50256)),
		NumLayers:           config.NumLayers,
		NumHeads:            config.NumHeads,
		HeadDim:             config.HeadDim,
	}

	// Build image config (Moondream2 uses SigLIP vision encoder)
	imageSize := getIntFromMoondreamConfig(rawConfig, "image_size", 378)
	config.ImageConfig = &backends.ImageConfig{
		Width:         imageSize,
		Height:        imageSize,
		Channels:      3,
		Mean:          [3]float32{0.5, 0.5, 0.5},  // SigLIP normalization
		Std:           [3]float32{0.5, 0.5, 0.5},  // SigLIP normalization
		RescaleFactor: 1.0 / 255.0,
	}

	// Try to load preprocessor_config.json for more accurate image config
	preprocPath := filepath.Join(modelPath, "preprocessor_config.json")
	if preprocData, err := os.ReadFile(preprocPath); err == nil {
		var preproc map[string]any
		if json.Unmarshal(preprocData, &preproc) == nil {
			if size := getIntFromMoondreamConfig(preproc, "size", 0); size > 0 {
				config.ImageConfig.Width = size
				config.ImageConfig.Height = size
			}
			if mean, ok := preproc["image_mean"].([]any); ok && len(mean) == 3 {
				for i, v := range mean {
					if f, ok := v.(float64); ok {
						config.ImageConfig.Mean[i] = float32(f)
					}
				}
			}
			if std, ok := preproc["image_std"].([]any); ok && len(std) == 3 {
				for i, v := range std {
					if f, ok := v.(float64); ok {
						config.ImageConfig.Std[i] = float32(f)
					}
				}
			}
		}
	}

	return config, nil
}

func getIntFromMoondreamConfig(config map[string]any, key string, defaultVal int) int {
	if v, ok := config[key].(float64); ok {
		return int(v)
	}
	return defaultVal
}

// =============================================================================
// Moondream2 Model
// =============================================================================

// moondream2Model implements backends.Model for Moondream2 architecture.
// Moondream2 uses a 3-stage encoder:
//   - vision_encoder: pixel_values → image_features (SigLIP)
//   - projection: image_features → projected_features (MLP to Phi-2 embedding space)
//   - decoder: projected_features + input_ids → logits (Phi-2 decoder)
type moondream2Model struct {
	config *Moondream2ModelConfig

	// Moondream2 sessions
	visionEncoderSession backends.Session // vision_encoder.onnx (SigLIP)
	projectionSession    backends.Session // projection.onnx (MLP)
	decoderSession       backends.Session // decoder_model.onnx (Phi-2)

	backendType backends.BackendType
}

// LoadMoondream2Model loads a Moondream2 model using the given session factory.
func LoadMoondream2Model(modelPath string, factory backends.SessionFactory, opts ...backends.SessionOption) (backends.Model, error) {
	// Load configuration
	config, err := LoadMoondream2ModelConfig(modelPath)
	if err != nil {
		return nil, fmt.Errorf("loading model config: %w", err)
	}

	// Validate required files
	if config.VisionEncoderPath == "" {
		return nil, fmt.Errorf("vision_encoder.onnx not found in %s", modelPath)
	}
	if config.ProjectionPath == "" {
		return nil, fmt.Errorf("projection.onnx not found in %s", modelPath)
	}
	if config.DecoderPath == "" {
		return nil, fmt.Errorf("decoder ONNX file not found in %s", modelPath)
	}

	// Create sessions
	visionEncoderSession, err := factory.CreateSession(config.VisionEncoderPath, opts...)
	if err != nil {
		return nil, fmt.Errorf("creating vision encoder session: %w", err)
	}

	projectionSession, err := factory.CreateSession(config.ProjectionPath, opts...)
	if err != nil {
		visionEncoderSession.Close()
		return nil, fmt.Errorf("creating projection session: %w", err)
	}

	decoderSession, err := factory.CreateSession(config.DecoderPath, opts...)
	if err != nil {
		visionEncoderSession.Close()
		projectionSession.Close()
		return nil, fmt.Errorf("creating decoder session: %w", err)
	}

	return &moondream2Model{
		config:               config,
		visionEncoderSession: visionEncoderSession,
		projectionSession:    projectionSession,
		decoderSession:       decoderSession,
		backendType:          factory.Backend(),
	}, nil
}

// Forward runs the Moondream2 model.
// - If ImagePixels is set (and EncoderOutput is nil): runs vision encoder + projection
// - If EncoderOutput is set: runs decoder step
func (m *moondream2Model) Forward(ctx context.Context, inputs *backends.ModelInputs) (*backends.ModelOutput, error) {
	if inputs == nil {
		return nil, fmt.Errorf("nil inputs")
	}

	// If encoder output provided, run decoder
	if inputs.EncoderOutput != nil {
		return m.runDecoder(ctx, inputs)
	}

	// Otherwise run vision encoder + projection
	if inputs.ImagePixels == nil || len(inputs.ImagePixels) == 0 {
		return nil, fmt.Errorf("no image pixels or encoder output provided")
	}

	return m.runEncoder(ctx, inputs)
}

// runEncoder runs the vision encoder and projection layers.
// Pipeline: pixel_values → vision_encoder → hidden_states → projection → projected_features
func (m *moondream2Model) runEncoder(ctx context.Context, inputs *backends.ModelInputs) (*backends.ModelOutput, error) {
	batchSize := inputs.ImageBatch

	// Step 1: Run vision encoder on pixel_values
	pixelValues := backends.NamedTensor{
		Name:  "pixel_values",
		Shape: []int64{int64(batchSize), int64(inputs.ImageChannels), int64(inputs.ImageHeight), int64(inputs.ImageWidth)},
		Data:  inputs.ImagePixels,
	}

	visionOutputs, err := m.visionEncoderSession.Run([]backends.NamedTensor{pixelValues})
	if err != nil {
		return nil, fmt.Errorf("running vision encoder: %w", err)
	}

	if len(visionOutputs) == 0 {
		return nil, fmt.Errorf("no output from vision encoder")
	}

	// Get image features [batch, seq_len, hidden_size]
	imageFeatures := visionOutputs[0]
	imageFeaturesData, ok := imageFeatures.Data.([]float32)
	if !ok {
		return nil, fmt.Errorf("vision encoder output is not float32")
	}

	if len(imageFeatures.Shape) != 3 {
		return nil, fmt.Errorf("unexpected image features shape: %v (expected 3D)", imageFeatures.Shape)
	}

	// Step 2: Run projection layer
	projInputs := []backends.NamedTensor{{
		Name:  "hidden_states",
		Shape: imageFeatures.Shape,
		Data:  imageFeaturesData,
	}}

	projOutputs, err := m.projectionSession.Run(projInputs)
	if err != nil {
		return nil, fmt.Errorf("running projection: %w", err)
	}

	if len(projOutputs) == 0 {
		return nil, fmt.Errorf("no output from projection")
	}

	// Get projected features [batch, seq_len, hidden_size]
	projectedFeatures := projOutputs[0]
	projectedData, ok := projectedFeatures.Data.([]float32)
	if !ok {
		return nil, fmt.Errorf("projection output is not float32")
	}

	encoderOutput := &backends.EncoderOutput{
		HiddenStates: projectedData,
		Shape:        [3]int{int(projectedFeatures.Shape[0]), int(projectedFeatures.Shape[1]), int(projectedFeatures.Shape[2])},
	}

	return &backends.ModelOutput{
		EncoderOutput: encoderOutput,
	}, nil
}

// runDecoder performs one step of autoregressive decoding.
func (m *moondream2Model) runDecoder(ctx context.Context, inputs *backends.ModelInputs) (*backends.ModelOutput, error) {
	inputIDs := inputs.InputIDs
	encoderOutput := inputs.EncoderOutput

	batchSize := len(inputIDs)
	if batchSize == 0 {
		return nil, fmt.Errorf("empty input")
	}

	seqLen := len(inputIDs[0])

	// Get decoder input names
	inputInfo := m.decoderSession.InputInfo()
	inputNames := make(map[string]bool)
	for _, info := range inputInfo {
		inputNames[info.Name] = true
	}

	// Build decoder inputs
	var tensorInputs []backends.NamedTensor

	// Add input_ids (flatten to int64)
	flatInputIDs := make([]int64, batchSize*seqLen)
	for i := range batchSize {
		for j := range seqLen {
			flatInputIDs[i*seqLen+j] = int64(inputIDs[i][j])
		}
	}
	tensorInputs = append(tensorInputs, backends.NamedTensor{
		Name:  GetDecoderInputIDsName(inputNames),
		Shape: []int64{int64(batchSize), int64(seqLen)},
		Data:  flatInputIDs,
	})

	// Add encoder hidden states
	if inputNames["encoder_hidden_states"] || inputNames["inputs_embeds"] {
		name := "encoder_hidden_states"
		if inputNames["inputs_embeds"] && !inputNames["encoder_hidden_states"] {
			name = "inputs_embeds"
		}
		tensorInputs = append(tensorInputs, backends.NamedTensor{
			Name:  name,
			Shape: []int64{int64(encoderOutput.Shape[0]), int64(encoderOutput.Shape[1]), int64(encoderOutput.Shape[2])},
			Data:  encoderOutput.HiddenStates,
		})
	}

	// Add attention mask if needed
	if inputNames["attention_mask"] {
		mask := make([]int64, batchSize*seqLen)
		for i := range mask {
			mask[i] = 1
		}
		tensorInputs = append(tensorInputs, backends.NamedTensor{
			Name:  "attention_mask",
			Shape: []int64{int64(batchSize), int64(seqLen)},
			Data:  mask,
		})
	}

	// Add encoder attention mask if needed
	if inputNames["encoder_attention_mask"] {
		encSeqLen := encoderOutput.Shape[1]
		mask := make([]int64, batchSize*encSeqLen)
		for i := range mask {
			mask[i] = 1
		}
		tensorInputs = append(tensorInputs, backends.NamedTensor{
			Name:  "encoder_attention_mask",
			Shape: []int64{int64(batchSize), int64(encSeqLen)},
			Data:  mask,
		})
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

	// Reshape logits to [batch, vocab_size] (taking last position)
	vocabSize := int(logitsShape[len(logitsShape)-1])
	logits := make([][]float32, batchSize)
	for i := range batchSize {
		logits[i] = make([]float32, vocabSize)
		startIdx := i*seqLen*vocabSize + (seqLen-1)*vocabSize
		copy(logits[i], logitsData[startIdx:startIdx+vocabSize])
	}

	return &backends.ModelOutput{
		Logits: logits,
	}, nil
}

// DecoderConfig returns configuration needed for generation.
func (m *moondream2Model) DecoderConfig() *backends.DecoderConfig {
	return m.config.DecoderConfig
}

// ImageConfig returns configuration for image preprocessing.
func (m *moondream2Model) ImageConfig() *backends.ImageConfig {
	return m.config.ImageConfig
}

// Close releases resources associated with the model.
func (m *moondream2Model) Close() error {
	var errs []error

	if m.visionEncoderSession != nil {
		if err := m.visionEncoderSession.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing vision encoder: %w", err))
		}
		m.visionEncoderSession = nil
	}

	if m.projectionSession != nil {
		if err := m.projectionSession.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing projection: %w", err))
		}
		m.projectionSession = nil
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
func (m *moondream2Model) Name() string {
	return m.config.ModelPath
}

// Backend returns the backend type this model uses.
func (m *moondream2Model) Backend() backends.BackendType {
	return m.backendType
}

// =============================================================================
// Moondream2 Pipeline
// =============================================================================

// Moondream2Pipeline extends Vision2SeqPipeline with Moondream-specific behavior.
// Moondream is optimized for image understanding tasks with natural language prompts.
type Moondream2Pipeline struct {
	*Vision2SeqPipeline

	// Version detected from model path
	version string
}

// NewMoondream2Pipeline creates a new Moondream2Pipeline.
func NewMoondream2Pipeline(
	model backends.Model,
	tokenizer tokenizers.Tokenizer,
	config *Vision2SeqConfig,
) *Moondream2Pipeline {
	// Create base Vision2Seq pipeline
	base := NewVision2SeqPipeline(model, tokenizer, config)

	// Detect version from model path
	version := "2"
	if m, ok := model.(*moondream2Model); ok {
		pathLower := strings.ToLower(filepath.Base(m.config.ModelPath))
		if strings.Contains(pathLower, "moondream3") || strings.Contains(pathLower, "moondream-3") {
			version = "3"
		}
	}

	return &Moondream2Pipeline{
		Vision2SeqPipeline: base,
		version:            version,
	}
}

// Version returns the detected Moondream version (e.g., "2", "3").
func (p *Moondream2Pipeline) Version() string {
	return p.version
}

// Architecture returns the model architecture identifier.
func (p *Moondream2Pipeline) Architecture() string {
	return "moondream"
}

// =============================================================================
// Moondream2 Loader
// =============================================================================

// LoadMoondream2Pipeline loads a complete Moondream2 pipeline from a model directory.
func LoadMoondream2Pipeline(
	modelPath string,
	sessionManager *backends.SessionManager,
	modelBackends []string,
	opts ...Vision2SeqPipelineOption,
) (*Moondream2Pipeline, backends.BackendType, error) {
	// Get session factory from manager
	factory, backendType, err := sessionManager.GetSessionFactoryForModel(modelBackends)
	if err != nil {
		return nil, "", fmt.Errorf("getting session factory: %w", err)
	}

	// Load the tokenizer
	tokenizer, err := tokenizers.LoadTokenizer(modelPath)
	if err != nil {
		return nil, "", fmt.Errorf("loading tokenizer: %w", err)
	}

	// Load the Moondream2 model
	model, err := LoadMoondream2Model(modelPath, factory)
	if err != nil {
		return nil, "", fmt.Errorf("loading Moondream2 model: %w", err)
	}

	// Apply options
	config := &Vision2SeqConfig{}
	for _, opt := range opts {
		opt(config)
	}

	// Create the pipeline
	pipeline := NewMoondream2Pipeline(model, tokenizer, config)

	return pipeline, backendType, nil
}
