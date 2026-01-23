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

	"github.com/gomlx/go-huggingface/tokenizers"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
)

// ============================================================================
// Configuration Types
// ============================================================================

// Seq2SeqModelConfig holds parsed configuration for a Seq2Seq model.
// This is loaded from config.json and generation_config.json.
type Seq2SeqModelConfig struct {
	// Path to the model directory
	ModelPath string

	// Paths to ONNX files (if present)
	EncoderPath string
	DecoderPath string

	// Decoder configuration
	DecoderConfig *backends.DecoderConfig

	// Architecture details for KV-cache
	NumLayers  int
	NumHeads   int
	HeadDim    int
	HiddenSize int
}

// rawSeq2SeqConfig represents the model's config.json structure for seq2seq backends.
type rawSeq2SeqConfig struct {
	// Model type
	ModelType string `json:"model_type"`

	// Vocab and token IDs
	VocabSize           int   `json:"vocab_size"`
	EOSTokenID          any   `json:"eos_token_id"` // Can be int or []int
	BOSTokenID          int32 `json:"bos_token_id"`
	PadTokenID          any   `json:"pad_token_id"` // Can be int or null
	DecoderStartTokenID int32 `json:"decoder_start_token_id"`

	// Architecture - different names across models
	DecoderLayers         int `json:"decoder_layers"`
	NumDecoderLayers      int `json:"num_decoder_layers"`
	NumLayers             int `json:"num_layers"`
	DecoderAttentionHeads int `json:"decoder_attention_heads"`
	NumHeads              int `json:"num_heads"`
	DModel                int `json:"d_model"`
	HiddenSize            int `json:"hidden_size"`

	// Sequence length
	MaxPositionEmbeddings int `json:"max_position_embeddings"`
	MaxLength             int `json:"max_length"`
}

// rawSeq2SeqGenerationConfig represents generation_config.json for seq2seq.
type rawSeq2SeqGenerationConfig struct {
	MaxLength           int     `json:"max_length"`
	MaxNewTokens        int     `json:"max_new_tokens"`
	EOSTokenID          any     `json:"eos_token_id"`
	BOSTokenID          int32   `json:"bos_token_id"`
	PadTokenID          any     `json:"pad_token_id"`
	DecoderStartTokenID int32   `json:"decoder_start_token_id"`
	DoSample            bool    `json:"do_sample"`
	Temperature         float32 `json:"temperature"`
	TopK                int     `json:"top_k"`
	TopP                float32 `json:"top_p"`
	NumBeams            int     `json:"num_beams"`
}

// ============================================================================
// Configuration Loading
// ============================================================================

// LoadSeq2SeqModelConfig loads and parses configuration for a Seq2Seq model.
// This is backend-agnostic and can be used by both ONNX and GoMLX backends.
func LoadSeq2SeqModelConfig(modelPath string) (*Seq2SeqModelConfig, error) {
	// Find encoder ONNX file
	encoderPath := FindONNXFile(modelPath, []string{
		"encoder_model.onnx",
		"encoder.onnx",
	})

	// Find decoder ONNX file
	decoderPath := FindONNXFile(modelPath, []string{
		"decoder_model_merged.onnx", // Preferred: merged decoder with KV-cache
		"decoder_with_past_model.onnx",
		"decoder_model.onnx",
		"decoder.onnx",
	})

	// Load model configuration from config.json
	rawConfig, err := loadRawSeq2SeqConfig(modelPath)
	if err != nil {
		return nil, fmt.Errorf("loading model config: %w", err)
	}

	// Load generation config if available
	genConfig := loadSeq2SeqGenerationConfig(modelPath)

	// Build decoder config
	decoderConfig := buildSeq2SeqDecoderConfig(rawConfig, genConfig)

	// Determine architecture
	numLayers := FirstNonZero(rawConfig.DecoderLayers, rawConfig.NumDecoderLayers, rawConfig.NumLayers, 6)
	numHeads := FirstNonZero(rawConfig.DecoderAttentionHeads, rawConfig.NumHeads, 8)
	hiddenSize := FirstNonZero(rawConfig.DModel, rawConfig.HiddenSize, 768)
	headDim := hiddenSize / numHeads

	return &Seq2SeqModelConfig{
		ModelPath:     modelPath,
		EncoderPath:   encoderPath,
		DecoderPath:   decoderPath,
		DecoderConfig: decoderConfig,
		NumLayers:     numLayers,
		NumHeads:      numHeads,
		HeadDim:       headDim,
		HiddenSize:    hiddenSize,
	}, nil
}

// IsSeq2SeqModel checks if a model path contains a Seq2Seq model.
// Returns true for encoder-decoder text models like T5, BART, mT5.
func IsSeq2SeqModel(path string) bool {
	// Check for encoder and decoder ONNX files
	encoderPath := FindONNXFile(path, []string{
		"encoder_model.onnx",
		"encoder.onnx",
	})
	decoderPath := FindONNXFile(path, []string{
		"decoder_model_merged.onnx",
		"decoder_with_past_model.onnx",
		"decoder_model.onnx",
		"decoder.onnx",
	})

	// Must have both encoder and decoder
	if encoderPath == "" || decoderPath == "" {
		return false
	}

	// Check config.json for seq2seq model types
	rawConfig, err := loadRawSeq2SeqConfig(path)
	if err != nil {
		return false
	}

	// Check for seq2seq model types
	seq2seqTypes := map[string]bool{
		"t5":              true,
		"mt5":             true,
		"bart":            true,
		"mbart":           true,
		"pegasus":         true,
		"longt5":          true,
		"led":             true,
		"bigbird_pegasus": true,
	}

	return seq2seqTypes[rawConfig.ModelType]
}

// loadRawSeq2SeqConfig loads the model configuration from config.json.
func loadRawSeq2SeqConfig(path string) (*rawSeq2SeqConfig, error) {
	configPath := filepath.Join(path, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("reading config.json: %w", err)
	}

	var config rawSeq2SeqConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("parsing config.json: %w", err)
	}

	return &config, nil
}

// loadSeq2SeqGenerationConfig loads generation_config.json if it exists.
func loadSeq2SeqGenerationConfig(path string) *rawSeq2SeqGenerationConfig {
	configPath := filepath.Join(path, "generation_config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil
	}

	var config rawSeq2SeqGenerationConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil
	}

	return &config
}

// buildSeq2SeqDecoderConfig creates a DecoderConfig from the raw configs.
func buildSeq2SeqDecoderConfig(cfg *rawSeq2SeqConfig, genCfg *rawSeq2SeqGenerationConfig) *backends.DecoderConfig {
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
	// Override from generation config if present
	if genCfg != nil {
		switch v := genCfg.EOSTokenID.(type) {
		case float64:
			eosTokenID = int32(v)
		case []interface{}:
			if len(v) > 0 {
				if f, ok := v[0].(float64); ok {
					eosTokenID = int32(f)
				}
			}
		}
	}

	// Handle pad_token_id which can be int or null
	var padTokenID int32
	switch v := cfg.PadTokenID.(type) {
	case float64:
		padTokenID = int32(v)
	case nil:
		padTokenID = eosTokenID // Common fallback
	}

	// Decoder start token
	decoderStartTokenID := cfg.DecoderStartTokenID
	if genCfg != nil && genCfg.DecoderStartTokenID != 0 {
		decoderStartTokenID = genCfg.DecoderStartTokenID
	}
	if decoderStartTokenID == 0 {
		decoderStartTokenID = padTokenID // T5 uses pad_token as decoder_start
	}

	// Max length
	maxLength := FirstNonZero(cfg.MaxLength, cfg.MaxPositionEmbeddings, 512)
	if genCfg != nil && genCfg.MaxLength > 0 {
		maxLength = genCfg.MaxLength
	}

	numHeads := FirstNonZero(cfg.DecoderAttentionHeads, cfg.NumHeads, 8)
	hiddenSize := FirstNonZero(cfg.DModel, cfg.HiddenSize, 768)

	return &backends.DecoderConfig{
		VocabSize:           cfg.VocabSize,
		MaxLength:           maxLength,
		EOSTokenID:          eosTokenID,
		BOSTokenID:          cfg.BOSTokenID,
		PadTokenID:          padTokenID,
		DecoderStartTokenID: decoderStartTokenID,
		NumLayers:           FirstNonZero(cfg.DecoderLayers, cfg.NumDecoderLayers, cfg.NumLayers, 6),
		NumHeads:            numHeads,
		HeadDim:             hiddenSize / numHeads,
	}
}

// ============================================================================
// Model Management
// ============================================================================

// Ensure seq2SeqModel implements backends.Model
var _ backends.Model = (*seq2SeqModel)(nil)

// seq2SeqModel implements backends.Model for encoder-decoder text-to-text tasks.
// It uses separate encoder and decoder sessions (T5, BART, mT5, etc.).
type seq2SeqModel struct {
	config *Seq2SeqModelConfig

	encoderSession backends.Session
	decoderSession backends.Session

	backendType backends.BackendType
}

// NewSeq2SeqModel creates a Model from encoder and decoder sessions.
func NewSeq2SeqModel(
	config *Seq2SeqModelConfig,
	encoderSession backends.Session,
	decoderSession backends.Session,
	backendType backends.BackendType,
) backends.Model {
	return &seq2SeqModel{
		config:         config,
		encoderSession: encoderSession,
		decoderSession: decoderSession,
		backendType:    backendType,
	}
}

// LoadSeq2SeqModel loads a seq2seq Model using the given session factory.
// It automatically discovers encoder and decoder ONNX files and creates sessions.
func LoadSeq2SeqModel(modelPath string, factory backends.SessionFactory, opts ...backends.SessionOption) (backends.Model, error) {
	// Load configuration
	config, err := LoadSeq2SeqModelConfig(modelPath)
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

	return &seq2SeqModel{
		config:         config,
		encoderSession: encoderSession,
		decoderSession: decoderSession,
		backendType:    factory.Backend(),
	}, nil
}

// Forward runs encoder or decoder based on inputs.
// - If EncoderOutput is nil: runs encoder on InputIDs/AttentionMask, returns EncoderOutput
// - If EncoderOutput is set: runs decoder step, returns Logits and PastKeyValues
func (m *seq2SeqModel) Forward(ctx context.Context, inputs *backends.ModelInputs) (*backends.ModelOutput, error) {
	if inputs == nil || len(inputs.InputIDs) == 0 {
		return nil, fmt.Errorf("empty input")
	}

	// If no encoder output, run encoder
	if inputs.EncoderOutput == nil {
		return m.runEncoder(ctx, inputs)
	}

	// Otherwise run decoder step
	return m.runDecoder(ctx, inputs)
}

// runEncoder encodes input tokens into encoder hidden states.
func (m *seq2SeqModel) runEncoder(ctx context.Context, inputs *backends.ModelInputs) (*backends.ModelOutput, error) {
	inputIDs := inputs.InputIDs
	attentionMask := inputs.AttentionMask

	batchSize := len(inputIDs)
	seqLen := len(inputIDs[0])

	// Flatten input IDs to int64
	flatInputIDs := make([]int64, batchSize*seqLen)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			flatInputIDs[i*seqLen+j] = int64(inputIDs[i][j])
		}
	}

	// Flatten attention mask to int64
	flatMask := make([]int64, batchSize*seqLen)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			if attentionMask != nil && i < len(attentionMask) && j < len(attentionMask[i]) {
				flatMask[i*seqLen+j] = int64(attentionMask[i][j])
			} else {
				flatMask[i*seqLen+j] = 1 // default to attending
			}
		}
	}

	// Build encoder inputs
	tensorInputs := []backends.NamedTensor{
		{
			Name:  "input_ids",
			Shape: []int64{int64(batchSize), int64(seqLen)},
			Data:  flatInputIDs,
		},
		{
			Name:  "attention_mask",
			Shape: []int64{int64(batchSize), int64(seqLen)},
			Data:  flatMask,
		},
	}

	// Run encoder
	outputs, err := m.encoderSession.Run(tensorInputs)
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

// runDecoder performs one step of autoregressive decoding with cross-attention.
func (m *seq2SeqModel) runDecoder(ctx context.Context, inputs *backends.ModelInputs) (*backends.ModelOutput, error) {
	inputIDs := inputs.InputIDs
	encoderOutput := inputs.EncoderOutput
	pastKeyValues := inputs.PastKeyValues

	batchSize := len(inputIDs)
	seqLen := len(inputIDs[0])

	// Flatten input IDs to int64
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
func (m *seq2SeqModel) buildDecoderInputs(inputIDs []int64, batchSize, seqLen int, encoderOutput *backends.EncoderOutput, pastKV *backends.KVCache) ([]backends.NamedTensor, error) {
	var inputs []backends.NamedTensor

	// Get decoder input names from session
	inputInfo := m.decoderSession.InputInfo()
	inputNames := make(map[string]bool)
	for _, info := range inputInfo {
		inputNames[info.Name] = true
	}

	// Add input_ids (or decoder_input_ids)
	inputIDsName := "input_ids"
	if inputNames["decoder_input_ids"] {
		inputIDsName = "decoder_input_ids"
	}
	inputs = append(inputs, backends.NamedTensor{
		Name:  inputIDsName,
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
		useCacheBool := []bool{pastKV != nil && pastKV.SeqLen > 0}
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

// createPastKVTensor creates a tensor for past key/value cache.
func (m *seq2SeqModel) createPastKVTensor(name string, pastKV *backends.KVCache, batchSize int) backends.NamedTensor {
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
func (m *seq2SeqModel) extractKVCache(outputs []backends.NamedTensor, batchSize int, pastKV *backends.KVCache) *backends.KVCache {
	// Look for present_key_values or present outputs
	for _, output := range outputs {
		if IsPresentKeyValueOutput(output.Name) {
			// Found KV-cache output - build the cache structure
			seqLen := 1
			if pastKV != nil {
				seqLen = pastKV.SeqLen + 1
			}
			return &backends.KVCache{
				SeqLen:    seqLen,
				NumLayers: m.config.NumLayers,
				NumHeads:  m.config.NumHeads,
				HeadDim:   m.config.HeadDim,
				BatchSize: batchSize,
			}
		}
	}

	// If no explicit KV-cache output, return updated cache based on input
	if pastKV != nil {
		return &backends.KVCache{
			SeqLen:    pastKV.SeqLen + 1,
			NumLayers: m.config.NumLayers,
			NumHeads:  m.config.NumHeads,
			HeadDim:   m.config.HeadDim,
			BatchSize: batchSize,
		}
	}

	return &backends.KVCache{
		SeqLen:    1,
		NumLayers: m.config.NumLayers,
		NumHeads:  m.config.NumHeads,
		HeadDim:   m.config.HeadDim,
		BatchSize: batchSize,
	}
}

// DecoderConfig returns configuration needed for generation.
func (m *seq2SeqModel) DecoderConfig() *backends.DecoderConfig {
	return m.config.DecoderConfig
}

// Close releases resources associated with the model.
func (m *seq2SeqModel) Close() error {
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
func (m *seq2SeqModel) Name() string {
	return m.config.ModelPath
}

// Backend returns the backend type this model uses.
func (m *seq2SeqModel) Backend() backends.BackendType {
	return m.backendType
}

// ============================================================================
// Pipeline
// ============================================================================

// Seq2SeqPipeline handles text-to-text tasks using encoder-decoder models.
// It supports models like T5, BART, mT5 for tasks like translation, summarization, and question generation.
type Seq2SeqPipeline struct {
	// EncoderDecoderPipeline provides shared generation functionality.
	*EncoderDecoderPipeline
}

// Seq2SeqConfig holds configuration for creating a Seq2SeqPipeline.
type Seq2SeqConfig struct {
	// GenerationConfig for text generation. If nil, uses defaults.
	GenerationConfig *backends.GenerationConfig
}

// Seq2SeqResult is an alias for EncoderDecoderResult for backwards compatibility.
type Seq2SeqResult = EncoderDecoderResult

// NewSeq2SeqPipeline creates a new Seq2SeqPipeline.
func NewSeq2SeqPipeline(
	model backends.Model,
	tokenizer tokenizers.Tokenizer,
	config *Seq2SeqConfig,
) *Seq2SeqPipeline {
	if config == nil {
		config = &Seq2SeqConfig{}
	}

	// Create base encoder-decoder pipeline
	base := NewEncoderDecoderPipeline(model, tokenizer, config.GenerationConfig)

	return &Seq2SeqPipeline{
		EncoderDecoderPipeline: base,
	}
}

// Generate generates text from the input text.
func (p *Seq2SeqPipeline) Generate(ctx context.Context, input string) (*Seq2SeqResult, error) {
	// Encode input text
	encoderOutput, err := p.encodeText(ctx, input)
	if err != nil {
		return nil, err
	}

	// Get start tokens (decoder start token)
	startTokens := p.GetStartTokens("")

	// Generate using shared base
	return p.GenerateFromEncoderOutput(ctx, encoderOutput, startTokens)
}

// GenerateBatch generates text for multiple inputs.
func (p *Seq2SeqPipeline) GenerateBatch(ctx context.Context, inputs []string) ([]*Seq2SeqResult, error) {
	results := make([]*Seq2SeqResult, len(inputs))

	// Process inputs one at a time for now
	// TODO: Batch processing with proper batched encoder/decoder
	for i, input := range inputs {
		result, err := p.Generate(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("processing input %d: %w", i, err)
		}
		results[i] = result
	}

	return results, nil
}

// GenerateWithStreaming generates text and streams tokens as they are generated.
// The callback is called for each generated token. Return false to stop generation.
func (p *Seq2SeqPipeline) GenerateWithStreaming(
	ctx context.Context,
	input string,
	callback func(token int32, text string) bool,
) (*Seq2SeqResult, error) {
	// Encode input text
	encoderOutput, err := p.encodeText(ctx, input)
	if err != nil {
		return nil, err
	}

	// Get start tokens
	startTokens := p.GetStartTokens("")

	// Generate with streaming using shared base
	return p.GenerateFromEncoderOutputStreaming(ctx, encoderOutput, startTokens, callback)
}

// encodeText tokenizes and encodes the input text.
func (p *Seq2SeqPipeline) encodeText(ctx context.Context, input string) (*backends.EncoderOutput, error) {
	// Tokenize input
	tokens := p.Tokenizer.Encode(input)
	inputIDs := [][]int32{IntToInt32(tokens)}

	// Create attention mask (all 1s)
	attentionMask := make([][]int32, 1)
	attentionMask[0] = make([]int32, len(tokens))
	for i := range attentionMask[0] {
		attentionMask[0][i] = 1
	}

	// Run encoder
	output, err := p.Model.Forward(ctx, &backends.ModelInputs{
		InputIDs:      inputIDs,
		AttentionMask: attentionMask,
	})
	if err != nil {
		return nil, fmt.Errorf("encoding text: %w", err)
	}

	return output.EncoderOutput, nil
}

// ============================================================================
// Loader
// ============================================================================

// Seq2SeqPipelineOption is a functional option for configuring Seq2SeqPipeline loading.
type Seq2SeqPipelineOption func(*Seq2SeqConfig)

// WithSeq2SeqGenerationConfig sets the generation config for the pipeline.
func WithSeq2SeqGenerationConfig(config *backends.GenerationConfig) Seq2SeqPipelineOption {
	return func(c *Seq2SeqConfig) {
		c.GenerationConfig = config
	}
}

// LoadSeq2SeqPipeline loads a complete Seq2Seq pipeline from a model directory.
// It automatically loads the model, tokenizer, and creates the pipeline.
// This signature matches the encoder-based pipeline loaders for consistency.
func LoadSeq2SeqPipeline(
	modelPath string,
	sessionManager *backends.SessionManager,
	modelBackends []string,
	opts ...Seq2SeqPipelineOption,
) (*Seq2SeqPipeline, backends.BackendType, error) {
	// Get session factory from manager
	factory, backendType, err := sessionManager.GetSessionFactoryForModel(modelBackends)
	if err != nil {
		return nil, "", fmt.Errorf("getting session factory: %w", err)
	}

	// Load the model
	model, err := LoadSeq2SeqModel(modelPath, factory)
	if err != nil {
		return nil, "", fmt.Errorf("loading model: %w", err)
	}

	// Load the tokenizer
	tokenizer, err := LoadTokenizer(modelPath)
	if err != nil {
		model.Close()
		return nil, "", fmt.Errorf("loading tokenizer: %w", err)
	}

	// Apply options
	config := &Seq2SeqConfig{}
	for _, opt := range opts {
		opt(config)
	}

	// Create the pipeline
	pipeline := NewSeq2SeqPipeline(model, tokenizer, config)

	return pipeline, backendType, nil
}
