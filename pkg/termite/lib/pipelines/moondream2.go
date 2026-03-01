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
	"fmt"

	"github.com/gomlx/gomlx/pkg/core/tensors/bucketing"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
)

// =============================================================================
// Decoder-Only VLM Model Detection
// =============================================================================

// IsDecoderOnlyVLMModel checks if a model path contains a decoder-only VLM
// (e.g., Moondream2). Detected by the presence of vision_encoder.onnx and
// embed_tokens.onnx, with no encoder_model.onnx (which would indicate an
// encoder-decoder VLM like Florence-2).
func IsDecoderOnlyVLMModel(path string) bool {
	visionEncoder := FindONNXFile(path, []string{"vision_encoder.onnx"})
	embedTokens := FindONNXFile(path, []string{"embed_tokens.onnx"})
	encoderModel := FindONNXFile(path, []string{"encoder_model.onnx"})

	return visionEncoder != "" && embedTokens != "" && encoderModel == ""
}

// =============================================================================
// Decoder-Only VLM Model
// =============================================================================

// decoderOnlyVLMModel implements backends.Model for decoder-only VLM
// architectures (e.g., Moondream2). Uses a vision encoder to extract image
// features, embed_tokens to embed text, then concatenates them as inputs_embeds
// for a decoder-only transformer (no cross-attention):
//   - vision_encoder: pixel_values → image_features
//   - embed_tokens: input_ids → text_embeddings
//   - decoder: inputs_embeds (concat of [image_features | text_embeddings]) + position_ids → logits (first step)
//   - decoder: inputs_embeds (single token via embed_tokens) + past_key_values + position_ids → logits (subsequent steps)
//
// Unlike encoder-decoder VLMs (Florence-2), there is no separate encoder_model.
// Image features are injected as prefix tokens through concatenation with text
// embeddings in inputs_embeds. The decoder always takes inputs_embeds (not
// input_ids) and always outputs KV cache tensors.
type decoderOnlyVLMModel struct {
	config *Vision2SeqModelConfig

	// Model sessions
	visionEncoderSession backends.Session // vision_encoder.onnx
	embedTokensSession   backends.Session // embed_tokens.onnx
	decoderSession       backends.Session // decoder_model_merged.onnx (ONNX Runtime fallback)

	// Split decoder sessions for GoMLX backends (XLA, Go, CoreML).
	// The merged decoder's ONNX If node cannot be evaluated at runtime by these
	// backends. Instead we use separate ONNX files (decoder_model.onnx and
	// decoder_with_past_model.onnx) that are purpose-built for each phase.
	decoderFirstStepSession backends.Session // decoder_model.onnx (first step, no KV cache)
	decoderWithPastSession  backends.Session // decoder_with_past_model.onnx (subsequent steps, with KV cache)
	useSplitDecoders        bool

	// kvBucketStrategy buckets past_key_values sequence lengths to reduce
	// the number of unique shapes seen by JIT backends (XLA, CoreML).
	kvBucketStrategy bucketing.Strategy

	backendType backends.BackendType
}

// LoadDecoderOnlyVLMModel loads a decoder-only VLM model using the given session factory.
func LoadDecoderOnlyVLMModel(modelPath string, factory backends.SessionFactory, opts ...backends.SessionOption) (backends.Model, error) {
	// Load configuration
	config, err := LoadVision2SeqModelConfig(modelPath)
	if err != nil {
		return nil, fmt.Errorf("loading model config: %w", err)
	}

	// Find required ONNX files
	visionEncoderPath := FindONNXFile(modelPath, []string{"vision_encoder.onnx"})
	embedTokensPath := FindONNXFile(modelPath, []string{"embed_tokens.onnx"})
	decoderPath := FindONNXFile(modelPath, []string{
		"decoder_model_merged.onnx",
		"decoder_with_past.onnx",
		"decoder.onnx",
		"decoder_model.onnx",
	})

	if visionEncoderPath == "" {
		return nil, fmt.Errorf("vision_encoder.onnx not found in %s", modelPath)
	}
	if embedTokensPath == "" {
		return nil, fmt.Errorf("embed_tokens.onnx not found in %s", modelPath)
	}
	if decoderPath == "" {
		return nil, fmt.Errorf("decoder ONNX file not found in %s", modelPath)
	}

	config.DecoderPath = decoderPath

	// Create sessions with cascading cleanup on error
	visionEncoderSession, err := factory.CreateSession(visionEncoderPath, opts...)
	if err != nil {
		return nil, fmt.Errorf("creating vision encoder session: %w", err)
	}

	embedTokensSession, err := factory.CreateSession(embedTokensPath, opts...)
	if err != nil {
		visionEncoderSession.Close()
		return nil, fmt.Errorf("creating embed_tokens session: %w", err)
	}

	model := &decoderOnlyVLMModel{
		config:               config,
		visionEncoderSession: visionEncoderSession,
		embedTokensSession:   embedTokensSession,
		backendType:          factory.Backend(),
	}

	closeOnError := func() {
		visionEncoderSession.Close()
		embedTokensSession.Close()
	}

	// Create the main decoder session
	decoderSession, err := factory.CreateSession(decoderPath, opts...)
	if err != nil {
		closeOnError()
		return nil, fmt.Errorf("creating decoder session: %w", err)
	}
	model.decoderSession = decoderSession

	// Try to load split decoders for GoMLX backends (XLA, Go, CoreML).
	// These backends cannot evaluate ONNX If nodes at runtime, so the merged
	// decoder disables KV caching. The separate decoder_model.onnx and
	// decoder_with_past_model.onnx files are purpose-built for each phase.
	isGoMLXBackend := false
	switch factory.Backend() {
	case backends.BackendGo, backends.BackendXLA, backends.BackendCoreML:
		isGoMLXBackend = true
	}
	if isGoMLXBackend && config.DecoderFirstStepPath != "" && config.DecoderWithPastPath != "" {
		firstStepSession, err := factory.CreateSession(config.DecoderFirstStepPath, opts...)
		if err == nil {
			withPastOpts := append(opts, backends.WithDynamicAxes([]backends.DynamicAxisOverride{
				{InputName: "inputs_embeds", Axis: 1, ParamName: "decoder_sequence_length"},
				{InputName: "input_ids", Axis: 1, ParamName: "decoder_sequence_length"},
			}))
			withPastSession, err := factory.CreateSession(config.DecoderWithPastPath, withPastOpts...)
			if err == nil {
				model.decoderFirstStepSession = firstStepSession
				model.decoderWithPastSession = withPastSession
				model.useSplitDecoders = true

				switch factory.Backend() {
				case backends.BackendXLA, backends.BackendCoreML:
					model.kvBucketStrategy = bucketing.Pow2()
				}
			} else {
				firstStepSession.Close()
			}
		}
	}

	return model, nil
}

// Forward runs the decoder-only VLM model.
// - If ImagePixels is set (and EncoderOutput is nil): runs vision encoder
// - If EncoderOutput is set: runs decoder step
func (m *decoderOnlyVLMModel) Forward(ctx context.Context, inputs *backends.ModelInputs) (*backends.ModelOutput, error) {
	if inputs == nil {
		return nil, fmt.Errorf("nil inputs")
	}

	if inputs.EncoderOutput != nil {
		return m.runDecoder(ctx, inputs)
	}

	if inputs.ImagePixels == nil || len(inputs.ImagePixels) == 0 {
		return nil, fmt.Errorf("no image pixels or encoder output provided")
	}

	return m.runEncoder(ctx, inputs)
}

// runEncoder runs the vision encoder on pixel values.
// Unlike encoder-decoder VLMs, there is no separate encoder_model stage —
// the vision encoder output is concatenated with text embeddings in the
// decoder's inputs_embeds.
func (m *decoderOnlyVLMModel) runEncoder(ctx context.Context, inputs *backends.ModelInputs) (*backends.ModelOutput, error) {
	batchSize := inputs.ImageBatch

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

	imageFeatures := visionOutputs[0]
	imageFeaturesData, ok := imageFeatures.Data.([]float32)
	if !ok {
		return nil, fmt.Errorf("vision encoder output is not float32")
	}

	if len(imageFeatures.Shape) != 3 {
		return nil, fmt.Errorf("unexpected image features shape: %v (expected 3D)", imageFeatures.Shape)
	}

	encoderOutput := &backends.EncoderOutput{
		HiddenStates: imageFeaturesData,
		Shape:        [3]int{int(imageFeatures.Shape[0]), int(imageFeatures.Shape[1]), int(imageFeatures.Shape[2])},
	}

	return &backends.ModelOutput{
		EncoderOutput: encoderOutput,
	}, nil
}

// runDecoder performs one step of autoregressive decoding for a decoder-only VLM.
//
// First step (no KV cache): embed text tokens via embed_tokens, concatenate
// [image_features | text_embeds] into inputs_embeds, and run the decoder.
//
// Subsequent steps (with KV cache): embed the new token via embed_tokens,
// pass its embedding as inputs_embeds along with past_key_values and position_ids.
//
// The Moondream2 decoder always takes inputs_embeds (not input_ids) and always
// outputs present.* KV cache tensors. Unlike some merged decoders, there is no
// use_cache_branch input — the model switches behavior based on whether
// past_key_values has sequence length 0 or not.
func (m *decoderOnlyVLMModel) runDecoder(ctx context.Context, inputs *backends.ModelInputs) (*backends.ModelOutput, error) {
	inputIDs := inputs.InputIDs
	encoderOutput := inputs.EncoderOutput
	pastKeyValues := inputs.PastKeyValues

	batchSize := len(inputIDs)
	if batchSize == 0 {
		return nil, fmt.Errorf("empty input")
	}

	seqLen := len(inputIDs[0])
	isFirstStep := pastKeyValues == nil || pastKeyValues.SeqLen == 0

	// Choose decoder session
	var decoderSession backends.Session
	if m.useSplitDecoders {
		if isFirstStep {
			decoderSession = m.decoderFirstStepSession
		} else {
			decoderSession = m.decoderWithPastSession
		}
	} else {
		decoderSession = m.decoderSession
	}

	var tensorInputs []backends.NamedTensor

	if isFirstStep {
		// First step: embed text and concatenate with image features
		embeds, err := m.buildFirstStepInputs(decoderSession, inputIDs, batchSize, seqLen, encoderOutput)
		if err != nil {
			return nil, err
		}
		tensorInputs = embeds
	} else {
		// Subsequent steps: embed new token, pass with KV cache
		embeds, err := m.buildSubsequentStepInputs(decoderSession, inputIDs, batchSize, seqLen, pastKeyValues)
		if err != nil {
			return nil, err
		}
		tensorInputs = embeds
	}

	// Pad KV cache tensors for bucketing
	var realPastSeqLen int
	if m.kvBucketStrategy != nil && !isFirstStep {
		realPastSeqLen = kvCacheSeqLen(pastKeyValues)
		bucketedSeqLen := m.kvBucketStrategy.Bucket(realPastSeqLen)
		if bucketedSeqLen > realPastSeqLen {
			tensorInputs = padDecoderKVInputs(tensorInputs, realPastSeqLen, bucketedSeqLen)
		}
	}

	// Run decoder
	outputs, err := decoderSession.Run(tensorInputs)
	if err != nil {
		return nil, fmt.Errorf("running decoder: %w", err)
	}

	if len(outputs) == 0 {
		return nil, fmt.Errorf("no decoder output")
	}

	// Trim padded positions from present outputs
	if m.kvBucketStrategy != nil && !isFirstStep && realPastSeqLen > 0 {
		bucketedSeqLen := m.kvBucketStrategy.Bucket(realPastSeqLen)
		if bucketedSeqLen > realPastSeqLen {
			outputs = trimPresentKV(outputs, realPastSeqLen, bucketedSeqLen)
		}
	}

	// Extract logits (first output)
	logitsOutput := outputs[0]
	logitsData, ok := logitsOutput.Data.([]float32)
	if !ok {
		return nil, fmt.Errorf("logits tensor is not float32")
	}

	logitsShape := logitsOutput.Shape

	// Reshape logits to [batch, vocab_size] (taking last position)
	outputSeqLen := int(logitsShape[1])
	vocabSize := int(logitsShape[len(logitsShape)-1])
	logits := make([][]float32, batchSize)
	for i := range batchSize {
		logits[i] = make([]float32, vocabSize)
		startIdx := i*outputSeqLen*vocabSize + (outputSeqLen-1)*vocabSize
		copy(logits[i], logitsData[startIdx:startIdx+vocabSize])
	}

	// Always extract KV cache from decoder outputs. The merged decoder outputs
	// present.* tensors on every step. Returning them triggers the generation
	// loop to use the KV cache path (trimming InputIDs to just the last token).
	newKVCache := m.extractKVCache(outputs, batchSize, pastKeyValues)

	return &backends.ModelOutput{
		Logits:        logits,
		PastKeyValues: newKVCache,
	}, nil
}

// buildFirstStepInputs creates decoder inputs for the first step.
// Embeds text tokens via embed_tokens, then concatenates [image_features | text_embeds]
// into inputs_embeds for the decoder.
func (m *decoderOnlyVLMModel) buildFirstStepInputs(
	session backends.Session,
	inputIDs [][]int32,
	batchSize, seqLen int,
	encoderOutput *backends.EncoderOutput,
) ([]backends.NamedTensor, error) {
	// Flatten input IDs for embed_tokens
	flatInputIDs := make([]int64, batchSize*seqLen)
	for i := range batchSize {
		for j := range seqLen {
			flatInputIDs[i*seqLen+j] = int64(inputIDs[i][j])
		}
	}

	// Run embed_tokens on text input_ids
	embedInput := backends.NamedTensor{
		Name:  "input_ids",
		Shape: []int64{int64(batchSize), int64(seqLen)},
		Data:  flatInputIDs,
	}

	embedOutputs, err := m.embedTokensSession.Run([]backends.NamedTensor{embedInput})
	if err != nil {
		return nil, fmt.Errorf("running embed_tokens: %w", err)
	}
	if len(embedOutputs) == 0 {
		return nil, fmt.Errorf("no output from embed_tokens")
	}

	textEmbedsData, ok := embedOutputs[0].Data.([]float32)
	if !ok {
		return nil, fmt.Errorf("embed_tokens output is not float32")
	}

	hiddenSize := int(embedOutputs[0].Shape[2])
	imageSeqLen := encoderOutput.Shape[1]

	// Concatenate [image_features | text_embeds] → inputs_embeds
	totalSeqLen := imageSeqLen + seqLen
	inputsEmbeds := make([]float32, batchSize*totalSeqLen*hiddenSize)

	for b := range batchSize {
		// Copy image features
		for s := range imageSeqLen {
			srcIdx := b*imageSeqLen*hiddenSize + s*hiddenSize
			dstIdx := b*totalSeqLen*hiddenSize + s*hiddenSize
			copy(inputsEmbeds[dstIdx:dstIdx+hiddenSize], encoderOutput.HiddenStates[srcIdx:srcIdx+hiddenSize])
		}
		// Copy text embeds
		for s := range seqLen {
			srcIdx := b*seqLen*hiddenSize + s*hiddenSize
			dstIdx := b*totalSeqLen*hiddenSize + (imageSeqLen+s)*hiddenSize
			copy(inputsEmbeds[dstIdx:dstIdx+hiddenSize], textEmbedsData[srcIdx:srcIdx+hiddenSize])
		}
	}

	var inputs []backends.NamedTensor

	// Get session input names
	inputInfo := session.InputInfo()
	inputNames := make(map[string]bool)
	for _, info := range inputInfo {
		inputNames[info.Name] = true
	}

	// Add inputs_embeds
	inputs = append(inputs, backends.NamedTensor{
		Name:  "inputs_embeds",
		Shape: []int64{int64(batchSize), int64(totalSeqLen), int64(hiddenSize)},
		Data:  inputsEmbeds,
	})

	// Add attention mask if needed
	if inputNames["attention_mask"] {
		mask := make([]int64, batchSize*totalSeqLen)
		for i := range mask {
			mask[i] = 1
		}
		inputs = append(inputs, backends.NamedTensor{
			Name:  "attention_mask",
			Shape: []int64{int64(batchSize), int64(totalSeqLen)},
			Data:  mask,
		})
	}

	// Add position_ids if needed: [0, 1, 2, ..., totalSeqLen-1]
	if inputNames["position_ids"] {
		posIDs := make([]int64, batchSize*totalSeqLen)
		for b := range batchSize {
			for s := range totalSeqLen {
				posIDs[b*totalSeqLen+s] = int64(s)
			}
		}
		inputs = append(inputs, backends.NamedTensor{
			Name:  "position_ids",
			Shape: []int64{int64(batchSize), int64(totalSeqLen)},
			Data:  posIDs,
		})
	}

	// Add use_cache_branch if needed (first step → false)
	if inputNames["use_cache_branch"] {
		inputs = append(inputs, createUseCacheBranchTensor(inputInfo, false))
	}

	// Add zero-initialized past_key_values (decoder-only: no encoder KV)
	for _, info := range inputInfo {
		if IsPastKeyValueInput(info.Name) {
			inputs = append(inputs, m.createZeroPastKVTensor(info.Name, batchSize))
		}
	}

	return inputs, nil
}

// buildSubsequentStepInputs creates decoder inputs for subsequent steps.
// The Moondream2 decoder always takes inputs_embeds (not input_ids), so we
// run embed_tokens on the new token to get its embedding, then pass it
// along with the KV cache and position_ids.
func (m *decoderOnlyVLMModel) buildSubsequentStepInputs(
	session backends.Session,
	inputIDs [][]int32,
	batchSize, seqLen int,
	pastKV *backends.KVCache,
) ([]backends.NamedTensor, error) {
	// Flatten input IDs for embed_tokens
	flatInputIDs := make([]int64, batchSize*seqLen)
	for i := range batchSize {
		for j := range seqLen {
			flatInputIDs[i*seqLen+j] = int64(inputIDs[i][j])
		}
	}

	// Run embed_tokens to convert token(s) to embeddings
	embedInput := backends.NamedTensor{
		Name:  "input_ids",
		Shape: []int64{int64(batchSize), int64(seqLen)},
		Data:  flatInputIDs,
	}
	embedOutputs, err := m.embedTokensSession.Run([]backends.NamedTensor{embedInput})
	if err != nil {
		return nil, fmt.Errorf("running embed_tokens: %w", err)
	}
	if len(embedOutputs) == 0 {
		return nil, fmt.Errorf("no output from embed_tokens")
	}

	embedsData, ok := embedOutputs[0].Data.([]float32)
	if !ok {
		return nil, fmt.Errorf("embed_tokens output is not float32")
	}
	hiddenSize := int(embedOutputs[0].Shape[2])

	var inputs []backends.NamedTensor

	inputInfo := session.InputInfo()
	inputNames := make(map[string]bool)
	for _, info := range inputInfo {
		inputNames[info.Name] = true
	}

	// Add inputs_embeds (the decoder always takes inputs_embeds, not input_ids)
	inputs = append(inputs, backends.NamedTensor{
		Name:  "inputs_embeds",
		Shape: []int64{int64(batchSize), int64(seqLen), int64(hiddenSize)},
		Data:  embedsData,
	})

	// Get actual past sequence length from KV cache tensor shapes
	pastSeqLen := kvCacheSeqLen(pastKV)

	// Add attention mask covering all past + current tokens
	if inputNames["attention_mask"] {
		totalLen := pastSeqLen + seqLen
		mask := make([]int64, batchSize*totalLen)
		for i := range mask {
			mask[i] = 1
		}
		inputs = append(inputs, backends.NamedTensor{
			Name:  "attention_mask",
			Shape: []int64{int64(batchSize), int64(totalLen)},
			Data:  mask,
		})
	}

	// Add position_ids: [pastSeqLen, pastSeqLen+1, ..., pastSeqLen+seqLen-1]
	if inputNames["position_ids"] {
		posIDs := make([]int64, batchSize*seqLen)
		for b := range batchSize {
			for s := range seqLen {
				posIDs[b*seqLen+s] = int64(pastSeqLen + s)
			}
		}
		inputs = append(inputs, backends.NamedTensor{
			Name:  "position_ids",
			Shape: []int64{int64(batchSize), int64(seqLen)},
			Data:  posIDs,
		})
	}

	// Add use_cache_branch if needed (subsequent step → true)
	if inputNames["use_cache_branch"] {
		inputs = append(inputs, createUseCacheBranchTensor(inputInfo, true))
	}

	// Add past_key_values from cache
	for _, info := range inputInfo {
		if IsPastKeyValueInput(info.Name) {
			tensor := m.createPastKVTensor(info.Name, pastKV, batchSize)
			inputs = append(inputs, tensor)
		}
	}

	return inputs, nil
}

// createUseCacheBranchTensor creates the use_cache_branch tensor.
func createUseCacheBranchTensor(inputInfo []backends.TensorInfo, useCache bool) backends.NamedTensor {
	var dataType backends.DataType = backends.DataTypeBool
	for _, info := range inputInfo {
		if info.Name == "use_cache_branch" {
			dataType = info.DataType
			break
		}
	}

	if dataType == backends.DataTypeFloat32 {
		val := []float32{0}
		if useCache {
			val[0] = 1
		}
		return backends.NamedTensor{
			Name:  "use_cache_branch",
			Shape: []int64{1},
			Data:  val,
		}
	}
	return backends.NamedTensor{
		Name:  "use_cache_branch",
		Shape: []int64{1},
		Data:  []bool{useCache},
	}
}

// createZeroPastKVTensor creates zero-initialized past KV tensors for the first step.
// Decoder-only models have no encoder KV tensors — all are self-attention only.
func (m *decoderOnlyVLMModel) createZeroPastKVTensor(name string, batchSize int) backends.NamedTensor {
	numHeads := m.config.NumHeads
	headDim := m.config.HeadDim
	if numHeads == 0 {
		numHeads = 8
	}
	if headDim == 0 {
		headDim = 64
	}

	return backends.NamedTensor{
		Name:  name,
		Shape: []int64{int64(batchSize), int64(numHeads), 0, int64(headDim)},
		Data:  []float32{},
	}
}

// createPastKVTensor retrieves a cached KV tensor from the previous step.
func (m *decoderOnlyVLMModel) createPastKVTensor(name string, pastKV *backends.KVCache, batchSize int) backends.NamedTensor {
	if pastKV != nil && pastKV.SeqLen > 0 && pastKV.Tensors != nil {
		outputName := mapPastToPresent(name)
		if tensor, ok := pastKV.Tensors[outputName]; ok {
			return backends.NamedTensor{
				Name:  name,
				Shape: tensor.Shape,
				Data:  tensor.Data,
			}
		}
	}

	// Fallback to zero tensor
	return m.createZeroPastKVTensor(name, batchSize)
}

// extractKVCache extracts the KV cache from decoder outputs.
// Collects all present.* output tensors and stores them for the next step.
func (m *decoderOnlyVLMModel) extractKVCache(outputs []backends.NamedTensor, batchSize int, pastKV *backends.KVCache) *backends.KVCache {
	tensors := make(map[string]backends.NamedTensor)
	hasKVOutputs := false

	for _, output := range outputs {
		if IsPresentKeyValueOutput(output.Name) {
			hasKVOutputs = true
			data, ok := output.Data.([]float32)
			if ok {
				dataCopy := make([]float32, len(data))
				copy(dataCopy, data)
				shapeCopy := make([]int64, len(output.Shape))
				copy(shapeCopy, output.Shape)
				tensors[output.Name] = backends.NamedTensor{
					Name:  output.Name,
					Shape: shapeCopy,
					Data:  dataCopy,
				}
			}
		}
	}

	if hasKVOutputs {
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
			Tensors:   tensors,
		}
	}

	return nil
}

// kvCacheSeqLen returns the actual sequence length from KV cache tensor shapes.
// This reflects the total past sequence (including image tokens from the first step)
// rather than the step counter in KVCache.SeqLen.
func kvCacheSeqLen(pastKV *backends.KVCache) int {
	if pastKV == nil || pastKV.Tensors == nil {
		return 0
	}
	for _, tensor := range pastKV.Tensors {
		if len(tensor.Shape) == 4 {
			return int(tensor.Shape[2])
		}
	}
	return 0
}

// trimPresentKV removes zero-padding from all present.* KV output tensors.
// This is the decoder-only equivalent of trimPresentDecoderKV — since decoder-only
// models have no encoder KV tensors, all present outputs are trimmed.
//
// After a padded forward pass the present tensor has shape
// [batch, heads, bucketedSeqLen+1, headDim]. We keep positions [0:realSeqLen]
// and [bucketedSeqLen:bucketedSeqLen+1], producing [batch, heads, realSeqLen+1, headDim].
func trimPresentKV(outputs []backends.NamedTensor, realSeqLen, bucketedSeqLen int) []backends.NamedTensor {
	result := make([]backends.NamedTensor, len(outputs))
	for i, t := range outputs {
		if !IsPresentKeyValueOutput(t.Name) {
			result[i] = t
			continue
		}

		data, ok := t.Data.([]float32)
		if !ok || len(t.Shape) != 4 {
			result[i] = t
			continue
		}

		batch := int(t.Shape[0])
		heads := int(t.Shape[1])
		srcSeqLen := int(t.Shape[2]) // bucketedSeqLen + 1
		headDim := int(t.Shape[3])
		trimmedSeqLen := realSeqLen + 1

		trimmedSize := batch * heads * trimmedSeqLen * headDim
		trimmed := make([]float32, trimmedSize)

		for b := range batch {
			for h := range heads {
				srcBase := (b*heads + h) * srcSeqLen * headDim
				dstBase := (b*heads + h) * trimmedSeqLen * headDim

				// Copy the real past positions [0:realSeqLen].
				copy(trimmed[dstBase:dstBase+realSeqLen*headDim],
					data[srcBase:srcBase+realSeqLen*headDim])

				// Copy the new token position [bucketedSeqLen].
				newTokSrc := srcBase + bucketedSeqLen*headDim
				newTokDst := dstBase + realSeqLen*headDim
				copy(trimmed[newTokDst:newTokDst+headDim],
					data[newTokSrc:newTokSrc+headDim])
			}
		}

		result[i] = backends.NamedTensor{
			Name:  t.Name,
			Shape: []int64{t.Shape[0], t.Shape[1], int64(trimmedSeqLen), t.Shape[3]},
			Data:  trimmed,
		}
	}
	return result
}

// DecoderConfig returns configuration needed for generation.
func (m *decoderOnlyVLMModel) DecoderConfig() *backends.DecoderConfig {
	return m.config.DecoderConfig
}

// ImageConfig returns configuration for image preprocessing.
func (m *decoderOnlyVLMModel) ImageConfig() *backends.ImageConfig {
	return m.config.ImageConfig
}

// Close releases resources associated with the model.
func (m *decoderOnlyVLMModel) Close() error {
	var errs []error

	if m.visionEncoderSession != nil {
		if err := m.visionEncoderSession.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing vision encoder: %w", err))
		}
		m.visionEncoderSession = nil
	}

	if m.embedTokensSession != nil {
		if err := m.embedTokensSession.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing embed_tokens: %w", err))
		}
		m.embedTokensSession = nil
	}

	if m.decoderSession != nil {
		if err := m.decoderSession.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing decoder: %w", err))
		}
		m.decoderSession = nil
	}

	if m.decoderFirstStepSession != nil {
		if err := m.decoderFirstStepSession.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing first-step decoder: %w", err))
		}
		m.decoderFirstStepSession = nil
	}

	if m.decoderWithPastSession != nil {
		if err := m.decoderWithPastSession.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing with-past decoder: %w", err))
		}
		m.decoderWithPastSession = nil
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing model: %v", errs)
	}
	return nil
}

// Name returns the model name for logging and debugging.
func (m *decoderOnlyVLMModel) Name() string {
	return m.config.ModelPath
}

// Backend returns the backend type this model uses.
func (m *decoderOnlyVLMModel) Backend() backends.BackendType {
	return m.backendType
}
