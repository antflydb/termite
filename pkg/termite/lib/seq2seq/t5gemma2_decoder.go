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
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
	ort "github.com/yalue/onnxruntime_go"
	"go.uber.org/zap"
)

// runDecoderWithEmbeddings runs the decoder directly with pre-computed encoder hidden states.
// This bypasses the encoder and uses the embeddings as encoder_hidden_states directly
// for the decoder's cross-attention mechanism.
//
// The implementation creates ONNX tensors from the input embeddings and uses hugot's
// generation functions to run the autoregressive decoding loop.
func (g *T5Gemma2Generator) runDecoderWithEmbeddings(
	ctx context.Context,
	input *DecoderInput,
	opts DecodeOptions,
) (*GeneratedOutput, error) {
	// Verify decoder ONNX files exist
	decoderInitPath := filepath.Join(g.modelPath, "decoder-init.onnx")
	decoderPath := filepath.Join(g.modelPath, "decoder.onnx")

	if _, err := os.Stat(decoderInitPath); err != nil {
		return nil, fmt.Errorf("decoder-init.onnx not found: %w", err)
	}
	if _, err := os.Stat(decoderPath); err != nil {
		return nil, fmt.Errorf("decoder.onnx not found: %w", err)
	}

	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Prepare embeddings dimensions
	batchSize := 1 // Currently support single-input decoding
	seqLen := len(input.EncoderHiddenStates)
	hiddenSize := input.HiddenSize

	g.logger.Debug("Running decoder with embeddings",
		zap.Int("batch_size", batchSize),
		zap.Int("seq_len", seqLen),
		zap.Int("hidden_size", hiddenSize),
		zap.Int("max_tokens", opts.MaxTokens))

	// Flatten embeddings to [batch_size * seq_len * hidden_size]
	flatEmbeddings := make([]float32, batchSize*seqLen*hiddenSize)
	for i, emb := range input.EncoderHiddenStates {
		copy(flatEmbeddings[i*hiddenSize:], emb)
	}

	// Create encoder_hidden_states tensor [batch_size, seq_len, hidden_size]
	encoderHiddenStatesTensor, err := ort.NewTensor(
		ort.NewShape(int64(batchSize), int64(seqLen), int64(hiddenSize)),
		flatEmbeddings,
	)
	if err != nil {
		return nil, fmt.Errorf("creating encoder_hidden_states tensor: %w", err)
	}

	// Track whether tensors have been transferred to batch ownership.
	// If any operation fails before batch takes ownership, defers will clean up.
	tensorsOwnedByBatch := false
	defer func() {
		if !tensorsOwnedByBatch {
			encoderHiddenStatesTensor.Destroy()
		}
	}()

	// Create attention mask
	var attentionMask []int64
	if input.AttentionMask != nil && len(input.AttentionMask) > 0 {
		attentionMask = input.AttentionMask
	} else {
		// Default: attend to all positions
		attentionMask = make([]int64, seqLen)
		for i := range attentionMask {
			attentionMask[i] = 1
		}
	}

	// Create attention mask tensor [batch_size, seq_len]
	flatAttentionMask := make([]int64, batchSize*seqLen)
	copy(flatAttentionMask, attentionMask)

	encoderAttentionMaskTensor, err := ort.NewTensor(
		ort.NewShape(int64(batchSize), int64(seqLen)),
		flatAttentionMask,
	)
	if err != nil {
		return nil, fmt.Errorf("creating encoder_attention_mask tensor: %w", err)
	}
	defer func() {
		if !tensorsOwnedByBatch {
			encoderAttentionMaskTensor.Destroy()
		}
	}()

	// Create a Seq2SeqBatch and set encoder outputs directly
	batch := pipelines.NewSeq2SeqBatch(batchSize)
	batch.MaxInputLength = seqLen
	batch.SetEncoderHiddenStates(encoderHiddenStatesTensor)
	batch.SetEncoderAttentionMask(encoderAttentionMaskTensor)

	// Set cleanup function for encoder tensors - batch now owns the tensors
	batch.SetDestroyEncoder(func() error {
		err1 := encoderHiddenStatesTensor.Destroy()
		err2 := encoderAttentionMaskTensor.Destroy()
		if err1 != nil {
			return err1
		}
		return err2
	})
	tensorsOwnedByBatch = true // Tensors are now managed by batch.Destroy()

	// Ensure cleanup on exit
	defer func() {
		if destroyErr := batch.Destroy(); destroyErr != nil {
			g.logger.Warn("Error destroying batch", zap.Error(destroyErr))
		}
	}()

	// Run generation using hugot's backend functions
	// The seq2seqPipeline implements Seq2SeqPipelineInterface
	var genErr error
	if opts.Temperature > 0 && opts.Temperature != 1.0 {
		// Use sampling with temperature
		genErr = backends.RunSeq2SeqGenerationSampling(batch, g.seq2seqPipeline)
	} else {
		// Use greedy decoding
		genErr = backends.RunSeq2SeqGenerationGreedy(batch, g.seq2seqPipeline)
	}

	if genErr != nil {
		return nil, fmt.Errorf("generation failed: %w", genErr)
	}

	// Get generated tokens and decode to text
	generatedTokens := batch.GetGeneratedTokens()

	g.logger.Debug("Generation completed",
		zap.Int("num_sequences", len(generatedTokens)))

	// Decode tokens to text using the pipeline's tokenizer
	texts := make([][]string, len(generatedTokens))
	for i, tokens := range generatedTokens {
		// Convert int64 tokens to uint32 for decode function
		uint32Tokens := make([]uint32, len(tokens))
		for j, t := range tokens {
			uint32Tokens[j] = uint32(t)
		}

		// Decode tokens, skipping special tokens
		tokenizer := g.seq2seqPipeline.GetTokenizer()
		text, decodeErr := backends.Decode(uint32Tokens, tokenizer, true)
		if decodeErr != nil {
			return nil, fmt.Errorf("decoding tokens: %w", decodeErr)
		}
		texts[i] = []string{text}
	}

	// Convert tokens to uint32 format for output
	tokenOutput := make([][][]uint32, len(generatedTokens))
	for i, tokens := range generatedTokens {
		uint32Tokens := make([]uint32, len(tokens))
		for j, t := range tokens {
			uint32Tokens[j] = uint32(t)
		}
		tokenOutput[i] = [][]uint32{uint32Tokens}
	}

	return &GeneratedOutput{
		Texts:  texts,
		Tokens: tokenOutput,
	}, nil
}
