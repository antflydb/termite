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

package seq2seq

import (
	"context"
)

// DecoderInput represents input for embedding-to-text generation.
// This allows bypassing the encoder and generating text directly from
// pre-computed embeddings (encoder hidden states).
type DecoderInput struct {
	// EncoderHiddenStates contains the pre-computed encoder outputs.
	// Shape: [sequence_length, hidden_size] for a single input.
	// These are passed directly to the decoder as encoder_hidden_states.
	EncoderHiddenStates [][]float32

	// AttentionMask is an optional mask for the encoder hidden states.
	// Shape: [sequence_length]. If nil, all positions are attended to.
	AttentionMask []int64

	// HiddenSize is the dimension of the hidden states (for validation).
	// This should match the model's hidden_size configuration.
	HiddenSize int
}

// DecodeOptions configures the decoding/generation behavior.
type DecodeOptions struct {
	// MaxTokens is the maximum number of tokens to generate.
	// Default: 256
	MaxTokens int

	// Temperature controls randomness in sampling. Higher values (e.g., 1.0)
	// produce more diverse output, lower values (e.g., 0.1) produce more
	// deterministic output. Default: 1.0
	Temperature float32

	// TopP is the nucleus sampling probability threshold. Tokens are sampled
	// from the smallest set of tokens whose cumulative probability exceeds TopP.
	// Default: 1.0 (disabled)
	TopP float32

	// TopK limits sampling to the top K most likely tokens.
	// Default: 0 (disabled)
	TopK int

	// RepetitionPenalty penalizes repeated tokens. Values > 1.0 discourage
	// repetition, values < 1.0 encourage it. Default: 1.0
	RepetitionPenalty float32
}

// DefaultDecodeOptions returns sensible default options for decoding.
func DefaultDecodeOptions() DecodeOptions {
	return DecodeOptions{
		MaxTokens:         256,
		Temperature:       1.0,
		TopP:              1.0,
		TopK:              0,
		RepetitionPenalty: 1.2,
	}
}

// Decoder is the interface for models that support generating text from embeddings.
// This enables embedding-to-text generation workflows, including:
// - Embedding inversion (reconstructing text from embeddings)
// - Custom embedding injection (using external/manipulated embeddings)
// - Cross-modal generation (vision embeddings to text)
type Decoder interface {
	// DecodeFromEmbeddings generates text from pre-computed encoder hidden states.
	// This bypasses the encoder and runs only the decoder, using the provided
	// embeddings as encoder_hidden_states for cross-attention.
	DecodeFromEmbeddings(ctx context.Context, input *DecoderInput, opts DecodeOptions) (*GeneratedOutput, error)

	// HiddenSize returns the expected embedding dimension for validation.
	// Embeddings passed to DecodeFromEmbeddings must have this dimension.
	HiddenSize() int

	// Close releases resources held by the decoder.
	Close() error
}
