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

package embeddings

import (
	"context"

	"github.com/antflydb/antfly-go/libaf/ai"
	"github.com/antflydb/antfly-go/libaf/embeddings"
)

// HiddenStatesOutput contains raw encoder hidden states (before pooling).
type HiddenStatesOutput struct {
	// HiddenStates are raw encoder outputs [batch_size][seq_len][hidden_size].
	// Each input produces a sequence of hidden state vectors, one per token.
	HiddenStates [][][]float32

	// AttentionMask indicates valid token positions [batch_size][seq_len].
	// 1 = valid token, 0 = padding token.
	AttentionMask [][]int64
}

// HiddenStatesEmbedder extends Embedder with raw hidden state extraction.
// This is useful for:
// - Passing hidden states to the decode API for text generation
// - Custom pooling strategies (e.g., first token, last token, weighted)
// - Token-level similarity comparisons
type HiddenStatesEmbedder interface {
	embeddings.Embedder

	// EmbedWithHiddenStates returns raw encoder hidden states before pooling.
	// Unlike Embed() which returns mean-pooled vectors, this returns the full
	// sequence of token-level hidden states.
	//
	// The returned hidden states can be passed to the decode API's embeddings
	// field for text generation from custom embeddings.
	EmbedWithHiddenStates(ctx context.Context, contents [][]ai.ContentPart) (*HiddenStatesOutput, error)
}
