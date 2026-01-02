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

package tokenizer

import (
	_ "embed"
	"fmt"
	"strings"

	"github.com/pkoukk/tiktoken-go"
	tiktoken_loader "github.com/pkoukk/tiktoken-go-loader"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/decoder"
	"github.com/sugarme/tokenizer/model"
	"github.com/sugarme/tokenizer/model/wordpiece"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
	"github.com/sugarme/tokenizer/processor"
	"github.com/sugarme/tokenizer/util"
)

//go:embed vocab/bert-base-uncased-vocab.txt
var bertVocab string

// Tokenizer provides token counting for text chunking.
type Tokenizer interface {
	// CountTokens returns the number of tokens in the text.
	// Returns a character-based estimate on error.
	CountTokens(text string) int
}

// BertWordPieceTokenizer uses BERT's WordPiece tokenization.
// Good for general-purpose text and multilingual content.
type BertWordPieceTokenizer struct {
	tokenizer *tokenizer.Tokenizer
}

// NewBertWordPieceTokenizer creates a BERT tokenizer from embedded vocab.
// This avoids the filesystem dependency of pretrained.BertBaseUncased()
// which fails in CI/Docker environments.
func NewBertWordPieceTokenizer() (*BertWordPieceTokenizer, error) {
	// Parse vocab file (one token per line, ID is line number)
	vocab := make(model.Vocab)
	for i, line := range strings.Split(bertVocab, "\n") {
		if line != "" {
			vocab[line] = i
		}
	}

	// Create WordPiece model with [UNK] as unknown token
	opts := util.NewParams(map[string]any{
		"unk_token": "[UNK]",
	})
	wp, err := wordpiece.New(vocab, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to create wordpiece model: %w", err)
	}

	tk := tokenizer.NewTokenizer(wp)

	// Configure BERT normalizer: clean text, lowercase, handle Chinese chars, strip accents
	tk.WithNormalizer(normalizer.NewBertNormalizer(true, true, true, true))
	tk.WithPreTokenizer(pretokenizer.NewBertPreTokenizer())

	// Configure post-processor with SEP and CLS tokens
	sepId, ok := tk.TokenToId("[SEP]")
	if !ok {
		return nil, fmt.Errorf("cannot find ID for [SEP] token")
	}
	clsId, ok := tk.TokenToId("[CLS]")
	if !ok {
		return nil, fmt.Errorf("cannot find ID for [CLS] token")
	}

	tk.WithPostProcessor(processor.NewBertProcessing(
		processor.PostToken{Id: sepId, Value: "[SEP]"},
		processor.PostToken{Id: clsId, Value: "[CLS]"},
	))

	// Add special tokens
	tk.AddSpecialTokens([]tokenizer.AddedToken{tokenizer.NewAddedToken("[MASK]", true)})
	tk.AddSpecialTokens([]tokenizer.AddedToken{tokenizer.NewAddedToken("[SEP]", true)})
	tk.AddSpecialTokens([]tokenizer.AddedToken{tokenizer.NewAddedToken("[CLS]", true)})

	tk.WithDecoder(decoder.DefaultWordpieceDecoder())

	return &BertWordPieceTokenizer{tokenizer: tk}, nil
}

// CountTokens returns the number of tokens in the text.
// Uses a recover wrapper to handle panics from the underlying tokenizer library
// (github.com/sugarme/tokenizer has a bounds check bug in BertNormalizer.TransformRange).
func (t *BertWordPieceTokenizer) CountTokens(text string) (count int) {
	if text == "" {
		return 0
	}

	// Recover from panics in the underlying tokenizer library
	defer func() {
		if r := recover(); r != nil {
			// Fallback: rough approximation (1 token ≈ 4 chars for English)
			count = len(text) / 4
		}
	}()

	enc, err := t.tokenizer.EncodeSingle(text)
	if err != nil {
		// Fallback: rough approximation (1 token ≈ 4 chars for English)
		return len(text) / 4
	}

	return len(enc.Ids)
}

// BPETokenizer uses OpenAI's tiktoken BPE tokenization.
// Good for GPT-style models and code.
type BPETokenizer struct {
	tiktoken *tiktoken.Tiktoken
}

func init() {
	// Set the offline loader for tiktoken to avoid network requests
	tiktoken.SetBpeLoader(tiktoken_loader.NewOfflineLoader())
}

// NewBPETokenizer creates a BPE tokenizer using tiktoken-go with embedded dictionaries.
// The encoding parameter specifies which BPE encoding to use:
// - "cl100k_base": GPT-4, GPT-3.5-turbo, text-embedding-ada-002 (recommended)
// - "o200k_base": GPT-4o models
// - "p50k_base": Codex models
// - "r50k_base": GPT-3 models (davinci, curie, etc.)
func NewBPETokenizer(encoding string) (*BPETokenizer, error) {
	if encoding == "" {
		encoding = "cl100k_base" // Default to GPT-4 encoding
	}

	tk, err := tiktoken.GetEncoding(encoding)
	if err != nil {
		return nil, fmt.Errorf("failed to get tiktoken encoding %q: %w", encoding, err)
	}

	return &BPETokenizer{tiktoken: tk}, nil
}

// CountTokens returns the number of tokens in the text.
func (t *BPETokenizer) CountTokens(text string) int {
	if text == "" {
		return 0
	}

	tokens := t.tiktoken.Encode(text, nil, nil)
	return len(tokens)
}
