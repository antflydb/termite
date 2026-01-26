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

package pipelines

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/daulet/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	goTokenizers "github.com/gomlx/go-huggingface/tokenizers"
)

// rustTokenizer wraps the Rust HuggingFace tokenizers library.
// This is significantly faster than the pure Go implementation.
type rustTokenizer struct {
	tk     *tokenizers.Tokenizer
	config *api.Config
}

// Ensure rustTokenizer implements tokenizers.Tokenizer
var _ goTokenizers.Tokenizer = (*rustTokenizer)(nil)

// loadRustTokenizer attempts to load a tokenizer using the Rust library.
// Returns nil if the tokenizer can't be loaded (falls back to Go implementation).
func loadRustTokenizer(modelPath string, config *api.Config) (goTokenizers.Tokenizer, error) {
	tokenizerJSONPath := filepath.Join(modelPath, "tokenizer.json")

	data, err := os.ReadFile(tokenizerJSONPath)
	if err != nil {
		return nil, fmt.Errorf("reading tokenizer.json: %w", err)
	}

	tk, err := tokenizers.FromBytes(data)
	if err != nil {
		return nil, fmt.Errorf("loading Rust tokenizer: %w", err)
	}

	return &rustTokenizer{
		tk:     tk,
		config: config,
	}, nil
}

// Encode returns the text encoded into a sequence of token IDs.
func (t *rustTokenizer) Encode(text string) []int {
	output := t.tk.EncodeWithOptions(text, true,
		tokenizers.WithReturnTokens(),
	)

	result := make([]int, len(output.IDs))
	for i, id := range output.IDs {
		result[i] = int(id)
	}
	return result
}

// Decode returns the text from a sequence of token IDs.
func (t *rustTokenizer) Decode(ids []int) string {
	// Convert []int to []uint32
	uids := make([]uint32, len(ids))
	for i, id := range ids {
		uids[i] = uint32(id)
	}
	return t.tk.Decode(uids, true)
}

// SpecialTokenID returns the ID for the given special token.
func (t *rustTokenizer) SpecialTokenID(token api.SpecialToken) (int, error) {
	if t.config == nil {
		return 0, fmt.Errorf("no tokenizer config available")
	}

	var tokenStr string
	switch token {
	case api.TokUnknown:
		tokenStr = t.config.UnkToken
	case api.TokPad:
		tokenStr = t.config.PadToken
	case api.TokBeginningOfSentence:
		tokenStr = t.config.BosToken
	case api.TokEndOfSentence:
		tokenStr = t.config.EosToken
	case api.TokClassification:
		tokenStr = t.config.ClsToken
	case api.TokMask:
		tokenStr = t.config.MaskToken
	default:
		return 0, fmt.Errorf("unknown special token: %s (%d)", token, int(token))
	}

	if tokenStr == "" {
		return 0, fmt.Errorf("special token %s not defined in config", token)
	}

	// Encode the special token to get its ID
	output := t.tk.EncodeWithOptions(tokenStr, false)
	if len(output.IDs) == 0 {
		return 0, fmt.Errorf("special token %s not found in vocabulary", tokenStr)
	}

	return int(output.IDs[0]), nil
}

// Close releases the Rust tokenizer resources.
func (t *rustTokenizer) Close() error {
	if t.tk != nil {
		return t.tk.Close()
	}
	return nil
}

// rustTokenizerAvailable returns true when the Rust tokenizer is available.
// Set TOKENIZER_BACKEND=go to force the pure Go tokenizer.
// Valid values: "rust" (default), "go"
func rustTokenizerAvailable() bool {
	backend := os.Getenv("TOKENIZER_BACKEND")
	return backend != "go"
}
