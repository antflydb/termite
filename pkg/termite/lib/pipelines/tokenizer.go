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
	"fmt"
	"os"
	"path/filepath"

	esentencepiece "github.com/eliben/go-sentencepiece"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/go-huggingface/tokenizers/hftokenizer"
)

// LoadTokenizer loads a tokenizer from a local model directory.
// It auto-detects the tokenizer type (HuggingFace tokenizer.json or SentencePiece tokenizer.model).
func LoadTokenizer(modelPath string) (tokenizers.Tokenizer, error) {
	// First, try to load tokenizer_config.json for class information
	var config *api.Config
	configPath := filepath.Join(modelPath, "tokenizer_config.json")
	if _, err := os.Stat(configPath); err == nil {
		config, err = api.ParseConfigFile(configPath)
		if err != nil {
			return nil, fmt.Errorf("parsing tokenizer config: %w", err)
		}
	}

	// Try tokenizer.json (HuggingFace Tokenizers format - BPE, WordPiece, etc.)
	tokenizerJSONPath := filepath.Join(modelPath, "tokenizer.json")
	if _, err := os.Stat(tokenizerJSONPath); err == nil {
		tok, err := hftokenizer.NewFromFile(config, tokenizerJSONPath)
		if err != nil {
			return nil, fmt.Errorf("loading tokenizer.json: %w", err)
		}
		return tok, nil
	}

	// Try tokenizer.model (SentencePiece format)
	spModelPath := filepath.Join(modelPath, "tokenizer.model")
	if _, err := os.Stat(spModelPath); err == nil {
		proc, err := esentencepiece.NewProcessorFromPath(spModelPath)
		if err != nil {
			return nil, fmt.Errorf("loading tokenizer.model: %w", err)
		}
		return &sentencepieceTokenizer{
			Processor: proc,
			Info:      proc.ModelInfo(),
		}, nil
	}

	return nil, fmt.Errorf("no tokenizer found in %s (expected tokenizer.json or tokenizer.model)", modelPath)
}

// sentencepieceTokenizer wraps esentencepiece.Processor to implement tokenizers.Tokenizer.
type sentencepieceTokenizer struct {
	*esentencepiece.Processor
	Info *esentencepiece.ModelInfo
}

// Ensure sentencepieceTokenizer implements tokenizers.Tokenizer
var _ tokenizers.Tokenizer = (*sentencepieceTokenizer)(nil)

// Encode returns the text encoded into a sequence of token IDs.
func (t *sentencepieceTokenizer) Encode(text string) []int {
	tokens := t.Processor.Encode(text)
	result := make([]int, len(tokens))
	for i, tok := range tokens {
		result[i] = tok.ID
	}
	return result
}

// Decode returns the text from a sequence of token IDs.
func (t *sentencepieceTokenizer) Decode(ids []int) string {
	return t.Processor.Decode(ids)
}

// SpecialTokenID returns the ID for the given special token, or an error if unknown.
func (t *sentencepieceTokenizer) SpecialTokenID(token api.SpecialToken) (int, error) {
	switch token {
	case api.TokUnknown:
		return t.Info.UnknownID, nil
	case api.TokPad:
		return t.Info.PadID, nil
	case api.TokBeginningOfSentence:
		return t.Info.BeginningOfSentenceID, nil
	case api.TokEndOfSentence:
		return t.Info.EndOfSentenceID, nil
	default:
		return 0, fmt.Errorf("unknown special token: %s (%d)", token, int(token))
	}
}

// MustLoadTokenizer loads a tokenizer and panics on error.
// Useful for tests and initialization code.
func MustLoadTokenizer(modelPath string) tokenizers.Tokenizer {
	tok, err := LoadTokenizer(modelPath)
	if err != nil {
		panic(fmt.Sprintf("failed to load tokenizer: %v", err))
	}
	return tok
}
