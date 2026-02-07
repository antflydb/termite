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

package chunking

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/antflydb/antfly-go/libaf/chunking"
	"github.com/antflydb/termite/pkg/termite/lib/tokenizers"
)

// Fixed chunker model names
const (
	// ModelFixedBert uses BERT's WordPiece tokenization (~30k vocab).
	// Good for general-purpose text and multilingual content.
	ModelFixedBert = "fixed-bert-tokenizer"

	// ModelFixedBPE uses OpenAI's tiktoken BPE tokenization (cl100k_base, ~100k vocab).
	// Good for GPT-style models and code.
	ModelFixedBPE = "fixed-bpe-tokenizer"
)

// Ensure FixedChunker implements the Chunker interface
var _ chunking.Chunker = (*FixedChunker)(nil)

// FixedChunkerConfig contains configuration for the fixed chunker.
type FixedChunkerConfig struct {
	// Model specifies which tokenizer to use (ModelFixedBert or ModelFixedBPE)
	Model string

	// TargetTokens is the target number of tokens per chunk
	TargetTokens int

	// OverlapTokens is the number of overlapping tokens between chunks
	OverlapTokens int

	// Separator is the text separator for splitting
	Separator string

	// MaxChunks is the maximum number of chunks to generate
	MaxChunks int
}

// DefaultFixedChunkerConfig returns sensible defaults for the fixed chunker.
func DefaultFixedChunkerConfig() FixedChunkerConfig {
	return FixedChunkerConfig{
		Model:         ModelFixedBert,
		TargetTokens:  500,
		OverlapTokens: 50,
		Separator:     "\n\n",
		MaxChunks:     50,
	}
}

// FixedChunker splits text into fixed-size chunks
// while respecting token count targets
type FixedChunker struct {
	config    FixedChunkerConfig
	tokenizer tokenizers.TokenCounter
}

// NewFixedChunker creates a chunker that splits text into fixed-size chunks.
// Supported models:
// - "fixed-bert-tokenizer": BERT WordPiece tokenization (~30k vocab)
// - "fixed-bpe-tokenizer": OpenAI tiktoken BPE tokenization (cl100k_base, ~100k vocab)
func NewFixedChunker(config FixedChunkerConfig) (*FixedChunker, error) {
	// Apply defaults for zero values
	if config.Model == "" {
		config.Model = ModelFixedBert
	}
	if config.TargetTokens <= 0 {
		config.TargetTokens = 500
	}
	if config.OverlapTokens < 0 {
		config.OverlapTokens = 50
	}
	if config.Separator == "" {
		config.Separator = "\n\n"
	}
	if config.MaxChunks <= 0 {
		config.MaxChunks = 50
	}

	// Validate config
	if config.OverlapTokens >= config.TargetTokens {
		return nil, errors.New("overlap_tokens must be less than target_tokens")
	}

	// Create tokenizer
	tk, err := tokenizers.NewTokenCounter()
	if err != nil {
		return nil, fmt.Errorf("failed to create token counter: %w", err)
	}

	return &FixedChunker{
		config:    config,
		tokenizer: tk,
	}, nil
}

// Chunk splits text into chunks with per-request config overrides.
func (s *FixedChunker) Chunk(ctx context.Context, text string, opts chunking.ChunkOptions) ([]chunking.Chunk, error) {
	if text == "" {
		return nil, nil
	}

	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Resolve effective config by applying overrides (zero values mean "use default")
	effectiveConfig := s.config
	if opts.MaxChunks != 0 {
		effectiveConfig.MaxChunks = opts.MaxChunks
	}
	if opts.TargetTokens != 0 {
		effectiveConfig.TargetTokens = opts.TargetTokens
	}
	if opts.OverlapTokens != 0 {
		effectiveConfig.OverlapTokens = opts.OverlapTokens
	}
	if opts.Separator != "" {
		effectiveConfig.Separator = opts.Separator
	}
	// Note: Threshold is not applicable to FixedChunker (only used by ONNX models)

	// Split on separator to get candidate sections
	sections := strings.Split(text, effectiveConfig.Separator)

	// If splitting on the separator didn't produce multiple sections,
	// try progressively less strict separators
	if len(sections) <= 1 {
		// Try single newlines
		sections = strings.Split(text, "\n")
		if len(sections) <= 1 {
			// Try periods followed by space (sentences)
			sections = strings.Split(text, ". ")
			// Re-add the period to each section except the last
			for i := 0; i < len(sections)-1; i++ {
				sections[i] = sections[i] + "."
			}
		}
	}

	if len(sections) == 0 {
		return nil, nil
	}

	chunks := make([]chunking.Chunk, 0)
	currentChunk := strings.Builder{}
	currentStartChar := 0
	currentTokens := 0
	previousChunkText := "" // For overlap

	for _, section := range sections {
		// Check context cancellation periodically
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		section = strings.TrimSpace(section)
		if section == "" {
			continue
		}

		// Count tokens in this section
		sectionTokens := s.tokenizer.CountTokens(section)

		// Check if adding this section would exceed target
		if currentTokens > 0 && currentTokens+sectionTokens > effectiveConfig.TargetTokens {
			// Finalize current chunk
			chunkText := strings.TrimSpace(currentChunk.String())
			if chunkText != "" {
				chunks = append(chunks, chunking.NewTextChunk(
					uint32(len(chunks)),
					chunkText,
					currentStartChar,
					currentStartChar+len(chunkText),
				))

				// Check max chunks limit
				if len(chunks) >= effectiveConfig.MaxChunks {
					break
				}

				previousChunkText = chunkText
			}

			// Start new chunk with overlap from previous chunk
			currentChunk.Reset()
			currentStartChar = currentStartChar + len(chunkText) + len(effectiveConfig.Separator)

			// Add overlap from previous chunk if configured
			if effectiveConfig.OverlapTokens > 0 && previousChunkText != "" {
				overlapText := s.extractOverlap(previousChunkText, effectiveConfig.OverlapTokens)
				if overlapText != "" {
					currentChunk.WriteString(overlapText)
					currentChunk.WriteString(" ")
					currentTokens = s.tokenizer.CountTokens(overlapText)
				} else {
					currentTokens = 0
				}
			} else {
				currentTokens = 0
			}
		}

		// Add section to current chunk
		if currentChunk.Len() > 0 {
			currentChunk.WriteString(effectiveConfig.Separator)
		}
		currentChunk.WriteString(section)
		currentTokens += sectionTokens
	}

	// Add final chunk
	chunkText := strings.TrimSpace(currentChunk.String())
	if chunkText != "" && len(chunks) < effectiveConfig.MaxChunks {
		chunks = append(chunks, chunking.NewTextChunk(
			uint32(len(chunks)),
			chunkText,
			currentStartChar,
			currentStartChar+len(chunkText),
		))
	}

	// If no chunks were created (text too short), return single chunk
	if len(chunks) == 0 {
		chunks = append(chunks, chunking.NewTextChunk(
			0,
			strings.TrimSpace(text),
			0,
			len(text),
		))
	}

	return chunks, nil
}

// extractOverlap extracts the last N tokens from text for overlap
func (s *FixedChunker) extractOverlap(text string, targetTokens int) string {
	if text == "" || targetTokens <= 0 {
		return ""
	}

	// Fallback: take last ~targetTokens*4 characters
	// This is more reliable than trying to decode tokens, which requires a properly configured decoder
	overlapChars := targetTokens * 4
	if len(text) <= overlapChars {
		return text
	}
	return text[len(text)-overlapChars:]
}

// Close releases tokenizer resources
func (s *FixedChunker) Close() error {
	// Tokenizer doesn't need explicit closing
	return nil
}
