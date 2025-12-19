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

// Package generation provides text generation (LLM) functionality for Termite.
package generation

import "context"

// Message represents a chat message in OpenAI-compatible format.
type Message struct {
	Role    string `json:"role"`    // "system", "user", or "assistant"
	Content string `json:"content"` // The message content
}

// GenerateOptions configures text generation parameters.
type GenerateOptions struct {
	MaxTokens   int     `json:"max_tokens,omitempty"`   // Maximum tokens to generate
	Temperature float32 `json:"temperature,omitempty"` // Sampling temperature (0.0 = deterministic)
	TopP        float32 `json:"top_p,omitempty"`       // Nucleus sampling probability
	TopK        int     `json:"top_k,omitempty"`       // Top-k sampling
}

// GenerateResult contains the output from text generation.
type GenerateResult struct {
	Text         string `json:"text"`          // Generated text
	TokensUsed   int    `json:"tokens_used"`   // Number of tokens generated
	FinishReason string `json:"finish_reason"` // "stop" or "length"
}

// Generator is the interface for text generation models.
type Generator interface {
	// Generate produces text from the given messages using the specified options.
	Generate(ctx context.Context, messages []Message, opts GenerateOptions) (*GenerateResult, error)

	// Close releases any resources held by the generator.
	Close() error
}
