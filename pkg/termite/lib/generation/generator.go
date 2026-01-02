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

// ContentPart represents a part of a message content.
// It can be either text or an image URL.
type ContentPart struct {
	Type     string `json:"type"`                // "text" or "image_url"
	Text     string `json:"text,omitempty"`      // Text content (when type is "text")
	ImageURL string `json:"image_url,omitempty"` // Image URL or data URI (when type is "image_url")
}

// TextPart creates a text content part.
func TextPart(text string) ContentPart {
	return ContentPart{Type: "text", Text: text}
}

// ImagePart creates an image content part from a URL or data URI.
func ImagePart(url string) ContentPart {
	return ContentPart{Type: "image_url", ImageURL: url}
}

// Message represents a chat message in OpenAI-compatible format.
// Content can be either a simple string or an array of content parts for multimodal messages.
type Message struct {
	Role    string        `json:"role"`    // "system", "user", or "assistant"
	Content string        `json:"content"` // Simple text content (for backward compatibility)
	Parts   []ContentPart `json:"parts"`   // Multimodal content parts (optional, takes precedence if set)
}

// GetTextContent returns the text content of a message.
// If Parts is set, returns concatenated text from text parts.
// Otherwise returns the Content string.
func (m Message) GetTextContent() string {
	if len(m.Parts) > 0 {
		var text string
		for _, part := range m.Parts {
			if part.Type == "text" {
				text += part.Text
			}
		}
		return text
	}
	return m.Content
}

// HasImages returns true if the message contains any image content parts.
func (m Message) HasImages() bool {
	for _, part := range m.Parts {
		if part.Type == "image_url" {
			return true
		}
	}
	return false
}

// ToolDefinition represents a tool (function) the model can call.
type ToolDefinition struct {
	Type     string             `json:"type"`     // Always "function"
	Function FunctionDefinition `json:"function"` // The function definition
}

// FunctionDefinition describes a callable function.
type FunctionDefinition struct {
	Name        string                 `json:"name"`                  // Function name
	Description string                 `json:"description,omitempty"` // What the function does
	Parameters  map[string]interface{} `json:"parameters,omitempty"`  // JSON Schema for parameters
	Strict      bool                   `json:"strict,omitempty"`      // Enforce strict validation
}

// ToolCall represents a function call made by the model.
type ToolCall struct {
	ID       string           `json:"id"`       // Unique identifier
	Type     string           `json:"type"`     // Always "function"
	Function ToolCallFunction `json:"function"` // The function being called
}

// ToolCallFunction contains the function name and arguments.
type ToolCallFunction struct {
	Name      string `json:"name"`      // Function name
	Arguments string `json:"arguments"` // JSON-encoded arguments
}

// GenerateOptions configures text generation parameters.
type GenerateOptions struct {
	MaxTokens          int              `json:"max_tokens,omitempty"`           // Maximum tokens to generate
	Temperature        float32          `json:"temperature,omitempty"`          // Sampling temperature (0.0 = deterministic)
	TopP               float32          `json:"top_p,omitempty"`                // Nucleus sampling probability
	TopK               int              `json:"top_k,omitempty"`                // Top-k sampling
	Tools              []ToolDefinition `json:"tools,omitempty"`                // Tools the model can call
	ToolChoice         string           `json:"tool_choice,omitempty"`          // "auto", "none", or "required"
	ForcedFunctionName string           `json:"forced_function_name,omitempty"` // Force calling a specific function by name
}

// GenerateResult contains the output from text generation.
type GenerateResult struct {
	Text         string     `json:"text"`                    // Generated text
	TokensUsed   int        `json:"tokens_used"`             // Number of tokens generated
	FinishReason string     `json:"finish_reason"`           // "stop", "length", or "tool_calls"
	ToolCalls    []ToolCall `json:"tool_calls,omitempty"`    // Tool calls made by the model
}

// TokenDelta represents a single generated token in streaming mode.
type TokenDelta struct {
	Token string // The generated token text
	Index int    // Sequence index (for batch generation, usually 0)
}

// Generator is the interface for text generation models.
type Generator interface {
	// Generate produces text from the given messages using the specified options.
	Generate(ctx context.Context, messages []Message, opts GenerateOptions) (*GenerateResult, error)

	// Close releases any resources held by the generator.
	Close() error
}

// StreamingGenerator extends Generator with streaming support.
// All generators in Termite implement streaming natively.
// Use type assertion to access streaming:
//
//	if sg, ok := generator.(StreamingGenerator); ok {
//	    tokens, errs, err := sg.GenerateStream(ctx, messages, opts)
//	    // consume tokens channel
//	}
type StreamingGenerator interface {
	Generator

	// GenerateStream produces tokens one at a time via channels.
	// Returns:
	//   - tokens: channel of TokenDelta, closed when generation completes
	//   - errs: channel of errors during generation, closed when done
	//   - err: initialization error (if non-nil, channels are nil)
	GenerateStream(ctx context.Context, messages []Message, opts GenerateOptions) (
		tokens <-chan TokenDelta,
		errs <-chan error,
		err error,
	)
}

// ToolSupporter is implemented by generators that support tool calling.
// Use type assertion to check if a generator supports tools:
//
//	if ts, ok := generator.(ToolSupporter); ok && ts.SupportsTools() {
//	    parser := ts.ToolParser()
//	    // use parser to format tools and parse output
//	}
type ToolSupporter interface {
	// SupportsTools returns true if this generator supports tool calling.
	SupportsTools() bool

	// ToolParser returns the tool parser for this generator, or nil if not supported.
	ToolParser() ToolParser

	// ToolCallFormat returns the tool call format name (e.g., "functiongemma").
	ToolCallFormat() string
}
