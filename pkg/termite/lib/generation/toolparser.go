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

package generation

import (
	"fmt"
	"sync"
)

// ToolParser handles prompt formatting and output parsing for tool calling.
// Different model families implement this interface with their specific formats.
type ToolParser interface {
	// Name returns the parser identifier (e.g., "functiongemma", "json", "hermes")
	Name() string

	// FormatToolsPrompt creates tool declarations for the system prompt.
	FormatToolsPrompt(tools []ToolDefinition) string

	// Feed processes incoming tokens and detects complete tool calls.
	// Returns newly completed tool calls (for real-time streaming emission).
	Feed(token string) []ToolCall

	// Finish completes parsing and returns all tool calls and remaining text.
	Finish() (toolCalls []ToolCall, remainingText string)

	// Reset clears parser state for reuse.
	Reset()
}

// ToolParserFactory creates a ToolParser for a given model path.
type ToolParserFactory func(modelPath string) (ToolParser, error)

// toolParserRegistry maps format names to factory functions.
var (
	toolParserRegistry   = make(map[string]ToolParserFactory)
	toolParserRegistryMu sync.RWMutex
)

func init() {
	// Register built-in parsers
	RegisterToolParser("functiongemma", NewFunctionGemmaParser)
}

// RegisterToolParser adds a new parser factory to the registry.
func RegisterToolParser(name string, factory ToolParserFactory) {
	toolParserRegistryMu.Lock()
	defer toolParserRegistryMu.Unlock()
	toolParserRegistry[name] = factory
}

// GetToolParser returns a parser for the given format name.
// Returns nil, nil if format is empty (no tool calling support).
// Returns an error if format is unknown.
func GetToolParser(format string, modelPath string) (ToolParser, error) {
	if format == "" {
		return nil, nil // No tool calling support
	}

	toolParserRegistryMu.RLock()
	factory, ok := toolParserRegistry[format]
	toolParserRegistryMu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("unknown tool_call_format: %s", format)
	}

	return factory(modelPath)
}

// ListToolParsers returns the names of all registered tool parsers.
func ListToolParsers() []string {
	toolParserRegistryMu.RLock()
	defer toolParserRegistryMu.RUnlock()

	names := make([]string, 0, len(toolParserRegistry))
	for name := range toolParserRegistry {
		names = append(names, name)
	}
	return names
}
