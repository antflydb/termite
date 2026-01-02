# FunctionGemma Tool Calling Support

## Summary

Add tool calling support to Termite's `/api/generate` endpoint. Uses a **parser registry** with format explicitly declared in model config - following vLLM's approach.

**Key insight**: Different model families use different output formats (FunctionGemma's escape-token format, JSON, Hermes, etc.). The format cannot be reliably auto-detected, so we explicitly declare it in `genai_config.json`. This works for renamed/fine-tuned models since the config travels with the model.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      API Request                                │
│  tools: [{type: "function", function: {...}}]                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Model Config (genai_config.json)              │
│  { "model_type": "generator", "tool_call_format": "functiongemma" }│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Parser Registry                               │
│  "functiongemma" → FunctionGemmaParser                          │
│  "json"          → JSONToolParser (future)                      │
│  "hermes"        → HermesParser (future)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Selected ToolParser                           │
│  - Prompt formatting (inject tool declarations)                 │
│  - Output parsing (detect tool calls in stream)                 │
└─────────────────────────────────────────────────────────────────┘
```

## Files to Modify

| File | Changes |
|------|---------|
| `pkg/termite/openapi.yaml` | Add tool schemas (same as OpenAI) |
| `pkg/termite/lib/generation/generator.go` | Add tool types to options/result |
| `pkg/termite/lib/generation/toolparser.go` | **NEW** - ToolParser interface + registry |
| `pkg/termite/lib/generation/toolparser_functiongemma.go` | **NEW** - FunctionGemma parser implementation |
| `pkg/termite/lib/generation/genai_config.go` | Add `tool_call_format` field |
| `pkg/termite/lib/generation/hugot.go` | Load parser from genai_config |
| `pkg/termite/api.go` | Handle tools in request/response |
| `scripts/export_model_to_registry.py` | Auto-detect and set `tool_call_format` during export |

## Implementation Steps

### Step 1: OpenAPI Schema Updates (`pkg/termite/openapi.yaml`)

Add OpenAI-compatible tool schemas. These are the same regardless of which model parser is used:

```yaml
# New schemas
Tool:
  type: object
  required: [type, function]
  properties:
    type:
      type: string
      enum: [function]
    function:
      $ref: "#/components/schemas/FunctionDefinition"

FunctionDefinition:
  type: object
  required: [name]
  properties:
    name:
      type: string
    description:
      type: string
    parameters:
      type: object
      additionalProperties: true
    strict:
      type: boolean

ToolCall:
  type: object
  required: [id, type, function]
  properties:
    id:
      type: string
    type:
      type: string
      enum: [function]
    function:
      $ref: "#/components/schemas/ToolCallFunction"

ToolCallFunction:
  type: object
  required: [name, arguments]
  properties:
    name:
      type: string
    arguments:
      type: string

# Extend GenerateRequest
tools:
  type: array
  items:
    $ref: "#/components/schemas/Tool"
tool_choice:
  type: string
  enum: [auto, none, required]

# Extend GenerateMessage and GenerateDelta
tool_calls:
  type: array
  items:
    $ref: "#/components/schemas/ToolCall"

# Extend Role enum
enum: [system, user, assistant, tool]

# Add tool_call_id for tool response messages
tool_call_id:
  type: string
```

### Step 2: Generator Types (`pkg/termite/lib/generation/generator.go`)

Add tool types to the generation package:

```go
// Tool types (mirror OpenAPI schemas)
type ToolDefinition struct {
    Type     string             `json:"type"`
    Function FunctionDefinition `json:"function"`
}

type FunctionDefinition struct {
    Name        string                 `json:"name"`
    Description string                 `json:"description,omitempty"`
    Parameters  map[string]interface{} `json:"parameters,omitempty"`
    Strict      bool                   `json:"strict,omitempty"`
}

type ToolCall struct {
    ID       string           `json:"id"`
    Type     string           `json:"type"`
    Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
    Name      string `json:"name"`
    Arguments string `json:"arguments"` // JSON string
}

// Extend GenerateOptions
type GenerateOptions struct {
    MaxTokens   int
    Temperature float32
    TopP        float32
    TopK        int
    Tools       []ToolDefinition  // NEW
    ToolChoice  string            // NEW: "auto", "none", "required"
}

// Extend GenerateResult
type GenerateResult struct {
    Text         string
    TokensUsed   int
    FinishReason string     // "stop", "length", "tool_calls"
    ToolCalls    []ToolCall // NEW
}
```

### Step 3: Tool Parser Interface & Registry (`pkg/termite/lib/generation/toolparser.go` - NEW)

Define interface and registry for tool parsers:

```go
package generation

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

// Parser registry - maps format names to factory functions
var toolParserRegistry = map[string]func(modelPath string) (ToolParser, error){
    "functiongemma": NewFunctionGemmaParser,
    // Future: "json", "hermes", etc.
}

// GetToolParser returns a parser for the given format name.
// Returns nil if format is empty or unknown.
func GetToolParser(format string, modelPath string) (ToolParser, error) {
    if format == "" {
        return nil, nil // No tool calling support
    }
    factory, ok := toolParserRegistry[format]
    if !ok {
        return nil, fmt.Errorf("unknown tool_call_format: %s", format)
    }
    return factory(modelPath)
}

// RegisterToolParser adds a new parser to the registry.
func RegisterToolParser(name string, factory func(modelPath string) (ToolParser, error)) {
    toolParserRegistry[name] = factory
}
```

### Step 3b: FunctionGemma Parser (`pkg/termite/lib/generation/toolparser_functiongemma.go` - NEW)

FunctionGemma-specific implementation that reads tokens from tokenizer config:

```go
package generation

import (
    "encoding/json"
    "fmt"
    "os"
    "path/filepath"
    "strings"
)

// FunctionGemmaTokens holds special tokens read from model's tokenizer config.
type FunctionGemmaTokens struct {
    StartFunctionDecl string `json:"start_function_declaration"`
    EndFunctionDecl   string `json:"end_function_declaration"`
    StartFunctionCall string `json:"start_function_call"`
    EndFunctionCall   string `json:"end_function_call"`
    Escape            string `json:"escape"`
}

// FunctionGemmaParser handles FunctionGemma's tool calling format.
type FunctionGemmaParser struct {
    tokens        FunctionGemmaTokens
    buffer        strings.Builder
    toolCalls     []ToolCall
    textSegments  []string
    callIDCounter int
}

// NewFunctionGemmaParser creates a parser, loading tokens from model config.
func NewFunctionGemmaParser(modelPath string) (ToolParser, error) {
    tokens, err := loadFunctionGemmaTokens(modelPath)
    if err != nil {
        return nil, fmt.Errorf("load FunctionGemma tokens: %w", err)
    }
    return &FunctionGemmaParser{tokens: tokens}, nil
}

func loadFunctionGemmaTokens(modelPath string) (FunctionGemmaTokens, error) {
    // Try special_tokens_map.json first
    tokenMapPath := filepath.Join(modelPath, "special_tokens_map.json")
    if tokens, err := loadTokensFromFile(tokenMapPath); err == nil {
        return tokens, nil
    }

    // Fall back to tokenizer_config.json
    tokenizerPath := filepath.Join(modelPath, "tokenizer_config.json")
    return loadTokensFromFile(tokenizerPath)
}

func loadTokensFromFile(path string) (FunctionGemmaTokens, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return FunctionGemmaTokens{}, err
    }
    var tokens FunctionGemmaTokens
    if err := json.Unmarshal(data, &tokens); err != nil {
        return FunctionGemmaTokens{}, err
    }
    if tokens.StartFunctionCall == "" || tokens.EndFunctionCall == "" {
        return FunctionGemmaTokens{}, fmt.Errorf("missing required tokens")
    }
    return tokens, nil
}

func (p *FunctionGemmaParser) Name() string { return "functiongemma" }

func (p *FunctionGemmaParser) FormatToolsPrompt(tools []ToolDefinition) string {
    var sb strings.Builder
    sb.WriteString("You are a model that can do function calling with the following functions.\n\n")

    for _, tool := range tools {
        sb.WriteString(p.tokens.StartFunctionDecl)
        sb.WriteString("declaration:")
        sb.WriteString(tool.Function.Name)
        sb.WriteString("{description:")
        sb.WriteString(p.tokens.Escape)
        sb.WriteString(tool.Function.Description)
        sb.WriteString(p.tokens.Escape)
        sb.WriteString(",parameters:")
        sb.WriteString(p.formatParams(tool.Function.Parameters))
        sb.WriteString("}")
        sb.WriteString(p.tokens.EndFunctionDecl)
        sb.WriteString("\n")
    }
    return sb.String()
}

// ... (formatParams, Feed, parseCall, splitParams, Finish, Reset - same logic as before)
```

### Step 4: genai_config.json Schema (`pkg/termite/lib/generation/genai_config.go`)

Add `tool_call_format` field to the existing genai config:

```go
// GenAIConfig represents the model's genai_config.json
type GenAIConfig struct {
    ModelType      string `json:"model_type"`       // "generator"
    ToolCallFormat string `json:"tool_call_format"` // NEW: "functiongemma", "json", etc.
    // ... existing fields ...
}
```

Example `genai_config.json` for FunctionGemma:
```json
{
    "model_type": "generator",
    "tool_call_format": "functiongemma"
}
```

### Step 5: Model Loading (`pkg/termite/lib/generation/hugot.go`)

Load tool parser from genai_config.json using the registry:

```go
type HugotGenerator struct {
    // ... existing fields ...
    imageToken  string     // image placeholder token
    toolParser  ToolParser // tool call parser (from genai_config.json)
}

func NewHugotGenerator(modelPath string, ...) (*HugotGenerator, error) {
    // ... existing initialization ...

    // Read genai_config.json
    config := readGenAIConfig(modelPath)

    // Read image token (existing)
    imageToken := readImageToken(modelPath)

    // Load tool parser from registry based on config (NEW)
    var toolParser ToolParser
    if config.ToolCallFormat != "" {
        var err error
        toolParser, err = GetToolParser(config.ToolCallFormat, modelPath)
        if err != nil {
            return nil, fmt.Errorf("load tool parser: %w", err)
        }
        logger.Info("Loaded tool parser",
            zap.String("format", config.ToolCallFormat))
    }

    return &HugotGenerator{
        // ... existing fields ...
        imageToken: imageToken,
        toolParser: toolParser,
    }, nil
}

func (g *HugotGenerator) SupportsTools() bool {
    return g.toolParser != nil
}

func (g *HugotGenerator) ToolParser() ToolParser {
    return g.toolParser
}
```

### Step 6: API Handler Updates (`pkg/termite/api.go`)

Handle tools in request/response and emit tool calls during streaming:

```go
// In handleApiGenerate():

// Convert API tools to internal types
var tools []generation.ToolDefinition
if req.Tools != nil {
    for _, t := range *req.Tools {
        tools = append(tools, generation.ToolDefinition{
            Type: string(t.Type),
            Function: generation.FunctionDefinition{
                Name:        t.Function.Name,
                Description: deref(t.Function.Description),
                Parameters:  t.Function.Parameters,
                Strict:      deref(t.Function.Strict),
            },
        })
    }
}

opts := generation.GenerateOptions{
    // ... existing fields ...
    Tools:      tools,
    ToolChoice: parseToolChoice(req.ToolChoice),
}

// Check if model supports tools
if len(tools) > 0 {
    if gen, ok := generator.(*generation.HugotGenerator); ok && !gen.SupportsTools() {
        writeJSONError(w, http.StatusBadRequest, "model does not support tool calling")
        return
    }
}

// Non-streaming: add tool_calls to response message
if len(result.ToolCalls) > 0 {
    apiToolCalls := convertToolCalls(result.ToolCalls)
    message.ToolCalls = &apiToolCalls
}

// Map finish reason
switch result.FinishReason {
case "tool_calls":
    finishReason = FinishReasonToolCalls
// ...
}

// Streaming: emit tool calls in real-time
func (ln *TermiteNode) handleStreamingGenerate(...) {
    // Get parser if tools provided
    var parser generation.ToolParser
    if len(opts.Tools) > 0 {
        if gen, ok := generator.(*generation.HugotGenerator); ok {
            parser = gen.ToolParser()
        }
    }

    var allToolCalls []generation.ToolCall

    for token := range tokenChan {
        // Feed to parser if active
        var newCalls []generation.ToolCall
        if parser != nil {
            newCalls = parser.Feed(token.Token)
        }

        // Send token chunk
        // ...

        // Emit newly detected tool calls immediately
        for _, tc := range newCalls {
            allToolCalls = append(allToolCalls, tc)

            // Send tool call chunk
            toolCallChunk := GenerateChunk{
                // ...
                Choices: []GenerateChunkChoice{{
                    Index: 0,
                    Delta: GenerateDelta{ToolCalls: convertToolCalls([]generation.ToolCall{tc})},
                }},
            }
            // ...
        }
    }

    // Final chunk with finish_reason
    finishReason := FinishReasonStop
    if len(allToolCalls) > 0 {
        finishReason = FinishReasonToolCalls
    }
    // ...
}
```

### Step 7: Export Script Update (`scripts/export_model_to_registry.py`)

Auto-detect tool calling format and write to `genai_config.json`:

```python
def detect_tool_call_format(model_id: str, tokenizer_config: dict) -> str | None:
    """Detect tool calling format from model ID and tokenizer config."""
    model_lower = model_id.lower()

    # Check for FunctionGemma (has specific tokens in tokenizer config)
    if "functiongemma" in model_lower:
        return "functiongemma"

    # Check tokenizer config for FunctionGemma-style tokens
    special_tokens = tokenizer_config.get("added_tokens_decoder", {})
    for token_info in special_tokens.values():
        content = token_info.get("content", "")
        if "<start_function_call>" in content:
            return "functiongemma"

    # Check for Hermes-style (future)
    if "hermes" in model_lower:
        return "hermes"

    # Check for models with chat_template that includes tool_use
    chat_template = tokenizer_config.get("chat_template", "")
    if isinstance(chat_template, dict) and "tool_use" in chat_template:
        return "json"  # Standard JSON format

    return None  # No tool calling support detected

def export_generator_model(model_id: str, output_dir: Path) -> None:
    # ... existing export logic ...

    # Load tokenizer config for detection
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    tokenizer_config = {}
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path) as f:
            tokenizer_config = json.load(f)

    # Detect and set tool_call_format
    tool_format = detect_tool_call_format(model_id, tokenizer_config)

    # Write genai_config.json
    genai_config = {
        "model_type": "generator",
    }
    if tool_format:
        genai_config["tool_call_format"] = tool_format
        logger.info(f"Detected tool_call_format: {tool_format}")

    with open(output_dir / "genai_config.json", "w") as f:
        json.dump(genai_config, f, indent=2)
```

**Key points:**
- Format is auto-detected during export and stored in `genai_config.json`
- Works for renamed/fine-tuned models since config travels with model files
- Can be manually overridden by editing `genai_config.json`
- Detection checks: model name, tokenizer special tokens, chat_template

### Step 8: Code Generation & Testing

```bash
# Generate code from OpenAPI
cd pkg/termite && make generate

# Build
go build ./...

# Test parser
go test ./pkg/termite/lib/generation/... -v -run TestFunctionGemma

# Export FunctionGemma model
./scripts/export_model_to_registry.py generator google/functiongemma-270m-it

# Integration test with actual model
go test ./e2e/... -v -run TestToolCalling
```

## Testing

1. **Unit tests** (`pkg/termite/lib/generation/toolparser_functiongemma_test.go`):
   - `TestFormatToolsPrompt` - verify FunctionGemma declaration format
   - `TestFeed` - streaming token parsing
   - `TestFinish` - complete parsing with multiple calls
   - `TestParseCall` - individual call parsing

2. **Integration test** with actual FunctionGemma model:
   - Export model with `./scripts/export_model_to_registry.py generator google/functiongemma-270m-it`
   - Test non-streaming tool call
   - Test streaming with real-time emission

3. **OpenAI SDK compatibility**:
   ```python
   from openai import OpenAI
   client = OpenAI(base_url="http://localhost:8084/api/v1", api_key="unused")

   response = client.chat.completions.create(
       model="functiongemma-270m-it",
       messages=[{"role": "user", "content": "What's the weather in London?"}],
       tools=[{
           "type": "function",
           "function": {
               "name": "get_weather",
               "description": "Get current weather",
               "parameters": {
                   "type": "object",
                   "properties": {"location": {"type": "string"}},
                   "required": ["location"]
               }
           }
       }]
   )
   ```

## Design Decisions

### Explicit Format Declaration (vLLM-style)
Different model families use different output formats that require different parsing logic. Rather than magic auto-detection:

- **Format declared in `genai_config.json`**: `"tool_call_format": "functiongemma"`
- **Auto-detected during export**: Export script detects format and writes to config
- **Works for renamed/fine-tuned models**: Config travels with model files
- **Follows vLLM's approach**: They use `--tool-call-parser` flag for same reason

### Parser Registry Pattern
```go
// Format name → Parser factory
var toolParserRegistry = map[string]func(modelPath string) (ToolParser, error){
    "functiongemma": NewFunctionGemmaParser,
    "json":          NewJSONToolParser,      // future
    "hermes":        NewHermesParser,        // future
}

// Load parser based on config
parser, err := GetToolParser(config.ToolCallFormat, modelPath)
```

Benefits:
- Easy to add new formats (just register a new parser)
- Clear separation between format families
- Each parser can read its own config (tokens, templates, etc.)

### Tokens Read by Parser, Not Generic Loader
Each parser implementation reads its own tokens from config files:
- FunctionGemma parser reads `<start_function_call>`, `<escape>` from tokenizer config
- JSON parser would read different tokens
- This avoids a one-size-fits-all token struct that doesn't fit all formats

### Streaming Parser Interface
The `Feed(token) → []ToolCall` interface enables:
- Real-time tool call detection during streaming
- Immediate emission to client (like OpenAI does)
- Reuse between streaming and non-streaming paths

## Adding New Formats

To add support for a new model family (e.g., Hermes):

1. Create `toolparser_hermes.go` implementing `ToolParser` interface
2. Register in `toolparser.go`: `"hermes": NewHermesParser`
3. Update export script detection to recognize Hermes models
