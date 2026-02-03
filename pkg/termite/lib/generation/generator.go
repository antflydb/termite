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
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync/atomic"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure PooledPipelineGenerator implements the Generator and StreamingGenerator interfaces
var _ Generator = (*PooledPipelineGenerator)(nil)
var _ StreamingGenerator = (*PooledPipelineGenerator)(nil)

// PooledPipelineGenerator manages multiple TextGenerationPipelines for concurrent text generation.
// Each request acquires a pipeline slot via semaphore, enabling true parallelism.
type PooledPipelineGenerator struct {
	pipelines      []*pipelines.TextGenerationPipeline
	sem            *semaphore.Weighted
	nextPipeline   atomic.Uint64
	logger         *zap.Logger
	poolSize       int
	modelPath      string
	toolParser     ToolParser
	toolCallFormat string
}

// PooledPipelineGeneratorConfig holds configuration for creating a PooledPipelineGenerator.
type PooledPipelineGeneratorConfig struct {
	// ModelPath is the path to the model directory.
	ModelPath string

	// PoolSize is the number of concurrent pipelines (0 = auto-detect from CPU count).
	PoolSize int

	// GenerationConfig holds text generation parameters. If nil, uses defaults.
	GenerationConfig *backends.GenerationConfig

	// Logger for logging. If nil, uses a no-op logger.
	Logger *zap.Logger
}

// NewPooledPipelineGenerator creates a new pooled generator using the session-based pipeline architecture.
func NewPooledPipelineGenerator(
	cfg *PooledPipelineGeneratorConfig,
	sessionManager *backends.SessionManager,
	modelBackends []string,
) (*PooledPipelineGenerator, backends.BackendType, error) {
	if cfg == nil {
		return nil, "", errors.New("config is required")
	}
	if cfg.ModelPath == "" {
		return nil, "", errors.New("model path is required")
	}

	logger := cfg.Logger
	if logger == nil {
		logger = zap.NewNop()
	}

	poolSize := cfg.PoolSize
	if poolSize <= 0 {
		poolSize = min(runtime.NumCPU(), 4)
	}

	logger.Info("Initializing pooled pipeline generator",
		zap.String("modelPath", cfg.ModelPath),
		zap.Int("poolSize", poolSize))

	// Build pipeline options with tool parser factory
	var opts []pipelines.TextGenerationPipelineOption
	if cfg.GenerationConfig != nil {
		opts = append(opts, pipelines.WithTextGenerationConfig(cfg.GenerationConfig))
	}

	// Pass the tool parser factory to the pipeline loader
	// This enables the pipeline to automatically load tool parsers from genai_config.json
	opts = append(opts, pipelines.WithToolParserFactory(func(modelPath string) (pipelines.ToolParser, error) {
		parser, err := GetToolParser("", modelPath)
		if err != nil {
			return nil, err
		}
		if parser == nil {
			return nil, nil
		}
		return &toolParserAdapter{parser}, nil
	}))

	// Create pooled pipelines using LoadTextGenerationPipeline
	pipelineSlice := make([]*pipelines.TextGenerationPipeline, poolSize)
	var backendType backends.BackendType

	for i := 0; i < poolSize; i++ {
		pipeline, bt, err := pipelines.LoadTextGenerationPipeline(
			cfg.ModelPath,
			sessionManager,
			modelBackends,
			opts...,
		)
		if err != nil {
			// Clean up already-created pipelines
			for j := 0; j < i; j++ {
				if pipelineSlice[j] != nil {
					pipelineSlice[j].Close()
				}
			}
			return nil, "", fmt.Errorf("loading text generation pipeline %d: %w", i, err)
		}
		pipelineSlice[i] = pipeline
		backendType = bt
		logger.Debug("Created pipeline", zap.Int("index", i))
	}

	// Get tool parser info from first pipeline
	var toolParser ToolParser
	var toolCallFormat string
	if len(pipelineSlice) > 0 && pipelineSlice[0].SupportsTools() {
		toolCallFormat = pipelineSlice[0].ToolCallFormat()
		if pipelineParser := pipelineSlice[0].GetToolParser(); pipelineParser != nil {
			// Unwrap the adapter to get our ToolParser
			if adapter, ok := pipelineParser.(*toolParserAdapter); ok {
				toolParser = adapter.inner
			}
		}
		// If we have a format but no parser from pipeline, try to load it directly
		if toolParser == nil && toolCallFormat != "" {
			var err error
			toolParser, err = GetToolParser(toolCallFormat, cfg.ModelPath)
			if err != nil {
				logger.Warn("Failed to load tool parser",
					zap.String("format", toolCallFormat),
					zap.Error(err))
			} else {
				logger.Info("Loaded tool parser from model config",
					zap.String("format", toolCallFormat))
			}
		}
	}

	logger.Info("Created pooled pipeline generator",
		zap.Int("poolSize", poolSize),
		zap.String("backend", string(backendType)))

	return &PooledPipelineGenerator{
		pipelines:      pipelineSlice,
		sem:            semaphore.NewWeighted(int64(poolSize)),
		logger:         logger,
		poolSize:       poolSize,
		modelPath:      cfg.ModelPath,
		toolParser:     toolParser,
		toolCallFormat: toolCallFormat,
	}, backendType, nil
}

// toolParserAdapter adapts our ToolParser to pipelines.ToolParser
type toolParserAdapter struct {
	inner ToolParser
}

func (a *toolParserAdapter) Name() string {
	return a.inner.Name()
}

func (a *toolParserAdapter) FormatToolsPrompt(tools []pipelines.ToolDefinition) string {
	// Convert pipelines.ToolDefinition to generation.ToolDefinition
	genTools := make([]ToolDefinition, len(tools))
	for i, t := range tools {
		genTools[i] = ToolDefinition{
			Type: t.Type,
			Function: FunctionDefinition{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  t.Function.Parameters,
				Strict:      t.Function.Strict,
			},
		}
	}
	return a.inner.FormatToolsPrompt(genTools)
}

func (a *toolParserAdapter) Feed(token string) []pipelines.ToolCall {
	calls := a.inner.Feed(token)
	// Convert generation.ToolCall to pipelines.ToolCall
	result := make([]pipelines.ToolCall, len(calls))
	for i, c := range calls {
		result[i] = pipelines.ToolCall{
			ID:   c.ID,
			Type: c.Type,
			Function: pipelines.ToolCallFunction{
				Name:      c.Function.Name,
				Arguments: c.Function.Arguments,
			},
		}
	}
	return result
}

func (a *toolParserAdapter) Finish() ([]pipelines.ToolCall, string) {
	calls, text := a.inner.Finish()
	result := make([]pipelines.ToolCall, len(calls))
	for i, c := range calls {
		result[i] = pipelines.ToolCall{
			ID:   c.ID,
			Type: c.Type,
			Function: pipelines.ToolCallFunction{
				Name:      c.Function.Name,
				Arguments: c.Function.Arguments,
			},
		}
	}
	return result, text
}

func (a *toolParserAdapter) Reset() {
	a.inner.Reset()
}

// Generate produces text from the given messages.
// Thread-safe: uses semaphore to limit concurrent pipeline access.
func (p *PooledPipelineGenerator) Generate(ctx context.Context, messages []Message, opts GenerateOptions) (*GenerateResult, error) {
	if len(messages) == 0 {
		return nil, errors.New("messages are required")
	}

	// Acquire semaphore slot (blocks if all pipelines busy)
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}
	defer p.sem.Release(1)

	// Round-robin pipeline selection
	idx := int(p.nextPipeline.Add(1) % uint64(p.poolSize))
	pipeline := p.pipelines[idx]

	p.logger.Debug("Using pipeline for generation",
		zap.Int("pipelineIndex", idx),
		zap.Int("numMessages", len(messages)))

	// Convert messages to prompt string
	prompt := messagesToPrompt(messages)

	// Apply options
	if opts.MaxTokens > 0 {
		pipeline.GenerationConfig.MaxNewTokens = opts.MaxTokens
	}
	if opts.Temperature > 0 {
		pipeline.GenerationConfig.Temperature = opts.Temperature
		pipeline.GenerationConfig.DoSample = true
	}
	if opts.TopP > 0 {
		pipeline.GenerationConfig.TopP = opts.TopP
		pipeline.GenerationConfig.DoSample = true
	}
	if opts.TopK > 0 {
		pipeline.GenerationConfig.TopK = opts.TopK
		pipeline.GenerationConfig.DoSample = true
	}

	// Run pipeline
	result, err := pipeline.Generate(ctx, prompt)
	if err != nil {
		p.logger.Error("Pipeline generation failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, fmt.Errorf("running text generation: %w", err)
	}

	finishReason := "stop"
	if !result.StoppedAtEOS {
		finishReason = "length"
	}

	genResult := &GenerateResult{
		Text:         result.Text,
		TokensUsed:   result.TokenCount,
		FinishReason: finishReason,
	}

	p.logger.Debug("Generation complete",
		zap.Int("pipelineIndex", idx),
		zap.Int("responseLength", len(result.Text)),
		zap.Int("tokensGenerated", result.TokenCount),
		zap.String("finishReason", genResult.FinishReason))

	return genResult, nil
}

// GenerateStream produces tokens one at a time via channels.
// Thread-safe: uses semaphore to limit concurrent pipeline access.
func (p *PooledPipelineGenerator) GenerateStream(ctx context.Context, messages []Message, opts GenerateOptions) (<-chan TokenDelta, <-chan error, error) {
	if len(messages) == 0 {
		return nil, nil, errors.New("messages are required")
	}

	// Acquire semaphore slot (blocks if all pipelines busy)
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}

	// Round-robin pipeline selection
	idx := int(p.nextPipeline.Add(1) % uint64(p.poolSize))
	pipeline := p.pipelines[idx]

	p.logger.Debug("Using pipeline for streaming generation",
		zap.Int("pipelineIndex", idx),
		zap.Int("numMessages", len(messages)))

	// Convert messages to prompt string
	prompt := messagesToPrompt(messages)

	// Apply options
	if opts.MaxTokens > 0 {
		pipeline.GenerationConfig.MaxNewTokens = opts.MaxTokens
	}
	if opts.Temperature > 0 {
		pipeline.GenerationConfig.Temperature = opts.Temperature
		pipeline.GenerationConfig.DoSample = true
	}
	if opts.TopP > 0 {
		pipeline.GenerationConfig.TopP = opts.TopP
		pipeline.GenerationConfig.DoSample = true
	}
	if opts.TopK > 0 {
		pipeline.GenerationConfig.TopK = opts.TopK
		pipeline.GenerationConfig.DoSample = true
	}

	// Create output channels
	tokenChan := make(chan TokenDelta)
	errChan := make(chan error, 1)

	go func() {
		defer p.sem.Release(1) // Release semaphore when done streaming
		defer close(tokenChan)
		defer close(errChan)

		// Streaming callback
		callback := func(token int32, text string) bool {
			select {
			case <-ctx.Done():
				return false
			case tokenChan <- TokenDelta{Token: text, Index: 0}:
				return true
			}
		}

		// Run with streaming
		_, err := pipeline.GenerateWithStreaming(ctx, prompt, callback)
		if err != nil {
			select {
			case errChan <- err:
			default:
			}
		}

		p.logger.Debug("Streaming generation complete", zap.Int("pipelineIndex", idx))
	}()

	return tokenChan, errChan, nil
}

// SupportsTools returns true if this generator supports tool calling.
func (p *PooledPipelineGenerator) SupportsTools() bool {
	return p.toolParser != nil
}

// ToolParser returns the tool parser for this generator, or nil if not supported.
func (p *PooledPipelineGenerator) ToolParser() ToolParser {
	return p.toolParser
}

// ToolCallFormat returns the tool call format name (e.g., "functiongemma").
func (p *PooledPipelineGenerator) ToolCallFormat() string {
	return p.toolCallFormat
}

// Close releases resources.
func (p *PooledPipelineGenerator) Close() error {
	p.logger.Info("Closing pooled pipeline generator", zap.Int("poolSize", p.poolSize))

	var errs []error
	for i, pipeline := range p.pipelines {
		if pipeline != nil {
			if err := pipeline.Close(); err != nil {
				p.logger.Warn("Error closing pipeline",
					zap.Int("index", i),
					zap.Error(err))
				errs = append(errs, err)
			}
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing generator: %v", errs)
	}
	return nil
}

// messagesToPrompt converts messages to a simple prompt string.
// For more sophisticated chat templating, consider using a proper chat template.
func messagesToPrompt(messages []Message) string {
	var prompt string
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			prompt += fmt.Sprintf("System: %s\n\n", msg.GetTextContent())
		case "user":
			prompt += fmt.Sprintf("User: %s\n\n", msg.GetTextContent())
		case "assistant":
			prompt += fmt.Sprintf("Assistant: %s\n\n", msg.GetTextContent())
		default:
			prompt += msg.GetTextContent() + "\n\n"
		}
	}
	prompt += "Assistant: "
	return prompt
}
