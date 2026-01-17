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
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure HugotGenerator implements the Generator, StreamingGenerator, and ToolSupporter interfaces
var _ Generator = (*HugotGenerator)(nil)
var _ StreamingGenerator = (*HugotGenerator)(nil)
var _ ToolSupporter = (*HugotGenerator)(nil)
var _ Generator = (*PooledHugotGenerator)(nil)
var _ StreamingGenerator = (*PooledHugotGenerator)(nil)
var _ ToolSupporter = (*PooledHugotGenerator)(nil)

// toHugotMessages converts internal Message types to Hugot's backends.Message format.
// Supports multimodal content via the Parts field.
// The imageToken parameter specifies the placeholder token for images (e.g., "<start_of_image>").
// If imageToken is empty, no image tokens are prepended.
func toHugotMessages(messages []Message, imageToken string) []backends.Message {
	hugotMessages := make([]backends.Message, len(messages))
	for i, m := range messages {
		msg := backends.Message{
			Role: m.Role,
		}

		// If Parts is set, extract text and images from content parts
		if len(m.Parts) > 0 {
			var textParts []string
			var imageURLs []string

			for _, part := range m.Parts {
				switch part.Type {
				case "text":
					if part.Text != "" {
						textParts = append(textParts, part.Text)
					}
				case "image_url":
					if part.ImageURL != "" {
						imageURLs = append(imageURLs, part.ImageURL)
					}
				}
			}

			// Set ImageURLs if any images were found
			if len(imageURLs) > 0 {
				msg.ImageURLs = imageURLs
			}

			// Combine all text parts into Content
			// For multimodal models, prepend image tokens for each image
			// so the model knows where to insert image embeddings.
			// The token format is read from the model's special_tokens_map.json.
			if len(textParts) > 0 || len(imageURLs) > 0 {
				// Add image token before text (one per image)
				imageTokens := ""
				if imageToken != "" {
					for range imageURLs {
						imageTokens += imageToken
					}
				}
				msg.Content = imageTokens
				for _, text := range textParts {
					msg.Content += text
				}
			}
		} else {
			// Backward compatibility: use Content field directly if Parts is empty
			msg.Content = m.Content
		}

		hugotMessages[i] = msg
	}
	return hugotMessages
}

// specialTokensMap represents the structure of special_tokens_map.json
type specialTokensMap struct {
	BOIToken   string `json:"boi_token"`   // Begin of image token (e.g., "<start_of_image>")
	EOIToken   string `json:"eoi_token"`   // End of image token (e.g., "<end_of_image>")
	ImageToken string `json:"image_token"` // Image soft token (e.g., "<image_soft_token>")
}

// GenAIConfig represents the model's genai_config.json with Termite extensions.
type GenAIConfig struct {
	// Standard ONNX Runtime GenAI fields
	Model struct {
		Type      string `json:"type"`
		VocabSize int    `json:"vocab_size"`
	} `json:"model"`

	// Termite extension: tool calling format
	ToolCallFormat string `json:"tool_call_format,omitempty"` // "functiongemma", "json", "hermes", etc.
}

// readGenAIConfig reads the genai_config.json file from the model directory.
func readGenAIConfig(modelPath string) *GenAIConfig {
	configPath := filepath.Join(modelPath, "genai_config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil
	}

	var config GenAIConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil
	}

	return &config
}

// readImageToken reads the image placeholder token from the model's special_tokens_map.json.
// Returns the boi_token (begin of image) if available, otherwise returns an empty string.
func readImageToken(modelPath string) string {
	tokenMapPath := filepath.Join(modelPath, "special_tokens_map.json")
	data, err := os.ReadFile(tokenMapPath)
	if err != nil {
		return "" // File not found or not readable
	}

	var tokenMap specialTokensMap
	if err := json.Unmarshal(data, &tokenMap); err != nil {
		return "" // Invalid JSON
	}

	// Prefer boi_token (begin of image) as it's the marker placed in the prompt
	if tokenMap.BOIToken != "" {
		return tokenMap.BOIToken
	}
	return ""
}

// HugotGenerator wraps a Hugot TextGenerationPipeline for LLM inference.
type HugotGenerator struct {
	session        *khugot.Session
	pipeline       *pipelines.TextGenerationPipeline // streaming-enabled pipeline
	logger         *zap.Logger
	sessionShared  bool       // true if session is shared and shouldn't be destroyed
	imageToken     string     // image placeholder token from model's special_tokens_map.json
	toolParser     ToolParser // tool call parser (from genai_config.json)
	toolCallFormat string     // the tool call format name (e.g., "functiongemma")
}

// NewHugotGenerator creates a new generator using the Hugot runtime.
// Note: For generative models, hugot uses genai_config.json to determine model files,
// so onnxFilename is ignored.
func NewHugotGenerator(modelPath string, logger *zap.Logger) (*HugotGenerator, error) {
	return NewHugotGeneratorWithSession(modelPath, nil, logger)
}

// NewHugotGeneratorWithSession creates a new generator using an optional shared Hugot session.
// If sharedSession is nil, a new session is created.
// If sharedSession is provided, it will be reused (important for ONNX Runtime which allows only one session).
// Note: For generative models, hugot uses genai_config.json to determine model files.
func NewHugotGeneratorWithSession(modelPath string, sharedSession *khugot.Session, logger *zap.Logger) (*HugotGenerator, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Initializing Hugot generator",
		zap.String("modelPath", modelPath),
		zap.String("backend", hugot.BackendName()))

	// Use shared session or create a new one
	session, err := hugot.NewSessionOrUseExisting(sharedSession)
	if err != nil {
		logger.Error("Failed to create Hugot session", zap.Error(err))
		return nil, fmt.Errorf("creating hugot session: %w", err)
	}
	sessionShared := (sharedSession != nil)
	if sessionShared {
		logger.Info("Using shared Hugot session", zap.String("backend", hugot.BackendName()))
	} else {
		logger.Info("Created new Hugot session", zap.String("backend", hugot.BackendName()))
	}

	// Create text generation pipeline with streaming enabled
	// We use a single streaming pipeline for both streaming and non-streaming calls
	// Note: OnnxFilename is intentionally not set - generative models use genai_config.json
	pipelineName := fmt.Sprintf("generator:%s", modelPath)
	pipelineConfig := khugot.TextGenerationConfig{
		ModelPath: modelPath,
		Name:      pipelineName,
		Options: []backends.PipelineOption[*pipelines.TextGenerationPipeline]{
			pipelines.WithMaxLength(2048),
			pipelines.WithStreaming(),
		},
	}

	pipeline, err := khugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		// Only destroy session if we created it (not shared)
		if !sessionShared {
			_ = session.Destroy()
		}
		logger.Error("Failed to create pipeline", zap.Error(err))
		return nil, fmt.Errorf("creating text generation pipeline: %w", err)
	}
	logger.Info("Successfully created streaming-enabled text generation pipeline")

	// Read image token from model's special_tokens_map.json
	imageToken := readImageToken(modelPath)
	if imageToken != "" {
		logger.Info("Loaded image token from model config", zap.String("imageToken", imageToken))
	}

	// Read genai_config.json for tool calling support
	var toolParser ToolParser
	var toolCallFormat string
	if config := readGenAIConfig(modelPath); config != nil && config.ToolCallFormat != "" {
		toolCallFormat = config.ToolCallFormat
		var err error
		toolParser, err = GetToolParser(config.ToolCallFormat, modelPath)
		if err != nil {
			logger.Warn("Failed to load tool parser",
				zap.String("format", config.ToolCallFormat),
				zap.Error(err))
		} else {
			logger.Info("Loaded tool parser from model config",
				zap.String("format", config.ToolCallFormat))
		}
	}

	return &HugotGenerator{
		session:        session,
		pipeline:       pipeline,
		logger:         logger,
		sessionShared:  sessionShared,
		imageToken:     imageToken,
		toolParser:     toolParser,
		toolCallFormat: toolCallFormat,
	}, nil
}

// SupportsTools returns true if this generator supports tool calling.
func (g *HugotGenerator) SupportsTools() bool {
	return g.toolParser != nil
}

// ToolParser returns the tool parser for this generator, or nil if not supported.
func (g *HugotGenerator) ToolParser() ToolParser {
	return g.toolParser
}

// ToolCallFormat returns the tool call format name (e.g., "functiongemma").
func (g *HugotGenerator) ToolCallFormat() string {
	return g.toolCallFormat
}

// convertMessages converts internal Message types to Hugot's backends.Message format
// using the generator's configured image token.
func (g *HugotGenerator) convertMessages(messages []Message) []backends.Message {
	return toHugotMessages(messages, g.imageToken)
}

// NewHugotGeneratorWithSessionManager creates a new generator using a SessionManager.
// This is the preferred way to create generators when using shared session management.
// Returns the generator, the backend type used, and any error.
func NewHugotGeneratorWithSessionManager(modelPath string, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*HugotGenerator, hugot.BackendType, error) {
	if modelPath == "" {
		return nil, "", errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	if sessionManager == nil {
		// Fall back to creating a new session
		model, err := NewHugotGeneratorWithSession(modelPath, nil, logger)
		if err != nil {
			return nil, "", err
		}
		return model, hugot.BackendType(""), nil
	}

	logger.Info("Initializing Hugot generator with SessionManager",
		zap.String("modelPath", modelPath))

	// Get session from SessionManager with backend restrictions
	session, backendUsed, err := sessionManager.GetSessionForModel(modelBackends)
	if err != nil {
		return nil, "", fmt.Errorf("getting session from manager: %w", err)
	}

	logger.Info("Got session from SessionManager",
		zap.String("backend", string(backendUsed)))

	// Create text generation pipeline with streaming enabled
	pipelineName := fmt.Sprintf("generator:%s", modelPath)
	pipelineConfig := khugot.TextGenerationConfig{
		ModelPath: modelPath,
		Name:      pipelineName,
		Options: []backends.PipelineOption[*pipelines.TextGenerationPipeline]{
			pipelines.WithMaxLength(2048),
			pipelines.WithStreaming(),
		},
	}

	pipeline, err := khugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		logger.Error("Failed to create pipeline", zap.Error(err))
		return nil, "", fmt.Errorf("creating text generation pipeline: %w", err)
	}
	logger.Info("Successfully created streaming-enabled text generation pipeline")

	// Read image token from model's special_tokens_map.json
	imageToken := readImageToken(modelPath)
	if imageToken != "" {
		logger.Info("Loaded image token from model config", zap.String("imageToken", imageToken))
	}

	// Read genai_config.json for tool calling support
	var toolParser ToolParser
	var toolCallFormat string
	if config := readGenAIConfig(modelPath); config != nil && config.ToolCallFormat != "" {
		toolCallFormat = config.ToolCallFormat
		var err error
		toolParser, err = GetToolParser(config.ToolCallFormat, modelPath)
		if err != nil {
			logger.Warn("Failed to load tool parser",
				zap.String("format", config.ToolCallFormat),
				zap.Error(err))
		} else {
			logger.Info("Loaded tool parser from model config",
				zap.String("format", config.ToolCallFormat))
		}
	}

	return &HugotGenerator{
		session:        session,
		pipeline:       pipeline,
		logger:         logger,
		sessionShared:  true, // SessionManager-provided sessions are always shared
		imageToken:     imageToken,
		toolParser:     toolParser,
		toolCallFormat: toolCallFormat,
	}, backendUsed, nil
}

// Generate produces text from the given messages.
// Uses the streaming pipeline internally and collects all tokens into the response.
func (g *HugotGenerator) Generate(ctx context.Context, messages []Message, opts GenerateOptions) (*GenerateResult, error) {
	if len(messages) == 0 {
		return nil, errors.New("messages are required")
	}

	g.logger.Debug("Starting generation",
		zap.Int("numMessages", len(messages)),
		zap.Int("maxTokens", opts.MaxTokens),
	)

	// Convert messages to Hugot format using the generator's image token
	hugotMessages := g.convertMessages(messages)

	// Debug: log converted message content
	for i, msg := range hugotMessages {
		g.logger.Debug("Converted message",
			zap.Int("index", i),
			zap.String("role", msg.Role),
			zap.String("content", msg.Content),
			zap.Int("numImages", len(msg.ImageURLs)),
			zap.String("imageToken", g.imageToken),
		)
	}

	output, err := g.pipeline.RunMessages(ctx, [][]backends.Message{hugotMessages})
	if err != nil {
		g.logger.Error("Pipeline generation failed", zap.Error(err))
		return nil, fmt.Errorf("running text generation: %w", err)
	}

	// Collect all tokens from the stream
	var generatedText string
	var tokenCount int
	var genErr error

	for delta := range output.TokenStream {
		generatedText += delta.Token
		tokenCount++
	}

	// Check for errors from the error stream
	for err := range output.ErrorStream {
		genErr = err
	}

	if genErr != nil {
		g.logger.Error("Generation error", zap.Error(genErr))
		return nil, fmt.Errorf("generation error: %w", genErr)
	}

	result := &GenerateResult{
		Text:         generatedText,
		TokensUsed:   tokenCount,
		FinishReason: "stop",
	}

	g.logger.Debug("Generation complete",
		zap.Int("responseLength", len(generatedText)),
		zap.Int("tokensGenerated", tokenCount),
		zap.String("finishReason", result.FinishReason),
	)

	return result, nil
}

// GenerateStream produces tokens one at a time via channels.
func (g *HugotGenerator) GenerateStream(ctx context.Context, messages []Message, opts GenerateOptions) (<-chan TokenDelta, <-chan error, error) {
	if len(messages) == 0 {
		return nil, nil, errors.New("messages are required")
	}

	g.logger.Debug("Starting streaming generation",
		zap.Int("numMessages", len(messages)),
		zap.Int("maxTokens", opts.MaxTokens),
	)

	// Convert messages to Hugot format using the generator's image token
	hugotMessages := g.convertMessages(messages)
	output, err := g.pipeline.RunMessages(ctx, [][]backends.Message{hugotMessages})
	if err != nil {
		g.logger.Error("Streaming pipeline generation failed", zap.Error(err))
		return nil, nil, fmt.Errorf("running streaming text generation: %w", err)
	}

	// Adapt hugot's channels to our TokenDelta format
	tokenChan := make(chan TokenDelta)
	errChan := make(chan error, 1)

	go func() {
		defer close(tokenChan)
		defer close(errChan)

		// Read from hugot's token stream
		for delta := range output.TokenStream {
			select {
			case <-ctx.Done():
				return
			case tokenChan <- TokenDelta{Token: delta.Token, Index: delta.Index}:
			}
		}

		// Forward any errors
		for err := range output.ErrorStream {
			select {
			case errChan <- err:
			default:
			}
		}

		g.logger.Debug("Streaming generation complete")
	}()

	return tokenChan, errChan, nil
}

// Close releases resources.
// Only destroys the session if it was created by this generator (not shared).
func (g *HugotGenerator) Close() error {
	if g.session != nil && !g.sessionShared {
		g.logger.Info("Destroying Hugot session (owned by this generator)")
		return g.session.Destroy()
	} else if g.sessionShared {
		g.logger.Debug("Skipping session destruction (shared session)")
	}
	return nil
}

// PooledHugotGenerator manages multiple ONNX pipelines for concurrent text generation.
// Each request acquires a pipeline slot via semaphore, enabling true parallelism.
type PooledHugotGenerator struct {
	session        *khugot.Session
	pipelines      []*pipelines.TextGenerationPipeline
	sem            *semaphore.Weighted
	nextPipeline   atomic.Uint64
	logger         *zap.Logger
	sessionShared  bool
	poolSize       int
	imageToken     string     // image placeholder token from model's special_tokens_map.json
	toolParser     ToolParser // tool call parser (from genai_config.json)
	toolCallFormat string     // the tool call format name (e.g., "functiongemma")

	// Synchronization for safe Close() behavior
	closed    atomic.Bool    // Prevents new operations after Close()
	wg        sync.WaitGroup // Waits for in-flight operations to complete
	closeOnce sync.Once      // Ensures Close() runs exactly once
	closeErr  error          // Stores error from Close()
}

// ErrGeneratorClosed is returned when Generate is called on a closed generator.
var ErrGeneratorClosed = errors.New("generator is closed")

// NewPooledHugotGenerator creates a new pooled generator using the Hugot runtime.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
// Note: For generative models, hugot uses genai_config.json to determine model files.
func NewPooledHugotGenerator(modelPath string, poolSize int, logger *zap.Logger) (*PooledHugotGenerator, error) {
	return NewPooledHugotGeneratorWithSession(modelPath, poolSize, nil, logger)
}

// NewPooledHugotGeneratorWithSession creates a new pooled generator using an optional shared Hugot session.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
// Note: For generative models, hugot uses genai_config.json to determine model files.
func NewPooledHugotGeneratorWithSession(modelPath string, poolSize int, sharedSession *khugot.Session, logger *zap.Logger) (*PooledHugotGenerator, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	// Auto-detect pool size from CPU count if not specified
	if poolSize <= 0 {
		poolSize = runtime.NumCPU()
	}

	logger.Info("Initializing pooled Hugot generator",
		zap.String("modelPath", modelPath),
		zap.Int("poolSize", poolSize),
		zap.String("backend", hugot.BackendName()))

	// Use shared session or create a new one
	session, err := hugot.NewSessionOrUseExisting(sharedSession)
	if err != nil {
		logger.Error("Failed to create Hugot session", zap.Error(err))
		return nil, fmt.Errorf("creating hugot session: %w", err)
	}
	sessionShared := (sharedSession != nil)
	if sessionShared {
		logger.Info("Using shared Hugot session", zap.String("backend", hugot.BackendName()))
	} else {
		logger.Info("Created new Hugot session", zap.String("backend", hugot.BackendName()))
	}

	// Create N streaming-enabled pipelines with unique names
	// Note: OnnxFilename is intentionally not set - generative models use genai_config.json
	pipelinesList := make([]*pipelines.TextGenerationPipeline, poolSize)
	for i := 0; i < poolSize; i++ {
		pipelineName := fmt.Sprintf("generator:%s:%d", modelPath, i)
		pipelineConfig := khugot.TextGenerationConfig{
			ModelPath: modelPath,
			Name:      pipelineName,
			Options: []backends.PipelineOption[*pipelines.TextGenerationPipeline]{
				pipelines.WithMaxLength(2048),
				pipelines.WithStreaming(),
			},
		}

		pipeline, err := khugot.NewPipeline(session, pipelineConfig)
		if err != nil {
			// Clean up already-created pipelines
			if !sessionShared {
				_ = session.Destroy()
			}
			logger.Error("Failed to create pipeline",
				zap.Int("index", i),
				zap.Error(err))
			return nil, fmt.Errorf("creating text generation pipeline %d: %w", i, err)
		}
		pipelinesList[i] = pipeline
		logger.Debug("Created streaming-enabled pipeline", zap.Int("index", i), zap.String("name", pipelineName))
	}

	logger.Info("Successfully created pooled streaming-enabled pipelines", zap.Int("count", poolSize))

	// Read image token from model's special_tokens_map.json
	imageToken := readImageToken(modelPath)
	if imageToken != "" {
		logger.Info("Loaded image token from model config", zap.String("imageToken", imageToken))
	}

	// Read genai_config.json for tool calling support
	var toolParser ToolParser
	var toolCallFormat string
	if config := readGenAIConfig(modelPath); config != nil && config.ToolCallFormat != "" {
		toolCallFormat = config.ToolCallFormat
		var err error
		toolParser, err = GetToolParser(config.ToolCallFormat, modelPath)
		if err != nil {
			logger.Warn("Failed to load tool parser",
				zap.String("format", config.ToolCallFormat),
				zap.Error(err))
		} else {
			logger.Info("Loaded tool parser from model config",
				zap.String("format", config.ToolCallFormat))
		}
	}

	return &PooledHugotGenerator{
		session:        session,
		pipelines:      pipelinesList,
		sem:            semaphore.NewWeighted(int64(poolSize)),
		logger:         logger,
		sessionShared:  sessionShared,
		poolSize:       poolSize,
		imageToken:     imageToken,
		toolParser:     toolParser,
		toolCallFormat: toolCallFormat,
	}, nil
}

// SupportsTools returns true if this generator supports tool calling.
func (p *PooledHugotGenerator) SupportsTools() bool {
	return p.toolParser != nil
}

// ToolParser returns the tool parser for this generator, or nil if not supported.
func (p *PooledHugotGenerator) ToolParser() ToolParser {
	return p.toolParser
}

// ToolCallFormat returns the tool call format name (e.g., "functiongemma").
func (p *PooledHugotGenerator) ToolCallFormat() string {
	return p.toolCallFormat
}

// convertMessages converts internal Message types to Hugot's backends.Message format
// using the generator's configured image token.
func (p *PooledHugotGenerator) convertMessages(messages []Message) []backends.Message {
	return toHugotMessages(messages, p.imageToken)
}

// Generate produces text from the given messages.
// Thread-safe: uses semaphore to limit concurrent pipeline access.
// Uses the streaming pipeline internally and collects all tokens into the response.
func (p *PooledHugotGenerator) Generate(ctx context.Context, messages []Message, opts GenerateOptions) (*GenerateResult, error) {
	// Check if closed before starting
	if p.closed.Load() {
		return nil, ErrGeneratorClosed
	}

	// Track this in-flight operation so Close() waits for us
	p.wg.Add(1)
	defer p.wg.Done()

	// Double-check after registration (handles race with Close())
	if p.closed.Load() {
		return nil, ErrGeneratorClosed
	}

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

	// Convert messages to Hugot format using the generator's image token
	hugotMessages := p.convertMessages(messages)
	output, err := pipeline.RunMessages(ctx, [][]backends.Message{hugotMessages})
	if err != nil {
		p.logger.Error("Pipeline generation failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, fmt.Errorf("running text generation: %w", err)
	}

	// Collect all tokens from the stream
	var generatedText string
	var tokenCount int
	var genErr error

	for delta := range output.TokenStream {
		generatedText += delta.Token
		tokenCount++
	}

	// Check for errors from the error stream
	for err := range output.ErrorStream {
		genErr = err
	}

	if genErr != nil {
		p.logger.Error("Generation error",
			zap.Int("pipelineIndex", idx),
			zap.Error(genErr))
		return nil, fmt.Errorf("generation error: %w", genErr)
	}

	result := &GenerateResult{
		Text:         generatedText,
		TokensUsed:   tokenCount,
		FinishReason: "stop",
	}

	p.logger.Debug("Generation complete",
		zap.Int("pipelineIndex", idx),
		zap.Int("responseLength", len(generatedText)),
		zap.Int("tokensGenerated", tokenCount),
		zap.String("finishReason", result.FinishReason))

	return result, nil
}

// GenerateStream produces tokens one at a time via channels.
// Thread-safe: uses semaphore to limit concurrent pipeline access.
func (p *PooledHugotGenerator) GenerateStream(ctx context.Context, messages []Message, opts GenerateOptions) (<-chan TokenDelta, <-chan error, error) {
	// Check if closed before starting
	if p.closed.Load() {
		return nil, nil, ErrGeneratorClosed
	}

	// Note: wg.Done() is called in the goroutine when streaming completes
	p.wg.Add(1)

	// Double-check after registration (handles race with Close())
	if p.closed.Load() {
		p.wg.Done()
		return nil, nil, ErrGeneratorClosed
	}

	if len(messages) == 0 {
		p.wg.Done()
		return nil, nil, errors.New("messages are required")
	}

	// Acquire semaphore slot (blocks if all pipelines busy)
	if err := p.sem.Acquire(ctx, 1); err != nil {
		p.wg.Done()
		return nil, nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}

	// Round-robin pipeline selection
	idx := int(p.nextPipeline.Add(1) % uint64(p.poolSize))
	pipeline := p.pipelines[idx]

	p.logger.Debug("Using pipeline for streaming generation",
		zap.Int("pipelineIndex", idx),
		zap.Int("numMessages", len(messages)))

	// Convert messages to Hugot format using the generator's image token
	hugotMessages := p.convertMessages(messages)
	output, err := pipeline.RunMessages(ctx, [][]backends.Message{hugotMessages})
	if err != nil {
		p.sem.Release(1)
		p.wg.Done()
		p.logger.Error("Streaming pipeline generation failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, nil, fmt.Errorf("running streaming text generation: %w", err)
	}

	// Adapt hugot's channels to our TokenDelta format
	tokenChan := make(chan TokenDelta)
	errChan := make(chan error, 1)

	go func() {
		defer p.wg.Done()         // Signal completion to WaitGroup
		defer p.sem.Release(1)    // Release semaphore when done streaming
		defer close(tokenChan)
		defer close(errChan)

		// Read from hugot's token stream
		for delta := range output.TokenStream {
			select {
			case <-ctx.Done():
				return
			case tokenChan <- TokenDelta{Token: delta.Token, Index: delta.Index}:
			}
		}

		// Forward any errors
		for err := range output.ErrorStream {
			select {
			case errChan <- err:
			default:
			}
		}

		p.logger.Debug("Streaming generation complete", zap.Int("pipelineIndex", idx))
	}()

	return tokenChan, errChan, nil
}

// Close releases resources.
// Only destroys the session if it was created by this generator (not shared).
// Thread-safe: waits for in-flight operations to complete before destroying.
// Safe to call multiple times (only the first call takes effect).
func (p *PooledHugotGenerator) Close() error {
	p.closeOnce.Do(func() {
		// Set closed flag to prevent new operations
		p.closed.Store(true)

		// Wait for all in-flight operations to complete
		p.wg.Wait()

		// Now safe to destroy the session
		if p.session != nil && !p.sessionShared {
			p.logger.Info("Destroying Hugot session (owned by this pooled generator)")
			p.closeErr = p.session.Destroy()
		} else if p.sessionShared {
			p.logger.Debug("Skipping session destruction (shared session)")
		}
	})
	return p.closeErr
}
