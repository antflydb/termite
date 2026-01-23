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

package generation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/knights-analytics/ortgenai"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure OrtgenaiGenerator implements the Generator, StreamingGenerator, and ToolSupporter interfaces
var _ Generator = (*OrtgenaiGenerator)(nil)
var _ StreamingGenerator = (*OrtgenaiGenerator)(nil)
var _ ToolSupporter = (*OrtgenaiGenerator)(nil)
var _ Generator = (*PooledOrtgenaiGenerator)(nil)
var _ StreamingGenerator = (*PooledOrtgenaiGenerator)(nil)
var _ ToolSupporter = (*PooledOrtgenaiGenerator)(nil)

func init() {
	// Auto-detect and set GenAI library path
	if genaiPath := getGenAILibraryPath(); genaiPath != "" {
		ortgenai.SetSharedLibraryPath(genaiPath)
	}
}

// getGenAILibraryPath returns the path to libonnxruntime-genai.so/.dylib.
func getGenAILibraryPath() string {
	libName := getGenAILibraryName()
	platform := runtime.GOOS + "-" + runtime.GOARCH

	// Check explicit ORTGENAI_DYLIB_PATH first
	if path := os.Getenv("ORTGENAI_DYLIB_PATH"); path != "" {
		return path
	}

	// Check ONNXRUNTIME_ROOT (GenAI libs are often installed alongside ONNX Runtime)
	if root := os.Getenv("ONNXRUNTIME_ROOT"); root != "" {
		// Try platform-specific path first
		platformPath := filepath.Join(root, platform, "lib", libName)
		if _, err := os.Stat(platformPath); err == nil {
			return platformPath
		}
		// Try direct lib path
		directPath := filepath.Join(root, "lib", libName)
		if _, err := os.Stat(directPath); err == nil {
			return directPath
		}
	}

	// Check LD_LIBRARY_PATH / DYLD_LIBRARY_PATH
	ldPath := os.Getenv("LD_LIBRARY_PATH")
	if runtime.GOOS == "darwin" {
		if dyldPath := os.Getenv("DYLD_LIBRARY_PATH"); dyldPath != "" {
			ldPath = dyldPath
		}
	}
	if ldPath != "" {
		for _, dir := range filepath.SplitList(ldPath) {
			libPath := filepath.Join(dir, libName)
			if _, err := os.Stat(libPath); err == nil {
				return libPath
			}
		}
	}

	return ""
}

// getGenAILibraryName returns the platform-specific library name.
func getGenAILibraryName() string {
	switch runtime.GOOS {
	case "windows":
		return "onnxruntime-genai.dll"
	case "darwin":
		return "libonnxruntime-genai.dylib"
	default:
		return "libonnxruntime-genai.so"
	}
}

// toOrtgenaiMessages converts internal Message types to ortgenai.Message format.
func toOrtgenaiMessages(messages []Message) []ortgenai.Message {
	ortMessages := make([]ortgenai.Message, len(messages))
	for i, m := range messages {
		// For multimodal messages, concatenate text parts
		content := m.Content
		if len(m.Parts) > 0 {
			var textParts []string
			for _, part := range m.Parts {
				if part.Type == "text" && part.Text != "" {
					textParts = append(textParts, part.Text)
				}
			}
			content = strings.Join(textParts, "")
		}

		ortMessages[i] = ortgenai.Message{
			Role:    m.Role,
			Content: content,
		}
	}
	return ortMessages
}

// hasImages checks if any message contains image parts.
func hasImages(messages []Message) bool {
	for _, m := range messages {
		if m.HasImages() {
			return true
		}
	}
	return false
}

// extractImageURLs extracts all image URLs from messages.
func extractImageURLs(messages []Message) []string {
	var urls []string
	for _, m := range messages {
		for _, part := range m.Parts {
			if part.Type == "image_url" && part.ImageURL != "" {
				urls = append(urls, part.ImageURL)
			}
		}
	}
	return urls
}

// extractPrompt extracts the text prompt from messages.
func extractPrompt(messages []Message) string {
	var prompts []string
	for _, m := range messages {
		text := m.GetTextContent()
		if text != "" {
			prompts = append(prompts, text)
		}
	}
	return strings.Join(prompts, "\n")
}

// OrtgenaiGenerator wraps an ortgenai Session for LLM inference.
type OrtgenaiGenerator struct {
	session        *ortgenai.Session
	logger         *zap.Logger
	modelPath      string
	toolParser     ToolParser
	toolCallFormat string

	// Environment initialization tracking
	envInitOnce sync.Once
	envInitErr  error
}

// NewOrtgenaiGenerator creates a new generator using the ortgenai runtime.
func NewOrtgenaiGenerator(modelPath string, logger *zap.Logger) (*OrtgenaiGenerator, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Initializing ortgenai generator",
		zap.String("modelPath", modelPath))

	// Initialize ortgenai environment
	if err := ortgenai.InitializeEnvironment(); err != nil {
		// Ignore "already initialized" errors
		if !strings.Contains(err.Error(), "already") {
			logger.Error("Failed to initialize ortgenai environment", zap.Error(err))
			return nil, fmt.Errorf("initializing ortgenai environment: %w", err)
		}
	}

	// Create generative session
	session, err := ortgenai.CreateGenerativeSession(modelPath)
	if err != nil {
		logger.Error("Failed to create ortgenai session", zap.Error(err))
		return nil, fmt.Errorf("creating ortgenai session: %w", err)
	}

	logger.Info("Successfully created ortgenai session")

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

	return &OrtgenaiGenerator{
		session:        session,
		logger:         logger,
		modelPath:      modelPath,
		toolParser:     toolParser,
		toolCallFormat: toolCallFormat,
	}, nil
}

// SupportsTools returns true if this generator supports tool calling.
func (g *OrtgenaiGenerator) SupportsTools() bool {
	return g.toolParser != nil
}

// ToolParser returns the tool parser for this generator, or nil if not supported.
func (g *OrtgenaiGenerator) ToolParser() ToolParser {
	return g.toolParser
}

// ToolCallFormat returns the tool call format name (e.g., "functiongemma").
func (g *OrtgenaiGenerator) ToolCallFormat() string {
	return g.toolCallFormat
}

// Generate produces text from the given messages.
// Supports both text-only and multimodal (image+text) generation.
func (g *OrtgenaiGenerator) Generate(ctx context.Context, messages []Message, opts GenerateOptions) (*GenerateResult, error) {
	if len(messages) == 0 {
		return nil, errors.New("messages are required")
	}

	// Configure generation options
	maxLen := opts.MaxTokens
	if maxLen <= 0 {
		maxLen = 2048
	}
	genOpts := &ortgenai.GenerationOptions{
		MaxLength: maxLen,
		BatchSize: 1,
	}

	var outputChan <-chan ortgenai.SequenceDelta
	var errChan <-chan error
	var err error

	// Multimodal cleanup functions (deferred after setup)
	var cleanup []func()
	defer func() {
		for _, fn := range cleanup {
			fn()
		}
	}()

	if hasImages(messages) {
		// Multimodal path
		imageURLs := extractImageURLs(messages)
		prompt := extractPrompt(messages)

		g.logger.Debug("Starting multimodal generation",
			zap.Int("numImages", len(imageURLs)),
			zap.Int("maxTokens", maxLen))

		images, err := ortgenai.LoadImages(imageURLs)
		if err != nil {
			return nil, fmt.Errorf("loading images: %w", err)
		}
		cleanup = append(cleanup, images.Destroy)

		processor, err := ortgenai.CreateMultiModalProcessor(g.session.GetModel())
		if err != nil {
			return nil, fmt.Errorf("creating multimodal processor: %w", err)
		}
		cleanup = append(cleanup, processor.Destroy)

		tensors, err := processor.ProcessImages(prompt, images)
		if err != nil {
			return nil, fmt.Errorf("processing images: %w", err)
		}
		cleanup = append(cleanup, tensors.Destroy)

		outputChan, errChan, err = g.session.GenerateWithTensors(ctx, tensors, genOpts)
	} else {
		// Text-only path
		g.logger.Debug("Starting text-only generation",
			zap.Int("numMessages", len(messages)),
			zap.Int("maxTokens", maxLen))

		ortMessages := toOrtgenaiMessages(messages)
		outputChan, errChan, err = g.session.Generate(ctx, [][]ortgenai.Message{ortMessages}, genOpts)
	}

	if err != nil {
		return nil, fmt.Errorf("starting generation: %w", err)
	}

	// Collect tokens
	var generatedText strings.Builder
	var tokenCount int
	for delta := range outputChan {
		generatedText.WriteString(delta.Tokens)
		tokenCount++
	}

	// Check for errors
	for err := range errChan {
		if err != nil {
			return nil, fmt.Errorf("generation error: %w", err)
		}
	}

	g.logger.Debug("Generation complete",
		zap.Int("responseLength", generatedText.Len()),
		zap.Int("tokensGenerated", tokenCount))

	return &GenerateResult{
		Text:         generatedText.String(),
		TokensUsed:   tokenCount,
		FinishReason: "stop",
	}, nil
}

// GenerateStream produces tokens one at a time via channels.
func (g *OrtgenaiGenerator) GenerateStream(ctx context.Context, messages []Message, opts GenerateOptions) (<-chan TokenDelta, <-chan error, error) {
	if len(messages) == 0 {
		return nil, nil, errors.New("messages are required")
	}

	g.logger.Debug("Starting streaming generation",
		zap.Int("numMessages", len(messages)),
		zap.Int("maxTokens", opts.MaxTokens))

	// Convert messages to ortgenai format
	ortMessages := toOrtgenaiMessages(messages)

	// Configure generation options
	genOpts := &ortgenai.GenerationOptions{
		BatchSize: 1,
	}
	if opts.MaxTokens > 0 {
		genOpts.MaxLength = opts.MaxTokens
	} else {
		genOpts.MaxLength = 2048
	}

	// Run generation
	outputChan, ortErrChan, err := g.session.Generate(ctx, [][]ortgenai.Message{ortMessages}, genOpts)
	if err != nil {
		g.logger.Error("Streaming generation failed", zap.Error(err))
		return nil, nil, fmt.Errorf("running streaming generation: %w", err)
	}

	// Adapt ortgenai's channels to our TokenDelta format
	tokenChan := make(chan TokenDelta)
	errChan := make(chan error, 1)

	go func() {
		defer close(tokenChan)
		defer close(errChan)

		// Read from ortgenai's output stream
		for delta := range outputChan {
			select {
			case <-ctx.Done():
				return
			case tokenChan <- TokenDelta{Token: delta.Tokens, Index: delta.Sequence}:
			}
		}

		// Forward any errors
		for err := range ortErrChan {
			if err != nil {
				select {
				case errChan <- err:
				default:
				}
			}
		}

		g.logger.Debug("Streaming generation complete")
	}()

	return tokenChan, errChan, nil
}

// Close releases resources.
func (g *OrtgenaiGenerator) Close() error {
	if g.session != nil {
		g.logger.Info("Destroying ortgenai session")
		g.session.Destroy()
		g.session = nil
	}
	return nil
}

// GetStatistics returns generation performance statistics.
func (g *OrtgenaiGenerator) GetStatistics() *ortgenai.Statistics {
	if g.session == nil {
		return nil
	}
	stats := g.session.GetStatistics()
	return &stats
}

// PooledOrtgenaiGenerator manages multiple ortgenai sessions for concurrent text generation.
// Each request acquires a session slot via semaphore, enabling true parallelism.
type PooledOrtgenaiGenerator struct {
	sessions       []*ortgenai.Session
	sem            *semaphore.Weighted
	nextSession    atomic.Uint64
	logger         *zap.Logger
	poolSize       int
	modelPath      string
	toolParser     ToolParser
	toolCallFormat string
}

// NewPooledOrtgenaiGenerator creates a new pooled generator using the ortgenai runtime.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
func NewPooledOrtgenaiGenerator(modelPath string, poolSize int, logger *zap.Logger) (*PooledOrtgenaiGenerator, error) {
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

	logger.Info("Initializing pooled ortgenai generator",
		zap.String("modelPath", modelPath),
		zap.Int("poolSize", poolSize))

	// Initialize ortgenai environment
	if err := ortgenai.InitializeEnvironment(); err != nil {
		// Ignore "already initialized" errors
		if !strings.Contains(err.Error(), "already") {
			logger.Error("Failed to initialize ortgenai environment", zap.Error(err))
			return nil, fmt.Errorf("initializing ortgenai environment: %w", err)
		}
	}

	// Create N sessions
	sessions := make([]*ortgenai.Session, poolSize)
	for i := 0; i < poolSize; i++ {
		session, err := ortgenai.CreateGenerativeSession(modelPath)
		if err != nil {
			// Clean up already-created sessions
			for j := 0; j < i; j++ {
				sessions[j].Destroy()
			}
			logger.Error("Failed to create ortgenai session",
				zap.Int("index", i),
				zap.Error(err))
			return nil, fmt.Errorf("creating ortgenai session %d: %w", i, err)
		}
		sessions[i] = session
		logger.Debug("Created ortgenai session", zap.Int("index", i))
	}

	logger.Info("Successfully created pooled ortgenai sessions", zap.Int("count", poolSize))

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

	return &PooledOrtgenaiGenerator{
		sessions:       sessions,
		sem:            semaphore.NewWeighted(int64(poolSize)),
		logger:         logger,
		poolSize:       poolSize,
		modelPath:      modelPath,
		toolParser:     toolParser,
		toolCallFormat: toolCallFormat,
	}, nil
}

// SupportsTools returns true if this generator supports tool calling.
func (p *PooledOrtgenaiGenerator) SupportsTools() bool {
	return p.toolParser != nil
}

// ToolParser returns the tool parser for this generator, or nil if not supported.
func (p *PooledOrtgenaiGenerator) ToolParser() ToolParser {
	return p.toolParser
}

// ToolCallFormat returns the tool call format name (e.g., "functiongemma").
func (p *PooledOrtgenaiGenerator) ToolCallFormat() string {
	return p.toolCallFormat
}

// Generate produces text from the given messages.
// Thread-safe: uses semaphore to limit concurrent session access.
// Supports both text-only and multimodal (image+text) generation.
func (p *PooledOrtgenaiGenerator) Generate(ctx context.Context, messages []Message, opts GenerateOptions) (*GenerateResult, error) {
	if len(messages) == 0 {
		return nil, errors.New("messages are required")
	}

	// Acquire semaphore slot (blocks if all sessions busy)
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("acquiring session slot: %w", err)
	}
	defer p.sem.Release(1)

	// Round-robin session selection
	idx := int(p.nextSession.Add(1) % uint64(p.poolSize))
	session := p.sessions[idx]

	// Configure generation options
	maxLen := opts.MaxTokens
	if maxLen <= 0 {
		maxLen = 2048
	}
	genOpts := &ortgenai.GenerationOptions{
		MaxLength: maxLen,
		BatchSize: 1,
	}

	var outputChan <-chan ortgenai.SequenceDelta
	var errChan <-chan error
	var err error

	// Multimodal cleanup
	var cleanup []func()
	defer func() {
		for _, fn := range cleanup {
			fn()
		}
	}()

	if hasImages(messages) {
		imageURLs := extractImageURLs(messages)
		prompt := extractPrompt(messages)

		p.logger.Debug("Starting multimodal generation",
			zap.Int("sessionIndex", idx),
			zap.Int("numImages", len(imageURLs)))

		images, err := ortgenai.LoadImages(imageURLs)
		if err != nil {
			return nil, fmt.Errorf("loading images: %w", err)
		}
		cleanup = append(cleanup, images.Destroy)

		processor, err := ortgenai.CreateMultiModalProcessor(session.GetModel())
		if err != nil {
			return nil, fmt.Errorf("creating multimodal processor: %w", err)
		}
		cleanup = append(cleanup, processor.Destroy)

		tensors, err := processor.ProcessImages(prompt, images)
		if err != nil {
			return nil, fmt.Errorf("processing images: %w", err)
		}
		cleanup = append(cleanup, tensors.Destroy)

		outputChan, errChan, err = session.GenerateWithTensors(ctx, tensors, genOpts)
	} else {
		p.logger.Debug("Starting text-only generation",
			zap.Int("sessionIndex", idx),
			zap.Int("numMessages", len(messages)))

		ortMessages := toOrtgenaiMessages(messages)
		outputChan, errChan, err = session.Generate(ctx, [][]ortgenai.Message{ortMessages}, genOpts)
	}

	if err != nil {
		return nil, fmt.Errorf("starting generation: %w", err)
	}

	// Collect tokens
	var generatedText strings.Builder
	var tokenCount int
	for delta := range outputChan {
		generatedText.WriteString(delta.Tokens)
		tokenCount++
	}

	// Check for errors
	for err := range errChan {
		if err != nil {
			return nil, fmt.Errorf("generation error: %w", err)
		}
	}

	p.logger.Debug("Generation complete",
		zap.Int("sessionIndex", idx),
		zap.Int("responseLength", generatedText.Len()),
		zap.Int("tokensGenerated", tokenCount))

	return &GenerateResult{
		Text:         generatedText.String(),
		TokensUsed:   tokenCount,
		FinishReason: "stop",
	}, nil
}

// GenerateStream produces tokens one at a time via channels.
// Thread-safe: uses semaphore to limit concurrent session access.
func (p *PooledOrtgenaiGenerator) GenerateStream(ctx context.Context, messages []Message, opts GenerateOptions) (<-chan TokenDelta, <-chan error, error) {
	if len(messages) == 0 {
		return nil, nil, errors.New("messages are required")
	}

	// Acquire semaphore slot (blocks if all sessions busy)
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, nil, fmt.Errorf("acquiring session slot: %w", err)
	}

	// Round-robin session selection
	idx := int(p.nextSession.Add(1) % uint64(p.poolSize))
	session := p.sessions[idx]

	p.logger.Debug("Using session for streaming generation",
		zap.Int("sessionIndex", idx),
		zap.Int("numMessages", len(messages)))

	// Convert messages to ortgenai format
	ortMessages := toOrtgenaiMessages(messages)

	// Configure generation options
	genOpts := &ortgenai.GenerationOptions{
		BatchSize: 1,
	}
	if opts.MaxTokens > 0 {
		genOpts.MaxLength = opts.MaxTokens
	} else {
		genOpts.MaxLength = 2048
	}

	// Run generation
	outputChan, ortErrChan, err := session.Generate(ctx, [][]ortgenai.Message{ortMessages}, genOpts)
	if err != nil {
		p.sem.Release(1)
		p.logger.Error("Streaming generation failed",
			zap.Int("sessionIndex", idx),
			zap.Error(err))
		return nil, nil, fmt.Errorf("running streaming generation: %w", err)
	}

	// Adapt ortgenai's channels to our TokenDelta format
	tokenChan := make(chan TokenDelta)
	errChan := make(chan error, 1)

	go func() {
		defer p.sem.Release(1) // Release semaphore when done streaming
		defer close(tokenChan)
		defer close(errChan)

		// Read from ortgenai's output stream
		for delta := range outputChan {
			select {
			case <-ctx.Done():
				return
			case tokenChan <- TokenDelta{Token: delta.Tokens, Index: delta.Sequence}:
			}
		}

		// Forward any errors
		for err := range ortErrChan {
			if err != nil {
				select {
				case errChan <- err:
				default:
				}
			}
		}

		p.logger.Debug("Streaming generation complete", zap.Int("sessionIndex", idx))
	}()

	return tokenChan, errChan, nil
}

// Close releases resources.
func (p *PooledOrtgenaiGenerator) Close() error {
	p.logger.Info("Destroying pooled ortgenai sessions", zap.Int("count", p.poolSize))

	for i, session := range p.sessions {
		if session != nil {
			session.Destroy()
			p.sessions[i] = nil
		}
	}

	return nil
}

// GetStatistics returns generation performance statistics from the first session.
func (p *PooledOrtgenaiGenerator) GetStatistics() *ortgenai.Statistics {
	if len(p.sessions) == 0 || p.sessions[0] == nil {
		return nil
	}
	stats := p.sessions[0].GetStatistics()
	return &stats
}

// ortgenaiGenAIConfig represents the model's genai_config.json with Termite extensions.
type ortgenaiGenAIConfig struct {
	Model struct {
		Type      string `json:"type"`
		VocabSize int    `json:"vocab_size"`
	} `json:"model"`

	// Termite extension: tool calling format
	ToolCallFormat string `json:"tool_call_format,omitempty"`
}

// readOrtgenaiConfig reads the genai_config.json file from the model directory.
func readOrtgenaiConfig(modelPath string) *ortgenaiGenAIConfig {
	configPath := filepath.Join(modelPath, "genai_config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil
	}

	var config ortgenaiGenAIConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil
	}

	return &config
}
