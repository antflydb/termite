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
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// LoadGenerator loads a text generation model using the available backends.
// With ONNX support, this tries the pipeline-based approach first,
// then falls back to the GenerativeSessionFactory for generative models.
func LoadGenerator(
	modelPath string,
	poolSize int,
	logger *zap.Logger,
	sessionManager *backends.SessionManager,
	modelBackends []string,
) (Generator, backends.BackendType, error) {
	// Try the pipeline-based approach first
	cfg := &PooledPipelineGeneratorConfig{
		ModelPath: modelPath,
		PoolSize:  poolSize,
		Logger:    logger,
	}
	generator, backendType, err := NewPooledPipelineGenerator(cfg, sessionManager, modelBackends)
	if err == nil {
		return generator, backendType, nil
	}

	// Check if we should fall back to GenerativeSessionFactory for generative models.
	// Fall back when:
	// 1. Session factory not supported (encoder model factory can't handle generative models)
	// 2. Model has genai_config.json (ortgenai-native format, like FunctionGemma from registry)
	hasGenaiConfig := false
	if _, statErr := os.Stat(filepath.Join(modelPath, "genai_config.json")); statErr == nil {
		hasGenaiConfig = true
	}
	shouldFallback := strings.Contains(err.Error(), "session factory") || hasGenaiConfig
	if shouldFallback {
		logger.Debug("Pipeline approach failed, falling back to GenerativeSessionFactory",
			zap.String("modelPath", modelPath),
			zap.Error(err))

		genFactory, bt, factoryErr := sessionManager.GetGenerativeSessionFactoryForModel(modelBackends)
		if factoryErr != nil {
			// Return the original error since it's likely more informative
			return nil, "", err
		}

		genGenerator, genErr := NewPooledGenerativeSessionGenerator(modelPath, poolSize, genFactory, logger)
		if genErr != nil {
			// Return the original error
			return nil, "", err
		}

		logger.Info("Loaded generator using GenerativeSessionFactory",
			zap.String("modelPath", modelPath),
			zap.String("backend", string(bt)))

		return genGenerator, bt, nil
	}

	// Return the original error for other failure cases
	return nil, "", err
}

// Ensure PooledGenerativeSessionGenerator implements the interfaces
var _ Generator = (*PooledGenerativeSessionGenerator)(nil)
var _ StreamingGenerator = (*PooledGenerativeSessionGenerator)(nil)

// PooledGenerativeSessionGenerator wraps multiple GenerativeSessions for concurrent generation.
// It adapts the backends.GenerativeSession interface to the generation.Generator interface.
type PooledGenerativeSessionGenerator struct {
	sessions       []backends.GenerativeSession
	sem            *semaphore.Weighted
	nextSession    atomic.Uint64
	logger         *zap.Logger
	poolSize       int
	modelPath      string
	toolParser     ToolParser
	toolCallFormat string
}

// NewPooledGenerativeSessionGenerator creates a new pooled generator using GenerativeSessionFactory.
func NewPooledGenerativeSessionGenerator(
	modelPath string,
	poolSize int,
	factory backends.GenerativeSessionFactory,
	logger *zap.Logger,
) (*PooledGenerativeSessionGenerator, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	if poolSize <= 0 {
		poolSize = min(runtime.NumCPU(), 4)
	}

	logger.Info("Initializing pooled generative session generator",
		zap.String("modelPath", modelPath),
		zap.Int("poolSize", poolSize))

	// Create N sessions
	sessions := make([]backends.GenerativeSession, poolSize)
	for i := 0; i < poolSize; i++ {
		session, err := factory.CreateGenerativeSession(modelPath)
		if err != nil {
			// Clean up already-created sessions
			for j := 0; j < i; j++ {
				sessions[j].Close()
			}
			return nil, err
		}
		sessions[i] = session
		logger.Debug("Created generative session", zap.Int("index", i))
	}

	logger.Info("Successfully created pooled generative sessions", zap.Int("count", poolSize))

	// Try to load tool parser from genai_config.json
	var toolParser ToolParser
	var toolCallFormat string
	if format := readToolCallFormat(modelPath); format != "" {
		if parser, err := GetToolParser(format, modelPath); err == nil && parser != nil {
			toolParser = parser
			toolCallFormat = format
			logger.Info("Loaded tool parser from model config",
				zap.String("format", toolCallFormat))
		} else if err != nil {
			logger.Warn("Failed to load tool parser",
				zap.String("format", format),
				zap.Error(err))
		}
	}

	return &PooledGenerativeSessionGenerator{
		sessions:       sessions,
		sem:            semaphore.NewWeighted(int64(poolSize)),
		logger:         logger,
		poolSize:       poolSize,
		modelPath:      modelPath,
		toolParser:     toolParser,
		toolCallFormat: toolCallFormat,
	}, nil
}

// Generate produces text from the given messages.
func (p *PooledGenerativeSessionGenerator) Generate(ctx context.Context, messages []Message, opts GenerateOptions) (*GenerateResult, error) {
	// Acquire semaphore slot
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, err
	}
	defer p.sem.Release(1)

	// Round-robin session selection
	idx := int(p.nextSession.Add(1) % uint64(p.poolSize))
	session := p.sessions[idx]

	// Convert messages to backend format
	backendMsgs := toBackendMessages(messages)
	backendOpts := toBackendOptions(opts)

	// Generate
	result, err := session.Generate(ctx, backendMsgs, backendOpts)
	if err != nil {
		return nil, err
	}

	return &GenerateResult{
		Text:         result.Text,
		TokensUsed:   result.TokensUsed,
		FinishReason: result.FinishReason,
	}, nil
}

// GenerateStream produces tokens one at a time via channels.
func (p *PooledGenerativeSessionGenerator) GenerateStream(ctx context.Context, messages []Message, opts GenerateOptions) (<-chan TokenDelta, <-chan error, error) {
	// Acquire semaphore slot
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, nil, err
	}

	// Round-robin session selection
	idx := int(p.nextSession.Add(1) % uint64(p.poolSize))
	session := p.sessions[idx]

	// Convert messages to backend format
	backendMsgs := toBackendMessages(messages)
	backendOpts := toBackendOptions(opts)

	// Start streaming
	backendTokenChan, backendErrChan, err := session.GenerateStream(ctx, backendMsgs, backendOpts)
	if err != nil {
		p.sem.Release(1)
		return nil, nil, err
	}

	// Adapt backend channels to generation channels
	tokenChan := make(chan TokenDelta)
	errChan := make(chan error, 1)

	go func() {
		defer p.sem.Release(1)
		defer close(tokenChan)
		defer close(errChan)

		for token := range backendTokenChan {
			select {
			case <-ctx.Done():
				return
			case tokenChan <- TokenDelta{Token: token.Token, Index: token.Index}:
			}
		}

		for err := range backendErrChan {
			if err != nil {
				select {
				case errChan <- err:
				default:
				}
			}
		}
	}()

	return tokenChan, errChan, nil
}

// SupportsTools returns true if this generator supports tool calling.
func (p *PooledGenerativeSessionGenerator) SupportsTools() bool {
	return p.toolParser != nil
}

// ToolParser returns the tool parser for this generator.
func (p *PooledGenerativeSessionGenerator) ToolParser() ToolParser {
	return p.toolParser
}

// ToolCallFormat returns the tool call format name.
func (p *PooledGenerativeSessionGenerator) ToolCallFormat() string {
	return p.toolCallFormat
}

// Close releases resources.
func (p *PooledGenerativeSessionGenerator) Close() error {
	p.logger.Info("Closing pooled generative session generator", zap.Int("poolSize", p.poolSize))

	for _, session := range p.sessions {
		if session != nil {
			session.Close()
		}
	}
	return nil
}

// toBackendMessages converts generation.Message to backends.GenerativeMessage.
func toBackendMessages(messages []Message) []backends.GenerativeMessage {
	result := make([]backends.GenerativeMessage, len(messages))
	for i, m := range messages {
		// Extract text content
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

		// Extract image URLs
		var imageURLs []string
		for _, part := range m.Parts {
			if part.Type == "image_url" && part.ImageURL != "" {
				imageURLs = append(imageURLs, part.ImageURL)
			}
		}

		result[i] = backends.GenerativeMessage{
			Role:      m.Role,
			Content:   content,
			ImageURLs: imageURLs,
		}
	}
	return result
}

// toBackendOptions converts generation.GenerateOptions to backends.GenerativeOptions.
func toBackendOptions(opts GenerateOptions) *backends.GenerativeOptions {
	return &backends.GenerativeOptions{
		MaxTokens:   opts.MaxTokens,
		Temperature: opts.Temperature,
		TopP:        opts.TopP,
		TopK:        opts.TopK,
		StopTokens:  opts.StopTokens,
	}
}

// genaiConfigToolFormat holds the tool_call_format from genai_config.json.
type genaiConfigToolFormat struct {
	ToolCallFormat string `json:"tool_call_format"`
}

// readToolCallFormat reads the tool_call_format from genai_config.json.
// Returns empty string if the file doesn't exist or doesn't have tool_call_format.
func readToolCallFormat(modelPath string) string {
	configPath := filepath.Join(modelPath, "genai_config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}

	var config genaiConfigToolFormat
	if err := json.Unmarshal(data, &config); err != nil {
		return ""
	}

	return config.ToolCallFormat
}
