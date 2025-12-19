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

	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure HugotGenerator implements the Generator interface
var _ Generator = (*HugotGenerator)(nil)
var _ Generator = (*PooledHugotGenerator)(nil)

// toHugotMessages converts internal Message types to Hugot's backends.Message format.
func toHugotMessages(messages []Message) []backends.Message {
	hugotMessages := make([]backends.Message, len(messages))
	for i, m := range messages {
		hugotMessages[i] = backends.Message{
			Role:    m.Role,
			Content: m.Content,
		}
	}
	return hugotMessages
}

// HugotGenerator wraps a Hugot TextGenerationPipeline for LLM inference.
type HugotGenerator struct {
	session       *khugot.Session
	pipeline      *pipelines.TextGenerationPipeline
	logger        *zap.Logger
	sessionShared bool // true if session is shared and shouldn't be destroyed
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

	// Create text generation pipeline configuration
	// Note: OnnxFilename is intentionally not set - generative models use genai_config.json
	pipelineName := fmt.Sprintf("generator:%s", modelPath)
	pipelineConfig := khugot.TextGenerationConfig{
		ModelPath: modelPath,
		Name:      pipelineName,
		Options: []backends.PipelineOption[*pipelines.TextGenerationPipeline]{
			pipelines.WithMaxLength(2048),
		},
	}

	// Create the pipeline
	pipeline, err := khugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		// Only destroy session if we created it (not shared)
		if !sessionShared {
			_ = session.Destroy()
		}
		logger.Error("Failed to create pipeline", zap.Error(err))
		return nil, fmt.Errorf("creating text generation pipeline: %w", err)
	}
	logger.Info("Successfully created text generation pipeline")

	return &HugotGenerator{
		session:       session,
		pipeline:      pipeline,
		logger:        logger,
		sessionShared: sessionShared,
	}, nil
}

// Generate produces text from the given messages.
func (g *HugotGenerator) Generate(ctx context.Context, messages []Message, opts GenerateOptions) (*GenerateResult, error) {
	if len(messages) == 0 {
		return nil, errors.New("messages are required")
	}

	g.logger.Debug("Starting generation",
		zap.Int("numMessages", len(messages)),
		zap.Int("maxTokens", opts.MaxTokens),
	)

	// Convert messages to Hugot format and run generation
	hugotMessages := toHugotMessages(messages)
	output, err := g.pipeline.RunMessages(ctx, [][]backends.Message{hugotMessages})
	if err != nil {
		g.logger.Error("Pipeline generation failed", zap.Error(err))
		return nil, fmt.Errorf("running text generation: %w", err)
	}

	// Get the first response from the batch
	generatedText := ""
	if len(output.Responses) > 0 {
		generatedText = output.Responses[0]
	}

	result := &GenerateResult{
		Text:         generatedText,
		TokensUsed:   0, // Token count not directly available from new API
		FinishReason: "stop",
	}

	g.logger.Debug("Generation complete",
		zap.Int("responseLength", len(generatedText)),
		zap.String("finishReason", result.FinishReason),
	)

	return result, nil
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
	session       *khugot.Session
	pipelines     []*pipelines.TextGenerationPipeline
	sem           *semaphore.Weighted
	nextPipeline  atomic.Uint64
	logger        *zap.Logger
	sessionShared bool
	poolSize      int
}

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

	// Create N pipelines with unique names
	// Note: OnnxFilename is intentionally not set - generative models use genai_config.json
	pipelinesList := make([]*pipelines.TextGenerationPipeline, poolSize)
	for i := 0; i < poolSize; i++ {
		pipelineName := fmt.Sprintf("generator:%s:%d", modelPath, i)
		pipelineConfig := khugot.TextGenerationConfig{
			ModelPath: modelPath,
			Name:      pipelineName,
			Options: []backends.PipelineOption[*pipelines.TextGenerationPipeline]{
				pipelines.WithMaxLength(2048),
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
		logger.Debug("Created pipeline", zap.Int("index", i), zap.String("name", pipelineName))
	}

	logger.Info("Successfully created pooled text generation pipelines", zap.Int("count", poolSize))

	return &PooledHugotGenerator{
		session:       session,
		pipelines:     pipelinesList,
		sem:           semaphore.NewWeighted(int64(poolSize)),
		logger:        logger,
		sessionShared: sessionShared,
		poolSize:      poolSize,
	}, nil
}

// Generate produces text from the given messages.
// Thread-safe: uses semaphore to limit concurrent pipeline access.
func (p *PooledHugotGenerator) Generate(ctx context.Context, messages []Message, opts GenerateOptions) (*GenerateResult, error) {
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

	// Convert messages to Hugot format and run generation
	hugotMessages := toHugotMessages(messages)
	output, err := pipeline.RunMessages(ctx, [][]backends.Message{hugotMessages})
	if err != nil {
		p.logger.Error("Pipeline generation failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, fmt.Errorf("running text generation: %w", err)
	}

	// Get the first response from the batch
	generatedText := ""
	if len(output.Responses) > 0 {
		generatedText = output.Responses[0]
	}

	result := &GenerateResult{
		Text:         generatedText,
		TokensUsed:   0, // Token count not directly available from new API
		FinishReason: "stop",
	}

	p.logger.Debug("Generation complete",
		zap.Int("pipelineIndex", idx),
		zap.Int("responseLength", len(generatedText)),
		zap.String("finishReason", result.FinishReason))

	return result, nil
}

// Close releases resources.
// Only destroys the session if it was created by this generator (not shared).
func (p *PooledHugotGenerator) Close() error {
	if p.session != nil && !p.sessionShared {
		p.logger.Info("Destroying Hugot session (owned by this pooled generator)")
		return p.session.Destroy()
	} else if p.sessionShared {
		p.logger.Debug("Skipping session destruction (shared session)")
	}
	return nil
}
