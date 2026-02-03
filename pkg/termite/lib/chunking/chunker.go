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
	"fmt"
	"runtime"
	"sync/atomic"

	"github.com/antflydb/antfly-go/libaf/chunking"
	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure PooledChunker implements the Chunker interface
var _ chunking.Chunker = (*PooledChunker)(nil)

// ChunkerConfig contains configuration for the chunker.
type ChunkerConfig struct {
	// MaxChunks is the maximum number of chunks to generate per document.
	MaxChunks int

	// Threshold is the minimum confidence threshold for separator detection.
	Threshold float32

	// TargetTokens is the target number of tokens per chunk.
	TargetTokens int
}

// DefaultChunkerConfig returns sensible defaults for the chunker.
func DefaultChunkerConfig() ChunkerConfig {
	return ChunkerConfig{
		MaxChunks:    50,
		Threshold:    0.5,
		TargetTokens: 500,
	}
}

// PooledChunkerConfig holds configuration for creating a PooledChunker.
type PooledChunkerConfig struct {
	// ModelPath is the path to the model directory
	ModelPath string

	// PoolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count)
	PoolSize int

	// ChunkerConfig contains chunking-specific settings
	ChunkerConfig ChunkerConfig

	// ModelBackends specifies which backends this model supports (nil = all backends)
	ModelBackends []string

	// Logger for logging (nil = no logging)
	Logger *zap.Logger
}

// PooledChunker manages multiple ChunkingPipeline instances for concurrent chunking.
// Uses the new backends package (go-huggingface + gomlx/onnxruntime).
type PooledChunker struct {
	pipelines    []*pipelines.ChunkingPipeline
	sem          *semaphore.Weighted
	nextPipeline atomic.Uint64
	config       ChunkerConfig
	logger       *zap.Logger
	poolSize     int
	backendType  backends.BackendType
}

// NewPooledChunker creates a new ChunkingPipeline-based pooled chunker.
// This is the new implementation using go-huggingface tokenizers and the backends package.
func NewPooledChunker(
	cfg PooledChunkerConfig,
	sessionManager *backends.SessionManager,
) (*PooledChunker, backends.BackendType, error) {
	if cfg.ModelPath == "" {
		return nil, "", fmt.Errorf("model path is required")
	}

	logger := cfg.Logger
	if logger == nil {
		logger = zap.NewNop()
	}

	// Apply defaults for zero values
	chunkerCfg := cfg.ChunkerConfig
	if chunkerCfg.MaxChunks <= 0 {
		chunkerCfg.MaxChunks = 50
	}
	if chunkerCfg.Threshold <= 0 {
		chunkerCfg.Threshold = 0.5
	}
	if chunkerCfg.TargetTokens <= 0 {
		chunkerCfg.TargetTokens = 500
	}

	// Auto-detect pool size from CPU count if not specified
	poolSize := cfg.PoolSize
	if poolSize <= 0 {
		poolSize = runtime.NumCPU()
	}

	logger.Info("Initializing pooled chunker",
		zap.String("modelPath", cfg.ModelPath),
		zap.Int("poolSize", poolSize),
		zap.Int("maxChunks", chunkerCfg.MaxChunks),
		zap.Float32("threshold", chunkerCfg.Threshold),
		zap.Int("targetTokens", chunkerCfg.TargetTokens))

	// Create N ChunkingPipelines
	pipelinesList := make([]*pipelines.ChunkingPipeline, poolSize)
	var backendUsed backends.BackendType

	for i := 0; i < poolSize; i++ {
		pipeline, bt, err := pipelines.LoadChunkingPipeline(
			cfg.ModelPath,
			sessionManager,
			cfg.ModelBackends,
			pipelines.WithChunkingThreshold(chunkerCfg.Threshold),
			pipelines.WithChunkingTargetTokens(chunkerCfg.TargetTokens),
		)
		if err != nil {
			// Clean up already-created pipelines
			for j := 0; j < i; j++ {
				if pipelinesList[j] != nil {
					_ = pipelinesList[j].Close()
				}
			}
			logger.Error("Failed to create chunking pipeline",
				zap.Int("index", i),
				zap.Error(err))
			return nil, "", fmt.Errorf("creating chunking pipeline %d: %w", i, err)
		}
		pipelinesList[i] = pipeline
		backendUsed = bt
		logger.Debug("Created chunking pipeline", zap.Int("index", i), zap.String("backend", string(bt)))
	}

	logger.Info("Successfully created pooled chunker pipelines",
		zap.Int("count", poolSize),
		zap.String("backend", string(backendUsed)))

	return &PooledChunker{
		pipelines:   pipelinesList,
		sem:         semaphore.NewWeighted(int64(poolSize)),
		config:      chunkerCfg,
		logger:      logger,
		poolSize:    poolSize,
		backendType: backendUsed,
	}, backendUsed, nil
}

// BackendType returns the backend type used by this chunker
func (p *PooledChunker) BackendType() backends.BackendType {
	return p.backendType
}

// Chunk splits text using neural token classification.
// Thread-safe: uses semaphore to limit concurrent pipeline access.
// Note: per-request options (opts) are currently ignored; pipeline uses config from creation time.
func (p *PooledChunker) Chunk(ctx context.Context, text string, opts chunking.ChunkOptions) ([]chunking.Chunk, error) {
	if text == "" {
		p.logger.Debug("Chunk called with empty text")
		return nil, nil
	}

	// Acquire semaphore slot (blocks if all pipelines busy)
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}
	defer p.sem.Release(1)

	// Round-robin pipeline selection
	idx := int(p.nextPipeline.Add(1) % uint64(p.poolSize))
	pipeline := p.pipelines[idx]

	textLen := len(text)
	textPreview := text
	if textLen > 100 {
		textPreview = text[:100] + "..."
	}

	p.logger.Debug("Starting chunking",
		zap.Int("pipelineIndex", idx),
		zap.Int("text_length", textLen),
		zap.String("text_preview", textPreview))

	// Delegate to ChunkingPipeline.Chunk
	pipelineChunks, err := pipeline.Chunk(ctx, text)
	if err != nil {
		p.logger.Error("Chunking failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, fmt.Errorf("chunking text: %w", err)
	}

	// Convert pipelines.Chunk to chunking.Chunk
	result := make([]chunking.Chunk, len(pipelineChunks))
	for i, c := range pipelineChunks {
		result[i] = chunking.Chunk{
			Id:        uint32(c.Index),
			Text:      c.Text,
			StartChar: c.Start,
			EndChar:   c.End,
		}
	}

	// Enforce max chunks limit if configured
	if p.config.MaxChunks > 0 && len(result) > p.config.MaxChunks {
		p.logger.Debug("Limiting chunks",
			zap.Int("from", len(result)),
			zap.Int("to", p.config.MaxChunks))
		result = result[:p.config.MaxChunks]
	}

	p.logger.Info("Chunking completed",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_chunks", len(result)),
		zap.Int("text_length", textLen))

	return result, nil
}

// Close releases resources.
func (p *PooledChunker) Close() error {
	var lastErr error
	for i, pipeline := range p.pipelines {
		if pipeline != nil {
			if err := pipeline.Close(); err != nil {
				p.logger.Warn("Failed to close pipeline",
					zap.Int("index", i),
					zap.Error(err))
				lastErr = err
			}
		}
	}
	p.pipelines = nil
	return lastErr
}
