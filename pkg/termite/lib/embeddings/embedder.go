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

package embeddings

import (
	"context"
	"fmt"
	"runtime"
	"sync/atomic"

	"github.com/antflydb/antfly-go/libaf/ai"
	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure PooledEmbedder implements the Embedder interface
var _ embeddings.Embedder = (*PooledEmbedder)(nil)

// DefaultEmbeddingBatchSize is the default batch size for embedding inference.
//
// The ONNX Runtime CoreML Execution Provider cannot handle batch sizes > 1 for embedding models.
// This is specific to the CoreML EP bridge layer - pure ONNX Runtime CPU handles batching fine.
// With CoreML, any batch size > 1 causes: "Error executing model: Unable to compute the prediction
// using a neural network model (error code: -1)".
//
// See batch_test.go for validation of this limitation.
const DefaultEmbeddingBatchSize = 1

// PooledEmbedder manages multiple EmbeddingPipeline instances for concurrent embedding generation.
// Uses the new backends package (go-huggingface + gomlx/onnxruntime) instead of hugot.
type PooledEmbedder struct {
	pipelines    []*pipelines.EmbeddingPipeline
	sem          *semaphore.Weighted
	nextPipeline atomic.Uint64
	logger       *zap.Logger
	poolSize     int
	caps         embeddings.EmbedderCapabilities
	batchSize    int
	backendType  backends.BackendType
}

// PooledEmbedderConfig holds configuration for creating a PooledEmbedder.
type PooledEmbedderConfig struct {
	// ModelPath is the path to the model directory
	ModelPath string

	// PoolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count)
	PoolSize int

	// BatchSize is the inference batch size (0 = use default)
	BatchSize int

	// Normalize enables L2 normalization of embeddings
	Normalize bool

	// Pooling specifies the pooling strategy ("mean", "cls", "max")
	Pooling backends.PoolingStrategy

	// ModelBackends specifies which backends this model supports (nil = all backends)
	ModelBackends []string

	// Logger for logging (nil = no logging)
	Logger *zap.Logger
}

// NewPooledEmbedder creates a new EmbeddingPipeline-based pooled embedder.
// This is the new implementation using go-huggingface tokenizers and the backends package.
func NewPooledEmbedder(
	cfg PooledEmbedderConfig,
	sessionManager *backends.SessionManager,
) (*PooledEmbedder, backends.BackendType, error) {
	if cfg.ModelPath == "" {
		return nil, "", fmt.Errorf("model path is required")
	}

	logger := cfg.Logger
	if logger == nil {
		logger = zap.NewNop()
	}

	// Auto-detect pool size from CPU count if not specified
	poolSize := cfg.PoolSize
	if poolSize <= 0 {
		poolSize = runtime.NumCPU()
	}

	// Default batch size
	batchSize := cfg.BatchSize
	if batchSize <= 0 {
		batchSize = DefaultEmbeddingBatchSize
	}

	// Default pooling strategy
	pooling := cfg.Pooling
	if pooling == "" {
		pooling = backends.PoolingMean
	}

	logger.Info("Initializing pooled embedder",
		zap.String("modelPath", cfg.ModelPath),
		zap.Int("poolSize", poolSize),
		zap.Int("batchSize", batchSize))

	// Create N pipelines
	pipelinesList := make([]*pipelines.EmbeddingPipeline, poolSize)
	var backendUsed backends.BackendType

	for i := 0; i < poolSize; i++ {
		// LoadEmbeddingPipelines returns text and visual pipelines; we only need text
		textPipeline, _, bt, err := pipelines.LoadEmbeddingPipelines(
			cfg.ModelPath,
			sessionManager,
			cfg.ModelBackends,
			pipelines.WithEmbeddingNormalization(cfg.Normalize),
			pipelines.WithPoolingStrategy(pooling),
		)
		if err != nil {
			// Clean up already-created pipelines
			for j := 0; j < i; j++ {
				if pipelinesList[j] != nil {
					_ = pipelinesList[j].Close()
				}
			}
			logger.Error("Failed to create pipeline",
				zap.Int("index", i),
				zap.Error(err))
			return nil, "", fmt.Errorf("creating pipeline %d: %w", i, err)
		}
		if textPipeline == nil {
			// Clean up already-created pipelines
			for j := 0; j < i; j++ {
				if pipelinesList[j] != nil {
					_ = pipelinesList[j].Close()
				}
			}
			return nil, "", fmt.Errorf("model at %s does not have a text encoder", cfg.ModelPath)
		}
		pipelinesList[i] = textPipeline
		backendUsed = bt
		logger.Debug("Created pipeline", zap.Int("index", i), zap.String("backend", string(bt)))
	}

	logger.Info("Successfully created pooled pipelines",
		zap.Int("count", poolSize),
		zap.String("backend", string(backendUsed)))

	return &PooledEmbedder{
		pipelines:   pipelinesList,
		sem:         semaphore.NewWeighted(int64(poolSize)),
		logger:      logger,
		poolSize:    poolSize,
		caps:        embeddings.TextOnlyCapabilities(),
		batchSize:   batchSize,
		backendType: backendUsed,
	}, backendUsed, nil
}

// Capabilities returns the capabilities of this embedder
func (p *PooledEmbedder) Capabilities() embeddings.EmbedderCapabilities {
	return p.caps
}

// BackendType returns the backend type used by this embedder
func (p *PooledEmbedder) BackendType() backends.BackendType {
	return p.backendType
}

// Embed generates embeddings for the given content.
// Thread-safe: uses semaphore to limit concurrent pipeline access.
func (p *PooledEmbedder) Embed(ctx context.Context, contents [][]ai.ContentPart) ([][]float32, error) {
	if len(contents) == 0 {
		return [][]float32{}, nil
	}

	// Acquire semaphore slot (blocks if all pipelines busy)
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}
	defer p.sem.Release(1)

	// Round-robin pipeline selection
	idx := int(p.nextPipeline.Add(1) % uint64(p.poolSize))
	pipeline := p.pipelines[idx]

	// Extract text from content parts
	texts := embeddings.ExtractText(contents)

	// Process in batches
	result := make([][]float32, 0, len(texts))
	batchSize := p.batchSize

	numBatches := (len(texts) + batchSize - 1) / batchSize
	p.logger.Debug("Processing embeddings in batches",
		zap.Int("pipelineIndex", idx),
		zap.Int("numTexts", len(texts)),
		zap.Int("batchSize", batchSize),
		zap.Int("numBatches", numBatches))

	for batchStart := 0; batchStart < len(texts); batchStart += batchSize {
		// Check context cancellation between batches
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		batchEnd := batchStart + batchSize
		if batchEnd > len(texts) {
			batchEnd = len(texts)
		}
		batch := texts[batchStart:batchEnd]

		// Run inference using EmbeddingPipeline.Embed which handles tokenization,
		// pooling, and normalization internally
		batchEmbeddings, err := pipeline.Embed(ctx, batch)
		if err != nil {
			p.logger.Error("Pipeline inference failed",
				zap.Int("pipelineIndex", idx),
				zap.Int("batchStart", batchStart),
				zap.Int("batchSize", len(batch)),
				zap.Error(err))
			return nil, fmt.Errorf("running embedding inference (batch %d-%d): %w", batchStart, batchEnd, err)
		}

		// Validate embeddings
		for i, embedding := range batchEmbeddings {
			if len(embedding) == 0 {
				p.logger.Error("Empty embedding returned",
					zap.Int("pipelineIndex", idx),
					zap.Int("index", batchStart+i))
				return nil, fmt.Errorf("empty embedding at index %d", batchStart+i)
			}
		}

		result = append(result, batchEmbeddings...)
	}

	p.logger.Debug("Embedding generation complete",
		zap.Int("pipelineIndex", idx),
		zap.Int("numEmbeddings", len(result)))

	return result, nil
}

// Close releases resources.
func (p *PooledEmbedder) Close() error {
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
