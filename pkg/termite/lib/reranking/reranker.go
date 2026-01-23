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

package reranking

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync/atomic"

	"github.com/antflydb/antfly-go/libaf/reranking"
	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure PooledReranker implements the Model interface
var _ reranking.Model = (*PooledReranker)(nil)

// PooledReranker manages multiple RerankingPipeline instances for concurrent reranking.
// Uses the new backends package (go-huggingface + gomlx/onnxruntime) instead of hugot.
type PooledReranker struct {
	pipelines    []*pipelines.RerankingPipeline
	sem          *semaphore.Weighted
	nextPipeline atomic.Uint64
	logger       *zap.Logger
	poolSize     int
	backendType  backends.BackendType
}

// PooledRerankerConfig holds configuration for creating a PooledReranker.
type PooledRerankerConfig struct {
	// ModelPath is the path to the model directory
	ModelPath string

	// PoolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count)
	PoolSize int

	// MaxLength is the maximum sequence length for tokenization (0 = use default 512)
	MaxLength int

	// ModelBackends specifies which backends this model supports (nil = all backends)
	ModelBackends []string

	// Logger for logging (nil = no logging)
	Logger *zap.Logger
}

// NewPooledReranker creates a new RerankingPipeline-based pooled reranker.
// This is the new implementation using go-huggingface tokenizers and the backends package.
func NewPooledReranker(
	cfg PooledRerankerConfig,
	sessionManager *backends.SessionManager,
) (*PooledReranker, backends.BackendType, error) {
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

	logger.Info("Initializing pooled reranker",
		zap.String("modelPath", cfg.ModelPath),
		zap.Int("poolSize", poolSize))

	// Build loader options
	var loaderOpts []pipelines.RerankingLoaderOption
	if cfg.MaxLength > 0 {
		loaderOpts = append(loaderOpts, pipelines.WithRerankingMaxLength(cfg.MaxLength))
	}

	// Create N pipelines
	pipelinesList := make([]*pipelines.RerankingPipeline, poolSize)
	var backendUsed backends.BackendType

	for i := 0; i < poolSize; i++ {
		pipeline, bt, err := pipelines.LoadRerankingPipeline(
			cfg.ModelPath,
			sessionManager,
			cfg.ModelBackends,
			loaderOpts...,
		)
		if err != nil {
			// Clean up already-created pipelines
			for j := 0; j < i; j++ {
				if pipelinesList[j] != nil {
					_ = pipelinesList[j].Close()
				}
			}
			logger.Error("Failed to create reranking pipeline",
				zap.Int("index", i),
				zap.Error(err))
			return nil, "", fmt.Errorf("creating reranking pipeline %d: %w", i, err)
		}
		pipelinesList[i] = pipeline
		backendUsed = bt
		logger.Debug("Created reranking pipeline", zap.Int("index", i), zap.String("backend", string(bt)))
	}

	logger.Info("Successfully created pooled reranker pipelines",
		zap.Int("count", poolSize),
		zap.String("backend", string(backendUsed)))

	return &PooledReranker{
		pipelines:   pipelinesList,
		sem:         semaphore.NewWeighted(int64(poolSize)),
		logger:      logger,
		poolSize:    poolSize,
		backendType: backendUsed,
	}, backendUsed, nil
}

// BackendType returns the backend type used by this reranker
func (p *PooledReranker) BackendType() backends.BackendType {
	return p.backendType
}

// Rerank scores documents based on relevance to the query.
// Thread-safe: uses semaphore to limit concurrent pipeline access.
// Returns one score per document, higher scores indicate more relevance.
func (p *PooledReranker) Rerank(ctx context.Context, query string, documents []string) ([]float32, error) {
	if len(documents) == 0 {
		return []float32{}, nil
	}

	if query == "" {
		return nil, errors.New("query is required for reranking")
	}

	// Acquire semaphore slot (blocks if all pipelines busy)
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}
	defer p.sem.Release(1)

	// Round-robin pipeline selection
	idx := int(p.nextPipeline.Add(1) % uint64(p.poolSize))
	pipeline := p.pipelines[idx]

	p.logger.Debug("Using pipeline for reranking",
		zap.Int("pipelineIndex", idx),
		zap.Int("numDocuments", len(documents)))

	// Delegate to the RerankingPipeline which handles tokenization, inference, and score extraction
	scores, err := pipeline.Rerank(ctx, query, documents)
	if err != nil {
		p.logger.Error("Reranking pipeline failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, fmt.Errorf("running reranking pipeline: %w", err)
	}

	p.logger.Debug("Reranking complete",
		zap.Int("pipelineIndex", idx),
		zap.Int("numScores", len(scores)),
		zap.Float32("minScore", minScore(scores)),
		zap.Float32("maxScore", maxScore(scores)))

	return scores, nil
}

// Close releases resources.
func (p *PooledReranker) Close() error {
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

// minScore returns the minimum score from a slice of scores.
func minScore(scores []float32) float32 {
	if len(scores) == 0 {
		return 0
	}
	min := scores[0]
	for _, s := range scores[1:] {
		if s < min {
			min = s
		}
	}
	return min
}

// maxScore returns the maximum score from a slice of scores.
func maxScore(scores []float32) float32 {
	if len(scores) == 0 {
		return 0
	}
	max := scores[0]
	for _, s := range scores[1:] {
		if s > max {
			max = s
		}
	}
	return max
}
