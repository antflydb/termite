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

	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure PooledSparseEmbedder implements the SparseEmbedder interface
var _ embeddings.SparseEmbedder = (*PooledSparseEmbedder)(nil)

// PooledSparseEmbedder manages multiple sparse embedding pipelines for concurrent inference.
// Mirrors PooledEmbedder's semaphore + round-robin pattern.
type PooledSparseEmbedder struct {
	pipelines    []*pipelines.SparseEmbeddingPipeline
	sem          *semaphore.Weighted
	nextPipeline atomic.Uint64
	logger       *zap.Logger
	poolSize     int
	batchSize    int
	backendType  backends.BackendType
}

// PooledSparseEmbedderConfig holds configuration for creating a PooledSparseEmbedder.
type PooledSparseEmbedderConfig struct {
	ModelPath     string
	PoolSize      int
	BatchSize     int
	TopK          int
	MinWeight     float32
	ModelBackends []string
	Logger        *zap.Logger
}

// NewPooledSparseEmbedder creates a new pool of sparse embedding pipelines.
func NewPooledSparseEmbedder(
	cfg PooledSparseEmbedderConfig,
	sessionManager *backends.SessionManager,
) (*PooledSparseEmbedder, backends.BackendType, error) {
	if cfg.ModelPath == "" {
		return nil, "", fmt.Errorf("model path is required")
	}

	logger := cfg.Logger
	if logger == nil {
		logger = zap.NewNop()
	}

	poolSize := cfg.PoolSize
	if poolSize <= 0 {
		poolSize = runtime.NumCPU()
	}

	batchSize := cfg.BatchSize
	if batchSize <= 0 {
		batchSize = DefaultEmbeddingBatchSize
	}

	// Build loader options
	var opts []pipelines.SparseEmbeddingLoaderOption
	if cfg.TopK > 0 {
		opts = append(opts, pipelines.WithSparseTopK(cfg.TopK))
	}
	if cfg.MinWeight > 0 {
		opts = append(opts, pipelines.WithSparseMinWeight(cfg.MinWeight))
	}

	logger.Info("Initializing pooled sparse embedder",
		zap.String("modelPath", cfg.ModelPath),
		zap.Int("poolSize", poolSize),
		zap.Int("batchSize", batchSize))

	pipelinesList := make([]*pipelines.SparseEmbeddingPipeline, poolSize)
	var backendUsed backends.BackendType

	for i := 0; i < poolSize; i++ {
		pipeline, bt, err := pipelines.LoadSparseEmbeddingPipeline(
			cfg.ModelPath,
			sessionManager,
			cfg.ModelBackends,
			opts...,
		)
		if err != nil {
			// Clean up already-created pipelines
			for j := 0; j < i; j++ {
				_ = pipelinesList[j].Close()
			}
			logger.Error("Failed to create sparse pipeline",
				zap.Int("index", i),
				zap.Error(err))
			return nil, "", fmt.Errorf("creating sparse pipeline %d: %w", i, err)
		}
		pipelinesList[i] = pipeline
		backendUsed = bt
	}

	logger.Info("Successfully created pooled sparse embedder",
		zap.Int("poolSize", poolSize),
		zap.String("backend", string(backendUsed)))

	return &PooledSparseEmbedder{
		pipelines:   pipelinesList,
		sem:         semaphore.NewWeighted(int64(poolSize)),
		logger:      logger,
		poolSize:    poolSize,
		batchSize:   batchSize,
		backendType: backendUsed,
	}, backendUsed, nil
}

// SparseEmbed generates sparse embeddings for the given texts.
// Thread-safe: uses semaphore to limit concurrent pipeline access.
func (p *PooledSparseEmbedder) SparseEmbed(ctx context.Context, texts []string) ([]embeddings.SparseVector, error) {
	if len(texts) == 0 {
		return []embeddings.SparseVector{}, nil
	}

	// Acquire semaphore slot
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}
	defer p.sem.Release(1)

	// Round-robin pipeline selection
	idx := int(p.nextPipeline.Add(1) % uint64(p.poolSize))
	pipeline := p.pipelines[idx]

	// Process in batches
	result := make([]embeddings.SparseVector, 0, len(texts))
	batchSize := p.batchSize

	for batchStart := 0; batchStart < len(texts); batchStart += batchSize {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		batchEnd := min(batchStart+batchSize, len(texts))
		batch := texts[batchStart:batchEnd]

		batchEmbeddings, err := pipeline.Embed(ctx, batch)
		if err != nil {
			p.logger.Error("Sparse pipeline inference failed",
				zap.Int("pipelineIndex", idx),
				zap.Int("batchStart", batchStart),
				zap.Error(err))
			return nil, fmt.Errorf("running sparse embedding inference (batch %d-%d): %w", batchStart, batchEnd, err)
		}

		result = append(result, batchEmbeddings...)
	}

	return result, nil
}

// BackendType returns the backend type used by this embedder.
func (p *PooledSparseEmbedder) BackendType() backends.BackendType {
	return p.backendType
}

// Close releases resources.
func (p *PooledSparseEmbedder) Close() error {
	var lastErr error
	for i := range p.pipelines {
		if err := p.pipelines[i].Close(); err != nil {
			p.logger.Warn("Failed to close sparse pipeline",
				zap.Int("index", i),
				zap.Error(err))
			lastErr = err
		}
	}
	p.pipelines = nil
	return lastErr
}
