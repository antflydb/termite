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
	"sync"
	"sync/atomic"

	"github.com/antflydb/antfly-go/libaf/reranking"
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure PooledHugotReranker implements the Model interface
var _ reranking.Model = (*PooledHugotReranker)(nil)

// Helper functions
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

// PooledHugotReranker manages multiple ONNX pipelines for concurrent reranking.
// Each request acquires a pipeline slot via semaphore, enabling true parallelism.
type PooledHugotReranker struct {
	session       *khugot.Session
	pipelines     []*pipelines.CrossEncoderPipeline
	sem           *semaphore.Weighted
	nextPipeline  atomic.Uint64
	logger        *zap.Logger
	sessionShared bool
	poolSize      int

	// Synchronization for safe Close() behavior
	closed    atomic.Bool    // Prevents new operations after Close()
	wg        sync.WaitGroup // Waits for in-flight operations to complete
	closeOnce sync.Once      // Ensures Close() runs exactly once
	closeErr  error          // Stores error from Close()
}

// ErrRerankerClosed is returned when Rerank is called on a closed reranker.
var ErrRerankerClosed = errors.New("reranker is closed")

// NewPooledHugotReranker creates a new pooled reranker using the Hugot ONNX runtime.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_f16.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
func NewPooledHugotReranker(modelPath string, onnxFilename string, poolSize int, logger *zap.Logger) (*PooledHugotReranker, error) {
	return NewPooledHugotRerankerWithSession(modelPath, onnxFilename, poolSize, nil, logger)
}

// NewPooledHugotRerankerWithSession creates a new pooled reranker using an optional shared Hugot session.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_f16.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
func NewPooledHugotRerankerWithSession(modelPath string, onnxFilename string, poolSize int, sharedSession *khugot.Session, logger *zap.Logger) (*PooledHugotReranker, error) {
	reranker, _, err := newPooledHugotRerankerInternal(modelPath, onnxFilename, poolSize, sharedSession, nil, nil, logger)
	return reranker, err
}

// NewPooledHugotRerankerWithSessionManager creates a new pooled reranker using a SessionManager.
// The SessionManager handles backend selection based on priority and model compatibility.
// Returns the reranker, the backend type that was used, and any error.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_f16.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
// modelBackends specifies which backends this model supports (nil = all backends).
func NewPooledHugotRerankerWithSessionManager(modelPath string, onnxFilename string, poolSize int, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*PooledHugotReranker, hugot.BackendType, error) {
	return newPooledHugotRerankerInternal(modelPath, onnxFilename, poolSize, nil, sessionManager, modelBackends, logger)
}

// newPooledHugotRerankerInternal is the shared implementation for creating pooled rerankers.
// Either sharedSession or sessionManager should be provided, not both.
func newPooledHugotRerankerInternal(modelPath string, onnxFilename string, poolSize int, sharedSession *khugot.Session, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*PooledHugotReranker, hugot.BackendType, error) {
	if modelPath == "" {
		return nil, hugot.BackendGo, errors.New("model path is required")
	}

	// Default to model.onnx if not specified
	if onnxFilename == "" {
		onnxFilename = "model.onnx"
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	// Auto-detect pool size from CPU count if not specified
	if poolSize <= 0 {
		poolSize = runtime.NumCPU()
	}

	logger.Info("Initializing pooled Hugot reranker",
		zap.String("modelPath", modelPath),
		zap.String("onnxFilename", onnxFilename),
		zap.Int("poolSize", poolSize),
		zap.String("backend", hugot.BackendName()))

	// Get session from SessionManager, shared session, or create new one
	var session *khugot.Session
	var backendUsed hugot.BackendType
	var err error
	var sessionShared bool

	if sessionManager != nil {
		// Use SessionManager for backend selection
		session, backendUsed, err = sessionManager.GetSessionForModel(modelBackends)
		if err != nil {
			logger.Error("Failed to get session from SessionManager", zap.Error(err))
			return nil, backendUsed, fmt.Errorf("getting session from SessionManager: %w", err)
		}
		sessionShared = true // SessionManager owns the session
		logger.Info("Using session from SessionManager",
			zap.String("backend", string(backendUsed)))
	} else if sharedSession != nil {
		// Use provided shared session
		session = sharedSession
		sessionShared = true
		backendUsed = hugot.GetDefaultBackend().Type()
		logger.Info("Using shared Hugot session", zap.String("backend", hugot.BackendName()))
	} else {
		// Create new session
		session, err = hugot.NewSession()
		if err != nil {
			logger.Error("Failed to create Hugot session", zap.Error(err))
			return nil, hugot.BackendGo, fmt.Errorf("creating hugot session: %w", err)
		}
		sessionShared = false
		backendUsed = hugot.GetDefaultBackend().Type()
		logger.Info("Created new Hugot session", zap.String("backend", hugot.BackendName()))
	}

	// Create N pipelines with unique names
	pipelinesList := make([]*pipelines.CrossEncoderPipeline, poolSize)
	for i := 0; i < poolSize; i++ {
		pipelineName := fmt.Sprintf("%s:%s:%d", modelPath, onnxFilename, i)
		pipelineConfig := khugot.CrossEncoderConfig{
			ModelPath:    modelPath,
			Name:         pipelineName,
			OnnxFilename: onnxFilename,
			Options: []backends.PipelineOption[*pipelines.CrossEncoderPipeline]{
				pipelines.WithBatchSize(16),
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
			return nil, backendUsed, fmt.Errorf("creating cross-encoder pipeline %d: %w", i, err)
		}
		pipelinesList[i] = pipeline
	}

	logger.Info("Successfully created pooled cross-encoder pipelines", zap.Int("count", poolSize))

	return &PooledHugotReranker{
		session:       session,
		pipelines:     pipelinesList,
		sem:           semaphore.NewWeighted(int64(poolSize)),
		logger:        logger,
		sessionShared: sessionShared,
		poolSize:      poolSize,
	}, backendUsed, nil
}

// Rerank scores pre-rendered prompts based on relevance to the query.
// Thread-safe: uses semaphore to limit concurrent pipeline access.
func (p *PooledHugotReranker) Rerank(ctx context.Context, query string, prompts []string) ([]float32, error) {
	// Check if closed before starting
	if p.closed.Load() {
		return nil, ErrRerankerClosed
	}

	// Track this in-flight operation so Close() waits for us
	p.wg.Add(1)
	defer p.wg.Done()

	// Double-check after registration (handles race with Close())
	if p.closed.Load() {
		return nil, ErrRerankerClosed
	}

	if len(prompts) == 0 {
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
		zap.Int("numPrompts", len(prompts)))

	// Run cross-encoder inference
	output, err := pipeline.RunPipeline(query, prompts)
	if err != nil {
		p.logger.Error("Pipeline inference failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, fmt.Errorf("running cross-encoder: %w", err)
	}

	// Extract scores from output
	scores := make([]float32, len(prompts))
	for i, result := range output.Results {
		scores[i] = result.Score
	}

	p.logger.Debug("Reranking complete",
		zap.Int("pipelineIndex", idx),
		zap.Int("numScores", len(scores)),
		zap.Float32("minScore", minScore(scores)),
		zap.Float32("maxScore", maxScore(scores)))

	return scores, nil
}

// Close releases resources.
// Only destroys the session if it was created by this reranker (not shared).
// Thread-safe: waits for in-flight operations to complete before destroying.
// Safe to call multiple times (only the first call takes effect).
func (p *PooledHugotReranker) Close() error {
	p.closeOnce.Do(func() {
		// Set closed flag to prevent new operations
		p.closed.Store(true)

		// Wait for all in-flight operations to complete
		p.wg.Wait()

		// Now safe to destroy the session
		if p.session != nil && !p.sessionShared {
			p.logger.Info("Destroying Hugot session (owned by this pooled reranker)")
			p.closeErr = p.session.Destroy()
		} else if p.sessionShared {
			p.logger.Debug("Skipping session destruction (shared session)")
		}
	})
	return p.closeErr
}
