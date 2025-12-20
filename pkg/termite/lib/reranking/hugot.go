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
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure HugotReranker implements the Model interface
var _ reranking.Model = (*HugotReranker)(nil)
var _ reranking.Model = (*PooledHugotReranker)(nil)

// HugotReranker uses ONNX-based cross-encoder for reranking
type HugotReranker struct {
	session       *khugot.Session
	pipeline      *pipelines.CrossEncoderPipeline
	logger        *zap.Logger
	sessionShared bool // true if session is shared and shouldn't be destroyed
}

// NewHugotReranker creates a new reranker using the Hugot ONNX runtime.
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_f16.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
func NewHugotReranker(modelPath string, onnxFilename string, logger *zap.Logger) (*HugotReranker, error) {
	return NewHugotRerankerWithSession(modelPath, onnxFilename, nil, logger)
}

// NewHugotRerankerWithSession creates a new reranker using an optional shared Hugot session.
// If sharedSession is nil, a new session is created.
// If sharedSession is provided, it will be reused (important for ONNX Runtime which allows only one session).
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_f16.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
func NewHugotRerankerWithSession(modelPath string, onnxFilename string, sharedSession *khugot.Session, logger *zap.Logger) (*HugotReranker, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	// Default to model.onnx if not specified
	if onnxFilename == "" {
		onnxFilename = "model.onnx"
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Initializing Hugot reranker",
		zap.String("modelPath", modelPath),
		zap.String("onnxFilename", onnxFilename),
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

	// Create cross-encoder pipeline configuration
	// Use modelPath + onnxFilename as pipeline name to ensure uniqueness when multiple models share a session
	// This handles the case where both standard and quantized models are in the same directory
	pipelineName := fmt.Sprintf("%s:%s", modelPath, onnxFilename)
	pipelineConfig := khugot.CrossEncoderConfig{
		ModelPath:    modelPath,
		Name:         pipelineName, // Include onnxFilename to differentiate standard vs quantized
		OnnxFilename: onnxFilename,
		Options: []backends.PipelineOption[*pipelines.CrossEncoderPipeline]{
			pipelines.WithBatchSize(16),
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
		return nil, fmt.Errorf("creating cross-encoder pipeline: %w", err)
	}
	logger.Info("Successfully created cross-encoder pipeline")

	return &HugotReranker{
		session:       session,
		pipeline:      pipeline,
		logger:        logger,
		sessionShared: sessionShared,
	}, nil
}

// Rerank scores pre-rendered prompts based on relevance to the query
func (h *HugotReranker) Rerank(ctx context.Context, query string, prompts []string) ([]float32, error) {
	h.logger.Info("Rerank method ENTRY",
		zap.String("rerankerPtr", fmt.Sprintf("%p", h)),
		zap.String("pipelinePtr", fmt.Sprintf("%p", h.pipeline)),
		zap.String("sessionPtr", fmt.Sprintf("%p", h.session)),
		zap.Int("numPrompts", len(prompts)),
	)
	defer h.logger.Info("Rerank method EXIT")

	if len(prompts) == 0 {
		return []float32{}, nil
	}

	if query == "" {
		return nil, errors.New("query is required for reranking")
	}

	h.logger.Debug("Starting reranking",
		zap.String("query", query),
		zap.Int("numPrompts", len(prompts)),
	)

	// Run cross-encoder inference
	h.logger.Info("About to call pipeline.RunPipeline")
	output, err := h.pipeline.RunPipeline(query, prompts)
	h.logger.Info("pipeline.RunPipeline completed",
		zap.Bool("hasError", err != nil))
	if err != nil {
		h.logger.Error("Pipeline inference failed", zap.Error(err))
		return nil, fmt.Errorf("running cross-encoder: %w", err)
	}

	// Extract scores from output
	scores := make([]float32, len(prompts))
	for i, result := range output.Results {
		scores[i] = result.Score
	}

	h.logger.Debug("Reranking complete",
		zap.Int("numScores", len(scores)),
		zap.Float32("minScore", minScore(scores)),
		zap.Float32("maxScore", maxScore(scores)),
	)

	return scores, nil
}

// Close releases resources
// Only destroys the session if it was created by this reranker (not shared)
func (h *HugotReranker) Close() error {
	if h.session != nil && !h.sessionShared {
		h.logger.Info("Destroying Hugot session (owned by this reranker)")
		return h.session.Destroy()
	} else if h.sessionShared {
		h.logger.Debug("Skipping session destruction (shared session)")
	}
	return nil
}

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
}

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
		logger.Debug("Created pipeline", zap.Int("index", i), zap.String("name", pipelineName))
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
func (p *PooledHugotReranker) Close() error {
	if p.session != nil && !p.sessionShared {
		p.logger.Info("Destroying Hugot session (owned by this pooled reranker)")
		return p.session.Destroy()
	} else if p.sessionShared {
		p.logger.Debug("Skipping session destruction (shared session)")
	}
	return nil
}
