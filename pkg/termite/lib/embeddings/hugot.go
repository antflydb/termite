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
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync/atomic"

	"github.com/antflydb/antfly-go/libaf/ai"
	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure HugotEmbedder implements the Embedder interface
var _ embeddings.Embedder = (*HugotEmbedder)(nil)
var _ embeddings.Embedder = (*PooledHugotEmbedder)(nil)

// HugotEmbedder uses ONNX-based feature extraction for embeddings
type HugotEmbedder struct {
	session       *khugot.Session
	pipeline      *pipelines.FeatureExtractionPipeline
	logger        *zap.Logger
	sessionShared bool // true if session is shared and shouldn't be destroyed
	caps          embeddings.EmbedderCapabilities
}

// NewHugotEmbedder creates a new embedder using the Hugot ONNX runtime.
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_f16.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
func NewHugotEmbedder(modelPath string, onnxFilename string, logger *zap.Logger) (*HugotEmbedder, error) {
	return NewHugotEmbedderWithSession(modelPath, onnxFilename, nil, logger)
}

// NewHugotEmbedderWithSession creates a new embedder using an optional shared Hugot session.
// If sharedSession is nil, a new session is created.
// If sharedSession is provided, it will be reused (important for ONNX Runtime which allows only one session).
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_f16.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
func NewHugotEmbedderWithSession(modelPath string, onnxFilename string, sharedSession *khugot.Session, logger *zap.Logger) (*HugotEmbedder, error) {
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

	logger.Info("Initializing Hugot embedder",
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

	// Create feature extraction pipeline configuration
	// Use modelPath + onnxFilename as pipeline name to ensure uniqueness when multiple models share a session
	// This handles the case where both standard and quantized models are in the same directory
	pipelineName := fmt.Sprintf("%s:%s", modelPath, onnxFilename)
	pipelineConfig := khugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         pipelineName, // Include onnxFilename to differentiate standard vs quantized
		OnnxFilename: onnxFilename,
		Options: []backends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
			pipelines.WithNormalization(),
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
		return nil, fmt.Errorf("creating feature extraction pipeline: %w", err)
	}
	logger.Info("Successfully created feature extraction pipeline")

	return &HugotEmbedder{
		session:       session,
		pipeline:      pipeline,
		logger:        logger,
		sessionShared: sessionShared,
		caps:          embeddings.TextOnlyCapabilities(), // ONNX embedders are typically text-only
	}, nil
}

// Capabilities returns the capabilities of this embedder
func (h *HugotEmbedder) Capabilities() embeddings.EmbedderCapabilities {
	return h.caps
}

// Embed generates embeddings for the given content
// Extracts [CLS] token embeddings and normalizes them
func (h *HugotEmbedder) Embed(ctx context.Context, contents [][]ai.ContentPart) ([][]float32, error) {
	h.logger.Debug("Embed method ENTRY",
		zap.String("embedderPtr", fmt.Sprintf("%p", h)),
		zap.String("pipelinePtr", fmt.Sprintf("%p", h.pipeline)),
		zap.String("sessionPtr", fmt.Sprintf("%p", h.session)),
		zap.Int("numContents", len(contents)),
	)
	defer h.logger.Debug("Embed method EXIT")

	if len(contents) == 0 {
		return [][]float32{}, nil
	}

	// Hugot only supports text embeddings
	values := embeddings.ExtractText(contents)

	h.logger.Debug("Starting embedding generation",
		zap.Int("numTexts", len(values)),
	)

	// Run feature extraction inference
	h.logger.Debug("About to call pipeline.RunPipeline")
	output, err := h.pipeline.RunPipeline(values)
	h.logger.Debug("pipeline.RunPipeline completed",
		zap.Bool("hasError", err != nil))
	if err != nil {
		h.logger.Error("Pipeline inference failed", zap.Error(err))
		return nil, fmt.Errorf("running feature extraction: %w", err)
	}

	// Extract embeddings from output
	// The pipeline returns last_hidden_state with shape [batch_size, seq_len, hidden_size]
	// We extract [CLS] token (first token) at position [:, 0, :]
	result := make([][]float32, len(output.Embeddings))

	for i, embedding := range output.Embeddings {
		// The FeatureExtractionPipeline already extracts mean pooling by default
		// but for BGE models we specifically want [CLS] token
		// Check if we got the expected embedding (should be pre-normalized by pipeline)
		if len(embedding) == 0 {
			h.logger.Error("Empty embedding returned",
				zap.Int("index", i))
			return nil, fmt.Errorf("empty embedding at index %d", i)
		}

		// Normalize the embedding (L2 normalization)
		// BGE and similar models typically use normalized embeddings
		normalized := normalizeL2(embedding)
		result[i] = normalized
	}

	h.logger.Debug("Embedding generation complete",
		zap.Int("numEmbeddings", len(result)),
	)

	return result, nil
}

// normalizeL2 performs L2 normalization on a vector
func normalizeL2(vec []float32) []float32 {
	// Calculate L2 norm
	var sum float32
	for _, v := range vec {
		sum += v * v
	}

	if sum == 0 {
		return vec // Avoid division by zero
	}

	// Normalize
	norm := float32(1.0) / float32(math.Sqrt(float64(sum)))
	normalized := make([]float32, len(vec))
	for i, v := range vec {
		normalized[i] = v * norm
	}

	return normalized
}

// Close releases resources
// Only destroys the session if it was created by this embedder (not shared)
func (h *HugotEmbedder) Close() error {
	if h.session != nil && !h.sessionShared {
		h.logger.Info("Destroying Hugot session (owned by this embedder)")
		return h.session.Destroy()
	} else if h.sessionShared {
		h.logger.Debug("Skipping session destruction (shared session)")
	}
	return nil
}

// PooledHugotEmbedder manages multiple ONNX pipelines for concurrent embedding generation.
// Each request acquires a pipeline slot via semaphore, enabling true parallelism.
type PooledHugotEmbedder struct {
	session       *khugot.Session
	pipelines     []*pipelines.FeatureExtractionPipeline
	sem           *semaphore.Weighted
	nextPipeline  atomic.Uint64
	logger        *zap.Logger
	sessionShared bool
	poolSize      int
	caps          embeddings.EmbedderCapabilities
}

// NewPooledHugotEmbedder creates a new pooled embedder using the Hugot ONNX runtime.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_f16.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
func NewPooledHugotEmbedder(modelPath string, onnxFilename string, poolSize int, logger *zap.Logger) (*PooledHugotEmbedder, error) {
	return NewPooledHugotEmbedderWithSession(modelPath, onnxFilename, poolSize, nil, logger)
}

// NewPooledHugotEmbedderWithSession creates a new pooled embedder using an optional shared Hugot session.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_f16.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
func NewPooledHugotEmbedderWithSession(modelPath string, onnxFilename string, poolSize int, sharedSession *khugot.Session, logger *zap.Logger) (*PooledHugotEmbedder, error) {
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

	// Auto-detect pool size from CPU count if not specified
	if poolSize <= 0 {
		poolSize = runtime.NumCPU()
	}

	logger.Info("Initializing pooled Hugot embedder",
		zap.String("modelPath", modelPath),
		zap.String("onnxFilename", onnxFilename),
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
	pipelinesList := make([]*pipelines.FeatureExtractionPipeline, poolSize)
	for i := 0; i < poolSize; i++ {
		pipelineName := fmt.Sprintf("%s:%s:%d", modelPath, onnxFilename, i)
		pipelineConfig := khugot.FeatureExtractionConfig{
			ModelPath:    modelPath,
			Name:         pipelineName,
			OnnxFilename: onnxFilename,
			Options: []backends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
				pipelines.WithNormalization(),
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
			return nil, fmt.Errorf("creating feature extraction pipeline %d: %w", i, err)
		}
		pipelinesList[i] = pipeline
		logger.Debug("Created pipeline", zap.Int("index", i), zap.String("name", pipelineName))
	}

	logger.Info("Successfully created pooled feature extraction pipelines", zap.Int("count", poolSize))

	return &PooledHugotEmbedder{
		session:       session,
		pipelines:     pipelinesList,
		sem:           semaphore.NewWeighted(int64(poolSize)),
		logger:        logger,
		sessionShared: sessionShared,
		poolSize:      poolSize,
		caps:          embeddings.TextOnlyCapabilities(),
	}, nil
}

// NewPooledHugotEmbedderWithSessionManager creates a new pooled embedder using a SessionManager.
// The SessionManager handles backend selection based on priority and model restrictions.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
// modelBackends specifies which backends this model supports (nil = all backends).
// Returns the embedder and the backend type that was used.
func NewPooledHugotEmbedderWithSessionManager(
	modelPath string,
	onnxFilename string,
	poolSize int,
	sessionManager *hugot.SessionManager,
	modelBackends []string,
	logger *zap.Logger,
) (*PooledHugotEmbedder, hugot.BackendType, error) {
	if modelPath == "" {
		return nil, "", errors.New("model path is required")
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

	// Get session from session manager
	var session *khugot.Session
	var backendUsed hugot.BackendType
	var err error

	if sessionManager != nil {
		session, backendUsed, err = sessionManager.GetSessionForModel(modelBackends)
		if err != nil {
			return nil, "", fmt.Errorf("getting session from manager: %w", err)
		}
		logger.Info("Using session from SessionManager",
			zap.String("backend", string(backendUsed)))
	} else {
		// Fallback to default session creation for backward compatibility
		session, err = hugot.NewSession()
		if err != nil {
			return nil, "", fmt.Errorf("creating hugot session: %w", err)
		}
		backendUsed = hugot.GetDefaultBackend().Type()
	}

	logger.Info("Initializing pooled Hugot embedder",
		zap.String("modelPath", modelPath),
		zap.String("onnxFilename", onnxFilename),
		zap.Int("poolSize", poolSize),
		zap.String("backend", string(backendUsed)))

	// Create N pipelines with unique names
	pipelinesList := make([]*pipelines.FeatureExtractionPipeline, poolSize)
	for i := 0; i < poolSize; i++ {
		pipelineName := fmt.Sprintf("%s:%s:%d", modelPath, onnxFilename, i)
		pipelineConfig := khugot.FeatureExtractionConfig{
			ModelPath:    modelPath,
			Name:         pipelineName,
			OnnxFilename: onnxFilename,
			Options: []backends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
				pipelines.WithNormalization(),
			},
		}

		pipeline, err := khugot.NewPipeline(session, pipelineConfig)
		if err != nil {
			logger.Error("Failed to create pipeline",
				zap.Int("index", i),
				zap.Error(err))
			return nil, "", fmt.Errorf("creating feature extraction pipeline %d: %w", i, err)
		}
		pipelinesList[i] = pipeline
		logger.Debug("Created pipeline", zap.Int("index", i), zap.String("name", pipelineName))
	}

	logger.Info("Successfully created pooled feature extraction pipelines", zap.Int("count", poolSize))

	return &PooledHugotEmbedder{
		session:       session,
		pipelines:     pipelinesList,
		sem:           semaphore.NewWeighted(int64(poolSize)),
		logger:        logger,
		sessionShared: true, // SessionManager owns the session
		poolSize:      poolSize,
		caps:          embeddings.TextOnlyCapabilities(),
	}, backendUsed, nil
}

// Capabilities returns the capabilities of this embedder
func (p *PooledHugotEmbedder) Capabilities() embeddings.EmbedderCapabilities {
	return p.caps
}

// Embed generates embeddings for the given content.
// Thread-safe: uses semaphore to limit concurrent pipeline access.
func (p *PooledHugotEmbedder) Embed(ctx context.Context, contents [][]ai.ContentPart) ([][]float32, error) {
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

	// Hugot only supports text embeddings
	values := embeddings.ExtractText(contents)

	p.logger.Debug("Starting embedding generation",
		zap.Int("pipelineIndex", idx),
		zap.Int("numTexts", len(values)))

	// Run feature extraction inference
	output, err := pipeline.RunPipeline(values)
	if err != nil {
		p.logger.Error("Pipeline inference failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, fmt.Errorf("running feature extraction: %w", err)
	}

	// Extract embeddings from output
	result := make([][]float32, len(output.Embeddings))
	for i, embedding := range output.Embeddings {
		if len(embedding) == 0 {
			p.logger.Error("Empty embedding returned",
				zap.Int("pipelineIndex", idx),
				zap.Int("index", i))
			return nil, fmt.Errorf("empty embedding at index %d", i)
		}
		// Normalize the embedding (L2 normalization)
		normalized := normalizeL2(embedding)
		result[i] = normalized
	}

	p.logger.Debug("Embedding generation complete",
		zap.Int("pipelineIndex", idx),
		zap.Int("numEmbeddings", len(result)))

	return result, nil
}

// Close releases resources.
// Only destroys the session if it was created by this embedder (not shared).
func (p *PooledHugotEmbedder) Close() error {
	if p.session != nil && !p.sessionShared {
		p.logger.Info("Destroying Hugot session (owned by this pooled embedder)")
		return p.session.Destroy()
	} else if p.sessionShared {
		p.logger.Debug("Skipping session destruction (shared session)")
	}
	return nil
}
