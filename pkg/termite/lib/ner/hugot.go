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

package ner

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync/atomic"

	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure HugotNER and PooledHugotNER implement the Model interface
var _ Model = (*HugotNER)(nil)
var _ Model = (*PooledHugotNER)(nil)

// HugotNER uses ONNX-based token classification for named entity recognition.
type HugotNER struct {
	session       *khugot.Session
	pipeline      *pipelines.TokenClassificationPipeline
	config        *NERConfig
	logger        *zap.Logger
	sessionShared bool // true if session is shared and shouldn't be destroyed
}

// NewHugotNER creates a new NER model using the Hugot ONNX runtime.
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
func NewHugotNER(modelPath string, onnxFilename string, logger *zap.Logger) (*HugotNER, error) {
	return NewHugotNERWithSession(modelPath, onnxFilename, nil, logger)
}

// NewHugotNERWithSession creates a new NER model using an optional shared Hugot session.
// If sharedSession is nil, a new session is created.
// If sharedSession is provided, it will be reused (important for ONNX Runtime which allows only one session).
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
func NewHugotNERWithSession(modelPath string, onnxFilename string, sharedSession *khugot.Session, logger *zap.Logger) (*HugotNER, error) {
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

	// Load NER configuration (labels)
	config, err := LoadNERConfig(modelPath)
	if err != nil {
		logger.Warn("Failed to load NER config, using default labels",
			zap.String("modelPath", modelPath),
			zap.Error(err))
		// Use default CoNLL labels if config not found
		config = &NERConfig{
			Labels: []string{"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"},
		}
	}

	logger.Info("Initializing Hugot NER",
		zap.String("modelPath", modelPath),
		zap.String("onnxFilename", onnxFilename),
		zap.Int("numLabels", len(config.Labels)),
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

	// Create token classification pipeline configuration
	// Use modelPath + onnxFilename as pipeline name to ensure uniqueness when multiple models share a session
	pipelineName := fmt.Sprintf("ner:%s:%s", modelPath, onnxFilename)
	pipelineConfig := khugot.TokenClassificationConfig{
		ModelPath:    modelPath,
		Name:         pipelineName,
		OnnxFilename: onnxFilename,
	}

	// Create the pipeline
	pipeline, err := khugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		// Only destroy session if we created it (not shared)
		if !sessionShared {
			_ = session.Destroy()
		}
		logger.Error("Failed to create pipeline", zap.Error(err))
		return nil, fmt.Errorf("creating token classification pipeline: %w", err)
	}

	// Set aggregation strategy to "SIMPLE" to group adjacent tokens with same entity type
	pipeline.AggregationStrategy = "SIMPLE"
	logger.Info("Successfully created NER token classification pipeline")

	return &HugotNER{
		session:       session,
		pipeline:      pipeline,
		config:        config,
		logger:        logger,
		sessionShared: sessionShared,
	}, nil
}

// NewHugotNERWithSessionManager creates a new NER model using a SessionManager.
// The SessionManager handles backend selection based on priority and model compatibility.
// modelBackends restricts which backends can be used (empty = all backends allowed).
func NewHugotNERWithSessionManager(modelPath string, onnxFilename string, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*HugotNER, hugot.BackendType, error) {
	if modelPath == "" {
		return nil, "", errors.New("model path is required")
	}

	if onnxFilename == "" {
		onnxFilename = "model.onnx"
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	if sessionManager == nil {
		// Fall back to creating a new session
		model, err := NewHugotNERWithSession(modelPath, onnxFilename, nil, logger)
		if err != nil {
			return nil, "", err
		}
		return model, hugot.BackendType(""), nil
	}

	// Load NER configuration (labels)
	config, err := LoadNERConfig(modelPath)
	if err != nil {
		logger.Warn("Failed to load NER config, using default labels",
			zap.String("modelPath", modelPath),
			zap.Error(err))
		config = &NERConfig{
			Labels: []string{"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"},
		}
	}

	logger.Info("Initializing Hugot NER with SessionManager",
		zap.String("modelPath", modelPath),
		zap.String("onnxFilename", onnxFilename),
		zap.Int("numLabels", len(config.Labels)))

	// Get session from SessionManager with backend restrictions
	session, backendUsed, err := sessionManager.GetSessionForModel(modelBackends)
	if err != nil {
		logger.Error("Failed to get session from SessionManager", zap.Error(err))
		return nil, "", fmt.Errorf("getting session from SessionManager: %w", err)
	}

	// Create token classification pipeline
	pipelineName := fmt.Sprintf("ner:%s:%s", modelPath, onnxFilename)
	pipelineConfig := khugot.TokenClassificationConfig{
		ModelPath:    modelPath,
		Name:         pipelineName,
		OnnxFilename: onnxFilename,
	}

	pipeline, err := khugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		logger.Error("Failed to create pipeline", zap.Error(err))
		return nil, "", fmt.Errorf("creating token classification pipeline: %w", err)
	}

	pipeline.AggregationStrategy = "SIMPLE"
	logger.Info("Successfully created NER token classification pipeline",
		zap.String("backend", string(backendUsed)))

	return &HugotNER{
		session:       session,
		pipeline:      pipeline,
		config:        config,
		logger:        logger,
		sessionShared: true, // SessionManager owns the session
	}, backendUsed, nil
}

// Recognize extracts named entities from the given texts.
func (h *HugotNER) Recognize(ctx context.Context, texts []string) ([][]Entity, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	h.logger.Debug("Running NER on texts",
		zap.Int("num_texts", len(texts)))

	// Run token classification pipeline
	output, err := h.pipeline.RunPipeline(texts)
	if err != nil {
		h.logger.Error("Token classification failed", zap.Error(err))
		return nil, fmt.Errorf("running token classification: %w", err)
	}

	if len(output.Entities) != len(texts) {
		h.logger.Warn("Entity count mismatch",
			zap.Int("expected", len(texts)),
			zap.Int("got", len(output.Entities)))
	}

	// Convert pipeline output to NER entities
	results := make([][]Entity, len(texts))
	for i, textEntities := range output.Entities {
		results[i] = h.parseEntities(texts[i], textEntities)
	}

	h.logger.Debug("NER completed",
		zap.Int("num_texts", len(texts)),
		zap.Int("total_entities", countEntities(results)))

	return results, nil
}

// parseEntities converts pipeline entities to NER entities.
// Handles BIO format labels (B-PER, I-PER, etc.) and entity aggregation.
func (h *HugotNER) parseEntities(text string, pipelineEntities []pipelines.Entity) []Entity {
	if len(pipelineEntities) == 0 {
		return nil
	}

	entities := make([]Entity, 0, len(pipelineEntities))

	for _, pe := range pipelineEntities {
		// Skip O (outside) labels
		if IsBIOOutside(pe.Entity) {
			continue
		}

		// Get the entity text from the original text using character offsets
		start := int(pe.Start)
		end := int(pe.End)
		if start < 0 || end > len(text) || start >= end {
			h.logger.Debug("Invalid entity offsets",
				zap.Int("start", start),
				zap.Int("end", end),
				zap.Int("text_len", len(text)))
			continue
		}

		entityText := text[start:end]
		label := NormalizeLabel(pe.Entity)

		if label == "" {
			continue
		}

		entities = append(entities, Entity{
			Text:  entityText,
			Label: label,
			Start: start,
			End:   end,
			Score: pe.Score,
		})
	}

	return entities
}

// Close releases resources.
// Only destroys the session if it was created by this NER model (not shared).
func (h *HugotNER) Close() error {
	if h.session != nil && !h.sessionShared {
		h.logger.Info("Destroying Hugot session (owned by this NER model)")
		return h.session.Destroy()
	} else if h.sessionShared {
		h.logger.Debug("Skipping session destruction (shared session)")
	}
	return nil
}

// PooledHugotNER manages multiple ONNX pipelines for concurrent NER.
// Each request acquires a pipeline slot via semaphore, enabling true parallelism.
type PooledHugotNER struct {
	session       *khugot.Session
	pipelines     []*pipelines.TokenClassificationPipeline
	config        *NERConfig
	sem           *semaphore.Weighted
	nextPipeline  atomic.Uint64
	logger        *zap.Logger
	sessionShared bool
	poolSize      int
}

// NewPooledHugotNER creates a new pooled NER model using the Hugot ONNX runtime.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
func NewPooledHugotNER(modelPath string, onnxFilename string, poolSize int, logger *zap.Logger) (*PooledHugotNER, error) {
	return NewPooledHugotNERWithSession(modelPath, onnxFilename, poolSize, nil, logger)
}

// NewPooledHugotNERWithSession creates a new pooled NER model using an optional shared Hugot session.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
func NewPooledHugotNERWithSession(modelPath string, onnxFilename string, poolSize int, sharedSession *khugot.Session, logger *zap.Logger) (*PooledHugotNER, error) {
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

	// Load NER configuration (labels)
	config, err := LoadNERConfig(modelPath)
	if err != nil {
		logger.Warn("Failed to load NER config, using default labels",
			zap.String("modelPath", modelPath),
			zap.Error(err))
		// Use default CoNLL labels if config not found
		config = &NERConfig{
			Labels: []string{"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"},
		}
	}

	logger.Info("Initializing pooled Hugot NER",
		zap.String("modelPath", modelPath),
		zap.String("onnxFilename", onnxFilename),
		zap.Int("poolSize", poolSize),
		zap.Int("numLabels", len(config.Labels)),
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
	pipelinesList := make([]*pipelines.TokenClassificationPipeline, poolSize)
	for i := 0; i < poolSize; i++ {
		pipelineName := fmt.Sprintf("ner:%s:%s:%d", modelPath, onnxFilename, i)
		pipelineConfig := khugot.TokenClassificationConfig{
			ModelPath:    modelPath,
			Name:         pipelineName,
			OnnxFilename: onnxFilename,
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
			return nil, fmt.Errorf("creating token classification pipeline %d: %w", i, err)
		}

		// Set aggregation strategy to "SIMPLE" for entity grouping
		pipeline.AggregationStrategy = "SIMPLE"
		pipelinesList[i] = pipeline
		logger.Debug("Created pipeline", zap.Int("index", i), zap.String("name", pipelineName))
	}

	logger.Info("Successfully created pooled NER pipelines", zap.Int("count", poolSize))

	return &PooledHugotNER{
		session:       session,
		pipelines:     pipelinesList,
		config:        config,
		sem:           semaphore.NewWeighted(int64(poolSize)),
		logger:        logger,
		sessionShared: sessionShared,
		poolSize:      poolSize,
	}, nil
}

// NewPooledHugotNERWithSessionManager creates a new pooled NER model using a SessionManager.
// The SessionManager handles backend selection based on priority and model compatibility.
// modelBackends restricts which backends can be used (empty = all backends allowed).
func NewPooledHugotNERWithSessionManager(modelPath string, onnxFilename string, poolSize int, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*PooledHugotNER, hugot.BackendType, error) {
	if modelPath == "" {
		return nil, "", errors.New("model path is required")
	}

	if onnxFilename == "" {
		onnxFilename = "model.onnx"
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	if sessionManager == nil {
		// Fall back to creating a new session
		model, err := NewPooledHugotNERWithSession(modelPath, onnxFilename, poolSize, nil, logger)
		if err != nil {
			return nil, "", err
		}
		return model, hugot.BackendType(""), nil
	}

	if poolSize <= 0 {
		poolSize = runtime.NumCPU()
	}

	// Load NER configuration (labels)
	config, err := LoadNERConfig(modelPath)
	if err != nil {
		logger.Warn("Failed to load NER config, using default labels",
			zap.String("modelPath", modelPath),
			zap.Error(err))
		config = &NERConfig{
			Labels: []string{"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"},
		}
	}

	logger.Info("Initializing pooled Hugot NER with SessionManager",
		zap.String("modelPath", modelPath),
		zap.String("onnxFilename", onnxFilename),
		zap.Int("poolSize", poolSize),
		zap.Int("numLabels", len(config.Labels)))

	// Get session from SessionManager with backend restrictions
	session, backendUsed, err := sessionManager.GetSessionForModel(modelBackends)
	if err != nil {
		logger.Error("Failed to get session from SessionManager", zap.Error(err))
		return nil, "", fmt.Errorf("getting session from SessionManager: %w", err)
	}

	// Create N pipelines with unique names
	pipelinesList := make([]*pipelines.TokenClassificationPipeline, poolSize)
	for i := 0; i < poolSize; i++ {
		pipelineName := fmt.Sprintf("ner:%s:%s:%d", modelPath, onnxFilename, i)
		pipelineConfig := khugot.TokenClassificationConfig{
			ModelPath:    modelPath,
			Name:         pipelineName,
			OnnxFilename: onnxFilename,
		}

		pipeline, err := khugot.NewPipeline(session, pipelineConfig)
		if err != nil {
			logger.Error("Failed to create pipeline",
				zap.Int("index", i),
				zap.Error(err))
			return nil, "", fmt.Errorf("creating token classification pipeline %d: %w", i, err)
		}

		pipeline.AggregationStrategy = "SIMPLE"
		pipelinesList[i] = pipeline
		logger.Debug("Created pipeline", zap.Int("index", i), zap.String("name", pipelineName))
	}

	logger.Info("Successfully created pooled NER pipelines",
		zap.Int("count", poolSize),
		zap.String("backend", string(backendUsed)))

	return &PooledHugotNER{
		session:       session,
		pipelines:     pipelinesList,
		config:        config,
		sem:           semaphore.NewWeighted(int64(poolSize)),
		logger:        logger,
		sessionShared: true, // SessionManager owns the session
		poolSize:      poolSize,
	}, backendUsed, nil
}

// Recognize extracts named entities from the given texts.
// Thread-safe: uses semaphore to limit concurrent pipeline access.
func (p *PooledHugotNER) Recognize(ctx context.Context, texts []string) ([][]Entity, error) {
	if len(texts) == 0 {
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

	p.logger.Debug("Using pipeline for NER",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_texts", len(texts)))

	// Run token classification pipeline
	output, err := pipeline.RunPipeline(texts)
	if err != nil {
		p.logger.Error("Pipeline inference failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, fmt.Errorf("running token classification: %w", err)
	}

	// Convert pipeline output to NER entities
	results := make([][]Entity, len(texts))
	for i, textEntities := range output.Entities {
		results[i] = p.parseEntities(texts[i], textEntities)
	}

	p.logger.Debug("NER completed",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_texts", len(texts)),
		zap.Int("total_entities", countEntities(results)))

	return results, nil
}

// parseEntities converts pipeline entities to NER entities.
func (p *PooledHugotNER) parseEntities(text string, pipelineEntities []pipelines.Entity) []Entity {
	if len(pipelineEntities) == 0 {
		return nil
	}

	entities := make([]Entity, 0, len(pipelineEntities))

	for _, pe := range pipelineEntities {
		// Skip O (outside) labels
		if IsBIOOutside(pe.Entity) {
			continue
		}

		// Get the entity text from the original text using character offsets
		start := int(pe.Start)
		end := int(pe.End)
		if start < 0 || end > len(text) || start >= end {
			p.logger.Debug("Invalid entity offsets",
				zap.Int("start", start),
				zap.Int("end", end),
				zap.Int("text_len", len(text)))
			continue
		}

		entityText := text[start:end]
		label := NormalizeLabel(pe.Entity)

		if label == "" {
			continue
		}

		entities = append(entities, Entity{
			Text:  entityText,
			Label: label,
			Start: start,
			End:   end,
			Score: pe.Score,
		})
	}

	return entities
}

// Close releases resources.
// Only destroys the session if it was created by this NER model (not shared).
func (p *PooledHugotNER) Close() error {
	if p.session != nil && !p.sessionShared {
		p.logger.Info("Destroying Hugot session (owned by this pooled NER model)")
		return p.session.Destroy()
	} else if p.sessionShared {
		p.logger.Debug("Skipping session destruction (shared session)")
	}
	return nil
}

// countEntities returns the total number of entities across all texts.
func countEntities(results [][]Entity) int {
	count := 0
	for _, entities := range results {
		count += len(entities)
	}
	return count
}
