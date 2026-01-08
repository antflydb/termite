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

package classification

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"

	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure HugotClassifier implements Classifier
var _ Classifier = (*HugotClassifier)(nil)

// HugotClassifier implements zero-shot classification using the Hugot ONNX runtime.
// It uses NLI-based models like mDeBERTa for zero-shot text classification.
type HugotClassifier struct {
	session       *khugot.Session
	pipeline      *pipelines.ZeroShotClassificationPipeline
	logger        *zap.Logger
	sessionShared bool
	config        Config
}

// PooledHugotClassifier manages multiple zero-shot classification pipelines for concurrent requests.
type PooledHugotClassifier struct {
	session       *khugot.Session
	pipelines     []*pipelines.ZeroShotClassificationPipeline
	sem           *semaphore.Weighted
	nextPipeline  atomic.Uint64
	logger        *zap.Logger
	sessionShared bool
	poolSize      int
	config        Config
}

// NewHugotClassifier creates a new zero-shot classifier using the Hugot ONNX runtime.
func NewHugotClassifier(modelPath string, logger *zap.Logger) (*HugotClassifier, error) {
	return NewHugotClassifierWithSession(modelPath, nil, logger)
}

// NewHugotClassifierWithSession creates a new zero-shot classifier with an optional shared session.
func NewHugotClassifierWithSession(modelPath string, sharedSession *khugot.Session, logger *zap.Logger) (*HugotClassifier, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Initializing Hugot Zero-Shot Classifier",
		zap.String("modelPath", modelPath),
		zap.String("backend", hugot.BackendName()))

	// Load config if available
	config := Config{
		HypothesisTemplate: DefaultHypothesisTemplate,
		MultiLabel:         false,
		Threshold:          0.0,
	}
	configPath := filepath.Join(modelPath, "zsc_config.json")
	if data, err := os.ReadFile(configPath); err == nil {
		if err := json.Unmarshal(data, &config); err != nil {
			logger.Warn("Failed to parse ZSC config", zap.Error(err))
		} else {
			logger.Info("Loaded ZSC config",
				zap.String("hypothesis_template", config.HypothesisTemplate),
				zap.Bool("multi_label", config.MultiLabel),
				zap.Float32("threshold", config.Threshold))
		}
	}

	// Use shared session or create new one
	session, err := hugot.NewSessionOrUseExisting(sharedSession)
	if err != nil {
		logger.Error("Failed to create Hugot session", zap.Error(err))
		return nil, fmt.Errorf("creating hugot session: %w", err)
	}
	sessionShared := (sharedSession != nil)

	// Determine ONNX file
	onnxFilename := "model.onnx"

	// Create zero-shot classification pipeline
	pipelineName := fmt.Sprintf("zsc:%s:%s", modelPath, onnxFilename)
	pipelineOptions := []khugot.ZeroShotClassificationOption{
		pipelines.WithHypothesisTemplate(config.HypothesisTemplate),
	}

	if config.MultiLabel {
		pipelineOptions = append(pipelineOptions, pipelines.WithMultilabel(true))
	}

	pipelineConfig := khugot.ZeroShotClassificationConfig{
		ModelPath:    modelPath,
		OnnxFilename: onnxFilename,
		Name:         pipelineName,
		Options:      pipelineOptions,
	}

	pipeline, err := khugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		if !sessionShared {
			session.Destroy()
		}
		logger.Error("Failed to create ZeroShotClassification pipeline", zap.Error(err))
		return nil, fmt.Errorf("creating ZeroShotClassification pipeline: %w", err)
	}

	logger.Info("Zero-Shot Classifier initialization complete",
		zap.String("hypothesis_template", config.HypothesisTemplate),
		zap.Bool("multi_label", config.MultiLabel))

	return &HugotClassifier{
		session:       session,
		pipeline:      pipeline,
		logger:        logger,
		sessionShared: sessionShared,
		config:        config,
	}, nil
}

// NewHugotClassifierWithSessionManager creates a new zero-shot classifier using a SessionManager.
func NewHugotClassifierWithSessionManager(modelPath string, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*HugotClassifier, hugot.BackendType, error) {
	if modelPath == "" {
		return nil, "", errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	if sessionManager == nil {
		model, err := NewHugotClassifierWithSession(modelPath, nil, logger)
		if err != nil {
			return nil, "", err
		}
		return model, hugot.BackendType(""), nil
	}

	logger.Info("Initializing Hugot Zero-Shot Classifier with SessionManager",
		zap.String("modelPath", modelPath))

	// Load config
	config := Config{
		HypothesisTemplate: DefaultHypothesisTemplate,
		MultiLabel:         false,
		Threshold:          0.0,
	}
	configPath := filepath.Join(modelPath, "zsc_config.json")
	if data, err := os.ReadFile(configPath); err == nil {
		if err := json.Unmarshal(data, &config); err != nil {
			logger.Warn("Failed to parse ZSC config", zap.Error(err))
		}
	}

	// Get session from SessionManager
	session, backendUsed, err := sessionManager.GetSessionForModel(modelBackends)
	if err != nil {
		logger.Error("Failed to get session from SessionManager", zap.Error(err))
		return nil, "", fmt.Errorf("getting session from SessionManager: %w", err)
	}

	// Create pipeline
	onnxFilename := "model.onnx"
	pipelineName := fmt.Sprintf("zsc:%s:%s", modelPath, onnxFilename)
	pipelineOptions := []khugot.ZeroShotClassificationOption{
		pipelines.WithHypothesisTemplate(config.HypothesisTemplate),
	}

	if config.MultiLabel {
		pipelineOptions = append(pipelineOptions, pipelines.WithMultilabel(true))
	}

	pipelineConfig := khugot.ZeroShotClassificationConfig{
		ModelPath:    modelPath,
		OnnxFilename: onnxFilename,
		Name:         pipelineName,
		Options:      pipelineOptions,
	}

	pipeline, err := khugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		logger.Error("Failed to create ZeroShotClassification pipeline", zap.Error(err))
		return nil, "", fmt.Errorf("creating ZeroShotClassification pipeline: %w", err)
	}

	logger.Info("Zero-Shot Classifier initialization complete",
		zap.String("backend", string(backendUsed)))

	return &HugotClassifier{
		session:       session,
		pipeline:      pipeline,
		logger:        logger,
		sessionShared: true,
		config:        config,
	}, backendUsed, nil
}

// Classify classifies texts using the specified candidate labels.
func (c *HugotClassifier) Classify(ctx context.Context, texts []string, labels []string) ([][]Classification, error) {
	return c.ClassifyWithHypothesis(ctx, texts, labels, c.config.HypothesisTemplate)
}

// ClassifyWithHypothesis classifies texts using a custom hypothesis template.
func (c *HugotClassifier) ClassifyWithHypothesis(ctx context.Context, texts []string, labels []string, hypothesisTemplate string) ([][]Classification, error) {
	if len(texts) == 0 {
		return [][]Classification{}, nil
	}

	if len(labels) == 0 {
		return nil, errors.New("at least one label is required")
	}

	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	c.logger.Debug("Starting zero-shot classification",
		zap.Int("num_texts", len(texts)),
		zap.Strings("labels", labels),
		zap.String("hypothesis_template", hypothesisTemplate))

	// Set labels and hypothesis template on the pipeline
	c.pipeline.Labels = labels
	if hypothesisTemplate != "" {
		c.pipeline.HypothesisTemplate = hypothesisTemplate
	}

	// Run the pipeline
	output, err := c.pipeline.RunPipeline(texts)
	if err != nil {
		c.logger.Error("Zero-shot classification failed", zap.Error(err))
		return nil, fmt.Errorf("running zero-shot classification: %w", err)
	}

	// Convert output to our Classification type
	results := convertZSCOutput(output)

	c.logger.Debug("Zero-shot classification completed",
		zap.Int("num_texts", len(texts)),
		zap.Int("num_results", len(results)))

	return results, nil
}

// MultiLabelClassify classifies texts allowing multiple labels per text.
func (c *HugotClassifier) MultiLabelClassify(ctx context.Context, texts []string, labels []string) ([][]Classification, error) {
	if len(texts) == 0 {
		return [][]Classification{}, nil
	}

	if len(labels) == 0 {
		return nil, errors.New("at least one label is required")
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	c.logger.Debug("Starting multi-label classification",
		zap.Int("num_texts", len(texts)),
		zap.Strings("labels", labels))

	// Set labels and multi-label mode on the pipeline
	c.pipeline.Labels = labels
	c.pipeline.Multilabel = true

	// Run the pipeline
	output, err := c.pipeline.RunPipeline(texts)
	if err != nil {
		c.logger.Error("Multi-label classification failed", zap.Error(err))
		return nil, fmt.Errorf("running multi-label classification: %w", err)
	}

	// Reset multilabel for future calls
	c.pipeline.Multilabel = false

	results := convertZSCOutput(output)

	// Apply threshold if configured
	if c.config.Threshold > 0 {
		for i := range results {
			filtered := make([]Classification, 0, len(results[i]))
			for _, cls := range results[i] {
				if cls.Score >= c.config.Threshold {
					filtered = append(filtered, cls)
				}
			}
			results[i] = filtered
		}
	}

	return results, nil
}

// Close releases resources.
func (c *HugotClassifier) Close() error {
	if c.session != nil && !c.sessionShared {
		c.logger.Info("Destroying Hugot session (owned by this ZSC model)")
		c.session.Destroy()
	}
	return nil
}

// Config returns the classifier configuration.
func (c *HugotClassifier) Config() Config {
	return c.config
}

// convertZSCOutput converts pipeline output to Classification slices.
func convertZSCOutput(output *pipelines.ZeroShotOutput) [][]Classification {
	if output == nil {
		return [][]Classification{}
	}

	results := make([][]Classification, len(output.ClassificationOutputs))
	for i, out := range output.ClassificationOutputs {
		// SortedValues is already sorted by score descending
		classifications := make([]Classification, len(out.SortedValues))
		for j, kv := range out.SortedValues {
			classifications[j] = Classification{
				Label: kv.Key,
				Score: float32(kv.Value),
			}
		}
		results[i] = classifications
	}
	return results
}

// =============================================================================
// Pooled Implementation for Concurrent Access
// =============================================================================

// NewPooledHugotClassifier creates a pooled zero-shot classifier for concurrent requests.
func NewPooledHugotClassifier(modelPath string, poolSize int, logger *zap.Logger) (*PooledHugotClassifier, error) {
	return NewPooledHugotClassifierWithSession(modelPath, poolSize, nil, logger)
}

// NewPooledHugotClassifierWithSession creates a pooled zero-shot classifier with an optional shared session.
func NewPooledHugotClassifierWithSession(modelPath string, poolSize int, sharedSession *khugot.Session, logger *zap.Logger) (*PooledHugotClassifier, error) {
	classifier, _, err := newPooledHugotClassifierInternal(modelPath, poolSize, sharedSession, nil, nil, logger)
	return classifier, err
}

// NewPooledHugotClassifierWithSessionManager creates a pooled zero-shot classifier using a SessionManager.
func NewPooledHugotClassifierWithSessionManager(modelPath string, poolSize int, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*PooledHugotClassifier, hugot.BackendType, error) {
	return newPooledHugotClassifierInternal(modelPath, poolSize, nil, sessionManager, modelBackends, logger)
}

func newPooledHugotClassifierInternal(modelPath string, poolSize int, sharedSession *khugot.Session, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*PooledHugotClassifier, hugot.BackendType, error) {
	if modelPath == "" {
		return nil, hugot.BackendGo, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	// Auto-detect pool size
	if poolSize <= 0 {
		poolSize = runtime.NumCPU()
		if poolSize > 4 {
			poolSize = 4 // Cap at 4 for ZSC models
		}
	}

	logger.Info("Initializing pooled Hugot Zero-Shot Classifier",
		zap.String("modelPath", modelPath),
		zap.Int("poolSize", poolSize),
		zap.String("backend", hugot.BackendName()))

	// Load config
	config := Config{
		HypothesisTemplate: DefaultHypothesisTemplate,
		MultiLabel:         false,
		Threshold:          0.0,
	}
	configPath := filepath.Join(modelPath, "zsc_config.json")
	if data, err := os.ReadFile(configPath); err == nil {
		if err := json.Unmarshal(data, &config); err != nil {
			logger.Warn("Failed to parse ZSC config", zap.Error(err))
		}
	}

	// Get or create session
	var session *khugot.Session
	var backendUsed hugot.BackendType
	var err error
	var sessionShared bool

	if sessionManager != nil {
		session, backendUsed, err = sessionManager.GetSessionForModel(modelBackends)
		if err != nil {
			return nil, backendUsed, fmt.Errorf("getting session from SessionManager: %w", err)
		}
		sessionShared = true
	} else if sharedSession != nil {
		session = sharedSession
		sessionShared = true
		backendUsed = hugot.GetDefaultBackend().Type()
	} else {
		session, err = hugot.NewSession()
		if err != nil {
			return nil, hugot.BackendGo, fmt.Errorf("creating hugot session: %w", err)
		}
		sessionShared = false
		backendUsed = hugot.GetDefaultBackend().Type()
	}

	// Create pipelines
	onnxFilename := "model.onnx"
	pipelinesList := make([]*pipelines.ZeroShotClassificationPipeline, poolSize)

	for i := 0; i < poolSize; i++ {
		pipelineName := fmt.Sprintf("zsc:%s:%s:%d", modelPath, onnxFilename, i)
		// Provide default labels (required by hugot) - will be overridden per request
		defaultLabels := []string{"positive", "negative", "neutral"}
		pipelineOptions := []khugot.ZeroShotClassificationOption{
			pipelines.WithHypothesisTemplate(config.HypothesisTemplate),
			pipelines.WithLabels(defaultLabels),
		}
		if config.MultiLabel {
			pipelineOptions = append(pipelineOptions, pipelines.WithMultilabel(true))
		}

		pipelineConfig := khugot.ZeroShotClassificationConfig{
			ModelPath:    modelPath,
			OnnxFilename: onnxFilename,
			Name:         pipelineName,
			Options:      pipelineOptions,
		}

		pipeline, err := khugot.NewPipeline(session, pipelineConfig)
		if err != nil {
			if !sessionShared {
				_ = session.Destroy()
			}
			return nil, backendUsed, fmt.Errorf("creating ZSC pipeline %d: %w", i, err)
		}
		pipelinesList[i] = pipeline
	}

	logger.Info("Successfully created pooled ZSC pipelines", zap.Int("count", poolSize))

	return &PooledHugotClassifier{
		session:       session,
		pipelines:     pipelinesList,
		sem:           semaphore.NewWeighted(int64(poolSize)),
		logger:        logger,
		sessionShared: sessionShared,
		poolSize:      poolSize,
		config:        config,
	}, backendUsed, nil
}

// Classify classifies texts using the specified candidate labels.
func (p *PooledHugotClassifier) Classify(ctx context.Context, texts []string, labels []string) ([][]Classification, error) {
	return p.ClassifyWithHypothesis(ctx, texts, labels, p.config.HypothesisTemplate)
}

// ClassifyWithHypothesis classifies texts using a custom hypothesis template.
func (p *PooledHugotClassifier) ClassifyWithHypothesis(ctx context.Context, texts []string, labels []string, hypothesisTemplate string) ([][]Classification, error) {
	if len(texts) == 0 {
		return [][]Classification{}, nil
	}

	if len(labels) == 0 {
		return nil, errors.New("at least one label is required")
	}

	// Acquire semaphore slot
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}
	defer p.sem.Release(1)

	// Round-robin pipeline selection
	idx := int(p.nextPipeline.Add(1) % uint64(p.poolSize))
	pipeline := p.pipelines[idx]

	p.logger.Debug("Using pipeline for ZSC",
		zap.Int("pipelineIndex", idx),
		zap.Int("numTexts", len(texts)))

	// Set labels and hypothesis template on the pipeline
	pipeline.Labels = labels
	if hypothesisTemplate != "" {
		pipeline.HypothesisTemplate = hypothesisTemplate
	}

	output, err := pipeline.RunPipeline(texts)
	if err != nil {
		p.logger.Error("Pipeline inference failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, fmt.Errorf("running zero-shot classification: %w", err)
	}

	results := convertZSCOutput(output)

	p.logger.Debug("ZSC complete",
		zap.Int("pipelineIndex", idx),
		zap.Int("numResults", len(results)))

	return results, nil
}

// MultiLabelClassify classifies texts allowing multiple labels per text.
func (p *PooledHugotClassifier) MultiLabelClassify(ctx context.Context, texts []string, labels []string) ([][]Classification, error) {
	if len(texts) == 0 {
		return [][]Classification{}, nil
	}

	if len(labels) == 0 {
		return nil, errors.New("at least one label is required")
	}

	// Acquire semaphore slot
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}
	defer p.sem.Release(1)

	// Round-robin pipeline selection
	idx := int(p.nextPipeline.Add(1) % uint64(p.poolSize))
	pipeline := p.pipelines[idx]

	// Set labels and multi-label mode on the pipeline
	pipeline.Labels = labels
	pipeline.Multilabel = true

	output, err := pipeline.RunPipeline(texts)

	// Reset multilabel for future calls
	pipeline.Multilabel = false

	if err != nil {
		p.logger.Error("Multi-label pipeline inference failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, fmt.Errorf("running multi-label classification: %w", err)
	}

	results := convertZSCOutput(output)

	// Apply threshold if configured
	if p.config.Threshold > 0 {
		for i := range results {
			filtered := make([]Classification, 0, len(results[i]))
			for _, cls := range results[i] {
				if cls.Score >= p.config.Threshold {
					filtered = append(filtered, cls)
				}
			}
			results[i] = filtered
		}
	}

	return results, nil
}

// Close releases resources.
func (p *PooledHugotClassifier) Close() error {
	if p.session != nil && !p.sessionShared {
		p.logger.Info("Destroying Hugot session (owned by this pooled ZSC)")
		return p.session.Destroy()
	}
	return nil
}

// Config returns the classifier configuration.
func (p *PooledHugotClassifier) Config() Config {
	return p.config
}

// =============================================================================
// Model Detection
// =============================================================================

// IsClassifierModel checks if the model path contains a zero-shot classification model.
func IsClassifierModel(modelPath string) bool {
	// Check for zsc_config.json
	configPath := filepath.Join(modelPath, "zsc_config.json")
	if _, err := os.Stat(configPath); err == nil {
		return true
	}

	// Check for known ZSC model patterns in name
	modelName := strings.ToLower(filepath.Base(modelPath))
	return strings.Contains(modelName, "mnli") ||
		strings.Contains(modelName, "xnli") ||
		strings.Contains(modelName, "nli") ||
		strings.Contains(modelName, "zero-shot") ||
		strings.Contains(modelName, "zeroshot")
}
