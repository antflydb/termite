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
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync/atomic"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure PooledClassifier implements Classifier
var _ Classifier = (*PooledClassifier)(nil)

// PooledClassifierConfig holds configuration for creating a PooledClassifier.
type PooledClassifierConfig struct {
	// ModelPath is the path to the model directory
	ModelPath string

	// PoolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count)
	PoolSize int

	// ModelBackends specifies which backends this model supports (nil = all backends)
	ModelBackends []string

	// Logger for logging (nil = no logging)
	Logger *zap.Logger
}

// PooledClassifier manages multiple ClassificationPipeline instances for concurrent zero-shot classification.
// Uses the new backends package (go-huggingface + gomlx/onnxruntime) instead of hugot.
type PooledClassifier struct {
	pipelines    []*pipelines.ClassificationPipeline
	sem          *semaphore.Weighted
	nextPipeline atomic.Uint64
	config       Config
	logger       *zap.Logger
	poolSize     int
	backendType  backends.BackendType
}

// NewPooledClassifier creates a new ClassificationPipeline-based pooled zero-shot classifier.
// This is the new implementation using go-huggingface tokenizers and the backends package.
func NewPooledClassifier(
	cfg PooledClassifierConfig,
	sessionManager *backends.SessionManager,
) (*PooledClassifier, backends.BackendType, error) {
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
		if poolSize > 4 {
			poolSize = 4 // Cap at 4 for ZSC models (memory intensive)
		}
	}

	// Load ZSC configuration (hypothesis template, multi-label settings)
	zscConfig, err := LoadZSCConfig(cfg.ModelPath)
	if err != nil {
		logger.Warn("Failed to load ZSC config, using defaults",
			zap.String("modelPath", cfg.ModelPath),
			zap.Error(err))
		zscConfig = &Config{
			HypothesisTemplate: DefaultHypothesisTemplate,
			MultiLabel:         false,
			Threshold:          0.0,
		}
	}

	logger.Info("Initializing pooled zero-shot classifier",
		zap.String("modelPath", cfg.ModelPath),
		zap.Int("poolSize", poolSize),
		zap.String("hypothesisTemplate", zscConfig.HypothesisTemplate))

	// Create N ClassificationPipelines
	pipelinesList := make([]*pipelines.ClassificationPipeline, poolSize)
	var backendUsed backends.BackendType

	for i := 0; i < poolSize; i++ {
		pipeline, bt, err := pipelines.LoadClassificationPipeline(
			cfg.ModelPath,
			sessionManager,
			cfg.ModelBackends,
			pipelines.WithHypothesisTemplate(zscConfig.HypothesisTemplate),
			pipelines.WithClassificationMultiLabel(zscConfig.MultiLabel),
		)
		if err != nil {
			// Clean up already-created pipelines
			for j := 0; j < i; j++ {
				if pipelinesList[j] != nil {
					_ = pipelinesList[j].Close()
				}
			}
			logger.Error("Failed to create classification pipeline",
				zap.Int("index", i),
				zap.Error(err))
			return nil, "", fmt.Errorf("creating classification pipeline %d: %w", i, err)
		}
		pipelinesList[i] = pipeline
		backendUsed = bt
		logger.Debug("Created classification pipeline", zap.Int("index", i), zap.String("backend", string(bt)))
	}

	logger.Info("Successfully created pooled ZSC pipelines",
		zap.Int("count", poolSize),
		zap.String("backend", string(backendUsed)))

	return &PooledClassifier{
		pipelines:   pipelinesList,
		sem:         semaphore.NewWeighted(int64(poolSize)),
		config:      *zscConfig,
		logger:      logger,
		poolSize:    poolSize,
		backendType: backendUsed,
	}, backendUsed, nil
}

// BackendType returns the backend type used by this classifier
func (p *PooledClassifier) BackendType() backends.BackendType {
	return p.backendType
}

// Classify classifies texts using the specified candidate labels.
// Thread-safe: uses semaphore to limit concurrent pipeline access.
func (p *PooledClassifier) Classify(ctx context.Context, texts []string, labels []string) ([][]Classification, error) {
	return p.ClassifyWithHypothesis(ctx, texts, labels, p.config.HypothesisTemplate)
}

// ClassifyWithHypothesis classifies texts using a custom hypothesis template.
// Note: The hypothesis template is currently ignored; pipeline uses template from creation time.
func (p *PooledClassifier) ClassifyWithHypothesis(ctx context.Context, texts []string, labels []string, hypothesisTemplate string) ([][]Classification, error) {
	if len(texts) == 0 {
		return [][]Classification{}, nil
	}

	if len(labels) == 0 {
		return nil, fmt.Errorf("at least one label is required")
	}

	// Acquire semaphore slot (blocks if all pipelines busy)
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}
	defer p.sem.Release(1)

	// Round-robin pipeline selection
	idx := int(p.nextPipeline.Add(1) % uint64(p.poolSize))
	pipeline := p.pipelines[idx]

	p.logger.Debug("Using pipeline for ZSC",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_texts", len(texts)),
		zap.Int("num_labels", len(labels)))

	// Delegate to ClassificationPipeline.ClassifyWithLabels
	pipelineResults, err := pipeline.ClassifyWithLabels(ctx, texts, labels)
	if err != nil {
		p.logger.Error("ZSC failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, fmt.Errorf("classifying texts: %w", err)
	}

	// Convert pipelines.ClassificationResult to classification.Classification
	results := make([][]Classification, len(pipelineResults))
	for i, result := range pipelineResults {
		// Build sorted list of (label, score) pairs from Scores map
		classifications := make([]Classification, 0, len(result.Scores))
		for label, score := range result.Scores {
			classifications = append(classifications, Classification{
				Label: label,
				Score: score,
			})
		}
		// Sort by score descending
		sort.Slice(classifications, func(a, b int) bool {
			return classifications[a].Score > classifications[b].Score
		})
		results[i] = classifications
	}

	p.logger.Debug("ZSC completed",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_texts", len(texts)),
		zap.Int("num_results", len(results)))

	return results, nil
}

// MultiLabelClassify classifies texts allowing multiple labels per text.
// Note: Multi-label mode must be configured at pipeline creation time.
func (p *PooledClassifier) MultiLabelClassify(ctx context.Context, texts []string, labels []string) ([][]Classification, error) {
	// For multi-label, we use the same ClassifyWithLabels method
	// The pipeline handles multi-label mode based on its configuration
	results, err := p.ClassifyWithHypothesis(ctx, texts, labels, p.config.HypothesisTemplate)
	if err != nil {
		return nil, err
	}

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
func (p *PooledClassifier) Close() error {
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

// Config returns the classifier configuration.
func (p *PooledClassifier) Config() Config {
	return p.config
}

// LoadZSCConfig loads the zero-shot classification configuration from the model directory.
func LoadZSCConfig(modelPath string) (*Config, error) {
	configPath := filepath.Join(modelPath, "zsc_config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("reading zsc_config.json: %w", err)
	}

	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("parsing zsc_config.json: %w", err)
	}

	// Set defaults for empty values
	if config.HypothesisTemplate == "" {
		config.HypothesisTemplate = DefaultHypothesisTemplate
	}

	return &config, nil
}
