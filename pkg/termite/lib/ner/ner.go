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

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// ErrNotSupported is returned when a model doesn't support a particular operation.
// Check capabilities before calling methods to avoid this error.
var ErrNotSupported = errors.New("operation not supported by this model")

// Entity represents a named entity extracted from text.
type Entity struct {
	// Text is the entity text (e.g., "John Smith")
	Text string `json:"text"`
	// Label is the entity type (e.g., "PER", "ORG", "LOC", "MISC")
	Label string `json:"label"`
	// Start is the character offset where the entity begins
	Start int `json:"start"`
	// End is the character offset where the entity ends (exclusive)
	End int `json:"end"`
	// Score is the confidence score (0.0 to 1.0)
	Score float32 `json:"score"`
}

// Model defines the interface for Named Entity Recognition models.
type Model interface {
	// Recognize extracts named entities from the given texts.
	// Returns a slice of entities for each input text.
	Recognize(ctx context.Context, texts []string) ([][]Entity, error)

	// Close releases any resources held by the model.
	Close() error
}

// Relation represents a relationship between two entities.
type Relation struct {
	// HeadEntity is the source entity in the relationship
	HeadEntity Entity `json:"head"`
	// TailEntity is the target entity in the relationship
	TailEntity Entity `json:"tail"`
	// Label is the relationship type (e.g., "founded", "works_at", "located_in")
	Label string `json:"label"`
	// Score is the model's confidence in this relationship (0.0-1.0)
	Score float32 `json:"score"`
}

// Answer represents an extracted answer span from question answering.
type Answer struct {
	// Text is the answer text extracted from the context
	Text string `json:"text"`
	// Start is the character offset where the answer begins in the context
	Start int `json:"start"`
	// End is the character offset where the answer ends (exclusive)
	End int `json:"end"`
	// Score is the model's confidence in this answer (0.0-1.0)
	Score float32 `json:"score"`
}

// Recognizer extends Model with zero-shot NER and extraction capabilities.
// Models implementing this interface can recognize any entity types specified
// at inference time without requiring model retraining (e.g., GLiNER, REBEL).
//
// Not all models support all methods. Use capabilities to check support:
//   - "labels": Supports RecognizeWithLabels
//   - "zeroshot": Supports arbitrary labels at inference time
//   - "relations": Supports ExtractRelations
//   - "answers": Supports ExtractAnswers
//
// Methods return ErrNotSupported if the model lacks the required capability.
type Recognizer interface {
	Model

	// RecognizeWithLabels extracts entities of the specified types.
	// For zero-shot models like GLiNER, labels can be arbitrary entity types.
	// For traditional NER models, labels should match the trained entity types.
	RecognizeWithLabels(ctx context.Context, texts []string, labels []string) ([][]Entity, error)

	// Labels returns the default entity labels this model uses.
	Labels() []string

	// ExtractRelations extracts both entities and relationships between them.
	// Returns ErrNotSupported if the model doesn't have the "relations" capability.
	ExtractRelations(ctx context.Context, texts []string, entityLabels []string, relationLabels []string) ([][]Entity, [][]Relation, error)

	// ExtractAnswers performs extractive question answering.
	// Given questions and contexts, extracts answer spans from the contexts.
	// Returns ErrNotSupported if the model doesn't have the "answers" capability.
	ExtractAnswers(ctx context.Context, questions []string, contexts []string) ([]Answer, error)

	// RelationLabels returns the default relation labels this model uses.
	// Returns nil if the model doesn't support relation extraction.
	RelationLabels() []string
}

// Ensure PooledNER implements the Model interface
var _ Model = (*PooledNER)(nil)

// PooledNERConfig holds configuration for creating a PooledNER.
type PooledNERConfig struct {
	// ModelPath is the path to the model directory
	ModelPath string

	// PoolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count)
	PoolSize int

	// ModelBackends specifies which backends this model supports (nil = all backends)
	ModelBackends []string

	// Logger for logging (nil = no logging)
	Logger *zap.Logger
}

// PooledNER manages multiple NERPipeline instances for concurrent NER.
// Uses the new backends package (go-huggingface + gomlx/onnxruntime) instead of hugot.
type PooledNER struct {
	pipelines    []*pipelines.NERPipeline
	sem          *semaphore.Weighted
	nextPipeline atomic.Uint64
	logger       *zap.Logger
	poolSize     int
	backendType  backends.BackendType
}

// NewPooledNER creates a new NERPipeline-based pooled NER model.
// This is the new implementation using go-huggingface tokenizers and the backends package.
func NewPooledNER(
	cfg PooledNERConfig,
	sessionManager *backends.SessionManager,
) (*PooledNER, backends.BackendType, error) {
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

	logger.Info("Initializing pooled NER",
		zap.String("modelPath", cfg.ModelPath),
		zap.Int("poolSize", poolSize))

	// Create N NER pipelines
	pipelinesList := make([]*pipelines.NERPipeline, poolSize)
	var backendUsed backends.BackendType

	for i := 0; i < poolSize; i++ {
		pipeline, bt, err := pipelines.LoadNERPipeline(
			cfg.ModelPath,
			sessionManager,
			cfg.ModelBackends,
		)
		if err != nil {
			// Clean up already-created pipelines
			for j := 0; j < i; j++ {
				if pipelinesList[j] != nil {
					_ = pipelinesList[j].Close()
				}
			}
			logger.Error("Failed to create NER pipeline",
				zap.Int("index", i),
				zap.Error(err))
			return nil, "", fmt.Errorf("creating NER pipeline %d: %w", i, err)
		}
		pipelinesList[i] = pipeline
		backendUsed = bt
		logger.Debug("Created NER pipeline", zap.Int("index", i), zap.String("backend", string(bt)))
	}

	logger.Info("Successfully created pooled NER pipelines",
		zap.Int("count", poolSize),
		zap.String("backend", string(backendUsed)))

	return &PooledNER{
		pipelines:   pipelinesList,
		sem:         semaphore.NewWeighted(int64(poolSize)),
		logger:      logger,
		poolSize:    poolSize,
		backendType: backendUsed,
	}, backendUsed, nil
}

// BackendType returns the backend type used by this NER model
func (p *PooledNER) BackendType() backends.BackendType {
	return p.backendType
}

// Recognize extracts named entities from the given texts.
// Thread-safe: uses semaphore to limit concurrent pipeline access.
func (p *PooledNER) Recognize(ctx context.Context, texts []string) ([][]Entity, error) {
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

	// Delegate to NERPipeline.ExtractEntities
	pipelineEntities, err := pipeline.ExtractEntities(ctx, texts)
	if err != nil {
		p.logger.Error("NER failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, fmt.Errorf("extracting entities: %w", err)
	}

	// Convert pipelines.Entity to ner.Entity
	results := make([][]Entity, len(pipelineEntities))
	for i, entities := range pipelineEntities {
		results[i] = make([]Entity, len(entities))
		for j, e := range entities {
			results[i][j] = Entity{
				Text:  e.Text,
				Label: e.Label,
				Start: e.Start,
				End:   e.End,
				Score: e.Score,
			}
		}
	}

	p.logger.Debug("NER completed",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_texts", len(texts)),
		zap.Int("total_entities", countEntities(results)))

	return results, nil
}

// Close releases resources.
func (p *PooledNER) Close() error {
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

// countEntities counts the total number of entities across all texts.
func countEntities(results [][]Entity) int {
	count := 0
	for _, entities := range results {
		count += len(entities)
	}
	return count
}
