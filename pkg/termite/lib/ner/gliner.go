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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure PooledGLiNER implements both Model and Recognizer interfaces.
var (
	_ Model      = (*PooledGLiNER)(nil)
	_ Recognizer = (*PooledGLiNER)(nil)
)

// =============================================================================
// GLiNER Config Types
// =============================================================================

// GLiNERModelType represents the type of GLiNER model architecture.
type GLiNERModelType string

const (
	// GLiNERModelUniEncoder is the standard GLiNER model, best for <30 entity types.
	GLiNERModelUniEncoder GLiNERModelType = "uniencoder"
	// GLiNERModelBiEncoder is optimized for 50-200+ entity types with pre-computed embeddings.
	GLiNERModelBiEncoder GLiNERModelType = "biencoder"
	// GLiNERModelTokenLevel is optimized for extracting long entity spans (multi-sentence).
	GLiNERModelTokenLevel GLiNERModelType = "token_level"
	// GLiNERModelMultiTask supports multiple tasks: NER, classification, QA, relation extraction.
	GLiNERModelMultiTask GLiNERModelType = "multitask"
)

// GLiNERConfig holds configuration for GLiNER models.
type GLiNERConfig struct {
	// MaxWidth is the maximum entity span width in tokens.
	MaxWidth int `json:"max_width"`
	// DefaultLabels are the entity labels to use if none specified.
	DefaultLabels []string `json:"default_labels"`
	// Threshold is the score threshold for entity detection (0.0-1.0).
	Threshold float32 `json:"threshold"`
	// FlatNER if true, don't allow nested/overlapping entities (default: true).
	FlatNER bool `json:"flat_ner"`
	// MultiLabel if true, allow entities to have multiple labels (default: false).
	MultiLabel bool `json:"multi_label"`
	// ModelType indicates the GLiNER architecture variant.
	ModelType GLiNERModelType `json:"model_type,omitempty"`
	// RelationLabels are default relation types for relationship extraction.
	RelationLabels []string `json:"relation_labels,omitempty"`
	// RelationThreshold is the score threshold for relationship detection (0.0-1.0).
	RelationThreshold float32 `json:"relation_threshold,omitempty"`
}

// DefaultGLiNERConfig returns a GLiNERConfig with sensible defaults.
func DefaultGLiNERConfig() GLiNERConfig {
	return GLiNERConfig{
		MaxWidth:          12, // default max entity span width
		DefaultLabels:     []string{"person", "organization", "location", "date", "product"},
		Threshold:         0.5,
		FlatNER:           true,  // default to flat NER (no overlapping entities)
		MultiLabel:        false, // default to single label per entity
		ModelType:         GLiNERModelUniEncoder,
		RelationThreshold: 0.5,
	}
}

// LoadGLiNERConfig loads GLiNER configuration from the model directory.
// Returns the default config if gliner_config.json is not found.
func LoadGLiNERConfig(modelPath string) GLiNERConfig {
	config := DefaultGLiNERConfig()

	configPath := filepath.Join(modelPath, "gliner_config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return config
	}

	if err := json.Unmarshal(data, &config); err != nil {
		return DefaultGLiNERConfig()
	}

	// Detect model type from model name if not specified in config
	if config.ModelType == "" {
		config.ModelType = detectGLiNERModelType(modelPath)
	}

	return config
}

// detectGLiNERModelType attempts to detect the model type from the model name.
func detectGLiNERModelType(modelPath string) GLiNERModelType {
	modelName := strings.ToLower(filepath.Base(modelPath))

	switch {
	case strings.Contains(modelName, "multitask"):
		return GLiNERModelMultiTask
	case strings.Contains(modelName, "biencoder") || strings.Contains(modelName, "bi-"):
		return GLiNERModelBiEncoder
	case strings.Contains(modelName, "token") || strings.Contains(modelName, "large"):
		// Token-level models are often the larger variants
		return GLiNERModelTokenLevel
	default:
		return GLiNERModelUniEncoder
	}
}

// IsGLiNERModel checks if the model path contains a GLiNER model
// by looking for gliner_config.json or gliner in the model name.
func IsGLiNERModel(modelPath string) bool {
	// Check for gliner_config.json
	configPath := filepath.Join(modelPath, "gliner_config.json")
	if _, err := os.Stat(configPath); err == nil {
		return true
	}

	// Check if model name contains "gliner"
	modelName := strings.ToLower(filepath.Base(modelPath))
	return strings.Contains(modelName, "gliner")
}

// =============================================================================
// Pooled GLiNER Implementation
// =============================================================================

// PooledGLiNERConfig holds configuration for creating a PooledGLiNER.
type PooledGLiNERConfig struct {
	// ModelPath is the path to the model directory.
	ModelPath string

	// PoolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
	PoolSize int

	// Quantized if true, use quantized model (model_quantized.onnx).
	Quantized bool

	// ModelBackends specifies which backends this model supports (nil = all backends).
	ModelBackends []string

	// Logger for logging (nil = no logging).
	Logger *zap.Logger
}

// PooledGLiNER manages multiple GLiNER pipelines for concurrent zero-shot NER.
// Uses the pipelines.GLiNERPipeline for inference.
type PooledGLiNER struct {
	pipelineList   []*pipelines.GLiNERPipeline
	sem            *semaphore.Weighted
	nextPipeline   atomic.Uint64
	logger         *zap.Logger
	poolSize       int
	backendType    backends.BackendType
	config         GLiNERConfig
	labels         []string // Default labels from config
	relationLabels []string // Default relation labels (if multitask model)
}

// NewPooledGLiNER creates a new pooled GLiNER model with session management.
func NewPooledGLiNER(
	cfg PooledGLiNERConfig,
	sessionManager *backends.SessionManager,
) (*PooledGLiNER, backends.BackendType, error) {
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
		poolSize = min(runtime.NumCPU(), 4)
	}

	// Load GLiNER config
	config := LoadGLiNERConfig(cfg.ModelPath)

	logger.Info("Initializing pooled GLiNER",
		zap.String("modelPath", cfg.ModelPath),
		zap.Int("poolSize", poolSize),
		zap.Bool("quantized", cfg.Quantized))

	logger.Info("Loaded GLiNER config",
		zap.Int("max_width", config.MaxWidth),
		zap.Strings("default_labels", config.DefaultLabels),
		zap.Float32("threshold", config.Threshold),
		zap.Bool("flat_ner", config.FlatNER),
		zap.Bool("multi_label", config.MultiLabel),
		zap.String("model_type", string(config.ModelType)))

	// Build loader options
	loaderOpts := []pipelines.GLiNERLoaderOption{
		pipelines.WithGLiNERThreshold(config.Threshold),
		pipelines.WithGLiNERMaxWidth(config.MaxWidth),
		pipelines.WithGLiNERFlatNER(config.FlatNER),
		pipelines.WithGLiNERMultiLabel(config.MultiLabel),
		pipelines.WithGLiNERLabels(config.DefaultLabels),
		pipelines.WithGLiNERQuantized(cfg.Quantized),
	}

	// Create N GLiNER pipelines
	pipelinesList := make([]*pipelines.GLiNERPipeline, poolSize)
	var backendType backends.BackendType

	for i := 0; i < poolSize; i++ {
		pipeline, bt, err := pipelines.LoadGLiNERPipeline(
			cfg.ModelPath,
			sessionManager,
			cfg.ModelBackends,
			loaderOpts...,
		)
		if err != nil {
			// Clean up already-created pipelines
			for j := 0; j < i; j++ {
				if pipelinesList[j] != nil {
					pipelinesList[j].Close()
				}
			}
			logger.Error("Failed to create GLiNER pipeline",
				zap.Int("index", i),
				zap.Error(err))
			return nil, "", fmt.Errorf("creating GLiNER pipeline %d: %w", i, err)
		}
		pipelinesList[i] = pipeline
		backendType = bt
		logger.Debug("Created GLiNER pipeline", zap.Int("index", i))
	}

	logger.Info("Successfully created pooled GLiNER pipelines",
		zap.Int("count", poolSize),
		zap.String("backend", string(backendType)),
		zap.Strings("default_labels", config.DefaultLabels))

	return &PooledGLiNER{
		pipelineList:   pipelinesList,
		sem:            semaphore.NewWeighted(int64(poolSize)),
		logger:         logger,
		poolSize:       poolSize,
		backendType:    backendType,
		config:         config,
		labels:         config.DefaultLabels,
		relationLabels: config.RelationLabels,
	}, backendType, nil
}

// BackendType returns the backend type used by this GLiNER model.
func (p *PooledGLiNER) BackendType() backends.BackendType {
	return p.backendType
}

// Config returns the GLiNER configuration.
func (p *PooledGLiNER) Config() GLiNERConfig {
	return p.config
}

// =============================================================================
// Model Interface Implementation
// =============================================================================

// Recognize extracts named entities using default labels.
// Implements ner.Model interface.
func (p *PooledGLiNER) Recognize(ctx context.Context, texts []string) ([][]Entity, error) {
	return p.RecognizeWithLabels(ctx, texts, p.labels)
}

// Close releases resources.
// Implements ner.Model interface.
func (p *PooledGLiNER) Close() error {
	p.logger.Info("Closing PooledGLiNER")
	for _, pipeline := range p.pipelineList {
		if pipeline != nil {
			pipeline.Close()
		}
	}
	p.pipelineList = nil
	return nil
}

// =============================================================================
// Recognizer Interface Implementation
// =============================================================================

// RecognizeWithLabels extracts entities of the specified types (zero-shot NER).
// This is the key feature of GLiNER - it can extract any entity type without retraining.
// Implements ner.Recognizer interface.
func (p *PooledGLiNER) RecognizeWithLabels(ctx context.Context, texts []string, labels []string) ([][]Entity, error) {
	if len(texts) == 0 {
		return [][]Entity{}, nil
	}

	if len(labels) == 0 {
		labels = p.labels
	}

	// Acquire semaphore slot (blocks if all pipelines busy)
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}
	defer p.sem.Release(1)

	// Round-robin pipeline selection
	idx := int(p.nextPipeline.Add(1) % uint64(p.poolSize))
	pipeline := p.pipelineList[idx]

	p.logger.Debug("Starting GLiNER recognition",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_texts", len(texts)),
		zap.Strings("labels", labels))

	// Run the GLiNER pipeline with the specified labels
	output, err := pipeline.RecognizeWithLabels(ctx, texts, labels)
	if err != nil {
		p.logger.Error("GLiNER recognition failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, fmt.Errorf("running GLiNER pipeline: %w", err)
	}

	// Convert pipeline output to our Entity type
	results := make([][]Entity, len(texts))
	for i, entities := range output.Entities {
		results[i] = convertGLiNEREntities(entities)
	}

	p.logger.Debug("GLiNER recognition completed",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_texts", len(texts)),
		zap.Int("total_entities", countGLiNEREntities(results)))

	return results, nil
}

// Labels returns the default entity labels this model uses.
// Implements ner.Recognizer interface.
func (p *PooledGLiNER) Labels() []string {
	return p.labels
}

// ExtractRelations extracts both entities and relationships between them.
// This requires a multitask GLiNER model that supports relation extraction.
// Returns ErrNotSupported if the model doesn't have the "relations" capability.
// Implements ner.Recognizer interface.
func (p *PooledGLiNER) ExtractRelations(ctx context.Context, texts []string, entityLabels []string, relationLabels []string) ([][]Entity, [][]Relation, error) {
	if len(texts) == 0 {
		return [][]Entity{}, [][]Relation{}, nil
	}

	if !p.SupportsRelationExtraction() {
		return nil, nil, ErrNotSupported
	}

	if len(entityLabels) == 0 {
		entityLabels = p.labels
	}
	if len(relationLabels) == 0 {
		relationLabels = p.relationLabels
	}

	// Acquire semaphore slot
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}
	defer p.sem.Release(1)

	// Round-robin pipeline selection
	idx := int(p.nextPipeline.Add(1) % uint64(p.poolSize))
	pipeline := p.pipelineList[idx]

	p.logger.Debug("Starting GLiNER recognition with relations",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_texts", len(texts)),
		zap.Strings("entity_labels", entityLabels),
		zap.Strings("relation_labels", relationLabels))

	// For now, relation extraction is not implemented in the new pipeline
	// Just extract entities and return empty relations
	output, err := pipeline.RecognizeWithLabels(ctx, texts, entityLabels)
	if err != nil {
		p.logger.Error("GLiNER recognition with relations failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, nil, fmt.Errorf("running GLiNER with relations: %w", err)
	}

	// Convert entities
	entities := make([][]Entity, len(texts))
	for i, ents := range output.Entities {
		entities[i] = convertGLiNEREntities(ents)
	}

	// Return empty relations (not yet implemented)
	relations := make([][]Relation, len(texts))

	p.logger.Debug("GLiNER recognition with relations completed",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_texts", len(texts)),
		zap.Int("total_entities", countGLiNEREntities(entities)),
		zap.Int("total_relations", countGLiNERRelations(relations)))

	return entities, relations, nil
}

// ExtractAnswers performs extractive question answering.
// Given questions and contexts, extracts answer spans from the contexts.
// Returns ErrNotSupported if the model doesn't have the "answers" capability.
// Implements ner.Recognizer interface.
func (p *PooledGLiNER) ExtractAnswers(ctx context.Context, questions []string, contexts []string) ([]Answer, error) {
	if len(questions) == 0 || len(contexts) == 0 {
		return []Answer{}, nil
	}

	if len(questions) != len(contexts) {
		return nil, fmt.Errorf("questions and contexts must have the same length: got %d questions and %d contexts", len(questions), len(contexts))
	}

	if !p.SupportsQA() {
		return nil, ErrNotSupported
	}

	// Acquire semaphore slot
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}
	defer p.sem.Release(1)

	// Round-robin pipeline selection
	idx := int(p.nextPipeline.Add(1) % uint64(p.poolSize))
	pipeline := p.pipelineList[idx]

	p.logger.Debug("Starting GLiNER question answering",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_questions", len(questions)))

	// Process each question-context pair
	// GLiNER multitask models treat QA as finding spans that answer the question
	answers := make([]Answer, len(questions))
	for i, question := range questions {
		context := contexts[i]

		// Use the question as the "label" to extract from the context
		// This leverages GLiNER's zero-shot capability to find answer spans
		output, err := pipeline.RecognizeWithLabels(ctx, []string{context}, []string{question})
		if err != nil {
			p.logger.Error("GLiNER QA failed",
				zap.Int("index", i),
				zap.String("question", question),
				zap.Error(err))
			return nil, fmt.Errorf("running GLiNER QA for question %d: %w", i, err)
		}

		// Take the highest-scoring entity as the answer
		if len(output.Entities) > 0 && len(output.Entities[0]) > 0 {
			// Find best entity by score
			best := output.Entities[0][0]
			for _, e := range output.Entities[0][1:] {
				if e.Score > best.Score {
					best = e
				}
			}
			answers[i] = Answer{
				Text:  best.Text,
				Start: best.Start,
				End:   best.End,
				Score: best.Score,
			}
		} else {
			// No answer found
			answers[i] = Answer{
				Text:  "",
				Start: 0,
				End:   0,
				Score: 0,
			}
		}
	}

	p.logger.Debug("GLiNER QA completed",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_answers", len(answers)))

	return answers, nil
}

// RelationLabels returns the default relation labels this model uses.
// Returns nil if the model doesn't support relation extraction.
// Implements ner.Recognizer interface.
func (p *PooledGLiNER) RelationLabels() []string {
	return p.relationLabels
}

// =============================================================================
// Capability Methods
// =============================================================================

// SupportsRelationExtraction returns true if the model supports relation extraction.
func (p *PooledGLiNER) SupportsRelationExtraction() bool {
	if len(p.pipelineList) == 0 {
		return false
	}
	return p.pipelineList[0].SupportsRelationExtraction()
}

// SupportsQA returns true if the model supports question answering.
func (p *PooledGLiNER) SupportsQA() bool {
	return p.config.ModelType == GLiNERModelMultiTask
}

// =============================================================================
// BiEncoder Label Caching
// =============================================================================

// IsBiEncoder returns true if this is a BiEncoder model that supports label caching.
func (p *PooledGLiNER) IsBiEncoder() bool {
	if len(p.pipelineList) == 0 {
		return false
	}
	return p.pipelineList[0].IsBiEncoder()
}

// PrecomputeLabelEmbeddings precomputes and caches embeddings for the given labels.
// This is useful for BiEncoder models where label embeddings can be computed once
// and reused across many inference calls with the same labels.
//
// For BiEncoder models, this runs the labels through the label encoder to get
// embeddings that can be reused. For UniEncoder models, this is a no-op since
// labels are encoded together with the text.
func (p *PooledGLiNER) PrecomputeLabelEmbeddings(labels []string) error {
	if len(p.pipelineList) == 0 {
		return nil
	}

	// Precompute embeddings for all pipelines in the pool
	// Since each pipeline has its own session, we need to compute for each one
	for i, pipeline := range p.pipelineList {
		if err := pipeline.PrecomputeLabelEmbeddings(labels); err != nil {
			p.logger.Error("Failed to precompute label embeddings",
				zap.Int("pipelineIndex", i),
				zap.Error(err))
			return fmt.Errorf("precomputing label embeddings for pipeline %d: %w", i, err)
		}
	}

	p.logger.Debug("Precomputed label embeddings",
		zap.Int("num_labels", len(labels)),
		zap.Strings("labels", labels))

	return nil
}

// HasCachedLabelEmbeddings returns true if label embeddings are currently cached.
func (p *PooledGLiNER) HasCachedLabelEmbeddings() bool {
	if len(p.pipelineList) == 0 {
		return false
	}
	return p.pipelineList[0].HasCachedLabelEmbeddings()
}

// CachedLabels returns the list of labels that are currently cached.
func (p *PooledGLiNER) CachedLabels() []string {
	if len(p.pipelineList) == 0 {
		return nil
	}
	return p.pipelineList[0].CachedLabels()
}

// ClearLabelEmbeddingCache clears all cached label embeddings across all pooled pipelines.
func (p *PooledGLiNER) ClearLabelEmbeddingCache() {
	for _, pipeline := range p.pipelineList {
		pipeline.ClearLabelEmbeddingCache()
	}

	p.logger.Debug("Cleared label embedding cache")
}

// =============================================================================
// Helper Functions
// =============================================================================

// convertGLiNEREntities converts pipeline entities to our Entity type.
func convertGLiNEREntities(pipelineEntities []pipelines.GLiNEREntity) []Entity {
	entities := make([]Entity, len(pipelineEntities))
	for i, e := range pipelineEntities {
		entities[i] = Entity{
			Text:  e.Text,
			Label: e.Label,
			Start: e.Start,
			End:   e.End,
			Score: e.Score,
		}
	}
	return entities
}

// convertGLiNERRelations converts pipeline relations to our Relation type.
func convertGLiNERRelations(pipelineRelations []pipelines.GLiNERRelation) []Relation {
	relations := make([]Relation, len(pipelineRelations))
	for i, r := range pipelineRelations {
		relations[i] = Relation{
			HeadEntity: Entity{
				Text:  r.HeadEntity.Text,
				Label: r.HeadEntity.Label,
				Start: r.HeadEntity.Start,
				End:   r.HeadEntity.End,
				Score: r.HeadEntity.Score,
			},
			TailEntity: Entity{
				Text:  r.TailEntity.Text,
				Label: r.TailEntity.Label,
				Start: r.TailEntity.Start,
				End:   r.TailEntity.End,
				Score: r.TailEntity.Score,
			},
			Label: r.Label,
			Score: r.Score,
		}
	}
	return relations
}

// countGLiNEREntities counts the total number of entities across all texts.
func countGLiNEREntities(results [][]Entity) int {
	count := 0
	for _, entities := range results {
		count += len(entities)
	}
	return count
}

// countGLiNERRelations counts the total number of relations across all texts.
func countGLiNERRelations(relations [][]Relation) int {
	count := 0
	for _, rels := range relations {
		count += len(rels)
	}
	return count
}
