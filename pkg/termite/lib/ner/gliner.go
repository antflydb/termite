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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
	"go.uber.org/zap"
)

// Ensure HugotGLiNER implements the GLiNERModel and RelationExtractionModel interfaces
var _ GLiNERModel = (*HugotGLiNER)(nil)
var _ RelationExtractionModel = (*HugotGLiNER)(nil)

// GLiNERModelType represents the type of GLiNER model architecture
type GLiNERModelType string

const (
	// GLiNERModelUniEncoder is the standard GLiNER model, best for <30 entity types
	GLiNERModelUniEncoder GLiNERModelType = "uniencoder"
	// GLiNERModelBiEncoder is optimized for 50-200+ entity types with pre-computed embeddings
	GLiNERModelBiEncoder GLiNERModelType = "biencoder"
	// GLiNERModelTokenLevel is optimized for extracting long entity spans (multi-sentence)
	GLiNERModelTokenLevel GLiNERModelType = "token_level"
	// GLiNERModelMultiTask supports multiple tasks: NER, classification, QA, relation extraction
	GLiNERModelMultiTask GLiNERModelType = "multitask"
)

// GLiNERConfig holds configuration for GLiNER models
type GLiNERConfig struct {
	// MaxWidth is the maximum entity span width in tokens
	MaxWidth int `json:"max_width"`
	// DefaultLabels are the entity labels to use if none specified
	DefaultLabels []string `json:"default_labels"`
	// Threshold is the score threshold for entity detection (0.0-1.0)
	Threshold float32 `json:"threshold"`
	// FlatNER if true, don't allow nested/overlapping entities (default: true)
	FlatNER bool `json:"flat_ner"`
	// MultiLabel if true, allow entities to have multiple labels (default: false)
	MultiLabel bool `json:"multi_label"`
	// ModelType indicates the GLiNER architecture variant
	ModelType GLiNERModelType `json:"model_type,omitempty"`
	// RelationLabels are default relation types for relationship extraction
	RelationLabels []string `json:"relation_labels,omitempty"`
	// RelationThreshold is the score threshold for relationship detection (0.0-1.0)
	RelationThreshold float32 `json:"relation_threshold,omitempty"`
}

// HugotGLiNER implements GLiNERModel using the hugot GLiNER pipeline.
// GLiNER models are zero-shot NER models that can recognize any entity types.
type HugotGLiNER struct {
	session        *khugot.Session
	pipeline       *pipelines.GLiNERPipeline
	logger         *zap.Logger
	sessionShared  bool
	config         GLiNERConfig
	labels         []string // Default labels from config
	relationLabels []string // Default relation labels (if multitask model)
}

// NewHugotGLiNER creates a new GLiNER model using the Hugot ONNX runtime.
func NewHugotGLiNER(modelPath string, quantized bool, logger *zap.Logger) (*HugotGLiNER, error) {
	return NewHugotGLiNERWithSession(modelPath, quantized, nil, logger)
}

// NewHugotGLiNERWithSession creates a new GLiNER model with an optional shared session.
func NewHugotGLiNERWithSession(modelPath string, quantized bool, sharedSession *khugot.Session, logger *zap.Logger) (*HugotGLiNER, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Initializing Hugot GLiNER model",
		zap.String("modelPath", modelPath),
		zap.Bool("quantized", quantized),
		zap.String("backend", hugot.BackendName()))

	// Load GLiNER config if available
	config := GLiNERConfig{
		MaxWidth:          12, // default max entity span width
		DefaultLabels:     []string{"person", "organization", "location", "date", "product"},
		Threshold:         0.5,
		FlatNER:           true,  // default to flat NER (no overlapping entities)
		MultiLabel:        false, // default to single label per entity
		ModelType:         GLiNERModelUniEncoder,
		RelationThreshold: 0.5,
	}
	configPath := filepath.Join(modelPath, "gliner_config.json")
	if data, err := os.ReadFile(configPath); err == nil {
		if err := json.Unmarshal(data, &config); err != nil {
			logger.Warn("Failed to parse GLiNER config", zap.Error(err))
		} else {
			logger.Info("Loaded GLiNER config",
				zap.Int("max_width", config.MaxWidth),
				zap.Strings("default_labels", config.DefaultLabels),
				zap.Float32("threshold", config.Threshold),
				zap.Bool("flat_ner", config.FlatNER),
				zap.Bool("multi_label", config.MultiLabel),
				zap.String("model_type", string(config.ModelType)))
		}
	}

	// Detect model type from model name if not specified in config
	if config.ModelType == "" {
		config.ModelType = detectGLiNERModelType(modelPath)
	}

	// Use shared session or create a new one
	session, err := hugot.NewSessionOrUseExisting(sharedSession)
	if err != nil {
		logger.Error("Failed to create Hugot session", zap.Error(err))
		return nil, fmt.Errorf("creating hugot session: %w", err)
	}
	sessionShared := (sharedSession != nil)

	// Determine which ONNX file to use
	onnxFilename := "model.onnx"
	if quantized {
		onnxFilename = "model_quantized.onnx"
	}

	// Create GLiNER pipeline with proper configuration
	pipelineName := fmt.Sprintf("gliner:%s:%s", modelPath, onnxFilename)
	pipelineOptions := []khugot.GLiNEROption{
		pipelines.WithGLiNERLabels(config.DefaultLabels),
		pipelines.WithGLiNERMaxWidth(config.MaxWidth),
		pipelines.WithGLiNERThreshold(config.Threshold),
	}

	// Apply FlatNER option based on config
	if config.FlatNER {
		pipelineOptions = append(pipelineOptions, pipelines.WithGLiNERFlatNER())
	}

	// Apply MultiLabel option based on config
	if config.MultiLabel {
		pipelineOptions = append(pipelineOptions, pipelines.WithGLiNERMultiLabel())
	}

	pipelineConfig := khugot.GLiNERConfig{
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
		logger.Error("Failed to create GLiNER pipeline", zap.Error(err))
		return nil, fmt.Errorf("creating GLiNER pipeline: %w", err)
	}

	logger.Info("GLiNER model initialization complete",
		zap.Strings("default_labels", config.DefaultLabels),
		zap.Int("max_width", config.MaxWidth),
		zap.String("model_type", string(config.ModelType)),
		zap.Bool("flat_ner", config.FlatNER),
		zap.Bool("multi_label", config.MultiLabel))

	return &HugotGLiNER{
		session:        session,
		pipeline:       pipeline,
		logger:         logger,
		sessionShared:  sessionShared,
		config:         config,
		labels:         config.DefaultLabels,
		relationLabels: config.RelationLabels,
	}, nil
}

// NewHugotGLiNERWithSessionManager creates a new GLiNER model using a SessionManager.
// The SessionManager handles backend selection based on priority and model compatibility.
func NewHugotGLiNERWithSessionManager(modelPath string, quantized bool, sessionManager *hugot.SessionManager, logger *zap.Logger) (*HugotGLiNER, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	if sessionManager == nil {
		// Fall back to creating a new session
		return NewHugotGLiNERWithSession(modelPath, quantized, nil, logger)
	}

	logger.Info("Initializing Hugot GLiNER model with SessionManager",
		zap.String("modelPath", modelPath),
		zap.Bool("quantized", quantized))

	// Load GLiNER config if available
	config := GLiNERConfig{
		MaxWidth:          12,
		DefaultLabels:     []string{"person", "organization", "location", "date", "product"},
		Threshold:         0.5,
		FlatNER:           true,
		MultiLabel:        false,
		ModelType:         GLiNERModelUniEncoder,
		RelationThreshold: 0.5,
	}
	configPath := filepath.Join(modelPath, "gliner_config.json")
	if data, err := os.ReadFile(configPath); err == nil {
		if err := json.Unmarshal(data, &config); err != nil {
			logger.Warn("Failed to parse GLiNER config", zap.Error(err))
		}
	}

	if config.ModelType == "" {
		config.ModelType = detectGLiNERModelType(modelPath)
	}

	// Get session from SessionManager
	session, _, err := sessionManager.GetSessionForModel(nil)
	if err != nil {
		logger.Error("Failed to get session from SessionManager", zap.Error(err))
		return nil, fmt.Errorf("getting session from SessionManager: %w", err)
	}

	// Determine which ONNX file to use
	onnxFilename := "model.onnx"
	if quantized {
		onnxFilename = "model_quantized.onnx"
	}

	// Create GLiNER pipeline
	pipelineName := fmt.Sprintf("gliner:%s:%s", modelPath, onnxFilename)
	pipelineOptions := []khugot.GLiNEROption{
		pipelines.WithGLiNERLabels(config.DefaultLabels),
		pipelines.WithGLiNERMaxWidth(config.MaxWidth),
		pipelines.WithGLiNERThreshold(config.Threshold),
	}

	if config.FlatNER {
		pipelineOptions = append(pipelineOptions, pipelines.WithGLiNERFlatNER())
	}
	if config.MultiLabel {
		pipelineOptions = append(pipelineOptions, pipelines.WithGLiNERMultiLabel())
	}

	pipelineConfig := khugot.GLiNERConfig{
		ModelPath:    modelPath,
		OnnxFilename: onnxFilename,
		Name:         pipelineName,
		Options:      pipelineOptions,
	}

	pipeline, err := khugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		logger.Error("Failed to create GLiNER pipeline", zap.Error(err))
		return nil, fmt.Errorf("creating GLiNER pipeline: %w", err)
	}

	logger.Info("GLiNER model initialization complete",
		zap.Strings("default_labels", config.DefaultLabels),
		zap.Int("max_width", config.MaxWidth),
		zap.String("model_type", string(config.ModelType)))

	return &HugotGLiNER{
		session:        session,
		pipeline:       pipeline,
		logger:         logger,
		sessionShared:  true, // SessionManager owns the session
		config:         config,
		labels:         config.DefaultLabels,
		relationLabels: config.RelationLabels,
	}, nil
}

// Recognize extracts named entities using default labels.
func (g *HugotGLiNER) Recognize(ctx context.Context, texts []string) ([][]Entity, error) {
	return g.RecognizeWithLabels(ctx, texts, g.labels)
}

// RecognizeWithLabels extracts entities of the specified types (zero-shot NER).
// This is the key feature of GLiNER - it can extract any entity type without retraining.
func (g *HugotGLiNER) RecognizeWithLabels(ctx context.Context, texts []string, labels []string) ([][]Entity, error) {
	if len(texts) == 0 {
		return [][]Entity{}, nil
	}

	if len(labels) == 0 {
		labels = g.labels
	}

	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	g.logger.Debug("Starting GLiNER recognition",
		zap.Int("num_texts", len(texts)),
		zap.Strings("labels", labels))

	// Run the GLiNER pipeline with the specified labels
	output, err := g.pipeline.RunPipelineWithLabels(texts, labels)
	if err != nil {
		g.logger.Error("GLiNER recognition failed", zap.Error(err))
		return nil, fmt.Errorf("running GLiNER pipeline: %w", err)
	}

	// Convert pipeline output to our Entity type
	results := make([][]Entity, len(texts))
	for i, entities := range output.Entities {
		results[i] = convertGLiNEREntities(entities)
	}

	g.logger.Debug("GLiNER recognition completed",
		zap.Int("num_texts", len(texts)),
		zap.Int("total_entities", countEntities(results)))

	return results, nil
}

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

// Labels returns the default entity labels this model uses.
func (g *HugotGLiNER) Labels() []string {
	return g.labels
}

// Close releases resources.
func (g *HugotGLiNER) Close() error {
	if g.session != nil && !g.sessionShared {
		g.logger.Info("Destroying Hugot session (owned by this GLiNER model)")
		g.session.Destroy()
	}
	return nil
}

// Config returns the GLiNER configuration.
func (g *HugotGLiNER) Config() GLiNERConfig {
	return g.config
}

// ModelType returns the detected model type.
func (g *HugotGLiNER) ModelType() GLiNERModelType {
	return g.config.ModelType
}

// IsTokenLevel returns true if this is a token-level model optimized for long spans.
func (g *HugotGLiNER) IsTokenLevel() bool {
	return g.config.ModelType == GLiNERModelTokenLevel
}

// IsMultiTask returns true if this model supports multiple tasks (NER, relations, etc.).
func (g *HugotGLiNER) IsMultiTask() bool {
	return g.config.ModelType == GLiNERModelMultiTask
}

// RelationLabels returns the default relation labels (for multitask models).
func (g *HugotGLiNER) RelationLabels() []string {
	return g.relationLabels
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

// IsTokenLevelModel checks if the model is a token-level variant.
// Token-level models are better for extracting long entity spans.
func IsTokenLevelModel(modelPath string) bool {
	modelName := strings.ToLower(filepath.Base(modelPath))
	return strings.Contains(modelName, "token") ||
		strings.Contains(modelName, "multitask") ||
		strings.Contains(modelName, "large")
}

// IsMultiTaskModel checks if the model supports multiple tasks.
func IsMultiTaskModel(modelPath string) bool {
	modelName := strings.ToLower(filepath.Base(modelPath))
	return strings.Contains(modelName, "multitask") ||
		strings.Contains(modelName, "relex")
}

// =============================================================================
// BiEncoder Label Caching
// =============================================================================

// PrecomputeLabelEmbeddings precomputes and caches embeddings for the given labels.
// This is useful for BiEncoder models where label embeddings can be computed once
// and reused across many inference calls with the same labels.
func (g *HugotGLiNER) PrecomputeLabelEmbeddings(labels []string) error {
	return g.pipeline.PrecomputeLabelEmbeddings(labels)
}

// HasCachedLabelEmbeddings returns true if label embeddings are cached.
func (g *HugotGLiNER) HasCachedLabelEmbeddings() bool {
	return g.pipeline.HasCachedLabelEmbeddings()
}

// ClearLabelEmbeddingCache clears the cached label embeddings.
func (g *HugotGLiNER) ClearLabelEmbeddingCache() {
	g.pipeline.ClearLabelEmbeddingCache()
}

// RecognizeWithCachedEmbeddings runs inference using cached label embeddings.
// This is faster than RecognizeWithLabels when the same labels are used repeatedly.
func (g *HugotGLiNER) RecognizeWithCachedEmbeddings(ctx context.Context, texts []string, labels []string) ([][]Entity, error) {
	if len(texts) == 0 {
		return [][]Entity{}, nil
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	output, err := g.pipeline.RunWithCachedEmbeddings(texts, labels)
	if err != nil {
		return nil, fmt.Errorf("running GLiNER with cached embeddings: %w", err)
	}

	results := make([][]Entity, len(texts))
	for i, entities := range output.Entities {
		results[i] = convertGLiNEREntities(entities)
	}

	return results, nil
}

// =============================================================================
// Sequence Packing
// =============================================================================

// RecognizeWithPacking runs inference with sequence packing optimization.
// This combines multiple short sequences into single transformer passes for better throughput.
func (g *HugotGLiNER) RecognizeWithPacking(ctx context.Context, texts []string, labels []string) ([][]Entity, error) {
	if len(texts) == 0 {
		return [][]Entity{}, nil
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	output, err := g.pipeline.RunPipelineWithPacking(texts, labels)
	if err != nil {
		return nil, fmt.Errorf("running GLiNER with packing: %w", err)
	}

	results := make([][]Entity, len(texts))
	for i, entities := range output.Entities {
		results[i] = convertGLiNEREntities(entities)
	}

	return results, nil
}

// =============================================================================
// Relation Extraction
// =============================================================================

// RecognizeWithRelations extracts both entities and relationships between them.
// This requires a multitask GLiNER model that supports relation extraction.
func (g *HugotGLiNER) RecognizeWithRelations(ctx context.Context, texts []string, entityLabels []string, relationLabels []string) ([][]Entity, [][]Relation, error) {
	if len(texts) == 0 {
		return [][]Entity{}, [][]Relation{}, nil
	}

	if len(entityLabels) == 0 {
		entityLabels = g.labels
	}
	if len(relationLabels) == 0 {
		relationLabels = g.relationLabels
	}

	select {
	case <-ctx.Done():
		return nil, nil, ctx.Err()
	default:
	}

	g.logger.Debug("Starting GLiNER recognition with relations",
		zap.Int("num_texts", len(texts)),
		zap.Strings("entity_labels", entityLabels),
		zap.Strings("relation_labels", relationLabels))

	output, err := g.pipeline.RunPipelineWithRelations(texts, entityLabels, relationLabels)
	if err != nil {
		g.logger.Error("GLiNER recognition with relations failed", zap.Error(err))
		return nil, nil, fmt.Errorf("running GLiNER with relations: %w", err)
	}

	// Convert entities
	entities := make([][]Entity, len(texts))
	for i, ents := range output.Entities {
		entities[i] = convertGLiNEREntities(ents)
	}

	// Convert relations
	relations := make([][]Relation, len(texts))
	for i, rels := range output.Relations {
		relations[i] = convertGLiNERRelations(rels)
	}

	g.logger.Debug("GLiNER recognition with relations completed",
		zap.Int("num_texts", len(texts)),
		zap.Int("total_entities", countEntities(entities)),
		zap.Int("total_relations", countRelations(relations)))

	return entities, relations, nil
}

// SupportsRelationExtraction returns true if the model supports relation extraction.
func (g *HugotGLiNER) SupportsRelationExtraction() bool {
	return g.pipeline.SupportsRelationExtraction()
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

// countRelations counts the total number of relations across all texts.
func countRelations(relations [][]Relation) int {
	count := 0
	for _, rels := range relations {
		count += len(rels)
	}
	return count
}
