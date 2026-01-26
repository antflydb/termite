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

// Ensure PooledREBEL implements the Recognizer interface
var _ Recognizer = (*PooledREBEL)(nil)

// REBELConfig holds configuration for REBEL models.
type REBELConfig struct {
	// ModelID is the original HuggingFace model ID.
	ModelID string `json:"model_id"`
	// ModelType should be "rebel".
	ModelType string `json:"model_type"`
	// MaxLength is the maximum number of tokens to generate.
	MaxLength int `json:"max_length"`
	// NumBeams is the number of beams for beam search.
	NumBeams int `json:"num_beams"`
	// Task is the model task (e.g., "relation_extraction").
	Task string `json:"task"`
	// TripletToken is the token marking triplet boundaries (default: "<triplet>").
	TripletToken string `json:"triplet_token"`
	// SubjectToken is the token marking subject boundaries (default: "<subj>").
	SubjectToken string `json:"subject_token"`
	// ObjectToken is the token marking object boundaries (default: "<obj>").
	ObjectToken string `json:"object_token"`
	// Multilingual indicates if this is a multilingual model.
	Multilingual bool `json:"multilingual"`
}

// DefaultREBELConfig returns the default REBEL configuration.
func DefaultREBELConfig() REBELConfig {
	return REBELConfig{
		MaxLength:    256,
		NumBeams:     3,
		TripletToken: "<triplet>",
		SubjectToken: "<subj>",
		ObjectToken:  "<obj>",
		Task:         "relation_extraction",
	}
}

// LoadREBELConfig loads REBEL configuration from the model directory.
// It looks for rebel_config.json and falls back to defaults if not found.
func LoadREBELConfig(modelPath string) REBELConfig {
	config := DefaultREBELConfig()

	configPath := filepath.Join(modelPath, "rebel_config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return config
	}

	if err := json.Unmarshal(data, &config); err != nil {
		return DefaultREBELConfig()
	}

	// Ensure defaults for missing tokens
	if config.TripletToken == "" {
		config.TripletToken = "<triplet>"
	}
	if config.SubjectToken == "" {
		config.SubjectToken = "<subj>"
	}
	if config.ObjectToken == "" {
		config.ObjectToken = "<obj>"
	}

	return config
}

// IsREBELModel checks if the model path contains a REBEL model.
// It looks for rebel_config.json or encoder/decoder ONNX files typical of REBEL.
func IsREBELModel(modelPath string) bool {
	// Check for rebel_config.json
	configPath := filepath.Join(modelPath, "rebel_config.json")
	if _, err := os.Stat(configPath); err == nil {
		return true
	}

	// Check if model name contains "rebel"
	modelName := strings.ToLower(filepath.Base(modelPath))
	if strings.Contains(modelName, "rebel") {
		// Verify it has the expected files
		encoderPath := filepath.Join(modelPath, "encoder_model.onnx")
		decoderPath := filepath.Join(modelPath, "decoder_model.onnx")
		if _, err := os.Stat(encoderPath); err == nil {
			if _, err := os.Stat(decoderPath); err == nil {
				return true
			}
		}
	}

	return false
}

// triplet represents an extracted relation triplet from REBEL output.
type triplet struct {
	Subject  string
	Object   string
	Relation string
	Score    float32
}

// parseREBELTriplets parses REBEL's generated text into structured triplets.
//
// REBEL output format with special tokens:
// <s><triplet> Subject <subj> Object <obj> relation <triplet> Subject2 <subj> Object2 <obj> relation2 </s>
//
// Fallback format (when tokenizer strips special tokens):
// Subject  Object  relation  Subject2  Object2  relation2
// (elements separated by double spaces)
//
// The score parameter is the generation confidence score, applied to all triplets.
func parseREBELTriplets(text string, config REBELConfig, score float32) []triplet {
	var triplets []triplet

	// Remove start/end tokens
	text = strings.ReplaceAll(text, "<s>", "")
	text = strings.ReplaceAll(text, "</s>", "")
	text = strings.ReplaceAll(text, "<pad>", "")
	text = strings.TrimSpace(text)

	// Check if special tokens are present
	hasSpecialTokens := strings.Contains(text, config.TripletToken) ||
		strings.Contains(text, config.SubjectToken) ||
		strings.Contains(text, config.ObjectToken)

	if hasSpecialTokens {
		// Parse using special tokens
		parts := strings.Split(text, config.TripletToken)
		for _, part := range parts {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			if t := parseTripletPart(part, config, score); t != nil {
				triplets = append(triplets, *t)
			}
		}
	} else {
		// Fallback: parse by double-space separation
		triplets = parseREBELOutputNoTokens(text, score)
	}

	return triplets
}

// parseTripletPart parses a single triplet from REBEL output.
func parseTripletPart(part string, config REBELConfig, score float32) *triplet {
	subjToken := config.SubjectToken
	objToken := config.ObjectToken

	if !strings.Contains(part, subjToken) || !strings.Contains(part, objToken) {
		return nil
	}

	// Split by <subj> first
	subjSplit := strings.SplitN(part, subjToken, 2)
	if len(subjSplit) != 2 {
		return nil
	}
	subject := strings.TrimSpace(subjSplit[0])

	// The rest contains object and relation
	rest := subjSplit[1]

	// Split by <obj>
	objSplit := strings.SplitN(rest, objToken, 2)
	if len(objSplit) != 2 {
		return nil
	}
	object := strings.TrimSpace(objSplit[0])
	relation := strings.TrimSpace(objSplit[1])

	if subject == "" || object == "" || relation == "" {
		return nil
	}

	return &triplet{
		Subject:  subject,
		Object:   object,
		Relation: relation,
		Score:    score,
	}
}

// parseREBELOutputNoTokens parses REBEL output when special tokens are stripped.
// The format is elements separated by double spaces: "Subject  Object  relation  ..."
func parseREBELOutputNoTokens(text string, score float32) []triplet {
	var triplets []triplet

	// Split by double space
	parts := strings.Split(text, "  ")

	// Filter out empty parts
	var elements []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			elements = append(elements, p)
		}
	}

	// Each triplet consists of 3 elements: subject, object, relation
	for i := 0; i+2 < len(elements); i += 3 {
		subject := elements[i]
		object := elements[i+1]
		relation := elements[i+2]

		if subject != "" && object != "" && relation != "" {
			triplets = append(triplets, triplet{
				Subject:  subject,
				Object:   object,
				Relation: relation,
				Score:    score,
			})
		}
	}

	return triplets
}

// tripletsToNER converts REBEL triplets to NER entities and relations.
func tripletsToNER(text string, triplets []triplet) ([]Entity, []Relation) {
	// Track unique entities to avoid duplicates
	entityMap := make(map[string]Entity)
	var relations []Relation

	for _, t := range triplets {
		// Find or create subject entity
		subjectKey := t.Subject
		if _, exists := entityMap[subjectKey]; !exists {
			start, end := findSpan(text, t.Subject)
			entityMap[subjectKey] = Entity{
				Text:  t.Subject,
				Label: "ENTITY", // REBEL doesn't provide entity types
				Start: start,
				End:   end,
				Score: t.Score,
			}
		}

		// Find or create object entity
		objectKey := t.Object
		if _, exists := entityMap[objectKey]; !exists {
			start, end := findSpan(text, t.Object)
			entityMap[objectKey] = Entity{
				Text:  t.Object,
				Label: "ENTITY",
				Start: start,
				End:   end,
				Score: t.Score,
			}
		}

		// Create relation
		relations = append(relations, Relation{
			HeadEntity: entityMap[subjectKey],
			TailEntity: entityMap[objectKey],
			Label:      t.Relation,
			Score:      t.Score,
		})
	}

	// Convert entity map to slice
	entities := make([]Entity, 0, len(entityMap))
	for _, e := range entityMap {
		entities = append(entities, e)
	}

	return entities, relations
}

// findSpan finds the character offsets of a substring in text.
// Returns -1, -1 if not found.
func findSpan(text, substring string) (int, int) {
	idx := strings.Index(strings.ToLower(text), strings.ToLower(substring))
	if idx == -1 {
		return -1, -1
	}
	return idx, idx + len(substring)
}

// PooledREBELConfig holds configuration for creating a PooledREBEL.
type PooledREBELConfig struct {
	// ModelPath is the path to the model directory
	ModelPath string

	// PoolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count)
	PoolSize int

	// ModelBackends specifies which backends this model supports (nil = all backends)
	ModelBackends []string

	// Logger for logging (nil = no logging)
	Logger *zap.Logger
}

// PooledREBEL manages multiple Seq2SeqPipeline instances for concurrent relation extraction.
// Uses the new pipelines package (go-huggingface + gomlx/onnxruntime) instead of hugot.
type PooledREBEL struct {
	pipelines    []*pipelines.Seq2SeqPipeline
	sem          *semaphore.Weighted
	nextPipeline atomic.Uint64
	logger       *zap.Logger
	poolSize     int
	backendType  backends.BackendType
	config       REBELConfig
}

// NewPooledREBEL creates a new Seq2SeqPipeline-based pooled REBEL model.
// This is the new implementation using go-huggingface tokenizers and the backends package.
func NewPooledREBEL(
	cfg PooledREBELConfig,
	sessionManager *backends.SessionManager,
) (*PooledREBEL, backends.BackendType, error) {
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

	// Load REBEL config
	rebelConfig := LoadREBELConfig(cfg.ModelPath)

	logger.Info("Initializing pooled REBEL",
		zap.String("modelPath", cfg.ModelPath),
		zap.Int("poolSize", poolSize),
		zap.String("model_id", rebelConfig.ModelID),
		zap.Int("max_length", rebelConfig.MaxLength))

	// Create N Seq2Seq pipelines
	pipelinesList := make([]*pipelines.Seq2SeqPipeline, poolSize)
	var backendUsed backends.BackendType

	for i := 0; i < poolSize; i++ {
		pipeline, bt, err := pipelines.LoadSeq2SeqPipeline(
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
			logger.Error("Failed to create Seq2Seq pipeline for REBEL",
				zap.Int("index", i),
				zap.Error(err))
			return nil, "", fmt.Errorf("creating Seq2Seq pipeline %d: %w", i, err)
		}
		pipelinesList[i] = pipeline
		backendUsed = bt
		logger.Debug("Created REBEL Seq2Seq pipeline", zap.Int("index", i), zap.String("backend", string(bt)))
	}

	logger.Info("Successfully created pooled REBEL pipelines",
		zap.Int("count", poolSize),
		zap.String("backend", string(backendUsed)))

	return &PooledREBEL{
		pipelines:   pipelinesList,
		sem:         semaphore.NewWeighted(int64(poolSize)),
		logger:      logger,
		poolSize:    poolSize,
		backendType: backendUsed,
		config:      rebelConfig,
	}, backendUsed, nil
}

// BackendType returns the backend type used by this REBEL model
func (p *PooledREBEL) BackendType() backends.BackendType {
	return p.backendType
}

// Config returns the REBEL configuration.
func (p *PooledREBEL) Config() REBELConfig {
	return p.config
}

// --- Model interface ---

// Recognize extracts entities from the given texts.
// REBEL is primarily a relation extractor, so this returns the entities
// extracted as subjects and objects from relation triplets.
func (p *PooledREBEL) Recognize(ctx context.Context, texts []string) ([][]Entity, error) {
	entities, _, err := p.ExtractRelations(ctx, texts, nil, nil)
	return entities, err
}

// Close releases resources.
func (p *PooledREBEL) Close() error {
	var lastErr error
	for i, pipeline := range p.pipelines {
		if pipeline != nil {
			if err := pipeline.Close(); err != nil {
				p.logger.Warn("Failed to close REBEL pipeline",
					zap.Int("index", i),
					zap.Error(err))
				lastErr = err
			}
		}
	}
	p.pipelines = nil
	return lastErr
}

// --- Recognizer interface ---

// RecognizeWithLabels extracts entities of the specified types.
// REBEL doesn't support custom entity labels - it extracts whatever entities
// appear in relation triplets. The labels parameter is ignored.
func (p *PooledREBEL) RecognizeWithLabels(ctx context.Context, texts []string, labels []string) ([][]Entity, error) {
	return p.Recognize(ctx, texts)
}

// Labels returns the default entity labels this model uses.
// REBEL doesn't have predefined entity labels - it extracts entities from relations.
func (p *PooledREBEL) Labels() []string {
	return []string{} // REBEL extracts entities dynamically from relations
}

// ExtractRelations extracts relation triplets from the given texts.
// Returns entities (subjects and objects) and relations between them.
// The entityLabels and relationLabels parameters are ignored by REBEL.
func (p *PooledREBEL) ExtractRelations(ctx context.Context, texts []string, entityLabels []string, relationLabels []string) ([][]Entity, [][]Relation, error) {
	if len(texts) == 0 {
		return [][]Entity{}, [][]Relation{}, nil
	}

	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, nil, ctx.Err()
	default:
	}

	// Acquire semaphore slot (blocks if all pipelines busy)
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}
	defer p.sem.Release(1)

	// Round-robin pipeline selection
	idx := int(p.nextPipeline.Add(1) % uint64(p.poolSize))
	pipeline := p.pipelines[idx]

	p.logger.Debug("Starting REBEL relation extraction",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_inputs", len(texts)))

	// Process each text
	allEntities := make([][]Entity, len(texts))
	allRelations := make([][]Relation, len(texts))

	for i, text := range texts {
		// Run the pipeline
		result, err := pipeline.Generate(ctx, text)
		if err != nil {
			p.logger.Error("REBEL generation failed",
				zap.Int("text_index", i),
				zap.Error(err))
			return nil, nil, fmt.Errorf("running REBEL pipeline on text %d: %w", i, err)
		}

		// Parse triplets from generated text
		rawOutput := result.Text
		p.logger.Debug("REBEL raw output",
			zap.Int("text_index", i),
			zap.String("raw_output", rawOutput),
			zap.Float32("score", result.Score),
			zap.String("triplet_token", p.config.TripletToken),
			zap.String("subject_token", p.config.SubjectToken),
			zap.String("object_token", p.config.ObjectToken))
		triplets := parseREBELTriplets(rawOutput, p.config, result.Score)

		// Convert triplets to entities and relations
		entities, relations := tripletsToNER(text, triplets)
		allEntities[i] = entities
		allRelations[i] = relations

		p.logger.Debug("REBEL extraction completed",
			zap.Int("text_index", i),
			zap.Int("triplets_parsed", len(triplets)),
			zap.Int("entities", len(entities)),
			zap.Int("relations", len(relations)))
	}

	p.logger.Info("REBEL relation extraction completed",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_inputs", len(texts)))

	return allEntities, allRelations, nil
}

// ExtractAnswers performs extractive question answering.
// REBEL does not support question answering - use GLiNER multitask models instead.
func (p *PooledREBEL) ExtractAnswers(ctx context.Context, questions []string, contexts []string) ([]Answer, error) {
	return nil, ErrNotSupported
}

// RelationLabels returns the default relation labels this model uses.
// REBEL extracts relations dynamically - it doesn't have a fixed set of labels.
func (p *PooledREBEL) RelationLabels() []string {
	return []string{} // REBEL extracts relation types dynamically
}
