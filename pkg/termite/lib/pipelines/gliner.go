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

package pipelines

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/gomlx/go-huggingface/tokenizers"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
)

// ============================================================================
// GLiNER Config Types and Loading
// ============================================================================

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
	// GLiNERModelGLiNER2 is the unified GLiNER2 multi-task model from Fastino.
	// Supports NER, classification, structured extraction, and relation extraction.
	GLiNERModelGLiNER2 GLiNERModelType = "gliner2"
)

// GLiNERModelConfig holds parsed configuration for a GLiNER model.
type GLiNERModelConfig struct {
	// Path to the model directory
	ModelPath string

	// ModelFile is the ONNX file for the GLiNER model
	ModelFile string

	// MaxWidth is the maximum entity span width in tokens
	MaxWidth int

	// MaxLength is the maximum sequence length
	MaxLength int

	// DefaultLabels are the entity labels to use if none specified
	DefaultLabels []string

	// Threshold is the score threshold for entity detection (0.0-1.0)
	Threshold float32

	// FlatNER if true, don't allow nested/overlapping entities
	FlatNER bool

	// MultiLabel if true, allow entities to have multiple labels
	MultiLabel bool

	// ModelType indicates the GLiNER architecture variant
	ModelType GLiNERModelType

	// RelationLabels are default relation types for relationship extraction
	RelationLabels []string

	// RelationThreshold is the score threshold for relationship detection
	RelationThreshold float32

	// WordsJoiner is the character used to join words (typically space)
	WordsJoiner string
}

// LoadGLiNERModelConfig loads and parses configuration for a GLiNER model.
func LoadGLiNERModelConfig(modelPath string) (*GLiNERModelConfig, error) {
	config := &GLiNERModelConfig{
		ModelPath:         modelPath,
		MaxWidth:          12,
		MaxLength:         512,
		DefaultLabels:     []string{"person", "organization", "location", "date", "product"},
		Threshold:         0.5,
		FlatNER:           true,
		MultiLabel:        false,
		ModelType:         GLiNERModelUniEncoder,
		RelationThreshold: 0.5,
		WordsJoiner:       " ",
	}

	// Detect model file
	config.ModelFile = FindONNXFile(modelPath, []string{
		"model.onnx",
		"model_quantized.onnx",
		"gliner.onnx",
	})

	if config.ModelFile == "" {
		return nil, fmt.Errorf("no ONNX model file found in %s", modelPath)
	}

	// Load gliner_config.json if present
	glinerConfigPath := filepath.Join(modelPath, "gliner_config.json")
	if data, err := os.ReadFile(glinerConfigPath); err == nil {
		var rawConfig rawGLiNERConfig
		if err := json.Unmarshal(data, &rawConfig); err == nil {
			if rawConfig.MaxWidth > 0 {
				config.MaxWidth = rawConfig.MaxWidth
			}
			if rawConfig.MaxLength > 0 {
				config.MaxLength = rawConfig.MaxLength
			}
			if len(rawConfig.Labels) > 0 {
				config.DefaultLabels = rawConfig.Labels
			}
			if rawConfig.Threshold > 0 {
				config.Threshold = rawConfig.Threshold
			}
			if rawConfig.FlatNER {
				config.FlatNER = rawConfig.FlatNER
			}
			if rawConfig.MultiLabel {
				config.MultiLabel = rawConfig.MultiLabel
			}
			if rawConfig.ModelType != "" {
				config.ModelType = GLiNERModelType(rawConfig.ModelType)
			}
			if len(rawConfig.RelationLabels) > 0 {
				config.RelationLabels = rawConfig.RelationLabels
			}
			if rawConfig.RelationThreshold > 0 {
				config.RelationThreshold = rawConfig.RelationThreshold
			}
			if rawConfig.WordsJoiner != "" {
				config.WordsJoiner = rawConfig.WordsJoiner
			}
		}
	}

	// Detect model type from model name if not specified in config
	if config.ModelType == "" || config.ModelType == GLiNERModelUniEncoder {
		config.ModelType = detectGLiNERModelType(modelPath)
	}

	return config, nil
}

// rawGLiNERConfig represents gliner_config.json structure.
type rawGLiNERConfig struct {
	MaxWidth          int      `json:"max_width"`
	MaxLength         int      `json:"max_len"`
	Labels            []string `json:"labels"`
	Threshold         float32  `json:"threshold"`
	FlatNER           bool     `json:"flat_ner"`
	MultiLabel        bool     `json:"multi_label"`
	ModelType         string   `json:"model_type"`
	RelationLabels    []string `json:"relation_labels"`
	RelationThreshold float32  `json:"relation_threshold"`
	WordsJoiner       string   `json:"words_joiner"`
}

// detectGLiNERModelType attempts to detect the model type from the model name.
func detectGLiNERModelType(modelPath string) GLiNERModelType {
	modelName := strings.ToLower(filepath.Base(modelPath))
	parentDir := strings.ToLower(filepath.Base(filepath.Dir(modelPath)))

	// Check for GLiNER2 models (from Fastino)
	// GLiNER2 models have "gliner2" in name or are from "fastino" organization
	if strings.Contains(modelName, "gliner2") ||
		(strings.Contains(parentDir, "fastino") && strings.Contains(modelName, "gliner")) {
		return GLiNERModelGLiNER2
	}

	switch {
	case strings.Contains(modelName, "multitask"):
		return GLiNERModelMultiTask
	case strings.Contains(modelName, "biencoder") || strings.Contains(modelName, "bi-"):
		return GLiNERModelBiEncoder
	case strings.Contains(modelName, "token") || strings.Contains(modelName, "large"):
		return GLiNERModelTokenLevel
	default:
		return GLiNERModelUniEncoder
	}
}

// IsGLiNERModel checks if a model path contains a GLiNER model.
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

// ============================================================================
// GLiNER Entity Types
// ============================================================================

// GLiNEREntity represents a named entity extracted by GLiNER.
type GLiNEREntity struct {
	// Text is the entity text
	Text string `json:"text"`
	// Label is the entity type
	Label string `json:"label"`
	// Start is the character offset where the entity begins
	Start int `json:"start"`
	// End is the character offset where the entity ends (exclusive)
	End int `json:"end"`
	// Score is the confidence score (0.0 to 1.0)
	Score float32 `json:"score"`
}

// GLiNERRelation represents a relationship between two entities.
type GLiNERRelation struct {
	// HeadEntity is the source entity
	HeadEntity GLiNEREntity `json:"head"`
	// TailEntity is the target entity
	TailEntity GLiNEREntity `json:"tail"`
	// Label is the relationship type
	Label string `json:"label"`
	// Score is the confidence score
	Score float32 `json:"score"`
}

// GLiNEROutput holds the output from GLiNER inference.
type GLiNEROutput struct {
	// Entities holds entities for each input text
	Entities [][]GLiNEREntity
	// Relations holds relations for each input text (if supported)
	Relations [][]GLiNERRelation
}

// GLiNER2TaskType represents different task types for GLiNER2.
type GLiNER2TaskType int

const (
	// GLiNER2TaskNER is standard named entity recognition
	GLiNER2TaskNER GLiNER2TaskType = iota
	// GLiNER2TaskRelations extracts relationships between entities
	GLiNER2TaskRelations
	// GLiNER2TaskClassification performs text classification
	GLiNER2TaskClassification
)

// GLiNER2Classification represents a text classification result.
type GLiNER2Classification struct {
	// Label is the classification label
	Label string `json:"label"`
	// Score is the confidence score (0.0 to 1.0)
	Score float32 `json:"score"`
}

// GLiNER2ClassificationConfig holds configuration for classification.
type GLiNER2ClassificationConfig struct {
	// Threshold is the score threshold for positive classification
	Threshold float32
	// MultiLabel if true, allow multiple labels per text
	MultiLabel bool
	// TopK returns top K predictions (0 = all above threshold)
	TopK int
}

// DefaultGLiNER2ClassificationConfig returns sensible defaults.
func DefaultGLiNER2ClassificationConfig() *GLiNER2ClassificationConfig {
	return &GLiNER2ClassificationConfig{
		Threshold:  0.5,
		MultiLabel: false,
		TopK:       1,
	}
}

// GLiNER2RelationOutput holds the output from relation extraction.
type GLiNER2RelationOutput struct {
	// Entities holds all extracted entities
	Entities [][]GLiNEREntity
	// Relations holds extracted relations
	Relations [][]GLiNERRelation
}

// ============================================================================
// GLiNER Pipeline
// ============================================================================

// labelEmbeddingCache stores precomputed label embeddings for BiEncoder models.
// This allows labels to be encoded once and reused across many inference calls.
type labelEmbeddingCache struct {
	mu         sync.RWMutex
	embeddings map[string][]float32 // label -> embedding
	labels     []string             // cached labels in order
}

// GLiNERPipeline wraps a GLiNER model for zero-shot Named Entity Recognition.
// Unlike traditional NER models, GLiNER can extract entities of any type
// specified at inference time without requiring retraining.
type GLiNERPipeline struct {
	// Session is the low-level ONNX session for running inference
	Session backends.Session

	// LabelEncoderSession is an optional separate session for encoding labels (BiEncoder only)
	LabelEncoderSession backends.Session

	// Tokenizer handles text-to-token conversion
	Tokenizer tokenizers.Tokenizer

	// Config holds model configuration
	Config *GLiNERModelConfig

	// PipelineConfig holds pipeline-specific configuration
	PipelineConfig *GLiNERPipelineConfig

	// backend type used
	backendType backends.BackendType

	// labelCache stores precomputed label embeddings (BiEncoder models only)
	labelCache *labelEmbeddingCache
}

// GLiNERPipelineConfig holds configuration for GLiNER inference.
type GLiNERPipelineConfig struct {
	// Threshold is the score threshold for entity detection
	Threshold float32

	// MaxWidth is the maximum entity span width in tokens
	MaxWidth int

	// FlatNER if true, don't allow nested/overlapping entities
	FlatNER bool

	// MultiLabel if true, allow entities to have multiple labels
	MultiLabel bool

	// DefaultLabels are the entity labels to use if none specified
	DefaultLabels []string
}

// DefaultGLiNERPipelineConfig returns sensible defaults for GLiNER.
func DefaultGLiNERPipelineConfig() *GLiNERPipelineConfig {
	return &GLiNERPipelineConfig{
		Threshold:     0.5,
		MaxWidth:      12,
		FlatNER:       true,
		MultiLabel:    false,
		DefaultLabels: []string{"person", "organization", "location"},
	}
}

// NewGLiNERPipeline creates a new GLiNER pipeline from a session.
func NewGLiNERPipeline(
	session backends.Session,
	tokenizer tokenizers.Tokenizer,
	modelConfig *GLiNERModelConfig,
	pipelineConfig *GLiNERPipelineConfig,
	backendType backends.BackendType,
) *GLiNERPipeline {
	if pipelineConfig == nil {
		pipelineConfig = DefaultGLiNERPipelineConfig()
	}

	// Override pipeline config with model config values if not explicitly set
	if modelConfig != nil {
		if pipelineConfig.Threshold == 0 {
			pipelineConfig.Threshold = modelConfig.Threshold
		}
		if pipelineConfig.MaxWidth == 0 {
			pipelineConfig.MaxWidth = modelConfig.MaxWidth
		}
		if len(pipelineConfig.DefaultLabels) == 0 {
			pipelineConfig.DefaultLabels = modelConfig.DefaultLabels
		}
	}

	p := &GLiNERPipeline{
		Session:        session,
		Tokenizer:      tokenizer,
		Config:         modelConfig,
		PipelineConfig: pipelineConfig,
		backendType:    backendType,
	}

	// Initialize label cache for BiEncoder models
	if modelConfig != nil && modelConfig.ModelType == GLiNERModelBiEncoder {
		p.labelCache = &labelEmbeddingCache{
			embeddings: make(map[string][]float32),
		}
	}

	return p
}

// Recognize extracts entities from texts using the default labels.
func (p *GLiNERPipeline) Recognize(ctx context.Context, texts []string) (*GLiNEROutput, error) {
	return p.RecognizeWithLabels(ctx, texts, p.PipelineConfig.DefaultLabels)
}

// RecognizeWithLabels extracts entities of the specified types (zero-shot NER).
// This is the key feature of GLiNER - it can extract any entity type without retraining.
func (p *GLiNERPipeline) RecognizeWithLabels(ctx context.Context, texts []string, labels []string) (*GLiNEROutput, error) {
	if len(texts) == 0 {
		return &GLiNEROutput{Entities: [][]GLiNEREntity{}}, nil
	}

	if len(labels) == 0 {
		labels = p.PipelineConfig.DefaultLabels
	}

	// Process each text
	allEntities := make([][]GLiNEREntity, len(texts))
	for i, text := range texts {
		entities, err := p.processText(ctx, text, labels)
		if err != nil {
			return nil, fmt.Errorf("processing text %d: %w", i, err)
		}
		allEntities[i] = entities
	}

	return &GLiNEROutput{Entities: allEntities}, nil
}

// processText processes a single text with the given labels.
func (p *GLiNERPipeline) processText(ctx context.Context, text string, labels []string) ([]GLiNEREntity, error) {
	if text == "" {
		return nil, nil
	}

	// Tokenize the text and track word boundaries
	words, wordStartChars, wordEndChars := p.splitIntoWords(text)
	if len(words) == 0 {
		return nil, nil
	}

	// Build the prompt with labels
	// GLiNER typically expects: <<entity_type1>><<entity_type2>>... followed by text tokens
	prompt := p.buildPrompt(labels)

	// Tokenize prompt and text
	promptTokens := p.Tokenizer.Encode(prompt)
	textTokens := p.tokenizeWords(words)

	// Build model inputs
	inputs, err := p.buildInputs(promptTokens, textTokens, words, labels)
	if err != nil {
		return nil, fmt.Errorf("building inputs: %w", err)
	}

	// Run model inference
	outputs, err := p.Session.Run(inputs)
	if err != nil {
		return nil, fmt.Errorf("running inference: %w", err)
	}

	// Parse outputs to extract entities
	entities, err := p.parseOutputs(outputs, words, wordStartChars, wordEndChars, labels, text)
	if err != nil {
		return nil, fmt.Errorf("parsing outputs: %w", err)
	}

	return entities, nil
}

// splitIntoWords splits text into words and returns word boundaries.
func (p *GLiNERPipeline) splitIntoWords(text string) ([]string, []int, []int) {
	var words []string
	var startChars, endChars []int

	wordStart := -1
	for i, r := range text {
		if isWordChar(r) {
			if wordStart == -1 {
				wordStart = i
			}
		} else {
			if wordStart != -1 {
				words = append(words, text[wordStart:i])
				startChars = append(startChars, wordStart)
				endChars = append(endChars, i)
				wordStart = -1
			}
		}
	}

	// Handle last word
	if wordStart != -1 {
		words = append(words, text[wordStart:])
		startChars = append(startChars, wordStart)
		endChars = append(endChars, len(text))
	}

	return words, startChars, endChars
}

// isWordChar returns true if the rune is part of a word.
func isWordChar(r rune) bool {
	return r != ' ' && r != '\t' && r != '\n' && r != '\r'
}

// buildPrompt constructs the label prompt for GLiNER.
// GLiNER ONNX models expect labels in format: <<ENT>>label1<<SEP>><<ENT>>label2<<SEP>>...
// where <<ENT>> and <<SEP>> are special tokens in the vocabulary.
func (p *GLiNERPipeline) buildPrompt(labels []string) string {
	var sb strings.Builder
	for _, label := range labels {
		sb.WriteString("<<ENT>>")
		sb.WriteString(label)
		sb.WriteString("<<SEP>>")
	}
	return sb.String()
}

// buildPromptForTask constructs the label prompt for different GLiNER2 tasks.
// Each task type uses a different prompt format:
// - NER: <<ENT>>label<<SEP>>
// - Relations: <<REL>>entity::relation<<SEP>>
// - Classification: <<CLS>>label<<SEP>>
func (p *GLiNERPipeline) buildPromptForTask(labels []string, taskType GLiNER2TaskType) string {
	var sb strings.Builder

	var prefix string
	switch taskType {
	case GLiNER2TaskNER:
		prefix = "<<ENT>>"
	case GLiNER2TaskRelations:
		prefix = "<<REL>>"
	case GLiNER2TaskClassification:
		prefix = "<<CLS>>"
	default:
		prefix = "<<ENT>>"
	}

	for _, label := range labels {
		sb.WriteString(prefix)
		sb.WriteString(label)
		sb.WriteString("<<SEP>>")
	}

	return sb.String()
}

// buildCompositeRelationLabels creates composite labels for relation extraction.
// For each entity type and relation type, creates "entity::relation" label.
// This follows GLiNER2's approach where relations are expressed as composite labels.
func (p *GLiNERPipeline) buildCompositeRelationLabels(entityLabels []string, relationLabels []string) []string {
	compositeLabels := make([]string, 0, len(entityLabels)*len(relationLabels))
	for _, entity := range entityLabels {
		for _, relation := range relationLabels {
			compositeLabels = append(compositeLabels, entity+"::"+relation)
		}
	}
	return compositeLabels
}

// ExtractRelations extracts entities and relationships between them.
// This is a GLiNER2-specific feature that uses composite labels.
//
// The approach:
// 1. First extract all entities with the given entity labels
// 2. Then use composite labels (entity::relation) to find head entities
// 3. Match head entities with potential tail entities
func (p *GLiNERPipeline) ExtractRelations(
	ctx context.Context,
	texts []string,
	entityLabels []string,
	relationLabels []string,
) (*GLiNER2RelationOutput, error) {
	if !p.IsGLiNER2() {
		return nil, fmt.Errorf("relation extraction requires GLiNER2 model")
	}

	if len(texts) == 0 {
		return &GLiNER2RelationOutput{
			Entities:  [][]GLiNEREntity{},
			Relations: [][]GLiNERRelation{},
		}, nil
	}

	if len(entityLabels) == 0 {
		entityLabels = p.PipelineConfig.DefaultLabels
	}

	if len(relationLabels) == 0 && p.Config != nil {
		relationLabels = p.Config.RelationLabels
	}

	// Process each text
	allEntities := make([][]GLiNEREntity, len(texts))
	allRelations := make([][]GLiNERRelation, len(texts))

	for i, text := range texts {
		entities, relations, err := p.processTextForRelations(ctx, text, entityLabels, relationLabels)
		if err != nil {
			return nil, fmt.Errorf("processing text %d for relations: %w", i, err)
		}
		allEntities[i] = entities
		allRelations[i] = relations
	}

	return &GLiNER2RelationOutput{
		Entities:  allEntities,
		Relations: allRelations,
	}, nil
}

// processTextForRelations extracts both entities and relations from a single text.
func (p *GLiNERPipeline) processTextForRelations(
	ctx context.Context,
	text string,
	entityLabels []string,
	relationLabels []string,
) ([]GLiNEREntity, []GLiNERRelation, error) {
	if text == "" {
		return nil, nil, nil
	}

	// Step 1: Extract all entities first
	entities, err := p.processText(ctx, text, entityLabels)
	if err != nil {
		return nil, nil, fmt.Errorf("extracting entities: %w", err)
	}

	if len(entities) < 2 || len(relationLabels) == 0 {
		// Need at least 2 entities for a relation
		return entities, nil, nil
	}

	// Step 2: Build composite labels for relation extraction
	compositeLabels := p.buildCompositeRelationLabels(entityLabels, relationLabels)

	// Step 3: Extract relation head entities using composite labels
	relationHeadSpans, err := p.processText(ctx, text, compositeLabels)
	if err != nil {
		return entities, nil, fmt.Errorf("extracting relation heads: %w", err)
	}

	// Step 4: Match relation heads to entities and find tail entities
	relations := p.matchRelations(entities, relationHeadSpans, entityLabels)

	return entities, relations, nil
}

// matchRelations matches extracted relation head spans to entities and finds relations.
func (p *GLiNERPipeline) matchRelations(
	entities []GLiNEREntity,
	relationHeadSpans []GLiNEREntity,
	entityLabels []string,
) []GLiNERRelation {
	var relations []GLiNERRelation

	// Build a map of entity positions for quick lookup
	entityByPos := make(map[string]*GLiNEREntity)
	for i := range entities {
		key := fmt.Sprintf("%d-%d", entities[i].Start, entities[i].End)
		entityByPos[key] = &entities[i]
	}

	// Process each relation head span
	for _, headSpan := range relationHeadSpans {
		// Parse the composite label: "entity_type::relation"
		parts := strings.SplitN(headSpan.Label, "::", 2)
		if len(parts) != 2 {
			continue
		}
		headEntityType := parts[0]
		relationLabel := parts[1]

		// Find the matching entity for this head span
		headKey := fmt.Sprintf("%d-%d", headSpan.Start, headSpan.End)
		headEntity := entityByPos[headKey]
		if headEntity == nil {
			// Create a head entity from the span
			headEntity = &GLiNEREntity{
				Text:  headSpan.Text,
				Label: headEntityType,
				Start: headSpan.Start,
				End:   headSpan.End,
				Score: headSpan.Score,
			}
		}

		// Find potential tail entities (non-overlapping entities)
		for i := range entities {
			tail := &entities[i]

			// Skip if same span or overlapping
			if overlapsSpan(headSpan.Start, headSpan.End, tail.Start, tail.End) {
				continue
			}

			// Create relation
			relations = append(relations, GLiNERRelation{
				HeadEntity: *headEntity,
				TailEntity: *tail,
				Label:      relationLabel,
				Score:      (headSpan.Score + tail.Score) / 2, // Average confidence
			})
		}
	}

	// Apply threshold filtering if configured
	if p.Config != nil && p.Config.RelationThreshold > 0 {
		filtered := make([]GLiNERRelation, 0, len(relations))
		for _, rel := range relations {
			if rel.Score >= p.Config.RelationThreshold {
				filtered = append(filtered, rel)
			}
		}
		relations = filtered
	}

	return relations
}

// overlapsSpan checks if two spans overlap.
func overlapsSpan(start1, end1, start2, end2 int) bool {
	return start1 < end2 && start2 < end1
}

// ClassifyText performs text classification using GLiNER2.
// Classification is implemented by treating the entire text as a single span
// and scoring it against each class label.
func (p *GLiNERPipeline) ClassifyText(
	ctx context.Context,
	texts []string,
	labels []string,
	config *GLiNER2ClassificationConfig,
) ([][]GLiNER2Classification, error) {
	if !p.IsGLiNER2() {
		return nil, fmt.Errorf("classification requires GLiNER2 model")
	}

	if len(texts) == 0 {
		return [][]GLiNER2Classification{}, nil
	}

	if len(labels) == 0 {
		return nil, fmt.Errorf("classification labels are required")
	}

	if config == nil {
		config = DefaultGLiNER2ClassificationConfig()
	}

	results := make([][]GLiNER2Classification, len(texts))

	for i, text := range texts {
		classifications, err := p.classifySingleText(ctx, text, labels, config)
		if err != nil {
			return nil, fmt.Errorf("classifying text %d: %w", i, err)
		}
		results[i] = classifications
	}

	return results, nil
}

// classifySingleText classifies a single text against the given labels.
// We use the span-based approach where we score each label against the entire text.
func (p *GLiNERPipeline) classifySingleText(
	ctx context.Context,
	text string,
	labels []string,
	config *GLiNER2ClassificationConfig,
) ([]GLiNER2Classification, error) {
	if text == "" {
		return nil, nil
	}

	// For classification, we use the same approach as NER but interpret results differently.
	// Each label is treated as a potential "entity type" for the entire text span.
	// We use the classification prompt format.
	prompt := p.buildPromptForTask(labels, GLiNER2TaskClassification)

	// Tokenize
	words, _, _ := p.splitIntoWords(text)
	if len(words) == 0 {
		return nil, nil
	}

	promptTokens := p.Tokenizer.Encode(prompt)
	textTokens := p.tokenizeWords(words)

	// Build inputs
	inputs, err := p.buildInputs(promptTokens, textTokens, words, labels)
	if err != nil {
		return nil, fmt.Errorf("building inputs: %w", err)
	}

	// Run inference
	outputs, err := p.Session.Run(inputs)
	if err != nil {
		return nil, fmt.Errorf("running inference: %w", err)
	}

	// Parse classification outputs
	classifications := p.parseClassificationOutputs(outputs, labels, config)

	return classifications, nil
}

// parseClassificationOutputs extracts classification results from model output.
// For classification, we look at the highest-scoring span for each label
// and use that as the classification confidence.
func (p *GLiNERPipeline) parseClassificationOutputs(
	outputs []backends.NamedTensor,
	labels []string,
	config *GLiNER2ClassificationConfig,
) []GLiNER2Classification {
	// Find the logits tensor
	var logits []float32
	var logitsShape []int64

	for _, out := range outputs {
		if out.Name == "logits" || len(outputs) == 1 {
			switch data := out.Data.(type) {
			case []float32:
				logits = data
				logitsShape = out.Shape
			}
			break
		}
	}

	if logits == nil {
		return nil
	}

	// For GLiNER2 classification, we aggregate span scores per label
	// The output shape is typically [batch, num_spans, num_labels] or [batch, num_spans, 1]
	// For classification with single span (whole text), we want the max score per label

	numLabels := len(labels)
	results := make([]GLiNER2Classification, 0, numLabels)

	if len(logitsShape) >= 3 {
		// Shape: [batch, num_spans, ...]
		numSpans := int(logitsShape[1])

		// For each label, find the maximum score across all spans
		// In classification mode, we typically have one label encoded per run,
		// so we need to interpret the output based on how we encoded the labels
		if numLabels == 1 || (len(logitsShape) > 2 && logitsShape[2] == 1) {
			// Single label per call or single output channel
			// Find max score across spans and apply sigmoid
			maxScore := float32(-1000)
			for i := 0; i < numSpans && i < len(logits); i++ {
				if logits[i] > maxScore {
					maxScore = logits[i]
				}
			}

			// Apply sigmoid to convert logit to probability
			score := sigmoid(maxScore)

			for _, label := range labels {
				if config.MultiLabel || score >= config.Threshold {
					results = append(results, GLiNER2Classification{
						Label: label,
						Score: score,
					})
				}
			}
		} else {
			// Multiple labels with separate output channels
			labelsPerSpan := 1
			if len(logitsShape) > 2 {
				labelsPerSpan = int(logitsShape[2])
			}

			for labelIdx, label := range labels {
				if labelIdx >= labelsPerSpan {
					break
				}

				// Find max score for this label across all spans
				maxScore := float32(-1000)
				for spanIdx := 0; spanIdx < numSpans; spanIdx++ {
					idx := spanIdx*labelsPerSpan + labelIdx
					if idx < len(logits) && logits[idx] > maxScore {
						maxScore = logits[idx]
					}
				}

				score := sigmoid(maxScore)
				results = append(results, GLiNER2Classification{
					Label: label,
					Score: score,
				})
			}
		}
	} else {
		// Fallback: treat as flat array of scores
		for i, label := range labels {
			if i >= len(logits) {
				break
			}
			score := sigmoid(logits[i])
			results = append(results, GLiNER2Classification{
				Label: label,
				Score: score,
			})
		}
	}

	// Apply filtering based on config
	if !config.MultiLabel {
		// Single-label: return only the highest scoring
		if len(results) > 0 {
			sort.Slice(results, func(i, j int) bool {
				return results[i].Score > results[j].Score
			})
			if config.TopK > 0 && len(results) > config.TopK {
				results = results[:config.TopK]
			} else {
				results = results[:1]
			}
		}
	} else {
		// Multi-label: filter by threshold
		filtered := make([]GLiNER2Classification, 0, len(results))
		for _, r := range results {
			if r.Score >= config.Threshold {
				filtered = append(filtered, r)
			}
		}
		results = filtered

		// Sort by score descending
		sort.Slice(results, func(i, j int) bool {
			return results[i].Score > results[j].Score
		})

		// Apply TopK if configured
		if config.TopK > 0 && len(results) > config.TopK {
			results = results[:config.TopK]
		}
	}

	return results
}

// sigmoid converts a logit to a probability.
func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

// tokenizeWords tokenizes each word and returns the tokens per word.
func (p *GLiNERPipeline) tokenizeWords(words []string) [][]int {
	result := make([][]int, len(words))
	for i, word := range words {
		result[i] = p.Tokenizer.Encode(word)
	}
	return result
}

// buildInputs constructs the model inputs for GLiNER inference.
func (p *GLiNERPipeline) buildInputs(promptTokens []int, textTokens [][]int, words []string, labels []string) ([]backends.NamedTensor, error) {
	// Count total text tokens (excluding special tokens added by tokenizer)
	totalTextTokens := 0
	for _, wt := range textTokens {
		// Each word's tokens may include special tokens, strip them
		for _, tok := range wt {
			if tok != 0 && tok != 1 && tok != 2 { // Skip PAD, CLS, SEP
				totalTextTokens++
			}
		}
	}

	// The tokenizer (DeBERTa) uses: [CLS]=1, [SEP]=2, [PAD]=0
	// The promptTokens already includes [CLS] at start and [SEP] at end
	// Strip the special tokens from promptTokens for cleaner handling
	cleanPromptTokens := make([]int, 0, len(promptTokens))
	for _, tok := range promptTokens {
		if tok != 1 && tok != 2 { // Skip CLS and SEP
			cleanPromptTokens = append(cleanPromptTokens, tok)
		}
	}

	// Build combined token sequence: [CLS] + prompt + text tokens + [SEP]
	// Note: GLiNER expects prompt and text in same segment (no separator between them)
	seqLen := len(cleanPromptTokens) + totalTextTokens + 2 // CLS at start, SEP at end

	// Limit sequence length
	maxLen := p.Config.MaxLength
	if seqLen > maxLen {
		seqLen = maxLen
	}

	// Build input_ids
	inputIDs := make([]int64, seqLen)
	attentionMask := make([]int64, seqLen)

	// DeBERTa special token IDs
	clsID := int64(1) // DeBERTa [CLS]
	sepID := int64(2) // DeBERTa [SEP]
	padID := int64(0) // DeBERTa [PAD]
	_ = padID         // unused for now, padding handled later

	idx := 0
	inputIDs[idx] = clsID
	attentionMask[idx] = 1
	idx++

	// Add prompt tokens (label markers)
	for _, tok := range cleanPromptTokens {
		if idx >= seqLen-1 {
			break
		}
		inputIDs[idx] = int64(tok)
		attentionMask[idx] = 1
		idx++
	}

	// Build words mask to track which tokens belong to which word
	wordsMask := make([]int64, seqLen)
	textLengths := make([]int64, 1)

	// Track position of first text token
	textStartIdx := idx

	// Add text tokens with word tracking
	// Skip special tokens (CLS=1, SEP=2, PAD=0) that tokenizer may add to each word
	wordIdx := int64(1) // Start at 1 (0 reserved for non-word tokens)
	for _, wordTokens := range textTokens {
		hasRealToken := false
		for _, tok := range wordTokens {
			// Skip special tokens that tokenizer adds
			if tok == 0 || tok == 1 || tok == 2 {
				continue
			}
			if idx >= seqLen-1 {
				break
			}
			inputIDs[idx] = int64(tok)
			attentionMask[idx] = 1
			wordsMask[idx] = wordIdx
			idx++
			hasRealToken = true
		}
		if hasRealToken {
			wordIdx++
		}
		if idx >= seqLen-1 {
			break
		}
	}

	// Record text length (number of text tokens)
	numTextTokens := idx - textStartIdx
	textLengths[0] = int64(numTextTokens)

	// Add final separator
	if idx < seqLen {
		inputIDs[idx] = sepID
		attentionMask[idx] = 1
		idx++
	}

	// Pad remaining
	for ; idx < seqLen; idx++ {
		inputIDs[idx] = padID
		attentionMask[idx] = 0
	}

	// Build span indices as a flat list of [numTextTokens * maxWidth] spans
	// The model operates on token positions, not word positions
	// It internally reshapes to [numTextTokens, maxWidth, ...]
	maxWidth := p.PipelineConfig.MaxWidth

	// Ensure at least 1 token
	if numTextTokens < 1 {
		numTextTokens = 1
	}

	// Total number of spans = numTextTokens * maxWidth (fixed grid)
	numSpans := numTextTokens * maxWidth

	// Build span_idx as [1, numSpans, 2] and span_mask as [1, numSpans]
	spanIdx := make([]int64, numSpans*2)
	spanMask := make([]bool, numSpans)

	for t := 0; t < numTextTokens; t++ {
		for wi := 0; wi < maxWidth; wi++ {
			spanI := t*maxWidth + wi
			start := t
			end := t + wi // span end position (token index)
			spanIdx[spanI*2] = int64(start)
			spanIdx[spanI*2+1] = int64(end)
			// Mask is true if the span end is within bounds
			spanMask[spanI] = end < numTextTokens
		}
	}

	// Build named tensors
	inputs := []backends.NamedTensor{
		{
			Name:  "input_ids",
			Shape: []int64{1, int64(seqLen)},
			Data:  inputIDs,
		},
		{
			Name:  "attention_mask",
			Shape: []int64{1, int64(seqLen)},
			Data:  attentionMask,
		},
		{
			Name:  "words_mask",
			Shape: []int64{1, int64(seqLen)},
			Data:  wordsMask,
		},
		{
			Name:  "text_lengths",
			Shape: []int64{1, 1},
			Data:  textLengths,
		},
		{
			Name:  "span_idx",
			Shape: []int64{1, int64(numSpans), 2},
			Data:  spanIdx,
		},
		{
			Name:  "span_mask",
			Shape: []int64{1, int64(numSpans)},
			Data:  spanMask,
		},
	}

	return inputs, nil
}

// parseOutputs extracts entities from model outputs.
func (p *GLiNERPipeline) parseOutputs(outputs []backends.NamedTensor, words []string, wordStartChars, wordEndChars []int, labels []string, originalText string) ([]GLiNEREntity, error) {
	// Find the logits output
	var logits []float32
	var logitsShape []int64

	for _, out := range outputs {
		if strings.Contains(strings.ToLower(out.Name), "logits") || out.Name == "output" {
			if data, ok := out.Data.([]float32); ok {
				logits = data
				logitsShape = out.Shape
				break
			}
		}
	}

	if logits == nil {
		// Try first float32 output
		for _, out := range outputs {
			if data, ok := out.Data.([]float32); ok {
				logits = data
				logitsShape = out.Shape
				break
			}
		}
	}

	if logits == nil {
		return nil, fmt.Errorf("no logits output found")
	}

	// Logits shape varies by model:
	// - GLiNER v1: [batch, num_tokens, max_width, num_labels] (4D)
	// - GLiNER2:   [batch, num_spans, 1] where num_spans = num_tokens * max_width (3D)
	// Both have identical flat data layout for single-label queries.
	// We need to use the actual shape from the output, not numWords.
	numLabels := len(labels)
	numWords := len(words)
	maxWidth := p.PipelineConfig.MaxWidth

	// Get dimensions from logits shape
	var numTokens int
	if len(logitsShape) >= 4 {
		// GLiNER v1 format: [batch, num_tokens, max_width, num_labels]
		numTokens = int(logitsShape[1])
		maxWidth = int(logitsShape[2])
		numLabels = int(logitsShape[3])
	} else if len(logitsShape) == 3 {
		// GLiNER2 format: [batch, num_spans, 1]
		// num_spans = num_tokens * max_width, numLabels = 1
		numSpans := int(logitsShape[1])
		numLabels = int(logitsShape[2])
		// Infer numTokens from num_spans / max_width
		if maxWidth > 0 && numSpans > 0 {
			numTokens = numSpans / maxWidth
		}
	}

	// Ensure at least 1 token, fall back to word count if not determined
	if numTokens < 1 {
		numTokens = numWords
	}

	// Extract entities from spans with scores above threshold
	// The span grid is [numTokens, maxWidth] where:
	// - First index is the token position
	// - Second index is the span width index (0 = width 1, 1 = width 2, etc.)
	// We need to map token positions back to word positions for entity extraction
	var entities []GLiNEREntity
	threshold := p.PipelineConfig.Threshold

	// For now, use word-based iteration since we need word boundaries for entity text
	// The logits are indexed by word position (after the prompt), not raw token position
	for w := 0; w < numWords; w++ {
		for wi := 0; wi < maxWidth; wi++ {
			start := w
			end := w + wi // span end position (word index)

			// Skip invalid spans (extending beyond text)
			if end >= numWords {
				continue
			}

			// Get scores for this span across all labels
			// Logits layout: [batch, token_pos, width, label]
			// Flat index: token_pos * maxWidth * numLabels + width * numLabels + label
			spanIdx := w*maxWidth*numLabels + wi*numLabels
			for labelIdx := 0; labelIdx < numLabels; labelIdx++ {
				logitIdx := spanIdx + labelIdx
				if logitIdx >= len(logits) {
					continue
				}

				score := sigmoid(logits[logitIdx])
				if score >= threshold {
					// Build entity text from words
					entityWords := words[start : end+1]
					entityText := strings.Join(entityWords, p.Config.WordsJoiner)

					// Get character positions
					charStart := wordStartChars[start]
					charEnd := wordEndChars[end]

					// Verify against original text
					if charStart < len(originalText) && charEnd <= len(originalText) {
						entityText = originalText[charStart:charEnd]
					}

					entities = append(entities, GLiNEREntity{
						Text:  entityText,
						Label: labels[labelIdx],
						Start: charStart,
						End:   charEnd,
						Score: score,
					})
				}
			}
		}
	}

	// Apply flat NER (remove overlapping entities) if enabled
	if p.PipelineConfig.FlatNER && len(entities) > 1 {
		entities = p.removeOverlappingEntities(entities)
	}

	// Sort by position
	sort.Slice(entities, func(i, j int) bool {
		if entities[i].Start != entities[j].Start {
			return entities[i].Start < entities[j].Start
		}
		return entities[i].End < entities[j].End
	})

	return entities, nil
}

// removeOverlappingEntities removes overlapping entities, keeping highest scoring ones.
func (p *GLiNERPipeline) removeOverlappingEntities(entities []GLiNEREntity) []GLiNEREntity {
	if len(entities) <= 1 {
		return entities
	}

	// Sort by score descending
	sorted := make([]GLiNEREntity, len(entities))
	copy(sorted, entities)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Score > sorted[j].Score
	})

	var result []GLiNEREntity
	for _, ent := range sorted {
		overlaps := false
		for _, existing := range result {
			if ent.Start < existing.End && ent.End > existing.Start {
				overlaps = true
				break
			}
		}
		if !overlaps {
			result = append(result, ent)
		}
	}

	return result
}

// Backend returns the backend type this pipeline uses.
func (p *GLiNERPipeline) Backend() backends.BackendType {
	return p.backendType
}

// Close releases resources held by the pipeline.
func (p *GLiNERPipeline) Close() error {
	var err error
	if p.Session != nil {
		err = p.Session.Close()
	}
	if p.LabelEncoderSession != nil {
		if closeErr := p.LabelEncoderSession.Close(); closeErr != nil && err == nil {
			err = closeErr
		}
	}
	// Clear the label cache
	if p.labelCache != nil {
		p.ClearLabelEmbeddingCache()
	}
	return err
}

// ============================================================================
// BiEncoder Label Embedding Caching
// ============================================================================

// IsBiEncoder returns true if this is a BiEncoder model that supports label caching.
func (p *GLiNERPipeline) IsBiEncoder() bool {
	return p.Config != nil && p.Config.ModelType == GLiNERModelBiEncoder && p.labelCache != nil
}

// PrecomputeLabelEmbeddings precomputes and caches embeddings for the given labels.
// This is useful for BiEncoder models where label embeddings can be computed once
// and reused across many inference calls with the same labels.
//
// For BiEncoder models, this runs the labels through the label encoder to get
// embeddings that can be reused. For UniEncoder models, this is a no-op since
// labels are encoded together with the text.
func (p *GLiNERPipeline) PrecomputeLabelEmbeddings(labels []string) error {
	if !p.IsBiEncoder() {
		// Not a BiEncoder model, nothing to precompute
		return nil
	}

	if len(labels) == 0 {
		return nil
	}

	p.labelCache.mu.Lock()
	defer p.labelCache.mu.Unlock()

	// Check which labels need to be computed
	var labelsToCompute []string
	for _, label := range labels {
		if _, exists := p.labelCache.embeddings[label]; !exists {
			labelsToCompute = append(labelsToCompute, label)
		}
	}

	if len(labelsToCompute) == 0 {
		// All labels already cached
		return nil
	}

	// Compute embeddings for new labels
	embeddings, err := p.computeLabelEmbeddings(labelsToCompute)
	if err != nil {
		return fmt.Errorf("computing label embeddings: %w", err)
	}

	// Cache the embeddings
	for i, label := range labelsToCompute {
		p.labelCache.embeddings[label] = embeddings[i]
	}

	// Update cached labels list
	p.labelCache.labels = labels

	return nil
}

// computeLabelEmbeddings computes embeddings for the given labels using the label encoder.
// This is an internal method used by PrecomputeLabelEmbeddings.
func (p *GLiNERPipeline) computeLabelEmbeddings(labels []string) ([][]float32, error) {
	// Build inputs for the label encoder
	// Each label is tokenized and passed through the encoder

	embeddings := make([][]float32, len(labels))

	for i, label := range labels {
		// Tokenize the label with special formatting for GLiNER
		// GLiNER expects labels in format: <<label>>
		formattedLabel := "<<" + label + ">>"
		tokens := p.Tokenizer.Encode(formattedLabel)

		// Build input tensors for label encoding
		seqLen := len(tokens) + 2 // +2 for CLS and SEP
		inputIDs := make([]int64, seqLen)
		attentionMask := make([]int64, seqLen)

		// CLS token
		inputIDs[0] = 101
		attentionMask[0] = 1

		// Label tokens
		for j, tok := range tokens {
			inputIDs[j+1] = int64(tok)
			attentionMask[j+1] = 1
		}

		// SEP token
		inputIDs[seqLen-1] = 102
		attentionMask[seqLen-1] = 1

		inputs := []backends.NamedTensor{
			{
				Name:  "input_ids",
				Shape: []int64{1, int64(seqLen)},
				Data:  inputIDs,
			},
			{
				Name:  "attention_mask",
				Shape: []int64{1, int64(seqLen)},
				Data:  attentionMask,
			},
		}

		// Use the label encoder session if available, otherwise use main session
		session := p.LabelEncoderSession
		if session == nil {
			session = p.Session
		}

		outputs, err := session.Run(inputs)
		if err != nil {
			return nil, fmt.Errorf("running label encoder for %q: %w", label, err)
		}

		// Extract the embedding from the output
		// Typically this is the [CLS] token representation or a pooled output
		embedding, err := extractLabelEmbedding(outputs)
		if err != nil {
			return nil, fmt.Errorf("extracting embedding for %q: %w", label, err)
		}

		embeddings[i] = embedding
	}

	return embeddings, nil
}

// extractLabelEmbedding extracts the label embedding from model outputs.
func extractLabelEmbedding(outputs []backends.NamedTensor) ([]float32, error) {
	// Look for the embedding output
	for _, out := range outputs {
		name := strings.ToLower(out.Name)
		if strings.Contains(name, "embedding") || strings.Contains(name, "pooler") || strings.Contains(name, "label") {
			if data, ok := out.Data.([]float32); ok {
				return data, nil
			}
		}
	}

	// Fall back to first float32 output
	for _, out := range outputs {
		if data, ok := out.Data.([]float32); ok {
			return data, nil
		}
	}

	return nil, fmt.Errorf("no float32 embedding found in outputs")
}

// HasCachedLabelEmbeddings returns true if label embeddings are currently cached.
func (p *GLiNERPipeline) HasCachedLabelEmbeddings() bool {
	if !p.IsBiEncoder() {
		return false
	}

	p.labelCache.mu.RLock()
	defer p.labelCache.mu.RUnlock()

	return len(p.labelCache.embeddings) > 0
}

// CachedLabels returns the list of labels that are currently cached.
func (p *GLiNERPipeline) CachedLabels() []string {
	if !p.IsBiEncoder() {
		return nil
	}

	p.labelCache.mu.RLock()
	defer p.labelCache.mu.RUnlock()

	result := make([]string, len(p.labelCache.labels))
	copy(result, p.labelCache.labels)
	return result
}

// GetCachedLabelEmbedding returns the cached embedding for a label, if available.
// Returns nil if the label is not cached or this is not a BiEncoder model.
func (p *GLiNERPipeline) GetCachedLabelEmbedding(label string) []float32 {
	if !p.IsBiEncoder() {
		return nil
	}

	p.labelCache.mu.RLock()
	defer p.labelCache.mu.RUnlock()

	return p.labelCache.embeddings[label]
}

// ClearLabelEmbeddingCache clears all cached label embeddings.
func (p *GLiNERPipeline) ClearLabelEmbeddingCache() {
	if p.labelCache == nil {
		return
	}

	p.labelCache.mu.Lock()
	defer p.labelCache.mu.Unlock()

	p.labelCache.embeddings = make(map[string][]float32)
	p.labelCache.labels = nil
}

// SupportsRelationExtraction returns true if the model supports relation extraction.
func (p *GLiNERPipeline) SupportsRelationExtraction() bool {
	if p.Config == nil {
		return false
	}
	// GLiNER2 and MultiTask models support relation extraction
	isRelationCapable := p.Config.ModelType == GLiNERModelMultiTask ||
		p.Config.ModelType == GLiNERModelGLiNER2
	return isRelationCapable
}

// SupportsClassification returns true if the model supports text classification.
func (p *GLiNERPipeline) SupportsClassification() bool {
	if p.Config == nil {
		return false
	}
	return p.Config.ModelType == GLiNERModelGLiNER2
}

// IsGLiNER2 returns true if this is a GLiNER2 model.
func (p *GLiNERPipeline) IsGLiNER2() bool {
	return p.Config != nil && p.Config.ModelType == GLiNERModelGLiNER2
}

// ============================================================================
// Loader Functions
// ============================================================================

// GLiNERLoaderOption configures GLiNER pipeline loading.
type GLiNERLoaderOption func(*glinerLoaderConfig)

type glinerLoaderConfig struct {
	threshold     float32
	maxWidth      int
	flatNER       bool
	multiLabel    bool
	defaultLabels []string
	quantized     bool
}

// WithGLiNERThreshold sets the score threshold for entity detection.
func WithGLiNERThreshold(threshold float32) GLiNERLoaderOption {
	return func(c *glinerLoaderConfig) {
		c.threshold = threshold
	}
}

// WithGLiNERMaxWidth sets the maximum entity span width.
func WithGLiNERMaxWidth(maxWidth int) GLiNERLoaderOption {
	return func(c *glinerLoaderConfig) {
		c.maxWidth = maxWidth
	}
}

// WithGLiNERFlatNER enables flat NER mode (no overlapping entities).
func WithGLiNERFlatNER(flatNER bool) GLiNERLoaderOption {
	return func(c *glinerLoaderConfig) {
		c.flatNER = flatNER
	}
}

// WithGLiNERMultiLabel enables multi-label mode.
func WithGLiNERMultiLabel(multiLabel bool) GLiNERLoaderOption {
	return func(c *glinerLoaderConfig) {
		c.multiLabel = multiLabel
	}
}

// WithGLiNERLabels sets the default labels.
func WithGLiNERLabels(labels []string) GLiNERLoaderOption {
	return func(c *glinerLoaderConfig) {
		c.defaultLabels = labels
	}
}

// WithGLiNERQuantized uses quantized model files if available.
func WithGLiNERQuantized(quantized bool) GLiNERLoaderOption {
	return func(c *glinerLoaderConfig) {
		c.quantized = quantized
	}
}

// LoadGLiNERPipeline loads a GLiNER pipeline from a model directory.
// Returns the pipeline and the backend type that was used.
func LoadGLiNERPipeline(
	modelPath string,
	sessionManager *backends.SessionManager,
	modelBackends []string,
	opts ...GLiNERLoaderOption,
) (*GLiNERPipeline, backends.BackendType, error) {
	// Apply options
	loaderCfg := &glinerLoaderConfig{
		threshold: 0.5,
		maxWidth:  12,
		flatNER:   true,
	}
	for _, opt := range opts {
		opt(loaderCfg)
	}

	// Load model configuration
	modelConfig, err := LoadGLiNERModelConfig(modelPath)
	if err != nil {
		return nil, "", fmt.Errorf("loading GLiNER config: %w", err)
	}

	// Override config with loader options
	if loaderCfg.threshold > 0 {
		modelConfig.Threshold = loaderCfg.threshold
	}
	if loaderCfg.maxWidth > 0 {
		modelConfig.MaxWidth = loaderCfg.maxWidth
	}
	if loaderCfg.quantized {
		quantizedFile := FindONNXFile(modelPath, []string{"model_quantized.onnx"})
		if quantizedFile != "" {
			modelConfig.ModelFile = quantizedFile
		}
	}

	// Get a session factory for the model
	factory, backendType, err := sessionManager.GetSessionFactoryForModel(modelBackends)
	if err != nil {
		return nil, "", fmt.Errorf("getting session factory: %w", err)
	}

	// Load tokenizer
	tokenizer, err := LoadTokenizer(modelPath)
	if err != nil {
		return nil, "", fmt.Errorf("loading tokenizer: %w", err)
	}

	// Create session for the ONNX model
	session, err := factory.CreateSession(modelConfig.ModelFile)
	if err != nil {
		return nil, "", fmt.Errorf("creating session: %w", err)
	}

	// Build pipeline config
	pipelineConfig := &GLiNERPipelineConfig{
		Threshold:     modelConfig.Threshold,
		MaxWidth:      modelConfig.MaxWidth,
		FlatNER:       loaderCfg.flatNER || modelConfig.FlatNER,
		MultiLabel:    loaderCfg.multiLabel || modelConfig.MultiLabel,
		DefaultLabels: modelConfig.DefaultLabels,
	}

	if len(loaderCfg.defaultLabels) > 0 {
		pipelineConfig.DefaultLabels = loaderCfg.defaultLabels
	}

	pipeline := NewGLiNERPipeline(session, tokenizer, modelConfig, pipelineConfig, backendType)

	return pipeline, backendType, nil
}

// ============================================================================
// Helper Functions
// ============================================================================

// charOffsetFromRunes converts rune index to byte offset.
func charOffsetFromRunes(text string, runeIdx int) int {
	offset := 0
	for i := 0; i < runeIdx && offset < len(text); i++ {
		_, size := utf8.DecodeRuneInString(text[offset:])
		offset += size
	}
	return offset
}
