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

package seq2seq

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

// RelationTriplet represents an extracted relation triplet.
type RelationTriplet struct {
	Subject  string  `json:"subject"`
	Object   string  `json:"object"`
	Relation string  `json:"relation"`
	Score    float32 `json:"score,omitempty"`
}

// RelationExtractionOutput represents the output from relation extraction.
type RelationExtractionOutput struct {
	// Triplets contains the extracted relation triplets for each input text.
	Triplets [][]RelationTriplet
	// RawOutputs contains the raw generated text from the model (for debugging).
	RawOutputs []string
}

// RelationExtractor is the interface for relation extraction models.
type RelationExtractor interface {
	// ExtractRelations extracts relation triplets from the given texts.
	ExtractRelations(ctx context.Context, texts []string) (*RelationExtractionOutput, error)

	// Close releases any resources held by the model.
	Close() error
}

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

// HugotREBEL implements the RelationExtractor interface using REBEL via Hugot.
type HugotREBEL struct {
	session       *khugot.Session
	pipeline      *pipelines.Seq2SeqPipeline
	logger        *zap.Logger
	sessionShared bool
	config        REBELConfig
}

// Ensure HugotREBEL implements RelationExtractor
var _ RelationExtractor = (*HugotREBEL)(nil)

// NewHugotREBEL creates a new REBEL model using the Hugot ONNX runtime.
func NewHugotREBEL(modelPath string, logger *zap.Logger) (*HugotREBEL, error) {
	return NewHugotREBELWithSession(modelPath, nil, logger)
}

// NewHugotREBELWithSession creates a new REBEL model with an optional shared session.
func NewHugotREBELWithSession(modelPath string, sharedSession *khugot.Session, logger *zap.Logger) (*HugotREBEL, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Initializing Hugot REBEL model",
		zap.String("modelPath", modelPath),
		zap.String("backend", hugot.BackendName()))

	// Load REBEL config
	config := REBELConfig{
		MaxLength:    256,
		NumBeams:     3,
		TripletToken: "<triplet>",
		SubjectToken: "<subj>",
		ObjectToken:  "<obj>",
		Task:         "relation_extraction",
	}
	configPath := filepath.Join(modelPath, "rebel_config.json")
	if data, err := os.ReadFile(configPath); err == nil {
		if err := json.Unmarshal(data, &config); err != nil {
			logger.Warn("Failed to parse REBEL config", zap.Error(err))
		} else {
			logger.Info("Loaded REBEL config",
				zap.String("model_id", config.ModelID),
				zap.Int("max_length", config.MaxLength),
				zap.Bool("multilingual", config.Multilingual))
		}
	}

	// Use shared session or create a new one
	session, err := hugot.NewSessionOrUseExisting(sharedSession)
	if err != nil {
		logger.Error("Failed to create Hugot session", zap.Error(err))
		return nil, fmt.Errorf("creating hugot session: %w", err)
	}
	sessionShared := (sharedSession != nil)

	// Create Seq2Seq pipeline for REBEL
	// IMPORTANT: REBEL uses special tokens (<triplet>, <subj>, <obj>) to structure output
	// These tokens are part of the model vocabulary, not special tokens that get stripped
	pipelineName := fmt.Sprintf("rebel:%s", filepath.Base(modelPath))
	pipelineOptions := []khugot.Seq2SeqOption{
		pipelines.WithSeq2SeqMaxTokens(config.MaxLength),
		pipelines.WithNumReturnSequences(1),
	}

	pipelineConfig := khugot.Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      pipelineName,
		Options:   pipelineOptions,
	}

	pipeline, err := khugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		if !sessionShared {
			session.Destroy()
		}
		logger.Error("Failed to create REBEL pipeline", zap.Error(err))
		return nil, fmt.Errorf("creating REBEL pipeline: %w", err)
	}

	logger.Info("REBEL model initialization complete",
		zap.String("model_id", config.ModelID),
		zap.Int("max_length", config.MaxLength))

	return &HugotREBEL{
		session:       session,
		pipeline:      pipeline,
		logger:        logger,
		sessionShared: sessionShared,
		config:        config,
	}, nil
}

// NewHugotREBELWithSessionManager creates a new REBEL model using a SessionManager.
func NewHugotREBELWithSessionManager(modelPath string, sessionManager *hugot.SessionManager, logger *zap.Logger) (*HugotREBEL, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	if sessionManager == nil {
		return NewHugotREBELWithSession(modelPath, nil, logger)
	}

	logger.Info("Initializing Hugot REBEL model with SessionManager",
		zap.String("modelPath", modelPath))

	// Load REBEL config
	config := REBELConfig{
		MaxLength:    256,
		NumBeams:     3,
		TripletToken: "<triplet>",
		SubjectToken: "<subj>",
		ObjectToken:  "<obj>",
		Task:         "relation_extraction",
	}
	configPath := filepath.Join(modelPath, "rebel_config.json")
	if data, err := os.ReadFile(configPath); err == nil {
		if err := json.Unmarshal(data, &config); err != nil {
			logger.Warn("Failed to parse REBEL config", zap.Error(err))
		}
	}

	// Get session from SessionManager
	session, _, err := sessionManager.GetSessionForModel(nil)
	if err != nil {
		logger.Error("Failed to get session from SessionManager", zap.Error(err))
		return nil, fmt.Errorf("getting session from SessionManager: %w", err)
	}

	// Create Seq2Seq pipeline for REBEL
	// IMPORTANT: REBEL uses special tokens (<triplet>, <subj>, <obj>) to structure output
	// These tokens are part of the model vocabulary, not special tokens that get stripped
	pipelineName := fmt.Sprintf("rebel:%s", filepath.Base(modelPath))
	pipelineOptions := []khugot.Seq2SeqOption{
		pipelines.WithSeq2SeqMaxTokens(config.MaxLength),
		pipelines.WithNumReturnSequences(1),
	}

	pipelineConfig := khugot.Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      pipelineName,
		Options:   pipelineOptions,
	}

	pipeline, err := khugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		logger.Error("Failed to create REBEL pipeline", zap.Error(err))
		return nil, fmt.Errorf("creating REBEL pipeline: %w", err)
	}

	logger.Info("REBEL model initialization complete",
		zap.String("model_id", config.ModelID),
		zap.Int("max_length", config.MaxLength))

	return &HugotREBEL{
		session:       session,
		pipeline:      pipeline,
		logger:        logger,
		sessionShared: true,
		config:        config,
	}, nil
}

// ExtractRelations extracts relation triplets from the given texts.
func (h *HugotREBEL) ExtractRelations(ctx context.Context, texts []string) (*RelationExtractionOutput, error) {
	if len(texts) == 0 {
		return &RelationExtractionOutput{
			Triplets:   [][]RelationTriplet{},
			RawOutputs: []string{},
		}, nil
	}

	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	h.logger.Debug("Starting REBEL relation extraction",
		zap.Int("num_inputs", len(texts)))

	// Run the pipeline
	output, err := h.pipeline.RunPipeline(texts)
	if err != nil {
		h.logger.Error("REBEL generation failed", zap.Error(err))
		return nil, fmt.Errorf("running REBEL pipeline: %w", err)
	}

	// Parse triplets from generated text
	result := &RelationExtractionOutput{
		Triplets:   make([][]RelationTriplet, len(texts)),
		RawOutputs: make([]string, len(texts)),
	}

	for i, generatedTexts := range output.GeneratedTexts {
		if len(generatedTexts) > 0 {
			rawOutput := generatedTexts[0]
			result.RawOutputs[i] = rawOutput
			result.Triplets[i] = h.parseREBELOutput(rawOutput)
			h.logger.Info("REBEL raw output",
				zap.Int("text_index", i),
				zap.String("raw_output", rawOutput),
				zap.Int("triplets_parsed", len(result.Triplets[i])))
		} else {
			result.Triplets[i] = []RelationTriplet{}
			h.logger.Warn("REBEL generated no output for text", zap.Int("text_index", i))
		}
	}

	h.logger.Info("REBEL relation extraction completed",
		zap.Int("num_inputs", len(texts)))

	return result, nil
}

// parseREBELOutput parses REBEL's generated text into structured triplets.
//
// REBEL output format:
// <s><triplet> Subject <subj> Object <obj> relation <triplet> Subject2 <subj> Object2 <obj> relation2 </s>
func (h *HugotREBEL) parseREBELOutput(text string) []RelationTriplet {
	triplets := []RelationTriplet{}

	// Remove start/end tokens
	text = strings.ReplaceAll(text, "<s>", "")
	text = strings.ReplaceAll(text, "</s>", "")
	text = strings.ReplaceAll(text, "<pad>", "")
	text = strings.TrimSpace(text)

	// Split by triplet token
	parts := strings.Split(text, h.config.TripletToken)

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		// Parse: "Subject <subj> Object <obj> relation"
		triplet := h.parseTripletPart(part)
		if triplet != nil {
			triplets = append(triplets, *triplet)
		}
	}

	return triplets
}

// parseTripletPart parses a single triplet from REBEL output.
func (h *HugotREBEL) parseTripletPart(part string) *RelationTriplet {
	subjToken := h.config.SubjectToken
	objToken := h.config.ObjectToken

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

	return &RelationTriplet{
		Subject:  subject,
		Object:   object,
		Relation: relation,
	}
}

// Config returns the REBEL configuration.
func (h *HugotREBEL) Config() REBELConfig {
	return h.config
}

// Close releases resources.
func (h *HugotREBEL) Close() error {
	var errs []error

	if h.pipeline != nil {
		if err := h.pipeline.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("destroying pipeline: %w", err))
		}
	}

	if h.session != nil && !h.sessionShared {
		h.logger.Info("Destroying Hugot session (owned by this REBEL model)")
		h.session.Destroy()
	}

	return errors.Join(errs...)
}

// IsREBELModel checks if the model path contains a REBEL model.
// It looks for rebel_config.json or "rebel" in the model name.
func IsREBELModel(modelPath string) bool {
	// Check for rebel_config.json
	configPath := filepath.Join(modelPath, "rebel_config.json")
	if _, err := os.Stat(configPath); err == nil {
		return true
	}

	// Check if model name contains "rebel"
	modelName := strings.ToLower(filepath.Base(modelPath))
	return strings.Contains(modelName, "rebel")
}
