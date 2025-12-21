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

	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
	"go.uber.org/zap"
)

// Ensure HugotSeq2Seq implements the Model and QuestionGenerator interfaces
var _ Model = (*HugotSeq2Seq)(nil)
var _ QuestionGenerator = (*HugotSeq2Seq)(nil)

// HugotSeq2Seq implements the Model and QuestionGenerator interfaces using Hugot's Seq2SeqPipeline.
type HugotSeq2Seq struct {
	session       *khugot.Session
	pipeline      *pipelines.Seq2SeqPipeline
	logger        *zap.Logger
	sessionShared bool
	config        Config
}

// NewHugotSeq2Seq creates a new Seq2Seq model using the Hugot ONNX runtime.
func NewHugotSeq2Seq(modelPath string, logger *zap.Logger) (*HugotSeq2Seq, error) {
	return NewHugotSeq2SeqWithSession(modelPath, nil, logger)
}

// NewHugotSeq2SeqWithSession creates a new Seq2Seq model with an optional shared session.
func NewHugotSeq2SeqWithSession(modelPath string, sharedSession *khugot.Session, logger *zap.Logger) (*HugotSeq2Seq, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Initializing Hugot Seq2Seq model",
		zap.String("modelPath", modelPath),
		zap.String("backend", hugot.BackendName()))

	// Load seq2seq config if available
	config := Config{
		MaxLength:          64,
		NumBeams:           1,
		NumReturnSequences: 1,
		DoSample:           false,
		TopP:               0.9,
		Temperature:        1.0,
	}
	configPath := filepath.Join(modelPath, "seq2seq_config.json")
	if data, err := os.ReadFile(configPath); err == nil {
		if err := json.Unmarshal(data, &config); err != nil {
			logger.Warn("Failed to parse seq2seq config", zap.Error(err))
		} else {
			logger.Info("Loaded seq2seq config",
				zap.String("task", config.Task),
				zap.Int("max_length", config.MaxLength),
				zap.Int("num_return_sequences", config.NumReturnSequences))
		}
	}

	// Use shared session or create a new one
	session, err := hugot.NewSessionOrUseExisting(sharedSession)
	if err != nil {
		logger.Error("Failed to create Hugot session", zap.Error(err))
		return nil, fmt.Errorf("creating hugot session: %w", err)
	}
	sessionShared := (sharedSession != nil)

	// Create Seq2Seq pipeline configuration
	pipelineName := fmt.Sprintf("seq2seq:%s", filepath.Base(modelPath))
	pipelineOptions := []khugot.Seq2SeqOption{
		pipelines.WithSeq2SeqMaxTokens(config.MaxLength),
		pipelines.WithNumReturnSequences(config.NumReturnSequences),
	}

	// Apply sampling if configured
	if config.DoSample && config.TopP > 0 && config.Temperature > 0 {
		pipelineOptions = append(pipelineOptions,
			pipelines.WithSampling(config.TopP, config.Temperature))
	}

	pipelineConfig := khugot.Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      pipelineName,
		Options:   pipelineOptions,
	}

	// Use NewPipeline to create the seq2seq pipeline through the session
	pipeline, err := khugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		if !sessionShared {
			session.Destroy()
		}
		logger.Error("Failed to create Seq2Seq pipeline", zap.Error(err))
		return nil, fmt.Errorf("creating Seq2Seq pipeline: %w", err)
	}

	logger.Info("Seq2Seq model initialization complete",
		zap.String("task", config.Task),
		zap.Int("max_length", config.MaxLength))

	return &HugotSeq2Seq{
		session:       session,
		pipeline:      pipeline,
		logger:        logger,
		sessionShared: sessionShared,
		config:        config,
	}, nil
}

// Generate runs the seq2seq model on the given inputs.
func (h *HugotSeq2Seq) Generate(ctx context.Context, inputs []string) (*GeneratedOutput, error) {
	if len(inputs) == 0 {
		return &GeneratedOutput{
			Texts:  [][]string{},
			Tokens: [][][]uint32{},
		}, nil
	}

	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	h.logger.Debug("Starting Seq2Seq generation",
		zap.Int("num_inputs", len(inputs)))

	// Run the pipeline
	output, err := h.pipeline.RunPipeline(inputs)
	if err != nil {
		h.logger.Error("Seq2Seq generation failed", zap.Error(err))
		return nil, fmt.Errorf("running Seq2Seq pipeline: %w", err)
	}

	h.logger.Debug("Seq2Seq generation completed",
		zap.Int("num_inputs", len(inputs)),
		zap.Int("total_outputs", len(output.GeneratedTexts)))

	return &GeneratedOutput{
		Texts:  output.GeneratedTexts,
		Tokens: output.GeneratedTokens,
	}, nil
}

// GenerateQuestions generates questions given answer-context pairs.
// This is a convenience method for LMQG-style models.
func (h *HugotSeq2Seq) GenerateQuestions(ctx context.Context, pairs []AnswerContextPair) (*GeneratedOutput, error) {
	if len(pairs) == 0 {
		return &GeneratedOutput{
			Texts:  [][]string{},
			Tokens: [][][]uint32{},
		}, nil
	}

	// Format inputs for LMQG models
	inputs := FormatLMQGInputBatch(pairs)

	return h.Generate(ctx, inputs)
}

// Config returns the model configuration.
func (h *HugotSeq2Seq) Config() Config {
	return h.config
}

// Close releases resources.
func (h *HugotSeq2Seq) Close() error {
	var errs []error

	if h.pipeline != nil {
		if err := h.pipeline.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("destroying pipeline: %w", err))
		}
	}

	if h.session != nil && !h.sessionShared {
		h.logger.Info("Destroying Hugot session (owned by this Seq2Seq model)")
		h.session.Destroy()
	}

	return errors.Join(errs...)
}

// IsSeq2SeqModel checks if the model path contains a seq2seq model
// by looking for required ONNX files (encoder.onnx, decoder-init.onnx, decoder.onnx).
func IsSeq2SeqModel(modelPath string) bool {
	requiredFiles := []string{"encoder.onnx", "decoder-init.onnx", "decoder.onnx"}
	for _, file := range requiredFiles {
		filePath := filepath.Join(modelPath, file)
		if _, err := os.Stat(filePath); err != nil {
			return false
		}
	}
	return true
}

// IsQuestionGenerationModel checks if the model is configured for question generation.
func IsQuestionGenerationModel(modelPath string) bool {
	if !IsSeq2SeqModel(modelPath) {
		return false
	}

	// Check config for task type
	configPath := filepath.Join(modelPath, "seq2seq_config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		// Check model name for hints
		modelName := filepath.Base(modelPath)
		return containsAny(modelName, "qg", "question", "squad")
	}

	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return false
	}

	return config.Task == "question_generation"
}

// containsAny checks if s contains any of the substrings (case-insensitive).
func containsAny(s string, substrings ...string) bool {
	sLower := filepath.Base(s)
	for _, sub := range substrings {
		if len(sub) > 0 {
			for i := 0; i <= len(sLower)-len(sub); i++ {
				if sLower[i:i+len(sub)] == sub {
					return true
				}
			}
		}
	}
	return false
}
