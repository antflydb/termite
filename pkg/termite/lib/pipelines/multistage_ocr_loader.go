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
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
)

// MultiStageMetadata describes the stages in a multi-stage OCR model.
// This is read from termite_metadata.json in the model directory.
type MultiStageMetadata struct {
	ModelType    string                     `json:"model_type"`
	PipelineType string                    `json:"pipeline_type"`
	Stages       map[string]StageMetadata  `json:"stages"`
}

// StageMetadata describes a single stage in a multi-stage model.
type StageMetadata struct {
	// ModelFile is the ONNX model filename (for single-session stages like detection, layout, order).
	ModelFile string `json:"model_file,omitempty"`
	// Type is the recognition type: "vision2seq" or "ctc".
	Type string `json:"type,omitempty"`
	// EncoderFile is the encoder ONNX filename (for vision2seq recognition).
	EncoderFile string `json:"encoder_file,omitempty"`
	// DecoderFile is the decoder ONNX filename (for vision2seq recognition).
	DecoderFile string `json:"decoder_file,omitempty"`
	// PostProcessor is the detection post-processor type: "heatmap" or "db".
	PostProcessor string `json:"post_processor,omitempty"`
	// CharDictFile is the character dictionary filename (for CTC recognition).
	CharDictFile string `json:"char_dict_file,omitempty"`
}

// IsMultiStageModel checks if a model directory contains a multi-stage OCR model
// by reading termite_metadata.json.
func IsMultiStageModel(modelPath string) bool {
	meta, err := LoadMultiStageMetadata(modelPath)
	if err != nil {
		return false
	}
	return meta.PipelineType == "multistage_ocr"
}

// LoadMultiStageMetadata reads the multi-stage metadata from a model directory.
func LoadMultiStageMetadata(modelPath string) (*MultiStageMetadata, error) {
	metaPath := filepath.Join(modelPath, "termite_metadata.json")
	data, err := os.ReadFile(metaPath)
	if err != nil {
		return nil, fmt.Errorf("reading metadata: %w", err)
	}

	var meta MultiStageMetadata
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil, fmt.Errorf("parsing metadata: %w", err)
	}

	return &meta, nil
}

// LoadMultiStageOCRPipeline loads a multi-stage OCR pipeline from a model directory.
// It reads termite_metadata.json to determine which stages to load, following the
// Florence-2 pattern of cascading session creation with cleanup on error
// (florence2.go:85-155).
func LoadMultiStageOCRPipeline(
	modelPath string,
	sessionManager *backends.SessionManager,
	modelBackends []string,
) (*MultiStageOCRPipeline, backends.BackendType, error) {
	// Read metadata
	meta, err := LoadMultiStageMetadata(modelPath)
	if err != nil {
		return nil, "", fmt.Errorf("loading metadata: %w", err)
	}

	if meta.PipelineType != "multistage_ocr" {
		return nil, "", fmt.Errorf("not a multi-stage OCR model: pipeline_type=%s", meta.PipelineType)
	}

	// Get session factory
	factory, backendType, err := sessionManager.GetSessionFactoryForModel(modelBackends)
	if err != nil {
		return nil, "", fmt.Errorf("getting session factory: %w", err)
	}

	// Load detection stage (required)
	detStage, ok := meta.Stages["detection"]
	if !ok {
		return nil, "", fmt.Errorf("no detection stage in metadata")
	}

	detPath := filepath.Join(modelPath, detStage.ModelFile)
	detSession, err := factory.CreateSession(detPath)
	if err != nil {
		return nil, "", fmt.Errorf("creating detection session: %w", err)
	}

	// Determine detection post-processor
	var detProcessor DetectionPostProcessor
	switch detStage.PostProcessor {
	case "db":
		detProcessor = NewDBPostProcessor(0.3, 0.5, 1.5, 10)
	case "heatmap":
		detProcessor = NewHeatmapPostProcessor(0.5, 50)
	default:
		detSession.Close()
		return nil, "", fmt.Errorf("unknown post-processor: %s", detStage.PostProcessor)
	}

	// Create detection image processor with defaults
	detImgProc := NewImageProcessor(backends.DefaultImageConfig())

	// Load recognition stage (required)
	recStage, ok := meta.Stages["recognition"]
	if !ok {
		detSession.Close()
		return nil, "", fmt.Errorf("no recognition stage in metadata")
	}

	var recognizer Recognizer

	switch recStage.Type {
	case "vision2seq":
		// Surya recognition: reuse existing Vision2SeqPipeline
		recPipeline, _, err := LoadVision2SeqPipeline(modelPath, sessionManager, modelBackends)
		if err != nil {
			detSession.Close()
			return nil, "", fmt.Errorf("loading Vision2Seq recognizer: %w", err)
		}
		recognizer = NewVision2SeqRecognizer(recPipeline)

	case "ctc":
		// PaddleOCR recognition: CTC decoder
		recPath := filepath.Join(modelPath, recStage.ModelFile)
		recSession, err := factory.CreateSession(recPath)
		if err != nil {
			detSession.Close()
			return nil, "", fmt.Errorf("creating CTC recognition session: %w", err)
		}

		// Load character dictionary
		dictPath := filepath.Join(modelPath, recStage.CharDictFile)
		charDict, err := loadCharDictFile(dictPath)
		if err != nil {
			detSession.Close()
			recSession.Close()
			return nil, "", fmt.Errorf("loading char dict: %w", err)
		}

		recImgProc := NewImageProcessor(backends.DefaultImageConfig())
		recognizer = NewCTCRecognizer(recSession, charDict, recImgProc)

	default:
		detSession.Close()
		return nil, "", fmt.Errorf("unknown recognition type: %s", recStage.Type)
	}

	// Build pipeline config
	config := &MultiStageOCRConfig{
		DetConfig: &DetectionConfig{
			InputWidth:  detImgProc.Config.Width,
			InputHeight: detImgProc.Config.Height,
			Threshold:   0.5,
			MinBoxArea:  50,
		},
		RecConfig: &RecognitionConfig{
			InputHeight: 48,
			InputWidth:  320,
		},
	}

	pipeline := NewMultiStageOCRPipeline(detSession, recognizer, detProcessor, detImgProc, config)

	// Load optional layout stage
	if layoutStage, ok := meta.Stages["layout"]; ok {
		layoutPath := filepath.Join(modelPath, layoutStage.ModelFile)
		layoutSession, err := factory.CreateSession(layoutPath)
		if err != nil {
			pipeline.Close()
			return nil, "", fmt.Errorf("creating layout session: %w", err)
		}
		pipeline.SetLayout(layoutSession)
		config.HasLayout = true
	}

	// Load optional order stage
	if orderStage, ok := meta.Stages["order"]; ok {
		orderPath := filepath.Join(modelPath, orderStage.ModelFile)
		orderSession, err := factory.CreateSession(orderPath)
		if err != nil {
			pipeline.Close()
			return nil, "", fmt.Errorf("creating order session: %w", err)
		}
		pipeline.SetOrder(orderSession)
		config.HasOrder = true
	}

	return pipeline, backendType, nil
}

// loadCharDictFile reads a character dictionary from a text file (one char per line).
func loadCharDictFile(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening char dict: %w", err)
	}
	defer f.Close()

	var dict []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		dict = append(dict, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scanning char dict: %w", err)
	}

	return dict, nil
}
