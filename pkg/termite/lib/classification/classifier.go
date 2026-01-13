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

// Package classification provides text classification functionality using NLI models.
// Supports zero-shot classification, allowing text to be classified into arbitrary
// categories without requiring training data for those specific categories.
package classification

import (
	"context"
)

// Classification represents a single classification prediction.
type Classification struct {
	// Label is the predicted category/class
	Label string `json:"label"`
	// Score is the confidence score (0.0 to 1.0)
	Score float32 `json:"score"`
}

// Classifier defines the interface for Zero-Shot Classification models.
// These models use Natural Language Inference (NLI) to classify text into
// arbitrary categories without requiring category-specific training.
type Classifier interface {
	// Classify classifies the given texts using the specified candidate labels.
	// Returns a slice of classification results for each input text.
	// Each result is a slice of Classifications, one per label, sorted by score descending.
	Classify(ctx context.Context, texts []string, labels []string) ([][]Classification, error)

	// ClassifyWithHypothesis classifies texts using a custom hypothesis template.
	// The template should contain "{}" which will be replaced with each label.
	// Example: "This text is about {}." or "Este texto trata sobre {}."
	ClassifyWithHypothesis(ctx context.Context, texts []string, labels []string, hypothesisTemplate string) ([][]Classification, error)

	// MultiLabelClassify classifies texts allowing multiple labels per text.
	// Unlike Classify which normalizes scores across labels (single-label),
	// this treats each label independently (multi-label classification).
	MultiLabelClassify(ctx context.Context, texts []string, labels []string) ([][]Classification, error)

	// Close releases any resources held by the classifier.
	Close() error
}

// DefaultHypothesisTemplate is the default template used for NLI-based zero-shot classification.
// The "{}" will be replaced with each candidate label.
const DefaultHypothesisTemplate = "This example is {}."

// Config holds configuration for zero-shot classification models.
type Config struct {
	// HypothesisTemplate is the template for constructing NLI hypotheses.
	// Use "{}" as placeholder for the label.
	// Default: "This example is {}."
	HypothesisTemplate string `json:"hypothesis_template,omitempty"`

	// MultiLabel enables multi-label classification mode.
	// When true, each label is evaluated independently.
	// When false (default), scores are normalized across all labels.
	MultiLabel bool `json:"multi_label,omitempty"`

	// Threshold is the minimum score for a classification to be included (0.0-1.0).
	// Only used in multi-label mode.
	Threshold float32 `json:"threshold,omitempty"`

	// ModelID is the original HuggingFace model identifier.
	ModelID string `json:"model_id,omitempty"`
}
