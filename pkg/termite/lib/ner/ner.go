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

// Package ner provides Named Entity Recognition functionality using ONNX models.
package ner

import (
	"context"
)

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

// GLiNERModel extends Model with zero-shot NER capabilities.
// GLiNER models can recognize any entity types specified at inference time
// without requiring model retraining.
type GLiNERModel interface {
	Model

	// RecognizeWithLabels extracts entities of the specified types (zero-shot NER).
	// This is the key feature of GLiNER - it can extract any entity type without retraining.
	RecognizeWithLabels(ctx context.Context, texts []string, labels []string) ([][]Entity, error)

	// Labels returns the default entity labels this model uses.
	Labels() []string
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

// RelationExtractionModel extends GLiNERModel with relationship extraction capabilities.
type RelationExtractionModel interface {
	GLiNERModel

	// RecognizeWithRelations extracts both entities and relationships between them.
	RecognizeWithRelations(ctx context.Context, texts []string, entityLabels []string, relationLabels []string) ([][]Entity, [][]Relation, error)

	// RelationLabels returns the default relation labels this model uses.
	RelationLabels() []string
}
