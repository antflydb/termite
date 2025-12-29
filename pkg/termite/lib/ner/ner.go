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

// Recognizer extends Model with zero-shot NER capabilities.
// Models implementing this interface can recognize any entity types specified
// at inference time without requiring model retraining (e.g., GLiNER).
type Recognizer interface {
	Model

	// RecognizeWithLabels extracts entities of the specified types.
	// For zero-shot models like GLiNER, labels can be arbitrary entity types.
	// For traditional NER models, labels should match the trained entity types.
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

// Extractor extends Recognizer with advanced extraction capabilities.
// Multitask models like GLiNER can extract relations and answer questions.
type Extractor interface {
	Recognizer

	// ExtractRelations extracts both entities and relationships between them.
	ExtractRelations(ctx context.Context, texts []string, entityLabels []string, relationLabels []string) ([][]Entity, [][]Relation, error)

	// ExtractAnswers performs extractive question answering.
	// Given questions and contexts, extracts answer spans from the contexts.
	ExtractAnswers(ctx context.Context, questions []string, contexts []string) ([]Answer, error)

	// RelationLabels returns the default relation labels this model uses.
	RelationLabels() []string
}
