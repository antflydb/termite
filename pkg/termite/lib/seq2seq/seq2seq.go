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

// Package seq2seq provides Seq2Seq text generation functionality using ONNX models.
package seq2seq

import (
	"context"
)

// GeneratedOutput represents the output from a seq2seq model.
type GeneratedOutput struct {
	// Texts contains the generated text sequences.
	// For each input, multiple sequences may be returned (depending on NumReturnSequences).
	Texts [][]string
	// Tokens contains the token IDs for each generated sequence.
	Tokens [][][]uint32
}

// Model is the interface for Seq2Seq text generation models.
// This includes encoder-decoder models like T5, FLAN-T5, BART, etc.
type Model interface {
	// Generate runs the seq2seq model on the given inputs.
	// Returns generated text sequences for each input.
	Generate(ctx context.Context, inputs []string) (*GeneratedOutput, error)

	// Close releases any resources held by the model.
	Close() error
}

// QuestionGenerator extends Model with question generation capabilities.
// This is specifically for LMQG-style models trained on SQuAD/QA datasets.
type QuestionGenerator interface {
	Model

	// GenerateQuestions generates questions given answer-context pairs.
	// The model will generate questions where the answer is the correct response.
	GenerateQuestions(ctx context.Context, pairs []AnswerContextPair) (*GeneratedOutput, error)
}

// AnswerContextPair holds an answer and its context for question generation.
type AnswerContextPair struct {
	// Answer is the text that should be the answer to the generated question.
	Answer string
	// Context is the passage containing the answer.
	Context string
}

// Config holds configuration for seq2seq models.
type Config struct {
	// ModelID is the original HuggingFace model ID.
	ModelID string `json:"model_id"`
	// Task indicates the model's intended use (e.g., "question_generation", "query_generation").
	Task string `json:"task"`
	// MaxLength is the maximum number of tokens to generate.
	MaxLength int `json:"max_length"`
	// NumBeams is the number of beams for beam search (1 = greedy).
	NumBeams int `json:"num_beams"`
	// NumReturnSequences is how many sequences to return per input.
	NumReturnSequences int `json:"num_return_sequences"`
	// InputFormat describes the expected input format (e.g., "generate question: <hl> {answer} <hl> {context}").
	InputFormat string `json:"input_format"`
	// DoSample enables sampling instead of greedy/beam search.
	DoSample bool `json:"do_sample"`
	// TopP is the nucleus sampling probability (used when DoSample=true).
	TopP float32 `json:"top_p"`
	// Temperature controls randomness (used when DoSample=true).
	Temperature float32 `json:"temperature"`
}

// FormatLMQGInput formats input for LMQG question generation models.
// These models expect input in the format: "generate question: <hl> {answer} <hl> {context}"
// where the answer is highlighted within the context.
//
// Example:
//
//	input := FormatLMQGInput("Beyonce", "Beyonce starred as Etta James in Cadillac Records.")
//	// Returns: "generate question: <hl> Beyonce <hl> Beyonce starred as Etta James in Cadillac Records."
func FormatLMQGInput(answer, context string) string {
	return "generate question: <hl> " + answer + " <hl> " + context
}

// FormatLMQGInputBatch formats multiple answer-context pairs for LMQG models.
func FormatLMQGInputBatch(pairs []AnswerContextPair) []string {
	inputs := make([]string, len(pairs))
	for i, pair := range pairs {
		inputs[i] = FormatLMQGInput(pair.Answer, pair.Context)
	}
	return inputs
}
