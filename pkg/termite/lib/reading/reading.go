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

// Package reading provides OCR and document understanding capabilities
// using Vision2Seq models like TrOCR, Donut, and Florence-2.
package reading

import (
	"context"
	"image"
)

// ModelType represents the type of Vision2Seq model for output parsing
type ModelType string

const (
	// ModelTypeTrOCR is a pure OCR model (microsoft/trocr-*)
	ModelTypeTrOCR ModelType = "trocr"
	// ModelTypeDonut is a document understanding model (naver-clova-ix/donut-*)
	ModelTypeDonut ModelType = "donut"
	// ModelTypeFlorence is a multi-task vision model (microsoft/Florence-2-*)
	ModelTypeFlorence ModelType = "florence"
	// ModelTypeGeneric is used when the model type is unknown
	ModelTypeGeneric ModelType = "generic"
)

// Result contains the output from reading an image.
type Result struct {
	// Text is the raw extracted text from the image
	Text string

	// Fields contains structured field values extracted by document understanding models.
	// Fields are flattened with dot notation for nested structures (e.g., "menu.nm", "menu.price").
	// This is populated by models like Donut that output structured data.
	Fields map[string]string
}

// Reader provides OCR and document understanding for images.
// It wraps Vision2Seq models (TrOCR, Donut, Florence-2) to extract text from images.
type Reader interface {
	// Read extracts text from the given images.
	// The optional prompt parameter allows specifying a task prompt for document understanding models:
	//   - TrOCR: prompt is ignored (pure OCR)
	//   - Donut CORD: "<s_cord-v2>" for receipt parsing
	//   - Donut DocVQA: "<s_docvqa><s_question>...</s_question><s_answer>" for visual QA
	//   - Florence-2: "<OCR>" for text extraction, "<CAPTION>" for captioning
	//
	// maxTokens limits the generated output length (0 uses model default).
	//
	// Returns one Result per input image.
	Read(ctx context.Context, images []image.Image, prompt string, maxTokens int) ([]Result, error)

	// Close releases model resources.
	Close() error
}
