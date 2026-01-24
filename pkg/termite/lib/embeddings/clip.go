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

package embeddings

import (
	"bytes"
	"context"
	"fmt"
	"image"
	_ "image/jpeg" // Register JPEG decoder
	_ "image/png"  // Register PNG decoder
	"strings"

	"github.com/antflydb/antfly-go/libaf/ai"
	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	"go.uber.org/zap"
)

// Ensure CLIPEmbedder implements the Embedder interface
var _ embeddings.Embedder = (*CLIPEmbedder)(nil)

// CLIPEmbedder wraps text and visual embedding pipelines for multimodal embedding.
// It uses pipelines.LoadEmbeddingPipelines() to load the model and supports
// both text-only and image-only embedding requests.
type CLIPEmbedder struct {
	textPipeline   *pipelines.EmbeddingPipeline
	visualPipeline *pipelines.EmbeddingPipeline
	backendType    backends.BackendType
	caps           embeddings.EmbedderCapabilities
	logger         *zap.Logger
}

// NewCLIPEmbedder creates a new CLIP-style multimodal embedder from a model path.
// Uses pipelines.LoadEmbeddingPipelines() to load text and visual encoders.
func NewCLIPEmbedder(
	modelPath string,
	quantized bool,
	sessionManager *backends.SessionManager,
	modelBackends []string,
	logger *zap.Logger,
) (*CLIPEmbedder, backends.BackendType, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Loading CLIP embedder",
		zap.String("modelPath", modelPath),
		zap.Bool("quantized", quantized))

	// Build loader options
	opts := []pipelines.EmbeddingLoaderOption{
		pipelines.WithEmbeddingNormalization(true),
		pipelines.WithQuantized(quantized),
	}

	// Load both text and visual pipelines
	textPipeline, visualPipeline, backendType, err := pipelines.LoadEmbeddingPipelines(
		modelPath,
		sessionManager,
		modelBackends,
		opts...,
	)
	if err != nil {
		return nil, "", fmt.Errorf("loading CLIP pipelines: %w", err)
	}

	// CLIP models should have both encoders
	if textPipeline == nil && visualPipeline == nil {
		return nil, "", fmt.Errorf("no text or visual encoder found in CLIP model at %s", modelPath)
	}

	// Build capabilities based on what pipelines are available
	caps := buildCLIPCapabilities(textPipeline, visualPipeline)

	logger.Info("Successfully loaded CLIP embedder",
		zap.String("backend", string(backendType)),
		zap.Bool("hasTextEncoder", textPipeline != nil),
		zap.Bool("hasVisualEncoder", visualPipeline != nil))

	return &CLIPEmbedder{
		textPipeline:   textPipeline,
		visualPipeline: visualPipeline,
		backendType:    backendType,
		caps:           caps,
		logger:         logger,
	}, backendType, nil
}

// buildCLIPCapabilities constructs EmbedderCapabilities based on available pipelines.
func buildCLIPCapabilities(textPipeline, visualPipeline *pipelines.EmbeddingPipeline) embeddings.EmbedderCapabilities {
	caps := embeddings.EmbedderCapabilities{
		SupportedMIMETypes: []embeddings.MIMETypeSupport{},
	}

	if textPipeline != nil {
		caps.SupportedMIMETypes = append(caps.SupportedMIMETypes,
			embeddings.MIMETypeSupport{MIMEType: "text/plain"})
	}

	if visualPipeline != nil {
		caps.SupportedMIMETypes = append(caps.SupportedMIMETypes,
			embeddings.MIMETypeSupport{MIMEType: "image/jpeg"},
			embeddings.MIMETypeSupport{MIMEType: "image/png"},
			embeddings.MIMETypeSupport{MIMEType: "image/*"})
	}

	return caps
}

// Capabilities returns the capabilities of this embedder.
func (e *CLIPEmbedder) Capabilities() embeddings.EmbedderCapabilities {
	return e.caps
}

// BackendType returns the backend type used by this embedder.
func (e *CLIPEmbedder) BackendType() backends.BackendType {
	return e.backendType
}

// Embed generates embeddings for the given content.
// Supports text content (via TextContent) and image content (via BinaryContent).
// Each input can contain either text OR an image, but not both (no fusion support).
func (e *CLIPEmbedder) Embed(ctx context.Context, contents [][]ai.ContentPart) ([][]float32, error) {
	if len(contents) == 0 {
		return [][]float32{}, nil
	}

	results := make([][]float32, len(contents))

	// Batch text and image inputs separately for efficiency
	textIndices := make([]int, 0)
	textInputs := make([]string, 0)
	imageIndices := make([]int, 0)
	imageInputs := make([]image.Image, 0)

	for i, parts := range contents {
		text, img, err := e.extractContent(parts)
		if err != nil {
			return nil, fmt.Errorf("extracting content at index %d: %w", i, err)
		}

		if text != "" {
			textIndices = append(textIndices, i)
			textInputs = append(textInputs, text)
		} else if img != nil {
			imageIndices = append(imageIndices, i)
			imageInputs = append(imageInputs, img)
		} else {
			return nil, fmt.Errorf("no text or image content found at index %d", i)
		}
	}

	// Process text inputs one at a time (CLIP text model only supports batch_size=1)
	if len(textInputs) > 0 {
		if e.textPipeline == nil {
			return nil, fmt.Errorf("text embedding requested but no text encoder available")
		}

		for i, text := range textInputs {
			embedding, err := e.textPipeline.EmbedOne(ctx, text)
			if err != nil {
				return nil, fmt.Errorf("embedding text %d: %w", i, err)
			}
			results[textIndices[i]] = embedding
		}
	}

	// Process image batch
	if len(imageInputs) > 0 {
		if e.visualPipeline == nil {
			return nil, fmt.Errorf("image embedding requested but no visual encoder available")
		}

		imageEmbeddings, err := e.visualPipeline.EmbedImages(ctx, imageInputs)
		if err != nil {
			return nil, fmt.Errorf("embedding images: %w", err)
		}

		for i, idx := range imageIndices {
			results[idx] = imageEmbeddings[i]
		}
	}

	return results, nil
}

// extractContent extracts text or image from content parts.
// Returns (text, image, error). Only one of text or image will be non-empty/non-nil.
func (e *CLIPEmbedder) extractContent(parts []ai.ContentPart) (string, image.Image, error) {
	for _, part := range parts {
		switch c := part.(type) {
		case ai.TextContent:
			if c.Text != "" {
				return c.Text, nil, nil
			}
		case ai.BinaryContent:
			if isImageMIME(c.MIMEType) {
				img, _, err := image.Decode(bytes.NewReader(c.Data))
				if err != nil {
					return "", nil, fmt.Errorf("decoding image: %w", err)
				}
				return "", img, nil
			}
		case ai.ImageURLContent:
			// URL content is not directly supported - treat as text for text embedding
			// This maintains backward compatibility
			if c.URL != "" {
				return c.URL, nil, nil
			}
		}
	}
	return "", nil, nil
}

// isImageMIME checks if the MIME type is an image type.
func isImageMIME(mimeType string) bool {
	return strings.HasPrefix(mimeType, "image/")
}

// Close releases resources held by the embedder.
func (e *CLIPEmbedder) Close() error {
	var errs []error

	if e.textPipeline != nil {
		if err := e.textPipeline.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing text pipeline: %w", err))
		}
	}

	if e.visualPipeline != nil {
		if err := e.visualPipeline.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing visual pipeline: %w", err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing CLIP embedder: %v", errs)
	}
	return nil
}
