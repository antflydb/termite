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

	"github.com/antflydb/antfly-go/libaf/ai"
	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	"go.uber.org/zap"
)

// Ensure CLIPCLAPEmbedder implements the Embedder interface
var _ embeddings.Embedder = (*CLIPCLAPEmbedder)(nil)

// CLIPCLAPEmbedder wraps text, visual, and audio embedding pipelines for unified
// multimodal embedding. It combines CLIP (text+image) and CLAP (audio) with a
// trained projection layer that maps audio embeddings into CLIP space, enabling
// cross-modal search between text, images, and audio in a single 512-dim space.
type CLIPCLAPEmbedder struct {
	textPipeline   *pipelines.EmbeddingPipeline
	visualPipeline *pipelines.EmbeddingPipeline
	audioPipeline  *pipelines.EmbeddingPipeline
	backendType    backends.BackendType
	caps           embeddings.EmbedderCapabilities
	logger         *zap.Logger
}

// NewCLIPCLAPEmbedder creates a new unified CLIP+CLAP multimodal embedder from a model path.
// Uses pipelines.LoadEmbeddingPipelines() to load text, visual, and audio encoders.
// The audio pipeline automatically picks up audio_projection.onnx as its projector.
func NewCLIPCLAPEmbedder(
	modelPath string,
	quantized bool,
	sessionManager *backends.SessionManager,
	modelBackends []string,
	logger *zap.Logger,
) (*CLIPCLAPEmbedder, backends.BackendType, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Loading CLIPCLAP embedder",
		zap.String("modelPath", modelPath),
		zap.Bool("quantized", quantized))

	// Build loader options
	opts := []pipelines.EmbeddingLoaderOption{
		pipelines.WithEmbeddingNormalization(true),
		pipelines.WithQuantized(quantized),
	}

	// Load all three pipelines
	textPipeline, visualPipeline, audioPipeline, backendType, err := pipelines.LoadEmbeddingPipelines(
		modelPath,
		sessionManager,
		modelBackends,
		opts...,
	)
	if err != nil {
		return nil, "", fmt.Errorf("loading CLIPCLAP pipelines: %w", err)
	}

	// CLIPCLAP models should have all three encoders
	if textPipeline == nil && visualPipeline == nil && audioPipeline == nil {
		return nil, "", fmt.Errorf("no text, visual, or audio encoder found in CLIPCLAP model at %s", modelPath)
	}

	// Build capabilities based on what pipelines are available
	caps := buildCLIPCLAPCapabilities(textPipeline, visualPipeline, audioPipeline)

	logger.Info("Successfully loaded CLIPCLAP embedder",
		zap.String("backend", string(backendType)),
		zap.Bool("hasTextEncoder", textPipeline != nil),
		zap.Bool("hasVisualEncoder", visualPipeline != nil),
		zap.Bool("hasAudioEncoder", audioPipeline != nil))

	return &CLIPCLAPEmbedder{
		textPipeline:   textPipeline,
		visualPipeline: visualPipeline,
		audioPipeline:  audioPipeline,
		backendType:    backendType,
		caps:           caps,
		logger:         logger,
	}, backendType, nil
}

// buildCLIPCLAPCapabilities constructs EmbedderCapabilities based on available pipelines.
func buildCLIPCLAPCapabilities(textPipeline, visualPipeline, audioPipeline *pipelines.EmbeddingPipeline) embeddings.EmbedderCapabilities {
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

	if audioPipeline != nil {
		caps.SupportedMIMETypes = append(caps.SupportedMIMETypes,
			embeddings.MIMETypeSupport{MIMEType: "audio/wav"},
			embeddings.MIMETypeSupport{MIMEType: "audio/wave"},
			embeddings.MIMETypeSupport{MIMEType: "audio/x-wav"},
			embeddings.MIMETypeSupport{MIMEType: "audio/*"})
	}

	return caps
}

// Capabilities returns the capabilities of this embedder.
func (e *CLIPCLAPEmbedder) Capabilities() embeddings.EmbedderCapabilities {
	return e.caps
}

// BackendType returns the backend type used by this embedder.
func (e *CLIPCLAPEmbedder) BackendType() backends.BackendType {
	return e.backendType
}

// Embed generates embeddings for the given content.
// Supports text content (via TextContent), image content, and audio content (via BinaryContent).
// Each input can contain text, an image, OR audio, but not multiple modalities.
func (e *CLIPCLAPEmbedder) Embed(ctx context.Context, contents [][]ai.ContentPart) ([][]float32, error) {
	if len(contents) == 0 {
		return [][]float32{}, nil
	}

	results := make([][]float32, len(contents))

	// Batch inputs by modality for efficiency
	textIndices := make([]int, 0)
	textInputs := make([]string, 0)
	imageIndices := make([]int, 0)
	imageInputs := make([]image.Image, 0)
	audioIndices := make([]int, 0)
	audioInputs := make([][]byte, 0)

	for i, parts := range contents {
		text, img, audio, err := e.extractContent(parts)
		if err != nil {
			return nil, fmt.Errorf("extracting content at index %d: %w", i, err)
		}

		if text != "" {
			textIndices = append(textIndices, i)
			textInputs = append(textInputs, text)
		} else if img != nil {
			imageIndices = append(imageIndices, i)
			imageInputs = append(imageInputs, img)
		} else if audio != nil {
			audioIndices = append(audioIndices, i)
			audioInputs = append(audioInputs, audio)
		} else {
			return nil, fmt.Errorf("no text, image, or audio content found at index %d", i)
		}
	}

	// Process text inputs one at a time (text models may only support batch_size=1)
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

	// Process audio inputs
	if len(audioInputs) > 0 {
		if e.audioPipeline == nil {
			return nil, fmt.Errorf("audio embedding requested but no audio encoder available")
		}

		audioEmbeddings, err := e.audioPipeline.EmbedAudio(ctx, audioInputs)
		if err != nil {
			return nil, fmt.Errorf("embedding audio: %w", err)
		}

		for i, idx := range audioIndices {
			results[idx] = audioEmbeddings[i]
		}
	}

	return results, nil
}

// extractContent extracts text, image, or audio from content parts.
// Returns (text, image, audioData, error). Only one will be non-empty/non-nil.
func (e *CLIPCLAPEmbedder) extractContent(parts []ai.ContentPart) (string, image.Image, []byte, error) {
	for _, part := range parts {
		switch c := part.(type) {
		case ai.TextContent:
			if c.Text != "" {
				return c.Text, nil, nil, nil
			}
		case ai.BinaryContent:
			if isImageMIME(c.MIMEType) {
				img, _, err := image.Decode(bytes.NewReader(c.Data))
				if err != nil {
					return "", nil, nil, fmt.Errorf("decoding image: %w", err)
				}
				return "", img, nil, nil
			}
			if isAudioMIME(c.MIMEType) {
				audioCopy := make([]byte, len(c.Data))
				copy(audioCopy, c.Data)
				return "", nil, audioCopy, nil
			}
		case ai.ImageURLContent:
			if c.URL != "" {
				return c.URL, nil, nil, nil
			}
		}
	}
	return "", nil, nil, nil
}

// Close releases resources held by the embedder.
func (e *CLIPCLAPEmbedder) Close() error {
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

	if e.audioPipeline != nil {
		if err := e.audioPipeline.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing audio pipeline: %w", err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing CLIPCLAP embedder: %v", errs)
	}
	return nil
}
