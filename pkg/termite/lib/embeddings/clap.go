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
	"strings"

	"github.com/antflydb/antfly-go/libaf/ai"
	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	"go.uber.org/zap"
)

// Ensure CLAPEmbedder implements the Embedder interface
var _ embeddings.Embedder = (*CLAPEmbedder)(nil)

// CLAPEmbedder wraps text and audio embedding pipelines for multimodal embedding.
// It uses pipelines.LoadCLAPPipelines() to load the model and supports
// both text-only and audio-only embedding requests.
// CLAP (Contrastive Language-Audio Pretraining) embeds audio and text into
// a shared embedding space, enabling cross-modal search between audio and text.
type CLAPEmbedder struct {
	textPipeline  *pipelines.EmbeddingPipeline
	audioPipeline *pipelines.EmbeddingPipeline
	backendType   backends.BackendType
	caps          embeddings.EmbedderCapabilities
	logger        *zap.Logger
}

// NewCLAPEmbedder creates a new CLAP-style multimodal embedder from a model path.
// Uses pipelines.LoadCLAPPipelines() to load text and audio encoders.
func NewCLAPEmbedder(
	modelPath string,
	quantized bool,
	sessionManager *backends.SessionManager,
	modelBackends []string,
	logger *zap.Logger,
) (*CLAPEmbedder, backends.BackendType, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Loading CLAP embedder",
		zap.String("modelPath", modelPath),
		zap.Bool("quantized", quantized))

	// Build loader options
	opts := []pipelines.EmbeddingLoaderOption{
		pipelines.WithEmbeddingNormalization(true),
		pipelines.WithQuantized(quantized),
	}

	// Load both text and audio pipelines
	textPipeline, audioPipeline, backendType, err := pipelines.LoadCLAPPipelines(
		modelPath,
		sessionManager,
		modelBackends,
		opts...,
	)
	if err != nil {
		return nil, "", fmt.Errorf("loading CLAP pipelines: %w", err)
	}

	// CLAP models should have both encoders
	if textPipeline == nil && audioPipeline == nil {
		return nil, "", fmt.Errorf("no text or audio encoder found in CLAP model at %s", modelPath)
	}

	// Build capabilities based on what pipelines are available
	caps := buildCLAPCapabilities(textPipeline, audioPipeline)

	logger.Info("Successfully loaded CLAP embedder",
		zap.String("backend", string(backendType)),
		zap.Bool("hasTextEncoder", textPipeline != nil),
		zap.Bool("hasAudioEncoder", audioPipeline != nil))

	return &CLAPEmbedder{
		textPipeline:  textPipeline,
		audioPipeline: audioPipeline,
		backendType:   backendType,
		caps:          caps,
		logger:        logger,
	}, backendType, nil
}

// buildCLAPCapabilities constructs EmbedderCapabilities based on available pipelines.
func buildCLAPCapabilities(textPipeline, audioPipeline *pipelines.EmbeddingPipeline) embeddings.EmbedderCapabilities {
	caps := embeddings.EmbedderCapabilities{
		SupportedMIMETypes: []embeddings.MIMETypeSupport{},
	}

	if textPipeline != nil {
		caps.SupportedMIMETypes = append(caps.SupportedMIMETypes,
			embeddings.MIMETypeSupport{MIMEType: "text/plain"})
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
func (e *CLAPEmbedder) Capabilities() embeddings.EmbedderCapabilities {
	return e.caps
}

// BackendType returns the backend type used by this embedder.
func (e *CLAPEmbedder) BackendType() backends.BackendType {
	return e.backendType
}

// Embed generates embeddings for the given content.
// Supports text content (via TextContent) and audio content (via BinaryContent).
// Each input can contain either text OR audio, but not both (no fusion support).
func (e *CLAPEmbedder) Embed(ctx context.Context, contents [][]ai.ContentPart) ([][]float32, error) {
	if len(contents) == 0 {
		return [][]float32{}, nil
	}

	results := make([][]float32, len(contents))

	// Batch text and audio inputs separately for efficiency
	textIndices := make([]int, 0)
	textInputs := make([]string, 0)
	audioIndices := make([]int, 0)
	audioInputs := make([][]byte, 0)

	for i, parts := range contents {
		text, audio, err := e.extractContent(parts)
		if err != nil {
			return nil, fmt.Errorf("extracting content at index %d: %w", i, err)
		}

		if text != "" {
			textIndices = append(textIndices, i)
			textInputs = append(textInputs, text)
		} else if audio != nil {
			audioIndices = append(audioIndices, i)
			audioInputs = append(audioInputs, audio)
		} else {
			return nil, fmt.Errorf("no text or audio content found at index %d", i)
		}
	}

	// Process text inputs one at a time (CLAP text model may only support batch_size=1)
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

// extractContent extracts text or audio from content parts.
// Returns (text, audioData, error). Only one of text or audioData will be non-empty/non-nil.
func (e *CLAPEmbedder) extractContent(parts []ai.ContentPart) (string, []byte, error) {
	for _, part := range parts {
		switch c := part.(type) {
		case ai.TextContent:
			if c.Text != "" {
				return c.Text, nil, nil
			}
		case ai.BinaryContent:
			if isAudioMIME(c.MIMEType) {
				// Return a copy to avoid aliasing issues
				audioCopy := make([]byte, len(c.Data))
				copy(audioCopy, c.Data)
				return "", audioCopy, nil
			}
		}
	}
	return "", nil, nil
}

// isAudioMIME checks if the MIME type is an audio type.
func isAudioMIME(mimeType string) bool {
	return strings.HasPrefix(mimeType, "audio/")
}

// Close releases resources held by the embedder.
func (e *CLAPEmbedder) Close() error {
	var errs []error

	if e.textPipeline != nil {
		if err := e.textPipeline.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing text pipeline: %w", err))
		}
	}

	if e.audioPipeline != nil {
		if err := e.audioPipeline.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing audio pipeline: %w", err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing CLAP embedder: %v", errs)
	}
	return nil
}

// EmbedAudioBytes is a convenience method to embed raw audio bytes directly.
// The audio should be in WAV format.
func (e *CLAPEmbedder) EmbedAudioBytes(ctx context.Context, audioData [][]byte) ([][]float32, error) {
	if e.audioPipeline == nil {
		return nil, fmt.Errorf("no audio encoder available")
	}
	return e.audioPipeline.EmbedAudio(ctx, audioData)
}

// EmbedAudio embeds a single audio file.
func (e *CLAPEmbedder) EmbedAudio(ctx context.Context, audioData []byte) ([]float32, error) {
	results, err := e.EmbedAudioBytes(ctx, [][]byte{audioData})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}
	return results[0], nil
}

// EmbedText embeds text using the text encoder.
func (e *CLAPEmbedder) EmbedText(ctx context.Context, text string) ([]float32, error) {
	if e.textPipeline == nil {
		return nil, fmt.Errorf("no text encoder available")
	}
	return e.textPipeline.EmbedOne(ctx, text)
}

// HasTextEncoder returns true if this embedder has a text encoder.
func (e *CLAPEmbedder) HasTextEncoder() bool {
	return e.textPipeline != nil
}

// HasAudioEncoder returns true if this embedder has an audio encoder.
func (e *CLAPEmbedder) HasAudioEncoder() bool {
	return e.audioPipeline != nil
}

// extractAudioFromReader extracts audio data from a reader.
// This is useful for streaming audio data.
func extractAudioFromReader(data []byte) ([]byte, error) {
	reader := bytes.NewReader(data)
	result := make([]byte, reader.Len())
	_, err := reader.Read(result)
	if err != nil {
		return nil, fmt.Errorf("reading audio data: %w", err)
	}
	return result, nil
}
