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

//go:build onnx && ORT

package embeddings

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"

	"github.com/antflydb/antfly-go/libaf/ai"
	libafembed "github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/util/imageutil"
	"go.uber.org/zap"
	_ "golang.org/x/image/webp"
)

// HugotCLIPEmbedder implements multimodal embeddings using CLIP models via hugot pipelines.
// It uses hugot's FeatureExtractionPipeline with ImageMode for the visual encoder
// and a regular text pipeline for the text encoder.
//
// Build with: CGO_ENABLED=1 go build -tags="onnx,ORT"
type HugotCLIPEmbedder struct {
	visualPipeline *pipelines.FeatureExtractionPipeline
	textPipeline   *pipelines.FeatureExtractionPipeline
	session        *khugot.Session
	config         *HugotCLIPConfig
	logger         *zap.Logger
	caps           libafembed.EmbedderCapabilities
	modelPath      string
	sessionShared  bool
}

// HugotCLIPConfig holds the CLIP model configuration
type HugotCLIPConfig struct {
	ModelType     string `json:"model_type"`
	ProjectionDim int    `json:"projection_dim"`
	ImageSize     int    `json:"image_size"`
}

// NewHugotCLIPEmbedder creates a new CLIP embedder using hugot pipelines.
// The directory should contain:
//   - visual_model.onnx (or visual_model_quantized.onnx)
//   - text_model.onnx (or text_model_quantized.onnx)
//   - config.json or clip_config.json
//   - tokenizer.json
//
// Build with -tags="onnx,ORT" to enable this embedder.
func NewHugotCLIPEmbedder(modelPath string, quantized bool, logger *zap.Logger) (*HugotCLIPEmbedder, error) {
	return NewHugotCLIPEmbedderWithSession(modelPath, quantized, nil, logger)
}

// NewHugotCLIPEmbedderWithSessionManager creates a new CLIP embedder using a SessionManager.
// The SessionManager handles backend selection and session reuse (required for ONNX Runtime which only allows one session).
// modelBackends restricts which backends can be used (nil = ONNX only, which is required for CLIP).
// Returns the embedder and the backend type that was used.
func NewHugotCLIPEmbedderWithSessionManager(modelPath string, quantized bool, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*HugotCLIPEmbedder, hugot.BackendType, error) {
	if sessionManager == nil {
		return nil, "", errors.New("sessionManager is required for CLIP embedder (ONNX Runtime only allows one session)")
	}

	// CLIP requires ONNX Runtime backend (not pure Go or XLA)
	// If modelBackends is nil, default to ONNX only
	if modelBackends == nil {
		modelBackends = []string{"onnx"}
	}
	session, backendUsed, err := sessionManager.GetSessionForModel(modelBackends)
	if err != nil {
		return nil, "", fmt.Errorf("getting session from manager: %w", err)
	}

	embedder, err := NewHugotCLIPEmbedderWithSession(modelPath, quantized, session, logger)
	if err != nil {
		return nil, "", err
	}

	// SessionManager owns the session, so mark as shared
	embedder.sessionShared = true

	return embedder, backendUsed, nil
}

// NewHugotCLIPEmbedderWithSession creates a new CLIP embedder using an optional shared session.
func NewHugotCLIPEmbedderWithSession(modelPath string, quantized bool, sharedSession *khugot.Session, logger *zap.Logger) (*HugotCLIPEmbedder, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Initializing HugotCLIP embedder",
		zap.String("modelPath", modelPath),
		zap.Bool("quantized", quantized),
		zap.String("backend", hugot.BackendName()))

	// Load configuration
	config, err := loadHugotCLIPConfig(modelPath)
	if err != nil {
		return nil, fmt.Errorf("loading CLIP config: %w", err)
	}

	// Determine ONNX filenames
	visualFile := "visual_model.onnx"
	textFile := "text_model.onnx"
	if quantized {
		visualFile = "visual_model_quantized.onnx"
		textFile = "text_model_quantized.onnx"
	}

	// Verify files exist
	visualPath := filepath.Join(modelPath, visualFile)
	textPath := filepath.Join(modelPath, textFile)
	if _, err := os.Stat(visualPath); err != nil {
		return nil, fmt.Errorf("visual model not found: %s", visualPath)
	}
	if _, err := os.Stat(textPath); err != nil {
		return nil, fmt.Errorf("text model not found: %s", textPath)
	}

	// Create or reuse session
	session, err := hugot.NewSessionOrUseExisting(sharedSession)
	if err != nil {
		return nil, fmt.Errorf("creating hugot session: %w", err)
	}
	sessionShared := (sharedSession != nil)

	// Get image size for preprocessing
	imageSize := config.ImageSize
	if imageSize == 0 {
		imageSize = 224 // Default for CLIP
	}

	// Create visual pipeline with image mode
	visualPipelineName := fmt.Sprintf("%s:visual:%s", modelPath, visualFile)
	visualConfig := khugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         visualPipelineName,
		OnnxFilename: visualFile,
		Options: []backends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
			pipelines.WithImageMode(),
			pipelines.WithPreprocessSteps[*pipelines.FeatureExtractionPipeline](
				imageutil.ResizeStep(imageSize),
				imageutil.CenterCropStep(imageSize, imageSize),
			),
			pipelines.WithNormalizationSteps[*pipelines.FeatureExtractionPipeline](
				imageutil.RescaleStep(),
				imageutil.CLIPPixelNormalizationStep(),
			),
			pipelines.WithNCHWFormat[*pipelines.FeatureExtractionPipeline](),
			pipelines.WithNormalization(),
		},
	}

	visualPipeline, err := khugot.NewPipeline(session, visualConfig)
	if err != nil {
		if !sessionShared {
			_ = session.Destroy()
		}
		return nil, fmt.Errorf("creating visual pipeline: %w", err)
	}

	// Create text pipeline (regular feature extraction)
	textPipelineName := fmt.Sprintf("%s:text:%s", modelPath, textFile)
	textConfig := khugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         textPipelineName,
		OnnxFilename: textFile,
		Options: []backends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
			pipelines.WithNormalization(),
		},
	}

	textPipeline, err := khugot.NewPipeline(session, textConfig)
	if err != nil {
		if !sessionShared {
			_ = session.Destroy()
		}
		return nil, fmt.Errorf("creating text pipeline: %w", err)
	}

	logger.Info("HugotCLIP embedder initialized",
		zap.Int("projectionDim", config.ProjectionDim),
		zap.Int("imageSize", imageSize))

	return &HugotCLIPEmbedder{
		visualPipeline: visualPipeline,
		textPipeline:   textPipeline,
		session:        session,
		config:         config,
		logger:         logger,
		modelPath:      modelPath,
		sessionShared:  sessionShared,
		caps: libafembed.EmbedderCapabilities{
			SupportedMIMETypes: []libafembed.MIMETypeSupport{
				{MIMEType: "text/plain"},
				{MIMEType: "image/png"},
				{MIMEType: "image/jpeg"},
				{MIMEType: "image/gif"},
				{MIMEType: "image/webp"},
			},
			Dimensions:       []int{config.ProjectionDim},
			DefaultDimension: config.ProjectionDim,
			SupportsFusion:   false,
		},
	}, nil
}

// Capabilities returns the embedder capabilities
func (c *HugotCLIPEmbedder) Capabilities() libafembed.EmbedderCapabilities {
	return c.caps
}

// Embed generates embeddings for the given content.
// For text content, uses the text encoder.
// For image content (BinaryContent), uses the visual encoder.
func (c *HugotCLIPEmbedder) Embed(ctx context.Context, contents [][]ai.ContentPart) ([][]float32, error) {
	if len(contents) == 0 {
		return [][]float32{}, nil
	}

	embeddings := make([][]float32, len(contents))

	for i, parts := range contents {
		var embedding []float32
		var err error

		for _, part := range parts {
			switch p := part.(type) {
			case ai.BinaryContent:
				if strings.HasPrefix(p.MIMEType, "image/") {
					embedding, err = c.embedImage(p.Data)
					if err != nil {
						return nil, fmt.Errorf("embedding image at index %d: %w", i, err)
					}
				}
			case ai.TextContent:
				embedding, err = c.embedText(p.Text)
				if err != nil {
					return nil, fmt.Errorf("embedding text at index %d: %w", i, err)
				}
			}

			if embedding != nil {
				break
			}
		}

		if embedding == nil {
			return nil, fmt.Errorf("no valid content found at index %d", i)
		}

		embeddings[i] = embedding
	}

	return embeddings, nil
}

// embedImage processes an image and returns its embedding using the visual pipeline
func (c *HugotCLIPEmbedder) embedImage(imageData []byte) ([]float32, error) {
	// Decode image
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return nil, fmt.Errorf("decoding image: %w", err)
	}

	// Run through visual pipeline
	output, err := c.visualPipeline.RunWithImages([]image.Image{img})
	if err != nil {
		return nil, fmt.Errorf("running visual pipeline: %w", err)
	}

	if len(output.Embeddings) == 0 || len(output.Embeddings[0]) == 0 {
		return nil, errors.New("no embedding returned from visual pipeline")
	}

	// Pipeline already normalizes if WithNormalization() was used
	return output.Embeddings[0], nil
}

// embedText tokenizes text and returns its embedding using the text pipeline
func (c *HugotCLIPEmbedder) embedText(text string) ([]float32, error) {
	// Run through text pipeline
	output, err := c.textPipeline.RunPipeline([]string{text})
	if err != nil {
		return nil, fmt.Errorf("running text pipeline: %w", err)
	}

	if len(output.Embeddings) == 0 || len(output.Embeddings[0]) == 0 {
		return nil, errors.New("no embedding returned from text pipeline")
	}

	// Pipeline already normalizes if WithNormalization() was used
	return output.Embeddings[0], nil
}

// Close releases resources
func (c *HugotCLIPEmbedder) Close() error {
	if c.session != nil && !c.sessionShared {
		c.logger.Info("Destroying Hugot session (owned by this CLIP embedder)")
		return c.session.Destroy()
	} else if c.sessionShared {
		c.logger.Debug("Skipping session destruction (shared session)")
	}
	return nil
}

// loadHugotCLIPConfig loads CLIP configuration from model directory
func loadHugotCLIPConfig(modelPath string) (*HugotCLIPConfig, error) {
	configPaths := []string{
		filepath.Join(modelPath, "clip_config.json"),
		filepath.Join(modelPath, "config.json"),
	}

	for _, path := range configPaths {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}

		// Try to parse as CLIP config with nested structure
		var nestedConfig struct {
			ModelType     string `json:"model_type"`
			ProjectionDim int    `json:"projection_dim"`
			VisionConfig  struct {
				ImageSize     int `json:"image_size"`
				ProjectionDim int `json:"projection_dim"`
			} `json:"vision_config"`
		}

		if err := json.Unmarshal(data, &nestedConfig); err == nil {
			projDim := nestedConfig.ProjectionDim
			if projDim == 0 {
				projDim = nestedConfig.VisionConfig.ProjectionDim
			}
			if projDim == 0 {
				projDim = 512 // Default CLIP dimension
			}

			imageSize := nestedConfig.VisionConfig.ImageSize
			if imageSize == 0 {
				imageSize = 224 // Default
			}

			return &HugotCLIPConfig{
				ModelType:     nestedConfig.ModelType,
				ProjectionDim: projDim,
				ImageSize:     imageSize,
			}, nil
		}
	}

	// Return default config for CLIP ViT-B/32
	return &HugotCLIPConfig{
		ModelType:     "clip",
		ProjectionDim: 512,
		ImageSize:     224,
	}, nil
}
