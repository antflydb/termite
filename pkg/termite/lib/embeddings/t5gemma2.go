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
	"sync"

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

// T5Gemma2Embedder implements multimodal embeddings using T5Gemma-2 encoder.
// It uses the encoder.onnx for text embeddings and vision_encoder.onnx for images.
//
// T5Gemma-2 is a multimodal encoder-decoder model from Google's Gemma 3 family,
// trained with UL2 objective. The encoder produces rich embeddings that can be
// used for similarity search and retrieval.
//
// The vision encoder is lazily loaded on first image embedding request to save memory.
//
// Build with: CGO_ENABLED=1 go build -tags="onnx,ORT"
type T5Gemma2Embedder struct {
	encoderPipeline *pipelines.FeatureExtractionPipeline
	visionPipeline  *pipelines.FeatureExtractionPipeline
	session         *khugot.Session
	config          *T5Gemma2Config
	logger          *zap.Logger
	caps            libafembed.EmbedderCapabilities
	modelPath       string
	sessionShared   bool

	// Lazy loading for vision encoder
	hasVisionFile   bool       // true if vision_encoder.onnx exists
	visionOnce      sync.Once  // ensures vision pipeline loaded only once
	visionLoadErr   error      // captures any error during lazy load
}

// T5Gemma2Config holds the T5Gemma-2 model configuration
type T5Gemma2Config struct {
	ModelType     string `json:"model_type"`
	HiddenSize    int    `json:"hidden_size"`
	VocabSize     int    `json:"vocab_size"`
	ImageSize     int    `json:"image_size"`
	ImageSeqLen   int    `json:"image_seq_length"` // Tokens per image (typically 256)
	Capabilities  []string `json:"capabilities"`
}

// NewT5Gemma2Embedder creates a new T5Gemma-2 embedder using hugot pipelines.
// The directory should contain:
//   - encoder.onnx (text + multimodal encoder)
//   - vision_encoder.onnx (SigLIP vision encoder)
//   - config.json
//   - tokenizer.json
//
// Build with -tags="onnx,ORT" to enable this embedder.
func NewT5Gemma2Embedder(modelPath string, logger *zap.Logger) (*T5Gemma2Embedder, error) {
	return NewT5Gemma2EmbedderWithSession(modelPath, nil, logger)
}

// NewT5Gemma2EmbedderWithSessionManager creates a new T5Gemma-2 embedder using a SessionManager.
// The SessionManager handles backend selection and session reuse (required for ONNX Runtime which only allows one session).
// Returns the embedder and the backend type that was used.
func NewT5Gemma2EmbedderWithSessionManager(modelPath string, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*T5Gemma2Embedder, hugot.BackendType, error) {
	if sessionManager == nil {
		return nil, "", errors.New("sessionManager is required for T5Gemma-2 embedder (ONNX Runtime only allows one session)")
	}

	// T5Gemma-2 requires ONNX Runtime backend (not pure Go or XLA)
	if modelBackends == nil {
		modelBackends = []string{"onnx"}
	}
	session, backendUsed, err := sessionManager.GetSessionForModel(modelBackends)
	if err != nil {
		return nil, "", fmt.Errorf("getting session from manager: %w", err)
	}

	embedder, err := NewT5Gemma2EmbedderWithSession(modelPath, session, logger)
	if err != nil {
		return nil, "", err
	}

	// SessionManager owns the session, so mark as shared
	embedder.sessionShared = true

	return embedder, backendUsed, nil
}

// NewT5Gemma2EmbedderWithSession creates a new T5Gemma-2 embedder using an optional shared session.
func NewT5Gemma2EmbedderWithSession(modelPath string, sharedSession *khugot.Session, logger *zap.Logger) (*T5Gemma2Embedder, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Initializing T5Gemma-2 embedder",
		zap.String("modelPath", modelPath),
		zap.String("backend", hugot.BackendName()))

	// Load configuration
	config, err := loadT5Gemma2Config(modelPath)
	if err != nil {
		return nil, fmt.Errorf("loading T5Gemma-2 config: %w", err)
	}

	// Verify required files exist
	encoderPath := filepath.Join(modelPath, "encoder.onnx")
	if _, err := os.Stat(encoderPath); err != nil {
		return nil, fmt.Errorf("encoder model not found: %s", encoderPath)
	}

	// Check for optional vision encoder
	visionPath := filepath.Join(modelPath, "vision_encoder.onnx")
	hasVision := false
	if _, err := os.Stat(visionPath); err == nil {
		hasVision = true
	}

	// Create or reuse session
	session, err := hugot.NewSessionOrUseExisting(sharedSession)
	if err != nil {
		return nil, fmt.Errorf("creating hugot session: %w", err)
	}
	sessionShared := (sharedSession != nil)

	// Create text encoder pipeline first (always needed)
	encoderPipelineName := fmt.Sprintf("%s:encoder:encoder.onnx", modelPath)
	encoderConfig := khugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         encoderPipelineName,
		OnnxFilename: "encoder.onnx",
		Options: []backends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
			pipelines.WithNormalization(),
		},
	}

	encoderPipeline, err := khugot.NewPipeline(session, encoderConfig)
	if err != nil {
		if !sessionShared {
			_ = session.Destroy()
		}
		return nil, fmt.Errorf("creating encoder pipeline: %w", err)
	}

	// Vision pipeline is loaded lazily on first image embed request to save memory.
	// We only check if the file exists here; actual loading happens in loadVisionPipeline().
	if hasVision {
		logger.Info("T5Gemma-2 embedder initialized (vision available, will load on first use)",
			zap.Int("hiddenSize", config.HiddenSize),
			zap.Int("imageSize", config.ImageSize))
	} else {
		logger.Info("T5Gemma-2 embedder initialized (text-only, no vision encoder file)",
			zap.Int("hiddenSize", config.HiddenSize))
	}

	// Build capabilities based on available features
	mimeTypes := []libafembed.MIMETypeSupport{
		{MIMEType: "text/plain"},
	}
	if hasVision {
		mimeTypes = append(mimeTypes,
			libafembed.MIMETypeSupport{MIMEType: "image/png"},
			libafembed.MIMETypeSupport{MIMEType: "image/jpeg"},
			libafembed.MIMETypeSupport{MIMEType: "image/gif"},
			libafembed.MIMETypeSupport{MIMEType: "image/webp"},
		)
	}

	return &T5Gemma2Embedder{
		encoderPipeline: encoderPipeline,
		visionPipeline:  nil, // Loaded lazily
		session:         session,
		config:          config,
		logger:          logger,
		modelPath:       modelPath,
		sessionShared:   sessionShared,
		hasVisionFile:   hasVision,
		caps: libafembed.EmbedderCapabilities{
			SupportedMIMETypes: mimeTypes,
			Dimensions:         []int{config.HiddenSize},
			DefaultDimension:   config.HiddenSize,
			SupportsFusion:     false,
		},
	}, nil
}

// Capabilities returns the embedder capabilities
func (t *T5Gemma2Embedder) Capabilities() libafembed.EmbedderCapabilities {
	return t.caps
}

// Embed generates embeddings for the given content.
// For text content, uses the text encoder.
// For image content (BinaryContent), uses the vision encoder.
// For mixed content, processes each modality appropriately.
func (t *T5Gemma2Embedder) Embed(ctx context.Context, contents [][]ai.ContentPart) ([][]float32, error) {
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
					embedding, err = t.embedImage(p.Data)
					if err != nil {
						return nil, fmt.Errorf("embedding image at index %d: %w", i, err)
					}
				}
			case ai.TextContent:
				embedding, err = t.embedText(p.Text)
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

// loadVisionPipeline lazily loads the vision encoder on first use.
// This saves memory when only text embeddings are needed.
func (t *T5Gemma2Embedder) loadVisionPipeline() error {
	t.visionOnce.Do(func() {
		if !t.hasVisionFile {
			t.visionLoadErr = errors.New("vision encoder not available: vision_encoder.onnx not found")
			return
		}

		t.logger.Info("Lazily loading vision encoder pipeline",
			zap.String("modelPath", t.modelPath))

		imageSize := t.config.ImageSize
		if imageSize == 0 {
			imageSize = 896 // Default for T5Gemma-2 (SigLIP)
		}

		visionPipelineName := fmt.Sprintf("%s:vision:vision_encoder.onnx", t.modelPath)
		visionConfig := khugot.FeatureExtractionConfig{
			ModelPath:    t.modelPath,
			Name:         visionPipelineName,
			OnnxFilename: "vision_encoder.onnx",
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

		pipeline, err := khugot.NewPipeline(t.session, visionConfig)
		if err != nil {
			t.visionLoadErr = fmt.Errorf("loading vision pipeline: %w", err)
			t.logger.Error("Failed to load vision encoder", zap.Error(err))
			return
		}

		t.visionPipeline = pipeline
		t.logger.Info("Vision encoder loaded successfully",
			zap.Int("imageSize", imageSize))
	})

	return t.visionLoadErr
}

// embedImage processes an image and returns its embedding using the vision pipeline.
// The vision pipeline is lazily loaded on first call to save memory.
func (t *T5Gemma2Embedder) embedImage(imageData []byte) ([]float32, error) {
	// Lazy load vision pipeline
	if err := t.loadVisionPipeline(); err != nil {
		return nil, err
	}

	// Decode image
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return nil, fmt.Errorf("decoding image: %w", err)
	}

	// Run through vision pipeline
	output, err := t.visionPipeline.RunWithImages([]image.Image{img})
	if err != nil {
		return nil, fmt.Errorf("running vision pipeline: %w", err)
	}

	if len(output.Embeddings) == 0 || len(output.Embeddings[0]) == 0 {
		return nil, errors.New("no embedding returned from vision pipeline")
	}

	// Pipeline already normalizes if WithNormalization() was used
	return output.Embeddings[0], nil
}

// HasVision returns true if this embedder supports image embeddings.
// Note: This returns true if vision_encoder.onnx exists, even if not yet loaded.
func (t *T5Gemma2Embedder) HasVision() bool {
	return t.hasVisionFile
}

// IsVisionLoaded returns true if the vision pipeline has been loaded.
func (t *T5Gemma2Embedder) IsVisionLoaded() bool {
	return t.visionPipeline != nil
}

// embedText tokenizes text and returns its embedding using the encoder pipeline
func (t *T5Gemma2Embedder) embedText(text string) ([]float32, error) {
	// Run through encoder pipeline
	output, err := t.encoderPipeline.RunPipeline([]string{text})
	if err != nil {
		return nil, fmt.Errorf("running encoder pipeline: %w", err)
	}

	if len(output.Embeddings) == 0 || len(output.Embeddings[0]) == 0 {
		return nil, errors.New("no embedding returned from encoder pipeline")
	}

	// Pipeline already normalizes if WithNormalization() was used
	return output.Embeddings[0], nil
}

// Close releases resources
func (t *T5Gemma2Embedder) Close() error {
	if t.session != nil && !t.sessionShared {
		t.logger.Info("Destroying Hugot session (owned by this T5Gemma-2 embedder)")
		return t.session.Destroy()
	} else if t.sessionShared {
		t.logger.Debug("Skipping session destruction (shared session)")
	}
	return nil
}

// loadT5Gemma2Config loads T5Gemma-2 configuration from model directory
func loadT5Gemma2Config(modelPath string) (*T5Gemma2Config, error) {
	// Try t5gemma2_config.json first (our custom config), then config.json
	configPaths := []string{
		filepath.Join(modelPath, "t5gemma2_config.json"),
		filepath.Join(modelPath, "config.json"),
	}

	for _, path := range configPaths {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}

		// Parse config
		var rawConfig struct {
			ModelType   string   `json:"model_type"`
			HiddenSize  int      `json:"hidden_size"`
			VocabSize   int      `json:"vocab_size"`
			Capabilities []string `json:"capabilities"`
			Encoder     struct {
				VisionConfig struct {
					ImageSize int `json:"image_size"`
				} `json:"vision_config"`
				MMTokensPerImage int `json:"mm_tokens_per_image"`
			} `json:"encoder"`
			Decoder struct {
				HiddenSize int `json:"hidden_size"`
			} `json:"decoder"`
		}

		if err := json.Unmarshal(data, &rawConfig); err != nil {
			continue
		}

		// Extract values with fallbacks
		hiddenSize := rawConfig.HiddenSize
		if hiddenSize == 0 {
			hiddenSize = rawConfig.Decoder.HiddenSize
		}
		if hiddenSize == 0 {
			hiddenSize = 640 // Default for T5Gemma-2 270M
		}

		imageSize := rawConfig.Encoder.VisionConfig.ImageSize
		if imageSize == 0 {
			imageSize = 896 // Default for SigLIP in T5Gemma-2
		}

		imageSeqLen := rawConfig.Encoder.MMTokensPerImage
		if imageSeqLen == 0 {
			imageSeqLen = 256 // Default
		}

		return &T5Gemma2Config{
			ModelType:    rawConfig.ModelType,
			HiddenSize:   hiddenSize,
			VocabSize:    rawConfig.VocabSize,
			ImageSize:    imageSize,
			ImageSeqLen:  imageSeqLen,
			Capabilities: rawConfig.Capabilities,
		}, nil
	}

	// Return default config for T5Gemma-2 270M
	return &T5Gemma2Config{
		ModelType:    "t5gemma2",
		HiddenSize:   640,
		VocabSize:    262144,
		ImageSize:    896,
		ImageSeqLen:  256,
		Capabilities: []string{"embeddings", "generation", "decoding", "multimodal"},
	}, nil
}

// IsT5Gemma2Model checks if a model directory contains T5Gemma-2 model files.
// Only encoder.onnx is required; vision_encoder.onnx is optional for text-only use.
func IsT5Gemma2Model(modelPath string) bool {
	encoderPath := filepath.Join(modelPath, "encoder.onnx")

	// Must have encoder (vision encoder is optional)
	if _, err := os.Stat(encoderPath); err != nil {
		return false
	}

	// Check config for model_type
	configPath := filepath.Join(modelPath, "config.json")
	if data, err := os.ReadFile(configPath); err == nil {
		var config struct {
			ModelType string `json:"model_type"`
		}
		if err := json.Unmarshal(data, &config); err == nil {
			return config.ModelType == "t5gemma2"
		}
	}

	// Also check t5gemma2_config.json
	t5gemma2ConfigPath := filepath.Join(modelPath, "t5gemma2_config.json")
	if _, err := os.Stat(t5gemma2ConfigPath); err == nil {
		return true
	}

	return false
}
