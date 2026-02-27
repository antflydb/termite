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

package reading

import (
	"context"
	"fmt"
	"image"
	"runtime"
	"strings"
	"sync/atomic"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
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
	// ModelTypePix2Struct is a visual question answering model (google/pix2struct-*)
	ModelTypePix2Struct ModelType = "pix2struct"
	// ModelTypeSurya is a multi-stage OCR model (vikp/surya_*)
	ModelTypeSurya ModelType = "surya"
	// ModelTypePaddleOCR is a multi-stage OCR model (PaddleOCR PP-OCRv4)
	ModelTypePaddleOCR ModelType = "paddleocr"
	// ModelTypeMoondream is a vision-language model (vikhyatk/moondream2)
	ModelTypeMoondream ModelType = "moondream"
	// ModelTypeGeneric is used when the model type is unknown
	ModelTypeGeneric ModelType = "generic"
)

// RecognizedRegion represents a text region with recognized text and bounding box.
// Populated by multi-stage OCR models (Surya, PaddleOCR).
type RecognizedRegion struct {
	// Text is the recognized text within this region.
	Text string

	// BBox is the bounding box [x1, y1, x2, y2] in pixel coordinates.
	BBox [4]float64

	// Confidence is the recognition confidence score.
	Confidence float64

	// Label is the semantic label (e.g., "text", "title", "table"), set by layout analysis.
	Label string
}

// Result contains the output from reading an image.
type Result struct {
	// Text is the raw extracted text from the image
	Text string

	// Fields contains structured field values extracted by document understanding models.
	// Fields are flattened with dot notation for nested structures (e.g., "menu.nm", "menu.price").
	// This is populated by models like Donut that output structured data.
	Fields map[string]string

	// Regions contains individual text regions with bounding boxes and recognized text.
	// Populated by multi-stage OCR models (Surya, PaddleOCR).
	Regions []RecognizedRegion
}

// Reader provides OCR and document understanding for images.
// It wraps Vision2Seq models (TrOCR, Donut, Florence-2, Moondream) to extract text from images.
type Reader interface {
	// Read extracts text from the given images.
	// The optional prompt parameter allows specifying a task prompt for document understanding models:
	//   - TrOCR: prompt is ignored (pure OCR)
	//   - Donut CORD: "<s_cord-v2>" for receipt parsing
	//   - Donut DocVQA: "<s_docvqa><s_question>...</s_question><s_answer>" for visual QA
	//   - Florence-2: "<OCR>" for text extraction, "<CAPTION>" for captioning
	//   - Moondream: natural language prompt (e.g., "Describe this image in detail")
	//
	// maxTokens limits the generated output length (0 uses model default).
	//
	// Returns one Result per input image. For Moondream, Result.Fields contains
	// structured output (mood, tags, possible_source) extracted from the JSON response.
	Read(ctx context.Context, images []image.Image, prompt string, maxTokens int) ([]Result, error)

	// Close releases model resources.
	Close() error
}

// Ensure PooledReader implements the Reader interface
var _ Reader = (*PooledReader)(nil)

// PooledReader manages multiple Vision2Seq pipelines for concurrent OCR/document reading.
// Each request acquires a pipeline slot via semaphore, enabling true parallelism.
type PooledReader struct {
	pipelines    []*pipelines.Vision2SeqPipeline
	sem          *semaphore.Weighted
	nextPipeline atomic.Uint64
	logger       *zap.Logger
	poolSize     int
	modelType    ModelType
	modelPath    string
}

// PooledReaderConfig holds configuration for creating a PooledReader.
type PooledReaderConfig struct {
	// ModelPath is the path to the Vision2Seq model.
	ModelPath string

	// PoolSize is the number of concurrent pipelines (0 = auto-detect from CPU count).
	PoolSize int

	// GenerationConfig holds text generation parameters. If nil, uses defaults.
	GenerationConfig *backends.GenerationConfig

	// ImageConfig holds image preprocessing parameters. If nil, uses model's default.
	ImageConfig *backends.ImageConfig

	// Logger for logging. If nil, uses a no-op logger.
	Logger *zap.Logger
}

// NewPooledReader creates a new pooled reader from the given configuration.
// sessionManager is used to load the vision2seq model.
func NewPooledReader(
	cfg *PooledReaderConfig,
	sessionManager *backends.SessionManager,
	modelBackends []string,
) (*PooledReader, backends.BackendType, error) {
	if cfg == nil {
		return nil, "", fmt.Errorf("config is required")
	}

	logger := cfg.Logger
	if logger == nil {
		logger = zap.NewNop()
	}

	poolSize := cfg.PoolSize
	if poolSize <= 0 {
		poolSize = min(runtime.NumCPU(), 4)
	}

	// Detect model type from path
	modelType := detectModelType(cfg.ModelPath)
	logger.Info("Detected reader model type",
		zap.String("path", cfg.ModelPath),
		zap.String("type", string(modelType)))

	// Build pipeline options
	var opts []pipelines.Vision2SeqPipelineOption
	if cfg.ImageConfig != nil {
		opts = append(opts, pipelines.WithVision2SeqImageConfig(cfg.ImageConfig))
	}
	if cfg.GenerationConfig != nil {
		opts = append(opts, pipelines.WithVision2SeqGenerationConfig(cfg.GenerationConfig))
	}

	// Create pooled pipelines using LoadVision2SeqPipeline
	pipelineSlice := make([]*pipelines.Vision2SeqPipeline, poolSize)
	var backendType backends.BackendType

	for i := 0; i < poolSize; i++ {
		pipeline, bt, err := pipelines.LoadVision2SeqPipeline(
			cfg.ModelPath,
			sessionManager,
			modelBackends,
			opts...,
		)
		if err != nil {
			// Clean up already-created pipelines
			for j := 0; j < i; j++ {
				if pipelineSlice[j] != nil {
					_ = pipelineSlice[j].Close()
				}
			}
			return nil, "", fmt.Errorf("loading Vision2Seq pipeline %d: %w", i, err)
		}
		pipelineSlice[i] = pipeline
		backendType = bt
	}

	reader := &PooledReader{
		pipelines: pipelineSlice,
		sem:       semaphore.NewWeighted(int64(poolSize)),
		logger:    logger,
		poolSize:  poolSize,
		modelType: modelType,
		modelPath: cfg.ModelPath,
	}

	logger.Info("Created pooled reader",
		zap.Int("poolSize", poolSize),
		zap.String("backend", string(backendType)),
		zap.String("modelType", string(modelType)))

	return reader, backendType, nil
}

// detectModelType determines the model type from the model path.
func detectModelType(modelPath string) ModelType {
	pathLower := strings.ToLower(modelPath)

	if strings.Contains(pathLower, "trocr") {
		return ModelTypeTrOCR
	}
	if strings.Contains(pathLower, "donut") {
		return ModelTypeDonut
	}
	if strings.Contains(pathLower, "florence") {
		return ModelTypeFlorence
	}
	if strings.Contains(pathLower, "pix2struct") {
		return ModelTypePix2Struct
	}
	if strings.Contains(pathLower, "surya") {
		return ModelTypeSurya
	}
	if strings.Contains(pathLower, "paddleocr") || strings.Contains(pathLower, "ppocr") {
		return ModelTypePaddleOCR
	}
	if strings.Contains(pathLower, "moondream") {
		return ModelTypeMoondream
	}

	return ModelTypeGeneric
}

// Read extracts text from the given images using the Vision2Seq model.
func (r *PooledReader) Read(ctx context.Context, images []image.Image, prompt string, maxTokens int) ([]Result, error) {
	if len(images) == 0 {
		return nil, fmt.Errorf("no images provided")
	}

	// Acquire a pipeline slot
	if err := r.sem.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}
	defer r.sem.Release(1)

	// Get the next pipeline in round-robin fashion
	idx := r.nextPipeline.Add(1) - 1
	pipeline := r.pipelines[idx%uint64(r.poolSize)]

	// Temporarily override max tokens if specified
	originalMaxTokens := pipeline.GenerationConfig.MaxNewTokens
	if maxTokens > 0 {
		pipeline.GenerationConfig.MaxNewTokens = maxTokens
	}
	defer func() {
		pipeline.GenerationConfig.MaxNewTokens = originalMaxTokens
	}()

	// Process each image
	results := make([]Result, len(images))
	for i, img := range images {
		var output *pipelines.Vision2SeqResult
		var err error

		if prompt != "" {
			output, err = pipeline.RunWithPrompt(ctx, img, prompt)
		} else {
			output, err = pipeline.Run(ctx, img)
		}

		if err != nil {
			return nil, fmt.Errorf("running Vision2Seq inference on image %d: %w", i, err)
		}

		results[i] = r.parseOutput(output.Text, prompt)
	}

	r.logger.Debug("Read completed",
		zap.Int("numImages", len(images)),
		zap.Int("numResults", len(results)),
		zap.String("prompt", truncateString(prompt, 50)))

	return results, nil
}

// parseOutput parses the raw model output based on model type.
func (r *PooledReader) parseOutput(text string, prompt string) Result {
	result := Result{
		Text: strings.TrimSpace(text),
	}

	switch r.modelType {
	case ModelTypeDonut:
		// Clean outer task tokens and parse structured fields
		result.Text = DonutCleanOutput(text)
		result.Fields = DonutParseFields(text)

		// Handle DocVQA specifically
		if strings.Contains(prompt, "<s_docvqa>") {
			result.Text = DonutParseDocVQAAnswer(text)
		}

	case ModelTypeFlorence:
		// Florence outputs are typically cleaner
		result.Text = FlorenceParseOCR(text)

	case ModelTypeMoondream:
		// Moondream outputs structured JSON with description, mood, tags, etc.
		description, fields := MoondreamParseFields(text)
		result.Text = description
		result.Fields = fields

	case ModelTypePix2Struct:
		// Pix2Struct outputs direct answers as plain text
		result.Text = strings.TrimSpace(text)

	case ModelTypeTrOCR, ModelTypeGeneric:
		// TrOCR and generic models output plain text
		result.Text = strings.TrimSpace(text)
	}

	return result
}

// Close releases all pipeline resources.
func (r *PooledReader) Close() error {
	r.logger.Info("Closing pooled reader", zap.Int("poolSize", r.poolSize))

	var errs []error

	// Close all pipelines
	for i, pipeline := range r.pipelines {
		if pipeline != nil {
			if err := pipeline.Close(); err != nil {
				r.logger.Warn("Error closing pipeline",
					zap.Int("index", i),
					zap.Error(err))
				errs = append(errs, err)
			}
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing reader: %v", errs)
	}
	return nil
}

// ModelType returns the detected model type.
func (r *PooledReader) ModelType() ModelType {
	return r.modelType
}

// truncateString truncates a string to maxLen, adding "..." if truncated.
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if maxLen <= 3 {
		return s[:maxLen]
	}
	return s[:maxLen-3] + "..."
}
