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

package reading

import (
	"context"
	"fmt"
	"image"
	"runtime"
	"strings"
	"sync/atomic"

	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	"github.com/antflydb/termite/pkg/termite/lib/ocr"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure PooledHugotReader implements the Reader interface
var _ Reader = (*PooledHugotReader)(nil)

// PooledHugotReader manages multiple Vision2Seq pipelines for concurrent OCR/document reading.
// Each request acquires a pipeline slot via semaphore, enabling true parallelism.
type PooledHugotReader struct {
	session       *khugot.Session
	pipelines     []*pipelines.Vision2SeqPipeline
	sem           *semaphore.Weighted
	nextPipeline  atomic.Uint64
	logger        *zap.Logger
	sessionShared bool
	poolSize      int
	modelType     ModelType
	modelPath     string
}

// NewPooledHugotReader creates a new pooled reader using the Hugot ONNX runtime.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
func NewPooledHugotReader(modelPath string, poolSize int, logger *zap.Logger) (*PooledHugotReader, error) {
	return NewPooledHugotReaderWithSession(modelPath, poolSize, nil, logger)
}

// NewPooledHugotReaderWithSession creates a new pooled reader using an optional shared Hugot session.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
func NewPooledHugotReaderWithSession(modelPath string, poolSize int, sharedSession *khugot.Session, logger *zap.Logger) (*PooledHugotReader, error) {
	reader, _, err := newPooledHugotReaderInternal(modelPath, poolSize, sharedSession, nil, nil, logger)
	return reader, err
}

// NewPooledHugotReaderWithSessionManager creates a new pooled reader using a SessionManager.
// The SessionManager handles backend selection based on priority and model compatibility.
// Returns the reader, the backend type that was used, and any error.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
// modelBackends specifies which backends this model supports (nil = all backends).
func NewPooledHugotReaderWithSessionManager(modelPath string, poolSize int, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*PooledHugotReader, hugot.BackendType, error) {
	return newPooledHugotReaderInternal(modelPath, poolSize, nil, sessionManager, modelBackends, logger)
}

func newPooledHugotReaderInternal(modelPath string, poolSize int, sharedSession *khugot.Session, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*PooledHugotReader, hugot.BackendType, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	if poolSize <= 0 {
		poolSize = min(runtime.NumCPU(), 4)
	}

	// Detect model type from path
	modelType := detectModelType(modelPath)
	logger.Info("Detected reader model type",
		zap.String("path", modelPath),
		zap.String("type", string(modelType)))

	var session *khugot.Session
	var backendUsed hugot.BackendType
	var sessionShared bool

	if sharedSession != nil {
		session = sharedSession
		sessionShared = true
		backendUsed = hugot.BackendONNX // Assume ONNX if using shared session
	} else if sessionManager != nil {
		// Use SessionManager for backend selection
		var err error
		session, backendUsed, err = sessionManager.GetSessionForModel(modelBackends)
		if err != nil {
			return nil, "", fmt.Errorf("getting session from manager: %w", err)
		}
		sessionShared = true // Session is managed by SessionManager
	} else {
		// Create our own session using the default backend
		var err error
		session, err = hugot.NewSession()
		if err != nil {
			return nil, "", fmt.Errorf("creating session: %w", err)
		}
		backendUsed = hugot.BackendONNX
		sessionShared = false
	}

	// Create pooled pipelines
	pipelineSlice := make([]*pipelines.Vision2SeqPipeline, poolSize)

	for i := 0; i < poolSize; i++ {
		config := khugot.Vision2SeqConfig{
			ModelPath: modelPath,
			Name:      fmt.Sprintf("reader-%d", i),
		}

		pipeline, err := khugot.NewPipeline[*pipelines.Vision2SeqPipeline](session, config)
		if err != nil {
			// Cleanup already created pipelines
			for j := 0; j < i; j++ {
				if pipelineSlice[j] != nil {
					_ = pipelineSlice[j].Destroy()
				}
			}
			if !sessionShared {
				_ = session.Destroy()
			}
			return nil, "", fmt.Errorf("creating Vision2Seq pipeline %d: %w", i, err)
		}
		pipelineSlice[i] = pipeline
	}

	reader := &PooledHugotReader{
		session:       session,
		pipelines:     pipelineSlice,
		sem:           semaphore.NewWeighted(int64(poolSize)),
		logger:        logger,
		sessionShared: sessionShared,
		poolSize:      poolSize,
		modelType:     modelType,
		modelPath:     modelPath,
	}

	logger.Info("Created pooled reader",
		zap.Int("poolSize", poolSize),
		zap.String("backend", string(backendUsed)),
		zap.String("modelType", string(modelType)))

	return reader, backendUsed, nil
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

	return ModelTypeGeneric
}

// Read extracts text from the given images using the Vision2Seq model.
func (r *PooledHugotReader) Read(ctx context.Context, images []image.Image, prompt string, maxTokens int) ([]Result, error) {
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

	// Set max tokens if specified
	if maxTokens > 0 {
		pipeline.MaxNewTokens = maxTokens
	}

	// Run inference
	var output *pipelines.Vision2SeqOutput
	var err error

	if prompt != "" {
		output, err = pipeline.RunWithPrompt(images, prompt)
	} else {
		output, err = pipeline.RunWithImages(images)
	}

	if err != nil {
		return nil, fmt.Errorf("running Vision2Seq inference: %w", err)
	}

	// Parse results based on model type
	results := make([]Result, len(output.GeneratedTexts))
	for i, text := range output.GeneratedTexts {
		results[i] = r.parseOutput(text, prompt)
	}

	r.logger.Debug("Read completed",
		zap.Int("numImages", len(images)),
		zap.Int("numResults", len(results)),
		zap.String("prompt", truncateString(prompt, 50)))

	return results, nil
}

// parseOutput parses the raw model output based on model type.
func (r *PooledHugotReader) parseOutput(text string, prompt string) Result {
	result := Result{
		Text: strings.TrimSpace(text),
	}

	switch r.modelType {
	case ModelTypeDonut:
		// Clean outer task tokens and parse structured fields
		result.Text = ocr.DonutCleanOutput(text)
		result.Fields = ocr.DonutParseFields(text)

		// Handle DocVQA specifically
		if strings.Contains(prompt, "<s_docvqa>") {
			result.Text = ocr.DonutParseDocVQAAnswer(text)
		}

	case ModelTypeFlorence:
		// Florence outputs are typically cleaner
		result.Text = ocr.FlorenceParseOCR(text)

	case ModelTypeTrOCR, ModelTypeGeneric:
		// TrOCR and generic models output plain text
		result.Text = strings.TrimSpace(text)
	}

	return result
}

// Close releases all pipeline resources.
func (r *PooledHugotReader) Close() error {
	r.logger.Info("Closing pooled reader", zap.Int("poolSize", r.poolSize))

	var errs []error

	// Destroy all pipelines
	for i, pipeline := range r.pipelines {
		if pipeline != nil {
			if err := pipeline.Destroy(); err != nil {
				r.logger.Warn("Error destroying pipeline",
					zap.Int("index", i),
					zap.Error(err))
				errs = append(errs, err)
			}
		}
	}

	// Only destroy session if we own it
	if !r.sessionShared && r.session != nil {
		if err := r.session.Destroy(); err != nil {
			r.logger.Warn("Error destroying session", zap.Error(err))
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing reader: %v", errs)
	}
	return nil
}

// ModelType returns the detected model type.
func (r *PooledHugotReader) ModelType() ModelType {
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
