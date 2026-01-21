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

package chunking

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/antflydb/antfly-go/libaf/chunking"
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// Ensure PooledHugotChunker implements the Chunker interface
var _ chunking.Chunker = (*PooledHugotChunker)(nil)

// HugotChunkerConfig contains configuration for the Hugot ONNX chunker.
type HugotChunkerConfig struct {
	// MaxChunks is the maximum number of chunks to generate per document.
	MaxChunks int

	// Threshold is the minimum confidence threshold for separator detection.
	Threshold float32

	// TargetTokens is the target number of tokens per chunk.
	TargetTokens int
}

// DefaultHugotChunkerConfig returns sensible defaults for the Hugot chunker.
func DefaultHugotChunkerConfig() HugotChunkerConfig {
	return HugotChunkerConfig{
		MaxChunks:    50,
		Threshold:    0.5,
		TargetTokens: 500,
	}
}

// PooledHugotChunker manages multiple ONNX pipelines for concurrent chunking.
// Each request acquires a pipeline slot via semaphore, enabling true parallelism.
type PooledHugotChunker struct {
	session       *khugot.Session
	pipelines     []*pipelines.TokenClassificationPipeline
	sem           *semaphore.Weighted
	nextPipeline  atomic.Uint64
	config        HugotChunkerConfig
	chunkLabel    string
	logger        *zap.Logger
	sessionShared bool
	poolSize      int

	// Synchronization for safe Close() behavior
	closed    atomic.Bool    // Prevents new operations after Close()
	wg        sync.WaitGroup // Waits for in-flight operations to complete
	closeOnce sync.Once      // Ensures Close() runs exactly once
	closeErr  error          // Stores error from Close()
}

// ErrChunkerClosed is returned when Chunk is called on a closed chunker.
var ErrChunkerClosed = errors.New("chunker is closed")

// NewPooledHugotChunker creates a new pooled chunker using the Hugot ONNX runtime.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_f16.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
func NewPooledHugotChunker(config HugotChunkerConfig, modelPath string, onnxFilename string, poolSize int, logger *zap.Logger) (*PooledHugotChunker, error) {
	return NewPooledHugotChunkerWithSession(config, modelPath, onnxFilename, poolSize, nil, logger)
}

// NewPooledHugotChunkerWithSession creates a new pooled chunker using an optional shared Hugot session.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_f16.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
func NewPooledHugotChunkerWithSession(config HugotChunkerConfig, modelPath string, onnxFilename string, poolSize int, sharedSession *khugot.Session, logger *zap.Logger) (*PooledHugotChunker, error) {
	chunker, _, err := newPooledHugotChunkerInternal(config, modelPath, onnxFilename, poolSize, sharedSession, nil, nil, logger)
	return chunker, err
}

// NewPooledHugotChunkerWithSessionManager creates a new pooled chunker using a SessionManager.
// The SessionManager handles backend selection based on priority and model compatibility.
// Returns the chunker, the backend type that was used, and any error.
// poolSize determines how many concurrent requests can be processed (0 = auto-detect from CPU count).
// onnxFilename specifies which ONNX file to load (e.g., "model.onnx", "model_f16.onnx", "model_i8.onnx").
// If empty, defaults to "model.onnx".
// modelBackends specifies which backends this model supports (nil = all backends).
func NewPooledHugotChunkerWithSessionManager(config HugotChunkerConfig, modelPath string, onnxFilename string, poolSize int, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*PooledHugotChunker, hugot.BackendType, error) {
	return newPooledHugotChunkerInternal(config, modelPath, onnxFilename, poolSize, nil, sessionManager, modelBackends, logger)
}

// newPooledHugotChunkerInternal is the shared implementation for creating pooled chunkers.
// Either sharedSession or sessionManager should be provided, not both.
func newPooledHugotChunkerInternal(config HugotChunkerConfig, modelPath string, onnxFilename string, poolSize int, sharedSession *khugot.Session, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*PooledHugotChunker, hugot.BackendType, error) {
	if modelPath == "" {
		return nil, hugot.BackendGo, errors.New("model path is required for Hugot chunking")
	}

	// Default to model.onnx if not specified
	if onnxFilename == "" {
		onnxFilename = "model.onnx"
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	// Apply defaults for zero values
	if config.MaxChunks <= 0 {
		config.MaxChunks = 50
	}
	if config.Threshold <= 0 {
		config.Threshold = 0.5
	}
	if config.TargetTokens <= 0 {
		config.TargetTokens = 500
	}

	// Auto-detect pool size from CPU count if not specified
	if poolSize <= 0 {
		poolSize = runtime.NumCPU()
	}

	logger.Info("Initializing pooled Hugot chunker",
		zap.String("model_path", modelPath),
		zap.String("onnxFilename", onnxFilename),
		zap.Int("poolSize", poolSize),
		zap.Int("max_chunks", config.MaxChunks),
		zap.Float32("threshold", config.Threshold),
		zap.Int("target_tokens", config.TargetTokens),
		zap.String("backend", hugot.BackendName()))

	// Get session from SessionManager, shared session, or create new one
	var session *khugot.Session
	var backendUsed hugot.BackendType
	var err error
	var sessionShared bool

	if sessionManager != nil {
		// Use SessionManager for backend selection
		session, backendUsed, err = sessionManager.GetSessionForModel(modelBackends)
		if err != nil {
			logger.Error("Failed to get session from SessionManager", zap.Error(err))
			return nil, backendUsed, fmt.Errorf("getting session from SessionManager: %w", err)
		}
		sessionShared = true // SessionManager owns the session
		logger.Info("Using session from SessionManager",
			zap.String("backend", string(backendUsed)))
	} else if sharedSession != nil {
		// Use provided shared session
		session = sharedSession
		sessionShared = true
		backendUsed = hugot.GetDefaultBackend().Type()
		logger.Info("Using shared Hugot session", zap.String("backend", hugot.BackendName()))
	} else {
		// Create new session
		session, err = hugot.NewSession()
		if err != nil {
			logger.Error("Failed to create Hugot session", zap.Error(err))
			return nil, hugot.BackendGo, fmt.Errorf("creating hugot session: %w", err)
		}
		sessionShared = false
		backendUsed = hugot.GetDefaultBackend().Type()
		logger.Info("Created new Hugot session", zap.String("backend", hugot.BackendName()))
	}

	// Create N pipelines with unique names
	pipelinesList := make([]*pipelines.TokenClassificationPipeline, poolSize)
	for i := 0; i < poolSize; i++ {
		pipelineName := fmt.Sprintf("%s:%s:%d", modelPath, onnxFilename, i)
		pipelineConfig := khugot.TokenClassificationConfig{
			ModelPath:    modelPath,
			OnnxFilename: onnxFilename,
			Name:         pipelineName,
		}

		pipeline, err := khugot.NewPipeline(session, pipelineConfig)
		if err != nil {
			// Clean up already-created pipelines
			if !sessionShared {
				_ = session.Destroy()
			}
			logger.Error("Failed to create pipeline",
				zap.Int("index", i),
				zap.Error(err))
			return nil, backendUsed, fmt.Errorf("creating token classification pipeline %d: %w", i, err)
		}
		// Set aggregation strategy to "SIMPLE" to group adjacent tokens
		pipeline.AggregationStrategy = "SIMPLE"
		pipelinesList[i] = pipeline
		logger.Debug("Created pipeline", zap.Int("index", i), zap.String("name", pipelineName))
	}

	logger.Info("Successfully created pooled token classification pipelines", zap.Int("count", poolSize))

	return &PooledHugotChunker{
		session:       session,
		pipelines:     pipelinesList,
		sem:           semaphore.NewWeighted(int64(poolSize)),
		config:        config,
		chunkLabel:    "separator",
		logger:        logger,
		sessionShared: sessionShared,
		poolSize:      poolSize,
	}, backendUsed, nil
}

// Chunk splits text using neural token classification with per-request config overrides.
// Thread-safe: uses semaphore to limit concurrent pipeline access.
func (p *PooledHugotChunker) Chunk(ctx context.Context, text string, opts chunking.ChunkOptions) ([]chunking.Chunk, error) {
	// Check if closed before starting
	if p.closed.Load() {
		return nil, ErrChunkerClosed
	}

	// Track this in-flight operation so Close() waits for us
	p.wg.Add(1)
	defer p.wg.Done()

	// Double-check after registration (handles race with Close())
	if p.closed.Load() {
		return nil, ErrChunkerClosed
	}

	if text == "" {
		p.logger.Debug("Chunk called with empty text")
		return nil, nil
	}

	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Resolve effective config by applying overrides (zero values mean "use default")
	effectiveConfig := p.config
	if opts.MaxChunks != 0 {
		effectiveConfig.MaxChunks = opts.MaxChunks
	}
	if opts.Threshold != 0 {
		effectiveConfig.Threshold = opts.Threshold
	}
	if opts.TargetTokens != 0 {
		effectiveConfig.TargetTokens = opts.TargetTokens
	}

	// Acquire semaphore slot (blocks if all pipelines busy)
	if err := p.sem.Acquire(ctx, 1); err != nil {
		return nil, fmt.Errorf("acquiring pipeline slot: %w", err)
	}
	defer p.sem.Release(1)

	// Round-robin pipeline selection
	idx := int(p.nextPipeline.Add(1) % uint64(p.poolSize))
	pipeline := p.pipelines[idx]

	textLen := len(text)
	textPreview := text
	if textLen > 100 {
		textPreview = text[:100] + "..."
	}

	p.logger.Debug("Starting chunking",
		zap.Int("pipelineIndex", idx),
		zap.Int("text_length", textLen),
		zap.String("text_preview", textPreview))

	// Run token classification pipeline
	output, err := pipeline.RunPipeline([]string{text})
	if err != nil {
		p.logger.Error("Token classification failed",
			zap.Int("pipelineIndex", idx),
			zap.Error(err))
		return nil, fmt.Errorf("running token classification: %w", err)
	}

	p.logger.Debug("Token classification completed",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_entity_groups", len(output.Entities)))

	if len(output.Entities) == 0 {
		p.logger.Error("No results from token classification")
		return nil, errors.New("no results from token classification")
	}

	// Parse token classification results to identify chunk boundaries using effective config
	chunks := p.parseChunkBoundaries(text, output.Entities[0], effectiveConfig)

	// Enforce max chunks limit
	if len(chunks) > effectiveConfig.MaxChunks {
		p.logger.Debug("Limiting chunks",
			zap.Int("from", len(chunks)),
			zap.Int("to", effectiveConfig.MaxChunks))
		chunks = chunks[:effectiveConfig.MaxChunks]
	}

	p.logger.Info("Chunking completed",
		zap.Int("pipelineIndex", idx),
		zap.Int("num_chunks", len(chunks)),
		zap.Int("text_length", textLen))

	return chunks, nil
}

// parseChunkBoundaries converts token classification results into chunks.
func (p *PooledHugotChunker) parseChunkBoundaries(text string, entities []pipelines.Entity, config HugotChunkerConfig) []chunking.Chunk {
	if len(entities) == 0 {
		return []chunking.Chunk{{
			Id:        0,
			Text:      strings.TrimSpace(text),
			StartChar: 0,
			EndChar:   len(text),
		}}
	}

	// Collect separator positions
	separatorPositions := make([]int, 0, len(entities))
	for _, entity := range entities {
		entityLower := strings.ToLower(entity.Entity)
		isSeparator := entityLower == "separator" ||
			strings.HasSuffix(entityLower, "separator") ||
			strings.HasPrefix(entityLower, "separator")

		if isSeparator && entity.Score >= config.Threshold {
			separatorPositions = append(separatorPositions, int(entity.End))
		}
	}

	if len(separatorPositions) == 0 {
		return []chunking.Chunk{{
			Id:        0,
			Text:      strings.TrimSpace(text),
			StartChar: 0,
			EndChar:   len(text),
		}}
	}

	sort.Ints(separatorPositions)

	// Create chunks by splitting text at separator positions
	chunks := make([]chunking.Chunk, 0, len(separatorPositions)+1)
	startPos := 0

	for _, sepPos := range separatorPositions {
		if sepPos > startPos && sepPos <= len(text) {
			chunkText := strings.TrimSpace(text[startPos:sepPos])
			if chunkText != "" {
				chunks = append(chunks, chunking.Chunk{
					Id:        uint32(len(chunks)),
					Text:      chunkText,
					StartChar: startPos,
					EndChar:   sepPos,
				})
			}
			startPos = sepPos
		}
	}

	// Add final chunk
	if startPos < len(text) {
		chunkText := strings.TrimSpace(text[startPos:])
		if chunkText != "" {
			chunks = append(chunks, chunking.Chunk{
				Id:        uint32(len(chunks)),
				Text:      chunkText,
				StartChar: startPos,
				EndChar:   len(text),
			})
		}
	}

	if len(chunks) == 0 {
		chunks = append(chunks, chunking.Chunk{
			Id:        0,
			Text:      strings.TrimSpace(text),
			StartChar: 0,
			EndChar:   len(text),
		})
	}

	// Apply target tokens aggregation if configured
	if config.TargetTokens > 0 && len(chunks) > 1 {
		chunks = p.aggregateByTargetTokens(chunks, config)
	}

	return chunks
}

// estimateTokens returns approximate token count.
func (p *PooledHugotChunker) estimateTokens(text string) int {
	return len(text) / 4
}

// aggregateByTargetTokens combines consecutive chunks until they reach target token count.
func (p *PooledHugotChunker) aggregateByTargetTokens(chunks []chunking.Chunk, config HugotChunkerConfig) []chunking.Chunk {
	if len(chunks) == 0 {
		return chunks
	}

	aggregated := make([]chunking.Chunk, 0)
	currentTexts := make([]string, 0)
	currentTokens := 0
	currentStartChar := chunks[0].StartChar
	var lastEndChar int

	for i, chunk := range chunks {
		chunkTokens := p.estimateTokens(chunk.Text)
		lastEndChar = chunk.EndChar

		if currentTokens > 0 && currentTokens+chunkTokens > config.TargetTokens {
			combinedText := strings.Join(currentTexts, "\n\n")
			aggregated = append(aggregated, chunking.Chunk{
				Id:        uint32(len(aggregated)),
				Text:      combinedText,
				StartChar: currentStartChar,
				EndChar:   chunks[i-1].EndChar,
			})

			currentTexts = []string{chunk.Text}
			currentTokens = chunkTokens
			currentStartChar = chunk.StartChar
		} else {
			currentTexts = append(currentTexts, chunk.Text)
			currentTokens += chunkTokens
		}
	}

	if len(currentTexts) > 0 {
		combinedText := strings.Join(currentTexts, "\n\n")
		aggregated = append(aggregated, chunking.Chunk{
			Id:        uint32(len(aggregated)),
			Text:      combinedText,
			StartChar: currentStartChar,
			EndChar:   lastEndChar,
		})
	}

	return aggregated
}

// Close releases the Hugot session and resources.
// Only destroys the session if it was created by this chunker (not shared).
// Thread-safe: waits for in-flight operations to complete before destroying.
// Safe to call multiple times (only the first call takes effect).
func (p *PooledHugotChunker) Close() error {
	p.closeOnce.Do(func() {
		// Set closed flag to prevent new operations
		p.closed.Store(true)

		// Wait for all in-flight operations to complete
		p.wg.Wait()

		// Now safe to destroy the session
		if p.session != nil && !p.sessionShared {
			p.logger.Info("Destroying Hugot session (owned by this pooled chunker)")
			p.closeErr = p.session.Destroy()
		} else if p.sessionShared {
			p.logger.Debug("Skipping session destruction (shared session)")
		}
	})
	return p.closeErr
}
