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

//go:build go1.22

//go:generate go tool oapi-codegen --config=cfg.yaml ./openapi.yaml
package termite

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"runtime"
	"time"

	"github.com/antflydb/antfly-go/libaf/ai"
	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/antfly-go/libaf/s3"
	"github.com/antflydb/antfly-go/libaf/scraping"
	"github.com/antflydb/termite/pkg/termite/lib/generation"
	"github.com/antflydb/termite/pkg/termite/lib/ner"
	"github.com/bytedance/sonic/decoder"
	"github.com/bytedance/sonic/encoder"
	"go.uber.org/zap"
)

// NOTE: SerializeFloatArrays is in codec.go in this package

// TermiteAPI implements the generated ServerInterface
type TermiteAPI struct {
	logger *zap.Logger
	node   *TermiteNode
}

// NewTermiteAPI creates a new HTTP handler for the Termite API using generated code
func NewTermiteAPI(logger *zap.Logger, node *TermiteNode) http.Handler {
	api := &TermiteAPI{
		logger: logger,
		node:   node,
	}
	return HandlerWithOptions(api, StdHTTPServerOptions{
		BaseURL:    "/api",
		BaseRouter: http.NewServeMux(),
	})
}

// GenerateEmbeddings implements ServerInterface
func (t *TermiteAPI) GenerateEmbeddings(w http.ResponseWriter, r *http.Request) {
	t.node.handleApiEmbed(w, r)
}

// ChunkText implements ServerInterface
func (t *TermiteAPI) ChunkText(w http.ResponseWriter, r *http.Request) {
	t.node.handleApiChunk(w, r)
}

// RerankPrompts implements ServerInterface
func (t *TermiteAPI) RerankPrompts(w http.ResponseWriter, r *http.Request) {
	t.node.handleApiRerank(w, r)
}

// GenerateContent implements ServerInterface
func (t *TermiteAPI) GenerateContent(w http.ResponseWriter, r *http.Request) {
	t.node.handleApiGenerate(w, r)
}

// RecognizeEntities implements ServerInterface
func (t *TermiteAPI) RecognizeEntities(w http.ResponseWriter, r *http.Request) {
	t.node.handleApiRecognize(w, r)
}

// RewriteText implements ServerInterface
func (t *TermiteAPI) RewriteText(w http.ResponseWriter, r *http.Request) {
	t.node.handleApiRewrite(w, r)
}

// ListModels implements ServerInterface
func (t *TermiteAPI) ListModels(w http.ResponseWriter, r *http.Request) {
	resp := ModelsResponse{
		Chunkers:    []string{},
		Rerankers:   []string{},
		Embedders:   []string{},
		Generators:  []string{},
		Recognizers: []string{},
		Extractors:  []string{},
		Rewriters:   []string{},
	}

	if t.node.cachedChunker != nil {
		resp.Chunkers = t.node.cachedChunker.ListModels()
	}

	if t.node.embedderRegistry != nil {
		resp.Embedders = t.node.embedderRegistry.List()
	}

	if t.node.rerankerRegistry != nil {
		resp.Rerankers = t.node.rerankerRegistry.List()
	}

	if t.node.generatorRegistry != nil {
		resp.Generators = t.node.generatorRegistry.List()
	}

	if t.node.nerRegistry != nil {
		resp.Recognizers = t.node.nerRegistry.List()
		resp.Extractors = t.node.nerRegistry.ListRecognizers()
		// Populate recognizer info with capabilities
		capsMap := t.node.nerRegistry.ListWithCapabilities()
		if len(capsMap) > 0 {
			resp.RecognizerInfo = make(map[string]RecognizerModelInfo, len(capsMap))
			for name, caps := range capsMap {
				// Convert string capabilities to RecognizerCapability enum
				enumCaps := make([]RecognizerCapability, len(caps))
				for i, c := range caps {
					enumCaps[i] = RecognizerCapability(c)
				}
				resp.RecognizerInfo[name] = RecognizerModelInfo{
					Capabilities: enumCaps,
				}
			}
		}
	}

	if t.node.seq2seqRegistry != nil {
		resp.Rewriters = t.node.seq2seqRegistry.List()
	}

	w.Header().Set("Content-Type", "application/json")
	if err := encoder.NewStreamEncoder(w).Encode(resp); err != nil {
		t.logger.Error("encoding response", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// GetVersion implements ServerInterface
func (t *TermiteAPI) GetVersion(w http.ResponseWriter, r *http.Request) {
	resp := VersionResponse{
		Version:   Version,
		GitCommit: GitCommit,
		BuildTime: BuildTime,
		GoVersion: runtime.Version(),
	}

	w.Header().Set("Content-Type", "application/json")
	if err := encoder.NewStreamEncoder(w).Encode(resp); err != nil {
		t.logger.Error("encoding response", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// handleApiEmbed handles embedding generation requests using Ollama-compatible API
// with OpenAI-compatible multimodal extension for CLIP models.
func (ln *TermiteNode) handleApiEmbed(w http.ResponseWriter, r *http.Request) {
	defer func() { _ = r.Body.Close() }()

	// Check if embedder provider is available
	if ln.embedderRegistry == nil {
		http.Error(w, "embedding not available: no models configured", http.StatusServiceUnavailable)
		return
	}

	// Apply backpressure via request queue
	release, err := ln.requestQueue.Acquire(r.Context())
	if err != nil {
		switch err {
		case ErrQueueFull:
			RecordQueueRejection()
			WriteQueueFullResponse(w, 5*time.Second)
		case ErrRequestTimeout:
			RecordQueueTimeout()
			WriteTimeoutResponse(w)
		default:
			// Context cancelled
			http.Error(w, "request cancelled", http.StatusRequestTimeout)
		}
		return
	}
	defer release()

	// Update queue metrics
	UpdateQueueMetrics(ln.requestQueue.Stats())

	// Decode the request using generated types
	var req EmbedRequest
	if err := decoder.NewStreamDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("decoding request: %v", err), http.StatusBadRequest)
		return
	}

	// Validate model
	if req.Model == "" {
		http.Error(w, "model is required", http.StatusBadRequest)
		return
	}

	// Get embedder from provider (lazy loads if needed)
	embedder, err := ln.embedderRegistry.Get(req.Model)
	if err != nil {
		http.Error(w, fmt.Sprintf("model not found: %s", req.Model), http.StatusNotFound)
		return
	}

	// Parse input - supports text strings, arrays, and multimodal content parts
	// Uses scraping package for URL downloads with security config and S3 credentials
	contents, err := parseEmbedInput(r.Context(), req.Input, ln.contentSecurityConfig, ln.s3Credentials)
	if err != nil {
		http.Error(w, fmt.Sprintf("invalid input: %v", err), http.StatusBadRequest)
		return
	}

	if len(contents) == 0 {
		http.Error(w, "input is required", http.StatusBadRequest)
		return
	}

	// Validate MIME types against embedder capabilities
	if err := validateContentTypes(contents, embedder.Capabilities()); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Wrap embedder with caching for deduplicated requests
	cachedEmbedder := ln.embeddingCache.WrapEmbedder(embedder, req.Model)

	// Generate embeddings (with caching and singleflight deduplication)
	embeds, err := cachedEmbedder.Embed(r.Context(), contents)
	if err != nil {
		ln.logger.Error("failed to generate embeddings",
			zap.String("model", req.Model),
			zap.Error(err))
		http.Error(w, fmt.Sprintf("generating embeddings: %v", err), http.StatusInternalServerError)
		return
	}

	// Determine response format based on Accept header
	acceptHeader := r.Header.Get("Accept")

	switch acceptHeader {
	case "application/json":
		// JSON response using Ollama-compatible format
		resp := EmbedResponse{
			Model:      req.Model,
			Embeddings: embeds,
		}
		w.Header().Set("Content-Type", "application/json")
		if err := encoder.NewStreamEncoder(w).Encode(resp); err != nil {
			ln.logger.Error("encoding JSON response", zap.Error(err))
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

	default:
		// Default: binary serialization (application/octet-stream)
		w.Header().Set("Content-Type", "application/octet-stream")
		if err := SerializeFloatArrays(w, embeds); err != nil {
			ln.logger.Error("serializing embeddings", zap.Error(err))
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
}

// parseEmbedInput parses the EmbedRequest input which can be:
// - A single text string
// - An array of text strings (Ollama-compatible)
// - An array of ContentPart objects (OpenAI-compatible multimodal)
//
// For image_url content, supports:
// - Data URIs: data:image/png;base64,...
// - HTTP/HTTPS URLs: https://example.com/image.png
// - Local files: file:///path/to/image.png
// - S3 URLs: s3://endpoint/bucket/key
func parseEmbedInput(
	ctx context.Context,
	input EmbedRequest_Input,
	securityConfig *scraping.ContentSecurityConfig,
	s3Creds *s3.Credentials,
) ([][]ai.ContentPart, error) {
	// Try array of strings first (most common case)
	if arr, err := input.AsEmbedRequestInput1(); err == nil && len(arr) > 0 {
		contents := make([][]ai.ContentPart, len(arr))
		for i, t := range arr {
			contents[i] = []ai.ContentPart{ai.TextContent{Text: t}}
		}
		return contents, nil
	}

	// Try single string
	if str, err := input.AsEmbedRequestInput0(); err == nil && str != "" {
		return [][]ai.ContentPart{{ai.TextContent{Text: str}}}, nil
	}

	// Try multimodal content parts (OpenAI-compatible)
	if parts, err := input.AsEmbedRequestInput2(); err == nil && len(parts) > 0 {
		contents := make([][]ai.ContentPart, len(parts))
		for i, part := range parts {
			// Try text content
			if textPart, err := part.AsTextContentPart(); err == nil {
				contents[i] = []ai.ContentPart{ai.TextContent{Text: textPart.Text}}
				continue
			}

			// Try image URL content
			if imgPart, err := part.AsImageURLContentPart(); err == nil {
				// Use scraping package - handles data:, http://, https://, file://, s3://
				mimeType, data, err := scraping.DownloadContent(ctx, imgPart.ImageUrl.Url, securityConfig, s3Creds)
				if err != nil {
					return nil, fmt.Errorf("downloading image at index %d: %w", i, err)
				}
				contents[i] = []ai.ContentPart{ai.BinaryContent{
					MIMEType: mimeType,
					Data:     data,
				}}
				continue
			}

			return nil, fmt.Errorf("unknown content type at index %d", i)
		}
		return contents, nil
	}

	return nil, errors.New("input must be a string, array of strings, or array of content parts")
}

// validateContentTypes checks that all content types in the input are supported
// by the embedder's capabilities.
func validateContentTypes(contents [][]ai.ContentPart, caps embeddings.EmbedderCapabilities) error {
	// Build set of supported MIME types
	supported := make(map[string]bool)
	for _, m := range caps.SupportedMIMETypes {
		supported[m.MIMEType] = true
	}

	// Check each content part
	for i, parts := range contents {
		for _, part := range parts {
			switch p := part.(type) {
			case ai.TextContent:
				// Text is always supported via text/plain
				if !supported["text/plain"] {
					return fmt.Errorf("model does not support text input")
				}
			case ai.BinaryContent:
				if !supported[p.MIMEType] {
					return fmt.Errorf("unsupported MIME type at index %d: %s (model supports: %v)",
						i, p.MIMEType, getMIMETypeList(caps))
				}
			}
		}
	}

	return nil
}

// getMIMETypeList returns a list of supported MIME types for error messages.
func getMIMETypeList(caps embeddings.EmbedderCapabilities) []string {
	types := make([]string, len(caps.SupportedMIMETypes))
	for i, m := range caps.SupportedMIMETypes {
		types[i] = m.MIMEType
	}
	return types
}

// handleApiChunk handles text chunking requests
func (ln *TermiteNode) handleApiChunk(w http.ResponseWriter, r *http.Request) {
	defer func() { _ = r.Body.Close() }()

	// Apply backpressure via request queue
	release, err := ln.requestQueue.Acquire(r.Context())
	if err != nil {
		switch err {
		case ErrQueueFull:
			RecordQueueRejection()
			WriteQueueFullResponse(w, 5*time.Second)
		case ErrRequestTimeout:
			RecordQueueTimeout()
			WriteTimeoutResponse(w)
		default:
			http.Error(w, "request cancelled", http.StatusRequestTimeout)
		}
		return
	}
	defer release()

	// Update queue metrics
	UpdateQueueMetrics(ln.requestQueue.Stats())

	var req ChunkRequest
	if err := decoder.NewStreamDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("decoding request: %v", err), http.StatusBadRequest)
		return
	}

	// Validate the request
	if req.Text == "" {
		http.Error(w, "text is required", http.StatusBadRequest)
		return
	}

	// Convert ChunkConfig to internal chunkConfig type
	internalConfig := chunkConfig{
		Model:         req.Config.Model,
		TargetTokens:  req.Config.TargetTokens,
		OverlapTokens: req.Config.OverlapTokens,
		Separator:     req.Config.Separator,
		MaxChunks:     req.Config.MaxChunks,
		Threshold:     req.Config.Threshold,
	}

	// Use cached chunker to process the request
	chunks, cacheHit, err := ln.cachedChunker.Chunk(r.Context(), req.Text, internalConfig)
	if err != nil {
		ln.logger.Error("chunking failed", zap.Error(err))
		http.Error(w, fmt.Sprintf("chunking text: %v", err), http.StatusInternalServerError)
		return
	}

	// Record metrics
	modelUsed := internalConfig.Model
	if modelUsed == "" {
		modelUsed = "default"
	}
	RecordChunkerRequest(modelUsed)
	RecordChunkCreation(modelUsed, len(chunks))

	// Build response
	resp := ChunkResponse{
		Chunks:   chunks,
		Model:    internalConfig.Model,
		CacheHit: cacheHit,
	}

	// Return response
	w.Header().Set("Content-Type", "application/json")
	if err := encoder.NewStreamEncoder(w).Encode(resp); err != nil {
		ln.logger.Error("encoding response", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// handleApiRerank handles reranking requests
func (ln *TermiteNode) handleApiRerank(w http.ResponseWriter, r *http.Request) {
	defer func() { _ = r.Body.Close() }()

	// Check if reranking is available
	if ln.rerankerRegistry == nil || len(ln.rerankerRegistry.List()) == 0 {
		http.Error(w, "reranking not available", http.StatusServiceUnavailable)
		return
	}

	// Apply backpressure via request queue
	release, err := ln.requestQueue.Acquire(r.Context())
	if err != nil {
		switch err {
		case ErrQueueFull:
			RecordQueueRejection()
			WriteQueueFullResponse(w, 5*time.Second)
		case ErrRequestTimeout:
			RecordQueueTimeout()
			WriteTimeoutResponse(w)
		default:
			http.Error(w, "request cancelled", http.StatusRequestTimeout)
		}
		return
	}
	defer release()

	// Update queue metrics
	UpdateQueueMetrics(ln.requestQueue.Stats())

	// Decode request
	var req struct {
		Model   string   `json:"model"`   // Model name to use (required)
		Query   string   `json:"query"`   // Query text
		Prompts []string `json:"prompts"` // Pre-rendered document texts to rerank
	}
	if err := decoder.NewStreamDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Validate request
	if req.Model == "" {
		http.Error(w, "model is required", http.StatusBadRequest)
		return
	}
	if req.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}
	if len(req.Prompts) == 0 {
		http.Error(w, "prompts are required", http.StatusBadRequest)
		return
	}

	// Get model from registry
	reranker, err := ln.rerankerRegistry.Get(req.Model)
	if err != nil {
		http.Error(w, fmt.Sprintf("model not found: %s", req.Model), http.StatusNotFound)
		return
	}

	// Wrap reranker with caching for deduplicated requests
	cachedReranker := ln.rerankingCache.WrapReranker(reranker, req.Model)

	// Rerank prompts (with caching and singleflight deduplication)
	scores, err := cachedReranker.Rerank(r.Context(), req.Query, req.Prompts)
	if err != nil {
		ln.logger.Error("reranking failed",
			zap.String("model", req.Model),
			zap.String("query", req.Query),
			zap.Int("num_prompts", len(req.Prompts)),
			zap.Error(err))
		http.Error(w, fmt.Sprintf("reranking failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Record metrics
	RecordRerankerRequest(req.Model)
	RecordRerankingCreation(req.Model, len(req.Prompts))

	// Validate response
	if len(scores) != len(req.Prompts) {
		http.Error(w,
			fmt.Sprintf("expected %d scores, got %d", len(req.Prompts), len(scores)),
			http.StatusInternalServerError)
		return
	}

	ln.logger.Info("reranking request completed",
		zap.String("model", req.Model),
		zap.String("query", req.Query),
		zap.Int("num_prompts", len(req.Prompts)),
		zap.Int("num_scores", len(scores)))

	// Send response
	resp := struct {
		Model  string    `json:"model"`
		Scores []float32 `json:"scores"`
	}{
		Model:  req.Model,
		Scores: scores,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := encoder.NewStreamEncoder(w).Encode(resp); err != nil {
		ln.logger.Error("encoding response", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// handleApiRecognize handles NER (Named Entity Recognition) requests
func (ln *TermiteNode) handleApiRecognize(w http.ResponseWriter, r *http.Request) {
	defer func() { _ = r.Body.Close() }()

	// Check if NER is available
	if ln.nerRegistry == nil || len(ln.nerRegistry.List()) == 0 {
		http.Error(w, "NER not available: no models configured", http.StatusServiceUnavailable)
		return
	}

	// Apply backpressure via request queue
	release, err := ln.requestQueue.Acquire(r.Context())
	if err != nil {
		switch err {
		case ErrQueueFull:
			RecordQueueRejection()
			WriteQueueFullResponse(w, 5*time.Second)
		case ErrRequestTimeout:
			RecordQueueTimeout()
			WriteTimeoutResponse(w)
		default:
			http.Error(w, "request cancelled", http.StatusRequestTimeout)
		}
		return
	}
	defer release()

	// Update queue metrics
	UpdateQueueMetrics(ln.requestQueue.Stats())

	// Decode request
	var req struct {
		Model          string   `json:"model"`           // Model name to use (required)
		Texts          []string `json:"texts"`           // Texts to extract entities from
		Labels         []string `json:"labels"`          // Custom labels for GLiNER models (optional)
		RelationLabels []string `json:"relation_labels"` // Relation types to extract (optional, for models with relations capability)
	}
	if err := decoder.NewStreamDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Validate request
	if req.Model == "" {
		http.Error(w, "model is required", http.StatusBadRequest)
		return
	}
	if len(req.Texts) == 0 {
		http.Error(w, "texts are required", http.StatusBadRequest)
		return
	}

	var entities [][]ner.Entity
	var relations [][]ner.Relation

	// Check if the model supports relations and we should extract them
	hasRelationsCap := ln.nerRegistry.HasCapability(req.Model, "relations")

	// Check if this is a Recognizer (zero-shot capable)
	if ln.nerRegistry.IsRecognizer(req.Model) {
		recognizer, err := ln.nerRegistry.GetRecognizer(req.Model)
		if err != nil {
			http.Error(w, fmt.Sprintf("Recognizer not found: %s", req.Model), http.StatusNotFound)
			return
		}

		// If model supports relations, use ExtractRelations to get both entities and relations
		if hasRelationsCap {
			entities, relations, err = recognizer.ExtractRelations(r.Context(), req.Texts, req.Labels, req.RelationLabels)
			if err != nil {
				ln.logger.Error("Relation extraction failed",
					zap.String("model", req.Model),
					zap.Strings("labels", req.Labels),
					zap.Strings("relation_labels", req.RelationLabels),
					zap.Int("num_texts", len(req.Texts)),
					zap.Error(err))
				http.Error(w, fmt.Sprintf("Relation extraction failed: %v", err), http.StatusInternalServerError)
				return
			}
		} else if len(req.Labels) > 0 {
			// Use Recognizer with custom labels (zero-shot NER)
			entities, err = recognizer.RecognizeWithLabels(r.Context(), req.Texts, req.Labels)
			if err != nil {
				ln.logger.Error("Recognition failed",
					zap.String("model", req.Model),
					zap.Strings("labels", req.Labels),
					zap.Int("num_texts", len(req.Texts)),
					zap.Error(err))
				http.Error(w, fmt.Sprintf("Recognition failed: %v", err), http.StatusInternalServerError)
				return
			}
		} else {
			// Use Recognizer without custom labels (use default labels)
			entities, err = recognizer.RecognizeWithLabels(r.Context(), req.Texts, recognizer.Labels())
			if err != nil {
				ln.logger.Error("Recognition failed",
					zap.String("model", req.Model),
					zap.Int("num_texts", len(req.Texts)),
					zap.Error(err))
				http.Error(w, fmt.Sprintf("Recognition failed: %v", err), http.StatusInternalServerError)
				return
			}
		}
	} else {
		// Get standard NER model from registry
		model, err := ln.nerRegistry.Get(req.Model)
		if err != nil {
			http.Error(w, fmt.Sprintf("model not found: %s", req.Model), http.StatusNotFound)
			return
		}

		// Wrap model with caching for deduplicated requests
		cachedModel := ln.nerCache.WrapModel(model, req.Model)

		// Recognize entities (with caching and singleflight deduplication)
		entities, err = cachedModel.Recognize(r.Context(), req.Texts)
		if err != nil {
			ln.logger.Error("NER failed",
				zap.String("model", req.Model),
				zap.Int("num_texts", len(req.Texts)),
				zap.Error(err))
			http.Error(w, fmt.Sprintf("NER failed: %v", err), http.StatusInternalServerError)
			return
		}
	}

	// Record metrics
	RecordNERRequest(req.Model)
	totalEntities := 0
	for _, textEntities := range entities {
		totalEntities += len(textEntities)
	}
	RecordNERCreation(req.Model, totalEntities)

	ln.logger.Info("NER request completed",
		zap.String("model", req.Model),
		zap.Int("num_texts", len(req.Texts)),
		zap.Int("total_entities", totalEntities),
		zap.Int("total_relations", countRelations(relations)))

	// Convert internal Entity type to API response type
	apiEntities := make([][]RecognizeEntity, len(entities))
	for i, textEntities := range entities {
		apiEntities[i] = make([]RecognizeEntity, len(textEntities))
		for j, e := range textEntities {
			apiEntities[i][j] = RecognizeEntity{
				Text:  e.Text,
				Label: e.Label,
				Start: e.Start,
				End:   e.End,
				Score: e.Score,
			}
		}
	}

	// Convert internal Relation type to API response type
	var apiRelations [][]Relation
	if len(relations) > 0 {
		apiRelations = make([][]Relation, len(relations))
		for i, textRelations := range relations {
			apiRelations[i] = make([]Relation, len(textRelations))
			for j, rel := range textRelations {
				apiRelations[i][j] = Relation{
					Head: RecognizeEntity{
						Text:  rel.HeadEntity.Text,
						Label: rel.HeadEntity.Label,
						Start: rel.HeadEntity.Start,
						End:   rel.HeadEntity.End,
						Score: rel.HeadEntity.Score,
					},
					Tail: RecognizeEntity{
						Text:  rel.TailEntity.Text,
						Label: rel.TailEntity.Label,
						Start: rel.TailEntity.Start,
						End:   rel.TailEntity.End,
						Score: rel.TailEntity.Score,
					},
					Label: rel.Label,
					Score: rel.Score,
				}
			}
		}
	}

	// Send response
	nerResp := RecognizeResponse{
		Model:     req.Model,
		Entities:  apiEntities,
		Relations: apiRelations,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := encoder.NewStreamEncoder(w).Encode(nerResp); err != nil {
		ln.logger.Error("encoding response", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// countRelations returns the total number of relations across all texts
func countRelations(relations [][]ner.Relation) int {
	total := 0
	for _, textRelations := range relations {
		total += len(textRelations)
	}
	return total
}

// generateCompletionID generates a unique ID like OpenAI's "chatcmpl-xxx" format
func generateCompletionID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "chatcmpl-" + hex.EncodeToString(b)
}

// stringValue returns the string value of a pointer, or empty string if nil.
func stringValue(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

// boolValue returns the bool value of a pointer, or false if nil.
func boolValue(b *bool) bool {
	if b == nil {
		return false
	}
	return *b
}

// convertChatMessage converts an API ChatMessage to a generation.Message.
// Supports both simple string content and OpenAI-format array of content parts.
func convertChatMessage(msg ChatMessage) generation.Message {
	result := generation.Message{
		Role: string(msg.Role),
	}

	// Try as simple string first (most common case)
	if str, err := msg.Content.AsChatMessageContent0(); err == nil && str != "" {
		result.Content = str
		return result
	}

	// Try as array of content parts (OpenAI multimodal format)
	if parts, err := msg.Content.AsChatMessageContent1(); err == nil {
		for _, part := range parts {
			// Try as text content part
			if textPart, err := part.AsTextContentPart(); err == nil {
				result.Parts = append(result.Parts, generation.TextPart(textPart.Text))
				// Also set Content for backward compatibility with text-only generators
				if result.Content == "" {
					result.Content = textPart.Text
				}
			}
			// Try as image content part
			if imgPart, err := part.AsImageURLContentPart(); err == nil {
				result.Parts = append(result.Parts, generation.ImagePart(imgPart.ImageUrl.Url))
			}
		}
	}

	return result
}

// handleApiGenerate handles text generation requests using LLM models (OpenAI-compatible)
func (ln *TermiteNode) handleApiGenerate(w http.ResponseWriter, r *http.Request) {
	defer func() { _ = r.Body.Close() }()

	ln.logger.Info("Generate request received",
		zap.String("path", r.URL.Path),
		zap.String("content-type", r.Header.Get("Content-Type")))

	// Check if generation is available
	if ln.generatorRegistry == nil || len(ln.generatorRegistry.List()) == 0 {
		ln.logger.Warn("Generation not available: no models configured")
		http.Error(w, "generation not available: no models configured", http.StatusServiceUnavailable)
		return
	}

	// Apply backpressure via request queue
	release, err := ln.requestQueue.Acquire(r.Context())
	if err != nil {
		switch err {
		case ErrQueueFull:
			RecordQueueRejection()
			WriteQueueFullResponse(w, 5*time.Second)
		case ErrRequestTimeout:
			RecordQueueTimeout()
			WriteTimeoutResponse(w)
		default:
			http.Error(w, "request cancelled", http.StatusRequestTimeout)
		}
		return
	}
	defer release()

	// Update queue metrics
	UpdateQueueMetrics(ln.requestQueue.Stats())

	// Decode request
	var req GenerateRequest
	if err := decoder.NewStreamDecoder(r.Body).Decode(&req); err != nil {
		ln.logger.Error("Failed to decode generate request",
			zap.Error(err))
		http.Error(w, fmt.Sprintf("decoding request: %v", err), http.StatusBadRequest)
		return
	}

	ln.logger.Info("Generate request decoded",
		zap.String("model", req.Model),
		zap.Int("num_messages", len(req.Messages)))

	// Validate request
	if req.Model == "" {
		ln.logger.Warn("Generate request missing model")
		http.Error(w, "model is required", http.StatusBadRequest)
		return
	}
	if len(req.Messages) == 0 {
		ln.logger.Warn("Generate request missing messages")
		http.Error(w, "messages are required", http.StatusBadRequest)
		return
	}

	// Get generator from registry
	generator, err := ln.generatorRegistry.Get(req.Model)
	if err != nil {
		http.Error(w, fmt.Sprintf("model not found: %s", req.Model), http.StatusNotFound)
		return
	}

	// Check for tool support if tools are requested
	var toolParser generation.ToolParser
	if len(req.Tools) > 0 {
		ts, ok := generator.(generation.ToolSupporter)
		if !ok || !ts.SupportsTools() {
			http.Error(w, fmt.Sprintf("model %s does not support tool calling", req.Model), http.StatusBadRequest)
			return
		}
		toolParser = ts.ToolParser()
	}

	// Convert messages to internal format
	messages := make([]generation.Message, len(req.Messages))
	for i, m := range req.Messages {
		messages[i] = convertChatMessage(m)
	}

	// Set options from request, using defaults for zero values
	opts := generation.GenerateOptions{
		MaxTokens:   256,
		Temperature: 1.0,
		TopP:        1.0,
		TopK:        50,
	}
	if req.MaxTokens > 0 {
		opts.MaxTokens = req.MaxTokens
	}
	if req.Temperature > 0 {
		opts.Temperature = req.Temperature
	}
	if req.TopP > 0 {
		opts.TopP = req.TopP
	}
	if req.TopK > 0 {
		opts.TopK = req.TopK
	}
	// Try to extract tool choice from union type
	// First try string variant (auto, none, required)
	if tc, err := req.ToolChoice.AsToolChoice0(); err == nil && tc != "" {
		opts.ToolChoice = string(tc)
	} else if tc, err := req.ToolChoice.AsToolChoice1(); err == nil && tc.Function.Name != "" {
		// Function-specific variant: force calling a specific function
		opts.ToolChoice = "required"
		opts.ForcedFunctionName = tc.Function.Name
	}

	// If tools are provided, format tool declarations and prepend to system message
	if toolParser != nil && len(req.Tools) > 0 {
		// Convert API tools to internal format
		tools := make([]generation.ToolDefinition, len(req.Tools))
		for i, t := range req.Tools {
			tools[i] = generation.ToolDefinition{
				Type: string(t.Type),
				Function: generation.FunctionDefinition{
					Name:        t.Function.Name,
					Description: t.Function.Description,
					Parameters:  t.Function.Parameters,
					Strict:      t.Function.Strict,
				},
			}
		}

		// If a specific function is forced, filter tools to only that function
		if opts.ForcedFunctionName != "" {
			filteredTools := make([]generation.ToolDefinition, 0, 1)
			for _, tool := range tools {
				if tool.Function.Name == opts.ForcedFunctionName {
					filteredTools = append(filteredTools, tool)
					break
				}
			}
			if len(filteredTools) == 0 {
				http.Error(w, fmt.Sprintf("forced function %q not found in tools", opts.ForcedFunctionName), http.StatusBadRequest)
				return
			}
			tools = filteredTools
		}

		// Format tools prompt
		toolsPrompt := toolParser.FormatToolsPrompt(tools)

		// If a specific function is forced, add a directive to call it
		if opts.ForcedFunctionName != "" {
			toolsPrompt += fmt.Sprintf("\nYou MUST call the %s function. Do not respond with text, only call the function.\n", opts.ForcedFunctionName)
		}

		// Prepend to system message or create new one
		if len(messages) > 0 && messages[0].Role == "system" {
			messages[0].Content = toolsPrompt + "\n\n" + messages[0].Content
		} else {
			systemMsg := generation.Message{
				Role:    "system",
				Content: toolsPrompt,
			}
			messages = append([]generation.Message{systemMsg}, messages...)
		}
	}

	// Generate completion ID and timestamp
	completionID := generateCompletionID()
	created := int(time.Now().Unix())

	// Handle streaming vs non-streaming
	if req.Stream {
		ln.handleStreamingGenerate(w, r, req, generator, messages, opts, completionID, created)
		return
	}

	// Non-streaming: Generate text
	result, err := generator.Generate(r.Context(), messages, opts)
	if err != nil {
		ln.logger.Error("generation failed",
			zap.String("model", req.Model),
			zap.Int("num_messages", len(req.Messages)),
			zap.Error(err))
		http.Error(w, fmt.Sprintf("generation failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Record metrics
	RecordGeneratorRequest(req.Model)
	RecordTokenGeneration(req.Model, result.TokensUsed)

	// Parse tool calls from output if tools were requested
	var toolCalls []generation.ToolCall
	var responseText string
	if toolParser != nil && len(req.Tools) > 0 {
		// Feed the entire response to the parser
		toolParser.Reset()
		toolParser.Feed(result.Text)
		toolCalls, responseText = toolParser.Finish()

		ln.logger.Info("tool call parsing completed",
			zap.String("model", req.Model),
			zap.Int("tool_calls", len(toolCalls)),
			zap.Int("remaining_text_len", len(responseText)))
	} else {
		responseText = result.Text
	}

	ln.logger.Info("generation request completed",
		zap.String("model", req.Model),
		zap.Int("num_messages", len(req.Messages)),
		zap.Int("tokens_generated", result.TokensUsed),
		zap.Int("tool_calls", len(toolCalls)))

	// Map finish reason
	var finishReason FinishReason
	switch {
	case len(toolCalls) > 0:
		finishReason = FinishReasonToolCalls
	case result.FinishReason == "length":
		finishReason = FinishReasonLength
	default:
		finishReason = FinishReasonStop
	}

	// Estimate prompt tokens (rough estimate based on message content length)
	// TODO: Use actual tokenizer for accurate count
	promptTokens := 0
	for _, m := range messages {
		promptTokens += len(m.GetTextContent()) / 4 // Rough estimate: ~4 chars per token
	}

	// Build OpenAI-compatible response
	respMessage := GenerateMessage{
		Role: RoleAssistant,
	}

	// Set content or nil based on whether there are tool calls
	if len(toolCalls) > 0 {
		// When tool calls are present, content can be null or empty
		if responseText != "" {
			respMessage.Content = responseText
		}
		// Convert internal tool calls to API format
		apiToolCalls := make([]ToolCall, len(toolCalls))
		for i, tc := range toolCalls {
			apiToolCalls[i] = ToolCall{
				Id:   tc.ID,
				Type: ToolCallType(tc.Type),
				Function: ToolCallFunction{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			}
		}
		respMessage.ToolCalls = apiToolCalls
	} else {
		respMessage.Content = responseText
	}

	resp := GenerateResponse{
		Id:      completionID,
		Object:  GenerateResponseObjectChatCompletion,
		Created: created,
		Model:   req.Model,
		Choices: []GenerateChoice{
			{
				Index:        0,
				Message:      respMessage,
				FinishReason: finishReason,
				Logprobs:     nil,
			},
		},
		Usage: GenerateUsage{
			PromptTokens:     promptTokens,
			CompletionTokens: result.TokensUsed,
			TotalTokens:      promptTokens + result.TokensUsed,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := encoder.NewStreamEncoder(w).Encode(resp); err != nil {
		ln.logger.Error("encoding response", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// handleStreamingGenerate handles streaming generation with SSE
func (ln *TermiteNode) handleStreamingGenerate(
	w http.ResponseWriter,
	r *http.Request,
	req GenerateRequest,
	generator generation.Generator,
	messages []generation.Message,
	opts generation.GenerateOptions,
	completionID string,
	created int,
) {
	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	// Type-assert to StreamingGenerator for true token-by-token streaming
	streamingGen, ok := generator.(generation.StreamingGenerator)
	if !ok {
		ln.logger.Error("generator does not support streaming",
			zap.String("model", req.Model))
		fmt.Fprintf(w, "data: {\"error\": \"generator does not support streaming\"}\n\n")
		flusher.Flush()
		return
	}

	// Start streaming generation
	tokenChan, errChan, err := streamingGen.GenerateStream(r.Context(), messages, opts)
	if err != nil {
		ln.logger.Error("failed to start streaming generation",
			zap.String("model", req.Model),
			zap.Error(err))
		fmt.Fprintf(w, "data: {\"error\": \"%s\"}\n\n", err.Error())
		flusher.Flush()
		return
	}

	// Send first chunk with role
	firstChunk := GenerateChunk{
		Id:      completionID,
		Object:  GenerateChunkObjectChatCompletionChunk,
		Created: created,
		Model:   req.Model,
		Choices: []GenerateChunkChoice{
			{
				Index: 0,
				Delta: GenerateDelta{
					Role: RoleAssistant,
				},
			},
		},
	}
	data, _ := json.Marshal(firstChunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()

	// Stream tokens as they arrive
	var tokenCount int
	for token := range tokenChan {
		tokenCount++

		chunk := GenerateChunk{
			Id:      completionID,
			Object:  GenerateChunkObjectChatCompletionChunk,
			Created: created,
			Model:   req.Model,
			Choices: []GenerateChunkChoice{
				{
					Index: 0,
					Delta: GenerateDelta{
						Content: token.Token,
					},
				},
			},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	// Check for errors from the error channel
	select {
	case err := <-errChan:
		if err != nil {
			ln.logger.Error("streaming generation error",
				zap.String("model", req.Model),
				zap.Error(err))
			fmt.Fprintf(w, "data: {\"error\": \"%s\"}\n\n", err.Error())
			flusher.Flush()
			return
		}
	default:
	}

	// Record metrics
	RecordGeneratorRequest(req.Model)
	RecordTokenGeneration(req.Model, tokenCount)

	// Send final chunk with finish_reason
	finalChunk := GenerateChunk{
		Id:      completionID,
		Object:  GenerateChunkObjectChatCompletionChunk,
		Created: created,
		Model:   req.Model,
		Choices: []GenerateChunkChoice{
			{
				Index:        0,
				Delta:        GenerateDelta{},
				FinishReason: FinishReasonStop,
			},
		},
	}
	data, _ = json.Marshal(finalChunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()

	// Send [DONE] signal
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()

	ln.logger.Info("streaming generation completed",
		zap.String("model", req.Model),
		zap.Int("tokens_generated", tokenCount))
}

// handleApiRewrite handles Seq2Seq text rewriting requests
func (ln *TermiteNode) handleApiRewrite(w http.ResponseWriter, r *http.Request) {
	defer func() { _ = r.Body.Close() }()

	// Check if rewriting is available
	if ln.seq2seqRegistry == nil || len(ln.seq2seqRegistry.List()) == 0 {
		http.Error(w, "rewriting not available: no models configured", http.StatusServiceUnavailable)
		return
	}

	// Apply backpressure via request queue
	release, err := ln.requestQueue.Acquire(r.Context())
	if err != nil {
		switch err {
		case ErrQueueFull:
			RecordQueueRejection()
			WriteQueueFullResponse(w, 5*time.Second)
		case ErrRequestTimeout:
			RecordQueueTimeout()
			WriteTimeoutResponse(w)
		default:
			http.Error(w, "request cancelled", http.StatusRequestTimeout)
		}
		return
	}
	defer release()

	// Update queue metrics
	UpdateQueueMetrics(ln.requestQueue.Stats())

	// Decode request
	var req struct {
		Model  string   `json:"model"`  // Model name to use (required)
		Inputs []string `json:"inputs"` // Input texts to rewrite
	}
	if err := decoder.NewStreamDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Validate request
	if req.Model == "" {
		http.Error(w, "model is required", http.StatusBadRequest)
		return
	}
	if len(req.Inputs) == 0 {
		http.Error(w, "inputs are required", http.StatusBadRequest)
		return
	}

	// Get model from registry
	model, err := ln.seq2seqRegistry.Get(req.Model)
	if err != nil {
		http.Error(w, fmt.Sprintf("model not found: %s", req.Model), http.StatusNotFound)
		return
	}

	// Generate text
	output, err := model.Generate(r.Context(), req.Inputs)
	if err != nil {
		ln.logger.Error("rewriting failed",
			zap.String("model", req.Model),
			zap.Int("num_inputs", len(req.Inputs)),
			zap.Error(err))
		http.Error(w, fmt.Sprintf("rewriting failed: %v", err), http.StatusInternalServerError)
		return
	}

	ln.logger.Info("rewrite request completed",
		zap.String("model", req.Model),
		zap.Int("num_inputs", len(req.Inputs)),
		zap.Int("num_outputs", len(output.Texts)))

	// Send response
	resp := RewriteResponse{
		Model: req.Model,
		Texts: output.Texts,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := encoder.NewStreamEncoder(w).Encode(resp); err != nil {
		ln.logger.Error("encoding response", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}
