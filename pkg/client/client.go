/*
Copyright 2025 The Antfly Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

//go:generate go tool oapi-codegen --config=cfg.yaml ../termite/openapi.yaml

// Package client provides an auto-generated Go SDK client for the Termite API.
package client

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"strings"

	externalRef0 "github.com/antflydb/antfly-go/libaf/chunking"
	"github.com/antflydb/termite/pkg/client/oapi"
)

// NewChatMessage creates a ChatMessage with string content.
// This is a convenience helper for the common case of text-only messages.
func NewChatMessage(role oapi.Role, content string) oapi.ChatMessage {
	msg := oapi.ChatMessage{Role: role}
	_ = msg.Content.FromChatMessageContent0(content)
	return msg
}

// NewUserMessage creates a user ChatMessage with string content.
func NewUserMessage(content string) oapi.ChatMessage {
	return NewChatMessage(oapi.RoleUser, content)
}

// NewSystemMessage creates a system ChatMessage with string content.
func NewSystemMessage(content string) oapi.ChatMessage {
	return NewChatMessage(oapi.RoleSystem, content)
}

// NewAssistantMessage creates an assistant ChatMessage with string content.
func NewAssistantMessage(content string) oapi.ChatMessage {
	return NewChatMessage(oapi.RoleAssistant, content)
}

// NewMultimodalUserMessage creates a user ChatMessage with text and image content.
// The imageDataURI should be a base64 data URI like "data:image/png;base64,...".
func NewMultimodalUserMessage(text string, imageDataURIs ...string) (oapi.ChatMessage, error) {
	var parts []oapi.ContentPart

	// Add text part if provided
	if text != "" {
		var textPart oapi.ContentPart
		if err := textPart.FromTextContentPart(oapi.TextContentPart{
			Type: oapi.TextContentPartTypeText,
			Text: text,
		}); err != nil {
			return oapi.ChatMessage{}, fmt.Errorf("creating text part: %w", err)
		}
		parts = append(parts, textPart)
	}

	// Add image parts
	for _, dataURI := range imageDataURIs {
		var imagePart oapi.ContentPart
		if err := imagePart.FromImageURLContentPart(oapi.ImageURLContentPart{
			Type: oapi.ImageURLContentPartTypeImageUrl,
			ImageUrl: oapi.ImageURL{
				Url: dataURI,
			},
		}); err != nil {
			return oapi.ChatMessage{}, fmt.Errorf("creating image part: %w", err)
		}
		parts = append(parts, imagePart)
	}

	msg := oapi.ChatMessage{Role: oapi.RoleUser}
	if err := msg.Content.FromChatMessageContent1(parts); err != nil {
		return oapi.ChatMessage{}, fmt.Errorf("setting content parts: %w", err)
	}
	return msg, nil
}

// TermiteClient is a client for interacting with the Termite API.
type TermiteClient struct {
	client  *oapi.ClientWithResponses
	baseURL string
}

// NewTermiteClient creates a new Termite client.
// The baseURL should be the server address (e.g., "http://localhost:8080").
// The /api prefix is automatically appended.
func NewTermiteClient(baseURL string, httpClient *http.Client) (*TermiteClient, error) {
	// Append /api prefix for the Termite API
	apiURL := strings.TrimSuffix(baseURL, "/") + "/api"

	var opts []oapi.ClientOption
	if httpClient != nil {
		opts = append(opts, oapi.WithHTTPClient(httpClient))
	}

	client, err := oapi.NewClientWithResponses(apiURL, opts...)
	if err != nil {
		return nil, err
	}
	return &TermiteClient{
		client:  client,
		baseURL: apiURL,
	}, nil
}

// Client returns the underlying oapi-codegen client for direct API access.
func (c *TermiteClient) Client() *oapi.ClientWithResponses {
	return c.client
}

// Embed generates embeddings for the given text strings.
// Returns embeddings in binary format (most efficient).
func (c *TermiteClient) Embed(ctx context.Context, model string, input []string) ([][]float32, error) {
	// Build the input union type
	var inputUnion oapi.EmbedRequest_Input
	if err := inputUnion.FromEmbedRequestInput1(input); err != nil {
		return nil, fmt.Errorf("building input: %w", err)
	}

	req := oapi.EmbedRequest{
		Model: model,
		Input: inputUnion,
	}

	// Make request - server defaults to binary response (most efficient)
	resp, err := c.client.GenerateEmbeddingsWithResponse(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	// Check for error responses
	if resp.JSON400 != nil {
		return nil, fmt.Errorf("bad request: %s", resp.JSON400.Error)
	}
	if resp.JSON404 != nil {
		return nil, fmt.Errorf("model not found: %s", resp.JSON404.Error)
	}
	if resp.JSON500 != nil {
		return nil, fmt.Errorf("server error: %s", resp.JSON500.Error)
	}

	// Check content type to determine response format
	contentType := resp.HTTPResponse.Header.Get("Content-Type")
	if strings.Contains(contentType, "application/json") {
		// JSON response
		if resp.JSON200 != nil {
			return resp.JSON200.Embeddings, nil
		}
		return nil, fmt.Errorf("unexpected JSON response: %s", string(resp.Body))
	}

	// Binary response (default)
	if resp.StatusCode() != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode(), string(resp.Body))
	}

	embeddings, err := deserializeFloatArrays(bytes.NewReader(resp.Body))
	if err != nil {
		return nil, fmt.Errorf("deserializing embeddings: %w", err)
	}

	return embeddings, nil
}

// EmbedJSON generates embeddings and returns JSON response (includes model name).
func (c *TermiteClient) EmbedJSON(ctx context.Context, model string, input []string) (*oapi.EmbedResponse, error) {
	var inputUnion oapi.EmbedRequest_Input
	if err := inputUnion.FromEmbedRequestInput1(input); err != nil {
		return nil, fmt.Errorf("building input: %w", err)
	}

	req := oapi.EmbedRequest{
		Model: model,
		Input: inputUnion,
	}

	resp, err := c.client.GenerateEmbeddingsWithResponse(ctx, req, func(ctx context.Context, req *http.Request) error {
		req.Header.Set("Accept", "application/json")
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	if resp.JSON400 != nil {
		return nil, fmt.Errorf("bad request: %s", resp.JSON400.Error)
	}
	if resp.JSON404 != nil {
		return nil, fmt.Errorf("model not found: %s", resp.JSON404.Error)
	}
	if resp.JSON500 != nil {
		return nil, fmt.Errorf("server error: %s", resp.JSON500.Error)
	}
	if resp.JSON200 == nil {
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode(), string(resp.Body))
	}

	return resp.JSON200, nil
}

// ChunkConfig contains configuration for text chunking.
type ChunkConfig struct {
	Model         string
	TargetTokens  int
	OverlapTokens int
	Separator     string
	MaxChunks     int
	Threshold     float32
}

// Chunk splits text into smaller segments using semantic or fixed-size chunking.
func (c *TermiteClient) Chunk(ctx context.Context, text string, config ChunkConfig) ([]externalRef0.Chunk, error) {
	req := oapi.ChunkRequest{
		Text: text,
		Config: oapi.ChunkConfig{
			Model:         config.Model,
			TargetTokens:  config.TargetTokens,
			OverlapTokens: config.OverlapTokens,
			Separator:     config.Separator,
			MaxChunks:     config.MaxChunks,
			Threshold:     config.Threshold,
		},
	}

	resp, err := c.client.ChunkTextWithResponse(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	if resp.JSON400 != nil {
		return nil, fmt.Errorf("bad request: %s", resp.JSON400.Error)
	}
	if resp.JSON500 != nil {
		return nil, fmt.Errorf("server error: %s", resp.JSON500.Error)
	}
	if resp.JSON200 == nil {
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode(), string(resp.Body))
	}

	return resp.JSON200.Chunks, nil
}

// Rerank re-scores pre-rendered text prompts based on relevance to a query.
func (c *TermiteClient) Rerank(ctx context.Context, model string, query string, prompts []string) ([]float32, error) {
	req := oapi.RerankRequest{
		Model:   model,
		Query:   query,
		Prompts: prompts,
	}

	resp, err := c.client.RerankPromptsWithResponse(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	if resp.JSON400 != nil {
		return nil, fmt.Errorf("bad request: %s", resp.JSON400.Error)
	}
	if resp.JSON404 != nil {
		return nil, fmt.Errorf("model not found: %s", resp.JSON404.Error)
	}
	if resp.JSON500 != nil {
		return nil, fmt.Errorf("server error: %s", resp.JSON500.Error)
	}
	if resp.JSON503 != nil {
		return nil, fmt.Errorf("service unavailable: %s", resp.JSON503.Error)
	}
	if resp.JSON200 == nil {
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode(), string(resp.Body))
	}

	return resp.JSON200.Scores, nil
}

// ListModels returns available models for embedding, chunking, and reranking.
func (c *TermiteClient) ListModels(ctx context.Context) (*oapi.ModelsResponse, error) {
	resp, err := c.client.ListModelsWithResponse(ctx)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	if resp.JSON500 != nil {
		return nil, fmt.Errorf("server error: %s", resp.JSON500.Error)
	}
	if resp.JSON200 == nil {
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode(), string(resp.Body))
	}

	return resp.JSON200, nil
}

// GetVersion returns Termite version information.
func (c *TermiteClient) GetVersion(ctx context.Context) (*oapi.VersionResponse, error) {
	resp, err := c.client.GetVersionWithResponse(ctx)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	if resp.JSON200 == nil {
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode(), string(resp.Body))
	}

	return resp.JSON200, nil
}

// Recognize extracts named entities from text using a recognizer model.
// For GLiNER models, optional labels can be specified for zero-shot NER.
func (c *TermiteClient) Recognize(ctx context.Context, model string, texts []string, labels []string) (*oapi.RecognizeResponse, error) {
	req := oapi.RecognizeRequest{
		Model:  model,
		Texts:  texts,
		Labels: labels,
	}

	resp, err := c.client.RecognizeEntitiesWithResponse(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	if resp.JSON400 != nil {
		return nil, fmt.Errorf("bad request: %s", resp.JSON400.Error)
	}
	if resp.JSON404 != nil {
		return nil, fmt.Errorf("model not found: %s", resp.JSON404.Error)
	}
	if resp.JSON500 != nil {
		return nil, fmt.Errorf("server error: %s", resp.JSON500.Error)
	}
	if resp.JSON503 != nil {
		return nil, fmt.Errorf("service unavailable: %s", resp.JSON503.Error)
	}
	if resp.JSON200 == nil {
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode(), string(resp.Body))
	}

	return resp.JSON200, nil
}

// ExtractRelations extracts entities and relations between them from text.
// Uses models with the "relations" capability (e.g., REBEL, GLiNER multitask).
// entityLabels specifies the entity types to extract (optional, uses model defaults if empty).
// relationLabels specifies the relation types to extract (optional, uses model defaults if empty).
func (c *TermiteClient) ExtractRelations(ctx context.Context, model string, texts []string, entityLabels []string, relationLabels []string) (*oapi.RecognizeResponse, error) {
	req := oapi.RecognizeRequest{
		Model:          model,
		Texts:          texts,
		Labels:         entityLabels,
		RelationLabels: relationLabels,
	}

	resp, err := c.client.RecognizeEntitiesWithResponse(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	if resp.JSON400 != nil {
		return nil, fmt.Errorf("bad request: %s", resp.JSON400.Error)
	}
	if resp.JSON404 != nil {
		return nil, fmt.Errorf("model not found: %s", resp.JSON404.Error)
	}
	if resp.JSON500 != nil {
		return nil, fmt.Errorf("server error: %s", resp.JSON500.Error)
	}
	if resp.JSON503 != nil {
		return nil, fmt.Errorf("service unavailable: %s", resp.JSON503.Error)
	}
	if resp.JSON200 == nil {
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode(), string(resp.Body))
	}

	return resp.JSON200, nil
}

// RewriteText rewrites input texts using a Seq2Seq rewriter model.
func (c *TermiteClient) RewriteText(ctx context.Context, model string, inputs []string) (*oapi.RewriteResponse, error) {
	req := oapi.RewriteRequest{
		Model:  model,
		Inputs: inputs,
	}

	resp, err := c.client.RewriteTextWithResponse(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	if resp.JSON400 != nil {
		return nil, fmt.Errorf("bad request: %s", resp.JSON400.Error)
	}
	if resp.JSON404 != nil {
		return nil, fmt.Errorf("model not found: %s", resp.JSON404.Error)
	}
	if resp.JSON500 != nil {
		return nil, fmt.Errorf("server error: %s", resp.JSON500.Error)
	}
	if resp.JSON503 != nil {
		return nil, fmt.Errorf("service unavailable: %s", resp.JSON503.Error)
	}
	if resp.JSON200 == nil {
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode(), string(resp.Body))
	}

	return resp.JSON200, nil
}

// GenerateConfig contains configuration for text generation.
type GenerateConfig struct {
	MaxTokens   int
	Temperature float32
	TopP        float32
	TopK        int
	Tools       []oapi.Tool
	ToolChoice  oapi.ToolChoice
}

// ToolChoiceAuto returns a ToolChoice that lets the model decide whether to call a tool.
func ToolChoiceAuto() oapi.ToolChoice {
	var tc oapi.ToolChoice
	_ = tc.FromToolChoice0(oapi.ToolChoice0Auto)
	return tc
}

// ToolChoiceNone returns a ToolChoice that prevents the model from calling any tools.
func ToolChoiceNone() oapi.ToolChoice {
	var tc oapi.ToolChoice
	_ = tc.FromToolChoice0(oapi.ToolChoice0None)
	return tc
}

// ToolChoiceRequired returns a ToolChoice that forces the model to call at least one tool.
func ToolChoiceRequired() oapi.ToolChoice {
	var tc oapi.ToolChoice
	_ = tc.FromToolChoice0(oapi.ToolChoice0Required)
	return tc
}

// ToolChoiceFunction returns a ToolChoice that forces the model to call a specific function.
func ToolChoiceFunction(name string) oapi.ToolChoice {
	var tc oapi.ToolChoice
	_ = tc.FromToolChoice1(oapi.ToolChoice1{
		Type: oapi.ToolChoice1TypeFunction,
		Function: struct {
			Name string `json:"name"`
		}{Name: name},
	})
	return tc
}

// Generate generates text using an LLM model (non-streaming).
func (c *TermiteClient) Generate(ctx context.Context, model string, messages []oapi.ChatMessage, config *GenerateConfig) (*oapi.GenerateResponse, error) {
	req := oapi.GenerateRequest{
		Model:    model,
		Messages: messages,
	}

	if config != nil {
		if config.MaxTokens > 0 {
			req.MaxTokens = config.MaxTokens
		}
		if config.Temperature > 0 {
			req.Temperature = config.Temperature
		}
		if config.TopP > 0 {
			req.TopP = config.TopP
		}
		if config.TopK > 0 {
			req.TopK = config.TopK
		}
		if len(config.Tools) > 0 {
			req.Tools = config.Tools
		}
		req.ToolChoice = config.ToolChoice
	}

	resp, err := c.client.GenerateContentWithResponse(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}

	if resp.JSON400 != nil {
		return nil, fmt.Errorf("bad request: %s", resp.JSON400.Error)
	}
	if resp.JSON404 != nil {
		return nil, fmt.Errorf("model not found: %s", resp.JSON404.Error)
	}
	if resp.JSON500 != nil {
		return nil, fmt.Errorf("server error: %s", resp.JSON500.Error)
	}
	if resp.JSON503 != nil {
		return nil, fmt.Errorf("service unavailable: %s", resp.JSON503.Error)
	}
	if resp.JSON200 == nil {
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode(), string(resp.Body))
	}

	return resp.JSON200, nil
}

// deserializeFloatArrays reconstructs a 2D float32 array from binary format.
// Format: uint64(numVectors) + uint64(dimension) + float32 values in little endian
func deserializeFloatArrays(r io.Reader) ([][]float32, error) {
	var numVectors uint64
	if err := binary.Read(r, binary.LittleEndian, &numVectors); err != nil {
		return nil, fmt.Errorf("reading number of vectors: %w", err)
	}
	if numVectors == 0 {
		return [][]float32{}, nil
	}
	var dimension uint64
	if err := binary.Read(r, binary.LittleEndian, &dimension); err != nil {
		return nil, fmt.Errorf("reading dimension: %w", err)
	}
	result := make([][]float32, numVectors)
	for i := range numVectors {
		result[i] = make([]float32, dimension)
		for j := range dimension {
			if err := binary.Read(r, binary.LittleEndian, &result[i][j]); err != nil {
				return nil, fmt.Errorf("reading vector %d, dimension %d: %w", i, j, err)
			}
		}
	}
	return result, nil
}
