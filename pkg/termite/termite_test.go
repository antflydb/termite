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

package termite

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/antflydb/antfly-go/libaf/ai"
	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/antfly-go/libaf/reranking"
	"github.com/antflydb/termite/pkg/termite/lib/ner"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

// MockEmbedder implements the embeddings.Embedder interface for testing
type MockEmbedder struct {
	embedFunc func(ctx context.Context, values []string) ([][]float32, error)
	callCount atomic.Int32
}

func (m *MockEmbedder) Capabilities() embeddings.EmbedderCapabilities {
	return embeddings.TextOnlyCapabilities()
}

func (m *MockEmbedder) Embed(ctx context.Context, contents [][]ai.ContentPart) ([][]float32, error) {
	m.callCount.Add(1)
	values := embeddings.ExtractText(contents)
	if m.embedFunc != nil {
		return m.embedFunc(ctx, values)
	}
	// Default implementation returns simple embeddings
	result := make([][]float32, len(values))
	for i, v := range values {
		result[i] = []float32{float32(i), float32(len(v))}
	}
	return result, nil
}

func (m *MockEmbedder) GetCallCount() int32 {
	return m.callCount.Load()
}

func TestTermiteNode_HandleApiEmbed_NoRegistry(t *testing.T) {
	logger := zaptest.NewLogger(t)

	// Node without embedder registry (no models configured)
	node := &TermiteNode{
		logger: logger,
	}
	handler := NewTermiteAPI(logger, node)

	// Create Ollama-style request
	reqBody := EmbedRequest{
		Model: "test-model",
	}
	_ = reqBody.Input.FromEmbedRequestInput1([]string{"test1", "test2"})

	body, err := json.Marshal(reqBody)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/api/embed", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	// Should return 503 when no registry configured
	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
}

func TestTermiteNode_HandleApiEmbed_InvalidRequest(t *testing.T) {
	logger := zaptest.NewLogger(t)

	node := &TermiteNode{
		logger: logger,
	}
	handler := NewTermiteAPI(logger, node)

	// Test invalid JSON - should return 503 (registry check first) or 400
	req := httptest.NewRequest("POST", "/api/embed", bytes.NewReader([]byte("invalid json")))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	assert.True(t, w.Code == http.StatusServiceUnavailable || w.Code == http.StatusBadRequest)
}

func TestTermiteReranker_InitWithoutModel(t *testing.T) {
	logger := zaptest.NewLogger(t)

	// Should return nil when model is nil
	reranker, err := NewTermiteReranker(nil, logger)
	require.NoError(t, err)
	assert.Nil(t, reranker)
}

func TestHugotModel_InitWithoutModelPath(t *testing.T) {
	logger := zaptest.NewLogger(t)

	// Should return nil when model path is empty
	model, err := NewHugotModel("", logger)
	require.NoError(t, err)
	assert.Nil(t, model)
}

func TestHugotModel_InitWithInvalidModelPath(t *testing.T) {
	logger := zaptest.NewLogger(t)

	// Should return error when model path doesn't exist
	model, err := NewHugotModel("/nonexistent/path/to/model", logger)
	assert.Error(t, err)
	assert.Nil(t, model)
	assert.Contains(t, err.Error(), "does not exist")
}

// MockModel implements the reranking.Model interface for testing
type MockModel struct {
	rerankFunc func(ctx context.Context, query string, prompts []string) ([]float32, error)
	callCount  atomic.Int32
}

func (m *MockModel) Rerank(ctx context.Context, query string, prompts []string) ([]float32, error) {
	m.callCount.Add(1)
	if m.rerankFunc != nil {
		return m.rerankFunc(ctx, query, prompts)
	}
	// Default implementation returns simple scores
	scores := make([]float32, len(prompts))
	for i := range prompts {
		scores[i] = float32(len(prompts) - i) // Descending scores
	}
	return scores, nil
}

func (m *MockModel) GetCallCount() int32 {
	return m.callCount.Load()
}

func (m *MockModel) Close() error {
	return nil
}

func TestTermiteNode_HandleApiRerank_Success(t *testing.T) {
	logger := zaptest.NewLogger(t)

	// Create mock model
	mockModel := &MockModel{
		rerankFunc: func(ctx context.Context, query string, prompts []string) ([]float32, error) {
			// Return scores based on prompt count
			scores := make([]float32, len(prompts))
			for i := range prompts {
				scores[i] = float32(i) * 0.5
			}
			return scores, nil
		},
	}

	// Create mock registry with the mock model
	mockRegistry := &RerankerRegistry{
		models: map[string]reranking.Model{
			"test_model": mockModel,
		},
		logger: logger,
	}

	// Create Termite node with reranker registry
	node := &TermiteNode{
		logger:           logger,
		rerankerRegistry: mockRegistry,
		requestQueue: NewRequestQueue(RequestQueueConfig{
			MaxConcurrentRequests: 10,
			MaxQueueSize:          100,
		}, logger.Named("queue")),
		rerankingCache: NewRerankingCache(logger.Named("reranking-cache")),
	}
	handler := NewTermiteAPI(logger, node)

	// Create request (now requires model parameter)
	reqBody := struct {
		Model   string   `json:"model"`
		Query   string   `json:"query"`
		Prompts []string `json:"prompts"`
	}{
		Model: "test_model",
		Query: "test query",
		Prompts: []string{
			"first document",
			"second document",
			"third document",
		},
	}
	body, err := json.Marshal(reqBody)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/api/rerank", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	// Should return 200 OK
	assert.Equal(t, http.StatusOK, w.Code)

	// Decode response
	var resp struct {
		Model  string    `json:"model"`
		Scores []float32 `json:"scores"`
	}
	err = json.NewDecoder(w.Body).Decode(&resp)
	require.NoError(t, err)

	// Verify response
	assert.Equal(t, "test_model", resp.Model)
	assert.Len(t, resp.Scores, 3)
	assert.Equal(t, float32(0.0), resp.Scores[0])
	assert.Equal(t, float32(0.5), resp.Scores[1])
	assert.Equal(t, float32(1.0), resp.Scores[2])

	// Verify mock was called
	assert.Equal(t, int32(1), mockModel.GetCallCount())
}

func TestTermiteNode_HandleApiRerank_NotAvailable(t *testing.T) {
	logger := zaptest.NewLogger(t)

	// Create Termite node without reranker registry
	node := &TermiteNode{
		logger:           logger,
		rerankerRegistry: nil, // No reranker configured
	}
	handler := NewTermiteAPI(logger, node)

	// Create request (includes model parameter)
	reqBody := struct {
		Model   string   `json:"model"`
		Query   string   `json:"query"`
		Prompts []string `json:"prompts"`
	}{
		Model: "test_model",
		Query: "test query",
		Prompts: []string{
			"first document",
		},
	}
	body, err := json.Marshal(reqBody)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/api/rerank", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	// Should return 503 Service Unavailable
	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
	assert.Contains(t, w.Body.String(), "reranking not available")
}

func TestTermiteNode_HandleApiRerank_InvalidRequest(t *testing.T) {
	logger := zaptest.NewLogger(t)

	// Create mock model (won't be called due to validation errors)
	mockModel := &MockModel{}
	mockRegistry := &RerankerRegistry{
		models: map[string]reranking.Model{
			"test_model": mockModel,
		},
		logger: logger,
	}

	node := &TermiteNode{
		logger:           logger,
		rerankerRegistry: mockRegistry,
		requestQueue: NewRequestQueue(RequestQueueConfig{
			MaxConcurrentRequests: 10,
			MaxQueueSize:          100,
		}, logger.Named("queue")),
		rerankingCache: NewRerankingCache(logger.Named("reranking-cache")),
	}
	handler := NewTermiteAPI(logger, node)

	tests := []struct {
		name       string
		body       string
		wantStatus int
		wantError  string
	}{
		{
			name:       "invalid JSON",
			body:       "invalid json",
			wantStatus: http.StatusBadRequest,
			wantError:  "",
		},
		{
			name: "missing model",
			body: `{
				"query": "test query",
				"prompts": ["test"]
			}`,
			wantStatus: http.StatusBadRequest,
			wantError:  "model is required",
		},
		{
			name: "missing query",
			body: `{
				"model": "test_model",
				"prompts": ["test"]
			}`,
			wantStatus: http.StatusBadRequest,
			wantError:  "query is required",
		},
		{
			name: "missing documents",
			body: `{
				"model": "test_model",
				"query": "test query"
			}`,
			wantStatus: http.StatusBadRequest,
			wantError:  "prompts are required",
		},
		{
			name: "empty documents",
			body: `{
				"model": "test_model",
				"query": "test query",
				"prompts": []
			}`,
			wantStatus: http.StatusBadRequest,
			wantError:  "prompts are required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("POST", "/api/rerank", bytes.NewReader([]byte(tt.body)))
			req.Header.Set("Content-Type", "application/json")

			w := httptest.NewRecorder()
			handler.ServeHTTP(w, req)

			assert.Equal(t, tt.wantStatus, w.Code)
			if tt.wantError != "" {
				assert.Contains(t, w.Body.String(), tt.wantError)
			}
		})
	}
}

// MockNER implements the ner.Model interface for testing
type MockNER struct {
	recognizeFunc func(ctx context.Context, texts []string) ([][]ner.Entity, error)
	callCount     atomic.Int32
}

func (m *MockNER) Recognize(ctx context.Context, texts []string) ([][]ner.Entity, error) {
	m.callCount.Add(1)
	if m.recognizeFunc != nil {
		return m.recognizeFunc(ctx, texts)
	}
	// Default implementation returns simple entities
	results := make([][]ner.Entity, len(texts))
	for i := range texts {
		results[i] = []ner.Entity{
			{Text: "Entity", Label: "PER", Start: 0, End: 6, Score: 0.99},
		}
	}
	return results, nil
}

func (m *MockNER) Close() error {
	return nil
}

func (m *MockNER) GetCallCount() int32 {
	return m.callCount.Load()
}

func TestTermiteNode_HandleApiNER_Success(t *testing.T) {
	logger := zaptest.NewLogger(t)

	// Create mock NER model
	mockNER := &MockNER{
		recognizeFunc: func(ctx context.Context, texts []string) ([][]ner.Entity, error) {
			results := make([][]ner.Entity, len(texts))
			for i := range texts {
				results[i] = []ner.Entity{
					{Text: "John Smith", Label: "PER", Start: 0, End: 10, Score: 0.99},
					{Text: "Google", Label: "ORG", Start: 20, End: 26, Score: 0.98},
				}
			}
			return results, nil
		},
	}

	// Create mock registry with the mock NER model
	mockRegistry := &NERRegistry{
		models: map[string]ner.Model{
			"bert-base-ner": mockNER,
		},
		logger: logger,
	}

	// Create Termite node with NER registry
	node := &TermiteNode{
		logger:      logger,
		nerRegistry: mockRegistry,
		requestQueue: NewRequestQueue(RequestQueueConfig{
			MaxConcurrentRequests: 10,
			MaxQueueSize:          100,
		}, logger.Named("queue")),
		nerCache: NewNERCache(logger.Named("ner-cache")),
	}
	handler := NewTermiteAPI(logger, node)

	// Create NER request
	reqBody := struct {
		Model string   `json:"model"`
		Texts []string `json:"texts"`
	}{
		Model: "bert-base-ner",
		Texts: []string{
			"John Smith works at Google.",
			"Apple Inc. is in Cupertino.",
		},
	}
	body, err := json.Marshal(reqBody)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/api/recognize", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	// Should return 200 OK
	assert.Equal(t, http.StatusOK, w.Code)

	// Decode response
	var resp struct {
		Model    string              `json:"model"`
		Entities [][]RecognizeEntity `json:"entities"`
	}
	err = json.NewDecoder(w.Body).Decode(&resp)
	require.NoError(t, err)

	// Verify response
	assert.Equal(t, "bert-base-ner", resp.Model)
	assert.Len(t, resp.Entities, 2)
	assert.Len(t, resp.Entities[0], 2)
	assert.Equal(t, "John Smith", resp.Entities[0][0].Text)
	assert.Equal(t, "PER", resp.Entities[0][0].Label)
	assert.Equal(t, "Google", resp.Entities[0][1].Text)
	assert.Equal(t, "ORG", resp.Entities[0][1].Label)

	// Verify mock was called
	assert.Equal(t, int32(1), mockNER.GetCallCount())
}

func TestTermiteNode_HandleApiNER_NotAvailable(t *testing.T) {
	logger := zaptest.NewLogger(t)

	// Create Termite node without NER registry
	node := &TermiteNode{
		logger:      logger,
		nerRegistry: nil, // No NER configured
	}
	handler := NewTermiteAPI(logger, node)

	// Create NER request
	reqBody := struct {
		Model string   `json:"model"`
		Texts []string `json:"texts"`
	}{
		Model: "bert-base-ner",
		Texts: []string{"Test text"},
	}
	body, err := json.Marshal(reqBody)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/api/recognize", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	// Should return 503 Service Unavailable
	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
	assert.Contains(t, w.Body.String(), "NER not available")
}

func TestTermiteNode_HandleApiNER_InvalidRequest(t *testing.T) {
	logger := zaptest.NewLogger(t)

	// Create mock NER model
	mockNER := &MockNER{}
	mockRegistry := &NERRegistry{
		models: map[string]ner.Model{
			"bert-base-ner": mockNER,
		},
		logger: logger,
	}

	node := &TermiteNode{
		logger:      logger,
		nerRegistry: mockRegistry,
		requestQueue: NewRequestQueue(RequestQueueConfig{
			MaxConcurrentRequests: 10,
			MaxQueueSize:          100,
		}, logger.Named("queue")),
		nerCache: NewNERCache(logger.Named("ner-cache")),
	}
	handler := NewTermiteAPI(logger, node)

	tests := []struct {
		name       string
		body       string
		wantStatus int
		wantError  string
	}{
		{
			name:       "invalid JSON",
			body:       "invalid json",
			wantStatus: http.StatusBadRequest,
			wantError:  "",
		},
		{
			name: "missing model",
			body: `{
				"texts": ["test"]
			}`,
			wantStatus: http.StatusBadRequest,
			wantError:  "model is required",
		},
		{
			name: "missing texts",
			body: `{
				"model": "bert-base-ner"
			}`,
			wantStatus: http.StatusBadRequest,
			wantError:  "texts are required",
		},
		{
			name: "empty texts",
			body: `{
				"model": "bert-base-ner",
				"texts": []
			}`,
			wantStatus: http.StatusBadRequest,
			wantError:  "texts are required",
		},
		{
			name: "model not found",
			body: `{
				"model": "nonexistent-model",
				"texts": ["test"]
			}`,
			wantStatus: http.StatusNotFound,
			wantError:  "model not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("POST", "/api/recognize", bytes.NewReader([]byte(tt.body)))
			req.Header.Set("Content-Type", "application/json")

			w := httptest.NewRecorder()
			handler.ServeHTTP(w, req)

			assert.Equal(t, tt.wantStatus, w.Code)
			if tt.wantError != "" {
				assert.Contains(t, w.Body.String(), tt.wantError)
			}
		})
	}
}
