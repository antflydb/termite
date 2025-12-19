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

package generation

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestHugotGenerator_InitWithoutModel(t *testing.T) {
	_, err := NewHugotGeneratorWithSession("", nil, zap.NewNop())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "model path is required")
}

func TestHugotGenerator_InitWithInvalidModelPath(t *testing.T) {
	_, err := NewHugotGeneratorWithSession("/nonexistent/path", nil, zap.NewNop())
	// Should fail because model doesn't exist
	require.Error(t, err)
}

func TestPooledHugotGenerator_InitWithoutModel(t *testing.T) {
	_, err := NewPooledHugotGeneratorWithSession("", 2, nil, zap.NewNop())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "model path is required")
}

func TestGenerateOptions_Defaults(t *testing.T) {
	opts := GenerateOptions{}

	// Verify zero values
	assert.Equal(t, 0, opts.MaxTokens)
	assert.Equal(t, float32(0), opts.Temperature)
	assert.Equal(t, float32(0), opts.TopP)
	assert.Equal(t, 0, opts.TopK)
}

func TestMessage_Structure(t *testing.T) {
	msg := Message{
		Role:    "user",
		Content: "Hello, world!",
	}

	assert.Equal(t, "user", msg.Role)
	assert.Equal(t, "Hello, world!", msg.Content)
}

func TestGenerateResult_Structure(t *testing.T) {
	result := GenerateResult{
		Text:         "Generated text",
		TokensUsed:   42,
		FinishReason: "stop",
	}

	assert.Equal(t, "Generated text", result.Text)
	assert.Equal(t, 42, result.TokensUsed)
	assert.Equal(t, "stop", result.FinishReason)
}

// MockGenerator is a simple mock implementation for testing
type MockGenerator struct {
	generateFunc func(ctx context.Context, messages []Message, opts GenerateOptions) (*GenerateResult, error)
}

func (m *MockGenerator) Generate(ctx context.Context, messages []Message, opts GenerateOptions) (*GenerateResult, error) {
	if m.generateFunc != nil {
		return m.generateFunc(ctx, messages, opts)
	}
	return &GenerateResult{
		Text:         "Mock response",
		TokensUsed:   10,
		FinishReason: "stop",
	}, nil
}

func (m *MockGenerator) Close() error {
	return nil
}

func TestMockGenerator_Generate(t *testing.T) {
	mock := &MockGenerator{}

	messages := []Message{
		{Role: "user", Content: "Hello"},
	}
	opts := GenerateOptions{MaxTokens: 100}

	result, err := mock.Generate(context.Background(), messages, opts)
	require.NoError(t, err)
	assert.Equal(t, "Mock response", result.Text)
	assert.Equal(t, 10, result.TokensUsed)
	assert.Equal(t, "stop", result.FinishReason)
}

// Integration test - skipped unless model is available
func TestHugotGenerator_WithTinyRandomGemma3(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping model integration test in short mode")
	}

	// This test requires the tiny-random/gemma-3 model to be downloaded
	// with genai_config.json (ONNX Runtime GenAI format)
	modelPath := "../../../../testdata/generators/tiny-random-gemma-3"

	gen, err := NewHugotGeneratorWithSession(modelPath, nil, zap.NewNop())
	if err != nil {
		t.Skipf("Model not available at %s: %v", modelPath, err)
	}
	defer gen.Close()

	messages := []Message{
		{Role: "user", Content: "Hello"},
	}
	opts := GenerateOptions{
		MaxTokens: 10,
	}

	result, err := gen.Generate(context.Background(), messages, opts)
	require.NoError(t, err)
	require.NotEmpty(t, result.Text)
	t.Logf("Generated text: %s", result.Text)
}
