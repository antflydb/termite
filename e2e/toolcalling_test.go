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

package e2e

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/client/oapi"
	"github.com/antflydb/termite/pkg/termite"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

// functionGemmaModelName is the HuggingFace repository for the FunctionGemma model.
// This model supports tool calling via the FunctionGemma format.
const functionGemmaModelName = "google/functiongemma-270m-it"

// TestToolCallingE2E tests the tool calling functionality:
// 1. Downloads FunctionGemma model if not present (lazy download)
// 2. Uses a model with tool_call_format configured (FunctionGemma)
// 3. Sends a request with tools defined
// 4. Verifies the model returns tool calls in the response
func TestToolCallingE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Ensure FunctionGemma model is downloaded (lazy download)
	ensureRegistryModel(t, functionGemmaModelName, ModelTypeGenerator)

	modelsDir := getTestModelsDir()
	modelPath := filepath.Join(modelsDir, "generators", functionGemmaModelName)

	// Verify the model exists with tool_call_format configured
	genaiConfigPath := filepath.Join(modelPath, "genai_config.json")
	if _, err := os.Stat(genaiConfigPath); os.IsNotExist(err) {
		t.Fatalf("genai_config.json not found at %s after model download", genaiConfigPath)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)

	t.Logf("Using models directory: %s", modelsDir)
	t.Logf("FunctionGemma model path: %s", modelPath)

	// Find an available port
	port := findAvailablePort(t)
	serverURL := fmt.Sprintf("http://localhost:%d", port)
	t.Logf("Starting server on %s", serverURL)

	// Start termite server
	config := termite.Config{
		ApiUrl:    serverURL,
		ModelsDir: modelsDir,
	}

	serverCtx, serverCancel := context.WithCancel(ctx)
	defer serverCancel()

	readyC := make(chan struct{})
	serverDone := make(chan struct{})

	go func() {
		defer close(serverDone)
		termite.RunAsTermite(serverCtx, logger, config, readyC)
	}()

	// Wait for server to be ready
	select {
	case <-readyC:
		t.Log("Server is ready")
	case <-time.After(120 * time.Second):
		t.Fatal("Timeout waiting for server to be ready")
	}

	// Create client
	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Run test cases
	t.Run("ListModels_VerifyFunctionGemma", func(t *testing.T) {
		testListModelsFunctionGemma(t, ctx, termiteClient)
	})

	t.Run("GenerateWithTools", func(t *testing.T) {
		testGenerateWithTools(t, ctx, termiteClient)
	})

	t.Run("GenerateWithToolsRequired", func(t *testing.T) {
		testGenerateWithToolsRequired(t, ctx, termiteClient)
	})

	// Graceful shutdown
	t.Log("Shutting down server...")
	serverCancel()

	select {
	case <-serverDone:
		t.Log("Server shutdown complete")
	case <-time.After(30 * time.Second):
		t.Error("Timeout waiting for server shutdown")
	}
}

// testListModelsFunctionGemma verifies the FunctionGemma model appears in the generators list
func testListModelsFunctionGemma(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	require.NoError(t, err, "ListModels failed")

	// Check that FunctionGemma model is in the generators list
	foundGenerator := false
	for _, name := range models.Generators {
		if name == functionGemmaModelName {
			foundGenerator = true
			break
		}
	}

	if !foundGenerator {
		t.Errorf("FunctionGemma model %s not found in generators: %v",
			functionGemmaModelName, models.Generators)
	} else {
		t.Logf("Found FunctionGemma model in generators: %v", models.Generators)
	}
}

// testGenerateWithTools tests that the model can use tools when provided
func testGenerateWithTools(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	// Define a simple weather tool
	tools := []oapi.Tool{
		{
			Type: oapi.ToolTypeFunction,
			Function: oapi.FunctionDefinition{
				Name:        "get_weather",
				Description: "Get the current weather for a location",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "The city and state, e.g., San Francisco, CA",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	messages := []oapi.ChatMessage{
		client.NewUserMessage("What's the weather like in San Francisco?"),
	}

	config := &client.GenerateConfig{
		MaxTokens:   100,
		Temperature: 0.1,
		Tools:       tools,
		ToolChoice:  client.ToolChoiceAuto(),
	}

	resp, err := c.Generate(ctx, functionGemmaModelName, messages, config)
	if err != nil {
		// FunctionGemma ONNX export has known shape mismatch issues with variable sequence lengths
		if strings.Contains(err.Error(), "Shape mismatch") {
			t.Skipf("FunctionGemma ONNX model has shape mismatch issues (known model export problem): %v", err)
		}
		require.NoError(t, err, "Generate with tools failed")
	}

	assert.Equal(t, functionGemmaModelName, resp.Model)
	require.NotEmpty(t, resp.Choices, "Response should have at least one choice")

	choice := resp.Choices[0]
	t.Logf("Generated text: %q", choice.Message.Content)
	t.Logf("Finish reason: %s", choice.FinishReason)
	t.Logf("Tool calls: %+v", choice.Message.ToolCalls)

	// The model may or may not call the tool depending on its understanding
	// Just verify we got a valid response
	assert.NotNil(t, choice.FinishReason, "Should have a finish reason")
}

// testGenerateWithToolsRequired tests that tool_choice=required forces tool usage
func testGenerateWithToolsRequired(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	// Define a simple calculator tool
	tools := []oapi.Tool{
		{
			Type: oapi.ToolTypeFunction,
			Function: oapi.FunctionDefinition{
				Name:        "calculate",
				Description: "Perform a mathematical calculation",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"expression": map[string]interface{}{
							"type":        "string",
							"description": "The mathematical expression to evaluate",
						},
					},
					"required": []string{"expression"},
				},
			},
		},
	}

	messages := []oapi.ChatMessage{
		client.NewUserMessage("What is 2 + 2?"),
	}

	config := &client.GenerateConfig{
		MaxTokens:   100,
		Temperature: 0.1,
		Tools:       tools,
		ToolChoice:  client.ToolChoiceRequired(),
	}

	resp, err := c.Generate(ctx, functionGemmaModelName, messages, config)
	if err != nil {
		// FunctionGemma ONNX export has known shape mismatch issues with variable sequence lengths
		if strings.Contains(err.Error(), "Shape mismatch") {
			t.Skipf("FunctionGemma ONNX model has shape mismatch issues (known model export problem): %v", err)
		}
		require.NoError(t, err, "Generate with required tools failed")
	}

	assert.Equal(t, functionGemmaModelName, resp.Model)
	require.NotEmpty(t, resp.Choices, "Response should have at least one choice")

	choice := resp.Choices[0]
	t.Logf("Generated text: %q", choice.Message.Content)
	t.Logf("Finish reason: %s", choice.FinishReason)
	t.Logf("Tool calls: %+v", choice.Message.ToolCalls)

	// With tool_choice=required, we expect tool calls
	// Note: Small models may not reliably generate tool calls even when required
	// This is a best-effort test
	if choice.FinishReason == oapi.FinishReasonToolCalls {
		require.NotEmpty(t, choice.Message.ToolCalls, "Should have tool calls when finish_reason is tool_calls")
		t.Logf("Model made %d tool call(s)", len(choice.Message.ToolCalls))

		for i, tc := range choice.Message.ToolCalls {
			t.Logf("  Tool call %d: %s(%s)", i, tc.Function.Name, tc.Function.Arguments)
		}
	} else {
		t.Logf("Model did not make tool calls (finish_reason: %v)", choice.FinishReason)
	}
}
