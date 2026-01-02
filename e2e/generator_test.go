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
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/client/oapi"
	"github.com/antflydb/termite/pkg/termite"
	"github.com/antflydb/termite/pkg/termite/lib/cli"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

// generatorModelName is the HuggingFace repository for the generator model.
// This uses the onnxruntime-genai compatible Gemma 3 model.
// Uses owner/model format matching the directory structure.
const generatorModelName = "onnxruntime/Gemma-3-ONNX"

// TestGeneratorE2E tests the Generator (LLM text generation) pipeline:
// 1. Downloads the generator model from HuggingFace if not present
// 2. Starts termite server with Generator model
// 3. Tests text generation from messages
func TestGeneratorE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Check if generator model exists locally, pull from HuggingFace if not
	modelsDir := getTestModelsDir()
	generatorsDir := getGeneratorModelsDir()
	modelPath := filepath.Join(generatorsDir, generatorModelName)

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Logf("Generator model not found at %s, pulling from HuggingFace: %s", modelPath, generatorModelName)

		// Pull the model from HuggingFace (auto-detects generator type)
		err := cli.PullFromHuggingFace(generatorModelName, cli.HuggingFaceOptions{
			ModelsDir: modelsDir,
			ModelType: "", // Auto-detect
		})
		if err != nil {
			t.Fatalf("Failed to pull generator model from HuggingFace: %v", err)
		}
		t.Logf("Successfully pulled generator model to %s", modelPath)
	}

	// Verify genai_config.json exists (required for onnxruntime-genai models)
	genaiConfigPath := filepath.Join(modelPath, "genai_config.json")
	if _, err := os.Stat(genaiConfigPath); os.IsNotExist(err) {
		t.Fatalf("genai_config.json not found at %s after pull. The model may not have downloaded correctly.", genaiConfigPath)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)

	t.Logf("Using models directory: %s", modelsDir)
	t.Logf("Generator model path: %s", modelPath)

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
	t.Run("ListModels", func(t *testing.T) {
		testListModelsGenerator(t, ctx, termiteClient)
	})

	t.Run("Generate", func(t *testing.T) {
		testGenerate(t, ctx, termiteClient)
	})

	t.Run("GenerateWithImage", func(t *testing.T) {
		testGenerateWithImage(t, ctx, termiteClient)
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

// testListModelsGenerator verifies the Generator model appears in the models list
func testListModelsGenerator(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	require.NoError(t, err, "ListModels failed")

	// Check that Generator model is in the generators list
	foundGenerator := false
	for _, name := range models.Generators {
		if name == generatorModelName {
			foundGenerator = true
			break
		}
	}

	if !foundGenerator {
		t.Errorf("Generator model %s not found in generators: %v",
			generatorModelName, models.Generators)
	} else {
		t.Logf("Found Generator model in generators: %v", models.Generators)
	}
}

// testGenerate tests text generation from messages
func testGenerate(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	messages := []oapi.ChatMessage{
		client.NewUserMessage("What is 2+2? Reply with just the number."),
	}

	config := &client.GenerateConfig{
		MaxTokens:   50,
		Temperature: 0.1, // Low temperature for more deterministic output
	}

	resp, err := c.Generate(ctx, generatorModelName, messages, config)
	require.NoError(t, err, "Generate failed")

	assert.Equal(t, generatorModelName, resp.Model)

	// OpenAI-compatible response has Choices array
	require.NotEmpty(t, resp.Choices, "Response should have at least one choice")

	generatedText := resp.Choices[0].Message.Content
	finishReason := resp.Choices[0].FinishReason
	completionTokens := resp.Usage.CompletionTokens

	// Log the generated text
	t.Logf("Generated text: %q", generatedText)
	t.Logf("Tokens used: %d", completionTokens)
	t.Logf("Finish reason: %s", finishReason)

	// Should have non-empty generated text
	assert.NotEmpty(t, generatedText, "Generated text should not be empty")

	// Should have reasonable token count
	assert.Greater(t, completionTokens, 0, "Should have used some tokens")
}

// testGenerateWithImage tests multimodal generation with an image
func testGenerateWithImage(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	// Create a simple test image (100x100 red square)
	imageData := createGeneratorTestImage(t, 100, 100, color.RGBA{255, 0, 0, 255})
	base64Image := base64.StdEncoding.EncodeToString(imageData)
	dataURI := "data:image/png;base64," + base64Image

	// Create multimodal message with text and image
	multimodalMsg, err := client.NewMultimodalUserMessage(
		"What color is this image? Reply with just the color name.",
		dataURI,
	)
	require.NoError(t, err, "Failed to create multimodal message")

	messages := []oapi.ChatMessage{multimodalMsg}

	config := &client.GenerateConfig{
		MaxTokens:   50,
		Temperature: 0.1, // Low temperature for more deterministic output
	}

	resp, err := c.Generate(ctx, generatorModelName, messages, config)
	require.NoError(t, err, "Generate with image failed")

	assert.Equal(t, generatorModelName, resp.Model)

	// OpenAI-compatible response has Choices array
	require.NotEmpty(t, resp.Choices, "Response should have at least one choice")

	generatedText := resp.Choices[0].Message.Content
	finishReason := resp.Choices[0].FinishReason
	completionTokens := resp.Usage.CompletionTokens

	// Log the generated text
	t.Logf("Generated text (multimodal): %q", generatedText)
	t.Logf("Tokens used: %d", completionTokens)
	t.Logf("Finish reason: %s", finishReason)

	// Should have non-empty generated text
	assert.NotEmpty(t, generatedText, "Generated text should not be empty")

	// Should have reasonable token count
	assert.Greater(t, completionTokens, 0, "Should have used some tokens")
}

// createGeneratorTestImage creates a PNG image with the specified dimensions and color
func createGeneratorTestImage(t *testing.T, width, height int, c color.Color) []byte {
	t.Helper()

	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, c)
		}
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("Failed to encode PNG: %v", err)
	}

	return buf.Bytes()
}
