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

//go:build onnx && ORT && darwin

package e2e

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/client/oapi"
	"github.com/antflydb/termite/pkg/termite"
	"github.com/antflydb/termite/pkg/termite/lib/cli"
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

// TestCoreMLBackendE2E tests CoreML backend configuration on macOS.
// This test verifies that:
// 1. The ONNX backend with CoreML is available on macOS
// 2. CoreML is configured with appropriate MLComputeUnits
// 3. Generation works with CoreML acceleration
func TestCoreMLBackendE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// This test only makes sense on macOS
	if runtime.GOOS != "darwin" {
		t.Skip("Skipping CoreML test on non-macOS platform")
	}

	t.Run("BackendAvailability", testCoreMLBackendAvailability)
	t.Run("GPUModeConfiguration", testCoreMLGPUModeConfiguration)
	t.Run("GeneratorWithCoreML", testGeneratorWithCoreML)
}

// testCoreMLBackendAvailability verifies the ONNX backend is available and configured for CoreML
func testCoreMLBackendAvailability(t *testing.T) {
	t.Helper()

	// Check ONNX backend is registered
	backend, ok := hugot.GetBackend(hugot.BackendONNX)
	require.True(t, ok, "ONNX backend should be registered")

	// Verify it's available
	require.True(t, backend.Available(), "ONNX backend should be available on macOS")

	// Check the name indicates CoreML
	name := backend.Name()
	t.Logf("ONNX backend name: %s", name)
	assert.Contains(t, name, "CoreML", "Backend name should indicate CoreML on macOS")

	// List all available backends
	available := hugot.ListAvailable()
	t.Logf("Available backends: %d", len(available))
	for _, b := range available {
		t.Logf("  - %s (%s): priority=%d", b.Name(), b.Type(), b.Priority())
	}

	// ONNX should be highest priority
	if len(available) > 0 {
		assert.Equal(t, hugot.BackendONNX, available[0].Type(),
			"ONNX should be the highest priority backend")
	}
}

// testCoreMLGPUModeConfiguration verifies GPU mode settings
func testCoreMLGPUModeConfiguration(t *testing.T) {
	t.Helper()

	// Get the default GPU mode
	mode := hugot.GetGPUMode()
	t.Logf("Default GPU mode: %s", mode)

	// Default should be auto
	assert.Equal(t, hugot.GPUModeAuto, mode,
		"Default GPU mode should be auto")

	// Test setting CoreML mode explicitly
	hugot.SetGPUMode(hugot.GPUModeCoreML)
	mode = hugot.GetGPUMode()
	assert.Equal(t, hugot.GPUModeCoreML, mode,
		"GPU mode should be coreml after setting")

	// Reset to auto
	hugot.SetGPUMode(hugot.GPUModeAuto)
	mode = hugot.GetGPUMode()
	assert.Equal(t, hugot.GPUModeAuto, mode,
		"GPU mode should be auto after reset")
}

// testGeneratorWithCoreML runs a generator with CoreML and verifies it works
func testGeneratorWithCoreML(t *testing.T) {
	t.Helper()

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
	require.NoError(t, err, "Failed to create client")

	// Test generation
	messages := []oapi.ChatMessage{
		client.NewUserMessage("What is 2+2? Reply with just the number."),
	}

	generateConfig := &client.GenerateConfig{
		MaxTokens:   50,
		Temperature: 0.1,
	}

	// Time the generation
	start := time.Now()
	resp, err := termiteClient.Generate(ctx, generatorModelName, messages, generateConfig)
	duration := time.Since(start)

	require.NoError(t, err, "Generate failed")
	require.NotEmpty(t, resp.Choices, "Response should have at least one choice")

	generatedText := resp.Choices[0].Message.Content
	completionTokens := resp.Usage.CompletionTokens

	t.Logf("Generated text: %q", generatedText)
	t.Logf("Tokens used: %d", completionTokens)
	t.Logf("Generation time: %v", duration)
	t.Logf("Tokens/sec: %.2f", float64(completionTokens)/duration.Seconds())

	// Verify output
	assert.NotEmpty(t, generatedText, "Generated text should not be empty")
	assert.Greater(t, completionTokens, 0, "Should have used some tokens")

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

// TestCoreMLDebugLogging tests with debug logging enabled to trace CoreML options
func TestCoreMLDebugLogging(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	if runtime.GOOS != "darwin" {
		t.Skip("Skipping CoreML test on non-macOS platform")
	}

	// Set TERMITE_DEBUG to enable CoreML compute plan profiling
	// This will log which hardware (ANE/GPU/CPU) executes each operation
	originalDebug := os.Getenv("TERMITE_DEBUG")
	os.Setenv("TERMITE_DEBUG", "1")
	defer func() {
		if originalDebug == "" {
			os.Unsetenv("TERMITE_DEBUG")
		} else {
			os.Setenv("TERMITE_DEBUG", originalDebug)
		}
	}()

	t.Log("TERMITE_DEBUG=1 set - CoreML will log compute plan profiling")

	// Run a simple backend check with debug enabled
	backend, ok := hugot.GetBackend(hugot.BackendONNX)
	require.True(t, ok, "ONNX backend should be registered")

	t.Logf("Backend: %s", backend.Name())
	t.Log("Check stdout for [DEBUG] messages showing CoreML options flow")

	// The debug logging added to hugot's model_ort.go will output:
	// [DEBUG] createORTGenerativeSession: CoreMLOptions = map[MLComputeUnits:ALL ProfileComputePlan:1]
	// [DEBUG] createORTGenerativeSession: providers = [CoreML], providerOptions = map[CoreML:map[...]]
}
