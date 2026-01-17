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

package generation

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"go.uber.org/zap"
)

// findModelPath searches for a generator model in common locations.
func findModelPath(t *testing.T) string {
	t.Helper()

	// Check common model locations
	homeDir, _ := os.UserHomeDir()
	paths := []string{
		filepath.Join(homeDir, ".termite", "models", "generators", "google", "gemma-2-2b-it"),
		"../../../../testdata/generators/tiny-random-gemma-3",
	}

	for _, p := range paths {
		// Check for genai_config.json which is required for ONNX RT GenAI
		if _, err := os.Stat(filepath.Join(p, "genai_config.json")); err == nil {
			t.Logf("Found model at %s", p)
			return p
		}
	}

	return ""
}

// TestCloseWhileGenerating tests the race condition between Close() and Generate().
// With the fix in place, the -race detector should not find any races.
func TestCloseWhileGenerating(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Generator model not found, skipping close race test")
	}

	const poolSize = 2
	logger := zap.NewNop()

	generator, err := NewPooledHugotGenerator(modelPath, poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create generator: %v", err)
	}

	ctx := context.Background()
	messages := []Message{
		{Role: "user", Content: "Write a short poem about clouds."},
	}
	opts := GenerateOptions{
		MaxTokens: 50,
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		// This runs concurrently with Close() - the -race detector catches unsync access
		_, _ = generator.Generate(ctx, messages, opts)
	}()

	time.Sleep(10 * time.Millisecond) // Let inference begin
	_ = generator.Close()
	wg.Wait()
}

// TestMultipleCloseIsSafe verifies that calling Close() multiple times
// doesn't cause issues (protected by sync.Once).
func TestMultipleCloseIsSafe(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Generator model not found, skipping multiple close test")
	}

	const poolSize = 2
	logger := zap.NewNop()

	generator, err := NewPooledHugotGenerator(modelPath, poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create generator: %v", err)
	}

	// Do one successful generate first
	ctx := context.Background()
	messages := []Message{
		{Role: "user", Content: "Hello"},
	}
	opts := GenerateOptions{
		MaxTokens: 5,
	}
	_, err = generator.Generate(ctx, messages, opts)
	if err != nil {
		t.Fatalf("Initial generate failed: %v", err)
	}

	// Now close multiple times concurrently
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = generator.Close() // Should not panic
		}()
	}
	wg.Wait()
}

// TestGenerateAfterClose verifies behavior when Generate is called after Close.
func TestGenerateAfterClose(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Generator model not found, skipping generate-after-close test")
	}

	const poolSize = 2
	logger := zap.NewNop()

	generator, err := NewPooledHugotGenerator(modelPath, poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create generator: %v", err)
	}

	// Close first
	err = generator.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Now try to generate
	ctx := context.Background()
	messages := []Message{
		{Role: "user", Content: "Hello"},
	}
	opts := GenerateOptions{
		MaxTokens: 5,
	}
	_, err = generator.Generate(ctx, messages, opts)
	if err == nil {
		t.Error("Expected error when generating after close, got nil")
	}
	if err != ErrGeneratorClosed {
		t.Errorf("Expected ErrGeneratorClosed, got: %v", err)
	}
}
