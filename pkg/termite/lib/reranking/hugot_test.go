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

package reranking

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"go.uber.org/zap"
)

// findModelPath searches for a reranker model in common locations.
func findModelPath(t *testing.T) string {
	t.Helper()

	// Check common model locations
	homeDir, _ := os.UserHomeDir()
	paths := []string{
		filepath.Join(homeDir, ".termite", "models", "rerankers", "mixedbread-ai", "mxbai-rerank-base-v1"),
		filepath.Join(homeDir, ".termite", "models", "rerankers", "BAAI", "bge-reranker-base"),
		"../../../../testdata/rerankers/mxbai-rerank-base-v1",
	}

	for _, p := range paths {
		// Check for model_i8.onnx first (smaller), then model.onnx
		for _, modelFile := range []string{"model_i8.onnx", "model.onnx"} {
			if _, err := os.Stat(filepath.Join(p, modelFile)); err == nil {
				t.Logf("Found model at %s with %s", p, modelFile)
				return p
			}
		}
	}

	return ""
}

// getOnnxFilename returns the ONNX filename to use based on what's available.
func getOnnxFilename(modelPath string) string {
	if _, err := os.Stat(filepath.Join(modelPath, "model_i8.onnx")); err == nil {
		return "model_i8.onnx"
	}
	return "model.onnx"
}

// TestCloseWhileReranking tests the race condition between Close() and Rerank().
// With the fix in place, the -race detector should not find any races.
func TestCloseWhileReranking(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Reranker model not found, skipping close race test")
	}

	const poolSize = 2
	logger := zap.NewNop()
	onnxFile := getOnnxFilename(modelPath)

	reranker, err := NewPooledHugotReranker(modelPath, onnxFile, poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create reranker: %v", err)
	}

	ctx := context.Background()
	query := "What is machine learning?"
	prompts := []string{
		"Machine learning is a subset of artificial intelligence.",
		"The weather today is sunny and warm.",
		"Deep learning uses neural networks with many layers.",
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		// This runs concurrently with Close() - the -race detector catches unsync access
		_, _ = reranker.Rerank(ctx, query, prompts)
	}()

	time.Sleep(10 * time.Millisecond) // Let inference begin
	_ = reranker.Close()
	wg.Wait()
}

// TestMultipleCloseIsSafe verifies that calling Close() multiple times
// doesn't cause issues (protected by sync.Once).
func TestMultipleCloseIsSafe(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Reranker model not found, skipping multiple close test")
	}

	const poolSize = 2
	logger := zap.NewNop()
	onnxFile := getOnnxFilename(modelPath)

	reranker, err := NewPooledHugotReranker(modelPath, onnxFile, poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create reranker: %v", err)
	}

	// Do one successful rerank first
	ctx := context.Background()
	query := "test query"
	prompts := []string{"test prompt"}
	_, err = reranker.Rerank(ctx, query, prompts)
	if err != nil {
		t.Fatalf("Initial rerank failed: %v", err)
	}

	// Now close multiple times concurrently
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = reranker.Close() // Should not panic
		}()
	}
	wg.Wait()
}

// TestRerankAfterClose verifies behavior when Rerank is called after Close.
func TestRerankAfterClose(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Reranker model not found, skipping rerank-after-close test")
	}

	const poolSize = 2
	logger := zap.NewNop()
	onnxFile := getOnnxFilename(modelPath)

	reranker, err := NewPooledHugotReranker(modelPath, onnxFile, poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create reranker: %v", err)
	}

	// Close first
	err = reranker.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Now try to rerank
	ctx := context.Background()
	query := "test query"
	prompts := []string{"test prompt"}
	_, err = reranker.Rerank(ctx, query, prompts)
	if err == nil {
		t.Error("Expected error when reranking after close, got nil")
	}
	if err != ErrRerankerClosed {
		t.Errorf("Expected ErrRerankerClosed, got: %v", err)
	}
}
