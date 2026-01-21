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

package chunking

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/antflydb/antfly-go/libaf/chunking"
	"go.uber.org/zap"
)

// findModelPath searches for a chunker model in common locations.
func findModelPath(t *testing.T) string {
	t.Helper()

	// Check common model locations
	homeDir, _ := os.UserHomeDir()
	paths := []string{
		filepath.Join(homeDir, ".termite", "models", "chunkers", "mirth", "chonky-mmbert-small-multilingual-1"),
		"../../../../testdata/chunkers/chonky-mmbert-small-multilingual-1",
	}

	for _, p := range paths {
		if _, err := os.Stat(filepath.Join(p, "model.onnx")); err == nil {
			t.Logf("Found model at %s", p)
			return p
		}
	}

	return ""
}

// TestCloseWhileChunking tests the race condition between Close() and Chunk().
// With the fix in place, the -race detector should not find any races.
func TestCloseWhileChunking(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Chunker model not found, skipping close race test")
	}

	const poolSize = 2
	logger := zap.NewNop()
	config := DefaultHugotChunkerConfig()

	chunker, err := NewPooledHugotChunker(config, modelPath, "model.onnx", poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create chunker: %v", err)
	}

	ctx := context.Background()
	// Use a longer text to make chunking take more time
	text := `Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.

These systems improve their performance on specific tasks over time without being explicitly programmed. Deep learning, a more specialized form of machine learning, uses artificial neural networks with many layers.

The field has seen tremendous growth in recent years, with applications ranging from image recognition to natural language processing. Companies across industries are adopting these technologies to automate processes and gain insights from their data.`

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		// This runs concurrently with Close() - the -race detector catches unsync access
		_, _ = chunker.Chunk(ctx, text, chunking.ChunkOptions{})
	}()

	time.Sleep(10 * time.Millisecond) // Let inference begin
	_ = chunker.Close()
	wg.Wait()
}

// TestMultipleCloseIsSafe verifies that calling Close() multiple times
// doesn't cause issues (protected by sync.Once).
func TestMultipleCloseIsSafe(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Chunker model not found, skipping multiple close test")
	}

	const poolSize = 2
	logger := zap.NewNop()
	config := DefaultHugotChunkerConfig()

	chunker, err := NewPooledHugotChunker(config, modelPath, "model.onnx", poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create chunker: %v", err)
	}

	// Do one successful chunk first
	ctx := context.Background()
	text := "This is a test document for chunking."
	_, err = chunker.Chunk(ctx, text, chunking.ChunkOptions{})
	if err != nil {
		t.Fatalf("Initial chunk failed: %v", err)
	}

	// Now close multiple times concurrently
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = chunker.Close() // Should not panic
		}()
	}
	wg.Wait()
}

// TestChunkAfterClose verifies behavior when Chunk is called after Close.
func TestChunkAfterClose(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Chunker model not found, skipping chunk-after-close test")
	}

	const poolSize = 2
	logger := zap.NewNop()
	config := DefaultHugotChunkerConfig()

	chunker, err := NewPooledHugotChunker(config, modelPath, "model.onnx", poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create chunker: %v", err)
	}

	// Close first
	err = chunker.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Now try to chunk
	ctx := context.Background()
	text := "Test document after close."
	_, err = chunker.Chunk(ctx, text, chunking.ChunkOptions{})
	if err == nil {
		t.Error("Expected error when chunking after close, got nil")
	}
	if err != ErrChunkerClosed {
		t.Errorf("Expected ErrChunkerClosed, got: %v", err)
	}
}
