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

package ner

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"go.uber.org/zap"
)

// findModelPath searches for a NER model in common locations.
func findModelPath(t *testing.T) string {
	t.Helper()

	// Check common model locations
	homeDir, _ := os.UserHomeDir()
	paths := []string{
		filepath.Join(homeDir, ".termite", "models", "ner", "dslim", "bert-base-NER"),
		filepath.Join(homeDir, ".termite", "models", "ner", "dbmdz", "bert-large-cased-finetuned-conll03-english"),
		"../../../../testdata/ner/bert-base-NER",
	}

	for _, p := range paths {
		if _, err := os.Stat(filepath.Join(p, "model.onnx")); err == nil {
			t.Logf("Found model at %s", p)
			return p
		}
	}

	return ""
}

// TestCloseWhileRecognizing tests the race condition between Close() and Recognize().
// With the fix in place, the -race detector should not find any races.
func TestCloseWhileRecognizing(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("NER model not found, skipping close race test")
	}

	const poolSize = 2
	logger := zap.NewNop()

	recognizer, err := NewPooledHugotNER(modelPath, "model.onnx", poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create NER recognizer: %v", err)
	}

	ctx := context.Background()
	texts := []string{
		"John Smith works at Google in Mountain View, California.",
		"Apple Inc. was founded by Steve Jobs in Cupertino.",
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		// This runs concurrently with Close() - the -race detector catches unsync access
		_, _ = recognizer.Recognize(ctx, texts)
	}()

	time.Sleep(10 * time.Millisecond) // Let inference begin
	_ = recognizer.Close()
	wg.Wait()
}

// TestMultipleCloseIsSafe verifies that calling Close() multiple times
// doesn't cause issues (protected by sync.Once).
func TestMultipleCloseIsSafe(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("NER model not found, skipping multiple close test")
	}

	const poolSize = 2
	logger := zap.NewNop()

	recognizer, err := NewPooledHugotNER(modelPath, "model.onnx", poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create NER recognizer: %v", err)
	}

	// Do one successful recognize first
	ctx := context.Background()
	texts := []string{"John works at Google."}
	_, err = recognizer.Recognize(ctx, texts)
	if err != nil {
		t.Fatalf("Initial recognize failed: %v", err)
	}

	// Now close multiple times concurrently
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = recognizer.Close() // Should not panic
		}()
	}
	wg.Wait()
}

// TestRecognizeAfterClose verifies behavior when Recognize is called after Close.
func TestRecognizeAfterClose(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("NER model not found, skipping recognize-after-close test")
	}

	const poolSize = 2
	logger := zap.NewNop()

	recognizer, err := NewPooledHugotNER(modelPath, "model.onnx", poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create NER recognizer: %v", err)
	}

	// Close first
	err = recognizer.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Now try to recognize
	ctx := context.Background()
	texts := []string{"John works at Google."}
	_, err = recognizer.Recognize(ctx, texts)
	if err == nil {
		t.Error("Expected error when recognizing after close, got nil")
	}
	if err != ErrNERClosed {
		t.Errorf("Expected ErrNERClosed, got: %v", err)
	}
}
