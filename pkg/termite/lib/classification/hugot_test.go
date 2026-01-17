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

package classification

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"go.uber.org/zap"
)

// findModelPath searches for a classifier model in common locations.
func findModelPath(t *testing.T) string {
	t.Helper()

	// Check common model locations
	homeDir, _ := os.UserHomeDir()
	paths := []string{
		filepath.Join(homeDir, ".termite", "models", "classifiers", "MoritzLaworski", "mDeBERTa-v3-base-mnli-xnli"),
		filepath.Join(homeDir, ".termite", "models", "classifiers", "facebook", "bart-large-mnli"),
		"../../../../testdata/classifiers/mDeBERTa-v3-base-mnli-xnli",
	}

	for _, p := range paths {
		if _, err := os.Stat(filepath.Join(p, "model.onnx")); err == nil {
			t.Logf("Found model at %s", p)
			return p
		}
	}

	return ""
}

// TestCloseWhileClassifying tests the race condition between Close() and Classify().
// With the fix in place, the -race detector should not find any races.
func TestCloseWhileClassifying(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Classifier model not found, skipping close race test")
	}

	const poolSize = 2
	logger := zap.NewNop()

	classifier, err := NewPooledHugotClassifier(modelPath, poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}

	ctx := context.Background()
	texts := []string{
		"I love this product, it's amazing!",
		"The weather is terrible today.",
	}
	labels := []string{"positive", "negative", "neutral"}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		// This runs concurrently with Close() - the -race detector catches unsync access
		_, _ = classifier.Classify(ctx, texts, labels)
	}()

	time.Sleep(10 * time.Millisecond) // Let inference begin
	_ = classifier.Close()
	wg.Wait()
}

// TestMultipleCloseIsSafe verifies that calling Close() multiple times
// doesn't cause issues (protected by sync.Once).
func TestMultipleCloseIsSafe(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Classifier model not found, skipping multiple close test")
	}

	const poolSize = 2
	logger := zap.NewNop()

	classifier, err := NewPooledHugotClassifier(modelPath, poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}

	// Do one successful classify first
	ctx := context.Background()
	texts := []string{"Test sentence"}
	labels := []string{"positive", "negative"}
	_, err = classifier.Classify(ctx, texts, labels)
	if err != nil {
		t.Fatalf("Initial classify failed: %v", err)
	}

	// Now close multiple times concurrently
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = classifier.Close() // Should not panic
		}()
	}
	wg.Wait()
}

// TestClassifyAfterClose verifies behavior when Classify is called after Close.
func TestClassifyAfterClose(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Classifier model not found, skipping classify-after-close test")
	}

	const poolSize = 2
	logger := zap.NewNop()

	classifier, err := NewPooledHugotClassifier(modelPath, poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}

	// Close first
	err = classifier.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Now try to classify
	ctx := context.Background()
	texts := []string{"Test sentence"}
	labels := []string{"positive", "negative"}
	_, err = classifier.Classify(ctx, texts, labels)
	if err == nil {
		t.Error("Expected error when classifying after close, got nil")
	}
	if err != ErrClassifierClosed {
		t.Errorf("Expected ErrClassifierClosed, got: %v", err)
	}
}
