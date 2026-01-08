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
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/termite"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

const (
	// mDeBERTa model name (downloaded from HuggingFace)
	mDebertaModelName = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
	// The local directory name after download includes the owner prefix
	mDebertaLocalName = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)

// TestMDebertaClassifierE2E tests the mDeBERTa zero-shot classification pipeline:
// 1. Downloads mDeBERTa model if not present (lazy download from HuggingFace)
// 2. Starts termite server with classifier model
// 3. Tests zero-shot classification with various labels
// 4. Tests multi-label classification
func TestMDebertaClassifierE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	// Ensure mDeBERTa model is downloaded from HuggingFace (lazy download)
	ensureHuggingFaceModel(t, mDebertaLocalName, mDebertaModelName, ModelTypeClassifier)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)

	// Use shared models directory from test harness
	modelsDir := getTestModelsDir()
	t.Logf("Using models directory: %s", modelsDir)

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
		testListModelsClassifier(t, ctx, termiteClient)
	})

	t.Run("ClassifyText", func(t *testing.T) {
		testClassifyText(t, ctx, termiteClient)
	})

	t.Run("ClassifyMultipleTexts", func(t *testing.T) {
		testClassifyMultipleTexts(t, ctx, termiteClient)
	})

	t.Run("ClassifyMultiLabel", func(t *testing.T) {
		testClassifyMultiLabel(t, ctx, termiteClient)
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

// testListModelsClassifier verifies the mDeBERTa model appears in the models list
func testListModelsClassifier(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	models, err := c.ListModels(ctx)
	require.NoError(t, err, "ListModels failed")

	// Check that mDeBERTa model is in the classifiers list
	foundClassifier := false
	for _, name := range models.Classifiers {
		if name == mDebertaLocalName {
			foundClassifier = true
			break
		}
	}

	if !foundClassifier {
		t.Errorf("mDeBERTa model %s not found in classifiers: %v",
			mDebertaLocalName, models.Classifiers)
	} else {
		t.Logf("Found mDeBERTa model in classifiers: %v", models.Classifiers)
	}
}

// testClassifyText tests basic zero-shot classification
func testClassifyText(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	texts := []string{
		"The new iPhone 15 Pro has an impressive camera system with advanced AI features.",
	}

	labels := []string{"technology", "sports", "politics", "entertainment"}

	resp, err := c.Classify(ctx, mDebertaLocalName, texts, labels)
	require.NoError(t, err, "Classify failed")

	assert.Equal(t, mDebertaLocalName, resp.Model)
	assert.Len(t, resp.Results, len(texts), "Should have results for each input text")

	// Log the classification results
	for i, textResults := range resp.Results {
		t.Logf("Text %d classifications:", i)
		for _, result := range textResults {
			t.Logf("  - %q: %.4f", result.Label, result.Score)
		}
	}

	// The text about iPhone should classify as technology
	assert.NotEmpty(t, resp.Results[0], "First text should have classification results")

	// Find the top label (highest score)
	topLabel := ""
	topScore := float32(0.0)
	for _, result := range resp.Results[0] {
		if result.Score > topScore {
			topScore = result.Score
			topLabel = result.Label
		}
	}

	t.Logf("Top classification: %q with score %.4f", topLabel, topScore)
	assert.Equal(t, "technology", topLabel, "iPhone text should classify as technology")
}

// testClassifyMultipleTexts tests classification of multiple texts at once
func testClassifyMultipleTexts(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	texts := []string{
		"The Lakers won the championship last night with an amazing comeback.",
		"The new climate bill passed the Senate with bipartisan support.",
		"Taylor Swift announced her new world tour dates for 2025.",
	}

	labels := []string{"sports", "politics", "entertainment", "business"}

	resp, err := c.Classify(ctx, mDebertaLocalName, texts, labels)
	require.NoError(t, err, "Classify failed")

	assert.Equal(t, mDebertaLocalName, resp.Model)
	assert.Len(t, resp.Results, len(texts), "Should have results for each input text")

	// Log and verify classifications
	expectedTopLabels := []string{"sports", "politics", "entertainment"}

	for i, textResults := range resp.Results {
		t.Logf("Text %d: %q", i, texts[i][:50])
		topLabel := ""
		topScore := float32(0.0)
		for _, result := range textResults {
			t.Logf("  - %q: %.4f", result.Label, result.Score)
			if result.Score > topScore {
				topScore = result.Score
				topLabel = result.Label
			}
		}
		t.Logf("  Top: %q (%.4f)", topLabel, topScore)
		assert.Equal(t, expectedTopLabels[i], topLabel, "Text %d should classify as %s", i, expectedTopLabels[i])
	}
}

// testClassifyMultiLabel tests multi-label classification where scores are independent
func testClassifyMultiLabel(t *testing.T, ctx context.Context, c *client.TermiteClient) {
	t.Helper()

	texts := []string{
		"The tech company's stock surged after announcing record quarterly earnings.",
	}

	// This text could reasonably be classified as both technology and business
	labels := []string{"technology", "business", "sports", "politics"}

	resp, err := c.ClassifyMultiLabel(ctx, mDebertaLocalName, texts, labels)
	require.NoError(t, err, "ClassifyMultiLabel failed")

	assert.Equal(t, mDebertaLocalName, resp.Model)
	assert.Len(t, resp.Results, len(texts), "Should have results for each input text")

	// Log the multi-label results
	t.Logf("Multi-label classification results:")
	for _, result := range resp.Results[0] {
		t.Logf("  - %q: %.4f", result.Label, result.Score)
	}

	// In multi-label mode, both technology and business should have relatively high scores
	var techScore, bizScore float32
	for _, result := range resp.Results[0] {
		if result.Label == "technology" {
			techScore = result.Score
		}
		if result.Label == "business" {
			bizScore = result.Score
		}
	}

	t.Logf("Technology score: %.4f, Business score: %.4f", techScore, bizScore)

	// Both should have scores above a threshold (indicating relevance)
	assert.Greater(t, techScore, float32(0.3), "Technology should have a high score for tech company news")
	assert.Greater(t, bizScore, float32(0.3), "Business should have a high score for stock/earnings news")
}
