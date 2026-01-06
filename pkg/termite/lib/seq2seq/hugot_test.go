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

package seq2seq

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

// getTestModelPath returns the path to a test model if available.
// Uses TERMITE_MODELS_DIR environment variable or skips the test.
func getTestModelPath(t *testing.T, modelType, modelName string) string {
	t.Helper()

	modelsDir := os.Getenv("TERMITE_MODELS_DIR")
	if modelsDir == "" {
		t.Skip("TERMITE_MODELS_DIR not set, skipping integration test")
	}

	modelPath := filepath.Join(modelsDir, modelType, modelName)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Model %s not found at %s, skipping", modelName, modelPath)
	}

	return modelPath
}

// TestHugotSeq2Seq_NewHugotSeq2Seq tests basic model initialization.
func TestHugotSeq2Seq_NewHugotSeq2Seq(t *testing.T) {
	modelPath := getTestModelPath(t, "rewriters", "lmqg/flan-t5-small-squad-qg")
	logger := zaptest.NewLogger(t)

	model, err := NewHugotSeq2Seq(modelPath, logger)
	require.NoError(t, err, "Failed to create HugotSeq2Seq")
	require.NotNil(t, model)

	defer func() {
		err := model.Close()
		assert.NoError(t, err, "Failed to close model")
	}()

	// Verify config was loaded
	config := model.Config()
	assert.NotEmpty(t, config.ModelID)
}

// TestHugotSeq2Seq_Generate tests basic text generation.
func TestHugotSeq2Seq_Generate(t *testing.T) {
	modelPath := getTestModelPath(t, "rewriters", "lmqg/flan-t5-small-squad-qg")
	logger := zaptest.NewLogger(t)

	model, err := NewHugotSeq2Seq(modelPath, logger)
	require.NoError(t, err)
	defer model.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	inputs := []string{
		"generate question: <hl> Beyonce <hl> Beyonce starred as Etta James in Cadillac Records.",
	}

	output, err := model.Generate(ctx, inputs)
	require.NoError(t, err)
	require.NotNil(t, output)

	assert.Len(t, output.Texts, 1, "Should have one output per input")
	assert.NotEmpty(t, output.Texts[0], "Should have generated text")
	t.Logf("Generated: %v", output.Texts[0])
}

// TestHugotSeq2Seq_Generate_Empty tests generation with empty input.
func TestHugotSeq2Seq_Generate_Empty(t *testing.T) {
	modelPath := getTestModelPath(t, "rewriters", "lmqg/flan-t5-small-squad-qg")
	logger := zaptest.NewLogger(t)

	model, err := NewHugotSeq2Seq(modelPath, logger)
	require.NoError(t, err)
	defer model.Close()

	ctx := context.Background()
	output, err := model.Generate(ctx, []string{})
	require.NoError(t, err)
	assert.Empty(t, output.Texts)
	assert.Empty(t, output.Tokens)
}

// TestHugotSeq2Seq_GenerateQuestions tests question generation.
func TestHugotSeq2Seq_GenerateQuestions(t *testing.T) {
	modelPath := getTestModelPath(t, "rewriters", "lmqg/flan-t5-small-squad-qg")
	logger := zaptest.NewLogger(t)

	model, err := NewHugotSeq2Seq(modelPath, logger)
	require.NoError(t, err)
	defer model.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	pairs := []AnswerContextPair{
		{
			Answer:  "Python",
			Context: "Python is a high-level, interpreted programming language.",
		},
	}

	output, err := model.GenerateQuestions(ctx, pairs)
	require.NoError(t, err)
	require.NotNil(t, output)

	assert.Len(t, output.Texts, 1)
	assert.NotEmpty(t, output.Texts[0])
	t.Logf("Generated question: %v", output.Texts[0])
}

// TestHugotSeq2Seq_GenerateQuestions_Empty tests with empty pairs.
func TestHugotSeq2Seq_GenerateQuestions_Empty(t *testing.T) {
	modelPath := getTestModelPath(t, "rewriters", "lmqg/flan-t5-small-squad-qg")
	logger := zaptest.NewLogger(t)

	model, err := NewHugotSeq2Seq(modelPath, logger)
	require.NoError(t, err)
	defer model.Close()

	ctx := context.Background()
	output, err := model.GenerateQuestions(ctx, []AnswerContextPair{})
	require.NoError(t, err)
	assert.Empty(t, output.Texts)
}

// TestHugotSeq2Seq_Paraphrase tests the paraphrase method.
func TestHugotSeq2Seq_Paraphrase(t *testing.T) {
	modelPath := getTestModelPath(t, "rewriters", "tuner007/pegasus_paraphrase")
	logger := zaptest.NewLogger(t)

	model, err := NewHugotSeq2Seq(modelPath, logger)
	require.NoError(t, err)
	defer model.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	texts := []string{
		"The ultimate test of your knowledge is your capacity to convey it to another.",
	}

	output, err := model.Paraphrase(ctx, texts)
	require.NoError(t, err)
	require.NotNil(t, output)

	assert.Len(t, output.Texts, 1, "Should have one output per input")
	assert.NotEmpty(t, output.Texts[0], "Should have generated paraphrases")

	// Log all paraphrases
	t.Logf("Input: %s", texts[0])
	for i, paraphrase := range output.Texts[0] {
		t.Logf("  Paraphrase %d: %s", i, paraphrase)
	}
}

// TestHugotSeq2Seq_Paraphrase_Empty tests paraphrase with empty input.
func TestHugotSeq2Seq_Paraphrase_Empty(t *testing.T) {
	modelPath := getTestModelPath(t, "rewriters", "tuner007/pegasus_paraphrase")
	logger := zaptest.NewLogger(t)

	model, err := NewHugotSeq2Seq(modelPath, logger)
	require.NoError(t, err)
	defer model.Close()

	ctx := context.Background()
	output, err := model.Paraphrase(ctx, []string{})
	require.NoError(t, err)
	assert.Empty(t, output.Texts)
}

// TestHugotSeq2Seq_ContextCancellation tests that context cancellation is respected.
func TestHugotSeq2Seq_ContextCancellation(t *testing.T) {
	modelPath := getTestModelPath(t, "rewriters", "lmqg/flan-t5-small-squad-qg")
	logger := zaptest.NewLogger(t)

	model, err := NewHugotSeq2Seq(modelPath, logger)
	require.NoError(t, err)
	defer model.Close()

	// Create an already-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	inputs := []string{"test input"}
	_, err = model.Generate(ctx, inputs)
	assert.Error(t, err, "Should return error for cancelled context")
	assert.ErrorIs(t, err, context.Canceled)
}

// TestHugotSeq2Seq_Config tests config retrieval.
func TestHugotSeq2Seq_Config(t *testing.T) {
	modelPath := getTestModelPath(t, "rewriters", "lmqg/flan-t5-small-squad-qg")
	logger := zaptest.NewLogger(t)

	model, err := NewHugotSeq2Seq(modelPath, logger)
	require.NoError(t, err)
	defer model.Close()

	config := model.Config()
	assert.NotEmpty(t, config.Task, "Config should have a task")
	assert.Greater(t, config.MaxLength, 0, "MaxLength should be positive")
}

// TestHugotSeq2Seq_Close tests proper resource cleanup.
func TestHugotSeq2Seq_Close(t *testing.T) {
	modelPath := getTestModelPath(t, "rewriters", "lmqg/flan-t5-small-squad-qg")
	logger := zaptest.NewLogger(t)

	model, err := NewHugotSeq2Seq(modelPath, logger)
	require.NoError(t, err)

	// Close should not error
	err = model.Close()
	assert.NoError(t, err)

	// Calling Close again should be safe
	err = model.Close()
	// Note: behavior after close is implementation-defined
}

// TestNewHugotSeq2Seq_InvalidPath tests error handling for invalid paths.
func TestNewHugotSeq2Seq_InvalidPath(t *testing.T) {
	logger := zaptest.NewLogger(t)

	_, err := NewHugotSeq2Seq("/nonexistent/path", logger)
	assert.Error(t, err, "Should error for nonexistent path")
}

// TestNewHugotSeq2Seq_EmptyPath tests error handling for empty path.
func TestNewHugotSeq2Seq_EmptyPath(t *testing.T) {
	logger := zaptest.NewLogger(t)

	_, err := NewHugotSeq2Seq("", logger)
	assert.Error(t, err, "Should error for empty path")
}

// TestHugotSeq2Seq_NilLogger tests that nil logger is handled.
func TestHugotSeq2Seq_NilLogger(t *testing.T) {
	modelPath := getTestModelPath(t, "rewriters", "lmqg/flan-t5-small-squad-qg")

	model, err := NewHugotSeq2Seq(modelPath, nil)
	require.NoError(t, err, "Should handle nil logger")
	defer model.Close()

	// Should still work
	ctx := context.Background()
	output, err := model.Generate(ctx, []string{"test input"})
	require.NoError(t, err)
	assert.NotNil(t, output)
}
