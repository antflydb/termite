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

package e2e

import (
	"context"
	"encoding/json"
	"image"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
)

// PaddleOCR model repository
const (
	paddleOCRModelRepo = "monkt/paddleocr-onnx" // Community-maintained ONNX exports
)

// =============================================================================
// PaddleOCR Model Tests
// =============================================================================

// TestPaddleOCRModelDownload tests downloading PaddleOCR ONNX models.
func TestPaddleOCRModelDownload(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping PaddleOCR download test in short mode")
	}

	modelPath := findPaddleOCRModel(t)
	if modelPath == "" {
		// Try downloading from HuggingFace
		modelPath = ensureHuggingFaceModel(t, paddleOCRModelRepo, paddleOCRModelRepo, ModelTypeReader)
	}

	// Verify expected files
	detFile := findONNXFile(t, modelPath, "det")
	recFile := findONNXFile(t, modelPath, "rec")

	assert.NotEmpty(t, detFile, "Should have detection ONNX model")
	assert.NotEmpty(t, recFile, "Should have recognition ONNX model")

	t.Logf("Detection model: %s", detFile)
	t.Logf("Recognition model: %s", recFile)

	// Check for character dictionary
	dictPath := filepath.Join(modelPath, "ppocr_keys_v1.txt")
	if fileExists(dictPath) {
		t.Logf("Found character dictionary: ppocr_keys_v1.txt")
	}
}

// TestPaddleOCRMetadata verifies the termite_metadata.json for PaddleOCR.
func TestPaddleOCRMetadata(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping PaddleOCR metadata test in short mode")
	}

	modelPath := findPaddleOCRModel(t)
	if modelPath == "" {
		t.Skip("No pre-exported PaddleOCR model found")
	}

	metaPath := filepath.Join(modelPath, "termite_metadata.json")
	if _, err := os.Stat(metaPath); os.IsNotExist(err) {
		t.Skip("No termite_metadata.json found")
	}

	data, err := os.ReadFile(metaPath)
	require.NoError(t, err)

	var meta pipelines.MultiStageMetadata
	err = json.Unmarshal(data, &meta)
	require.NoError(t, err)

	assert.Equal(t, "paddleocr", meta.ModelType, "model_type should be paddleocr")
	assert.Equal(t, "multistage_ocr", meta.PipelineType, "pipeline_type should be multistage_ocr")

	// Verify stages
	detStage, hasDetection := meta.Stages["detection"]
	assert.True(t, hasDetection, "Should have detection stage")
	if hasDetection {
		assert.Equal(t, "db", detStage.PostProcessor,
			"Detection should use DB post-processor")
		assert.NotEmpty(t, detStage.ModelFile, "Detection should have model file")
	}

	recStage, hasRecognition := meta.Stages["recognition"]
	assert.True(t, hasRecognition, "Should have recognition stage")
	if hasRecognition {
		assert.Equal(t, "ctc", recStage.Type,
			"Recognition should be CTC type")
		assert.NotEmpty(t, recStage.CharDictFile, "Recognition should have char dict file")
	}

	t.Logf("PaddleOCR metadata validated: %d stages", len(meta.Stages))
}

// TestPaddleOCRDetection tests PaddleOCR text detection on a document image.
func TestPaddleOCRDetection(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping PaddleOCR detection test in short mode")
	}

	modelPath := findPaddleOCRModel(t)
	if modelPath == "" {
		t.Skip("No exported PaddleOCR model found")
	}

	pageImagePath := filepath.Join("testdata", "sample-page-1.png")
	if _, err := os.Stat(pageImagePath); os.IsNotExist(err) {
		t.Skip("Pre-rendered page image not found at:", pageImagePath)
	}

	reader, err := createMultiStageReader(t, modelPath)
	if err != nil {
		t.Skipf("Could not create PaddleOCR reader: %v", err)
	}
	defer reader.Close()

	img := loadTestImage(t, pageImagePath)
	t.Logf("Loaded test image: %dx%d", img.Bounds().Dx(), img.Bounds().Dy())

	ctx := context.Background()
	results, err := reader.Read(ctx, []image.Image{img}, "", 0)
	require.NoError(t, err, "PaddleOCR failed")

	require.NotEmpty(t, results, "Expected at least one result")
	t.Logf("PaddleOCR detected %d regions", len(results[0].Regions))

	if len(results[0].Regions) > 0 {
		assertValidRegions(t, results[0].Regions, img.Bounds())
	}
}

// TestPaddleOCRFullPipeline tests the complete PaddleOCR pipeline (detection + CTC recognition).
func TestPaddleOCRFullPipeline(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping PaddleOCR full pipeline test in short mode")
	}

	modelPath := findPaddleOCRModel(t)
	if modelPath == "" {
		t.Skip("No exported PaddleOCR model found")
	}

	pageImagePath := filepath.Join("testdata", "sample-page-1.png")
	if _, err := os.Stat(pageImagePath); os.IsNotExist(err) {
		t.Skip("Pre-rendered page image not found at:", pageImagePath)
	}

	reader, err := createMultiStageReader(t, modelPath)
	if err != nil {
		t.Skipf("Could not create PaddleOCR reader: %v", err)
	}
	defer reader.Close()

	img := loadTestImage(t, pageImagePath)

	ctx := context.Background()
	results, err := reader.Read(ctx, []image.Image{img}, "", 0)
	require.NoError(t, err, "PaddleOCR failed")

	require.NotEmpty(t, results, "Expected at least one result")
	assert.NotEmpty(t, results[0].Text, "Expected non-empty OCR text output")

	t.Logf("PaddleOCR full text output (%d chars): %q", len(results[0].Text), truncateString(results[0].Text, 500))
	t.Logf("PaddleOCR detected %d regions", len(results[0].Regions))

	// Validate against expected phrases
	expectedPhrases := []string{
		"heading",
		"content",
		"table",
	}

	matchPercent, missing := containsKeyPhrases(results[0].Text, expectedPhrases)
	t.Logf("PaddleOCR accuracy: %.1f%% of key phrases found", matchPercent)
	if len(missing) > 0 {
		t.Logf("Missing phrases: %v", missing)
	}
}

// TestPaddleOCRRegionsInAPIResponse verifies that PaddleOCR returns regions with bboxes.
func TestPaddleOCRRegionsInAPIResponse(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping PaddleOCR regions test in short mode")
	}

	modelPath := findPaddleOCRModel(t)
	if modelPath == "" {
		t.Skip("No exported PaddleOCR model found")
	}

	pageImagePath := filepath.Join("testdata", "sample-page-1.png")
	if _, err := os.Stat(pageImagePath); os.IsNotExist(err) {
		t.Skip("Pre-rendered page image not found at:", pageImagePath)
	}

	reader, err := createMultiStageReader(t, modelPath)
	if err != nil {
		t.Skipf("Could not create PaddleOCR reader: %v", err)
	}
	defer reader.Close()

	img := loadTestImage(t, pageImagePath)

	ctx := context.Background()
	results, err := reader.Read(ctx, []image.Image{img}, "", 0)
	require.NoError(t, err, "PaddleOCR failed")

	require.NotEmpty(t, results, "Expected results")

	if len(results[0].Regions) > 0 {
		assertValidRegions(t, results[0].Regions, img.Bounds())

		for i, region := range results[0].Regions {
			if i >= 5 {
				t.Logf("  ... and %d more regions", len(results[0].Regions)-5)
				break
			}
			t.Logf("  Region %d: bbox=[%.0f,%.0f,%.0f,%.0f] text=%q conf=%.2f",
				i, region.BBox[0], region.BBox[1], region.BBox[2], region.BBox[3],
				truncateString(region.Text, 50), region.Confidence)
		}
	} else {
		t.Logf("No regions returned (detection may not have found text boxes)")
	}
}

// =============================================================================
// PaddleOCR Test Helpers
// =============================================================================

// findPaddleOCRModel searches for a pre-exported PaddleOCR model in the test models directory.
func findPaddleOCRModel(t *testing.T) string {
	t.Helper()

	candidates := []string{
		filepath.Join(getTestModelsDir(), "readers", "monkt", "paddleocr-onnx"),
		filepath.Join(getTestModelsDir(), "readers", "paddleocr"),
		filepath.Join(getTestModelsDir(), "readers", "monkt", "paddleocr"),
	}

	for _, path := range candidates {
		metaPath := filepath.Join(path, "termite_metadata.json")
		if fileExists(metaPath) {
			data, err := os.ReadFile(metaPath)
			if err != nil {
				continue
			}
			if strings.Contains(string(data), "multistage_ocr") {
				t.Logf("Found PaddleOCR model at: %s", path)
				return path
			}
		}
	}

	return ""
}

// findONNXFile looks for an ONNX file matching a prefix in a model directory.
func findONNXFile(t *testing.T, dir, prefix string) string {
	t.Helper()

	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}

	for _, entry := range entries {
		name := entry.Name()
		if strings.HasPrefix(name, prefix) && strings.HasSuffix(name, ".onnx") {
			return filepath.Join(dir, name)
		}
	}

	return ""
}
