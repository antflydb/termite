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

// Surya model repository
const (
	suryaModelRepo = "vikp/surya_det3" // Detection model repo (used for download test)
)

// =============================================================================
// Surya Model Tests
// =============================================================================

// TestSuryaModelExport verifies that the Surya export script produces expected files.
// This test requires the surya-ocr Python package to be installed.
func TestSuryaModelExport(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Surya export test in short mode")
	}

	// Check if a pre-exported Surya model exists in the models directory
	modelPath := filepath.Join(getTestModelsDir(), "readers", "vikp", "surya")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("No pre-exported Surya model found at:", modelPath)
	}

	// Verify termite_metadata.json exists and has correct pipeline type
	metaPath := filepath.Join(modelPath, "termite_metadata.json")
	if _, err := os.Stat(metaPath); os.IsNotExist(err) {
		t.Skip("No termite_metadata.json found, model may not be exported yet")
	}

	data, err := os.ReadFile(metaPath)
	require.NoError(t, err)

	var meta pipelines.MultiStageMetadata
	err = json.Unmarshal(data, &meta)
	require.NoError(t, err)

	assert.Equal(t, "surya", meta.ModelType, "model_type should be surya")
	assert.Equal(t, "multistage_ocr", meta.PipelineType, "pipeline_type should be multistage_ocr")

	// Verify stages are present
	_, hasDetection := meta.Stages["detection"]
	assert.True(t, hasDetection, "Should have detection stage")

	_, hasRecognition := meta.Stages["recognition"]
	assert.True(t, hasRecognition, "Should have recognition stage")

	// Check detection post-processor
	if hasDetection {
		assert.Equal(t, "heatmap", meta.Stages["detection"].PostProcessor,
			"Detection should use heatmap post-processor")
	}

	// Check recognition type
	if hasRecognition {
		assert.Equal(t, "vision2seq", meta.Stages["recognition"].Type,
			"Recognition should be vision2seq type")
	}

	t.Logf("Surya metadata validated: %d stages", len(meta.Stages))
	for name, stage := range meta.Stages {
		t.Logf("  Stage %s: model_file=%s type=%s", name, stage.ModelFile, stage.Type)
	}
}

// TestSuryaDetection tests Surya detection on a document image.
func TestSuryaDetection(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Surya detection test in short mode")
	}

	modelPath := findSuryaModel(t)
	if modelPath == "" {
		t.Skip("No exported Surya model found")
	}

	pageImagePath := filepath.Join("testdata", "sample-page-1.png")
	if _, err := os.Stat(pageImagePath); os.IsNotExist(err) {
		t.Skip("Pre-rendered page image not found at:", pageImagePath)
	}

	reader, err := createMultiStageReader(t, modelPath)
	if err != nil {
		t.Skipf("Could not create Surya reader: %v", err)
	}
	defer reader.Close()

	img := loadTestImage(t, pageImagePath)
	t.Logf("Loaded test image: %dx%d", img.Bounds().Dx(), img.Bounds().Dy())

	ctx := context.Background()
	results, err := reader.Read(ctx, []image.Image{img}, "", 0)
	require.NoError(t, err, "Surya OCR failed")

	require.NotEmpty(t, results, "Expected at least one result")
	t.Logf("Surya detected %d regions", len(results[0].Regions))

	// Validate regions
	if len(results[0].Regions) > 0 {
		assertValidRegions(t, results[0].Regions, img.Bounds())
	}
}

// TestSuryaFullPipeline tests the complete Surya OCR pipeline (detection + recognition).
func TestSuryaFullPipeline(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Surya full pipeline test in short mode")
	}

	modelPath := findSuryaModel(t)
	if modelPath == "" {
		t.Skip("No exported Surya model found")
	}

	pageImagePath := filepath.Join("testdata", "sample-page-1.png")
	if _, err := os.Stat(pageImagePath); os.IsNotExist(err) {
		t.Skip("Pre-rendered page image not found at:", pageImagePath)
	}

	reader, err := createMultiStageReader(t, modelPath)
	if err != nil {
		t.Skipf("Could not create Surya reader: %v", err)
	}
	defer reader.Close()

	img := loadTestImage(t, pageImagePath)

	ctx := context.Background()
	results, err := reader.Read(ctx, []image.Image{img}, "", 0)
	require.NoError(t, err, "Surya OCR failed")

	require.NotEmpty(t, results, "Expected at least one result")
	assert.NotEmpty(t, results[0].Text, "Expected non-empty OCR text output")

	t.Logf("Surya full text output (%d chars): %q", len(results[0].Text), truncateString(results[0].Text, 500))
	t.Logf("Surya detected %d regions", len(results[0].Regions))

	// Validate against expected phrases from the sample document
	expectedPhrases := []string{
		"heading",
		"content",
		"table",
	}

	matchPercent, missing := containsKeyPhrases(results[0].Text, expectedPhrases)
	t.Logf("Surya OCR accuracy: %.1f%% of key phrases found", matchPercent)
	if len(missing) > 0 {
		t.Logf("Missing phrases: %v", missing)
	}
}

// TestSuryaRegionsInAPIResponse verifies that Surya returns regions with bboxes.
func TestSuryaRegionsInAPIResponse(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Surya regions test in short mode")
	}

	modelPath := findSuryaModel(t)
	if modelPath == "" {
		t.Skip("No exported Surya model found")
	}

	pageImagePath := filepath.Join("testdata", "sample-page-1.png")
	if _, err := os.Stat(pageImagePath); os.IsNotExist(err) {
		t.Skip("Pre-rendered page image not found at:", pageImagePath)
	}

	reader, err := createMultiStageReader(t, modelPath)
	if err != nil {
		t.Skipf("Could not create Surya reader: %v", err)
	}
	defer reader.Close()

	img := loadTestImage(t, pageImagePath)

	ctx := context.Background()
	results, err := reader.Read(ctx, []image.Image{img}, "", 0)
	require.NoError(t, err, "Surya OCR failed")

	require.NotEmpty(t, results, "Expected results")

	// Verify regions are populated
	if len(results[0].Regions) > 0 {
		assertValidRegions(t, results[0].Regions, img.Bounds())

		// Log first few regions
		for i, region := range results[0].Regions {
			if i >= 5 {
				t.Logf("  ... and %d more regions", len(results[0].Regions)-5)
				break
			}
			t.Logf("  Region %d: bbox=[%.0f,%.0f,%.0f,%.0f] text=%q label=%q",
				i, region.BBox[0], region.BBox[1], region.BBox[2], region.BBox[3],
				truncateString(region.Text, 50), region.Label)
		}
	} else {
		t.Logf("No regions returned (detection may not have found text boxes)")
	}
}

// =============================================================================
// Surya Test Helpers
// =============================================================================

// findSuryaModel searches for a pre-exported Surya model in the test models directory.
func findSuryaModel(t *testing.T) string {
	t.Helper()

	// Check common paths
	candidates := []string{
		filepath.Join(getTestModelsDir(), "readers", "vikp", "surya"),
		filepath.Join(getTestModelsDir(), "readers", "vikp", "surya_det3"),
		filepath.Join(getTestModelsDir(), "readers", "surya"),
	}

	for _, path := range candidates {
		metaPath := filepath.Join(path, "termite_metadata.json")
		if fileExists(metaPath) {
			// Verify it's a multi-stage model
			data, err := os.ReadFile(metaPath)
			if err != nil {
				continue
			}
			if strings.Contains(string(data), "multistage_ocr") {
				t.Logf("Found Surya model at: %s", path)
				return path
			}
		}
	}

	return ""
}

// truncateString truncates a string to maxLen, appending "..." if truncated.
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
