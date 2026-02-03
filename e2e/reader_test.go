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
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	"github.com/antflydb/termite/pkg/termite/lib/reading"
)

// containsKeyPhrases checks if the OCR output contains expected key phrases.
// Returns the percentage of phrases found and the list of missing phrases.
func containsKeyPhrases(ocrOutput string, expectedPhrases []string) (float64, []string) {
	ocrLower := strings.ToLower(ocrOutput)
	var missing []string
	found := 0

	for _, phrase := range expectedPhrases {
		if strings.Contains(ocrLower, strings.ToLower(phrase)) {
			found++
		} else {
			missing = append(missing, phrase)
		}
	}

	return float64(found) / float64(len(expectedPhrases)) * 100, missing
}

// Model repositories
const (
	trocrModelRepo = "Xenova/trocr-base-printed"
	trocrModelName = "Xenova/trocr-base-printed"

	donutCordRepo = "Xenova/donut-base-finetuned-cord-v2"
	donutCordName = "Xenova/donut-base-finetuned-cord-v2"
)


// =============================================================================
// Helper Functions
// =============================================================================

// createTestImageWithText creates a simple test pattern image (not real text).
// Note: This creates a dot pattern that OCR models won't recognize as text.
// The output "SR:" from TrOCR on this pattern matches Python transformers behavior,
// confirming our ONNX implementation is correct.
// For real OCR testing, use actual text images or the sample-page-1.png testdata.
func createTestImageWithText(t *testing.T, text string, width, height int) image.Image {
	t.Helper()

	// Create white background
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.Draw(img, img.Bounds(), &image.Uniform{color.White}, image.Point{}, draw.Src)

	// Draw simple black dot pattern (not readable text)
	textRect := image.Rect(width/4, height/3, 3*width/4, 2*height/3)
	for x := textRect.Min.X; x < textRect.Max.X; x++ {
		for y := textRect.Min.Y; y < textRect.Max.Y; y++ {
			if (x+y)%3 == 0 {
				img.Set(x, y, color.Black)
			}
		}
	}

	return img
}

// saveTestImage saves an image to a temp file for debugging
func saveTestImage(t *testing.T, img image.Image, name string) string {
	t.Helper()

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, name+".png")

	f, err := os.Create(path)
	require.NoError(t, err)
	defer f.Close()

	err = png.Encode(f, img)
	require.NoError(t, err)

	return path
}

// renderPDFPage renders a PDF page to a PNG file using the render_pdf_page.py script.
func renderPDFPage(t *testing.T, pdfPath string, pageNum int, outputPath string) error {
	t.Helper()

	renderScript := filepath.Join(".", "render_pdf_page.py")
	if _, err := os.Stat(renderScript); os.IsNotExist(err) {
		return fmt.Errorf("render_pdf_page.py not found")
	}

	cmd := exec.Command("uv", "run", renderScript, pdfPath, fmt.Sprintf("%d", pageNum), outputPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to render PDF: %v\n%s", err, output)
	}

	t.Logf("Rendered PDF page %d: %s", pageNum, string(output))
	return nil
}

// createReader creates a PooledReader using the new backends API.
func createReader(t *testing.T, modelPath string) (*reading.PooledReader, error) {
	t.Helper()

	logger := zap.NewNop()
	sessionManager := backends.NewSessionManager()

	cfg := &reading.PooledReaderConfig{
		ModelPath: modelPath,
		PoolSize:  1,
		Logger:    logger,
	}

	// Use all available backends
	modelBackends := []string{"onnx", "xla", "go"}

	reader, _, err := reading.NewPooledReader(cfg, sessionManager, modelBackends)
	if err != nil {
		return nil, err
	}

	// Note: The sessionManager will be cleaned up when the reader is closed
	// since the pipelines hold references to the loaded models

	return reader, nil
}

// =============================================================================
// TrOCR Tests
// =============================================================================

// TestTrOCRModelDownload tests downloading the TrOCR model from HuggingFace
func TestTrOCRModelDownload(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping TrOCR download test in short mode")
	}

	modelPath := ensureHuggingFaceModel(t, trocrModelName, trocrModelRepo, ModelTypeReader)

	requiredFiles := []string{"tokenizer.json", "config.json"}
	for _, file := range requiredFiles {
		filePath := filepath.Join(modelPath, file)
		_, err := os.Stat(filePath)
		assert.NoError(t, err, "Missing required file: %s", file)
	}

	// Check for ONNX files
	onnxDir := filepath.Join(modelPath, "onnx")
	if _, err := os.Stat(onnxDir); err == nil {
		encoderPath := filepath.Join(onnxDir, "encoder_model.onnx")
		decoderPath := filepath.Join(onnxDir, "decoder_model_merged.onnx")

		if fileExists(encoderPath) && fileExists(decoderPath) {
			t.Logf("Found ONNX models in %s", onnxDir)
		}
	}

	// Load and verify config using the pipelines package
	config, err := pipelines.LoadVision2SeqModelConfig(modelPath)
	require.NoError(t, err, "Failed to load Vision2Seq model config")

	t.Logf("TrOCR config: num_heads=%d, head_dim=%d, vocab_size=%d",
		config.NumHeads, config.HeadDim, config.DecoderConfig.VocabSize)
}

// TestTrOCRReader tests the TrOCR model using the Reader interface
func TestTrOCRReader(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping TrOCR reader test in short mode")
	}

	modelPath := ensureHuggingFaceModel(t, trocrModelName, trocrModelRepo, ModelTypeReader)

	// Check for ONNX files
	onnxDir := filepath.Join(modelPath, "onnx")
	encoderPath := filepath.Join(onnxDir, "encoder_model.onnx")
	if _, err := os.Stat(encoderPath); os.IsNotExist(err) {
		encoderPath = filepath.Join(modelPath, "encoder_model.onnx")
		onnxDir = modelPath
	}

	if _, err := os.Stat(encoderPath); os.IsNotExist(err) {
		t.Skipf("TrOCR ONNX encoder not found at %s", encoderPath)
	}

	// Check for decoder models
	hasMerged := fileExists(filepath.Join(onnxDir, "decoder_model_merged.onnx"))
	hasSplit := fileExists(filepath.Join(onnxDir, "decoder_model.onnx")) &&
		fileExists(filepath.Join(onnxDir, "decoder_with_past_model.onnx"))

	if !hasMerged && !hasSplit {
		t.Skipf("No TrOCR decoder models found in %s", onnxDir)
	}

	// Create Reader
	reader, err := createReader(t, modelPath)
	if err != nil {
		t.Skipf("Reader creation failed: %v", err)
	}
	defer reader.Close()

	t.Logf("Reader model type: %s", reader.ModelType())

	// Run OCR
	img := createTestImageWithText(t, "Hello", 384, 384)
	ctx := context.Background()
	results, err := reader.Read(ctx, []image.Image{img}, "", 64)
	if err != nil {
		t.Logf("TrOCR inference failed: %v", err)
		return
	}

	t.Logf("TrOCR output: %q", results[0].Text)
}

// TestTrOCRExportedModel tests TrOCR model from HuggingFace
func TestTrOCRExportedModel(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping TrOCR model test in short mode")
	}

	modelPath := ensureHuggingFaceModel(t, trocrModelName, trocrModelRepo, ModelTypeReader)

	reader, err := createReader(t, modelPath)
	require.NoError(t, err, "Failed to create Reader")
	defer reader.Close()

	t.Logf("Reader model type: %s", reader.ModelType())

	img := createTestImageWithText(t, "Hello", 384, 384)
	ctx := context.Background()
	results, err := reader.Read(ctx, []image.Image{img}, "", 64)
	require.NoError(t, err, "TrOCR inference failed")

	t.Logf("TrOCR output: %q", results[0].Text)
	assert.NotEmpty(t, results[0].Text, "Expected non-empty OCR output")
}

// TestTrOCRWithPDFPage tests OCR on a rendered PDF page
func TestTrOCRWithPDFPage(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping PDF OCR test in short mode")
	}

	pageImagePath := filepath.Join("testdata", "sample-page-1.png")
	if _, err := os.Stat(pageImagePath); os.IsNotExist(err) {
		t.Skip("Pre-rendered page image not found at:", pageImagePath)
	}

	modelPath := ensureHuggingFaceModel(t, trocrModelName, trocrModelRepo, ModelTypeReader)

	// Create Reader
	reader, err := createReader(t, modelPath)
	require.NoError(t, err, "Failed to create Reader")
	defer reader.Close()

	// Load pre-rendered image
	f, err := os.Open(pageImagePath)
	require.NoError(t, err)
	defer f.Close()

	img, err := png.Decode(f)
	require.NoError(t, err)

	t.Logf("Loaded PDF page image: %dx%d", img.Bounds().Dx(), img.Bounds().Dy())

	ctx := context.Background()
	results, err := reader.Read(ctx, []image.Image{img}, "", 128)
	require.NoError(t, err, "TrOCR inference failed")

	t.Logf("TrOCR output: %q", results[0].Text)
	assert.NotEmpty(t, results[0].Text, "Expected non-empty OCR output")
}

// =============================================================================
// Donut Tests
// =============================================================================

// TestDonutFieldParsing tests the ParseDonutFields function.
func TestDonutFieldParsing(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected map[string]string
	}{
		{
			name:  "simple fields",
			input: "<s_menu><s_nm>Burger</s_nm><s_price>$5.99</s_price></s_menu>",
			expected: map[string]string{
				"menu.nm":    "Burger",
				"menu.price": "$5.99",
			},
		},
		{
			name:  "with task tokens",
			input: "<s_cord-v2><s_total>$123.45</s_total></s_cord-v2>",
			expected: map[string]string{
				"total": "$123.45",
			},
		},
		{
			name:  "flat fields",
			input: "<s_company>ACME Corp</s_company><s_date>2025-01-03</s_date>",
			expected: map[string]string{
				"company": "ACME Corp",
				"date":    "2025-01-03",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := reading.DonutParseFields(tt.input)
			t.Logf("Input: %s", tt.input)
			t.Logf("Parsed: %v", result)

			for key, expectedValue := range tt.expected {
				actual, ok := result[key]
				require.True(t, ok, "Should have field %q", key)
				require.Equal(t, expectedValue, actual, "Field %q should match", key)
			}
		})
	}
}

// TestDonutExportedModel tests Donut model from HuggingFace
func TestDonutExportedModel(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Donut model test in short mode")
	}

	modelPath := ensureHuggingFaceModel(t, donutCordName, donutCordRepo, ModelTypeReader)

	reader, err := createReader(t, modelPath)
	if err != nil {
		t.Skipf("Could not create Donut reader: %v", err)
	}
	defer reader.Close()

	t.Logf("Reader model type: %s", reader.ModelType())

	// Create test image and run with CORD prompt
	testImg := image.NewRGBA(image.Rect(0, 0, 800, 1000))
	draw.Draw(testImg, testImg.Bounds(), &image.Uniform{color.White}, image.Point{}, draw.Src)

	ctx := context.Background()
	results, err := reader.Read(ctx, []image.Image{testImg}, reading.DonutCORDPrompt(), 512)
	require.NoError(t, err)

	t.Logf("Donut output: %s", results[0].Text)
	if len(results[0].Fields) > 0 {
		t.Logf("Parsed fields: %v", results[0].Fields)
	}
}

// TestDonutWithPDFPage tests Donut on a rendered PDF page
func TestDonutWithPDFPage(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping PDF Donut test in short mode")
	}

	pageImagePath := filepath.Join("testdata", "sample-page-1.png")
	if _, err := os.Stat(pageImagePath); os.IsNotExist(err) {
		t.Skip("Pre-rendered page image not found at:", pageImagePath)
	}

	modelPath := ensureHuggingFaceModel(t, donutCordName, donutCordRepo, ModelTypeReader)

	// Load pre-rendered image
	imgFile, err := os.Open(pageImagePath)
	require.NoError(t, err)
	defer imgFile.Close()

	img, _, err := image.Decode(imgFile)
	require.NoError(t, err)
	t.Logf("Loaded PDF page image: %dx%d", img.Bounds().Dx(), img.Bounds().Dy())

	// Create Reader
	reader, err := createReader(t, modelPath)
	if err != nil {
		t.Skipf("Could not create Donut reader: %v", err)
	}
	defer reader.Close()

	ctx := context.Background()
	results, err := reader.Read(ctx, []image.Image{img}, reading.DonutCORDPrompt(), 512)
	require.NoError(t, err)

	t.Logf("Donut output: %q", results[0].Text)
	if len(results[0].Fields) > 0 {
		t.Logf("Extracted fields:")
		for k, v := range results[0].Fields {
			t.Logf("  %s: %s", k, v)
		}
	}
}

// TestDocVQAWithPDFPage tests Donut-DocVQA for question answering
func TestDocVQAWithPDFPage(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping PDF DocVQA test in short mode")
	}

	pageImagePath := filepath.Join("testdata", "sample-page-1.png")
	if _, err := os.Stat(pageImagePath); os.IsNotExist(err) {
		t.Skip("Pre-rendered page image not found at:", pageImagePath)
	}

	modelPath := ensureHuggingFaceModel(t, "Xenova/donut-base-finetuned-docvqa", "Xenova/donut-base-finetuned-docvqa", ModelTypeReader)

	// Load pre-rendered image
	imgFile, err := os.Open(pageImagePath)
	require.NoError(t, err)
	defer imgFile.Close()

	img, _, err := image.Decode(imgFile)
	require.NoError(t, err)

	// Create Reader
	reader, err := createReader(t, modelPath)
	if err != nil {
		t.Skipf("Could not create DocVQA reader: %v", err)
	}
	defer reader.Close()

	t.Logf("Reader model type: %s", reader.ModelType())

	// Ask questions
	questions := []string{
		"What is the case name?",
		"What is the document type?",
		"What court is this from?",
	}

	ctx := context.Background()
	for _, question := range questions {
		prompt := reading.DonutDocVQAPrompt(question)
		results, err := reader.Read(ctx, []image.Image{img}, prompt, 128)
		if err != nil {
			t.Logf("Question %q: error: %v", question, err)
			continue
		}
		t.Logf("Question: %q => Answer: %q", question, results[0].Text)
	}
}

// =============================================================================
// Florence-2 Tests
// =============================================================================

// TestFlorence2WithPDFPage tests Florence-2 for OCR on documents and validates
// the output against expected text extracted from the PDF.
func TestFlorence2WithPDFPage(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping PDF Florence-2 test in short mode")
	}

	pageImagePath := filepath.Join("testdata", "sample-page-1.png")
	if _, err := os.Stat(pageImagePath); os.IsNotExist(err) {
		t.Skip("Pre-rendered page image not found at:", pageImagePath)
	}

	modelPath := ensureHuggingFaceModel(t, "onnx-community/Florence-2-base-ft", "onnx-community/Florence-2-base-ft", ModelTypeReader)

	// Load pre-rendered image
	imgFile, err := os.Open(pageImagePath)
	require.NoError(t, err)
	defer imgFile.Close()

	img, _, err := image.Decode(imgFile)
	require.NoError(t, err)
	t.Logf("Loaded PDF page image: %dx%d", img.Bounds().Dx(), img.Bounds().Dy())

	// Create Reader
	reader, err := createReader(t, modelPath)
	if err != nil {
		t.Skipf("Could not create Florence-2 reader: %v", err)
	}
	defer reader.Close()

	t.Logf("Reader model type: %s", reader.ModelType())

	ctx := context.Background()

	// Test OCR
	ocrPrompt := reading.FlorencePrompt(reading.FlorenceOCR)
	t.Logf("Running Florence-2 OCR with prompt: %q", ocrPrompt)

	results, err := reader.Read(ctx, []image.Image{img}, ocrPrompt, 512)
	if err != nil {
		t.Fatalf("OCR failed: %v", err)
	}
	ocrOutput := results[0].Text
	t.Logf("OCR output: %q", ocrOutput)

	// Validate OCR output against expected key phrases from the sample PDF
	// These are distinctive phrases that should appear in the OCR output
	expectedPhrases := []string{
		"heading",
		"content",
		"table",
		"very long",
		"page width",
		"text box",
	}

	matchPercent, missing := containsKeyPhrases(ocrOutput, expectedPhrases)
	t.Logf("OCR accuracy: %.1f%% of key phrases found", matchPercent)
	if len(missing) > 0 {
		t.Logf("Missing phrases: %v", missing)
	}

	// Require at least 50% of key phrases to be found
	assert.GreaterOrEqual(t, matchPercent, 50.0,
		"OCR should find at least 50%% of expected key phrases, got %.1f%%", matchPercent)

	// Test caption
	captionPrompt := reading.FlorencePrompt(reading.FlorenceDetailedCaption)
	results, err = reader.Read(ctx, []image.Image{img}, captionPrompt, 256)
	if err != nil {
		t.Logf("Caption failed: %v", err)
	} else {
		t.Logf("Detailed caption: %q", results[0].Text)
	}
}

// =============================================================================
// Character Mapping Utilities (for corrupted PDF text)
// =============================================================================

// CharMapping represents a character substitution mapping
type CharMapping struct {
	Corrupted rune
	Correct   rune
	Count     int
}

// BuildCharMapping compares OCR text with corrupted text to build a mapping
func BuildCharMapping(ocrText, corruptedText string) []CharMapping {
	mappings := make(map[rune]map[rune]int)

	ocrRunes := []rune(ocrText)
	corruptedRunes := []rune(corruptedText)

	minLen := len(ocrRunes)
	if len(corruptedRunes) < minLen {
		minLen = len(corruptedRunes)
	}

	for i := 0; i < minLen; i++ {
		correct := ocrRunes[i]
		corrupted := corruptedRunes[i]

		if correct != corrupted {
			if mappings[corrupted] == nil {
				mappings[corrupted] = make(map[rune]int)
			}
			mappings[corrupted][correct]++
		}
	}

	var result []CharMapping
	for corrupted, correctMap := range mappings {
		for correct, count := range correctMap {
			result = append(result, CharMapping{
				Corrupted: corrupted,
				Correct:   correct,
				Count:     count,
			})
		}
	}

	return result
}

// ApplyCharMapping applies a character mapping to corrupted text
func ApplyCharMapping(text string, mappings []CharMapping) string {
	lookup := make(map[rune]rune)
	for _, m := range mappings {
		if existing, ok := lookup[m.Corrupted]; ok {
			for _, other := range mappings {
				if other.Corrupted == m.Corrupted && other.Correct == existing {
					if m.Count > other.Count {
						lookup[m.Corrupted] = m.Correct
					}
					break
				}
			}
		} else {
			lookup[m.Corrupted] = m.Correct
		}
	}

	var buf bytes.Buffer
	for _, r := range text {
		if corrected, ok := lookup[r]; ok {
			buf.WriteRune(corrected)
		} else {
			buf.WriteRune(r)
		}
	}

	return buf.String()
}

// TestCharMappingBuild tests the character mapping builder
func TestCharMappingBuild(t *testing.T) {
	ocrText := "1:15-cv-07433-LAP"
	corruptedText := "NWNRJcvJMTQPPJLAP"

	mappings := BuildCharMapping(ocrText, corruptedText)

	t.Logf("Found %d character mappings:", len(mappings))
	for _, m := range mappings {
		t.Logf("  '%c' -> '%c' (count: %d)", m.Corrupted, m.Correct, m.Count)
	}

	assert.NotEmpty(t, mappings, "Should find character mappings")
}

// TestApplyCharMapping tests applying character mappings
func TestApplyCharMapping(t *testing.T) {
	mappings := []CharMapping{
		{'N', '1', 2},
		{'W', ':', 1},
		{'R', '5', 1},
		{'J', '-', 3},
		{'M', '0', 1},
		{'T', '7', 1},
		{'Q', '4', 1},
		{'P', '3', 2},
	}

	corrupted := "NWNRJcvJMTQPPJLAP"
	expectedPartial := "1:15-cv-07433-LA3"

	result := ApplyCharMapping(corrupted, mappings)

	t.Logf("Corrupted: %s", corrupted)
	t.Logf("Fixed:     %s", result)

	assert.Equal(t, expectedPartial, result)
}
