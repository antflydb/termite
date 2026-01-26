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

package modelregistry

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/gomlx/go-huggingface/hub"
)

// HuggingFaceClient pulls ONNX models from HuggingFace Hub
type HuggingFaceClient struct {
	token           string
	progressHandler ProgressHandler
}

// HFClientOption configures the HuggingFace client
type HFClientOption func(*HuggingFaceClient)

// NewHuggingFaceClient creates a new HuggingFace client
func NewHuggingFaceClient(opts ...HFClientOption) *HuggingFaceClient {
	c := &HuggingFaceClient{}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// WithHFToken sets the HuggingFace API token for gated models
func WithHFToken(token string) HFClientOption {
	return func(c *HuggingFaceClient) { c.token = token }
}

// WithHFProgressHandler sets the progress handler for downloads
func WithHFProgressHandler(h ProgressHandler) HFClientOption {
	return func(c *HuggingFaceClient) { c.progressHandler = h }
}

// PullFromHuggingFace downloads ONNX model files from a HuggingFace repo.
// variant can be: "", "fp16", "q4", "q4f16", "quantized"
//
// The model is stored in the owner/model directory structure:
//
//	destDir/modelType/owner/model-name/
//
// A model_manifest.json is generated and saved with the model files.
func (c *HuggingFaceClient) PullFromHuggingFace(
	ctx context.Context,
	repoID string,
	modelType ModelType,
	destDir string,
	variant string,
) error {
	// Parse repo ID to get owner and model name
	ref, err := ParseModelRef(repoID)
	if err != nil {
		return fmt.Errorf("parsing repo ID: %w", err)
	}

	// If no owner in ref, try to extract from repoID directly (e.g., "BAAI/bge-small-en-v1.5")
	if ref.Owner == "" {
		parts := strings.SplitN(repoID, "/", 2)
		if len(parts) == 2 {
			ref.Owner = parts[0]
			ref.Name = parts[1]
		}
	}

	repo := hub.New(repoID)
	if c.token != "" {
		repo = repo.WithAuth(c.token)
	}

	// List all files in repo
	var files []string
	for fileName, err := range repo.IterFileNames() {
		if err != nil {
			return fmt.Errorf("listing files: %w", err)
		}
		files = append(files, fileName)
	}

	// Filter and select files to download based on model type
	var toDownload []string
	if modelType == ModelTypeGenerator || modelType == ModelTypeReader || modelType == ModelTypeTranscriber {
		// For generators, readers (Vision2Seq), and transcribers (Speech2Seq) -
		// all encoder-decoder models that need multiple ONNX files.
		// Auto-detect smallest variant if not specified.
		if variant == "" {
			variant = findSmallestGeneratorVariant(files)
			if variant != "" {
				fmt.Printf("Auto-selected variant: %s\n", variant)
			}
		}
		toDownload = selectGeneratorFiles(files, variant)
	} else {
		toDownload = selectONNXFiles(files, variant)
	}
	if len(toDownload) == 0 {
		return fmt.Errorf("no model files found in %s", repoID)
	}

	// Create destination directory with owner/model structure
	modelDir := filepath.Join(destDir, modelType.DirName(), ref.DirPath())
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return fmt.Errorf("creating directory: %w", err)
	}

	// Download each file
	for _, fileName := range toDownload {
		localPath, err := repo.DownloadFile(fileName)
		if err != nil {
			return fmt.Errorf("downloading %s: %w", fileName, err)
		}

		// Flatten path (e.g., "onnx/model.onnx" -> "model.onnx")
		destName := filepath.Base(fileName)
		destPath := filepath.Join(modelDir, destName)

		// Report progress before copy
		if c.progressHandler != nil {
			c.progressHandler(0, 0, destName)
		}

		// Copy from cache to destination
		if err := copyFile(localPath, destPath); err != nil {
			return fmt.Errorf("copying %s: %w", fileName, err)
		}

		// Report completion
		if c.progressHandler != nil {
			if info, err := os.Stat(destPath); err == nil {
				c.progressHandler(info.Size(), info.Size(), destName)
			}
		}
	}

	// Generate and save local manifest
	if err := c.generateAndSaveManifest(modelDir, repoID, ref, modelType); err != nil {
		// Log warning but don't fail the download
		fmt.Printf("Warning: failed to generate manifest: %v\n", err)
	}

	return nil
}

// generateAndSaveManifest creates a manifest for downloaded model files
func (c *HuggingFaceClient) generateAndSaveManifest(
	modelDir string,
	repoID string,
	ref ModelRef,
	modelType ModelType,
) error {
	// Scan downloaded files
	files, err := ScanModelFiles(modelDir)
	if err != nil {
		return fmt.Errorf("scanning files: %w", err)
	}

	if len(files) == 0 {
		return fmt.Errorf("no files found in %s", modelDir)
	}

	// Create manifest
	// Note: Backends is left empty (nil) meaning all backends are supported.
	// Models that require specific backends (e.g., ONNX-only due to unsupported ops)
	// should specify this in the registry manifest or hardcoded list.
	manifest := &ModelManifest{
		SchemaVersion: CurrentSchemaVersion,
		Name:          ref.Name,
		Source:        repoID,
		Owner:         ref.Owner,
		Type:          modelType,
		Files:         files,
		Provenance: &ModelProvenance{
			DownloadedFrom: "huggingface",
			DownloadedAt:   time.Now(),
		},
	}

	// Discover variants from downloaded files
	manifest.Variants = discoverVariantsFromFiles(files)

	// Detect capabilities for multimodal models
	for _, f := range files {
		if f.Name == "visual_model.onnx" || f.Name == "text_model.onnx" {
			manifest.Capabilities = append(manifest.Capabilities, CapabilityMultimodal)
			break
		}
	}

	// Save manifest
	manifestPath := filepath.Join(modelDir, ManifestFilename)
	return manifest.SaveTo(manifestPath)
}

// selectGeneratorFiles selects files needed for onnxruntime-genai models.
// This includes genai_config.json, all ONNX files, and tokenizer files.
// If variant is specified, only files from that subdirectory are selected.
// If variant is empty, it auto-selects the smallest cpu variant.
func selectGeneratorFiles(files []string, variant string) []string {
	// If no variant specified, auto-select the smallest cpu variant
	if variant == "" {
		variant = findSmallestGeneratorVariant(files)
	}

	var result []string

	// Files to include by exact basename match
	includeExact := map[string]bool{
		"genai_config.json":       true,
		"tokenizer.json":          true,
		"tokenizer.model":         true,
		"tokenizer_config.json":   true,
		"config.json":             true,
		"special_tokens_map.json": true,
		"added_tokens.json":       true,
		"generation_config.json":  true,
	}

	// Files to include by suffix
	includeSuffixes := []string{
		".onnx",
		".onnx.data",            // External data files
		".onnx_data",            // Alternative naming
		".txt",                  // Vocab files like vocab.txt, merges.txt
		".spm",                  // SentencePiece model files
		".tiktoken",             // Tiktoken encoding files
		".jinja",                // Chat template files (e.g., chat_template.jinja)
		"processor_config.json", // For multimodal models
	}

	for _, f := range files {
		// Filter by variant subdirectory if specified
		if variant != "" && !strings.HasPrefix(f, variant+"/") && f != variant {
			continue
		}

		base := filepath.Base(f)

		// Check exact matches
		if includeExact[base] {
			result = append(result, f)
			continue
		}

		// Check suffix matches
		for _, suffix := range includeSuffixes {
			if strings.HasSuffix(base, suffix) {
				result = append(result, f)
				break
			}
		}
	}

	return result
}

// findSmallestGeneratorVariant finds the smallest generator variant path.
// It looks for cpu-int4 variants first, then falls back to any cpu variant,
// then any variant with genai_config.json.
// It prefers smaller model sizes (e.g., 1b over 4b over 12b).
func findSmallestGeneratorVariant(files []string) string {
	// Find all directories containing genai_config.json
	variantDirs := make(map[string]bool)
	for _, f := range files {
		if filepath.Base(f) == "genai_config.json" {
			dir := filepath.Dir(f)
			if dir != "." {
				variantDirs[dir] = true
			}
		}
	}

	if len(variantDirs) == 0 {
		return "" // No variants found, download everything
	}

	// Priority order for selecting variants:
	// 1. cpu-int4 variants (smallest)
	// 2. Any cpu variant
	// 3. Any variant

	var cpuInt4Variants []string
	var cpuVariants []string
	var allVariants []string

	for dir := range variantDirs {
		allVariants = append(allVariants, dir)
		lowerDir := strings.ToLower(dir)
		if strings.Contains(lowerDir, "cpu") {
			cpuVariants = append(cpuVariants, dir)
			if strings.Contains(lowerDir, "int4") {
				cpuInt4Variants = append(cpuInt4Variants, dir)
			}
		}
	}

	// Sort by model size (prefer smaller: 1b < 4b < 12b < 27b)
	sortByModelSize := func(variants []string) {
		slices.SortFunc(variants, func(a, b string) int {
			sizeA := extractModelSize(a)
			sizeB := extractModelSize(b)
			if sizeA != sizeB {
				return sizeA - sizeB
			}
			// Fall back to alphabetical for same size
			return strings.Compare(a, b)
		})
	}

	sortByModelSize(cpuInt4Variants)
	sortByModelSize(cpuVariants)
	sortByModelSize(allVariants)

	// Return the first match in priority order
	if len(cpuInt4Variants) > 0 {
		return cpuInt4Variants[0]
	}
	if len(cpuVariants) > 0 {
		return cpuVariants[0]
	}
	return allVariants[0]
}

// extractModelSize extracts the numeric model size from a path like "gemma-3-4b-it/..."
// Returns a large number if no size found so those sort last.
func extractModelSize(path string) int {
	// Look for patterns like "1b", "4b", "12b", "27b" in the path
	lowerPath := strings.ToLower(path)

	// Common size patterns
	sizePatterns := []struct {
		pattern string
		size    int
	}{
		{"-1b", 1},
		{"-2b", 2},
		{"-3b", 3},
		{"-4b", 4},
		{"-7b", 7},
		{"-8b", 8},
		{"-12b", 12},
		{"-13b", 13},
		{"-27b", 27},
		{"-70b", 70},
		{"1b-", 1},
		{"2b-", 2},
		{"3b-", 3},
		{"4b-", 4},
		{"7b-", 7},
		{"8b-", 8},
		{"12b-", 12},
		{"13b-", 13},
		{"27b-", 27},
		{"70b-", 70},
	}

	for _, sp := range sizePatterns {
		if strings.Contains(lowerPath, sp.pattern) {
			return sp.size
		}
	}

	return 999 // No size found, sort last
}

// selectONNXFiles filters files based on variant preference.
// It returns tokenizer files plus the ONNX model file(s) matching the variant.
func selectONNXFiles(files []string, variant string) []string {
	var result []string

	// Always include tokenizer/config files from anywhere in the repo
	tokenizerFiles := []string{"tokenizer.json", "tokenizer.model", "tokenizer_config.json", "config.json", "special_tokens_map.json"}
	for _, tf := range tokenizerFiles {
		for _, f := range files {
			if filepath.Base(f) == tf {
				result = append(result, f)
				break
			}
		}
	}

	// Determine ONNX file pattern based on variant
	var onnxBase string
	switch variant {
	case "fp16":
		onnxBase = "model_fp16"
	case "q4":
		onnxBase = "model_q4"
	case "q4f16":
		onnxBase = "model_q4f16"
	case "quantized":
		onnxBase = "model_quantized"
	default:
		onnxBase = "model"
	}

	// Find matching ONNX files (model.onnx + model.onnx_data)
	for _, f := range files {
		base := filepath.Base(f)
		// Match exact model file or its data file
		if base == onnxBase+".onnx" || base == onnxBase+".onnx_data" {
			result = append(result, f)
		}
	}

	return result
}

// copyFile copies a file from src to dst
func copyFile(src, dst string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("opening source: %w", err)
	}
	defer func() { _ = srcFile.Close() }()

	dstFile, err := os.Create(dst)
	if err != nil {
		return fmt.Errorf("creating destination: %w", err)
	}

	if _, err := io.Copy(dstFile, srcFile); err != nil {
		_ = dstFile.Close()
		return fmt.Errorf("copying: %w", err)
	}

	return dstFile.Close()
}

// ValidVariants returns the list of valid ONNX variant names
func ValidVariants() []string {
	return []string{"", "fp16", "q4", "q4f16", "quantized"}
}

// IsValidVariant checks if a variant name is valid
func IsValidVariant(variant string) bool {
	return slices.Contains(ValidVariants(), variant)
}

// VariantDescription returns a human-readable description of a variant
func VariantDescription(variant string) string {
	switch variant {
	case "":
		return "full precision (default)"
	case "fp16":
		return "half precision (FP16)"
	case "q4":
		return "4-bit quantized"
	case "q4f16":
		return "4-bit quantized with FP16"
	case "quantized":
		return "INT8 quantized"
	default:
		return "unknown"
	}
}

// ListRepoFiles returns all files in a HuggingFace repo (useful for inspection)
func (c *HuggingFaceClient) ListRepoFiles(ctx context.Context, repoID string) ([]string, error) {
	repo := hub.New(repoID)
	if c.token != "" {
		repo = repo.WithAuth(c.token)
	}

	var files []string
	for fileName, err := range repo.IterFileNames() {
		if err != nil {
			return nil, fmt.Errorf("listing files: %w", err)
		}
		files = append(files, fileName)
	}
	return files, nil
}

// DetectAvailableVariants returns which ONNX variants are available in a repo
func (c *HuggingFaceClient) DetectAvailableVariants(ctx context.Context, repoID string) ([]string, error) {
	files, err := c.ListRepoFiles(ctx, repoID)
	if err != nil {
		return nil, err
	}

	variants := []string{}
	variantPatterns := map[string]string{
		"":          "model.onnx",
		"fp16":      "model_fp16.onnx",
		"q4":        "model_q4.onnx",
		"q4f16":     "model_q4f16.onnx",
		"quantized": "model_quantized.onnx",
	}

	for variant, pattern := range variantPatterns {
		for _, f := range files {
			if filepath.Base(f) == pattern {
				if variant == "" {
					variants = append(variants, "default")
				} else {
					variants = append(variants, variant)
				}
				break
			}
		}
	}

	return variants, nil
}

// ParseHuggingFaceRef parses a model reference like "hf:owner/repo" and returns the repo ID
func ParseHuggingFaceRef(ref string) (repoID string, isHF bool) {
	if after, ok := strings.CutPrefix(ref, "hf:"); ok {
		return after, true
	}
	return "", false
}

// DetectGeneratorVariants returns available onnxruntime-genai variants in a repo.
// These are subdirectories containing genai_config.json files.
func (c *HuggingFaceClient) DetectGeneratorVariants(ctx context.Context, repoID string) ([]string, error) {
	files, err := c.ListRepoFiles(ctx, repoID)
	if err != nil {
		return nil, err
	}

	// Find all directories containing genai_config.json
	variantDirs := make(map[string]bool)
	for _, f := range files {
		if filepath.Base(f) == "genai_config.json" {
			dir := filepath.Dir(f)
			if dir != "." {
				variantDirs[dir] = true
			}
		}
	}

	variants := make([]string, 0, len(variantDirs))
	for dir := range variantDirs {
		variants = append(variants, dir)
	}
	slices.Sort(variants)
	return variants, nil
}

// DetectModelType attempts to detect the model type from repo contents.
// It checks for genai_config.json (generator), encoder/decoder (rewriter),
// visual/text models (multimodal embedder), or regular model.onnx.
func (c *HuggingFaceClient) DetectModelType(ctx context.Context, repoID string) (ModelType, error) {
	files, err := c.ListRepoFiles(ctx, repoID)
	if err != nil {
		return "", fmt.Errorf("listing files: %w", err)
	}

	hasGenaiConfig := false
	hasEncoder := false
	hasDecoder := false
	hasVisual := false
	hasText := false
	hasModelOnnx := false

	for _, f := range files {
		base := filepath.Base(f)
		switch base {
		case "genai_config.json":
			hasGenaiConfig = true
		case "encoder.onnx":
			hasEncoder = true
		case "decoder.onnx":
			hasDecoder = true
		case "visual_model.onnx":
			hasVisual = true
		case "text_model.onnx":
			hasText = true
		case "model.onnx":
			hasModelOnnx = true
		}
	}

	// Check for generator (onnxruntime-genai format)
	if hasGenaiConfig {
		return ModelTypeGenerator, nil
	}

	// Check for seq2seq (rewriter)
	if hasEncoder && hasDecoder {
		return ModelTypeRewriter, nil
	}

	// Check for multimodal (could be embedder with CLIP)
	if hasVisual && hasText {
		return ModelTypeEmbedder, nil
	}

	// Check for standard model
	if hasModelOnnx {
		// Could be embedder, chunker, or reranker - can't tell without more context
		// Default to embedder as it's the most common
		return "", fmt.Errorf("cannot auto-detect model type (found model.onnx but could be embedder, chunker, or reranker)")
	}

	return "", fmt.Errorf("no recognizable model files found in repository")
}
