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

// Package cli provides shared CLI functions for termite model management.
// These functions are used by both the standalone termite binary and the antfly termite subcommand.
package cli

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"text/tabwriter"

	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
)

// PullOptions contains options for pulling models from the registry
type PullOptions struct {
	RegistryURL string
	ModelsDir   string
	Variants    []string // Variant IDs to download (e.g., ["f16", "i8"])
}

// HuggingFaceOptions contains options for pulling from HuggingFace
type HuggingFaceOptions struct {
	ModelsDir string
	ModelType string
	HFToken   string
	Variant   string
}

// ListOptions contains options for listing models
type ListOptions struct {
	RegistryURL string
	ModelsDir   string
	TypeFilter  string
	BinaryName  string // Used for help messages (e.g., "termite" or "antfly termite")
}

// knownVariants are the recognized model variant suffixes
var knownVariants = []string{"f32", "f16", "bf16", "i8", "i8-st", "i4"}

// parseModelRef parses a model reference like "bge-small-en-v1.5-i8" into
// name ("bge-small-en-v1.5") and variant ("i8"). If no known variant suffix
// is found, returns the original ref with empty variant.
func parseModelRef(ref string) (name, variant string) {
	for _, v := range knownVariants {
		suffix := "-" + v
		if before, ok := strings.CutSuffix(ref, suffix); ok {
			return before, v
		}
	}
	return ref, ""
}

// PullFromRegistry pulls a model from the Antfly model registry.
// modelRef can be "name" or "name-variant" (e.g., "bge-small-en-v1.5-i8").
func PullFromRegistry(modelRef string, opts PullOptions) error {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	// Parse model reference for inline variant suffix
	modelName, inlineVariant := parseModelRef(modelRef)
	variants := opts.Variants
	if inlineVariant != "" {
		// Add inline variant if not already in list
		found := slices.Contains(variants, inlineVariant)
		if !found {
			variants = append(variants, inlineVariant)
		}
	}

	client := modelregistry.NewClient(
		modelregistry.WithBaseURL(opts.RegistryURL),
		modelregistry.WithProgressHandler(PrintProgress),
	)

	fmt.Printf("Fetching manifest for %s...\n", modelName)
	manifest, err := client.FetchManifest(ctx, modelName)
	if err != nil {
		return fmt.Errorf("failed to fetch manifest: %w", err)
	}

	fmt.Printf("Model: %s\n", manifest.Name)
	fmt.Printf("Type:  %s\n", manifest.Type)
	if manifest.Description != "" {
		fmt.Printf("Description: %s\n", manifest.Description)
	}

	// Default to f32 if no variants specified (matches PullModel behavior)
	effectiveVariants := variants
	if len(effectiveVariants) == 0 {
		effectiveVariants = []string{modelregistry.VariantF32}
	}

	// Calculate size of what will actually be downloaded
	var totalSize int64
	requestedVariants := make(map[string]bool)
	for _, v := range effectiveVariants {
		requestedVariants[v] = true
	}

	// Count supporting files (non-ONNX) and f32 ONNX files if requested
	for _, f := range manifest.Files {
		if strings.HasSuffix(f.Name, ".onnx") {
			// Count all ONNX files in base manifest if f32 is requested
			// This supports both single-model (model.onnx) and multi-model (visual_model.onnx, text_model.onnx)
			if requestedVariants[modelregistry.VariantF32] {
				totalSize += f.Size
			}
		} else {
			// Always count supporting files
			totalSize += f.Size
		}
	}

	// Count requested variant files (non-f32)
	for _, variantID := range effectiveVariants {
		if variantID == modelregistry.VariantF32 {
			continue // Already counted above
		}
		if variantEntry, ok := manifest.Variants[variantID]; ok {
			// Sum sizes of all files in the variant (supports both single and multi-model variants)
			for _, variantFile := range variantEntry.Files {
				totalSize += variantFile.Size
			}
		}
	}

	fmt.Printf("Variants: %v\n", effectiveVariants)
	fmt.Printf("Total size: %s\n", FormatBytes(totalSize))
	fmt.Println()

	fmt.Println("Downloading files...")
	if err := client.PullModel(ctx, manifest, opts.ModelsDir, variants); err != nil {
		return fmt.Errorf("failed to pull model: %w", err)
	}

	// Build destination path with owner if present
	destDir := filepath.Join(opts.ModelsDir, manifest.Type.DirName(), manifest.FullName())
	fmt.Printf("\n✓ Model pulled successfully to %s\n", destDir)
	return nil
}

// PullFromHuggingFace pulls a model from HuggingFace
func PullFromHuggingFace(repoID string, opts HuggingFaceOptions) error {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	// Parse repoID to get owner/model
	ref, err := modelregistry.ParseModelRef(repoID)
	if err != nil {
		return fmt.Errorf("invalid model reference: %w", err)
	}

	// Model type can be auto-detected for generators, but required for others
	var modelType modelregistry.ModelType
	if opts.ModelType != "" {
		modelType, err = modelregistry.ParseModelType(opts.ModelType)
		if err != nil {
			return err
		}
	}

	if opts.Variant != "" && !modelregistry.IsValidVariant(opts.Variant) {
		return fmt.Errorf("invalid variant %q, valid options: fp16, q4, q4f16, quantized", opts.Variant)
	}

	hfToken := opts.HFToken
	if hfToken == "" {
		hfToken = os.Getenv("HF_TOKEN")
	}

	client := modelregistry.NewHuggingFaceClient(
		modelregistry.WithHFToken(hfToken),
		modelregistry.WithHFProgressHandler(PrintProgress),
	)

	fmt.Printf("Pulling from HuggingFace: %s\n", repoID)

	// Auto-detect model type if not specified
	if modelType == "" {
		fmt.Println("Detecting model type...")
		detected, err := client.DetectModelType(ctx, repoID)
		if err != nil {
			return fmt.Errorf("failed to detect model type: %w\nUse --type flag to specify manually (embedder, chunker, reranker, generator, recognizer, rewriter)", err)
		}
		modelType = detected
		fmt.Printf("Detected type: %s\n", modelType)
	} else {
		fmt.Printf("Type: %s\n", modelType)
	}

	if opts.Variant != "" {
		fmt.Printf("Variant: %s (%s)\n", opts.Variant, modelregistry.VariantDescription(opts.Variant))
	} else {
		fmt.Printf("Variant: %s\n", modelregistry.VariantDescription(""))
	}
	fmt.Println()
	fmt.Println("Downloading files...")

	if err := client.PullFromHuggingFace(ctx, repoID, modelType, opts.ModelsDir, opts.Variant); err != nil {
		return fmt.Errorf("failed to pull model: %w", err)
	}

	// Destination uses owner/model structure
	destDir := filepath.Join(opts.ModelsDir, modelType.DirName(), ref.DirPath())
	fmt.Printf("\n✓ Model pulled successfully to %s\n", destDir)
	return nil
}

// ListRemoteModels lists models available in the remote registry
func ListRemoteModels(opts ListOptions) error {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	client := modelregistry.NewClient(
		modelregistry.WithBaseURL(opts.RegistryURL),
	)

	fmt.Printf("Fetching model list from %s...\n\n", opts.RegistryURL)

	index, err := client.FetchIndex(ctx)
	if err != nil {
		return fmt.Errorf("failed to fetch registry index: %w", err)
	}

	if len(index.Models) == 0 {
		fmt.Println("No models available in registry")
		return nil
	}

	var filteredType modelregistry.ModelType
	if opts.TypeFilter != "" {
		filteredType, err = modelregistry.ParseModelType(opts.TypeFilter)
		if err != nil {
			return err
		}
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	_, _ = fmt.Fprintln(w, "NAME\tTYPE\tSIZE\tVARIANTS\tDESCRIPTION")

	for _, model := range index.Models {
		if filteredType != "" && model.Type != filteredType {
			continue
		}

		variantsStr := ""
		if len(model.Variants) > 0 {
			variantsStr = strings.Join(model.Variants, ",")
		}

		desc := model.Description
		if len(desc) > 50 {
			desc = desc[:47] + "..."
		}

		_, _ = fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n",
			model.Name,
			model.Type,
			FormatBytes(model.Size),
			variantsStr,
			desc,
		)
	}
	return w.Flush()
}

// ListLocalModels lists locally installed models
func ListLocalModels(opts ListOptions) error {
	fmt.Printf("Local models in %s:\n\n", opts.ModelsDir)

	modelTypes := []modelregistry.ModelType{
		modelregistry.ModelTypeEmbedder,
		modelregistry.ModelTypeChunker,
		modelregistry.ModelTypeReranker,
		modelregistry.ModelTypeGenerator,
		modelregistry.ModelTypeRecognizer,
		modelregistry.ModelTypeRewriter,
	}

	var filteredType modelregistry.ModelType
	if opts.TypeFilter != "" {
		var err error
		filteredType, err = modelregistry.ParseModelType(opts.TypeFilter)
		if err != nil {
			return err
		}
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	_, _ = fmt.Fprintln(w, "NAME\tTYPE\tSIZE\tVARIANTS\tSOURCE")

	totalModels := 0

	for _, modelType := range modelTypes {
		if filteredType != "" && modelType != filteredType {
			continue
		}

		typeDir := filepath.Join(opts.ModelsDir, modelType.DirName())
		entries, err := os.ReadDir(typeDir)
		if err != nil {
			continue
		}

		// Process entries - could be owner directories or legacy model directories
		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}

			entryPath := filepath.Join(typeDir, entry.Name())

			// Check if this is a model directory (has model files)
			if isModelDir(entryPath) {
				// Legacy flat structure: models/embedders/model-name/
				displayModel(w, entry.Name(), "", entryPath, modelType, &totalModels)
			} else {
				// New owner structure: models/embedders/owner/model-name/
				ownerDir := entryPath
				ownerName := entry.Name()
				subEntries, err := os.ReadDir(ownerDir)
				if err != nil {
					continue
				}
				for _, subEntry := range subEntries {
					if !subEntry.IsDir() {
						continue
					}
					modelDir := filepath.Join(ownerDir, subEntry.Name())
					if isModelDir(modelDir) {
						displayModel(w, subEntry.Name(), ownerName, modelDir, modelType, &totalModels)
					}
				}
			}
		}
	}
	if err := w.Flush(); err != nil {
		return err
	}

	if totalModels == 0 {
		binaryName := opts.BinaryName
		if binaryName == "" {
			binaryName = "termite"
		}
		fmt.Println("No models found locally.")
		fmt.Printf("\nUse '%s pull <model-name>' to download models.\n", binaryName)
		fmt.Printf("Use '%s list --remote' to see available models.\n", binaryName)
	}

	return nil
}

// isModelDir checks if a directory contains model files
func isModelDir(dir string) bool {
	// Check for standard model
	if _, err := os.Stat(filepath.Join(dir, "model.onnx")); err == nil {
		return true
	}
	// Check for generator model
	if _, err := os.Stat(filepath.Join(dir, "genai_config.json")); err == nil {
		return true
	}
	// Check for multimodal model
	if _, err := os.Stat(filepath.Join(dir, "visual_model.onnx")); err == nil {
		if _, err := os.Stat(filepath.Join(dir, "text_model.onnx")); err == nil {
			return true
		}
	}
	// Check for variant files
	for _, filename := range modelregistry.VariantFilenames {
		if _, err := os.Stat(filepath.Join(dir, filename)); err == nil {
			return true
		}
	}
	// Check for manifest
	if _, err := os.Stat(filepath.Join(dir, "model_manifest.json")); err == nil {
		return true
	}
	return false
}

// displayModel outputs a model row to the table writer
func displayModel(w *tabwriter.Writer, modelName, owner, modelDir string, modelType modelregistry.ModelType, totalModels *int) {
	standardPath := filepath.Join(modelDir, "model.onnx")
	genaiConfigPath := filepath.Join(modelDir, "genai_config.json")
	manifestPath := filepath.Join(modelDir, "model_manifest.json")

	hasStandard := false
	hasMultimodal := false
	hasGenerator := false
	var totalSize int64
	var variants []string
	var capabilities []string
	source := ""

	// Try to load manifest for source info
	if manifest, err := modelregistry.LoadManifestFromFile(manifestPath); err == nil {
		source = manifest.Source
	}

	// Check for standard model
	if info, err := os.Stat(standardPath); err == nil {
		hasStandard = true
		totalSize += info.Size()
	}

	// Check for generator model (genai_config.json)
	if _, err := os.Stat(genaiConfigPath); err == nil {
		hasGenerator = true
		capabilities = append(capabilities, "genai")
	}

	// Check for multimodal (CLIP-style) model files
	visualPath := filepath.Join(modelDir, "visual_model.onnx")
	textPath := filepath.Join(modelDir, "text_model.onnx")
	if visualInfo, err := os.Stat(visualPath); err == nil {
		if textInfo, err := os.Stat(textPath); err == nil {
			hasMultimodal = true
			capabilities = append(capabilities, "multimodal")
			totalSize += visualInfo.Size()
			totalSize += textInfo.Size()
		}
	}

	// Check for variant files
	for variantID, filename := range modelregistry.VariantFilenames {
		variantPath := filepath.Join(modelDir, filename)
		if info, err := os.Stat(variantPath); err == nil {
			variants = append(variants, variantID)
			totalSize += info.Size()
		}
	}

	if !hasStandard && !hasMultimodal && !hasGenerator && len(variants) == 0 {
		return
	}

	// Add size of other files (tokenizer, config, etc.)
	files, _ := os.ReadDir(modelDir)
	for _, f := range files {
		if f.IsDir() {
			continue
		}
		name := f.Name()
		// Skip ONNX files (already counted)
		if strings.HasSuffix(name, ".onnx") {
			continue
		}
		if info, err := f.Info(); err == nil {
			totalSize += info.Size()
		}
	}

	variantsStr := ""
	if len(variants) > 0 {
		variantsStr = strings.Join(variants, ",")
	}

	// Add capability info to display name for multimodal models
	displayType := string(modelType)
	if len(capabilities) > 0 {
		displayType = displayType + " [" + strings.Join(capabilities, ",") + "]"
	}

	// Format display name with owner if present
	displayName := modelName
	if owner != "" {
		displayName = owner + "/" + modelName
	}

	_, _ = fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n",
		displayName,
		displayType,
		FormatBytes(totalSize),
		variantsStr,
		source,
	)
	*totalModels++
}

// FormatBytes formats bytes as human-readable string
func FormatBytes(bytes int64) string {
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
	)

	switch {
	case bytes >= GB:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(GB))
	case bytes >= MB:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(MB))
	case bytes >= KB:
		return fmt.Sprintf("%.1f KB", float64(bytes)/float64(KB))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

// PrintProgress prints download progress to stdout
func PrintProgress(downloaded, total int64, filename string) {
	if total <= 0 {
		fmt.Printf("\r  %s: %s", filename, FormatBytes(downloaded))
		return
	}

	percent := float64(downloaded) / float64(total) * 100
	barWidth := 30
	filled := int(float64(barWidth) * float64(downloaded) / float64(total))

	bar := strings.Repeat("=", filled) + strings.Repeat("-", barWidth-filled)
	fmt.Printf("\r  %s: [%s] %.1f%% (%s/%s)",
		filename, bar, percent, FormatBytes(downloaded), FormatBytes(total))

	if downloaded >= total {
		fmt.Println()
	}
}
