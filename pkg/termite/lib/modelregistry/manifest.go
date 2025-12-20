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

// Package modelregistry provides functionality for pulling ONNX models from
// a remote registry (like Cloudflare R2) in an Ollama-style fashion.
package modelregistry

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ModelType represents the type of model (embedder, chunker, reranker)
type ModelType string

const (
	ModelTypeEmbedder ModelType = "embedder"
	ModelTypeChunker  ModelType = "chunker"
	ModelTypeReranker ModelType = "reranker"
)

// Model capabilities
const (
	// CapabilityMultimodal indicates the model can embed both images and text
	// (e.g., CLIP models with visual_model.onnx + text_model.onnx)
	CapabilityMultimodal = "multimodal"
)

// ParseModelType parses a string into a ModelType
func ParseModelType(s string) (ModelType, error) {
	switch strings.ToLower(s) {
	case "embedder", "embedders":
		return ModelTypeEmbedder, nil
	case "chunker", "chunkers":
		return ModelTypeChunker, nil
	case "reranker", "rerankers":
		return ModelTypeReranker, nil
	default:
		return "", fmt.Errorf("unknown model type: %s (valid: embedder, chunker, reranker)", s)
	}
}

// String returns the string representation of the model type
func (t ModelType) String() string {
	return string(t)
}

// DirName returns the directory name for this model type (plural form)
func (t ModelType) DirName() string {
	switch t {
	case ModelTypeEmbedder:
		return "embedders"
	case ModelTypeChunker:
		return "chunkers"
	case ModelTypeReranker:
		return "rerankers"
	default:
		return string(t) + "s"
	}
}

// ModelFile represents a single file in the model manifest
type ModelFile struct {
	// Name is the filename (e.g., "model.onnx", "tokenizer.json")
	Name string `json:"name"`
	// Digest is the SHA256 hash of the file (e.g., "sha256:abc123...")
	Digest string `json:"digest"`
	// Size is the file size in bytes
	Size int64 `json:"size"`
}

// Variant identifiers for quantized/precision model variants
const (
	// VariantF32 is the default FP32 model (model.onnx)
	VariantF32 = "f32"
	// VariantF16 is FP16 half precision
	VariantF16 = "f16"
	// VariantBF16 is BFloat16 precision
	VariantBF16 = "bf16"
	// VariantI8 is INT8 dynamic quantization
	VariantI8 = "i8"
	// VariantI8Static is INT8 static quantization with calibration
	VariantI8Static = "i8-st"
	// VariantI4 is INT4 quantization
	VariantI4 = "i4"
)

// VariantFilenames maps variant identifiers to their ONNX filenames
var VariantFilenames = map[string]string{
	VariantF32:      "model.onnx",
	VariantF16:      "model_f16.onnx",
	VariantBF16:     "model_bf16.onnx",
	VariantI8:       "model_i8.onnx",
	VariantI8Static: "model_i8-st.onnx",
	VariantI4:       "model_i4.onnx",
}

// FilenameToVariant maps ONNX filenames back to variant identifiers
var FilenameToVariant = map[string]string{
	"model.onnx":       VariantF32,
	"model_f16.onnx":   VariantF16,
	"model_bf16.onnx":  VariantBF16,
	"model_i8.onnx":    VariantI8,
	"model_i8-st.onnx": VariantI8Static,
	"model_i4.onnx":    VariantI4,
}

// ModelManifest describes an ONNX model and its files
type ModelManifest struct {
	// SchemaVersion is the manifest format version
	SchemaVersion int `json:"schemaVersion"`
	// Name is the model identifier (e.g., "bge-small-en-v1.5")
	Name string `json:"name"`
	// Type is the model type (embedder, chunker, reranker)
	Type ModelType `json:"type"`
	// Description is a human-readable description
	Description string `json:"description,omitempty"`
	// Capabilities lists special capabilities of the model.
	// Valid values: "multimodal" (for CLIP-style models that embed images and text)
	Capabilities []string `json:"capabilities,omitempty"`
	// Files lists all required files for the model (includes model.onnx)
	Files []ModelFile `json:"files"`
	// Variants maps variant identifiers to their model files.
	// For single-model types (embedder, chunker, reranker), this is a single file.
	// For multimodal embedders (CLIP), this can be an array of files (visual + text).
	// Use VariantFiles() to get the slice of files for a variant.
	Variants map[string]VariantEntry `json:"variants,omitempty"`
	// Backends lists supported inference backends for this model.
	// Valid values: "onnx", "xla", "go"
	// If empty, all backends are supported (default).
	Backends []string `json:"backends,omitempty"`
}

// VariantEntry can be either a single ModelFile or an array of ModelFiles.
// This supports both single-model variants and multi-model variants (like CLIP).
type VariantEntry struct {
	Files []ModelFile
}

// UnmarshalJSON handles both single file and array of files
func (v *VariantEntry) UnmarshalJSON(data []byte) error {
	// Try as array first
	var files []ModelFile
	if err := json.Unmarshal(data, &files); err == nil {
		v.Files = files
		return nil
	}

	// Try as single file
	var file ModelFile
	if err := json.Unmarshal(data, &file); err != nil {
		return err
	}
	v.Files = []ModelFile{file}
	return nil
}

// MarshalJSON serializes the variant entry
func (v VariantEntry) MarshalJSON() ([]byte, error) {
	if len(v.Files) == 1 {
		return json.Marshal(v.Files[0])
	}
	return json.Marshal(v.Files)
}

// SupportsBackend returns true if the model supports the given backend.
// If no backends are specified, all backends are supported.
func (m *ModelManifest) SupportsBackend(backend string) bool {
	if len(m.Backends) == 0 {
		return true // All backends supported by default
	}
	for _, b := range m.Backends {
		if b == backend {
			return true
		}
	}
	return false
}

// HasCapability returns true if the model has the specified capability.
func (m *ModelManifest) HasCapability(capability string) bool {
	for _, c := range m.Capabilities {
		if c == capability {
			return true
		}
	}
	return false
}

// IsMultimodal returns true if the model has the multimodal capability.
func (m *ModelManifest) IsMultimodal() bool {
	return m.HasCapability(CapabilityMultimodal)
}

// Validate checks that the manifest is well-formed
func (m *ModelManifest) Validate() error {
	if m.SchemaVersion != 1 {
		return fmt.Errorf("unsupported schema version: %d (expected 1)", m.SchemaVersion)
	}
	if m.Name == "" {
		return fmt.Errorf("manifest missing required field: name")
	}
	if m.Type == "" {
		return fmt.Errorf("manifest missing required field: type")
	}
	if _, err := ParseModelType(string(m.Type)); err != nil {
		return fmt.Errorf("invalid model type: %s", m.Type)
	}
	if len(m.Files) == 0 {
		return fmt.Errorf("manifest must have at least one file")
	}

	// Validate file entries
	hasModelOnnx := false
	hasVisualOnnx := false
	hasTextOnnx := false
	for _, f := range m.Files {
		switch f.Name {
		case "model.onnx":
			hasModelOnnx = true
		case "visual_model.onnx":
			hasVisualOnnx = true
		case "text_model.onnx":
			hasTextOnnx = true
		}
		if f.Name == "" {
			return fmt.Errorf("file entry missing name")
		}
		if f.Digest == "" {
			return fmt.Errorf("file %s missing digest", f.Name)
		}
		if !strings.HasPrefix(f.Digest, "sha256:") {
			return fmt.Errorf("file %s has invalid digest format (expected sha256:...)", f.Name)
		}
	}

	// Check for required ONNX files based on capability
	if m.IsMultimodal() {
		// Multimodal embedders (CLIP) require visual_model.onnx + text_model.onnx
		if !hasVisualOnnx || !hasTextOnnx {
			return fmt.Errorf("multimodal embedder must include visual_model.onnx and text_model.onnx")
		}
		// Multimodal models only support ONNX runtime
		if len(m.Backends) > 0 && !m.SupportsBackend("onnx") {
			return fmt.Errorf("multimodal embedders only support ONNX backend")
		}
	} else {
		// Standard models require model.onnx
		if !hasModelOnnx {
			return fmt.Errorf("manifest must include model.onnx file")
		}
	}

	// Validate variant files if present
	for variantID, variantEntry := range m.Variants {
		if len(variantEntry.Files) == 0 {
			return fmt.Errorf("variant %s has no files", variantID)
		}
		for _, variantFile := range variantEntry.Files {
			if variantFile.Name == "" {
				return fmt.Errorf("variant %s file missing name", variantID)
			}
			if variantFile.Digest == "" {
				return fmt.Errorf("variant %s file missing digest", variantID)
			}
		}
		// Validate variant ID is known
		if _, ok := VariantFilenames[variantID]; !ok {
			return fmt.Errorf("unknown variant identifier: %s (valid: f16, bf16, i8, i8-st, i4)", variantID)
		}
	}

	return nil
}

// ParseManifest parses a JSON manifest
func ParseManifest(data []byte) (*ModelManifest, error) {
	var manifest ModelManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, fmt.Errorf("parsing manifest: %w", err)
	}
	if err := manifest.Validate(); err != nil {
		return nil, err
	}
	return &manifest, nil
}

// RegistryIndex lists all available models in the registry
type RegistryIndex struct {
	// SchemaVersion is the index format version
	SchemaVersion int `json:"schemaVersion"`
	// Models lists all available model manifests
	Models []ModelIndexEntry `json:"models"`
}

// ModelIndexEntry is a summary of a model in the registry index
type ModelIndexEntry struct {
	// Name is the model identifier
	Name string `json:"name"`
	// Type is the model type
	Type ModelType `json:"type"`
	// Description is a human-readable description
	Description string `json:"description,omitempty"`
	// Capabilities lists special capabilities (e.g., ["multimodal"])
	Capabilities []string `json:"capabilities,omitempty"`
	// Size is the total size of all files in bytes
	Size int64 `json:"size,omitempty"`
	// Variants lists available variant identifiers (e.g., ["f16", "i8"])
	Variants []string `json:"variants,omitempty"`
}

// ParseRegistryIndex parses a JSON registry index
func ParseRegistryIndex(data []byte) (*RegistryIndex, error) {
	var index RegistryIndex
	if err := json.Unmarshal(data, &index); err != nil {
		return nil, fmt.Errorf("parsing registry index: %w", err)
	}
	if index.SchemaVersion != 1 {
		return nil, fmt.Errorf("unsupported index schema version: %d", index.SchemaVersion)
	}
	return &index, nil
}
