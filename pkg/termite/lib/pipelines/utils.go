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

package pipelines

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
)

// FirstNonZero returns the first non-zero value from the arguments.
// This is useful for config resolution where multiple fields may provide the same value.
func FirstNonZero(values ...int) int {
	for _, v := range values {
		if v != 0 {
			return v
		}
	}
	return 0
}

// FindONNXFile looks for an ONNX file in the given directory.
// It searches for the first matching file from the candidates list.
// If no exact candidate is found, it also checks for variant filenames
// (e.g., model.onnx → model_i8.onnx, text_model.onnx → text_model_f16.onnx).
// Also checks the "onnx/" subdirectory where some HuggingFace models store encoder files.
func FindONNXFile(dir string, candidates []string) string {
	searchDirs := []string{dir, filepath.Join(dir, "onnx")}

	for _, searchDir := range searchDirs {
		// First pass: try exact candidate names
		for _, name := range candidates {
			path := filepath.Join(searchDir, name)
			if _, err := os.Stat(path); err == nil {
				return path
			}
		}
		// Second pass: try variant filenames derived from each candidate
		for _, name := range candidates {
			stem := strings.TrimSuffix(name, ".onnx")
			if stem == name {
				continue // not an .onnx candidate
			}
			for _, suffix := range modelregistry.VariantSuffixes {
				path := filepath.Join(searchDir, stem+"_"+suffix+".onnx")
				if _, err := os.Stat(path); err == nil {
					return path
				}
			}
		}
	}
	return ""
}

// ParseTokenID converts a JSON-decoded token ID (float64 or []any) to int32.
// HuggingFace configs represent eos_token_id as either a number or an array;
// when an array, only the first element is used.
func ParseTokenID(v any) int32 {
	switch val := v.(type) {
	case float64:
		return int32(val)
	case []any:
		if len(val) > 0 {
			if f, ok := val[0].(float64); ok {
				return int32(f)
			}
		}
	}
	return 0
}

// IntToInt32 converts a slice of int to a slice of int32.
func IntToInt32(ids []int) []int32 {
	result := make([]int32, len(ids))
	for i, id := range ids {
		result[i] = int32(id)
	}
	return result
}

// Int32ToInt converts a slice of int32 to a slice of int.
func Int32ToInt(ids []int32) []int {
	result := make([]int, len(ids))
	for i, id := range ids {
		result[i] = int(id)
	}
	return result
}
