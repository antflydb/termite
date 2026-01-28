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
// Also checks the "onnx/" subdirectory where some HuggingFace models store encoder files.
func FindONNXFile(dir string, candidates []string) string {
	// Search directories: root directory and onnx/ subdirectory
	searchDirs := []string{dir, filepath.Join(dir, "onnx")}

	for _, searchDir := range searchDirs {
		for _, name := range candidates {
			path := filepath.Join(searchDir, name)
			if _, err := os.Stat(path); err == nil {
				return path
			}
		}
	}
	return ""
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
