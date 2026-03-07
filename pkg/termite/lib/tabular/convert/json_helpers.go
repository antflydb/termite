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

package convert

import (
	"fmt"
	"strconv"
	"strings"
)

// jsonObj returns a nested object, or nil if missing/wrong type.
func jsonObj(m map[string]any, key string) map[string]any {
	if m == nil {
		return nil
	}
	v, ok := m[key].(map[string]any)
	if !ok {
		return nil
	}
	return v
}

// jsonArray returns a nested array, or nil.
func jsonArray(m map[string]any, key string) []any {
	if m == nil {
		return nil
	}
	v, ok := m[key].([]any)
	if !ok {
		return nil
	}
	return v
}

// jsonStr returns a string value or "".
func jsonStr(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	switch v := m[key].(type) {
	case string:
		return v
	case float64:
		return fmt.Sprintf("%g", v)
	default:
		return ""
	}
}

// jsonStrDefault returns a string value or fallback.
func jsonStrDefault(m map[string]any, key, fallback string) string {
	s := jsonStr(m, key)
	if s == "" {
		return fallback
	}
	return s
}

// jsonInt returns an int value (from float64 or string).
func jsonInt(m map[string]any, key string) int {
	if m == nil {
		return 0
	}
	switch v := m[key].(type) {
	case float64:
		return int(v)
	case string:
		n, _ := strconv.Atoi(v)
		return n
	default:
		return 0
	}
}

// jsonHasKey returns true if the key exists in the map (regardless of value).
func jsonHasKey(m map[string]any, key string) bool {
	if m == nil {
		return false
	}
	_, ok := m[key]
	return ok
}

// jsonIntDefault returns an int value or fallback.
func jsonIntDefault(m map[string]any, key string, fallback int) int {
	if m == nil {
		return fallback
	}
	if _, ok := m[key]; !ok {
		return fallback
	}
	return jsonInt(m, key)
}

// jsonFloat returns a float64 value.
func jsonFloat(m map[string]any, key string) float64 {
	if m == nil {
		return 0
	}
	switch v := m[key].(type) {
	case float64:
		return v
	case string:
		f, _ := strconv.ParseFloat(v, 64)
		return f
	default:
		return 0
	}
}

// jsonBool converts a JSON value to bool.
func jsonBool(v any) bool {
	switch val := v.(type) {
	case bool:
		return val
	case float64:
		return val != 0
	case string:
		return val == "true" || val == "1"
	default:
		return false
	}
}

// jsonStringArray returns a []string from a JSON array.
func jsonStringArray(m map[string]any, key string) []string {
	arr := jsonArray(m, key)
	if arr == nil {
		return nil
	}
	result := make([]string, 0, len(arr))
	for _, v := range arr {
		if s, ok := v.(string); ok {
			result = append(result, s)
		}
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

// jsonFloatArray returns a []float64 from a JSON array.
func jsonFloatArray(m map[string]any, key string) []float64 {
	arr := jsonArray(m, key)
	if arr == nil {
		return nil
	}
	result := make([]float64, 0, len(arr))
	for _, v := range arr {
		if f, ok := v.(float64); ok {
			result = append(result, f)
		}
	}
	return result
}

// parseInt parses a string to int, returning fallback on error.
func parseInt(s string, fallback int) int {
	if s == "" {
		return fallback
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return fallback
	}
	return n
}

// parseFloat parses a string to float64, returning fallback on error.
func parseFloat(s string, fallback float64) float64 {
	if s == "" {
		return fallback
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return fallback
	}
	return f
}

// parseIntSlice parses a space-separated string of ints.
func parseIntSlice(s string) []int {
	fields := strings.Fields(s)
	result := make([]int, 0, len(fields))
	for _, f := range fields {
		n, _ := strconv.Atoi(f)
		result = append(result, n)
	}
	return result
}

// parseFloatSlice parses a space-separated string of floats.
func parseFloatSlice(s string) []float64 {
	fields := strings.Fields(s)
	result := make([]float64, 0, len(fields))
	for _, f := range fields {
		v, _ := strconv.ParseFloat(f, 64)
		result = append(result, v)
	}
	return result
}
