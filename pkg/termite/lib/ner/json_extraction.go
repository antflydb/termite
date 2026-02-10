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

package ner

import (
	"context"
	"fmt"
	"strings"
)

// FieldType represents the type of an extraction field.
type FieldType int

const (
	// FieldTypeStr keeps only the top-scoring span for a field.
	FieldTypeStr FieldType = iota
	// FieldTypeList keeps all extracted spans for a field.
	FieldTypeList
)

// SchemaField represents a single field in an extraction schema.
type SchemaField struct {
	// Name is the field name (used as the NER label).
	Name string
	// Type is the field type (str or list).
	Type FieldType
	// Choices contains valid options for choice fields (nil for non-choice fields).
	Choices []string
}

// ExtractionSchema represents a named structure to extract from text.
type ExtractionSchema struct {
	// Name is the structure name (e.g., "person").
	Name string
	// Fields are the fields to extract within this structure.
	Fields []SchemaField
}

// ExtractionConfig holds configuration for JSON extraction.
type ExtractionConfig struct {
	// Threshold is the score threshold for span extraction (0.0-1.0).
	Threshold float32
	// FlatNER if true, don't allow nested/overlapping entities.
	FlatNER bool
	// IncludeConfidence if true, include confidence scores in output.
	IncludeConfidence bool
	// IncludeSpans if true, include span offsets in output.
	IncludeSpans bool
}

// DefaultExtractionConfig returns sensible defaults for extraction.
func DefaultExtractionConfig() ExtractionConfig {
	return ExtractionConfig{
		Threshold: 0.3,
		FlatNER:   true,
	}
}

// ExtractedFieldValue represents a single extracted value for a field.
type ExtractedFieldValue struct {
	// Value is the extracted text.
	Value string `json:"value"`
	// Score is the confidence score (included when IncludeConfidence is true).
	Score float32 `json:"score,omitempty"`
	// Start is the character offset (included when IncludeSpans is true).
	// Pointer type so offset 0 is not silently dropped.
	Start *int `json:"start,omitempty"`
	// End is the character offset (included when IncludeSpans is true).
	// Pointer type so offset 0 is not silently dropped.
	End *int `json:"end,omitempty"`
}

// ExtractedInstance represents a single extracted instance of a structure.
// Keys are field names, values are either a single ExtractedFieldValue (for ::str)
// or a slice of ExtractedFieldValue (for ::list).
type ExtractedInstance map[string]any

// ExtractionResult holds the extraction results for a single text.
// Keys are structure names, values are slices of ExtractedInstance.
type ExtractionResult map[string][]ExtractedInstance

// JSONExtractor defines the interface for models that support structured JSON extraction.
type JSONExtractor interface {
	// ExtractJSON extracts structured JSON from text based on the given schemas.
	ExtractJSON(ctx context.Context, texts []string, schemas []ExtractionSchema, config ExtractionConfig) ([]ExtractionResult, error)

	// SupportsJSONExtraction returns true if the model supports JSON extraction.
	SupportsJSONExtraction() bool
}

// ParseSchemaString parses a schema map (e.g., {"person": ["name::str", "age::str", "skills::list"]})
// into ExtractionSchema values.
//
// Field syntax:
//   - "name::str"                 -> FieldTypeStr, no choices
//   - "skills::list"              -> FieldTypeList, no choices
//   - "role::[engineer|manager]"  -> FieldTypeStr with choices
//   - "name"                      -> FieldTypeStr (default if no :: separator)
func ParseSchemaString(schema map[string][]string) ([]ExtractionSchema, error) {
	schemas := make([]ExtractionSchema, 0, len(schema))

	for structName, fieldDefs := range schema {
		if structName == "" {
			return nil, fmt.Errorf("empty structure name")
		}
		if len(fieldDefs) == 0 {
			return nil, fmt.Errorf("structure %q has no fields", structName)
		}

		fields := make([]SchemaField, 0, len(fieldDefs))
		for _, fieldDef := range fieldDefs {
			field, err := parseFieldDef(fieldDef)
			if err != nil {
				return nil, fmt.Errorf("structure %q: %w", structName, err)
			}
			fields = append(fields, field)
		}

		schemas = append(schemas, ExtractionSchema{
			Name:   structName,
			Fields: fields,
		})
	}

	return schemas, nil
}

// parseFieldDef parses a single field definition string.
func parseFieldDef(def string) (SchemaField, error) {
	def = strings.TrimSpace(def)
	if def == "" {
		return SchemaField{}, fmt.Errorf("empty field definition")
	}

	// Split on "::" separator
	parts := strings.SplitN(def, "::", 3)

	field := SchemaField{
		Name: strings.TrimSpace(parts[0]),
		Type: FieldTypeStr, // default
	}

	if field.Name == "" {
		return SchemaField{}, fmt.Errorf("empty field name in %q", def)
	}

	if len(parts) == 1 {
		// Just a name, default to str
		return field, nil
	}

	// Check for choice fields: "field::[opt1|opt2]::str" or "field::[opt1|opt2]"
	for i := 1; i < len(parts); i++ {
		part := strings.TrimSpace(parts[i])

		if strings.HasPrefix(part, "[") && strings.HasSuffix(part, "]") {
			// Choice list
			choicesStr := part[1 : len(part)-1]
			choices := strings.Split(choicesStr, "|")
			for j, c := range choices {
				choices[j] = strings.TrimSpace(c)
			}
			if len(choices) < 2 {
				return SchemaField{}, fmt.Errorf("choice field %q must have at least 2 options", field.Name)
			}
			for _, c := range choices {
				if c == "" {
					return SchemaField{}, fmt.Errorf("choice field %q has empty option", field.Name)
				}
			}
			field.Choices = choices
		} else {
			// Type specifier
			switch strings.ToLower(part) {
			case "str", "string":
				field.Type = FieldTypeStr
			case "list", "array":
				field.Type = FieldTypeList
			default:
				return SchemaField{}, fmt.Errorf("unknown field type %q in %q (expected str or list)", part, def)
			}
		}
	}

	return field, nil
}
