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
	"testing"
)

func TestParseModelType(t *testing.T) {
	tests := []struct {
		input    string
		expected ModelType
		wantErr  bool
	}{
		{"embedder", ModelTypeEmbedder, false},
		{"embedders", ModelTypeEmbedder, false},
		{"EMBEDDER", ModelTypeEmbedder, false},
		{"chunker", ModelTypeChunker, false},
		{"chunkers", ModelTypeChunker, false},
		{"reranker", ModelTypeReranker, false},
		{"rerankers", ModelTypeReranker, false},
		{"ner", ModelTypeNER, false},
		{"NER", ModelTypeNER, false},
		{"unknown", "", true},
		{"", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, err := ParseModelType(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseModelType(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
				return
			}
			if got != tt.expected {
				t.Errorf("ParseModelType(%q) = %v, want %v", tt.input, got, tt.expected)
			}
		})
	}
}

func TestModelTypeDirName(t *testing.T) {
	tests := []struct {
		modelType ModelType
		expected  string
	}{
		{ModelTypeEmbedder, "embedders"},
		{ModelTypeChunker, "chunkers"},
		{ModelTypeReranker, "rerankers"},
		{ModelTypeNER, "ner"},
	}

	for _, tt := range tests {
		t.Run(string(tt.modelType), func(t *testing.T) {
			if got := tt.modelType.DirName(); got != tt.expected {
				t.Errorf("DirName() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestParseManifest(t *testing.T) {
	validManifest := `{
		"schemaVersion": 1,
		"name": "bge-small-en-v1.5",
		"type": "embedder",
		"description": "BGE small English embedding model",
		"files": [
			{"name": "model.onnx", "digest": "sha256:abc123", "size": 12345},
			{"name": "tokenizer.json", "digest": "sha256:def456", "size": 1000}
		],
		"variants": {
			"i8": {"name": "model_i8.onnx", "digest": "sha256:ghi789", "size": 6789},
			"f16": {"name": "model_f16.onnx", "digest": "sha256:jkl012", "size": 8000}
		}
	}`

	t.Run("valid manifest", func(t *testing.T) {
		manifest, err := ParseManifest([]byte(validManifest))
		if err != nil {
			t.Fatalf("ParseManifest() error = %v", err)
		}

		if manifest.Name != "bge-small-en-v1.5" {
			t.Errorf("Name = %v, want bge-small-en-v1.5", manifest.Name)
		}
		if manifest.Type != ModelTypeEmbedder {
			t.Errorf("Type = %v, want embedder", manifest.Type)
		}
		if len(manifest.Files) != 2 {
			t.Errorf("len(Files) = %v, want 2", len(manifest.Files))
		}
		if len(manifest.Variants) != 2 {
			t.Errorf("len(Variants) = %v, want 2", len(manifest.Variants))
		}
		if _, ok := manifest.Variants["i8"]; !ok {
			t.Error("Variants should contain 'i8' key")
		}
	})

	t.Run("missing name", func(t *testing.T) {
		data := `{"schemaVersion": 1, "type": "embedder", "files": [{"name": "model.onnx", "digest": "sha256:abc", "size": 1}]}`
		_, err := ParseManifest([]byte(data))
		if err == nil {
			t.Error("Expected error for missing name")
		}
	})

	t.Run("missing type", func(t *testing.T) {
		data := `{"schemaVersion": 1, "name": "test", "files": [{"name": "model.onnx", "digest": "sha256:abc", "size": 1}]}`
		_, err := ParseManifest([]byte(data))
		if err == nil {
			t.Error("Expected error for missing type")
		}
	})

	t.Run("missing model.onnx", func(t *testing.T) {
		data := `{"schemaVersion": 1, "name": "test", "type": "embedder", "files": [{"name": "tokenizer.json", "digest": "sha256:abc", "size": 1}]}`
		_, err := ParseManifest([]byte(data))
		if err == nil {
			t.Error("Expected error for missing model.onnx")
		}
	})

	t.Run("invalid schema version", func(t *testing.T) {
		data := `{"schemaVersion": 2, "name": "test", "type": "embedder", "files": [{"name": "model.onnx", "digest": "sha256:abc", "size": 1}]}`
		_, err := ParseManifest([]byte(data))
		if err == nil {
			t.Error("Expected error for invalid schema version")
		}
	})

	t.Run("invalid digest format", func(t *testing.T) {
		data := `{"schemaVersion": 1, "name": "test", "type": "embedder", "files": [{"name": "model.onnx", "digest": "md5:abc", "size": 1}]}`
		_, err := ParseManifest([]byte(data))
		if err == nil {
			t.Error("Expected error for invalid digest format")
		}
	})

	t.Run("invalid JSON", func(t *testing.T) {
		_, err := ParseManifest([]byte("not json"))
		if err == nil {
			t.Error("Expected error for invalid JSON")
		}
	})
}

func TestParseRegistryIndex(t *testing.T) {
	validIndex := `{
		"schemaVersion": 1,
		"models": [
			{"name": "bge-small-en-v1.5", "type": "embedder", "description": "BGE embedding", "size": 12345, "variants": ["i8", "f16"]},
			{"name": "mxbai-rerank-base-v1", "type": "reranker", "description": "Reranker", "size": 67890}
		]
	}`

	t.Run("valid index", func(t *testing.T) {
		index, err := ParseRegistryIndex([]byte(validIndex))
		if err != nil {
			t.Fatalf("ParseRegistryIndex() error = %v", err)
		}

		if len(index.Models) != 2 {
			t.Errorf("len(Models) = %v, want 2", len(index.Models))
		}
		if index.Models[0].Name != "bge-small-en-v1.5" {
			t.Errorf("Models[0].Name = %v, want bge-small-en-v1.5", index.Models[0].Name)
		}
		if len(index.Models[0].Variants) != 2 {
			t.Errorf("Models[0].Variants should have 2 entries, got %d", len(index.Models[0].Variants))
		}
	})

	t.Run("invalid schema version", func(t *testing.T) {
		data := `{"schemaVersion": 99, "models": []}`
		_, err := ParseRegistryIndex([]byte(data))
		if err == nil {
			t.Error("Expected error for invalid schema version")
		}
	})
}

func TestManifestValidate(t *testing.T) {
	tests := []struct {
		name    string
		modify  func(*ModelManifest)
		wantErr bool
	}{
		{
			name:    "valid manifest",
			modify:  func(m *ModelManifest) {},
			wantErr: false,
		},
		{
			name:    "invalid model type",
			modify:  func(m *ModelManifest) { m.Type = "invalid" },
			wantErr: true,
		},
		{
			name:    "empty files",
			modify:  func(m *ModelManifest) { m.Files = nil },
			wantErr: true,
		},
		{
			name: "file missing name",
			modify: func(m *ModelManifest) {
				m.Files[0].Name = ""
			},
			wantErr: true,
		},
		{
			name: "file missing digest",
			modify: func(m *ModelManifest) {
				m.Files[0].Digest = ""
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manifest := &ModelManifest{
				SchemaVersion: 1,
				Name:          "test-model",
				Type:          ModelTypeEmbedder,
				Files: []ModelFile{
					{Name: "model.onnx", Digest: "sha256:abc123", Size: 1000},
				},
			}
			tt.modify(manifest)

			err := manifest.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
