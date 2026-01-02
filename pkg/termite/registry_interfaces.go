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

package termite

import (
	"github.com/antflydb/antfly-go/libaf/reranking"
	"github.com/antflydb/termite/pkg/termite/lib/ner"
)

// RerankerRegistryInterface defines the interface for reranker model registries.
// This enables testing with mock implementations.
type RerankerRegistryInterface interface {
	// Get retrieves a reranker model by name, loading it if necessary
	Get(modelName string) (reranking.Model, error)
	// List returns all available model names
	List() []string
	// Close shuts down the registry and releases resources
	Close() error
}

// NERRegistryInterface defines the interface for NER model registries.
// This enables testing with mock implementations.
type NERRegistryInterface interface {
	// Get retrieves a NER model by name, loading it if necessary
	Get(modelName string) (ner.Model, error)
	// GetRecognizer retrieves a GLiNER/REBEL recognizer by name
	GetRecognizer(modelName string) (ner.Recognizer, error)
	// List returns all available model names
	List() []string
	// ListRecognizers returns names of models that support GLiNER/REBEL recognition
	ListRecognizers() []string
	// ListWithCapabilities returns a map of model names to their capabilities
	ListWithCapabilities() map[string][]string
	// HasCapability checks if a model has a specific capability
	HasCapability(modelName, capability string) bool
	// IsRecognizer checks if a model supports GLiNER/REBEL recognition
	IsRecognizer(modelName string) bool
	// Close shuts down the registry and releases resources
	Close() error
}

// Ensure concrete types implement the interfaces
var (
	_ RerankerRegistryInterface = (*RerankerRegistry)(nil)
	_ NERRegistryInterface      = (*NERRegistry)(nil)
)
