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
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
)

// testModelsDir is the shared models directory for all e2e tests
var testModelsDir string

// modelDownloadMutex ensures only one model downloads at a time to avoid
// duplicate downloads and provide clearer progress output
var modelDownloadMutex sync.Mutex

// TestMain sets up the e2e test environment (models directory only - downloads are lazy)
func TestMain(m *testing.M) {
	// Use TERMITE_MODELS_DIR if set, otherwise use a temp directory
	testModelsDir = os.Getenv("TERMITE_MODELS_DIR")
	if testModelsDir == "" {
		// Create temp directory for models
		var err error
		testModelsDir, err = os.MkdirTemp("", "termite-e2e-models-*")
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to create temp models dir: %v\n", err)
			os.Exit(1)
		}
		// Clean up temp directory after tests (unless KEEP_TEST_MODELS is set)
		if os.Getenv("KEEP_TEST_MODELS") != "true" {
			defer os.RemoveAll(testModelsDir)
		}
	}

	fmt.Printf("E2E Test Setup: Using models directory: %s\n", testModelsDir)
	fmt.Printf("E2E Test Setup: Models will be downloaded lazily as needed by each test\n")

	// Run tests
	code := m.Run()
	os.Exit(code)
}

// ModelType represents the type of model for determining directory placement
type ModelType string

const (
	ModelTypeEmbedder    ModelType = "embedders"
	ModelTypeReranker    ModelType = "rerankers"
	ModelTypeChunker     ModelType = "chunkers"
	ModelTypeRewriter    ModelType = "rewriters"
	ModelTypeRecognizer  ModelType = "recognizers"
	ModelTypeGenerator   ModelType = "generators"
	ModelTypeClassifier  ModelType = "classifiers"
	ModelTypeReader      ModelType = "readers"
	ModelTypeTranscriber ModelType = "transcribers"
)

// ensureRegistryModel downloads a model from the Antfly model registry if not present.
// It is safe to call from multiple tests - only downloads once.
// Returns the model path and any error.
func ensureRegistryModel(t *testing.T, modelName string, modelType ModelType) string {
	t.Helper()

	modelDownloadMutex.Lock()
	defer modelDownloadMutex.Unlock()

	modelPath := filepath.Join(testModelsDir, string(modelType), modelName)

	if _, err := os.Stat(modelPath); err == nil {
		t.Logf("Model %s already exists at %s", modelName, modelPath)
		return modelPath
	}

	t.Logf("Downloading model from registry: %s", modelName)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	// Track download progress per file (only log at milestones)
	lastMilestone := make(map[string]int)
	regClient := modelregistry.NewClient(
		modelregistry.WithProgressHandler(func(downloaded, total int64, filename string) {
			if total > 0 {
				percent := float64(downloaded) / float64(total) * 100
				milestone := int(percent / 25)
				if milestone > lastMilestone[filename] || (downloaded == total && lastMilestone[filename] < 4) {
					lastMilestone[filename] = milestone
					t.Logf("  %s: %.0f%%", filename, percent)
				}
			}
		}),
	)

	manifest, err := regClient.FetchManifest(ctx, modelName)
	if err != nil {
		t.Fatalf("Failed to fetch manifest for %s: %v", modelName, err)
	}

	// Pull the f32 variant (default)
	if err := regClient.PullModel(ctx, manifest, testModelsDir, []string{modelregistry.VariantF32}); err != nil {
		t.Fatalf("Failed to pull model %s: %v", modelName, err)
	}

	t.Logf("Successfully downloaded model: %s", modelName)
	return modelPath
}

// ensureHuggingFaceModel downloads a model directly from HuggingFace if not present.
// modelName is the name used in the local directory (e.g., "gliner_small-v2.1")
// repo is the HuggingFace repository (e.g., "onnx-community/gliner_small-v2.1")
// Returns the model path.
func ensureHuggingFaceModel(t *testing.T, modelName, repo string, modelType ModelType) string {
	t.Helper()

	modelDownloadMutex.Lock()
	defer modelDownloadMutex.Unlock()

	modelPath := filepath.Join(testModelsDir, string(modelType), modelName)

	if _, err := os.Stat(modelPath); err == nil {
		t.Logf("HuggingFace model %s already exists at %s", modelName, modelPath)
		return modelPath
	}

	t.Logf("Downloading model from HuggingFace: %s from %s", modelName, repo)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	// Track download progress per file (only log at milestones)
	lastMilestone := make(map[string]int)
	hfClient := modelregistry.NewHuggingFaceClient(
		modelregistry.WithHFProgressHandler(func(downloaded, total int64, filename string) {
			if total > 0 {
				percent := float64(downloaded) / float64(total) * 100
				milestone := int(percent / 25)
				if milestone > lastMilestone[filename] || (downloaded == total && lastMilestone[filename] < 4) {
					lastMilestone[filename] = milestone
					t.Logf("  %s: %.0f%%", filename, percent)
				}
			}
		}),
	)

	// Convert our ModelType to modelregistry.ModelType
	var regModelType modelregistry.ModelType
	switch modelType {
	case ModelTypeRecognizer:
		regModelType = modelregistry.ModelTypeRecognizer
	case ModelTypeRewriter:
		regModelType = modelregistry.ModelTypeRewriter
	case ModelTypeEmbedder:
		regModelType = modelregistry.ModelTypeEmbedder
	case ModelTypeReranker:
		regModelType = modelregistry.ModelTypeReranker
	case ModelTypeChunker:
		regModelType = modelregistry.ModelTypeChunker
	case ModelTypeGenerator:
		regModelType = modelregistry.ModelTypeGenerator
	case ModelTypeClassifier:
		regModelType = modelregistry.ModelTypeClassifier
	case ModelTypeReader:
		regModelType = modelregistry.ModelTypeReader
	case ModelTypeTranscriber:
		regModelType = modelregistry.ModelTypeTranscriber
	default:
		regModelType = modelregistry.ModelTypeEmbedder
	}

	if err := hfClient.PullFromHuggingFace(ctx, repo, regModelType, testModelsDir, ""); err != nil {
		t.Fatalf("Failed to pull HuggingFace model %s: %v", modelName, err)
	}

	t.Logf("Successfully downloaded HuggingFace model: %s", modelName)
	return modelPath
}

// findAvailablePort finds an available TCP port
func findAvailablePort(t *testing.T) int {
	t.Helper()

	listener, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Failed to find available port: %v", err)
	}
	defer listener.Close()

	return listener.Addr().(*net.TCPAddr).Port
}

// getTestModelsDir returns the shared models directory for tests
func getTestModelsDir() string {
	return testModelsDir
}

// getEmbedderModelsDir returns the embedders subdirectory
func getEmbedderModelsDir() string {
	return filepath.Join(testModelsDir, "embedders")
}

// getRerankerModelsDir returns the rerankers subdirectory
func getRerankerModelsDir() string {
	return filepath.Join(testModelsDir, "rerankers")
}

// getChunkerModelsDir returns the chunkers subdirectory
func getChunkerModelsDir() string {
	return filepath.Join(testModelsDir, "chunkers")
}

// getRecognizerModelsDir returns the recognizers subdirectory
func getRecognizerModelsDir() string {
	return filepath.Join(testModelsDir, "recognizers")
}

// getRewriterModelsDir returns the rewriters subdirectory
func getRewriterModelsDir() string {
	return filepath.Join(testModelsDir, "rewriters")
}

// getGeneratorModelsDir returns the generators subdirectory
func getGeneratorModelsDir() string {
	return filepath.Join(testModelsDir, "generators")
}

// getClassifierModelsDir returns the classifiers subdirectory
func getClassifierModelsDir() string {
	return filepath.Join(testModelsDir, "classifiers")
}

// getTranscriberModelsDir returns the transcribers subdirectory
func getTranscriberModelsDir() string {
	return filepath.Join(testModelsDir, "transcribers")
}

// fileExists checks if a file exists and is not a directory
func fileExists(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return !info.IsDir()
}
