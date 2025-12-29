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

//go:build onnx && ORT

package e2e

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
)

// Models to download for e2e tests (from model registry)
var testModels = []string{
	"bge-small-en-v1.5",
	"clip-vit-base-patch32",
	"flan-t5-small-squad-qg",
	// Uncomment when needed - these are larger models
	// "chonky-mmbert-small-multilingual-1",
	// "mxbai-rerank-base-v1",
}

// HuggingFace models for e2e tests
type hfModel struct {
	name      string
	repo      string
	modelType modelregistry.ModelType
}

var testHFModels = []hfModel{
	{
		name:      "gliner_small-v2.1",
		repo:      "onnx-community/gliner_small-v2.1",
		modelType: modelregistry.ModelTypeRecognizer,
	},
}

// testModelsDir is the shared models directory for all e2e tests
var testModelsDir string

// TestMain sets up the e2e test environment by downloading required models
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

	// Download models from registry
	if err := downloadTestModels(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to download test models: %v\n", err)
		os.Exit(1)
	}

	// Download HuggingFace models
	if err := downloadHuggingFaceModels(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to download HuggingFace models: %v\n", err)
		os.Exit(1)
	}

	// Run tests
	code := m.Run()
	os.Exit(code)
}

// downloadTestModels downloads all required models for e2e tests
func downloadTestModels() error {
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
					fmt.Printf("  %s: %.0f%%\n", filename, percent)
				}
			}
		}),
	)

	for _, modelName := range testModels {
		// Determine model path based on model type
		var modelPath string
		switch modelName {
		case "mxbai-rerank-base-v1":
			modelPath = filepath.Join(testModelsDir, "rerankers", modelName)
		case "chonky-mmbert-small-multilingual-1":
			modelPath = filepath.Join(testModelsDir, "chunkers", modelName)
		case "flan-t5-small-squad-qg":
			modelPath = filepath.Join(testModelsDir, "questionators", modelName)
		default:
			modelPath = filepath.Join(testModelsDir, "embedders", modelName)
		}

		if _, err := os.Stat(modelPath); err == nil {
			fmt.Printf("Model %s already exists, skipping download\n", modelName)
			continue
		}

		fmt.Printf("Downloading model: %s\n", modelName)

		manifest, err := regClient.FetchManifest(ctx, modelName)
		if err != nil {
			return fmt.Errorf("fetch manifest for %s: %w", modelName, err)
		}

		// Pull the f32 variant (default)
		if err := regClient.PullModel(ctx, manifest, testModelsDir, []string{modelregistry.VariantF32}); err != nil {
			return fmt.Errorf("pull model %s: %w", modelName, err)
		}

		fmt.Printf("Downloaded model: %s\n", modelName)
	}

	return nil
}

// downloadHuggingFaceModels downloads models directly from HuggingFace
func downloadHuggingFaceModels() error {
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
					fmt.Printf("  %s: %.0f%%\n", filename, percent)
				}
			}
		}),
	)

	for _, model := range testHFModels {
		// Get the model path based on type
		var modelPath string
		switch model.modelType {
		case modelregistry.ModelTypeRecognizer:
			modelPath = filepath.Join(testModelsDir, "recognizers", model.name)
		case modelregistry.ModelTypeQuestionator:
			modelPath = filepath.Join(testModelsDir, "questionators", model.name)
		default:
			modelPath = filepath.Join(testModelsDir, model.name)
		}

		if _, err := os.Stat(modelPath); err == nil {
			fmt.Printf("HuggingFace model %s already exists, skipping download\n", model.name)
			continue
		}

		fmt.Printf("Downloading HuggingFace model: %s from %s\n", model.name, model.repo)

		if err := hfClient.PullFromHuggingFace(ctx, model.repo, model.modelType, testModelsDir, ""); err != nil {
			return fmt.Errorf("pull HuggingFace model %s: %w", model.name, err)
		}

		fmt.Printf("Downloaded HuggingFace model: %s\n", model.name)
	}

	return nil
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

// getQuestionatorModelsDir returns the questionators subdirectory
func getQuestionatorModelsDir() string {
	return filepath.Join(testModelsDir, "questionators")
}
