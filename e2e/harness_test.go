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
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
)

// Models to download for e2e tests
var testModels = []string{
	"bge-small-en-v1.5",
	"clip-vit-base-patch32",
	// Uncomment when needed - these are larger models
	// "chonky-mmbert-small-multilingual-1",
	// "mxbai-rerank-base-v1",
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

	// Download models
	if err := downloadTestModels(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to download test models: %v\n", err)
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
		// Check if model already exists
		modelPath := filepath.Join(testModelsDir, "embedders", modelName)
		if modelName == "mxbai-rerank-base-v1" {
			modelPath = filepath.Join(testModelsDir, "rerankers", modelName)
		} else if modelName == "chonky-mmbert-small-multilingual-1" {
			modelPath = filepath.Join(testModelsDir, "chunkers", modelName)
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
