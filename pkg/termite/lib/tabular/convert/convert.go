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

// Package convert provides parsers that convert XGBoost, LightGBM, and CatBoost
// model files into Termite's tabular IR format.
package convert

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/antflydb/termite/pkg/termite/lib/tabular"
)

// SaveModel writes tabular_model.json and model_manifest.json to outDir.
func SaveModel(model *tabular.TabularModel, outDir string) error {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("creating output directory: %w", err)
	}

	// Write tabular_model.json
	data, err := json.MarshalIndent(model, "", "  ")
	if err != nil {
		return fmt.Errorf("marshalling model: %w", err)
	}
	modelPath := filepath.Join(outDir, tabular.TabularModelFilename)
	if err := os.WriteFile(modelPath, data, 0o644); err != nil {
		return fmt.Errorf("writing model file: %w", err)
	}

	// Write model_manifest.json
	manifest := buildManifest(model)
	mdata, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("marshalling manifest: %w", err)
	}
	manifestPath := filepath.Join(outDir, "model_manifest.json")
	if err := os.WriteFile(manifestPath, mdata, 0o644); err != nil {
		return fmt.Errorf("writing manifest file: %w", err)
	}

	return nil
}

type manifest struct {
	ModelType    string            `json:"model_type"`
	Capabilities []string         `json:"capabilities"`
	Metadata     manifestMetadata `json:"metadata"`
}

type manifestMetadata struct {
	Name            string `json:"name"`
	SourceFramework string `json:"source_framework"`
	Task            string `json:"task"`
}

func buildManifest(m *tabular.TabularModel) manifest {
	caps := []string{"tabular"}
	for _, stage := range m.Pipeline {
		switch stage.Type {
		case tabular.StageTreeEnsemble:
			caps = append(caps, "tree_ensemble")
		case tabular.StageLinear:
			caps = append(caps, "linear_model")
		case tabular.StageSVM:
			caps = append(caps, "svm")
		}
	}
	return manifest{
		ModelType:    "predictor",
		Capabilities: caps,
		Metadata: manifestMetadata{
			Name:            m.Metadata.Name,
			SourceFramework: m.Metadata.SourceFramework,
			Task:            string(m.Metadata.Task),
		},
	}
}

// PrintSummary prints a human-readable summary of the converted model.
func PrintSummary(model *tabular.TabularModel, outDir string) {
	fmt.Printf("Model saved to %s/\n", outDir)
	fmt.Printf("  tabular_model.json  (%s, %s)\n", model.Metadata.SourceFramework, model.Metadata.Task)

	for _, stage := range model.Pipeline {
		if stage.TreeEnsemble != nil {
			te := stage.TreeEnsemble
			totalNodes := len(te.Nodes.FeatureIndex)
			fmt.Printf("  Trees: %d, Nodes: %d, Max depth: %d, Features: %d\n",
				te.NumTrees, totalNodes, te.MaxDepth, te.NumFeatures)
		}
		if stage.Linear != nil {
			lm := stage.Linear
			fmt.Printf("  Linear: %dx%d\n", len(lm.Weights), len(lm.Weights[0]))
		}
	}

	fmt.Println("  model_manifest.json (type: predictor)")
}
