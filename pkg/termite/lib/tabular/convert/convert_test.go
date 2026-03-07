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
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// makeXGBJSON creates a minimal XGBoost JSON model for testing.
func makeXGBJSON(numFeatures int, objective string) map[string]any {
	return map[string]any{
		"learner": map[string]any{
			"learner_model_param": map[string]any{
				"num_feature": float64(numFeatures),
				"base_score":  "0.5",
				"num_class":   float64(0),
			},
			"objective": map[string]any{"name": objective},
			"gradient_booster": map[string]any{
				"model": map[string]any{
					"trees": []any{
						map[string]any{
							"nodes": []any{
								map[string]any{
									"nodeid":           float64(0),
									"split_feature_id": float64(0),
									"split_condition":  float64(0.5),
									"yes":              float64(1),
									"no":               float64(2),
									"missing":          float64(1),
								},
								map[string]any{"nodeid": float64(1), "leaf": float64(0.3)},
								map[string]any{"nodeid": float64(2), "leaf": float64(-0.2)},
							},
						},
						map[string]any{
							"nodes": []any{
								map[string]any{
									"nodeid":           float64(0),
									"split_feature_id": float64(1),
									"split_condition":  float64(1.0),
									"yes":              float64(1),
									"no":               float64(2),
									"missing":          float64(2),
								},
								map[string]any{"nodeid": float64(1), "leaf": float64(0.1)},
								map[string]any{"nodeid": float64(2), "leaf": float64(-0.1)},
							},
						},
					},
				},
			},
		},
	}
}

func writeJSONFile(t *testing.T, data map[string]any) string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "model-*.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.NewEncoder(f).Encode(data); err != nil {
		t.Fatal(err)
	}
	f.Close()
	return f.Name()
}

func TestParseXGBoost(t *testing.T) {
	data := makeXGBJSON(3, "binary:logistic")
	path := writeJSONFile(t, data)

	model, err := ParseXGBoost(path, "test-model")
	if err != nil {
		t.Fatal(err)
	}

	if model.SchemaVersion != 1 {
		t.Errorf("schema_version = %d, want 1", model.SchemaVersion)
	}
	if model.Metadata.Name != "test-model" {
		t.Errorf("name = %q, want %q", model.Metadata.Name, "test-model")
	}
	if model.Metadata.SourceFramework != "xgboost" {
		t.Errorf("source_framework = %q, want %q", model.Metadata.SourceFramework, "xgboost")
	}
	if model.Metadata.Task != "binary_classification" {
		t.Errorf("task = %q, want %q", model.Metadata.Task, "binary_classification")
	}
	if model.Metadata.NumFeatures != 3 {
		t.Errorf("num_features = %d, want 3", model.Metadata.NumFeatures)
	}
	if model.Output.Activation != "sigmoid" {
		t.Errorf("activation = %q, want %q", model.Output.Activation, "sigmoid")
	}
	if model.Output.NumOutputs != 1 {
		t.Errorf("num_outputs = %d, want 1", model.Output.NumOutputs)
	}

	if len(model.Pipeline) != 1 {
		t.Fatalf("pipeline len = %d, want 1", len(model.Pipeline))
	}
	te := model.Pipeline[0].TreeEnsemble
	if te == nil {
		t.Fatal("tree_ensemble is nil")
	}
	if te.NumTrees != 2 {
		t.Errorf("num_trees = %d, want 2", te.NumTrees)
	}
	if te.BaseScore != 0.5 {
		t.Errorf("base_score = %f, want 0.5", te.BaseScore)
	}

	// 2 trees * 3 nodes = 6 total
	if len(te.Nodes.FeatureIndex) != 6 {
		t.Errorf("total nodes = %d, want 6", len(te.Nodes.FeatureIndex))
	}
	if te.Nodes.TreeStarts[0] != 0 || te.Nodes.TreeStarts[1] != 3 {
		t.Errorf("tree_starts = %v, want [0, 3]", te.Nodes.TreeStarts)
	}
}

func TestXGBoostNodeStructure(t *testing.T) {
	data := makeXGBJSON(3, "binary:logistic")
	path := writeJSONFile(t, data)

	model, err := ParseXGBoost(path, "structure-test")
	if err != nil {
		t.Fatal(err)
	}
	te := model.Pipeline[0].TreeEnsemble

	// Tree 0: root splits on feature 0 at 0.5
	if te.Nodes.FeatureIndex[0] != 0 {
		t.Errorf("node 0 feature = %d, want 0", te.Nodes.FeatureIndex[0])
	}
	if te.Nodes.Threshold[0] != 0.5 {
		t.Errorf("node 0 threshold = %f, want 0.5", te.Nodes.Threshold[0])
	}
	if te.Nodes.FeatureIndex[1] != -1 {
		t.Errorf("node 1 should be leaf, got feature %d", te.Nodes.FeatureIndex[1])
	}
	if te.Nodes.LeafValue[1] != 0.3 {
		t.Errorf("node 1 leaf = %f, want 0.3", te.Nodes.LeafValue[1])
	}
	if te.Nodes.FeatureIndex[2] != -1 {
		t.Errorf("node 2 should be leaf, got feature %d", te.Nodes.FeatureIndex[2])
	}
	if te.Nodes.LeafValue[2] != -0.2 {
		t.Errorf("node 2 leaf = %f, want -0.2", te.Nodes.LeafValue[2])
	}
	if !te.Nodes.DefaultLeft[0] {
		t.Error("node 0 default_left should be true (missing=yes)")
	}

	// Tree 1 at index 3
	if te.Nodes.FeatureIndex[3] != 1 {
		t.Errorf("node 3 feature = %d, want 1", te.Nodes.FeatureIndex[3])
	}
	if te.Nodes.Threshold[3] != 1.0 {
		t.Errorf("node 3 threshold = %f, want 1.0", te.Nodes.Threshold[3])
	}
	if te.Nodes.DefaultLeft[3] {
		t.Error("node 3 default_left should be false (missing=no)")
	}
}

func TestXGBoostRegression(t *testing.T) {
	data := makeXGBJSON(3, "reg:squarederror")
	path := writeJSONFile(t, data)

	model, err := ParseXGBoost(path, "regressor")
	if err != nil {
		t.Fatal(err)
	}

	if model.Metadata.Task != "regression" {
		t.Errorf("task = %q, want regression", model.Metadata.Task)
	}
	if model.Output.Activation != "identity" {
		t.Errorf("activation = %q, want identity", model.Output.Activation)
	}
}

func TestXGBoostBracketedBaseScore(t *testing.T) {
	v := parseXGBScalar("[4.82E-1]", 0)
	if v < 0.481 || v > 0.483 {
		t.Errorf("parseXGBScalar([4.82E-1]) = %f, want ~0.482", v)
	}
}

func TestSaveModel(t *testing.T) {
	data := makeXGBJSON(3, "binary:logistic")
	path := writeJSONFile(t, data)

	model, err := ParseXGBoost(path, "save-test")
	if err != nil {
		t.Fatal(err)
	}

	outDir := filepath.Join(t.TempDir(), "output")
	if err := SaveModel(model, outDir); err != nil {
		t.Fatal(err)
	}

	// Check files exist
	modelFile := filepath.Join(outDir, "tabular_model.json")
	manifestFile := filepath.Join(outDir, "model_manifest.json")

	if _, err := os.Stat(modelFile); err != nil {
		t.Errorf("tabular_model.json not found: %v", err)
	}
	if _, err := os.Stat(manifestFile); err != nil {
		t.Errorf("model_manifest.json not found: %v", err)
	}

	// Validate model JSON
	raw, _ := os.ReadFile(modelFile)
	var loaded map[string]any
	if err := json.Unmarshal(raw, &loaded); err != nil {
		t.Fatal(err)
	}
	if loaded["schema_version"].(float64) != 1 {
		t.Error("loaded schema_version != 1")
	}

	// Validate manifest
	raw, _ = os.ReadFile(manifestFile)
	var mf map[string]any
	if err := json.Unmarshal(raw, &mf); err != nil {
		t.Fatal(err)
	}
	if mf["model_type"] != "predictor" {
		t.Errorf("manifest model_type = %v, want predictor", mf["model_type"])
	}
}

// makeLGBTextModel creates a minimal LightGBM text model.
func makeLGBTextModel() string {
	return `max_feature_idx=2
num_class=1
objective=binary

Tree=0
num_leaves=3
split_feature=0
threshold=0.5
decision_type=2
left_child=-1
right_child=-2
leaf_value=0.3 -0.2

Tree=1
num_leaves=3
split_feature=1
threshold=1.0
decision_type=0
left_child=-1
right_child=-2
leaf_value=0.1 -0.1

end of trees
`
}

func TestParseLightGBMText(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.txt")
	if err := os.WriteFile(path, []byte(makeLGBTextModel()), 0o644); err != nil {
		t.Fatal(err)
	}

	model, err := ParseLightGBM(path, "lgb-test")
	if err != nil {
		t.Fatal(err)
	}

	if model.Metadata.SourceFramework != "lightgbm" {
		t.Errorf("framework = %q, want lightgbm", model.Metadata.SourceFramework)
	}
	if model.Metadata.Task != "binary_classification" {
		t.Errorf("task = %q, want binary_classification", model.Metadata.Task)
	}
	if model.Metadata.NumFeatures != 3 {
		t.Errorf("num_features = %d, want 3", model.Metadata.NumFeatures)
	}

	te := model.Pipeline[0].TreeEnsemble
	if te.NumTrees != 2 {
		t.Errorf("num_trees = %d, want 2", te.NumTrees)
	}
}

func TestParseLightGBMJSON(t *testing.T) {
	data := map[string]any{
		"max_feature_idx": float64(2),
		"num_class":       float64(1),
		"objective":       "binary",
		"tree_info": []any{
			map[string]any{
				"tree_structure": map[string]any{
					"split_index":   float64(0),
					"split_feature": float64(0),
					"threshold":     float64(0.5),
					"default_left":  true,
					"left_child": map[string]any{
						"leaf_index": float64(0),
						"leaf_value": float64(0.3),
					},
					"right_child": map[string]any{
						"leaf_index": float64(1),
						"leaf_value": float64(-0.2),
					},
				},
			},
		},
	}

	path := writeJSONFile(t, data)
	model, err := ParseLightGBM(path, "lgb-json")
	if err != nil {
		t.Fatal(err)
	}

	if model.Metadata.SourceFramework != "lightgbm" {
		t.Errorf("framework = %q, want lightgbm", model.Metadata.SourceFramework)
	}
	te := model.Pipeline[0].TreeEnsemble
	if te.NumTrees != 1 {
		t.Errorf("num_trees = %d, want 1", te.NumTrees)
	}
	// 1 internal + 2 leaves = 3 nodes
	if len(te.Nodes.FeatureIndex) != 3 {
		t.Errorf("total nodes = %d, want 3", len(te.Nodes.FeatureIndex))
	}
}

func TestParseCatBoost(t *testing.T) {
	data := map[string]any{
		"model_info": map[string]any{
			"params": map[string]any{
				"loss_function": map[string]any{"type": "Logloss"},
			},
		},
		"features_info": map[string]any{
			"float_features": []any{
				map[string]any{"flat_feature_index": float64(0)},
				map[string]any{"flat_feature_index": float64(1)},
				map[string]any{"flat_feature_index": float64(2)},
			},
		},
		"scale_and_bias": []any{[]any{float64(1.0)}, []any{float64(0.0)}},
		"oblivious_trees": []any{
			map[string]any{
				"splits": []any{
					map[string]any{
						"float_feature_index": float64(0),
						"border":              float64(0.5),
					},
				},
				"leaf_values": []any{float64(0.3), float64(-0.2)},
			},
		},
	}

	path := writeJSONFile(t, data)
	model, err := ParseCatBoost(path, "cb-test")
	if err != nil {
		t.Fatal(err)
	}

	if model.Metadata.SourceFramework != "catboost" {
		t.Errorf("framework = %q, want catboost", model.Metadata.SourceFramework)
	}
	if model.Metadata.Task != "binary_classification" {
		t.Errorf("task = %q, want binary_classification", model.Metadata.Task)
	}
	if model.Metadata.NumFeatures != 3 {
		t.Errorf("num_features = %d, want 3", model.Metadata.NumFeatures)
	}

	te := model.Pipeline[0].TreeEnsemble
	if te.NumTrees != 1 {
		t.Errorf("num_trees = %d, want 1", te.NumTrees)
	}
	// Depth 1 oblivious tree → 1 internal + 2 leaves = 3 nodes
	if len(te.Nodes.FeatureIndex) != 3 {
		t.Errorf("total nodes = %d, want 3", len(te.Nodes.FeatureIndex))
	}
}

func TestValidateProducedModel(t *testing.T) {
	// Ensure every parser produces valid IR
	tests := []struct {
		name string
		fn   func() (string, error)
	}{
		{"xgboost", func() (string, error) {
			p := writeJSONFile(t, makeXGBJSON(3, "binary:logistic"))
			return p, nil
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path, err := tt.fn()
			if err != nil {
				t.Fatal(err)
			}
			model, err := ParseXGBoost(path, "validate-test")
			if err != nil {
				t.Fatal(err)
			}
			if err := model.Validate(); err != nil {
				t.Errorf("validation failed: %v", err)
			}
		})
	}
}
