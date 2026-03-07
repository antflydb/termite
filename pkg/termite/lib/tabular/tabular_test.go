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

package tabular

import (
	"context"
	"encoding/json"
	"math"
	"strings"
	"testing"
)

// Helper to build a minimal two-tree ensemble:
//
//	Tree 0 (root=0):
//	  node 0: feature[0] <= 0.5 ? left=1, right=2
//	  node 1: leaf = 0.3
//	  node 2: leaf = -0.1
//
//	Tree 1 (root=3):
//	  node 3: feature[1] <= 1.0 ? left=4, right=5
//	  node 4: leaf = 0.2
//	  node 5: leaf = -0.2
func twoTreeModel() *TabularModel {
	return &TabularModel{
		SchemaVersion: 1,
		Metadata: Metadata{
			Name:            "test-model",
			SourceFramework: "test",
			Task:            TaskBinaryClassification,
			NumFeatures:     2,
		},
		Pipeline: []Stage{
			{
				Type: StageTreeEnsemble,
				TreeEnsemble: &TreeEnsemble{
					Objective:   "binary_logistic",
					BaseScore:   0.0,
					NumTrees:    2,
					NumFeatures: 2,
					MaxDepth:    1,
					Nodes: TreeNodes{
						FeatureIndex: []int32{0, -1, -1, 1, -1, -1},
						Threshold:    []float64{0.5, 0, 0, 1.0, 0, 0},
						LeftChild:    []int32{1, -1, -1, 4, -1, -1},
						RightChild:   []int32{2, -1, -1, 5, -1, -1},
						LeafValue:    []float64{0, 0.3, -0.1, 0, 0.2, -0.2},
						DefaultLeft:  []bool{true, false, false, true, false, false},
						TreeStarts:   []int32{0, 3},
					},
				},
			},
		},
		Output: OutputCfg{
			Activation: ActivationSigmoid,
			NumOutputs: 1,
		},
	}
}

func TestTreeEnsembleBasic(t *testing.T) {
	m := twoTreeModel()
	pred, err := NewPredictor(m)
	if err != nil {
		t.Fatal(err)
	}

	// feature[0]=0.3 (<=0.5 -> leaf 0.3), feature[1]=0.5 (<=1.0 -> leaf 0.2)
	// raw = 0 + 0.3 + 0.2 = 0.5, sigmoid(0.5) ≈ 0.622
	out, err := pred.PredictSingle([]float32{0.3, 0.5})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 {
		t.Fatalf("expected 1 output, got %d", len(out))
	}
	expected := float32(1.0 / (1.0 + math.Exp(-0.5)))
	if diff := math.Abs(float64(out[0] - expected)); diff > 1e-5 {
		t.Errorf("expected %.6f, got %.6f (diff=%.6f)", expected, out[0], diff)
	}
}

func TestTreeEnsembleRightBranch(t *testing.T) {
	m := twoTreeModel()
	pred, err := NewPredictor(m)
	if err != nil {
		t.Fatal(err)
	}

	// feature[0]=0.7 (>0.5 -> leaf -0.1), feature[1]=1.5 (>1.0 -> leaf -0.2)
	// raw = 0 + (-0.1) + (-0.2) = -0.3, sigmoid(-0.3) ≈ 0.4256
	out, err := pred.PredictSingle([]float32{0.7, 1.5})
	if err != nil {
		t.Fatal(err)
	}
	expected := float32(1.0 / (1.0 + math.Exp(0.3)))
	if diff := math.Abs(float64(out[0] - expected)); diff > 1e-5 {
		t.Errorf("expected %.6f, got %.6f", expected, out[0])
	}
}

func TestTreeEnsembleBatch(t *testing.T) {
	m := twoTreeModel()
	pred, err := NewPredictor(m)
	if err != nil {
		t.Fatal(err)
	}

	batch := [][]float32{
		{0.3, 0.5},
		{0.7, 1.5},
	}
	results, err := pred.Predict(context.Background(), batch)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	// Just verify it runs without error and returns valid probabilities
	for i, r := range results {
		if r[0] < 0 || r[0] > 1 {
			t.Errorf("result[%d] = %.4f, not a valid probability", i, r[0])
		}
	}
}

func TestNaNDefaultLeft(t *testing.T) {
	m := twoTreeModel()
	pred, err := NewPredictor(m)
	if err != nil {
		t.Fatal(err)
	}

	// NaN feature[0] with default_left=true -> left child -> leaf 0.3
	nan := float32(math.NaN())
	out, err := pred.PredictSingle([]float32{nan, 0.5})
	if err != nil {
		t.Fatal(err)
	}
	// Same as left branch: raw = 0.3 + 0.2 = 0.5
	expected := float32(1.0 / (1.0 + math.Exp(-0.5)))
	if diff := math.Abs(float64(out[0] - expected)); diff > 1e-5 {
		t.Errorf("expected %.6f, got %.6f", expected, out[0])
	}
}

func TestLinearModel(t *testing.T) {
	m := &TabularModel{
		SchemaVersion: 1,
		Metadata: Metadata{
			Name:            "linear-test",
			SourceFramework: "test",
			Task:            TaskRegression,
			NumFeatures:     3,
		},
		Pipeline: []Stage{
			{
				Type: StageLinear,
				Linear: &LinearModel{
					Weights: [][]float64{{1.0, 2.0, 3.0}},
					Biases:  []float64{0.5},
				},
			},
		},
		Output: OutputCfg{
			Activation: ActivationIdentity,
			NumOutputs: 1,
		},
	}

	pred, err := NewPredictor(m)
	if err != nil {
		t.Fatal(err)
	}

	// 1*1 + 2*2 + 3*3 + 0.5 = 1 + 4 + 9 + 0.5 = 14.5
	out, err := pred.PredictSingle([]float32{1.0, 2.0, 3.0})
	if err != nil {
		t.Fatal(err)
	}
	if diff := math.Abs(float64(out[0]) - 14.5); diff > 1e-5 {
		t.Errorf("expected 14.5, got %.6f", out[0])
	}
}

func TestScalerPreprocessing(t *testing.T) {
	m := &TabularModel{
		SchemaVersion: 1,
		Metadata: Metadata{
			Name:            "scaler-test",
			SourceFramework: "test",
			Task:            TaskRegression,
			NumFeatures:     2,
		},
		Pipeline: []Stage{
			{
				Type: StageScaler,
				Scaler: &Scaler{
					Method: "standard",
					Center: []float64{10.0, 20.0},
					Scale:  []float64{2.0, 5.0},
				},
			},
			{
				Type: StageLinear,
				Linear: &LinearModel{
					Weights: [][]float64{{1.0, 1.0}},
					Biases:  []float64{0.0},
				},
			},
		},
		Output: OutputCfg{
			Activation: ActivationIdentity,
			NumOutputs: 1,
		},
	}

	pred, err := NewPredictor(m)
	if err != nil {
		t.Fatal(err)
	}

	// After scaling: [(12-10)/2, (25-20)/5] = [1.0, 1.0]
	// Linear: 1*1 + 1*1 + 0 = 2.0
	out, err := pred.PredictSingle([]float32{12.0, 25.0})
	if err != nil {
		t.Fatal(err)
	}
	if diff := math.Abs(float64(out[0]) - 2.0); diff > 1e-5 {
		t.Errorf("expected 2.0, got %.6f", out[0])
	}
}

func TestImputerPreprocessing(t *testing.T) {
	m := &TabularModel{
		SchemaVersion: 1,
		Metadata: Metadata{
			Name:            "imputer-test",
			SourceFramework: "test",
			Task:            TaskRegression,
			NumFeatures:     2,
		},
		Pipeline: []Stage{
			{
				Type: StageImputer,
				Imputer: &Imputer{
					Strategy:   "median",
					FillValues: []float64{5.0, 10.0},
				},
			},
			{
				Type: StageLinear,
				Linear: &LinearModel{
					Weights: [][]float64{{1.0, 1.0}},
					Biases:  []float64{0.0},
				},
			},
		},
		Output: OutputCfg{
			Activation: ActivationIdentity,
			NumOutputs: 1,
		},
	}

	pred, err := NewPredictor(m)
	if err != nil {
		t.Fatal(err)
	}

	// NaN imputed to [5.0, 10.0], linear: 5+10 = 15
	nan := float32(math.NaN())
	out, err := pred.PredictSingle([]float32{nan, nan})
	if err != nil {
		t.Fatal(err)
	}
	if diff := math.Abs(float64(out[0]) - 15.0); diff > 1e-5 {
		t.Errorf("expected 15.0, got %.6f", out[0])
	}
}

func TestJSONRoundTrip(t *testing.T) {
	m := twoTreeModel()
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}

	parsed, err := ParseModel(data)
	if err != nil {
		t.Fatal(err)
	}

	if parsed.Metadata.Name != m.Metadata.Name {
		t.Errorf("name mismatch: %s vs %s", parsed.Metadata.Name, m.Metadata.Name)
	}
	if parsed.Pipeline[0].TreeEnsemble.NumTrees != m.Pipeline[0].TreeEnsemble.NumTrees {
		t.Error("num_trees mismatch")
	}
}

func TestValidationErrors(t *testing.T) {
	tests := []struct {
		name    string
		modify  func(m *TabularModel)
		wantErr string
	}{
		{
			name:    "no name",
			modify:  func(m *TabularModel) { m.Metadata.Name = "" },
			wantErr: "missing name",
		},
		{
			name:    "no features",
			modify:  func(m *TabularModel) { m.Metadata.NumFeatures = 0 },
			wantErr: "num_features",
		},
		{
			name:    "no pipeline",
			modify:  func(m *TabularModel) { m.Pipeline = nil },
			wantErr: "at least one",
		},
		{
			name:    "no outputs",
			modify:  func(m *TabularModel) { m.Output.NumOutputs = 0 },
			wantErr: "num_outputs",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := twoTreeModel()
			tt.modify(m)
			data, _ := json.Marshal(m)
			_, err := ParseModel(data)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("expected error containing %q, got %q", tt.wantErr, err.Error())
			}
		})
	}
}

func TestFeatureCountMismatch(t *testing.T) {
	m := twoTreeModel()
	pred, err := NewPredictor(m)
	if err != nil {
		t.Fatal(err)
	}

	_, err = pred.PredictSingle([]float32{1.0}) // only 1 feature, need 2
	if err == nil {
		t.Fatal("expected error for wrong feature count")
	}
}

