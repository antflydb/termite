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
	"math"
	"testing"
)

func TestSVMLinearKernel(t *testing.T) {
	// Simple linear SVM: w=[1,2], b=-0.5
	// Decision = dot([1,2], features) - 0.5
	model := &TabularModel{
		SchemaVersion: 1,
		Metadata: Metadata{
			Name:        "test-svm-linear",
			Task:        TaskBinaryClassification,
			NumFeatures: 2,
		},
		Pipeline: []Stage{
			{
				Type: StageSVM,
				SVM: &SVMModel{
					Kernel: "linear",
					SupportVectors: [][]float64{
						{1, 0}, // sv1
						{0, 1}, // sv2
					},
					DualCoefficients: [][]float64{
						{1.0, 2.0}, // coefficients
					},
					Intercept: []float64{-0.5},
				},
			},
		},
		Output: OutputCfg{
			Activation: ActivationSigmoid,
			NumOutputs: 1,
		},
	}

	pred, err := NewPredictorWithOpts(model, OptimizeOptions{SkipOptimization: true})
	if err != nil {
		t.Fatal(err)
	}

	// features = [1, 1]
	// kernel(features, sv1=[1,0]) = dot([1,1],[1,0]) = 1
	// kernel(features, sv2=[0,1]) = dot([1,1],[0,1]) = 1
	// decision = 1.0*1 + 2.0*1 - 0.5 = 2.5
	// sigmoid(2.5) ≈ 0.924
	result, err := pred.PredictSingle([]float32{1, 1})
	if err != nil {
		t.Fatal(err)
	}

	if len(result) != 1 {
		t.Fatalf("expected 1 output, got %d", len(result))
	}

	expected := float32(1.0 / (1.0 + math.Exp(-2.5)))
	if diff := math.Abs(float64(result[0] - expected)); diff > 0.01 {
		t.Errorf("expected ~%.4f, got %.4f (diff %.6f)", expected, result[0], diff)
	}
}

func TestSVMRBFKernel(t *testing.T) {
	model := &TabularModel{
		SchemaVersion: 1,
		Metadata: Metadata{
			Name:        "test-svm-rbf",
			Task:        TaskBinaryClassification,
			NumFeatures: 2,
		},
		Pipeline: []Stage{
			{
				Type: StageSVM,
				SVM: &SVMModel{
					Kernel: "rbf",
					Gamma:  0.5,
					SupportVectors: [][]float64{
						{0, 0},
						{1, 1},
					},
					DualCoefficients: [][]float64{
						{1.0, -1.0},
					},
					Intercept: []float64{0.0},
				},
			},
		},
		Output: OutputCfg{
			Activation: ActivationIdentity,
			NumOutputs: 1,
		},
	}

	pred, err := NewPredictorWithOpts(model, OptimizeOptions{SkipOptimization: true})
	if err != nil {
		t.Fatal(err)
	}

	// features = [0, 0], close to sv1, far from sv2
	// kernel(f, sv1) = exp(-0.5 * 0) = 1.0
	// kernel(f, sv2) = exp(-0.5 * 2) = exp(-1) ≈ 0.368
	// decision = 1.0*1.0 + (-1.0)*0.368 = 0.632
	result, err := pred.PredictSingle([]float32{0, 0})
	if err != nil {
		t.Fatal(err)
	}

	expected := 1.0 - math.Exp(-1.0) // ≈ 0.632
	if diff := math.Abs(float64(result[0]) - expected); diff > 0.01 {
		t.Errorf("expected ~%.4f, got %.4f", expected, result[0])
	}
}

func TestSVMBatch(t *testing.T) {
	model := &TabularModel{
		SchemaVersion: 1,
		Metadata: Metadata{
			Name:        "test-svm-batch",
			Task:        TaskBinaryClassification,
			NumFeatures: 2,
		},
		Pipeline: []Stage{
			{
				Type: StageSVM,
				SVM: &SVMModel{
					Kernel: "linear",
					SupportVectors: [][]float64{
						{1, 0},
					},
					DualCoefficients: [][]float64{
						{1.0},
					},
					Intercept: []float64{0.0},
				},
			},
		},
		Output: OutputCfg{
			Activation: ActivationIdentity,
			NumOutputs: 1,
		},
	}

	pred, err := NewPredictorWithOpts(model, OptimizeOptions{SkipOptimization: true})
	if err != nil {
		t.Fatal(err)
	}

	batch := [][]float32{
		{1, 0},  // dot with sv=[1,0] = 1
		{0, 1},  // dot with sv=[1,0] = 0
		{2, 3},  // dot with sv=[1,0] = 2
	}

	results, err := pred.Predict(context.Background(), batch)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	expectedVals := []float32{1.0, 0.0, 2.0}
	for i, exp := range expectedVals {
		if diff := math.Abs(float64(results[i][0] - exp)); diff > 0.01 {
			t.Errorf("sample %d: expected %.4f, got %.4f", i, exp, results[i][0])
		}
	}
}
