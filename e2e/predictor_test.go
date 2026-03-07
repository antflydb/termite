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
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/termite"
	"github.com/antflydb/termite/pkg/termite/lib/tabular"
	"go.uber.org/zap/zaptest"
)

// TestPredictorModels is the E2E test suite for tabular predictor models.
// Unlike other E2E tests that require ONNX, predictor tests use pure Go
// inference and can run without build tags.
func TestPredictorModels(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E tests in short mode")
	}

	// Create test models in the models directory
	predictorsDir := filepath.Join(getTestModelsDir(), "predictors")
	if err := os.MkdirAll(predictorsDir, 0755); err != nil {
		t.Fatalf("Failed to create predictors dir: %v", err)
	}

	t.Run("XGBoostBinaryClassifier", func(t *testing.T) {
		runPredictorE2E(t, predictorsDir, buildXGBoostBinaryModel())
	})

	t.Run("LinearRegressor", func(t *testing.T) {
		runPredictorE2E(t, predictorsDir, buildLinearRegressionModel())
	})

	t.Run("TreeWithScaler", func(t *testing.T) {
		runPredictorE2E(t, predictorsDir, buildTreeWithScalerModel())
	})

	t.Run("SVMClassifier", func(t *testing.T) {
		runPredictorE2E(t, predictorsDir, buildSVMModel())
	})
}

const e2eOwner = "e2e-test"

func runPredictorE2E(t *testing.T, predictorsDir string, model *tabular.TabularModel) {
	t.Helper()

	// Write model to disk (owner/model-name structure required by discovery)
	modelDir := filepath.Join(predictorsDir, e2eOwner, model.Metadata.Name)
	apiModelName := e2eOwner + "/" + model.Metadata.Name
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatalf("Failed to create model dir: %v", err)
	}

	modelData, err := json.MarshalIndent(model, "", "  ")
	if err != nil {
		t.Fatalf("Failed to marshal model: %v", err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "tabular_model.json"), modelData, 0644); err != nil {
		t.Fatalf("Failed to write model: %v", err)
	}

	// Also write a model_manifest.json
	manifest := map[string]any{
		"model_type":   "predictor",
		"capabilities": []string{"tabular"},
	}
	manifestData, _ := json.MarshalIndent(manifest, "", "  ")
	if err := os.WriteFile(filepath.Join(modelDir, "model_manifest.json"), manifestData, 0644); err != nil {
		t.Fatalf("Failed to write manifest: %v", err)
	}

	// Start test server
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	logger := zaptest.NewLogger(t)
	port := findAvailablePort(t)
	serverURL := fmt.Sprintf("http://localhost:%d", port)

	config := termite.Config{
		ApiUrl:    serverURL,
		ModelsDir: getTestModelsDir(),
	}

	serverCtx, serverCancel := context.WithCancel(ctx)
	readyC := make(chan struct{})
	serverDone := make(chan struct{})

	go func() {
		defer close(serverDone)
		termite.RunAsTermite(serverCtx, logger, config, readyC)
	}()

	// Wait for readiness
	select {
	case <-readyC:
		t.Log("Server is ready")
	case <-time.After(10 * time.Second):
		serverCancel()
		t.Fatalf("Timeout waiting for server")
	}
	defer func() {
		serverCancel()
		<-serverDone
	}()

	// Create client
	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Test: List models includes our predictor
	t.Run("ListModels", func(t *testing.T) {
		models, err := termiteClient.ListModels(ctx)
		if err != nil {
			t.Fatalf("ListModels failed: %v", err)
		}

		if models.Predictors == nil {
			t.Fatal("Expected Predictors in model list, got nil")
		}

		if _, ok := models.Predictors[apiModelName]; !ok {
			t.Errorf("Expected model %q in predictors list, got: %v",
				apiModelName, models.Predictors)
		}
	})

	// Test: Predict with valid input
	t.Run("Predict", func(t *testing.T) {
		numFeatures := model.Metadata.NumFeatures
		input := make([][]float32, 3)
		for i := range input {
			input[i] = make([]float32, numFeatures)
			for j := range input[i] {
				input[i][j] = float32(i+1) * 0.1 * float32(j+1)
			}
		}

		resp, err := termiteClient.Predict(ctx, apiModelName, input)
		if err != nil {
			t.Fatalf("Predict failed: %v", err)
		}

		if resp.Model != apiModelName {
			t.Errorf("Expected model %q, got %q", apiModelName, resp.Model)
		}

		if len(resp.Predictions) != 3 {
			t.Fatalf("Expected 3 predictions, got %d", len(resp.Predictions))
		}

		// Verify predictions are valid numbers
		for i, pred := range resp.Predictions {
			if len(pred) != model.Output.NumOutputs {
				t.Errorf("Prediction %d: expected %d outputs, got %d",
					i, model.Output.NumOutputs, len(pred))
			}
			for j, v := range pred {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("Prediction %d output %d: invalid value %f", i, j, v)
				}
			}
		}

		// For sigmoid outputs, verify they're in [0, 1]
		if model.Output.Activation == tabular.ActivationSigmoid {
			for i, pred := range resp.Predictions {
				for j, v := range pred {
					if v < 0 || v > 1 {
						t.Errorf("Prediction %d output %d: sigmoid value %.4f not in [0,1]",
							i, j, v)
					}
				}
			}
		}

		t.Logf("Task: %s, Predictions: %v", resp.Task, resp.Predictions)
	})

	// Test: Wrong feature count returns error
	t.Run("WrongFeatureCount", func(t *testing.T) {
		badInput := [][]float32{{1.0, 2.0}} // Wrong number of features
		_, err := termiteClient.Predict(ctx, apiModelName, badInput)
		if err == nil {
			t.Error("Expected error for wrong feature count, got nil")
		}
	})
}

// ============================================================================
// Test model builders
// ============================================================================

func buildXGBoostBinaryModel() *tabular.TabularModel {
	// Simple 2-tree binary classifier with 5 features
	return &tabular.TabularModel{
		SchemaVersion: 1,
		Metadata: tabular.Metadata{
			Name:            "e2e-xgboost-binary",
			SourceFramework: "xgboost",
			Task:            tabular.TaskBinaryClassification,
			NumFeatures:     5,
		},
		Pipeline: []tabular.Stage{
			{
				Type: tabular.StageTreeEnsemble,
				TreeEnsemble: &tabular.TreeEnsemble{
					Objective:   "binary:logistic",
					BaseScore:   0.0,
					NumTrees:    2,
					NumFeatures: 5,
					MaxDepth:    2,
					Nodes: tabular.TreeNodes{
						// Tree 0: split on f0 at 0.5, then f1 at 0.3
						FeatureIndex: []int32{0, 1, -1, -1, -1, 2, -1, -1, -1},
						Threshold:    []float64{0.5, 0.3, 0, 0, 0, 1.0, 0, 0, 0},
						LeftChild:    []int32{1, 3, -1, -1, -1, 7, -1, -1, -1},
						RightChild:   []int32{2, 4, -1, -1, -1, 8, -1, -1, -1},
						LeafValue:    []float64{0, 0, 0.2, -0.1, 0.15, 0, -0.3, 0.1, -0.05},
						DefaultLeft:  []bool{true, true, false, false, false, true, false, false, false},
						TreeStarts:   []int32{0, 5},
					},
				},
			},
		},
		Output: tabular.OutputCfg{
			Activation: tabular.ActivationSigmoid,
			NumOutputs: 1,
		},
	}
}

func buildLinearRegressionModel() *tabular.TabularModel {
	return &tabular.TabularModel{
		SchemaVersion: 1,
		Metadata: tabular.Metadata{
			Name:            "e2e-linear-regressor",
			SourceFramework: "sklearn",
			Task:            tabular.TaskRegression,
			NumFeatures:     3,
		},
		Pipeline: []tabular.Stage{
			{
				Type: tabular.StageLinear,
				Linear: &tabular.LinearModel{
					Weights: [][]float64{{1.5, -0.5, 2.0}},
					Biases:  []float64{0.1},
				},
			},
		},
		Output: tabular.OutputCfg{
			Activation: tabular.ActivationIdentity,
			NumOutputs: 1,
		},
	}
}

func buildTreeWithScalerModel() *tabular.TabularModel {
	return &tabular.TabularModel{
		SchemaVersion: 1,
		Metadata: tabular.Metadata{
			Name:            "e2e-tree-scaled",
			SourceFramework: "sklearn",
			Task:            tabular.TaskBinaryClassification,
			NumFeatures:     3,
		},
		Pipeline: []tabular.Stage{
			{
				Type: tabular.StageScaler,
				Scaler: &tabular.Scaler{
					Method: "standard",
					Center: []float64{0.5, 1.0, 0.0},
					Scale:  []float64{0.1, 0.2, 0.5},
				},
			},
			{
				Type: tabular.StageTreeEnsemble,
				TreeEnsemble: &tabular.TreeEnsemble{
					Objective:   "binary_logistic",
					BaseScore:   0.0,
					NumTrees:    1,
					NumFeatures: 3,
					MaxDepth:    1,
					Nodes: tabular.TreeNodes{
						FeatureIndex: []int32{0, -1, -1},
						Threshold:    []float64{0.5, 0, 0},
						LeftChild:    []int32{1, -1, -1},
						RightChild:   []int32{2, -1, -1},
						LeafValue:    []float64{0, 0.5, -0.5},
						DefaultLeft:  []bool{true, false, false},
						TreeStarts:   []int32{0},
					},
				},
			},
		},
		Output: tabular.OutputCfg{
			Activation: tabular.ActivationSigmoid,
			NumOutputs: 1,
		},
	}
}

func buildSVMModel() *tabular.TabularModel {
	return &tabular.TabularModel{
		SchemaVersion: 1,
		Metadata: tabular.Metadata{
			Name:            "e2e-svm-classifier",
			SourceFramework: "sklearn",
			Task:            tabular.TaskBinaryClassification,
			NumFeatures:     4,
		},
		Pipeline: []tabular.Stage{
			{
				Type: tabular.StageSVM,
				SVM: &tabular.SVMModel{
					Kernel: "linear",
					SupportVectors: [][]float64{
						{1, 0, 0, 0},
						{0, 1, 0, 0},
						{0, 0, 1, 0},
					},
					DualCoefficients: [][]float64{
						{0.5, 0.3, -0.8},
					},
					Intercept: []float64{0.1},
				},
			},
		},
		Output: tabular.OutputCfg{
			Activation: tabular.ActivationSigmoid,
			NumOutputs: 1,
		},
	}
}
