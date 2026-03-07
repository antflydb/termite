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

// Package tabular provides inference for traditional ML models (tree ensembles,
// linear models, SVMs) using a SIMD-accelerated pure Go engine.
//
// Models are represented using a framework-agnostic intermediate representation
// (IR) that can be produced by converting XGBoost, LightGBM, sklearn, or CatBoost
// models via the termite-convert tool.
package tabular

import "context"

// TaskType describes the prediction task.
type TaskType string

const (
	TaskRegression           TaskType = "regression"
	TaskBinaryClassification TaskType = "binary_classification"
	TaskMulticlass           TaskType = "multiclass"
	TaskRanking              TaskType = "ranking"
)

// ActivationType describes the output activation function.
type ActivationType string

const (
	ActivationIdentity ActivationType = "identity"
	ActivationSigmoid  ActivationType = "sigmoid"
	ActivationSoftmax  ActivationType = "softmax"
	ActivationExp      ActivationType = "exp"
)

// StageType identifies a pipeline stage.
type StageType string

const (
	StageTreeEnsemble StageType = "tree_ensemble"
	StageLinear       StageType = "linear"
	StageSVM          StageType = "svm"
	StageScaler       StageType = "scaler"
	StageImputer      StageType = "imputer"
)

// TabularModel is the top-level IR for a traditional ML model.
type TabularModel struct {
	SchemaVersion int       `json:"schema_version"`
	Metadata      Metadata  `json:"metadata"`
	Pipeline      []Stage   `json:"pipeline"`
	Output        OutputCfg `json:"output"`
}

// Metadata describes the model's provenance and schema.
type Metadata struct {
	Name            string   `json:"name"`
	SourceFramework string   `json:"source_framework"`
	SourceVersion   string   `json:"source_version,omitempty"`
	Task            TaskType `json:"task"`
	NumFeatures     int      `json:"num_features"`
	NumClasses      int      `json:"num_classes,omitempty"`
	FeatureNames    []string `json:"feature_names,omitempty"`
	FeatureTypes    []string `json:"feature_types,omitempty"`
	CreatedAt       string   `json:"created_at,omitempty"`
}

// OutputCfg describes post-processing applied after the model stage.
type OutputCfg struct {
	Activation ActivationType `json:"activation"`
	NumOutputs int            `json:"num_outputs"`
}

// Stage is a single step in the inference pipeline.
// Exactly one of the typed fields will be non-nil.
type Stage struct {
	Type         StageType     `json:"type"`
	TreeEnsemble *TreeEnsemble `json:"tree_ensemble,omitempty"`
	Linear       *LinearModel  `json:"linear,omitempty"`
	SVM          *SVMModel     `json:"svm,omitempty"`
	Scaler       *Scaler       `json:"scaler,omitempty"`
	Imputer      *Imputer      `json:"imputer,omitempty"`
}

// TreeNodes stores tree data in a struct-of-arrays (SoA) layout for
// cache-friendly, SIMD-amenable traversal. All slices are the same length
// (total node count across all trees). Tree boundaries are given by TreeStarts.
type TreeNodes struct {
	FeatureIndex []int32   `json:"feature_index"` // -1 = leaf node
	Threshold    []float64 `json:"threshold"`
	LeftChild    []int32   `json:"left_child"`  // -1 = none
	RightChild   []int32   `json:"right_child"` // -1 = none
	LeafValue    []float64 `json:"leaf_value"`  // non-zero only for leaves
	DefaultLeft  []bool    `json:"default_left"`
	TreeStarts   []int32   `json:"tree_starts"` // index of each tree's root node
}

// TreeAnnotations holds optional optimiser metadata.
type TreeAnnotations struct {
	ThresholdPrecision   []string `json:"threshold_precision,omitempty"`
	DeadLeavesEliminated int      `json:"dead_leaves_eliminated,omitempty"`
	BranchOrder          string   `json:"branch_order,omitempty"`
}

// TreeEnsemble represents a gradient-boosted or random-forest model.
type TreeEnsemble struct {
	Objective   string          `json:"objective"`
	BaseScore   float64         `json:"base_score"`
	NumTrees    int             `json:"num_trees"`
	NumFeatures int             `json:"num_features"`
	MaxDepth    int             `json:"max_depth"`
	Nodes       TreeNodes       `json:"nodes"`
	Annotations TreeAnnotations `json:"annotations,omitempty"`
}

// LinearModel represents a logistic/linear regression.
// Activation is handled by Output.Activation in the pipeline, not here.
type LinearModel struct {
	Weights [][]float64 `json:"weights"` // [num_outputs][num_features]
	Biases  []float64   `json:"biases"`  // [num_outputs]
}

// SVMModel represents a support vector machine.
type SVMModel struct {
	Kernel          string      `json:"kernel"` // rbf, linear, poly, sigmoid
	SupportVectors  [][]float64 `json:"support_vectors"`
	DualCoefficients [][]float64 `json:"dual_coefficients"`
	Intercept       []float64   `json:"intercept"`
	Gamma           float64     `json:"gamma,omitempty"`
	Coef0           float64     `json:"coef0,omitempty"`
	Degree          int         `json:"degree,omitempty"`
}

// Scaler represents feature standardisation / normalisation.
type Scaler struct {
	Method string    `json:"method"` // standard, minmax, robust, maxabs
	Center []float64 `json:"center,omitempty"`
	Scale  []float64 `json:"scale,omitempty"`
	Min    []float64 `json:"min,omitempty"`
	Max    []float64 `json:"max,omitempty"`
}

// Imputer fills missing (NaN) values.
type Imputer struct {
	Strategy   string    `json:"strategy"` // mean, median, most_frequent, constant
	FillValues []float64 `json:"fill_values"`
}

// Predictor is the public interface for tabular model inference.
type Predictor interface {
	// Predict runs inference on a batch of samples.
	// features: [batch_size][num_features]
	// Returns: [batch_size][num_outputs]
	Predict(ctx context.Context, features [][]float32) ([][]float32, error)

	// PredictSingle runs inference on one sample (optimised fast path).
	PredictSingle(features []float32) ([]float32, error)

	// NumFeatures returns the expected input width.
	NumFeatures() int

	// NumOutputs returns the output width.
	NumOutputs() int

	// Metadata returns model metadata.
	ModelMetadata() *Metadata

	// Close releases resources.
	Close() error
}
