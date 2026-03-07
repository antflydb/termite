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
	"fmt"
	"math"
	"sync"
)

// pipelinePredictor implements Predictor by running preprocessing stages
// followed by a model engine and output activation.
type pipelinePredictor struct {
	model         *TabularModel
	preprocessors []preprocessor
	treeEngine    *treeEngine
	linearEngine  *linearEngine
	svmEngine     *svmEngine

	// Pool for scratch float64 buffers used during preprocessing.
	// Only needed when preprocessors are present.
	scratchPool sync.Pool
}

// LoadPredictor loads a tabular model from a directory and returns a ready-to-use Predictor.
// Load-time optimizations (dead leaf elimination) are applied automatically.
func LoadPredictor(modelDir string) (Predictor, error) {
	m, err := LoadModelFromDir(modelDir)
	if err != nil {
		return nil, err
	}
	return NewPredictor(m)
}

// LoadPredictorWithOpts loads a tabular model with custom optimization options.
func LoadPredictorWithOpts(modelDir string, opts OptimizeOptions) (Predictor, error) {
	m, err := LoadModelFromDir(modelDir)
	if err != nil {
		return nil, err
	}
	return NewPredictorWithOpts(m, opts)
}

// OptimizeOptions controls load-time optimization passes.
type OptimizeOptions struct {
	// DeadLeafThreshold is the fraction of max leaf value below which leaves
	// are pruned. Set to 0 to disable. Default: 0.001.
	DeadLeafThreshold float64

	// SkipOptimization disables all load-time optimizations.
	SkipOptimization bool
}

// NewPredictor builds a Predictor from a parsed TabularModel.
// Default load-time optimizations are applied (dead leaf elimination with threshold 0.001).
func NewPredictor(m *TabularModel) (Predictor, error) {
	return NewPredictorWithOpts(m, OptimizeOptions{DeadLeafThreshold: 0.001})
}

// NewPredictorWithOpts builds a Predictor with custom optimization options.
func NewPredictorWithOpts(m *TabularModel, opts OptimizeOptions) (Predictor, error) {
	if !opts.SkipOptimization && opts.DeadLeafThreshold > 0 {
		optimizeModel(m, opts)
	}

	p := &pipelinePredictor{model: m}

	for _, stage := range m.Pipeline {
		switch stage.Type {
		case StageScaler:
			p.preprocessors = append(p.preprocessors, newScalerProcessor(stage.Scaler))
		case StageImputer:
			p.preprocessors = append(p.preprocessors, newImputerProcessor(stage.Imputer))
		case StageTreeEnsemble:
			p.treeEngine = newTreeEngine(stage.TreeEnsemble)
		case StageLinear:
			p.linearEngine = newLinearEngine(stage.Linear)
		case StageSVM:
			p.svmEngine = newSVMEngine(stage.SVM)
		}
	}

	if p.treeEngine == nil && p.linearEngine == nil && p.svmEngine == nil {
		return nil, fmt.Errorf("no model engine found in pipeline")
	}

	// Multiclass tree ensembles (interleaved trees per class) are not yet
	// implemented. Reject at construction time rather than producing wrong output.
	if p.treeEngine != nil && m.Output.NumOutputs > 1 {
		return nil, fmt.Errorf("multiclass tree ensembles (num_outputs=%d) are not yet supported", m.Output.NumOutputs)
	}

	// Set up scratch pool if we have preprocessors
	if len(p.preprocessors) > 0 {
		numFeatures := m.Metadata.NumFeatures
		p.scratchPool = sync.Pool{
			New: func() any {
				buf := make([]float64, numFeatures)
				return &buf
			},
		}
	}

	return p, nil
}

func (p *pipelinePredictor) Predict(ctx context.Context, features [][]float32) ([][]float32, error) {
	// Validate feature dimensions
	numFeatures := p.model.Metadata.NumFeatures
	for i, f := range features {
		if len(f) != numFeatures {
			return nil, fmt.Errorf("sample %d: expected %d features, got %d", i, numFeatures, len(f))
		}
	}

	// Fast batch path: tree models without preprocessing use SIMD-accelerated
	// batch prediction with vectorized leaf accumulation and activation.
	// Only valid for single-output models (binary/regression). Multiclass models
	// interleave trees across classes and need the per-sample path.
	if p.treeEngine != nil && len(p.preprocessors) == 0 && p.model.Output.NumOutputs <= 1 {
		scores := p.treeEngine.predictBatch32(features)
		applyActivation32(scores, p.model.Output.Activation)
		results := make([][]float32, len(scores))
		for i, s := range scores {
			results[i] = []float32{s}
		}
		return results, nil
	}

	// General path: per-sample prediction
	results := make([][]float32, len(features))
	for i, f := range features {
		if ctx != nil {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		r, err := p.PredictSingle(f)
		if err != nil {
			return nil, fmt.Errorf("sample %d: %w", i, err)
		}
		results[i] = r
	}
	return results, nil
}

func (p *pipelinePredictor) PredictSingle(features []float32) ([]float32, error) {
	numFeatures := p.model.Metadata.NumFeatures
	if len(features) != numFeatures {
		return nil, fmt.Errorf("expected %d features, got %d", numFeatures, len(features))
	}

	// Fast path: no preprocessing → work entirely in float32, zero extra allocations
	if len(p.preprocessors) == 0 {
		return p.predictFast(features), nil
	}

	// Slow path: convert to float64, preprocess, then predict
	return p.predictWithPreprocessing(features), nil
}

// predictFast is the zero-allocation hot path for models without preprocessing.
// Works entirely in float32 for maximum cache efficiency.
func (p *pipelinePredictor) predictFast(features []float32) []float32 {
	if p.treeEngine != nil {
		v := p.treeEngine.predictSingle32(features)
		out := []float32{v}
		applyActivation32(out, p.model.Output.Activation)
		return out
	}

	// Linear/SVM models — need float64 for math
	f64 := make([]float64, len(features))
	for i, v := range features {
		f64[i] = float64(v)
	}

	var raw []float64
	if p.linearEngine != nil {
		raw = p.linearEngine.predictSingle(f64)
	} else if p.svmEngine != nil {
		raw = p.svmEngine.predictSingle(f64)
	}

	applyActivation(raw, p.model.Output.Activation)
	out := make([]float32, len(raw))
	for i, v := range raw {
		out[i] = float32(v)
	}
	return out
}

// predictWithPreprocessing handles models that require feature scaling/imputation.
func (p *pipelinePredictor) predictWithPreprocessing(features []float32) []float32 {
	// Get scratch buffer from pool
	bufPtr := p.scratchPool.Get().(*[]float64)
	f64 := *bufPtr
	defer p.scratchPool.Put(bufPtr)

	for i, v := range features {
		f64[i] = float64(v)
	}

	for _, pp := range p.preprocessors {
		pp.apply(f64)
	}

	var raw []float64
	if p.treeEngine != nil {
		v := p.treeEngine.predictSingle(f64)
		raw = []float64{v}
	} else if p.linearEngine != nil {
		raw = p.linearEngine.predictSingle(f64)
	} else if p.svmEngine != nil {
		raw = p.svmEngine.predictSingle(f64)
	} else {
		return []float32{0}
	}

	applyActivation(raw, p.model.Output.Activation)

	out := make([]float32, len(raw))
	for i, v := range raw {
		out[i] = float32(v)
	}
	return out
}

func (p *pipelinePredictor) NumFeatures() int {
	return p.model.Metadata.NumFeatures
}

func (p *pipelinePredictor) NumOutputs() int {
	return p.model.Output.NumOutputs
}

func (p *pipelinePredictor) ModelMetadata() *Metadata {
	return &p.model.Metadata
}

func (p *pipelinePredictor) Close() error {
	return nil
}

// optimizeModel applies load-time optimizations to the model in-place.
func optimizeModel(m *TabularModel, opts OptimizeOptions) {
	for i := range m.Pipeline {
		stage := &m.Pipeline[i]
		if stage.TreeEnsemble != nil {
			if opts.DeadLeafThreshold > 0 {
				eliminated := deadLeafElimination(stage.TreeEnsemble, opts.DeadLeafThreshold)
				stage.TreeEnsemble.Annotations.DeadLeavesEliminated += eliminated
			}
			// Run threshold precision annotation if not already set
			if len(stage.TreeEnsemble.Annotations.ThresholdPrecision) == 0 {
				annotateThresholdPrecision(stage.TreeEnsemble)
			}
		}
	}
}

// deadLeafElimination prunes negligible leaves in-place.
// Leaves with |value| < thresholdFraction * max|leafValue| are zeroed.
// When both children are dead, the parent collapses into a zero-value leaf.
func deadLeafElimination(te *TreeEnsemble, thresholdFraction float64) int {
	nodes := &te.Nodes
	n := len(nodes.FeatureIndex)
	if n == 0 {
		return 0
	}

	var maxVal float64
	for i := 0; i < n; i++ {
		if nodes.FeatureIndex[i] < 0 {
			v := math.Abs(nodes.LeafValue[i])
			if v > maxVal {
				maxVal = v
			}
		}
	}
	if maxVal == 0 {
		return 0
	}

	absThreshold := thresholdFraction * maxVal
	eliminated := 0

	isDead := make([]bool, n)
	for i := 0; i < n; i++ {
		if nodes.FeatureIndex[i] < 0 && math.Abs(nodes.LeafValue[i]) < absThreshold {
			isDead[i] = true
			eliminated++
		}
	}

	for changed := true; changed; {
		changed = false
		for i := 0; i < n; i++ {
			if nodes.FeatureIndex[i] >= 0 && !isDead[i] {
				left := nodes.LeftChild[i]
				right := nodes.RightChild[i]
				if left >= 0 && right >= 0 && isDead[left] && isDead[right] {
					nodes.FeatureIndex[i] = -1
					nodes.LeftChild[i] = -1
					nodes.RightChild[i] = -1
					nodes.LeafValue[i] = 0.0
					isDead[i] = true
					changed = true
				}
			}
		}
	}

	return eliminated
}

// annotateThresholdPrecision determines the minimum precision needed for each
// feature's thresholds and stores the result in annotations.
// Precision levels: "i8" (int8), "i16" (int16), "f16" (float16), "f32" (default).
func annotateThresholdPrecision(te *TreeEnsemble) {
	nodes := &te.Nodes
	n := len(nodes.FeatureIndex)
	if n == 0 || te.NumFeatures == 0 {
		return
	}

	// Collect thresholds per feature
	featureThresholds := make(map[int][]float64)
	for i := 0; i < n; i++ {
		fi := nodes.FeatureIndex[i]
		if fi >= 0 {
			featureThresholds[int(fi)] = append(featureThresholds[int(fi)], nodes.Threshold[i])
		}
	}

	precision := make([]string, te.NumFeatures)
	for i := range precision {
		precision[i] = "f32"
	}

	for fi, thresholds := range featureThresholds {
		if fi < 0 || fi >= te.NumFeatures {
			continue
		}
		precision[fi] = minPrecision(thresholds)
	}

	te.Annotations.ThresholdPrecision = precision
}

// minPrecision determines the minimum precision for a set of threshold values.
func minPrecision(values []float64) string {
	if len(values) == 0 {
		return "f32"
	}
	allInt := true
	minV := values[0]
	maxV := values[0]

	for _, v := range values {
		if v != float64(int64(v)) {
			allInt = false
		}
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}

	if allInt {
		if minV >= -128 && maxV <= 127 {
			return "i8"
		}
		if minV >= -32768 && maxV <= 32767 {
			return "i16"
		}
	}

	// Check float16: values that round-trip through float32 without
	// significant loss (relative tolerance 1e-3) and fit in f16 range.
	allF16 := true
	for _, v := range values {
		if v != 0 {
			ratio := v / float64(float32(v))
			if ratio < 0.999 || ratio > 1.001 {
				allF16 = false
				break
			}
		}
	}
	// For f16 safety, also check range
	if allF16 && minV >= -65504 && maxV <= 65504 {
		return "f16"
	}

	return "f32"
}


