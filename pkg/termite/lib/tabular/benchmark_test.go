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
	"math/rand"
	"testing"
)

// buildBenchModel creates a realistic tree ensemble for benchmarking.
// Parameters match real-world XGBoost/LightGBM configurations.
func buildBenchModel(numTrees, depth, numFeatures int) *TabularModel {
	rng := rand.New(rand.NewSource(42))

	// Complete binary tree: 2^(d+1) - 1 nodes per tree
	nodesPerTree := (1 << (depth + 1)) - 1
	totalNodes := numTrees * nodesPerTree

	featureIndex := make([]int32, totalNodes)
	threshold := make([]float64, totalNodes)
	leftChild := make([]int32, totalNodes)
	rightChild := make([]int32, totalNodes)
	leafValue := make([]float64, totalNodes)
	defaultLeft := make([]bool, totalNodes)
	treeStarts := make([]int32, numTrees)

	for t := 0; t < numTrees; t++ {
		base := t * nodesPerTree
		treeStarts[t] = int32(base)

		for i := 0; i < nodesPerTree; i++ {
			nodeIdx := base + i
			left := 2*i + 1
			right := 2*i + 2

			if left >= nodesPerTree {
				featureIndex[nodeIdx] = -1
				leftChild[nodeIdx] = -1
				rightChild[nodeIdx] = -1
				leafValue[nodeIdx] = rng.Float64()*0.2 - 0.1
			} else {
				featureIndex[nodeIdx] = int32(rng.Intn(numFeatures))
				threshold[nodeIdx] = rng.Float64()
				leftChild[nodeIdx] = int32(base + left)
				rightChild[nodeIdx] = int32(base + right)
				defaultLeft[nodeIdx] = rng.Float64() < 0.5
			}
		}
	}

	return &TabularModel{
		SchemaVersion: 1,
		Metadata: Metadata{
			Name:            "bench-model",
			SourceFramework: "xgboost",
			Task:            TaskBinaryClassification,
			NumFeatures:     numFeatures,
		},
		Pipeline: []Stage{
			{
				Type: StageTreeEnsemble,
				TreeEnsemble: &TreeEnsemble{
					Objective:   "binary_logistic",
					BaseScore:   0.0,
					NumTrees:    numTrees,
					NumFeatures: numFeatures,
					MaxDepth:    depth,
					Nodes: TreeNodes{
						FeatureIndex: featureIndex,
						Threshold:    threshold,
						LeftChild:    leftChild,
						RightChild:   rightChild,
						LeafValue:    leafValue,
						DefaultLeft:  defaultLeft,
						TreeStarts:   treeStarts,
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

func buildBenchModelWithScaler(numTrees, depth, numFeatures int) *TabularModel {
	m := buildBenchModel(numTrees, depth, numFeatures)
	rng := rand.New(rand.NewSource(99))

	center := make([]float64, numFeatures)
	scale := make([]float64, numFeatures)
	for i := range center {
		center[i] = rng.Float64() * 10
		scale[i] = rng.Float64()*5 + 0.1
	}

	// Prepend scaler stage
	m.Pipeline = append([]Stage{
		{
			Type: StageScaler,
			Scaler: &Scaler{
				Method: "standard",
				Center: center,
				Scale:  scale,
			},
		},
	}, m.Pipeline...)

	return m
}

func randomFeatures(rng *rand.Rand, n int) []float32 {
	f := make([]float32, n)
	for i := range f {
		f[i] = rng.Float32()
	}
	return f
}

// ============================================================================
// Benchmarks matching Timber's claimed test configuration:
//   "Apple M2 Pro, XGBoost 50 trees, 30 features"
//   Timber claims: ~2 us latency, 500k samples/sec
//   ONNX Runtime: ~80-150 us
//   Python XGBoost: ~670 us
// ============================================================================

// BenchmarkTimberComparison_50trees_30features matches Timber's primary benchmark.
// Timber claims ~2us on M2 Pro. We target <1us on M4.
func BenchmarkTimberComparison_50trees_30features(b *testing.B) {
	m := buildBenchModel(50, 8, 30)
	pred, err := NewPredictor(m)
	if err != nil {
		b.Fatal(err)
	}

	features := randomFeatures(rand.New(rand.NewSource(123)), 30)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = pred.PredictSingle(features)
	}
}

// BenchmarkTimberComparison_100trees_50features — medium model
func BenchmarkTimberComparison_100trees_50features(b *testing.B) {
	m := buildBenchModel(100, 8, 50)
	pred, err := NewPredictor(m)
	if err != nil {
		b.Fatal(err)
	}

	features := randomFeatures(rand.New(rand.NewSource(123)), 50)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = pred.PredictSingle(features)
	}
}

// BenchmarkTimberComparison_200trees_100features — large model
func BenchmarkTimberComparison_200trees_100features(b *testing.B) {
	m := buildBenchModel(200, 6, 100)
	pred, err := NewPredictor(m)
	if err != nil {
		b.Fatal(err)
	}

	features := randomFeatures(rand.New(rand.NewSource(123)), 100)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = pred.PredictSingle(features)
	}
}

// BenchmarkTimberComparison_500trees_200features — XL production model
func BenchmarkTimberComparison_500trees_200features(b *testing.B) {
	m := buildBenchModel(500, 8, 200)
	pred, err := NewPredictor(m)
	if err != nil {
		b.Fatal(err)
	}

	features := randomFeatures(rand.New(rand.NewSource(123)), 200)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = pred.PredictSingle(features)
	}
}

// BenchmarkWithPreprocessing — measures overhead of standard scaling
func BenchmarkWithPreprocessing_50trees_30features(b *testing.B) {
	m := buildBenchModelWithScaler(50, 8, 30)
	pred, err := NewPredictor(m)
	if err != nil {
		b.Fatal(err)
	}

	features := randomFeatures(rand.New(rand.NewSource(123)), 30)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = pred.PredictSingle(features)
	}
}

// BenchmarkBatch — throughput benchmark for batch processing
func BenchmarkBatch_50trees_30features(b *testing.B) {
	m := buildBenchModel(50, 8, 30)
	pred, err := NewPredictor(m)
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(123))
	batchSizes := []int{1, 10, 100, 1000}

	for _, size := range batchSizes {
		batch := make([][]float32, size)
		for i := range batch {
			batch[i] = randomFeatures(rng, 30)
		}

		b.Run(fmt.Sprintf("batch_%d", size), func(b *testing.B) {
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, _ = pred.Predict(context.Background(), batch)
			}

			b.ReportMetric(float64(size)/float64(b.Elapsed().Seconds())*float64(b.N), "samples/sec")
		})
	}
}

// BenchmarkTreeDepths — show impact of tree depth on latency
func BenchmarkTreeDepths(b *testing.B) {
	depths := []int{4, 6, 8, 10, 12}

	for _, depth := range depths {
		m := buildBenchModel(50, depth, 30)
		pred, err := NewPredictor(m)
		if err != nil {
			b.Fatal(err)
		}

		features := randomFeatures(rand.New(rand.NewSource(123)), 30)

		b.Run(fmt.Sprintf("depth_%d", depth), func(b *testing.B) {
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, _ = pred.PredictSingle(features)
			}
		})
	}
}
