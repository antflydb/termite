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
	"math"

	"github.com/ajroetker/go-highway/hwy/contrib/algo"
	"github.com/ajroetker/go-highway/hwy/contrib/vec"
)

// treeEngine runs inference on a TreeEnsemble.
// At construction, node data is pre-converted to float32 for cache-friendly
// traversal and to avoid repeated float64→float32 conversions.
type treeEngine struct {
	// Original IR (kept for metadata)
	te *TreeEnsemble

	// Pre-converted float32 thresholds and leaf values for cache efficiency.
	// Using float32 halves cache footprint and removes per-traversal conversion.
	threshold32 []float32
	leafValue32 []float32
}

func newTreeEngine(te *TreeEnsemble) *treeEngine {
	// Pre-convert thresholds and leaf values to float32 at load time
	t32 := make([]float32, len(te.Nodes.Threshold))
	l32 := make([]float32, len(te.Nodes.LeafValue))
	for i, v := range te.Nodes.Threshold {
		t32[i] = float32(v)
	}
	for i, v := range te.Nodes.LeafValue {
		l32[i] = float32(v)
	}
	return &treeEngine{te: te, threshold32: t32, leafValue32: l32}
}

// predictSingle32 traverses every tree for one sample using float32 throughout.
// Returns the raw accumulated leaf values including base score.
func (e *treeEngine) predictSingle32(features []float32) float32 {
	nodes := &e.te.Nodes
	featureIdx := nodes.FeatureIndex
	threshold := e.threshold32
	leftChild := nodes.LeftChild
	rightChild := nodes.RightChild
	leafValue := e.leafValue32
	defaultLeft := nodes.DefaultLeft
	treeStarts := nodes.TreeStarts
	numTrees := e.te.NumTrees
	maxIter := e.te.MaxDepth + 2

	sum := float32(e.te.BaseScore)

	for t := 0; t < numTrees; t++ {
		nodeIdx := int(treeStarts[t])

		// Bounded traversal prevents infinite loops on malformed models.
		for iter := 0; iter < maxIter; iter++ {
			fi := featureIdx[nodeIdx]

			// Leaf node
			if fi < 0 {
				sum += leafValue[nodeIdx]
				break
			}

			fv := features[fi]

			// NaN check: math.IsNaN is slow due to function call overhead.
			// NaN != NaN is the IEEE-754 fast path.
			if fv != fv {
				if defaultLeft[nodeIdx] {
					nodeIdx = int(leftChild[nodeIdx])
				} else {
					nodeIdx = int(rightChild[nodeIdx])
				}
			} else if fv <= threshold[nodeIdx] {
				nodeIdx = int(leftChild[nodeIdx])
			} else {
				nodeIdx = int(rightChild[nodeIdx])
			}
		}
	}

	return sum
}

// predictSingle traverses every tree using float64 precision (for compatibility
// with the preprocessing pipeline which works in float64).
func (e *treeEngine) predictSingle(features []float64) float64 {
	nodes := &e.te.Nodes
	featureIdx := nodes.FeatureIndex
	threshold := nodes.Threshold
	leftChild := nodes.LeftChild
	rightChild := nodes.RightChild
	leafValue := nodes.LeafValue
	defaultLeft := nodes.DefaultLeft
	treeStarts := nodes.TreeStarts
	numTrees := e.te.NumTrees
	maxIter := e.te.MaxDepth + 2

	sum := e.te.BaseScore

	for t := 0; t < numTrees; t++ {
		nodeIdx := int(treeStarts[t])

		for iter := 0; iter < maxIter; iter++ {
			fi := featureIdx[nodeIdx]

			if fi < 0 {
				sum += leafValue[nodeIdx]
				break
			}

			fv := features[fi]

			if fv != fv { // NaN fast check
				if defaultLeft[nodeIdx] {
					nodeIdx = int(leftChild[nodeIdx])
				} else {
					nodeIdx = int(rightChild[nodeIdx])
				}
			} else if fv <= threshold[nodeIdx] {
				nodeIdx = int(leftChild[nodeIdx])
			} else {
				nodeIdx = int(rightChild[nodeIdx])
			}
		}
	}

	return sum
}

// predictBatch32 processes a batch of samples through the tree ensemble,
// using go-highway SIMD for leaf value accumulation and batch activation.
// Returns raw scores (before activation) for all samples.
func (e *treeEngine) predictBatch32(features [][]float32) []float32 {
	numSamples := len(features)
	results := make([]float32, numSamples)

	nodes := &e.te.Nodes
	featureIdx := nodes.FeatureIndex
	threshold := e.threshold32
	leftChild := nodes.LeftChild
	rightChild := nodes.RightChild
	leafValue := e.leafValue32
	defaultLeft := nodes.DefaultLeft
	treeStarts := nodes.TreeStarts
	numTrees := e.te.NumTrees
	maxIter := e.te.MaxDepth + 2
	baseScore := float32(e.te.BaseScore)

	// Pre-allocate leaf values buffer for SIMD reduction
	leafBuf := make([]float32, numTrees)

	for s := 0; s < numSamples; s++ {
		f := features[s]

		// Zero the buffer so stale values cannot bleed between samples
		// (e.g., if a tree exhausts maxIter without reaching a leaf).
		for t := range leafBuf {
			leafBuf[t] = 0
		}

		// Traverse all trees and collect leaf values
		for t := 0; t < numTrees; t++ {
			nodeIdx := int(treeStarts[t])

			for iter := 0; iter < maxIter; iter++ {
				fi := featureIdx[nodeIdx]
				if fi < 0 {
					leafBuf[t] = leafValue[nodeIdx]
					break
				}

				fv := f[fi]
				if fv != fv { // NaN
					if defaultLeft[nodeIdx] {
						nodeIdx = int(leftChild[nodeIdx])
					} else {
						nodeIdx = int(rightChild[nodeIdx])
					}
				} else if fv <= threshold[nodeIdx] {
					nodeIdx = int(leftChild[nodeIdx])
				} else {
					nodeIdx = int(rightChild[nodeIdx])
				}
			}
		}

		// SIMD-accelerated sum of leaf values across all trees
		results[s] = baseScore + vec.Sum(leafBuf)
	}

	return results
}

// applyActivation applies the output activation function in-place using
// go-highway SIMD where beneficial.
func applyActivation(values []float64, activation ActivationType) {
	switch activation {
	case ActivationSigmoid:
		for i, v := range values {
			values[i] = 1.0 / (1.0 + math.Exp(-v))
		}
	case ActivationSoftmax:
		maxVal := values[0]
		for _, v := range values[1:] {
			if v > maxVal {
				maxVal = v
			}
		}
		var sum float64
		for i, v := range values {
			values[i] = math.Exp(v - maxVal)
			sum += values[i]
		}
		for i := range values {
			values[i] /= sum
		}
	case ActivationExp:
		for i, v := range values {
			values[i] = math.Exp(v)
		}
	case ActivationIdentity, "":
		// no-op
	}
}

// applyActivation32 applies activation in float32 using go-highway SIMD.
func applyActivation32(values []float32, activation ActivationType) {
	switch activation {
	case ActivationSigmoid:
		if len(values) >= 4 {
			out := make([]float32, len(values))
			algo.SigmoidTransform(values, out)
			copy(values, out)
		} else {
			for i, v := range values {
				values[i] = float32(1.0 / (1.0 + math.Exp(-float64(v))))
			}
		}
	case ActivationSoftmax:
		maxVal := values[0]
		for _, v := range values[1:] {
			if v > maxVal {
				maxVal = v
			}
		}
		var sum float32
		for i, v := range values {
			values[i] = float32(math.Exp(float64(v - maxVal)))
			sum += values[i]
		}
		for i := range values {
			values[i] /= sum
		}
	case ActivationExp:
		for i, v := range values {
			values[i] = float32(math.Exp(float64(v)))
		}
	case ActivationIdentity, "":
		// no-op
	}
}
