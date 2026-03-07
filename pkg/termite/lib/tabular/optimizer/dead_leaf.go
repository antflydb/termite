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

package optimizer

import (
	"math"

	"github.com/antflydb/termite/pkg/termite/lib/tabular"
)

// DeadLeafElimination prunes leaves whose |value| < thresholdFraction * max|leafValue|.
// When both children of an internal node become dead leaves, the subtree collapses
// into a single leaf with zero value. This typically reduces tree size by 10-30%.
//
// The operation modifies the TreeEnsemble in-place and returns the number of
// nodes eliminated.
func DeadLeafElimination(te *tabular.TreeEnsemble, thresholdFraction float64) int {
	nodes := &te.Nodes
	n := len(nodes.FeatureIndex)
	if n == 0 {
		return 0
	}

	eliminated := 0
	isDead := make([]bool, n)

	// Compute threshold per-tree so late correction trees with small leaf
	// values aren't pruned relative to early trees with large values.
	for t := 0; t < te.NumTrees; t++ {
		treeStart := int(nodes.TreeStarts[t])
		treeEnd := n
		if t+1 < te.NumTrees {
			treeEnd = int(nodes.TreeStarts[t+1])
		}

		var maxVal float64
		for i := treeStart; i < treeEnd; i++ {
			if nodes.FeatureIndex[i] < 0 {
				if v := math.Abs(nodes.LeafValue[i]); v > maxVal {
					maxVal = v
				}
			}
		}
		if maxVal == 0 {
			continue
		}

		absThreshold := thresholdFraction * maxVal
		for i := treeStart; i < treeEnd; i++ {
			if nodes.FeatureIndex[i] < 0 && math.Abs(nodes.LeafValue[i]) < absThreshold {
				isDead[i] = true
				eliminated++
			}
		}
	}

	// Bottom-up: collapse internal nodes where both children are dead.
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
					eliminated++
					changed = true
				}
			}
		}
	}

	return eliminated
}
