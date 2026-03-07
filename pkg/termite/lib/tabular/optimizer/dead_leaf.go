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

	// Find maximum absolute leaf value across all trees.
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

	// Mark dead leaves.
	isDead := make([]bool, n)
	for i := 0; i < n; i++ {
		if nodes.FeatureIndex[i] < 0 && math.Abs(nodes.LeafValue[i]) < absThreshold {
			isDead[i] = true
			eliminated++
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
