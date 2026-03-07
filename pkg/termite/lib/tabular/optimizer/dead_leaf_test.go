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
	"testing"

	"github.com/antflydb/termite/pkg/termite/lib/tabular"
)

func TestDeadLeafElimination_PrunesSmallLeaves(t *testing.T) {
	// Build a simple tree:
	//       [0] split on feature 0 at 0.5
	//      /   \
	//   [1]     [2]
	//  leaf=1.0  leaf=0.0001  <-- dead (|0.0001| < 0.001 * 1.0)
	te := &tabular.TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 1,
		MaxDepth:    1,
		Nodes: tabular.TreeNodes{
			FeatureIndex: []int32{0, -1, -1},
			Threshold:    []float64{0.5, 0, 0},
			LeftChild:    []int32{1, -1, -1},
			RightChild:   []int32{2, -1, -1},
			LeafValue:    []float64{0, 1.0, 0.0001},
			DefaultLeft:  []bool{true, false, false},
			TreeStarts:   []int32{0},
		},
	}

	eliminated := DeadLeafElimination(te, 0.001)

	if eliminated != 1 {
		t.Errorf("expected 1 eliminated, got %d", eliminated)
	}
	// Node 2 should still be marked as leaf with zero value (it was already a leaf)
	// The leaf value is small enough to be "dead" but we don't remove it from the array
	if te.Nodes.FeatureIndex[2] != -1 {
		t.Error("dead leaf should remain marked as leaf")
	}
}

func TestDeadLeafElimination_CollapsesSubtree(t *testing.T) {
	// Tree:
	//       [0] split on feature 0
	//      /   \
	//   [1]     [2] split on feature 1
	//  leaf=1.0  /   \
	//         [3]     [4]
	//       leaf=0   leaf=0  <-- both dead
	te := &tabular.TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 2,
		MaxDepth:    2,
		Nodes: tabular.TreeNodes{
			FeatureIndex: []int32{0, -1, 1, -1, -1},
			Threshold:    []float64{0.5, 0, 0.3, 0, 0},
			LeftChild:    []int32{1, -1, 3, -1, -1},
			RightChild:   []int32{2, -1, 4, -1, -1},
			LeafValue:    []float64{0, 1.0, 0, 0.00001, 0.00002},
			DefaultLeft:  []bool{true, false, true, false, false},
			TreeStarts:   []int32{0},
		},
	}

	eliminated := DeadLeafElimination(te, 0.001)

	// Nodes 3 and 4 are dead leaves, then node 2 collapses = 3 total
	if eliminated != 3 {
		t.Errorf("expected 3 eliminated, got %d", eliminated)
	}
	// Node 2 should now be a leaf
	if te.Nodes.FeatureIndex[2] != -1 {
		t.Error("collapsed node should be a leaf")
	}
	if te.Nodes.LeftChild[2] != -1 || te.Nodes.RightChild[2] != -1 {
		t.Error("collapsed node should have no children")
	}
}

func TestDeadLeafElimination_NoOp(t *testing.T) {
	// All leaves are significant -- nothing should be eliminated
	te := &tabular.TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 1,
		MaxDepth:    1,
		Nodes: tabular.TreeNodes{
			FeatureIndex: []int32{0, -1, -1},
			Threshold:    []float64{0.5, 0, 0},
			LeftChild:    []int32{1, -1, -1},
			RightChild:   []int32{2, -1, -1},
			LeafValue:    []float64{0, 0.5, -0.3},
			DefaultLeft:  []bool{true, false, false},
			TreeStarts:   []int32{0},
		},
	}

	eliminated := DeadLeafElimination(te, 0.001)

	if eliminated != 0 {
		t.Errorf("expected 0 eliminated, got %d", eliminated)
	}
}

func TestDeadLeafElimination_EmptyTree(t *testing.T) {
	te := &tabular.TreeEnsemble{
		Nodes: tabular.TreeNodes{},
	}

	eliminated := DeadLeafElimination(te, 0.001)
	if eliminated != 0 {
		t.Errorf("expected 0 eliminated for empty tree, got %d", eliminated)
	}
}
