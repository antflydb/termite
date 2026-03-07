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

package convert

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/antflydb/termite/pkg/termite/lib/tabular"
)

// ParseXGBoost parses an XGBoost JSON model file into Termite's tabular IR.
func ParseXGBoost(path string, name string) (*tabular.TabularModel, error) {
	if name == "" {
		base := filepath.Base(path)
		name = strings.TrimSuffix(base, filepath.Ext(base))
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading xgboost model: %w", err)
	}

	var modelData map[string]any
	if err := json.Unmarshal(data, &modelData); err != nil {
		return nil, fmt.Errorf("parsing xgboost JSON: %w", err)
	}

	return parseXGBoostJSON(modelData, name)
}

func parseXGBoostJSON(modelData map[string]any, name string) (*tabular.TabularModel, error) {
	learner := jsonObj(modelData, "learner")
	if learner == nil {
		learner = modelData
	}
	learnerParams := jsonObj(learner, "learner_model_param")
	gradientBooster := jsonObj(learner, "gradient_booster")
	gbtreeModel := jsonObj(gradientBooster, "model")

	numFeatures := jsonInt(learnerParams, "num_feature")
	baseScore := parseXGBScalar(jsonStr(learnerParams, "base_score"), 0.5)
	numClass := jsonInt(learnerParams, "num_class")

	objectiveData := jsonObj(learner, "objective")
	objectiveName := jsonStrDefault(objectiveData, "name", "reg:squarederror")
	task, activation := xgbObjectiveToTask(objectiveName, numClass)

	treesData := jsonArray(gbtreeModel, "trees")
	if len(treesData) == 0 {
		return nil, fmt.Errorf("no trees found in XGBoost model")
	}

	var (
		featureIndex []int32
		threshold    []float64
		leftChild    []int32
		rightChild   []int32
		leafValue    []float64
		defaultLeft  []bool
		treeStarts   []int32
		maxDepth     int
	)

	for _, treeRaw := range treesData {
		tree, ok := treeRaw.(map[string]any)
		if !ok {
			continue
		}
		treeStarts = append(treeStarts, int32(len(featureIndex)))
		depth := flattenXGBTree(tree, &featureIndex, &threshold, &leftChild, &rightChild, &leafValue, &defaultLeft)
		if depth > maxDepth {
			maxDepth = depth
		}
	}

	numTrees := len(treesData)

	featureNames := jsonStringArray(learner, "feature_names")

	if task == tabular.TaskMulticlass {
		return nil, fmt.Errorf("multiclass tree ensembles (num_class=%d) are not yet supported", numClass)
	}

	numOutputs := max(numClass, 1)
	if task == tabular.TaskBinaryClassification {
		numOutputs = 1
	}

	numClasses := 0
	if numClass > 2 {
		numClasses = numClass
	}

	return &tabular.TabularModel{
		SchemaVersion: 1,
		Metadata: tabular.Metadata{
			Name:            name,
			SourceFramework: "xgboost",
			SourceVersion:   jsonStr(learnerParams, "version"),
			Task:            task,
			NumFeatures:     numFeatures,
			NumClasses:      numClasses,
			FeatureNames:    featureNames,
			CreatedAt:       time.Now().UTC().Format(time.RFC3339),
		},
		Pipeline: []tabular.Stage{
			{
				Type: tabular.StageTreeEnsemble,
				TreeEnsemble: &tabular.TreeEnsemble{
					Objective:   objectiveName,
					BaseScore:   baseScore,
					NumTrees:    numTrees,
					NumFeatures: numFeatures,
					MaxDepth:    maxDepth,
					Nodes: tabular.TreeNodes{
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
		Output: tabular.OutputCfg{
			Activation: activation,
			NumOutputs: numOutputs,
		},
	}, nil
}

func flattenXGBTree(
	treeData map[string]any,
	featureIndex *[]int32,
	threshold *[]float64,
	leftChild *[]int32,
	rightChild *[]int32,
	leafValue *[]float64,
	defaultLeft *[]bool,
) int {
	base := int32(len(*featureIndex))

	// XGBoost 3.x format: flat parallel arrays (left_children, right_children, etc.)
	if leftArr := jsonFloatArray(treeData, "left_children"); len(leftArr) > 0 {
		return flattenXGBTreeV3(treeData, base, featureIndex, threshold, leftChild, rightChild, leafValue, defaultLeft)
	}

	// Modern format with "nodes" array
	nodes := jsonArray(treeData, "nodes")
	if len(nodes) == 0 {
		// Legacy nested format
		nodes = collectNodesRecursive(treeData)
	}

	// Determine slot count from max nodeid (may differ from len(nodes) in legacy format)
	maxNodeID := 0
	for _, nRaw := range nodes {
		if n, ok := nRaw.(map[string]any); ok {
			if nid := jsonInt(n, "nodeid"); nid > maxNodeID {
				maxNodeID = nid
			}
		}
	}
	numSlots := maxNodeID + 1
	if numSlots < len(nodes) {
		numSlots = len(nodes)
	}
	// Pre-allocate
	for range numSlots {
		*featureIndex = append(*featureIndex, -1)
		*threshold = append(*threshold, 0.0)
		*leftChild = append(*leftChild, -1)
		*rightChild = append(*rightChild, -1)
		*leafValue = append(*leafValue, 0.0)
		*defaultLeft = append(*defaultLeft, false)
	}

	maxDepth := 0
	for _, nodeRaw := range nodes {
		node, ok := nodeRaw.(map[string]any)
		if !ok {
			continue
		}
		nid := jsonInt(node, "nodeid")
		idx := int(base) + nid

		if _, hasLeaf := node["leaf"]; hasLeaf {
			(*featureIndex)[idx] = -1
			(*leafValue)[idx] = jsonFloat(node, "leaf")
			(*leftChild)[idx] = -1
			(*rightChild)[idx] = -1
		} else {
			fi := jsonInt(node, "split_feature_id")
			if !jsonHasKey(node, "split_feature_id") {
				fi = jsonInt(node, "split")
			}
			(*featureIndex)[idx] = int32(fi)

			th := jsonFloat(node, "split_condition")
			if !jsonHasKey(node, "split_condition") {
				th = jsonFloat(node, "threshold")
			}
			(*threshold)[idx] = th

			yes := jsonIntDefault(node, "yes", -1)
			no := jsonIntDefault(node, "no", -1)
			if yes >= 0 {
				(*leftChild)[idx] = base + int32(yes)
			}
			if no >= 0 {
				(*rightChild)[idx] = base + int32(no)
			}

			missing := jsonIntDefault(node, "missing", yes)
			(*defaultLeft)[idx] = (missing == yes)

			depth := jsonInt(node, "depth")
			if depth > maxDepth {
				maxDepth = depth
			}
		}
	}

	return maxDepth
}

// flattenXGBTreeV3 handles XGBoost 3.x JSON format with flat parallel arrays.
// Fields: left_children, right_children, split_indices, split_conditions,
// default_left, base_weights.
func flattenXGBTreeV3(
	treeData map[string]any,
	base int32,
	featureIndex *[]int32,
	threshold *[]float64,
	leftChild *[]int32,
	rightChild *[]int32,
	leafValue *[]float64,
	defaultLeft *[]bool,
) int {
	lefts := jsonFloatArray(treeData, "left_children")
	rights := jsonFloatArray(treeData, "right_children")
	splitIdx := jsonFloatArray(treeData, "split_indices")
	splitCond := jsonFloatArray(treeData, "split_conditions")
	defLeft := jsonFloatArray(treeData, "default_left")
	weights := jsonFloatArray(treeData, "base_weights")

	numNodes := len(lefts)

	maxDepth := 0
	// Compute depth per node
	depths := make([]int, numNodes)
	for i := 0; i < numNodes; i++ {
		l := int(lefts[i])
		r := int(rights[i])
		if l >= 0 && l < numNodes {
			depths[l] = depths[i] + 1
			if depths[l] > maxDepth {
				maxDepth = depths[l]
			}
		}
		if r >= 0 && r < numNodes {
			depths[r] = depths[i] + 1
			if depths[r] > maxDepth {
				maxDepth = depths[r]
			}
		}
	}

	for i := 0; i < numNodes; i++ {
		l := int(lefts[i])
		r := int(rights[i])

		isLeaf := l < 0 // leaf nodes have left_children = -1

		if isLeaf {
			*featureIndex = append(*featureIndex, -1)
			*threshold = append(*threshold, 0.0)
			*leftChild = append(*leftChild, -1)
			*rightChild = append(*rightChild, -1)
			lv := 0.0
			if i < len(weights) {
				lv = weights[i]
			}
			*leafValue = append(*leafValue, lv)
			*defaultLeft = append(*defaultLeft, false)
		} else {
			fi := int32(0)
			if i < len(splitIdx) {
				fi = int32(splitIdx[i])
			}
			*featureIndex = append(*featureIndex, fi)

			th := 0.0
			if i < len(splitCond) {
				th = splitCond[i]
			}
			*threshold = append(*threshold, th)

			*leftChild = append(*leftChild, base+int32(l))
			*rightChild = append(*rightChild, base+int32(r))
			*leafValue = append(*leafValue, 0.0)

			dl := false
			if i < len(defLeft) {
				dl = defLeft[i] != 0
			}
			*defaultLeft = append(*defaultLeft, dl)
		}
	}

	return maxDepth
}

func collectNodesRecursive(node map[string]any) []any {
	result := []any{node}
	if children, ok := node["children"].([]any); ok {
		for _, child := range children {
			if childMap, ok := child.(map[string]any); ok {
				result = append(result, collectNodesRecursive(childMap)...)
			}
		}
	}
	return result
}

// parseXGBScalar parses an XGBoost scalar which may be bracketed like "[4.82E-1]".
func parseXGBScalar(s string, fallback float64) float64 {
	if s == "" {
		return fallback
	}
	s = strings.Trim(s, "[] \t\n")
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return fallback
	}
	return v
}

func xgbObjectiveToTask(objective string, numClass int) (tabular.TaskType, tabular.ActivationType) {
	switch objective {
	case "binary:logistic":
		return tabular.TaskBinaryClassification, tabular.ActivationSigmoid
	case "binary:logitraw":
		return tabular.TaskBinaryClassification, tabular.ActivationIdentity
	case "binary:hinge":
		return tabular.TaskBinaryClassification, tabular.ActivationIdentity
	case "multi:softmax", "multi:softprob":
		return tabular.TaskMulticlass, tabular.ActivationSoftmax
	case "rank:pairwise", "rank:ndcg", "rank:map":
		return tabular.TaskRanking, tabular.ActivationIdentity
	case "count:poisson", "survival:cox":
		return tabular.TaskRegression, tabular.ActivationExp
	default:
		return tabular.TaskRegression, tabular.ActivationIdentity
	}
}
