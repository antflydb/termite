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
	"regexp"
	"strings"
	"time"

	"github.com/antflydb/termite/pkg/termite/lib/tabular"
)

// ParseLightGBM parses a LightGBM model file (text or JSON) into Termite's tabular IR.
func ParseLightGBM(path string, name string) (*tabular.TabularModel, error) {
	if name == "" {
		base := filepath.Base(path)
		name = strings.TrimSuffix(base, filepath.Ext(base))
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading lightgbm model: %w", err)
	}

	content := string(data)
	if strings.HasPrefix(strings.TrimSpace(content), "{") {
		var jsonData map[string]any
		if err := json.Unmarshal(data, &jsonData); err != nil {
			return nil, fmt.Errorf("parsing lightgbm JSON: %w", err)
		}
		return parseLGBJSON(jsonData, name)
	}
	return parseLGBText(content, name)
}

func parseLGBJSON(data map[string]any, name string) (*tabular.TabularModel, error) {
	numFeatures := jsonInt(data, "max_feature_idx") + 1
	numClass := jsonIntDefault(data, "num_class", 1)
	objective := jsonStrDefault(data, "objective", "regression")
	featureNames := jsonStringArray(data, "feature_names")
	treesData := jsonArray(data, "tree_info")

	if len(treesData) == 0 {
		return nil, fmt.Errorf("no trees found in LightGBM model")
	}

	task, activation := lgbObjectiveToTask(objective, numClass)

	if task == tabular.TaskMulticlass {
		return nil, fmt.Errorf("multiclass tree ensembles (num_class=%d) are not yet supported", numClass)
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
		treeInfo, ok := treeRaw.(map[string]any)
		if !ok {
			continue
		}
		tree := jsonObj(treeInfo, "tree_structure")
		if tree == nil {
			tree = treeInfo
		}

		treeStarts = append(treeStarts, int32(len(featureIndex)))

		// Collect all nodes via DFS and assign sequential indices
		var nodes []map[string]any
		treeDepth := collectLGBNodes(tree, &nodes)
		if treeDepth > maxDepth {
			maxDepth = treeDepth
		}

		base := int32(len(featureIndex))
		// Assign a DFS index to each node so children can be looked up in O(1)
		for i := range nodes {
			nodes[i]["__dfs_idx__"] = i
		}

		for _, node := range nodes {
			if _, hasLeaf := node["leaf_value"]; hasLeaf {
				featureIndex = append(featureIndex, -1)
				threshold = append(threshold, 0.0)
				leftChild = append(leftChild, -1)
				rightChild = append(rightChild, -1)
				leafValue = append(leafValue, jsonFloat(node, "leaf_value"))
				defaultLeft = append(defaultLeft, false)
			} else {
				featureIndex = append(featureIndex, int32(jsonInt(node, "split_feature")))
				threshold = append(threshold, jsonFloat(node, "threshold"))

				li := int32(-1)
				ri := int32(-1)
				if leftNode := jsonObj(node, "left_child"); leftNode != nil {
					if idx, ok := leftNode["__dfs_idx__"]; ok {
						if iidx, ok2 := idx.(int); ok2 {
							li = base + int32(iidx)
						}
					}
				}
				if rightNode := jsonObj(node, "right_child"); rightNode != nil {
					if idx, ok := rightNode["__dfs_idx__"]; ok {
						if iidx, ok2 := idx.(int); ok2 {
							ri = base + int32(iidx)
						}
					}
				}

				leftChild = append(leftChild, li)
				rightChild = append(rightChild, ri)
				leafValue = append(leafValue, 0.0)

				dl := true
				if v, ok := node["default_left"]; ok {
					dl = jsonBool(v)
				}
				defaultLeft = append(defaultLeft, dl)
			}
		}

		// Clean up temporary index markers
		for i := range nodes {
			delete(nodes[i], "__dfs_idx__")
		}
	}

	numTrees := len(treesData)
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
			SourceFramework: "lightgbm",
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
					Objective:   objective,
					BaseScore:   0.0,
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

// collectLGBNodes collects all nodes via DFS, returning the max depth of the subtree.
func collectLGBNodes(node map[string]any, nodes *[]map[string]any) int {
	*nodes = append(*nodes, node)
	d := 0
	if left := jsonObj(node, "left_child"); left != nil {
		if ld := collectLGBNodes(left, nodes) + 1; ld > d {
			d = ld
		}
	}
	if right := jsonObj(node, "right_child"); right != nil {
		if rd := collectLGBNodes(right, nodes) + 1; rd > d {
			d = rd
		}
	}
	return d
}

var treeHeaderRe = regexp.MustCompile(`^Tree=\d+$`)

func parseLGBText(content string, name string) (*tabular.TabularModel, error) {
	lines := strings.Split(content, "\n")

	// Parse header parameters (stop at first tree section)
	params := make(map[string]string)
	for _, line := range lines {
		if treeHeaderRe.MatchString(line) {
			break
		}
		if key, val, found := strings.Cut(line, "="); found {
			params[strings.TrimSpace(key)] = strings.TrimSpace(val)
		}
	}

	numFeatures := parseInt(params["max_feature_idx"], 0) + 1
	numClass := parseInt(params["num_class"], 1)
	objective := params["objective"]
	if objective == "" {
		objective = "regression"
	}
	var featureNames []string
	if fn := strings.TrimSpace(params["feature_names"]); fn != "" {
		featureNames = strings.Fields(fn)
	}

	task, activation := lgbObjectiveToTask(objective, numClass)

	if task == tabular.TaskMulticlass {
		return nil, fmt.Errorf("multiclass tree ensembles (num_class=%d) are not yet supported", numClass)
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

	treeSections := splitTreeSections(lines)
	for _, section := range treeSections {
		treeStarts = append(treeStarts, int32(len(featureIndex)))
		depth := parseTextTree(section, &featureIndex, &threshold, &leftChild, &rightChild, &leafValue, &defaultLeft)
		if depth > maxDepth {
			maxDepth = depth
		}
	}

	numTrees := len(treeSections)
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
			SourceFramework: "lightgbm",
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
					Objective:   objective,
					BaseScore:   0.0,
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

func splitTreeSections(lines []string) [][]string {
	var sections [][]string
	var current []string
	inTree := false
	terminated := false
	for _, line := range lines {
		if treeHeaderRe.MatchString(line) {
			if len(current) > 0 && inTree {
				sections = append(sections, current)
			}
			current = nil
			inTree = true
		} else if strings.HasPrefix(line, "end of trees") {
			if len(current) > 0 && inTree {
				sections = append(sections, current)
			}
			terminated = true
			break
		} else if inTree {
			current = append(current, line)
		}
	}
	// Flush last tree if file ended without "end of trees" sentinel
	if !terminated && inTree && len(current) > 0 {
		sections = append(sections, current)
	}
	return sections
}

func parseTextTree(
	section []string,
	featureIndex *[]int32,
	threshold *[]float64,
	leftChild *[]int32,
	rightChild *[]int32,
	leafValue *[]float64,
	defaultLeft *[]bool,
) int {
	arrays := make(map[string]string)
	for _, line := range section {
		if key, val, ok := strings.Cut(line, "="); ok {
			arrays[strings.TrimSpace(key)] = strings.TrimSpace(val)
		}
	}

	numLeaves := parseInt(arrays["num_leaves"], 0)
	numInternal := numLeaves - 1
	if numInternal <= 0 {
		// Single-leaf tree — leaf_value may be space-separated, take the first.
		vals := parseFloatSlice(arrays["leaf_value"])
		lv := 0.0
		if len(vals) > 0 {
			lv = vals[0]
		}
		*featureIndex = append(*featureIndex, -1)
		*threshold = append(*threshold, 0.0)
		*leftChild = append(*leftChild, -1)
		*rightChild = append(*rightChild, -1)
		*leafValue = append(*leafValue, lv)
		*defaultLeft = append(*defaultLeft, false)
		return 0
	}

	base := int32(len(*featureIndex))

	splits := parseIntSlice(arrays["split_feature"])
	thresholds := parseFloatSlice(arrays["threshold"])
	lefts := parseIntSlice(arrays["left_child"])
	rights := parseIntSlice(arrays["right_child"])
	leafValues := parseFloatSlice(arrays["leaf_value"])

	// Internal nodes first
	for i := 0; i < numInternal; i++ {
		fi := int32(-1)
		if i < len(splits) {
			fi = int32(splits[i])
		}
		*featureIndex = append(*featureIndex, fi)

		th := 0.0
		if i < len(thresholds) {
			th = thresholds[i]
		}
		*threshold = append(*threshold, th)

		leftRaw := -1
		if i < len(lefts) {
			leftRaw = lefts[i]
		}
		rightRaw := -1
		if i < len(rights) {
			rightRaw = rights[i]
		}

		if leftRaw < 0 {
			*leftChild = append(*leftChild, base+int32(numInternal)+(^int32(leftRaw)))
		} else {
			*leftChild = append(*leftChild, base+int32(leftRaw))
		}

		if rightRaw < 0 {
			*rightChild = append(*rightChild, base+int32(numInternal)+(^int32(rightRaw)))
		} else {
			*rightChild = append(*rightChild, base+int32(rightRaw))
		}

		*leafValue = append(*leafValue, 0.0)

		// LightGBM default missing direction: use default_bin if available,
		// otherwise default to left (LightGBM's default behavior).
		// Note: decision_type bit 0 controls comparison type (<=/<), not default direction.
		dl := true
		*defaultLeft = append(*defaultLeft, dl)
	}

	// Leaf nodes
	for i := 0; i < numLeaves; i++ {
		*featureIndex = append(*featureIndex, -1)
		*threshold = append(*threshold, 0.0)
		*leftChild = append(*leftChild, -1)
		*rightChild = append(*rightChild, -1)
		lv := 0.0
		if i < len(leafValues) {
			lv = leafValues[i]
		}
		*leafValue = append(*leafValue, lv)
		*defaultLeft = append(*defaultLeft, false)
	}

	md := parseInt(arrays["max_depth"], 0)
	return md
}

func lgbObjectiveToTask(objective string, numClass int) (tabular.TaskType, tabular.ActivationType) {
	fields := strings.Fields(objective)
	if len(fields) == 0 {
		return tabular.TaskRegression, tabular.ActivationIdentity
	}
	obj := strings.ToLower(fields[0])

	switch obj {
	case "binary", "cross_entropy", "cross_entropy_lambda":
		return tabular.TaskBinaryClassification, tabular.ActivationSigmoid
	case "multiclass", "multiclassova", "softmax", "multiclass_ova":
		return tabular.TaskMulticlass, tabular.ActivationSoftmax
	case "lambdarank", "rank_xendcg", "rank_pairwise":
		return tabular.TaskRanking, tabular.ActivationIdentity
	case "poisson", "gamma", "tweedie":
		return tabular.TaskRegression, tabular.ActivationExp
	default:
		return tabular.TaskRegression, tabular.ActivationIdentity
	}
}
