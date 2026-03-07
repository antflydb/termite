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
	"strconv"
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

		// Collect all nodes via DFS
		var nodes []map[string]any
		collectLGBNodes(tree, &nodes)

		base := int32(len(featureIndex))
		// Build ID map
		nodeMap := make(map[int]int32, len(nodes))
		for i := range nodes {
			nodeMap[i] = base + int32(i)
		}

		for i, node := range nodes {
			_ = i
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

				leftNode := jsonObj(node, "left_child")
				rightNode := jsonObj(node, "right_child")

				li := int32(-1)
				ri := int32(-1)
				// Find child index by searching for matching node in our collected list
				for j, n := range nodes {
					if sameNode(n, leftNode) {
						li = nodeMap[j]
					}
					if sameNode(n, rightNode) {
						ri = nodeMap[j]
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

				depth := jsonInt(node, "depth")
				if depth > maxDepth {
					maxDepth = depth
				}
			}
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

func collectLGBNodes(node map[string]any, nodes *[]map[string]any) {
	*nodes = append(*nodes, node)
	if left := jsonObj(node, "left_child"); left != nil {
		collectLGBNodes(left, nodes)
	}
	if right := jsonObj(node, "right_child"); right != nil {
		collectLGBNodes(right, nodes)
	}
}

// sameNode checks identity by comparing pointer-like fields (split_index or leaf_index).
func sameNode(a, b map[string]any) bool {
	if a == nil || b == nil {
		return false
	}
	// Compare by split_index for internal nodes
	if ai, ok := a["split_index"]; ok {
		if bi, ok := b["split_index"]; ok {
			return fmt.Sprint(ai) == fmt.Sprint(bi)
		}
	}
	// Compare by leaf_index for leaf nodes
	if ai, ok := a["leaf_index"]; ok {
		if bi, ok := b["leaf_index"]; ok {
			return fmt.Sprint(ai) == fmt.Sprint(bi)
		}
	}
	return false
}

var treeHeaderRe = regexp.MustCompile(`^Tree=\d+$`)

func parseLGBText(content string, name string) (*tabular.TabularModel, error) {
	lines := strings.Split(content, "\n")

	// Parse header parameters
	params := make(map[string]string)
	for _, line := range lines {
		if strings.Contains(line, "=") && !strings.HasPrefix(line, "Tree") {
			key, val, found := strings.Cut(line, "=")
			if found {
				params[strings.TrimSpace(key)] = strings.TrimSpace(val)
			}
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
			break
		} else if inTree {
			current = append(current, line)
		}
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
		// Single-leaf tree
		lv := parseFloat(arrays["leaf_value"], 0)
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
	defaults := strings.Fields(arrays["decision_type"])
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

		dl := true
		if i < len(defaults) {
			dt, _ := strconv.Atoi(defaults[i])
			dl = (dt & 2) != 0
		}
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
	obj := strings.ToLower(strings.Fields(objective)[0])

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
