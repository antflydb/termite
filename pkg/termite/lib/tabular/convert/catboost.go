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
	"strings"
	"time"

	"github.com/antflydb/termite/pkg/termite/lib/tabular"
)

// ParseCatBoost parses a CatBoost JSON model file into Termite's tabular IR.
func ParseCatBoost(path string, name string) (*tabular.TabularModel, error) {
	if name == "" {
		base := filepath.Base(path)
		name = strings.TrimSuffix(base, filepath.Ext(base))
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading catboost model: %w", err)
	}

	var modelData map[string]any
	if err := json.Unmarshal(data, &modelData); err != nil {
		return nil, fmt.Errorf("parsing catboost JSON: %w", err)
	}

	return parseCatBoostJSON(modelData, name)
}

func parseCatBoostJSON(data map[string]any, name string) (*tabular.TabularModel, error) {
	modelInfo := jsonObj(data, "model_info")
	obliviousTrees := jsonArray(data, "oblivious_trees")

	if len(obliviousTrees) == 0 {
		return nil, fmt.Errorf("no trees found in CatBoost model (ensure model was saved with format='json')")
	}

	params := jsonObj(modelInfo, "params")
	lossFunction := jsonObj(params, "loss_function")
	lossType := jsonStrDefault(lossFunction, "type", "RMSE")

	numFeatures := catboostCountFeatures(data, obliviousTrees)
	task, activation := catboostLossToTask(lossType)

	if task == tabular.TaskMulticlass {
		return nil, fmt.Errorf("multiclass tree ensembles are not yet supported")
	}

	// Scale and bias
	scale := 1.0
	bias := 0.0
	if sb, ok := data["scale_and_bias"].([]any); ok && len(sb) >= 2 {
		scale = anyToFloat(sb[0])
		bias = anyToFloat(sb[1])
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

	for _, treeRaw := range obliviousTrees {
		tree, ok := treeRaw.(map[string]any)
		if !ok {
			continue
		}
		treeStarts = append(treeStarts, int32(len(featureIndex)))
		depth := convertObliviousTree(
			tree, scale,
			&featureIndex, &threshold,
			&leftChild, &rightChild,
			&leafValue, &defaultLeft,
		)
		if depth > maxDepth {
			maxDepth = depth
		}
	}

	numTrees := len(obliviousTrees)

	numClasses := 0
	if task == tabular.TaskMulticlass {
		classParams := jsonObj(modelInfo, "class_params")
		numClasses = jsonInt(classParams, "class_count")
	}

	numOutputs := max(numClasses, 1)
	if task == tabular.TaskBinaryClassification {
		numOutputs = 1
	}

	displayClasses := 0
	if numClasses > 2 {
		displayClasses = numClasses
	}

	return &tabular.TabularModel{
		SchemaVersion: 1,
		Metadata: tabular.Metadata{
			Name:            name,
			SourceFramework: "catboost",
			SourceVersion:   jsonStr(modelInfo, "catboost_version_info"),
			Task:            task,
			NumFeatures:     numFeatures,
			NumClasses:      displayClasses,
			CreatedAt:       time.Now().UTC().Format(time.RFC3339),
		},
		Pipeline: []tabular.Stage{
			{
				Type: tabular.StageTreeEnsemble,
				TreeEnsemble: &tabular.TreeEnsemble{
					Objective:   lossType,
					BaseScore:   bias,
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

func convertObliviousTree(
	tree map[string]any,
	scale float64,
	featureIndex *[]int32,
	threshold *[]float64,
	leftChild *[]int32,
	rightChild *[]int32,
	leafValue *[]float64,
	defaultLeft *[]bool,
) int {
	splits := jsonArray(tree, "splits")
	leafValues := jsonFloatArray(tree, "leaf_values")

	depth := len(splits)
	if depth == 0 {
		lv := 0.0
		if len(leafValues) > 0 {
			lv = leafValues[0] * scale
		}
		*featureIndex = append(*featureIndex, -1)
		*threshold = append(*threshold, 0.0)
		*leftChild = append(*leftChild, -1)
		*rightChild = append(*rightChild, -1)
		*leafValue = append(*leafValue, lv)
		*defaultLeft = append(*defaultLeft, false)
		return 0
	}

	numInternal := (1 << depth) - 1
	numLeaves := 1 << depth
	totalNodes := numInternal + numLeaves
	base := int32(len(*featureIndex))

	// Pre-allocate
	for range totalNodes {
		*featureIndex = append(*featureIndex, -1)
		*threshold = append(*threshold, 0.0)
		*leftChild = append(*leftChild, -1)
		*rightChild = append(*rightChild, -1)
		*leafValue = append(*leafValue, 0.0)
		*defaultLeft = append(*defaultLeft, false)
	}

	// Fill internal nodes level by level
	for level := 0; level < depth; level++ {
		splitIdx := depth - 1 - level // CatBoost reverses the order
		split, ok := splits[splitIdx].(map[string]any)
		if !ok {
			continue
		}

		feat := jsonInt(split, "float_feature_index")
		if !jsonHasKey(split, "float_feature_index") {
			feat = jsonInt(split, "feature_index")
		}
		thresh := jsonFloat(split, "border")
		if !jsonHasKey(split, "border") {
			thresh = jsonFloat(split, "threshold")
		}

		levelStart := (1 << level) - 1
		levelEnd := (1 << (level + 1)) - 1

		for i := levelStart; i < levelEnd; i++ {
			idx := int(base) + i
			(*featureIndex)[idx] = int32(feat)
			(*threshold)[idx] = thresh
			(*leftChild)[idx] = base + int32(2*i+1)
			(*rightChild)[idx] = base + int32(2*i+2)
			(*defaultLeft)[idx] = true
		}
	}

	// Fill leaf nodes.
	// CatBoost leaf ordering uses a bit-convention where bit d (from the deepest
	// split) is 1 when the sample goes right at depth d. The BFS leaf order is
	// different, so we must reverse the bits of the leaf index to map correctly.
	for i := 0; i < numLeaves; i++ {
		// Reverse the path bits to get CatBoost's leaf index
		cbIdx := 0
		for d := 0; d < depth; d++ {
			if (i>>d)&1 == 1 {
				cbIdx |= 1 << (depth - 1 - d)
			}
		}
		idx := int(base) + numInternal + i
		if cbIdx < len(leafValues) {
			(*leafValue)[idx] = leafValues[cbIdx] * scale
		}
		(*featureIndex)[idx] = -1
		(*leftChild)[idx] = -1
		(*rightChild)[idx] = -1
	}

	return depth
}

func catboostCountFeatures(data map[string]any, trees []any) int {
	featuresInfo := jsonObj(data, "features_info")
	floatFeatures := jsonArray(featuresInfo, "float_features")
	if len(floatFeatures) > 0 {
		maxIdx := 0
		for _, fRaw := range floatFeatures {
			f, ok := fRaw.(map[string]any)
			if !ok {
				continue
			}
			idx := jsonInt(f, "flat_feature_index")
			if idx > maxIdx {
				maxIdx = idx
			}
		}
		return maxIdx + 1
	}

	maxFeat := 0
	for _, treeRaw := range trees {
		tree, ok := treeRaw.(map[string]any)
		if !ok {
			continue
		}
		for _, splitRaw := range jsonArray(tree, "splits") {
			split, ok := splitRaw.(map[string]any)
			if !ok {
				continue
			}
			fi := jsonInt(split, "float_feature_index")
			if !jsonHasKey(split, "float_feature_index") {
				fi = jsonInt(split, "feature_index")
			}
			if fi > maxFeat {
				maxFeat = fi
			}
		}
	}
	return maxFeat + 1
}

func catboostLossToTask(lossType string) (tabular.TaskType, tabular.ActivationType) {
	loss := strings.ToUpper(lossType)

	switch loss {
	case "LOGLOSS", "CROSSENTROPY":
		return tabular.TaskBinaryClassification, tabular.ActivationSigmoid
	case "MULTICLASS", "MULTICLASSONEHOT":
		return tabular.TaskMulticlass, tabular.ActivationSoftmax
	case "QUERYRMSE", "PAIRLOGIT", "PAIRLOGITPAIRWISE", "YETIARANKPAIRWISE", "QUERYSOFTMAX":
		return tabular.TaskRanking, tabular.ActivationIdentity
	case "POISSON":
		return tabular.TaskRegression, tabular.ActivationExp
	default:
		return tabular.TaskRegression, tabular.ActivationIdentity
	}
}

// anyToFloat extracts a float64 from a value that may be a float, []any with one element, etc.
func anyToFloat(v any) float64 {
	switch val := v.(type) {
	case float64:
		return val
	case []any:
		if len(val) > 0 {
			if f, ok := val[0].(float64); ok {
				return f
			}
		}
	}
	return 0
}
