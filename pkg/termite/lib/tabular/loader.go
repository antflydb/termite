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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// TabularModelFilename is the standard filename for the tabular IR.
const TabularModelFilename = "tabular_model.json"

// LoadModel reads and validates a TabularModel from a JSON file.
func LoadModel(path string) (*TabularModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading tabular model: %w", err)
	}
	return ParseModel(data)
}

// LoadModelFromDir loads the tabular model from a model directory.
func LoadModelFromDir(dir string) (*TabularModel, error) {
	return LoadModel(filepath.Join(dir, TabularModelFilename))
}

// ParseModel deserialises and validates a TabularModel from JSON bytes.
func ParseModel(data []byte) (*TabularModel, error) {
	var m TabularModel
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parsing tabular model: %w", err)
	}
	if err := m.Validate(); err != nil {
		return nil, err
	}
	return &m, nil
}

// Validate checks that the model IR is well-formed.
func (m *TabularModel) Validate() error {
	if m.SchemaVersion < 1 {
		return fmt.Errorf("unsupported tabular model schema version: %d", m.SchemaVersion)
	}
	if m.Metadata.Name == "" {
		return fmt.Errorf("tabular model missing name")
	}
	if m.Metadata.NumFeatures <= 0 {
		return fmt.Errorf("tabular model must have num_features > 0")
	}
	if len(m.Pipeline) == 0 {
		return fmt.Errorf("tabular model must have at least one pipeline stage")
	}
	if m.Output.NumOutputs <= 0 {
		return fmt.Errorf("tabular model must have num_outputs > 0")
	}

	hasModelStage := false
	for i, s := range m.Pipeline {
		switch s.Type {
		case StageTreeEnsemble:
			if s.TreeEnsemble == nil {
				return fmt.Errorf("stage %d: tree_ensemble type but no data", i)
			}
			if err := validateTreeEnsemble(s.TreeEnsemble); err != nil {
				return fmt.Errorf("stage %d: %w", i, err)
			}
			hasModelStage = true
		case StageLinear:
			if s.Linear == nil {
				return fmt.Errorf("stage %d: linear type but no data", i)
			}
			hasModelStage = true
		case StageSVM:
			if s.SVM == nil {
				return fmt.Errorf("stage %d: svm type but no data", i)
			}
			hasModelStage = true
		case StageScaler:
			if s.Scaler == nil {
				return fmt.Errorf("stage %d: scaler type but no data", i)
			}
		case StageImputer:
			if s.Imputer == nil {
				return fmt.Errorf("stage %d: imputer type but no data", i)
			}
		default:
			return fmt.Errorf("stage %d: unknown type %q", i, s.Type)
		}
	}
	if !hasModelStage {
		return fmt.Errorf("pipeline must contain at least one model stage (tree_ensemble, linear, or svm)")
	}
	return nil
}

func validateTreeEnsemble(te *TreeEnsemble) error {
	n := len(te.Nodes.FeatureIndex)
	if n == 0 {
		return fmt.Errorf("tree ensemble has no nodes")
	}
	if len(te.Nodes.Threshold) != n ||
		len(te.Nodes.LeftChild) != n ||
		len(te.Nodes.RightChild) != n ||
		len(te.Nodes.LeafValue) != n ||
		len(te.Nodes.DefaultLeft) != n {
		return fmt.Errorf("tree ensemble node arrays have inconsistent lengths")
	}
	if len(te.Nodes.TreeStarts) == 0 {
		return fmt.Errorf("tree ensemble has no tree_starts")
	}
	if te.NumTrees != len(te.Nodes.TreeStarts) {
		return fmt.Errorf("num_trees (%d) does not match tree_starts length (%d)", te.NumTrees, len(te.Nodes.TreeStarts))
	}
	// Validate TreeStarts indices are in bounds
	for i, start := range te.Nodes.TreeStarts {
		if int(start) < 0 || int(start) >= n {
			return fmt.Errorf("tree_starts[%d] = %d is out of bounds (0..%d)", i, start, n-1)
		}
	}
	// Validate feature indices and child pointers are in bounds
	for i := 0; i < n; i++ {
		fi := te.Nodes.FeatureIndex[i]
		if fi >= 0 {
			if te.NumFeatures > 0 && int(fi) >= te.NumFeatures {
				return fmt.Errorf("node %d: feature_index %d >= num_features %d", i, fi, te.NumFeatures)
			}
			lc := te.Nodes.LeftChild[i]
			rc := te.Nodes.RightChild[i]
			if lc < 0 || int(lc) >= n {
				return fmt.Errorf("node %d: left_child %d is out of bounds (0..%d)", i, lc, n-1)
			}
			if rc < 0 || int(rc) >= n {
				return fmt.Errorf("node %d: right_child %d is out of bounds (0..%d)", i, rc, n-1)
			}
		}
	}
	return nil
}
