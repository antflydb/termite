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

// Package optimizer provides load-time optimization passes for tabular models.
// These complement the Python-side conversion-time optimizations and can be
// applied to any model loaded from tabular_model.json.
package optimizer

import (
	"github.com/antflydb/termite/pkg/termite/lib/tabular"
)

// Options controls which optimization passes to run and their parameters.
type Options struct {
	// DeadLeafThreshold is the fraction of max leaf value below which leaves
	// are pruned. Set to 0 to disable. Default: 0.001.
	DeadLeafThreshold float64

	// EnableBranchSort reorders children so the statistically more likely
	// branch is tested first (requires annotations from conversion).
	EnableBranchSort bool
}

// DefaultOptions returns sensible defaults for production use.
func DefaultOptions() Options {
	return Options{
		DeadLeafThreshold: 0.001,
	}
}

// Optimize runs all enabled optimization passes on a TabularModel in-place.
// Returns the total number of nodes eliminated/modified.
func Optimize(m *tabular.TabularModel, opts Options) int {
	total := 0
	for i := range m.Pipeline {
		stage := &m.Pipeline[i]
		if stage.TreeEnsemble != nil {
			if opts.DeadLeafThreshold > 0 {
				eliminated := DeadLeafElimination(stage.TreeEnsemble, opts.DeadLeafThreshold)
				stage.TreeEnsemble.Annotations.DeadLeavesEliminated += eliminated
				total += eliminated
			}
		}
	}
	return total
}
