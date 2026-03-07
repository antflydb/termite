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

// linearEngine runs inference on a LinearModel (logistic/linear regression).
type linearEngine struct {
	lm *LinearModel
}

func newLinearEngine(lm *LinearModel) *linearEngine {
	return &linearEngine{lm: lm}
}

// predictSingle computes output = W*x + b for one sample.
// Returns [num_outputs] raw values. Activation is NOT applied here;
// the caller (engine.go) applies Output.Activation uniformly for all engines.
func (e *linearEngine) predictSingle(features []float64) []float64 {
	numOutputs := len(e.lm.Weights)
	result := make([]float64, numOutputs)

	for o := 0; o < numOutputs; o++ {
		w := e.lm.Weights[o]
		var sum float64
		// Use the shorter of weights and features to avoid out-of-bounds.
		// A mismatch here indicates a model/input size discrepancy, but the
		// caller (PredictSingle) already validates input feature count.
		n := len(w)
		if n > len(features) {
			n = len(features)
		}
		for i := 0; i < n; i++ {
			sum += w[i] * features[i]
		}
		if o < len(e.lm.Biases) {
			sum += e.lm.Biases[o]
		}
		result[o] = sum
	}

	return result
}
