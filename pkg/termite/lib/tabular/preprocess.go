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

import "math"

// preprocessor applies a Scaler or Imputer stage to a feature vector in-place.
type preprocessor interface {
	apply(features []float64)
}

// scalerProcessor applies feature scaling.
type scalerProcessor struct {
	method string
	center []float64
	scale  []float64
	min    []float64
	max    []float64
}

func newScalerProcessor(s *Scaler) *scalerProcessor {
	return &scalerProcessor{
		method: s.Method,
		center: s.Center,
		scale:  s.Scale,
		min:    s.Min,
		max:    s.Max,
	}
}

func (p *scalerProcessor) apply(features []float64) {
	switch p.method {
	case "standard":
		// (x - mean) / std
		for i := range features {
			if i < len(p.center) && i < len(p.scale) && p.scale[i] != 0 {
				features[i] = (features[i] - p.center[i]) / p.scale[i]
			}
		}
	case "minmax":
		// (x - min) / (max - min)
		for i := range features {
			if i < len(p.min) && i < len(p.max) {
				denom := p.max[i] - p.min[i]
				if denom != 0 {
					features[i] = (features[i] - p.min[i]) / denom
				}
			}
		}
	case "robust":
		// (x - center) / scale where center=median, scale=IQR
		for i := range features {
			if i < len(p.center) && i < len(p.scale) && p.scale[i] != 0 {
				features[i] = (features[i] - p.center[i]) / p.scale[i]
			}
		}
	case "maxabs":
		// x / max(|x|) — scale is max absolute value per feature
		for i := range features {
			if i < len(p.scale) && p.scale[i] != 0 {
				features[i] = features[i] / p.scale[i]
			}
		}
	}
}

// imputerProcessor fills NaN values.
type imputerProcessor struct {
	fillValues []float64
}

func newImputerProcessor(imp *Imputer) *imputerProcessor {
	return &imputerProcessor{fillValues: imp.FillValues}
}

func (p *imputerProcessor) apply(features []float64) {
	for i := range features {
		if math.IsNaN(features[i]) && i < len(p.fillValues) {
			features[i] = p.fillValues[i]
		}
	}
}
