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

// svmEngine runs inference for SVM models with RBF, linear, polynomial,
// and sigmoid kernels.
type svmEngine struct {
	svm *SVMModel
}

func newSVMEngine(svm *SVMModel) *svmEngine {
	return &svmEngine{svm: svm}
}

// predictSingle computes the SVM decision function for one sample.
// Returns [num_classes] raw decision values (before activation).
func (e *svmEngine) predictSingle(features []float64) []float64 {
	svm := e.svm
	numSV := len(svm.SupportVectors)
	numClasses := len(svm.DualCoefficients)
	if numClasses == 0 {
		numClasses = 1
	}

	// Compute kernel values between input and all support vectors
	kernelValues := make([]float64, numSV)
	for i, sv := range svm.SupportVectors {
		kernelValues[i] = e.kernel(features, sv)
	}

	// Compute decision function: sum(dual_coef * kernel) + intercept
	outputs := make([]float64, numClasses)
	for c := 0; c < numClasses; c++ {
		var sum float64
		coeffs := svm.DualCoefficients[c]
		for i := 0; i < numSV && i < len(coeffs); i++ {
			sum += coeffs[i] * kernelValues[i]
		}
		if c < len(svm.Intercept) {
			sum += svm.Intercept[c]
		}
		outputs[c] = sum
	}

	return outputs
}

// kernel computes the kernel function between two vectors.
func (e *svmEngine) kernel(x, y []float64) float64 {
	switch e.svm.Kernel {
	case "rbf":
		return e.rbfKernel(x, y)
	case "linear":
		return dotProduct(x, y)
	case "poly":
		return e.polyKernel(x, y)
	case "sigmoid":
		return e.sigmoidKernel(x, y)
	default:
		return e.rbfKernel(x, y)
	}
}

func (e *svmEngine) rbfKernel(x, y []float64) float64 {
	var sqDist float64
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	for i := 0; i < n; i++ {
		d := x[i] - y[i]
		sqDist += d * d
	}
	return math.Exp(-e.svm.Gamma * sqDist)
}

func (e *svmEngine) polyKernel(x, y []float64) float64 {
	dot := dotProduct(x, y)
	return math.Pow(e.svm.Gamma*dot+e.svm.Coef0, float64(e.svm.Degree))
}

func (e *svmEngine) sigmoidKernel(x, y []float64) float64 {
	dot := dotProduct(x, y)
	return math.Tanh(e.svm.Gamma*dot + e.svm.Coef0)
}

func dotProduct(x, y []float64) float64 {
	var sum float64
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	for i := 0; i < n; i++ {
		sum += x[i] * y[i]
	}
	return sum
}
