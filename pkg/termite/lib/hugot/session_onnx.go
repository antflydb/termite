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

//go:build onnx && ORT && !darwin

package hugot

import (
	"sync"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
)

var (
	// configuredGPUMode is set via SetGPUMode or config
	configuredGPUMode GPUMode = GPUModeAuto
	gpuModeMu         sync.RWMutex

	// Track whether CUDA is actually being used (after detection)
	cudaEnabled     bool
	cudaEnabledOnce sync.Once
)

// SetGPUMode sets the GPU mode for future sessions.
// Call this before creating any sessions to override auto-detection.
//
// Modes:
//   - "auto": Auto-detect GPU availability (default)
//   - "tpu": Force TPU
//   - "cuda": Force CUDA
//   - "coreml": Force CoreML (macOS)
//   - "off": CPU only
//
// Configure via TERMITE_GPU env var (viper auto-binding) or termite.yaml config.
func SetGPUMode(mode GPUMode) {
	gpuModeMu.Lock()
	defer gpuModeMu.Unlock()
	configuredGPUMode = mode
}

// GetGPUMode returns the currently configured GPU mode
func GetGPUMode() GPUMode {
	gpuModeMu.RLock()
	defer gpuModeMu.RUnlock()
	return configuredGPUMode
}

// useCUDA reports whether CUDA acceleration should be enabled.
// Uses auto-detection by default, can be overridden via SetGPUMode().
//
// Configure via TERMITE_GPU env var (viper auto-binding) or termite.yaml config.
//
// CUDA requires:
//   - NVIDIA GPU with CUDA support
//   - CUDA toolkit and cuDNN installed
//   - onnxruntime-gpu package (not standard onnxruntime)
func useCUDA() bool {
	cudaEnabledOnce.Do(func() {
		gpuModeMu.RLock()
		mode := configuredGPUMode
		gpuModeMu.RUnlock()

		cudaEnabled = ShouldUseGPU(mode)
	})
	return cudaEnabled
}

// IsCUDAEnabled returns whether CUDA is currently enabled
// (after auto-detection has run)
func IsCUDAEnabled() bool {
	useCUDA() // Ensure detection has run
	return cudaEnabled
}

// newSessionImpl creates a Hugot session using ONNX Runtime.
// By default auto-detects GPU availability. Override via SetGPUMode() before calling NewSession().
//
// Configure via TERMITE_GPU env var (viper auto-binding) or termite.yaml config.
//
// Runtime Requirements:
//   - Set LD_LIBRARY_PATH before running:
//     export LD_LIBRARY_PATH=/path/to/onnxruntime/lib
//   - For CUDA: export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:/usr/local/cuda/lib64
//
// Build Requirements:
//   - CGO must be enabled (CGO_ENABLED=1)
//   - ONNX Runtime libraries must be available at link time
//   - Tokenizers library available (CGO_LDFLAGS)
func newSessionImpl(opts ...options.WithOption) (*hugot.Session, error) {
	if useCUDA() {
		cudaOpts := []options.WithOption{options.WithCuda(nil)}
		opts = append(cudaOpts, opts...)
	}
	return hugot.NewORTSession(opts...)
}

// backendNameImpl returns the name of the ONNX Runtime backend.
func backendNameImpl() string {
	if useCUDA() {
		return "ONNX Runtime (CUDA)"
	}
	return "ONNX Runtime (CPU)"
}
