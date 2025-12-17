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

//go:build xla && XLA

package hugot

import (
	"os"
	"strings"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
)

// getXLADevice returns the XLA device to use, with autodetection.
//
// Priority order:
//  1. GOMLX_BACKEND environment variable (explicit override)
//  2. Autodetection: TPU > CUDA > CPU
//
// Environment variable examples:
//   - GOMLX_BACKEND="xla:tpu"  -> Force TPU
//   - GOMLX_BACKEND="xla:cuda" -> Force NVIDIA GPU
//   - GOMLX_BACKEND="xla:cpu"  -> Force CPU
//   - (unset)                  -> Autodetect best available
//
// For TPU, the PJRT plugin is loaded from:
//  1. PJRT_PLUGIN_LIBRARY_PATH if set
//  2. Standard library paths (/usr/local/lib, LD_LIBRARY_PATH)
//  3. GKE TPU nodes have libtpu.so pre-installed
func getXLADevice() string {
	backend := os.Getenv("GOMLX_BACKEND")
	if backend != "" {
		// Handle "xla:device" format
		if strings.HasPrefix(backend, "xla:") {
			return strings.TrimPrefix(backend, "xla:")
		}

		// Handle bare device names
		switch strings.ToLower(backend) {
		case "tpu", "cuda", "cpu":
			return strings.ToLower(backend)
		}
	}

	// Autodetect: check what hardware is available
	// XLA supports: tpu, cuda, cpu (not coreml)
	gpuInfo := DetectGPU()
	if gpuInfo.Available {
		switch gpuInfo.Type {
		case "tpu", "cuda":
			return gpuInfo.Type
		default:
			// coreml and other types not supported by XLA
			return "cpu"
		}
	}

	return "cpu"
}

// isTPU reports whether TPU acceleration is enabled.
func isTPU() bool {
	return getXLADevice() == "tpu"
}

// isCUDA reports whether CUDA acceleration is enabled.
func isCUDA() bool {
	return getXLADevice() == "cuda"
}

// newSessionImpl creates a Hugot session using the GoMLX XLA backend.
// This enables hardware acceleration via TPU, CUDA, or optimized CPU.
//
// Hardware is autodetected in priority order: TPU > CUDA > CPU.
// Use GOMLX_BACKEND environment variable to override autodetection.
//
// XLA Backend Advantages:
//   - State-of-the-art JIT compilation for TPU/GPU/CPU
//   - Same engine powering JAX, TensorFlow, PyTorch/XLA
//   - Automatic kernel fusion and optimization
//   - Distributed execution support for multi-TPU/GPU
//
// Requirements for TPU (autodetected):
//   - GKE cluster with TPU node pool, or
//   - libtpu.so available in library path
//
// Requirements for CUDA (autodetected):
//   - NVIDIA GPU with CUDA support
//   - CUDA toolkit installed (libcudart.so in library path)
func newSessionImpl(opts ...options.WithOption) (*hugot.Session, error) {
	device := getXLADevice()

	// Add device-specific options
	switch device {
	case "tpu":
		opts = append([]options.WithOption{options.WithTPU()}, opts...)
	case "cuda":
		opts = append([]options.WithOption{options.WithCuda(nil)}, opts...)
	}

	return hugot.NewXLASession(opts...)
}

// backendNameImpl returns a human-readable name of the XLA backend.
func backendNameImpl() string {
	device := getXLADevice()
	switch device {
	case "tpu":
		return "GoMLX XLA (TPU)"
	case "cuda":
		return "GoMLX XLA (CUDA)"
	default:
		return "GoMLX XLA (CPU)"
	}
}

// SetGPUMode sets the GPU mode for XLA backend.
//
//   - GPUModeAuto: Autodetect best available (TPU > CUDA > CPU)
//   - GPUModeCuda: Force CUDA
//   - GPUModeTpu: Force TPU
//   - GPUModeOff: Force CPU only
func SetGPUMode(mode GPUMode) {
	switch mode {
	case GPUModeCuda:
		os.Setenv("GOMLX_BACKEND", "xla:cuda")
	case GPUModeTpu:
		os.Setenv("GOMLX_BACKEND", "xla:tpu")
	case GPUModeOff:
		os.Setenv("GOMLX_BACKEND", "xla:cpu")
	case GPUModeAuto:
		// Clear any override to allow autodetection
		os.Unsetenv("GOMLX_BACKEND")
	}
}

// GetGPUMode returns the current GPU mode based on GOMLX_BACKEND.
func GetGPUMode() GPUMode {
	device := getXLADevice()
	switch device {
	case "cuda":
		return GPUModeCuda
	case "tpu":
		return GPUModeTpu
	default:
		return GPUModeOff
	}
}
