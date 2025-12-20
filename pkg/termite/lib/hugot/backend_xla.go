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
	"path/filepath"
	"strings"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
)

func init() {
	// Only use bundled PJRT plugin as fallback if no plugins in standard locations.
	// This allows user-installed plugins (via pjrt_installer) to take precedence.
	if os.Getenv("PJRT_PLUGIN_LIBRARY_PATH") == "" && !hasPluginInStandardPaths() {
		if libPath := findBundledPJRT(); libPath != "" {
			os.Setenv("PJRT_PLUGIN_LIBRARY_PATH", libPath)
		}
	}
	RegisterBackend(&xlaBackend{})
}

// hasPluginInStandardPaths checks if PJRT plugins exist in standard go-xla search paths.
func hasPluginInStandardPaths() bool {
	home, _ := os.UserHomeDir()
	standardPaths := []string{
		"/usr/local/lib/gomlx/pjrt",
		"/usr/local/lib/go-xla",
		filepath.Join(home, ".local/lib/gomlx/pjrt"),
		filepath.Join(home, ".local/lib/go-xla"),
		filepath.Join(home, "Library/Application Support/go-xla"), // macOS
	}

	for _, dir := range standardPaths {
		matches, _ := filepath.Glob(filepath.Join(dir, "pjrt_*plugin*"))
		if len(matches) > 0 {
			return true
		}
	}
	return false
}

// findBundledPJRT looks for PJRT plugin bundled with the binary.
// Returns the directory path if found, empty string otherwise.
func findBundledPJRT() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	exe, err = filepath.EvalSymlinks(exe)
	if err != nil {
		return ""
	}

	// Check lib/ directory next to the binary
	libDir := filepath.Join(filepath.Dir(exe), "lib")
	matches, _ := filepath.Glob(filepath.Join(libDir, "pjrt_c_api_cpu*plugin*"))
	if len(matches) > 0 {
		return libDir
	}

	return ""
}

// xlaBackend implements Backend using the GoMLX XLA runtime.
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
type xlaBackend struct{}

func (b *xlaBackend) Type() BackendType {
	return BackendXLA
}

func (b *xlaBackend) Name() string {
	device := b.getDevice()
	switch device {
	case "tpu":
		return "GoMLX XLA (TPU)"
	case "cuda":
		return "GoMLX XLA (CUDA)"
	default:
		return "GoMLX XLA (CPU)"
	}
}

func (b *xlaBackend) Available() bool {
	// XLA is available if the build includes XLA support
	return true
}

func (b *xlaBackend) Priority() int {
	// Second priority after ONNX
	return 20
}

func (b *xlaBackend) CreateSession(opts ...options.WithOption) (*hugot.Session, error) {
	device := b.getDevice()

	switch device {
	case "tpu":
		opts = append([]options.WithOption{options.WithTPU()}, opts...)
	case "cuda":
		opts = append([]options.WithOption{options.WithCuda(nil)}, opts...)
	}

	return hugot.NewXLASession(opts...)
}

// getDevice returns the XLA device to use, with autodetection.
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
func (b *xlaBackend) getDevice() string {
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
		}
	}

	return "cpu"
}

// SetDevice sets the XLA device via GOMLX_BACKEND environment variable.
// Call before creating any sessions to take effect.
func (b *xlaBackend) SetDevice(device string) {
	switch device {
	case "cuda", "tpu", "cpu":
		os.Setenv("GOMLX_BACKEND", "xla:"+device)
	case "":
		os.Unsetenv("GOMLX_BACKEND")
	}
}
