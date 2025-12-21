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
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
)

func init() {
	RegisterBackend(&onnxBackend{})
}

// onnxBackend implements Backend using ONNX Runtime (Linux/Windows).
// This is the fastest backend for CPU and CUDA inference.
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
type onnxBackend struct {
	// GPU mode configuration
	gpuMode   GPUMode
	gpuModeMu sync.RWMutex

	// Track whether CUDA is enabled (cached after first detection)
	cudaEnabled     bool
	cudaEnabledOnce sync.Once
}

func (b *onnxBackend) Type() BackendType {
	return BackendONNX
}

func (b *onnxBackend) Name() string {
	if b.useCUDA() {
		return "ONNX Runtime (CUDA)"
	}
	return "ONNX Runtime (CPU)"
}

func (b *onnxBackend) Available() bool {
	// ONNX is available if the build includes the ONNX Runtime
	// The build tags ensure this file is only included when ONNX is available
	return true
}

func (b *onnxBackend) Priority() int {
	// Highest priority when available
	return 10
}

func (b *onnxBackend) CreateSession(opts ...options.WithOption) (*hugot.Session, error) {
	var baseOpts []options.WithOption

	// Check for custom library path from environment
	if libPath := getOnnxLibraryPath(); libPath != "" {
		baseOpts = append(baseOpts, options.WithOnnxLibraryPath(libPath))
	}

	if b.useCUDA() {
		baseOpts = append(baseOpts, options.WithCuda(nil))
	}

	opts = append(baseOpts, opts...)
	return hugot.NewORTSession(opts...)
}

// getOnnxLibraryPath returns the directory containing libonnxruntime.so from environment.
// Checks ONNXRUNTIME_ROOT first, then LD_LIBRARY_PATH.
func getOnnxLibraryPath() string {
	platform := runtime.GOOS + "-" + runtime.GOARCH

	// Check ONNXRUNTIME_ROOT (set by Makefile)
	if root := os.Getenv("ONNXRUNTIME_ROOT"); root != "" {
		// Try platform-specific path first
		platformDir := filepath.Join(root, platform, "lib")
		if _, err := os.Stat(filepath.Join(platformDir, "libonnxruntime.so")); err == nil {
			return platformDir
		}
		// Try direct lib path
		directDir := filepath.Join(root, "lib")
		if _, err := os.Stat(filepath.Join(directDir, "libonnxruntime.so")); err == nil {
			return directDir
		}
	}

	// Check LD_LIBRARY_PATH
	if ldPath := os.Getenv("LD_LIBRARY_PATH"); ldPath != "" {
		// LD_LIBRARY_PATH can have multiple paths separated by ':'
		for _, dir := range filepath.SplitList(ldPath) {
			if _, err := os.Stat(filepath.Join(dir, "libonnxruntime.so")); err == nil {
				return dir
			}
		}
	}

	return ""
}

// SetGPUMode sets the GPU mode for this backend.
// Must be called before any sessions are created to take effect.
func (b *onnxBackend) SetGPUMode(mode GPUMode) {
	b.gpuModeMu.Lock()
	defer b.gpuModeMu.Unlock()
	b.gpuMode = mode
}

// GetGPUMode returns the current GPU mode.
func (b *onnxBackend) GetGPUMode() GPUMode {
	b.gpuModeMu.RLock()
	defer b.gpuModeMu.RUnlock()
	if b.gpuMode == "" {
		return GPUModeAuto
	}
	return b.gpuMode
}

// useCUDA determines if CUDA should be used.
// Uses auto-detection by default, can be overridden via SetGPUMode().
func (b *onnxBackend) useCUDA() bool {
	b.cudaEnabledOnce.Do(func() {
		b.gpuModeMu.RLock()
		mode := b.gpuMode
		b.gpuModeMu.RUnlock()

		b.cudaEnabled = ShouldUseGPU(mode)
	})
	return b.cudaEnabled
}
