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

//go:build onnx && ORT && darwin

package hugot

import (
	"os"
	"path/filepath"
	"sync"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/ortgenai"
)

func init() {
	RegisterBackend(&onnxDarwinBackend{})

	// Auto-detect and set GenAI library path
	if genaiPath := getGenAILibraryPath(); genaiPath != "" {
		ortgenai.SetSharedLibraryPath(genaiPath)
	}
}

// onnxDarwinBackend implements Backend using ONNX Runtime with CoreML on macOS.
// This provides hardware acceleration on Apple Silicon and Intel Macs via
// Neural Engine, GPU, or CPU execution depending on the model and hardware.
//
// Runtime Requirements:
//   - Set DYLD_LIBRARY_PATH before running:
//     export DYLD_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib
//
// Build Requirements:
//   - CGO must be enabled (CGO_ENABLED=1)
//   - ONNX Runtime libraries must be available at link time
//   - libomp installed (brew install libomp)
//   - Tokenizers library available (CGO_LDFLAGS)
type onnxDarwinBackend struct {
	gpuMode   GPUMode
	gpuModeMu sync.RWMutex
}

func (b *onnxDarwinBackend) Type() BackendType {
	return BackendONNX
}

func (b *onnxDarwinBackend) Name() string {
	return "ONNX Runtime (CoreML)"
}

func (b *onnxDarwinBackend) Available() bool {
	// CoreML is always available on macOS via ONNX Runtime
	return true
}

func (b *onnxDarwinBackend) Priority() int {
	// Highest priority when available
	return 10
}

func (b *onnxDarwinBackend) CreateSession(opts ...options.WithOption) (*hugot.Session, error) {
	var baseOpts []options.WithOption

	// Check for custom library path from environment
	libPath := getOnnxLibraryPath()
	if libPath != "" {
		baseOpts = append(baseOpts, options.WithOnnxLibraryPath(libPath))
	}

	// CoreML Execution Provider is DISABLED by default.
	//
	// There are multiple reasons to avoid CoreML EP:
	//
	// 1. CRITICAL: CoreML EP cannot handle ONNX models with external data files
	//    (.onnx.data). During model optimization, CoreML loses the path context
	//    needed to load external weights, causing "model_path must not be empty"
	//    errors. Most generative models use external data for their large weights.
	//
	// 2. Performance: Benchmarks show pure ONNX Runtime CPU significantly
	//    outperforms CoreML EP for embedding models due to bridge layer overhead:
	//    - CoreML (100 individual): 1.63s total, 16.33ms/request
	//    - Pure ONNX CPU (batch 100): 129ms total, 1.29ms/text (12.7x faster)
	//    - Single requests: CoreML 14.5ms vs ONNX CPU 1.95ms (7.4x faster)
	//
	// 3. Batching: CoreML EP cannot handle dynamic batch sizes > 1.
	//
	// To force-enable CoreML (experimental), set TERMITE_FORCE_COREML=1.
	// This may work for models with embedded weights and fixed batch sizes.
	// See pkg/termite/lib/embeddings/benchmark_batch_test.go for details.
	forceCoreML := os.Getenv("TERMITE_FORCE_COREML") != ""
	if forceCoreML && b.GetGPUMode() != GPUModeOff {
		coremlConfig := b.getCoreMLConfig()
		baseOpts = append(baseOpts, options.WithCoreML(coremlConfig))
	}

	opts = append(baseOpts, opts...)
	return hugot.NewORTSession(opts...)
}

// getCoreMLConfig returns the CoreML configuration based on the current GPU mode.
// Maps GPUMode to CoreML MLComputeUnits:
//   - GPUModeAuto/GPUModeCoreML: "ALL" (Neural Engine > GPU > CPU)
//   - GPUModeOff: "CPUOnly"
//   - GPUModeCuda/GPUModeTpu: "ALL" (not applicable on Mac, fallback to best available)
func (b *onnxDarwinBackend) getCoreMLConfig() map[string]string {
	mode := b.GetGPUMode()

	config := make(map[string]string)

	// Map GPUMode to MLComputeUnits
	switch mode {
	case GPUModeOff:
		config["MLComputeUnits"] = "CPUOnly"
	case GPUModeAuto, GPUModeCoreML, GPUModeCuda, GPUModeTpu, "":
		// Use ALL to let CoreML pick the best available hardware
		// (Neural Engine > GPU > CPU based on model and operation)
		config["MLComputeUnits"] = "ALL"
	default:
		config["MLComputeUnits"] = "ALL"
	}

	// Enable compute plan profiling when TERMITE_DEBUG is set
	// This logs which hardware (ANE/GPU/CPU) executes each operation
	if os.Getenv("TERMITE_DEBUG") != "" {
		config["ProfileComputePlan"] = "1"
	}

	return config
}

// getOnnxLibraryPath returns the directory containing libonnxruntime.dylib from environment.
// Checks ONNXRUNTIME_ROOT first, then DYLD_LIBRARY_PATH.
func getOnnxLibraryPath() string {
	// Check ONNXRUNTIME_ROOT (set by Makefile)
	if root := os.Getenv("ONNXRUNTIME_ROOT"); root != "" {
		// Try platform-specific path first
		platformDir := filepath.Join(root, "darwin-arm64", "lib")
		if _, err := os.Stat(filepath.Join(platformDir, "libonnxruntime.dylib")); err == nil {
			return platformDir
		}
		// Try direct lib path
		directDir := filepath.Join(root, "lib")
		if _, err := os.Stat(filepath.Join(directDir, "libonnxruntime.dylib")); err == nil {
			return directDir
		}
	}

	// Check DYLD_LIBRARY_PATH
	if dyldPath := os.Getenv("DYLD_LIBRARY_PATH"); dyldPath != "" {
		if _, err := os.Stat(filepath.Join(dyldPath, "libonnxruntime.dylib")); err == nil {
			return dyldPath
		}
	}

	return ""
}

// getGenAILibraryPath returns the path to libonnxruntime-genai.dylib.
// Checks ORTGENAI_DYLIB_PATH first, then looks in the same locations as ONNX Runtime.
func getGenAILibraryPath() string {
	const libName = "libonnxruntime-genai.dylib"

	// Check explicit ORTGENAI_DYLIB_PATH first
	if path := os.Getenv("ORTGENAI_DYLIB_PATH"); path != "" {
		return path
	}

	// Check ONNXRUNTIME_ROOT (GenAI libs are often installed alongside ONNX Runtime)
	if root := os.Getenv("ONNXRUNTIME_ROOT"); root != "" {
		// Try platform-specific path first
		platformPath := filepath.Join(root, "darwin-arm64", "lib", libName)
		if _, err := os.Stat(platformPath); err == nil {
			return platformPath
		}
		// Try direct lib path
		directPath := filepath.Join(root, "lib", libName)
		if _, err := os.Stat(directPath); err == nil {
			return directPath
		}
	}

	// Check DYLD_LIBRARY_PATH
	if dyldPath := os.Getenv("DYLD_LIBRARY_PATH"); dyldPath != "" {
		libPath := filepath.Join(dyldPath, libName)
		if _, err := os.Stat(libPath); err == nil {
			return libPath
		}
	}

	return ""
}

// SetGPUMode sets the GPU mode. On macOS, CoreML automatically uses the best
// available accelerator (Neural Engine, GPU, or CPU), so this is mostly informational.
func (b *onnxDarwinBackend) SetGPUMode(mode GPUMode) {
	b.gpuModeMu.Lock()
	defer b.gpuModeMu.Unlock()
	b.gpuMode = mode
}

// GetGPUMode returns the current GPU mode.
func (b *onnxDarwinBackend) GetGPUMode() GPUMode {
	b.gpuModeMu.RLock()
	defer b.gpuModeMu.RUnlock()
	if b.gpuMode == "" {
		return GPUModeAuto
	}
	return b.gpuMode
}
