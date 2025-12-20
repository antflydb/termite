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
	"sync"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
)

func init() {
	RegisterBackend(&onnxDarwinBackend{})
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
	// Prepend CoreML provider - user options can override if needed
	coremlOpts := []options.WithOption{options.WithCoreML(nil)}
	opts = append(coremlOpts, opts...)
	return hugot.NewORTSession(opts...)
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
