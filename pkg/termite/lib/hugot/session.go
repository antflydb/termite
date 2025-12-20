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

// Package hugot provides a unified interface for creating Hugot sessions
// with multi-backend support.
//
// Backends are selected based on build tags and availability:
//   - Pure Go (goMLX): Always available, no CGO required
//   - ONNX Runtime: Fastest inference, requires -tags="onnx,ORT"
//   - XLA: TPU/CUDA support via PJRT, requires -tags="xla,XLA"
//
// Multiple backends can coexist in a single binary:
//
//	go build -tags="onnx,ORT,xla,XLA" ./cmd/termite
//
// Example usage:
//
//	session, err := hugot.NewSession()  // Uses best available backend
//	manager := hugot.NewSessionManager()  // For explicit backend control
package hugot

import (
	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
)

// NewSession creates a new Hugot session using the best available backend.
//
// Backend selection follows the configured priority order (default: ONNX > XLA > Go).
// Use SetPriority() to customize the order before creating sessions.
//
// For multi-model scenarios, use NewSessionManager() to share sessions across models.
func NewSession(opts ...options.WithOption) (*hugot.Session, error) {
	backend := GetDefaultBackend()
	if backend == nil {
		// This shouldn't happen since Go backend is always registered,
		// but handle gracefully
		return nil, nil
	}
	return backend.CreateSession(opts...)
}

// NewSessionOrUseExisting returns the provided session if non-nil, otherwise creates a new one.
// This is useful when you want to share a single session across multiple models/pipelines.
//
// IMPORTANT: With ONNX Runtime backend, only ONE session can be active at a time.
// Use SessionManager for explicit control over session sharing.
//
// Example:
//
//	// Create shared session once
//	sharedSession, err := hugot.NewSession()
//	if err != nil {
//		return err
//	}
//	defer sharedSession.Destroy()
//
//	// Reuse session for multiple models
//	session1, _ := hugot.NewSessionOrUseExisting(sharedSession)  // Returns sharedSession
//	session2, _ := hugot.NewSessionOrUseExisting(nil)            // Creates new session
func NewSessionOrUseExisting(existingSession *hugot.Session, opts ...options.WithOption) (*hugot.Session, error) {
	if existingSession != nil {
		return existingSession, nil
	}
	return NewSession(opts...)
}

// BackendName returns a human-readable name of the default backend being used.
// Useful for logging and debugging.
//
// Example outputs:
//   - "ONNX Runtime (CUDA)"
//   - "ONNX Runtime (CoreML)"
//   - "GoMLX XLA (TPU)"
//   - "goMLX (Pure Go)"
func BackendName() string {
	b := GetDefaultBackend()
	if b == nil {
		return "No backend available"
	}
	return b.Name()
}

// GetGPUInfo returns information about detected GPU hardware.
// This is always available regardless of build tags.
func GetGPUInfo() GPUInfo {
	return DetectGPU()
}

// SetGPUMode sets the GPU mode for backends that support it.
// Call before creating any sessions to take effect.
//
// Modes:
//   - GPUModeAuto: Auto-detect GPU availability (default)
//   - GPUModeTpu: Force TPU (XLA backend)
//   - GPUModeCuda: Force CUDA
//   - GPUModeCoreML: Force CoreML (macOS ONNX)
//   - GPUModeOff: CPU only
//
// Configure via TERMITE_GPU env var or termite.yaml config.
func SetGPUMode(mode GPUMode) {
	// Set for ONNX backend if available
	if b, ok := GetBackend(BackendONNX); ok {
		if setter, ok := b.(interface{ SetGPUMode(GPUMode) }); ok {
			setter.SetGPUMode(mode)
		}
	}

	// Set for XLA backend if available (via environment variable)
	if b, ok := GetBackend(BackendXLA); ok {
		if setter, ok := b.(interface{ SetDevice(string) }); ok {
			switch mode {
			case GPUModeCuda:
				setter.SetDevice("cuda")
			case GPUModeTpu:
				setter.SetDevice("tpu")
			case GPUModeOff:
				setter.SetDevice("cpu")
			case GPUModeAuto:
				setter.SetDevice("")
			}
		}
	}
}

// GetGPUMode returns the current GPU mode.
// Returns GPUModeAuto if no specific mode has been set.
func GetGPUMode() GPUMode {
	// Check ONNX backend first
	if b, ok := GetBackend(BackendONNX); ok {
		if getter, ok := b.(interface{ GetGPUMode() GPUMode }); ok {
			return getter.GetGPUMode()
		}
	}

	// Default to auto
	return GPUModeAuto
}
