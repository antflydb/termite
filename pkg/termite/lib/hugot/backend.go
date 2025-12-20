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
// Multiple backends can coexist in a single binary when built with combined tags:
//
//	go build -tags="onnx,ORT,xla,XLA" ./cmd/termite
//
// Backend selection at runtime follows a configurable priority order (default: ONNX > XLA > Go).
// Models can restrict which backends they support via their manifest.
package hugot

import (
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
)

// BackendType identifies the inference backend
type BackendType string

const (
	// BackendGo is the pure Go backend (goMLX) - always available, no CGO required
	BackendGo BackendType = "go"

	// BackendONNX is the ONNX Runtime backend - fast CPU/GPU inference
	BackendONNX BackendType = "onnx"

	// BackendXLA is the GoMLX XLA backend - supports TPU/CUDA/CPU via PJRT
	BackendXLA BackendType = "xla"
)

// Backend represents an inference backend that can create Hugot sessions.
// Backends self-register via init() functions in their respective files.
type Backend interface {
	// Type returns the backend type identifier
	Type() BackendType

	// Name returns a human-readable name (e.g., "ONNX Runtime (CUDA)")
	Name() string

	// Available returns true if this backend can be used in the current environment.
	// This checks for required libraries, hardware, etc.
	Available() bool

	// Priority returns the default priority (lower = higher priority).
	// Used when no explicit priority is configured.
	// Recommended values: 10 for ONNX, 20 for XLA, 100 for Go (fallback)
	Priority() int

	// CreateSession creates a new Hugot session with the given options.
	CreateSession(opts ...options.WithOption) (*hugot.Session, error)
}

var (
	// registry holds all registered backends
	registry   = make(map[BackendType]Backend)
	registryMu sync.RWMutex

	// priority defines the order to try backends when selecting default.
	// Configurable via SetPriority(). Default: ONNX > XLA > Go
	defaultPriority = []BackendType{BackendONNX, BackendXLA, BackendGo}
	configPriority  []BackendType
	priorityMu      sync.RWMutex
)

// RegisterBackend registers a backend. Called by backend implementations in init().
// Thread-safe. Later registrations for the same type overwrite earlier ones.
func RegisterBackend(b Backend) {
	registryMu.Lock()
	defer registryMu.Unlock()
	registry[b.Type()] = b
}

// GetBackend returns the backend for the given type, if registered.
func GetBackend(t BackendType) (Backend, bool) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	b, ok := registry[t]
	return b, ok
}

// ListRegistered returns all registered backends (available or not).
// Sorted by priority (lowest priority number first).
func ListRegistered() []Backend {
	registryMu.RLock()
	defer registryMu.RUnlock()

	backends := make([]Backend, 0, len(registry))
	for _, b := range registry {
		backends = append(backends, b)
	}

	// Sort by priority for consistent ordering
	sort.Slice(backends, func(i, j int) bool {
		return backends[i].Priority() < backends[j].Priority()
	})

	return backends
}

// ListAvailable returns all backends that are currently available for use.
// Sorted by configured priority order.
func ListAvailable() []Backend {
	priority := GetPriority()

	registryMu.RLock()
	defer registryMu.RUnlock()

	// First add backends in priority order
	result := make([]Backend, 0, len(registry))
	seen := make(map[BackendType]bool)

	for _, t := range priority {
		if b, ok := registry[t]; ok && b.Available() {
			result = append(result, b)
			seen[t] = true
		}
	}

	// Then add any remaining available backends not in priority list
	for t, b := range registry {
		if !seen[t] && b.Available() {
			result = append(result, b)
		}
	}

	return result
}

// SetPriority sets the backend selection priority order.
// When selecting a default backend, the first available backend in this order is used.
// Call before creating any sessions to take effect.
func SetPriority(order []BackendType) {
	priorityMu.Lock()
	defer priorityMu.Unlock()
	configPriority = make([]BackendType, len(order))
	copy(configPriority, order)
}

// GetPriority returns the current backend priority order.
// Returns the configured priority if set, otherwise the default.
func GetPriority() []BackendType {
	priorityMu.RLock()
	defer priorityMu.RUnlock()
	if len(configPriority) > 0 {
		result := make([]BackendType, len(configPriority))
		copy(result, configPriority)
		return result
	}
	result := make([]BackendType, len(defaultPriority))
	copy(result, defaultPriority)
	return result
}

// GetDefaultBackend returns the first available backend according to priority order.
// Returns nil if no backends are available.
func GetDefaultBackend() Backend {
	priority := GetPriority()

	registryMu.RLock()
	defer registryMu.RUnlock()

	// Try backends in priority order
	for _, t := range priority {
		if b, ok := registry[t]; ok && b.Available() {
			return b
		}
	}

	// Fallback: any available backend
	for _, b := range registry {
		if b.Available() {
			return b
		}
	}

	return nil
}

// GetBackendWithFallback attempts to get the preferred backend, falling back to
// alternatives if unavailable. Returns the backend and its type.
func GetBackendWithFallback(preferred BackendType) (Backend, BackendType, error) {
	// Try preferred first
	if b, ok := GetBackend(preferred); ok && b.Available() {
		return b, preferred, nil
	}

	// Fallback to default
	b := GetDefaultBackend()
	if b == nil {
		return nil, "", fmt.Errorf("no available backends (preferred: %s)", preferred)
	}

	return b, b.Type(), nil
}

// ParseBackendType parses a string into BackendType.
// Returns an error for unrecognized values.
func ParseBackendType(s string) (BackendType, error) {
	switch s {
	case "onnx", "ONNX":
		return BackendONNX, nil
	case "xla", "XLA":
		return BackendXLA, nil
	case "go", "Go", "pure-go", "gomlx":
		return BackendGo, nil
	default:
		return "", fmt.Errorf("unknown backend type: %q (valid: onnx, xla, go)", s)
	}
}

// BackendTypeStrings returns valid backend type strings for documentation/validation.
func BackendTypeStrings() []string {
	return []string{"onnx", "xla", "go"}
}

// DeviceType identifies the hardware device for inference
type DeviceType string

const (
	// DeviceAuto auto-detects the best available device (default)
	DeviceAuto DeviceType = "auto"

	// DeviceCUDA uses NVIDIA CUDA GPU
	DeviceCUDA DeviceType = "cuda"

	// DeviceCoreML uses Apple CoreML (macOS only)
	DeviceCoreML DeviceType = "coreml"

	// DeviceTPU uses Google TPU
	DeviceTPU DeviceType = "tpu"

	// DeviceCPU forces CPU-only inference
	DeviceCPU DeviceType = "cpu"
)

// BackendSpec combines a backend type with a device specification.
// Used for configuring backend priority with device preferences.
type BackendSpec struct {
	Backend BackendType
	Device  DeviceType
}

// String returns the string representation (e.g., "onnx:cuda" or "go")
func (s BackendSpec) String() string {
	if s.Device == DeviceAuto || s.Device == "" {
		return string(s.Backend)
	}
	return string(s.Backend) + ":" + string(s.Device)
}

// ParseBackendSpec parses a "backend" or "backend:device" string.
// Examples: "onnx", "onnx:cuda", "xla:tpu", "go"
func ParseBackendSpec(s string) (BackendSpec, error) {
	parts := strings.SplitN(s, ":", 2)

	backend, err := ParseBackendType(parts[0])
	if err != nil {
		return BackendSpec{}, err
	}

	spec := BackendSpec{Backend: backend, Device: DeviceAuto}

	if len(parts) == 2 {
		device, err := ParseDeviceType(parts[1])
		if err != nil {
			return BackendSpec{}, err
		}
		spec.Device = device
	}

	return spec, nil
}

// ParseDeviceType parses a string into DeviceType.
func ParseDeviceType(s string) (DeviceType, error) {
	switch strings.ToLower(s) {
	case "auto", "":
		return DeviceAuto, nil
	case "cuda", "gpu":
		return DeviceCUDA, nil
	case "coreml":
		return DeviceCoreML, nil
	case "tpu":
		return DeviceTPU, nil
	case "cpu", "off":
		return DeviceCPU, nil
	default:
		return "", fmt.Errorf("unknown device type: %q (valid: auto, cuda, coreml, tpu, cpu)", s)
	}
}

// ParseBackendPriority parses a list of backend:device strings into BackendSpecs.
func ParseBackendPriority(priority []string) ([]BackendSpec, error) {
	specs := make([]BackendSpec, 0, len(priority))
	for _, s := range priority {
		spec, err := ParseBackendSpec(s)
		if err != nil {
			return nil, fmt.Errorf("invalid backend priority %q: %w", s, err)
		}
		specs = append(specs, spec)
	}
	return specs, nil
}

// ToGPUMode converts DeviceType to the legacy GPUMode for backward compatibility.
func (d DeviceType) ToGPUMode() GPUMode {
	switch d {
	case DeviceAuto:
		return GPUModeAuto
	case DeviceCUDA:
		return GPUModeCuda
	case DeviceCoreML:
		return GPUModeCoreML
	case DeviceTPU:
		return GPUModeTpu
	case DeviceCPU:
		return GPUModeOff
	default:
		return GPUModeAuto
	}
}
