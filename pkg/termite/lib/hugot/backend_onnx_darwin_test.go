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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCoreMLBackendRegistered(t *testing.T) {
	// Verify ONNX backend is registered
	backend, ok := GetBackend(BackendONNX)
	require.True(t, ok, "ONNX backend should be registered")

	// Verify it's available
	require.True(t, backend.Available(), "ONNX backend should be available on macOS")

	// Check the name indicates CoreML
	name := backend.Name()
	t.Logf("ONNX backend name: %s", name)
	assert.Contains(t, name, "CoreML", "Backend name should indicate CoreML on macOS")

	// Verify priority
	assert.Equal(t, 10, backend.Priority(), "ONNX should have priority 10")
}

func TestCoreMLBackendGPUMode(t *testing.T) {
	backend, ok := GetBackend(BackendONNX)
	require.True(t, ok, "ONNX backend should be registered")

	// Cast to onnxDarwinBackend to access GPUMode methods
	darwinBackend, ok := backend.(*onnxDarwinBackend)
	require.True(t, ok, "Backend should be onnxDarwinBackend on macOS")

	// Test default mode is auto
	mode := darwinBackend.GetGPUMode()
	assert.Equal(t, GPUModeAuto, mode, "Default GPU mode should be auto")

	// Test setting CoreML mode
	darwinBackend.SetGPUMode(GPUModeCoreML)
	mode = darwinBackend.GetGPUMode()
	assert.Equal(t, GPUModeCoreML, mode, "GPU mode should be coreml after setting")

	// Reset to auto
	darwinBackend.SetGPUMode(GPUModeAuto)
}

func TestCoreMLConfigWithGPUModes(t *testing.T) {
	backend, ok := GetBackend(BackendONNX)
	require.True(t, ok, "ONNX backend should be registered")

	darwinBackend, ok := backend.(*onnxDarwinBackend)
	require.True(t, ok, "Backend should be onnxDarwinBackend on macOS")

	tests := []struct {
		name              string
		gpuMode           GPUMode
		expectedMLUnits   string
	}{
		{
			name:            "auto mode uses ALL",
			gpuMode:         GPUModeAuto,
			expectedMLUnits: "ALL",
		},
		{
			name:            "coreml mode uses ALL",
			gpuMode:         GPUModeCoreML,
			expectedMLUnits: "ALL",
		},
		{
			name:            "off mode uses CPUOnly",
			gpuMode:         GPUModeOff,
			expectedMLUnits: "CPUOnly",
		},
		{
			name:            "cuda fallback to ALL on macOS",
			gpuMode:         GPUModeCuda,
			expectedMLUnits: "ALL",
		},
		{
			name:            "tpu fallback to ALL on macOS",
			gpuMode:         GPUModeTpu,
			expectedMLUnits: "ALL",
		},
		{
			name:            "empty string defaults to ALL",
			gpuMode:         "",
			expectedMLUnits: "ALL",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			darwinBackend.SetGPUMode(tt.gpuMode)
			config := darwinBackend.getCoreMLConfig()

			mlUnits, ok := config["MLComputeUnits"]
			require.True(t, ok, "Config should have MLComputeUnits")
			assert.Equal(t, tt.expectedMLUnits, mlUnits,
				"MLComputeUnits should be %s for GPU mode %s", tt.expectedMLUnits, tt.gpuMode)
		})
	}

	// Reset
	darwinBackend.SetGPUMode(GPUModeAuto)
}

func TestCoreMLConfigWithDebug(t *testing.T) {
	backend, ok := GetBackend(BackendONNX)
	require.True(t, ok, "ONNX backend should be registered")

	darwinBackend, ok := backend.(*onnxDarwinBackend)
	require.True(t, ok, "Backend should be onnxDarwinBackend on macOS")

	// Save and restore TERMITE_DEBUG
	originalDebug := os.Getenv("TERMITE_DEBUG")
	defer func() {
		if originalDebug == "" {
			os.Unsetenv("TERMITE_DEBUG")
		} else {
			os.Setenv("TERMITE_DEBUG", originalDebug)
		}
	}()

	// Test without debug
	os.Unsetenv("TERMITE_DEBUG")
	config := darwinBackend.getCoreMLConfig()
	_, hasProfile := config["ProfileComputePlan"]
	assert.False(t, hasProfile, "ProfileComputePlan should not be set without TERMITE_DEBUG")

	// Test with debug enabled
	os.Setenv("TERMITE_DEBUG", "1")
	config = darwinBackend.getCoreMLConfig()
	profileValue, hasProfile := config["ProfileComputePlan"]
	assert.True(t, hasProfile, "ProfileComputePlan should be set with TERMITE_DEBUG=1")
	assert.Equal(t, "1", profileValue, "ProfileComputePlan should be '1'")
}

func TestCoreMLLibraryPathDetection(t *testing.T) {
	// Test getOnnxLibraryPath function
	libPath := getOnnxLibraryPath()
	t.Logf("Detected ONNX library path: %s", libPath)

	// If ONNXRUNTIME_ROOT is set, we should find the library
	if onnxRoot := os.Getenv("ONNXRUNTIME_ROOT"); onnxRoot != "" {
		assert.NotEmpty(t, libPath, "Should find ONNX library when ONNXRUNTIME_ROOT is set")
	}

	// Test getGenAILibraryPath function
	genaiPath := getGenAILibraryPath()
	t.Logf("Detected GenAI library path: %s", genaiPath)

	// If ORTGENAI_DYLIB_PATH is set, we should find it
	if explicitPath := os.Getenv("ORTGENAI_DYLIB_PATH"); explicitPath != "" {
		assert.Equal(t, explicitPath, genaiPath,
			"Should return ORTGENAI_DYLIB_PATH when set")
	}
}
