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

package hugot

import (
	"testing"
)

func TestParseBackendSpec(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    BackendSpec
		wantErr bool
	}{
		{
			name:  "backend only - onnx",
			input: "onnx",
			want:  BackendSpec{Backend: BackendONNX, Device: DeviceAuto},
		},
		{
			name:  "backend only - xla",
			input: "xla",
			want:  BackendSpec{Backend: BackendXLA, Device: DeviceAuto},
		},
		{
			name:  "backend only - go",
			input: "go",
			want:  BackendSpec{Backend: BackendGo, Device: DeviceAuto},
		},
		{
			name:  "backend with device - onnx:cuda",
			input: "onnx:cuda",
			want:  BackendSpec{Backend: BackendONNX, Device: DeviceCUDA},
		},
		{
			name:  "backend with device - xla:tpu",
			input: "xla:tpu",
			want:  BackendSpec{Backend: BackendXLA, Device: DeviceTPU},
		},
		{
			name:  "backend with device - onnx:coreml",
			input: "onnx:coreml",
			want:  BackendSpec{Backend: BackendONNX, Device: DeviceCoreML},
		},
		{
			name:  "backend with device - onnx:cpu",
			input: "onnx:cpu",
			want:  BackendSpec{Backend: BackendONNX, Device: DeviceCPU},
		},
		{
			name:  "backend with auto device",
			input: "onnx:auto",
			want:  BackendSpec{Backend: BackendONNX, Device: DeviceAuto},
		},
		{
			name:  "gpu alias for cuda",
			input: "onnx:gpu",
			want:  BackendSpec{Backend: BackendONNX, Device: DeviceCUDA},
		},
		{
			name:  "off alias for cpu",
			input: "xla:off",
			want:  BackendSpec{Backend: BackendXLA, Device: DeviceCPU},
		},
		{
			name:    "invalid backend",
			input:   "invalid",
			wantErr: true,
		},
		{
			name:    "invalid device",
			input:   "onnx:invalid",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseBackendSpec(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseBackendSpec(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("ParseBackendSpec(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestParseBackendPriority(t *testing.T) {
	tests := []struct {
		name    string
		input   []string
		want    []BackendSpec
		wantErr bool
	}{
		{
			name:  "simple priority",
			input: []string{"onnx", "xla", "go"},
			want: []BackendSpec{
				{Backend: BackendONNX, Device: DeviceAuto},
				{Backend: BackendXLA, Device: DeviceAuto},
				{Backend: BackendGo, Device: DeviceAuto},
			},
		},
		{
			name:  "mixed priority with devices",
			input: []string{"onnx:cuda", "xla:tpu", "onnx:cpu", "go"},
			want: []BackendSpec{
				{Backend: BackendONNX, Device: DeviceCUDA},
				{Backend: BackendXLA, Device: DeviceTPU},
				{Backend: BackendONNX, Device: DeviceCPU},
				{Backend: BackendGo, Device: DeviceAuto},
			},
		},
		{
			name:  "empty priority",
			input: []string{},
			want:  []BackendSpec{},
		},
		{
			name:    "invalid entry",
			input:   []string{"onnx", "invalid", "go"},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseBackendPriority(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseBackendPriority(%v) error = %v, wantErr %v", tt.input, err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if len(got) != len(tt.want) {
					t.Errorf("ParseBackendPriority(%v) = %v, want %v", tt.input, got, tt.want)
					return
				}
				for i := range got {
					if got[i] != tt.want[i] {
						t.Errorf("ParseBackendPriority(%v)[%d] = %v, want %v", tt.input, i, got[i], tt.want[i])
					}
				}
			}
		})
	}
}

func TestBackendSpecString(t *testing.T) {
	tests := []struct {
		spec BackendSpec
		want string
	}{
		{BackendSpec{Backend: BackendONNX, Device: DeviceAuto}, "onnx"},
		{BackendSpec{Backend: BackendONNX, Device: DeviceCUDA}, "onnx:cuda"},
		{BackendSpec{Backend: BackendXLA, Device: DeviceTPU}, "xla:tpu"},
		{BackendSpec{Backend: BackendGo, Device: DeviceCPU}, "go:cpu"},
		{BackendSpec{Backend: BackendONNX, Device: ""}, "onnx"}, // empty device treated as auto
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := tt.spec.String(); got != tt.want {
				t.Errorf("BackendSpec.String() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDeviceTypeToGPUMode(t *testing.T) {
	tests := []struct {
		device DeviceType
		want   GPUMode
	}{
		{DeviceAuto, GPUModeAuto},
		{DeviceCUDA, GPUModeCuda},
		{DeviceCoreML, GPUModeCoreML},
		{DeviceTPU, GPUModeTpu},
		{DeviceCPU, GPUModeOff},
	}

	for _, tt := range tests {
		t.Run(string(tt.device), func(t *testing.T) {
			if got := tt.device.ToGPUMode(); got != tt.want {
				t.Errorf("DeviceType(%q).ToGPUMode() = %v, want %v", tt.device, got, tt.want)
			}
		})
	}
}
