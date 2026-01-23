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

package backends

// Session represents a low-level inference session that can run tensor computations.
// This is the primitive interface that backends provide - it handles tensor I/O
// without knowledge of model semantics (encoder-decoder, vision, etc.).
//
// Higher-level model types (Vision2SeqModel, EncoderDecoderModel) are built
// on top of Session in the pipelines package.
type Session interface {
	// Run executes the session with the given named inputs.
	// Returns named outputs as tensors.
	Run(inputs []NamedTensor) ([]NamedTensor, error)

	// InputInfo returns metadata about expected inputs.
	InputInfo() []TensorInfo

	// OutputInfo returns metadata about outputs.
	OutputInfo() []TensorInfo

	// Close releases resources associated with the session.
	Close() error
}

// NamedTensor associates a name with tensor data.
type NamedTensor struct {
	Name  string
	Shape []int64
	Data  interface{} // []float32, []int64, []int32, etc.
}

// TensorInfo describes a tensor's metadata.
type TensorInfo struct {
	Name     string
	Shape    []int64  // -1 for dynamic dimensions
	DataType DataType // float32, int64, etc.
}

// DataType represents tensor element types.
type DataType string

const (
	DataTypeFloat32 DataType = "float32"
	DataTypeFloat16 DataType = "float16"
	DataTypeInt64   DataType = "int64"
	DataTypeInt32   DataType = "int32"
	DataTypeBool    DataType = "bool"
)

// SessionFactory creates sessions from model files.
// Each backend implements this to provide its session creation mechanism.
type SessionFactory interface {
	// CreateSession creates a session from a model file (e.g., ONNX file).
	CreateSession(modelPath string, opts ...SessionOption) (Session, error)

	// Backend returns the backend type this factory uses.
	Backend() BackendType
}

// SessionFactoryProvider is an optional interface that backends can implement
// to provide access to a SessionFactory for creating raw sessions.
// This allows higher-level code to build custom model types from sessions.
type SessionFactoryProvider interface {
	// SessionFactory returns a factory for creating raw sessions.
	SessionFactory() SessionFactory
}

// SessionOption configures session creation.
type SessionOption func(*SessionConfig)

// SessionConfig holds configuration for session creation.
type SessionConfig struct {
	// NumThreads for inference (0 = auto)
	NumThreads int

	// GPUMode controls GPU acceleration
	GPUMode GPUMode

	// GraphOptimizationLevel for ONNX (0-3)
	GraphOptimizationLevel int
}

// DefaultSessionConfig returns sensible defaults.
func DefaultSessionConfig() *SessionConfig {
	return &SessionConfig{
		NumThreads:             0,
		GPUMode:                GPUModeAuto,
		GraphOptimizationLevel: 3,
	}
}

// WithSessionThreads sets the number of threads.
func WithSessionThreads(n int) SessionOption {
	return func(c *SessionConfig) {
		c.NumThreads = n
	}
}

// WithSessionGPUMode sets the GPU mode.
func WithSessionGPUMode(mode GPUMode) SessionOption {
	return func(c *SessionConfig) {
		c.GPUMode = mode
	}
}

// ApplySessionOptions applies options to a config.
func ApplySessionOptions(opts ...SessionOption) *SessionConfig {
	cfg := DefaultSessionConfig()
	for _, opt := range opts {
		opt(cfg)
	}
	return cfg
}
