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

package backends

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
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
type onnxDarwinBackend struct {
	gpuMode   GPUMode
	gpuModeMu sync.RWMutex

	initialized     bool
	initializedOnce sync.Once
	initErr         error
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

func (b *onnxDarwinBackend) Loader() ModelLoader {
	return &onnxDarwinModelLoader{backend: b}
}

// initONNX initializes the ONNX Runtime library.
func (b *onnxDarwinBackend) initONNX() error {
	b.initializedOnce.Do(func() {
		// Set library path if found
		if libPath := getOnnxLibraryPathDarwin(); libPath != "" {
			ort.SetSharedLibraryPath(filepath.Join(libPath, "libonnxruntime.dylib"))
		}

		// Initialize the environment
		b.initErr = ort.InitializeEnvironment()
		if b.initErr == nil {
			b.initialized = true
		}
	})
	return b.initErr
}

// getOnnxLibraryPathDarwin returns the directory containing libonnxruntime.dylib from environment.
// Checks ONNXRUNTIME_ROOT first, then DYLD_LIBRARY_PATH.
func getOnnxLibraryPathDarwin() string {
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

// onnxDarwinModelLoader implements ModelLoader for ONNX Runtime on macOS.
type onnxDarwinModelLoader struct {
	backend *onnxDarwinBackend
}

func (l *onnxDarwinModelLoader) Load(path string, opts ...LoadOption) (Model, error) {
	// Initialize ONNX Runtime if needed
	if err := l.backend.initONNX(); err != nil {
		return nil, fmt.Errorf("initializing ONNX Runtime: %w", err)
	}

	config := ApplyOptions(opts...)

	// Determine ONNX file path
	onnxPath := filepath.Join(path, config.ONNXFilename)
	if _, err := os.Stat(onnxPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("ONNX model not found: %s", onnxPath)
	}

	// Create session options
	sessionOpts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("creating session options: %w", err)
	}

	// Configure number of threads
	if config.NumThreads > 0 {
		if err := sessionOpts.SetIntraOpNumThreads(config.NumThreads); err != nil {
			sessionOpts.Destroy()
			return nil, fmt.Errorf("setting thread count: %w", err)
		}
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
	//    outperforms CoreML EP for embedding models due to bridge layer overhead.
	//
	// 3. Batching: CoreML EP cannot handle dynamic batch sizes > 1.
	//
	// NOTE: CoreML provider options are no longer exposed in onnxruntime_go v1.25.0+
	// The TERMITE_FORCE_COREML option is preserved for documentation only.
	_ = os.Getenv("TERMITE_FORCE_COREML")
	_ = config.GPUMode

	// Create the session
	session, err := ort.NewDynamicAdvancedSession(onnxPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"last_hidden_state"},
		sessionOpts)
	if err != nil {
		sessionOpts.Destroy()
		return nil, fmt.Errorf("creating ONNX session: %w", err)
	}

	return &onnxDarwinModel{
		path:        path,
		config:      config,
		session:     session,
		sessionOpts: sessionOpts,
	}, nil
}

func (l *onnxDarwinModelLoader) SupportsModel(path string) bool {
	// Check if the model directory contains an ONNX file
	matches, _ := filepath.Glob(filepath.Join(path, "*.onnx"))
	return len(matches) > 0
}

func (l *onnxDarwinModelLoader) Backend() BackendType {
	return BackendONNX
}

// onnxDarwinModel implements Model using ONNX Runtime on macOS.
type onnxDarwinModel struct {
	path        string
	config      *LoadConfig
	session     *ort.DynamicAdvancedSession
	sessionOpts *ort.SessionOptions
}

func (m *onnxDarwinModel) Forward(ctx context.Context, inputs *ModelInputs) (*ModelOutput, error) {
	if m.session == nil {
		return nil, fmt.Errorf("ONNX session not initialized")
	}

	// Convert inputs to ONNX tensors
	batchSize := len(inputs.InputIDs)
	if batchSize == 0 {
		return &ModelOutput{}, nil
	}

	seqLen := len(inputs.InputIDs[0])

	// Flatten input tensors
	flatInputIDs := make([]int64, batchSize*seqLen)
	flatAttentionMask := make([]int64, batchSize*seqLen)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			flatInputIDs[i*seqLen+j] = int64(inputs.InputIDs[i][j])
			flatAttentionMask[i*seqLen+j] = int64(inputs.AttentionMask[i][j])
		}
	}

	// Create input tensors
	inputIDsTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), int64(seqLen)), flatInputIDs)
	if err != nil {
		return nil, fmt.Errorf("creating input_ids tensor: %w", err)
	}
	defer inputIDsTensor.Destroy()

	attentionMaskTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), int64(seqLen)), flatAttentionMask)
	if err != nil {
		return nil, fmt.Errorf("creating attention_mask tensor: %w", err)
	}
	defer attentionMaskTensor.Destroy()

	// Run inference - pass nil output to let session allocate it
	outputTensors := []ort.Value{nil}
	err = m.session.Run([]ort.Value{inputIDsTensor, attentionMaskTensor}, outputTensors)
	if err != nil {
		return nil, fmt.Errorf("running ONNX inference: %w", err)
	}
	defer func() {
		for _, t := range outputTensors {
			if t != nil {
				t.Destroy()
			}
		}
	}()

	// Extract output - expect last_hidden_state with shape [batch, seq, hidden]
	if len(outputTensors) == 0 || outputTensors[0] == nil {
		return nil, fmt.Errorf("no output tensors returned")
	}

	// Get the output tensor and extract data
	outputTensor := outputTensors[0]
	outputShape := outputTensor.GetShape()
	if len(outputShape) < 3 {
		return nil, fmt.Errorf("unexpected output shape: %v", outputShape)
	}

	hiddenSize := int(outputShape[2])

	// Type assert to get the data
	floatTensor, ok := outputTensor.(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("output tensor is not float32")
	}
	outputData := floatTensor.GetData()

	// Reshape into [batch][seq][hidden]
	lastHiddenState := make([][][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		lastHiddenState[i] = make([][]float32, seqLen)
		for j := 0; j < seqLen; j++ {
			lastHiddenState[i][j] = make([]float32, hiddenSize)
			baseIdx := (i*seqLen + j) * hiddenSize
			copy(lastHiddenState[i][j], outputData[baseIdx:baseIdx+hiddenSize])
		}
	}

	// Apply pooling to get embeddings
	embeddings := m.poolHiddenStates(lastHiddenState, inputs.AttentionMask)

	return &ModelOutput{
		LastHiddenState: lastHiddenState,
		Embeddings:      embeddings,
	}, nil
}

// poolHiddenStates applies pooling to get [batch, hidden] embeddings.
func (m *onnxDarwinModel) poolHiddenStates(hiddenStates [][][]float32, attentionMask [][]int32) [][]float32 {
	batchSize := len(hiddenStates)
	if batchSize == 0 {
		return nil
	}
	hiddenSize := len(hiddenStates[0][0])

	embeddings := make([][]float32, batchSize)

	switch m.config.Pooling {
	case "cls":
		// Use [CLS] token (first token)
		for i := 0; i < batchSize; i++ {
			embeddings[i] = make([]float32, hiddenSize)
			copy(embeddings[i], hiddenStates[i][0])
		}
	case "max":
		// Max pooling over sequence
		for i := 0; i < batchSize; i++ {
			embeddings[i] = make([]float32, hiddenSize)
			for h := 0; h < hiddenSize; h++ {
				maxVal := float32(-1e9)
				for j := 0; j < len(hiddenStates[i]); j++ {
					if attentionMask[i][j] > 0 && hiddenStates[i][j][h] > maxVal {
						maxVal = hiddenStates[i][j][h]
					}
				}
				embeddings[i][h] = maxVal
			}
		}
	case "mean", "":
		// Mean pooling (default)
		for i := 0; i < batchSize; i++ {
			embeddings[i] = make([]float32, hiddenSize)
			count := float32(0)
			for j := 0; j < len(hiddenStates[i]); j++ {
				if attentionMask[i][j] > 0 {
					for h := 0; h < hiddenSize; h++ {
						embeddings[i][h] += hiddenStates[i][j][h]
					}
					count++
				}
			}
			if count > 0 {
				for h := 0; h < hiddenSize; h++ {
					embeddings[i][h] /= count
				}
			}
		}
	default:
		// No pooling - return first token
		for i := 0; i < batchSize; i++ {
			embeddings[i] = make([]float32, hiddenSize)
			copy(embeddings[i], hiddenStates[i][0])
		}
	}

	return embeddings
}

func (m *onnxDarwinModel) Close() error {
	if m.session != nil {
		m.session.Destroy()
		m.session = nil
	}
	if m.sessionOpts != nil {
		m.sessionOpts.Destroy()
		m.sessionOpts = nil
	}
	return nil
}

func (m *onnxDarwinModel) Name() string {
	return m.path
}

func (m *onnxDarwinModel) Backend() BackendType {
	return BackendONNX
}
