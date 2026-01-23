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

package backends

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
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
type onnxBackend struct {
	// GPU mode configuration
	gpuMode   GPUMode
	gpuModeMu sync.RWMutex

	// Track whether CUDA is enabled (cached after first detection)
	cudaEnabled     bool
	cudaEnabledOnce sync.Once

	// Track initialization state
	initialized     bool
	initializedOnce sync.Once
	initErr         error
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

func (b *onnxBackend) Loader() ModelLoader {
	return &onnxModelLoader{backend: b}
}

// SessionFactory returns a SessionFactory for creating raw ONNX sessions.
// This provides low-level access for building custom model types.
func (b *onnxBackend) SessionFactory() SessionFactory {
	return &onnxSessionFactory{backend: b}
}

// initONNX initializes the ONNX Runtime library.
func (b *onnxBackend) initONNX() error {
	b.initializedOnce.Do(func() {
		// Set library path if found
		if libPath := getOnnxLibraryPath(); libPath != "" {
			ort.SetSharedLibraryPath(filepath.Join(libPath, getOnnxLibraryName()))
		}

		// Initialize the environment
		b.initErr = ort.InitializeEnvironment()
		if b.initErr == nil {
			b.initialized = true
		}
	})
	return b.initErr
}

// getOnnxLibraryPath returns the directory containing libonnxruntime.so from environment.
// Checks ONNXRUNTIME_ROOT first, then LD_LIBRARY_PATH.
func getOnnxLibraryPath() string {
	platform := runtime.GOOS + "-" + runtime.GOARCH

	// Check ONNXRUNTIME_ROOT (set by Makefile)
	if root := os.Getenv("ONNXRUNTIME_ROOT"); root != "" {
		// Try platform-specific path first
		platformDir := filepath.Join(root, platform, "lib")
		if _, err := os.Stat(filepath.Join(platformDir, getOnnxLibraryName())); err == nil {
			return platformDir
		}
		// Try direct lib path
		directDir := filepath.Join(root, "lib")
		if _, err := os.Stat(filepath.Join(directDir, getOnnxLibraryName())); err == nil {
			return directDir
		}
	}

	// Check LD_LIBRARY_PATH
	if ldPath := os.Getenv("LD_LIBRARY_PATH"); ldPath != "" {
		// LD_LIBRARY_PATH can have multiple paths separated by ':'
		for _, dir := range filepath.SplitList(ldPath) {
			if _, err := os.Stat(filepath.Join(dir, getOnnxLibraryName())); err == nil {
				return dir
			}
		}
	}

	return ""
}

// getOnnxLibraryName returns the platform-specific library name.
func getOnnxLibraryName() string {
	switch runtime.GOOS {
	case "windows":
		return "onnxruntime.dll"
	case "darwin":
		return "libonnxruntime.dylib"
	default:
		return "libonnxruntime.so"
	}
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

// onnxModelLoader implements ModelLoader for ONNX Runtime.
type onnxModelLoader struct {
	backend *onnxBackend
}

func (l *onnxModelLoader) Load(path string, opts ...LoadOption) (Model, error) {
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

	// Enable CUDA if requested and available
	gpuMode := config.GPUMode
	if gpuMode == "" {
		gpuMode = l.backend.GetGPUMode()
	}
	useCUDA := gpuMode == GPUModeCuda || (gpuMode == GPUModeAuto && l.backend.useCUDA())
	if useCUDA {
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err == nil {
			if err := sessionOpts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
				// CUDA not available, fall back to CPU
				cudaOpts.Destroy()
			} else {
				defer cudaOpts.Destroy()
			}
		}
	}

	// Create the session
	session, err := ort.NewDynamicAdvancedSession(onnxPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"last_hidden_state"},
		sessionOpts)
	if err != nil {
		sessionOpts.Destroy()
		return nil, fmt.Errorf("creating ONNX session: %w", err)
	}

	return &onnxModel{
		path:        path,
		config:      config,
		session:     session,
		sessionOpts: sessionOpts,
	}, nil
}

func (l *onnxModelLoader) SupportsModel(path string) bool {
	// Check if the model directory contains an ONNX file
	matches, _ := filepath.Glob(filepath.Join(path, "*.onnx"))
	return len(matches) > 0
}

func (l *onnxModelLoader) Backend() BackendType {
	return BackendONNX
}

// onnxModel implements Model using ONNX Runtime.
type onnxModel struct {
	path        string
	config      *LoadConfig
	session     *ort.DynamicAdvancedSession
	sessionOpts *ort.SessionOptions
}

func (m *onnxModel) Forward(ctx context.Context, inputs *ModelInputs) (*ModelOutput, error) {
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

	// Run inference
	outputTensors, err := m.session.Run([]ort.ArbitraryTensor{inputIDsTensor, attentionMaskTensor})
	if err != nil {
		return nil, fmt.Errorf("running ONNX inference: %w", err)
	}
	defer func() {
		for _, t := range outputTensors {
			t.Destroy()
		}
	}()

	// Extract output - expect last_hidden_state with shape [batch, seq, hidden]
	if len(outputTensors) == 0 {
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
func (m *onnxModel) poolHiddenStates(hiddenStates [][][]float32, attentionMask [][]int32) [][]float32 {
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

func (m *onnxModel) Close() error {
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

func (m *onnxModel) Name() string {
	return m.path
}

func (m *onnxModel) Backend() BackendType {
	return BackendONNX
}

// onnxSessionFactory implements SessionFactory for ONNX Runtime.
type onnxSessionFactory struct {
	backend *onnxBackend
}

func (f *onnxSessionFactory) CreateSession(modelPath string, opts ...SessionOption) (Session, error) {
	// Initialize ONNX Runtime if needed
	if err := f.backend.initONNX(); err != nil {
		return nil, fmt.Errorf("initializing ONNX Runtime: %w", err)
	}

	cfg := ApplySessionOptions(opts...)

	// Get input/output info from the model
	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, fmt.Errorf("getting model info: %w", err)
	}

	inputNames := make([]string, len(inputs))
	inputInfo := make([]TensorInfo, len(inputs))
	for i, info := range inputs {
		inputNames[i] = info.Name
		inputInfo[i] = TensorInfo{
			Name:     info.Name,
			Shape:    info.Dimensions,
			DataType: onnxDataType(info.DataType),
		}
	}

	outputNames := make([]string, len(outputs))
	outputInfo := make([]TensorInfo, len(outputs))
	for i, info := range outputs {
		outputNames[i] = info.Name
		outputInfo[i] = TensorInfo{
			Name:     info.Name,
			Shape:    info.Dimensions,
			DataType: onnxDataType(info.DataType),
		}
	}

	// Create session options
	sessionOpts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("creating session options: %w", err)
	}

	// Configure number of threads
	if cfg.NumThreads > 0 {
		if err := sessionOpts.SetIntraOpNumThreads(cfg.NumThreads); err != nil {
			sessionOpts.Destroy()
			return nil, fmt.Errorf("setting thread count: %w", err)
		}
	}

	// Enable CUDA if requested and available
	gpuMode := cfg.GPUMode
	if gpuMode == "" {
		gpuMode = f.backend.GetGPUMode()
	}
	useCUDA := gpuMode == GPUModeCuda || (gpuMode == GPUModeAuto && f.backend.useCUDA())
	if useCUDA {
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err == nil {
			if err := sessionOpts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
				cudaOpts.Destroy()
			} else {
				defer cudaOpts.Destroy()
			}
		}
	}

	// Create the session
	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, sessionOpts)
	if err != nil {
		sessionOpts.Destroy()
		return nil, fmt.Errorf("creating ONNX session: %w", err)
	}

	return &onnxSession{
		session:     session,
		sessionOpts: sessionOpts,
		inputInfo:   inputInfo,
		outputInfo:  outputInfo,
	}, nil
}

func (f *onnxSessionFactory) Backend() BackendType {
	return BackendONNX
}

// onnxDataType converts ONNX data type to our DataType.
func onnxDataType(dt ort.TensorElementDataType) DataType {
	switch dt {
	case ort.TensorElementDataTypeFloat:
		return DataTypeFloat32
	case ort.TensorElementDataTypeInt64:
		return DataTypeInt64
	case ort.TensorElementDataTypeInt32:
		return DataTypeInt32
	case ort.TensorElementDataTypeBool:
		return DataTypeBool
	default:
		return DataTypeFloat32
	}
}

// onnxSession implements Session for ONNX Runtime.
type onnxSession struct {
	session     *ort.DynamicAdvancedSession
	sessionOpts *ort.SessionOptions
	inputInfo   []TensorInfo
	outputInfo  []TensorInfo
}

func (s *onnxSession) Run(inputs []NamedTensor) ([]NamedTensor, error) {
	if s.session == nil {
		return nil, fmt.Errorf("session is closed")
	}

	// Convert inputs to ONNX tensors
	ortInputs := make([]ort.Value, len(inputs))
	for i, input := range inputs {
		tensor, err := createOrtTensor(input)
		if err != nil {
			// Clean up already created tensors
			for j := 0; j < i; j++ {
				if ortInputs[j] != nil {
					ortInputs[j].Destroy()
				}
			}
			return nil, fmt.Errorf("creating input tensor %s: %w", input.Name, err)
		}
		ortInputs[i] = tensor
	}
	defer func() {
		for _, t := range ortInputs {
			if t != nil {
				t.Destroy()
			}
		}
	}()

	// Run inference
	ortOutputs := make([]ort.Value, len(s.outputInfo))
	for i := range ortOutputs {
		ortOutputs[i] = nil
	}

	if err := s.session.Run(ortInputs, ortOutputs); err != nil {
		return nil, fmt.Errorf("running ONNX session: %w", err)
	}
	defer func() {
		for _, t := range ortOutputs {
			if t != nil {
				t.Destroy()
			}
		}
	}()

	// Convert outputs to NamedTensors
	outputs := make([]NamedTensor, len(ortOutputs))
	for i, ortOutput := range ortOutputs {
		if ortOutput == nil {
			continue
		}
		output, err := extractOrtTensor(ortOutput, s.outputInfo[i].Name)
		if err != nil {
			return nil, fmt.Errorf("extracting output tensor %s: %w", s.outputInfo[i].Name, err)
		}
		outputs[i] = output
	}

	return outputs, nil
}

func (s *onnxSession) InputInfo() []TensorInfo {
	return s.inputInfo
}

func (s *onnxSession) OutputInfo() []TensorInfo {
	return s.outputInfo
}

func (s *onnxSession) Close() error {
	if s.session != nil {
		s.session.Destroy()
		s.session = nil
	}
	if s.sessionOpts != nil {
		s.sessionOpts.Destroy()
		s.sessionOpts = nil
	}
	return nil
}

// createOrtTensor creates an ORT tensor from a NamedTensor.
func createOrtTensor(input NamedTensor) (ort.Value, error) {
	shape := ort.NewShape(input.Shape...)

	switch data := input.Data.(type) {
	case []float32:
		return ort.NewTensor(shape, data)
	case []int64:
		return ort.NewTensor(shape, data)
	case []int32:
		// Convert to int64 for ONNX
		int64Data := make([]int64, len(data))
		for i, v := range data {
			int64Data[i] = int64(v)
		}
		return ort.NewTensor(shape, int64Data)
	case []bool:
		return ort.NewTensor(shape, data)
	default:
		return nil, fmt.Errorf("unsupported data type: %T", data)
	}
}

// extractOrtTensor extracts a NamedTensor from an ORT tensor.
func extractOrtTensor(ortTensor ort.Value, name string) (NamedTensor, error) {
	shape := ortTensor.GetShape()

	// Try float32 first (most common)
	if floatTensor, ok := ortTensor.(*ort.Tensor[float32]); ok {
		data := floatTensor.GetData()
		dataCopy := make([]float32, len(data))
		copy(dataCopy, data)
		return NamedTensor{
			Name:  name,
			Shape: shape,
			Data:  dataCopy,
		}, nil
	}

	// Try int64
	if int64Tensor, ok := ortTensor.(*ort.Tensor[int64]); ok {
		data := int64Tensor.GetData()
		dataCopy := make([]int64, len(data))
		copy(dataCopy, data)
		return NamedTensor{
			Name:  name,
			Shape: shape,
			Data:  dataCopy,
		}, nil
	}

	// Try int32
	if int32Tensor, ok := ortTensor.(*ort.Tensor[int32]); ok {
		data := int32Tensor.GetData()
		dataCopy := make([]int32, len(data))
		copy(dataCopy, data)
		return NamedTensor{
			Name:  name,
			Shape: shape,
			Data:  dataCopy,
		}, nil
	}

	return NamedTensor{}, fmt.Errorf("unsupported tensor type")
}
