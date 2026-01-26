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

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	hfmodels "github.com/ajroetker/huggingface-gomlx"
	"github.com/ajroetker/huggingface-gomlx/architectures/bert"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	mlctx "github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/onnx-gomlx/onnx"

	// Import Go backend - always available (pure Go, no CGO)
	_ "github.com/gomlx/gomlx/backends/simplego"
)

func init() {
	// Register Go backend (always available)
	// Note: The simplego package registers itself as "go" in the backends registry
	RegisterBackend(newGomlxBackend(BackendGo, "go"))
}

// gomlxBackend implements Backend using GoMLX for inference.
//
// This backend supports two model formats:
//   - HuggingFace: SafeTensors + config.json (via huggingface-gomlx)
//   - ONNX: .onnx model files (via onnx-gomlx)
//
// Two backends are registered:
//   - BackendGo: Pure Go engine (simplego), always available, slower
//   - BackendXLA: XLA engine, hardware accelerated (CUDA, TPU, optimized CPU), requires XLA/PJRT build tags
type gomlxBackend struct {
	backendType BackendType
	engineType  string // "go" or "xla"
	engineMgr   *engineManager
	available   *bool // cached availability check
}

// newGomlxBackend creates a new GoMLX backend with the specified type and engine.
func newGomlxBackend(backendType BackendType, engineType string) *gomlxBackend {
	return &gomlxBackend{
		backendType: backendType,
		engineType:  engineType,
	}
}

func (b *gomlxBackend) Type() BackendType {
	return b.backendType
}

func (b *gomlxBackend) Name() string {
	switch b.backendType {
	case BackendXLA:
		return "GoMLX (XLA)"
	case BackendCoreML:
		return "GoMLX (CoreML)"
	case BackendGo:
		return "GoMLX (Go)"
	default:
		return "GoMLX"
	}
}

func (b *gomlxBackend) Available() bool {
	if b.available != nil {
		return *b.available
	}

	// Test if we can create an engine of this type.
	// Use recover to catch panics from libraries that don't handle
	// missing dependencies gracefully (e.g., go-xla panics if PJRT
	// plugin fails to load due to GLIBC version mismatch).
	result := func() (available bool) {
		defer func() {
			if r := recover(); r != nil {
				available = false
			}
		}()
		_, err := backends.NewWithConfig(b.engineType)
		return err == nil
	}()

	b.available = &result
	return result
}

func (b *gomlxBackend) Priority() int {
	switch b.backendType {
	case BackendXLA:
		// XLA has higher priority than Go (lower number = higher priority)
		return 20
	case BackendCoreML:
		// CoreML is between XLA and Go (macOS only)
		return 25
	case BackendGo:
		// Go is always available fallback
		return 100
	default:
		return 50
	}
}

func (b *gomlxBackend) Loader() ModelLoader {
	if b.engineMgr == nil {
		b.engineMgr = newEngineManager()
	}
	return &gomlxModelLoader{backend: b}
}

// SessionFactory returns a SessionFactory for creating raw GoMLX sessions.
// This provides low-level access for building custom model types (e.g., seq2seq).
func (b *gomlxBackend) SessionFactory() SessionFactory {
	if b.engineMgr == nil {
		b.engineMgr = newEngineManager()
	}
	return &gomlxSessionFactory{backend: b}
}

// engineManager manages GoMLX backend engines.
type engineManager struct {
	mu            sync.RWMutex
	defaultEngine backends.Backend
}

func newEngineManager() *engineManager {
	return &engineManager{}
}

// getEngine returns the GoMLX backend engine, creating it if needed.
// If backendType is empty, auto-detects the best available backend.
func (m *engineManager) getEngine(backendType string) (backends.Backend, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// If specific backend requested, create it
	if backendType != "" {
		return safeNewBackend(backendType)
	}

	// Use cached default engine
	if m.defaultEngine != nil {
		return m.defaultEngine, nil
	}

	// Auto-detect: try xla first, fall back to go (simplego)
	engine, err := safeNewBackend("xla")
	if err != nil {
		// XLA not available, use go (simplego)
		engine, err = safeNewBackend("go")
		if err != nil {
			return nil, err
		}
	}

	m.defaultEngine = engine
	return engine, nil
}

// safeNewBackend creates a new backend, catching panics from libraries
// that don't handle missing dependencies gracefully.
func safeNewBackend(backendType string) (engine backends.Backend, err error) {
	defer func() {
		if r := recover(); r != nil {
			engine = nil
			err = fmt.Errorf("backend %q panicked during initialization: %v", backendType, r)
		}
	}()
	return backends.NewWithConfig(backendType)
}

// gomlxModelLoader implements ModelLoader for GoMLX inference.
type gomlxModelLoader struct {
	backend *gomlxBackend
}

func (l *gomlxModelLoader) Load(path string, opts ...LoadOption) (Model, error) {
	config := ApplyOptions(opts...)

	// Detect model format
	format := config.ModelFormat
	if format == ModelFormatAuto {
		format = detectGoMLXModelFormat(path)
	}

	// Get the inference backend
	engine, err := l.backend.engineMgr.getEngine(string(config.GoMLXBackend))
	if err != nil {
		return nil, fmt.Errorf("getting GoMLX engine: %w", err)
	}

	switch format {
	case ModelFormatHuggingFace:
		return l.loadHuggingFace(path, config, engine)
	case ModelFormatONNX:
		return l.loadONNX(path, config, engine)
	default:
		return nil, fmt.Errorf("unknown model format at %s", path)
	}
}

// loadHuggingFace loads a HuggingFace model (SafeTensors format).
func (l *gomlxModelLoader) loadHuggingFace(path string, config *LoadConfig, engine backends.Backend) (Model, error) {
	hfModel, err := newHFModel(path, engine, config.Pooling, config.Normalize)
	if err != nil {
		return nil, err
	}

	return &gomlxModelWrapper{
		hfModel:     hfModel,
		path:        path,
		config:      config,
		backendType: l.backend.backendType,
	}, nil
}

// loadONNX loads an ONNX model via onnx-gomlx.
func (l *gomlxModelLoader) loadONNX(path string, config *LoadConfig, engine backends.Backend) (Model, error) {
	// Find the ONNX file
	onnxPath := filepath.Join(path, config.ONNXFilename)
	if _, err := os.Stat(onnxPath); os.IsNotExist(err) {
		// Try to find any .onnx file
		matches, _ := filepath.Glob(filepath.Join(path, "*.onnx"))
		if len(matches) == 0 {
			return nil, fmt.Errorf("no ONNX file found in %s", path)
		}
		onnxPath = matches[0]
	}

	onnxModel, err := newONNXModel(onnxPath, engine, config.Pooling, config.Normalize)
	if err != nil {
		return nil, err
	}

	return &gomlxModelWrapper{
		onnxModel:   onnxModel,
		path:        path,
		config:      config,
		backendType: l.backend.backendType,
	}, nil
}

func (l *gomlxModelLoader) SupportsModel(path string) bool {
	format := detectGoMLXModelFormat(path)
	return format != ModelFormatAuto
}

func (l *gomlxModelLoader) Backend() BackendType {
	return l.backend.backendType
}

// detectGoMLXModelFormat auto-detects the model format from files present.
func detectGoMLXModelFormat(path string) ModelFormat {
	// Check for HuggingFace format (config.json + safetensors)
	configPath := filepath.Join(path, "config.json")
	if _, err := os.Stat(configPath); err == nil {
		// Check for SafeTensors
		safetensors, _ := filepath.Glob(filepath.Join(path, "*.safetensors"))
		if len(safetensors) > 0 {
			return ModelFormatHuggingFace
		}
		// Check for sharded model index
		indexPath := filepath.Join(path, "model.safetensors.index.json")
		if _, err := os.Stat(indexPath); err == nil {
			return ModelFormatHuggingFace
		}
	}

	// Check for ONNX format
	onnxFiles, _ := filepath.Glob(filepath.Join(path, "*.onnx"))
	if len(onnxFiles) > 0 {
		return ModelFormatONNX
	}

	return ModelFormatAuto // Unknown
}

// gomlxModelWrapper wraps hfModel or onnxModel to implement the backends.Model interface.
type gomlxModelWrapper struct {
	hfModel     *hfModel
	onnxModel   *onnxModel
	path        string
	config      *LoadConfig
	backendType BackendType
}

func (m *gomlxModelWrapper) Forward(ctx context.Context, inputs *ModelInputs) (*ModelOutput, error) {
	if m.hfModel != nil {
		return m.hfModel.forward(ctx, inputs.InputIDs, inputs.AttentionMask)
	}
	if m.onnxModel != nil {
		return m.onnxModel.forward(ctx, inputs.InputIDs, inputs.AttentionMask)
	}
	return nil, fmt.Errorf("no model loaded")
}

func (m *gomlxModelWrapper) Close() error {
	if m.hfModel != nil {
		return m.hfModel.close()
	}
	if m.onnxModel != nil {
		return m.onnxModel.close()
	}
	return nil
}

func (m *gomlxModelWrapper) Name() string {
	return m.path
}

func (m *gomlxModelWrapper) Backend() BackendType {
	return m.backendType
}

// =============================================================================
// HuggingFace Model (SafeTensors format)
// =============================================================================

// hfModel implements inference for HuggingFace models (SafeTensors format)
// using huggingface-gomlx.
type hfModel struct {
	path      string
	pooling   string
	normalize bool
	hfModel   *hfmodels.Model
	ctx       *mlctx.Context
	engine    backends.Backend
	exec      *mlctx.Exec // Compiled inference graph

	mu sync.Mutex
}

// newHFModel loads a HuggingFace model from a local directory.
func newHFModel(path string, engine backends.Backend, pooling string, normalize bool) (*hfModel, error) {
	// Load the model using huggingface-gomlx
	hfm, err := hfmodels.NewFromLocal(path)
	if err != nil {
		return nil, fmt.Errorf("loading HuggingFace model: %w", err)
	}

	// Create GoMLX context and load weights
	ctx := mlctx.New()
	if err := hfm.LoadWeightsIntoContext(ctx); err != nil {
		return nil, fmt.Errorf("loading weights into context: %w", err)
	}

	model := &hfModel{
		path:      path,
		pooling:   pooling,
		normalize: normalize,
		hfModel:   hfm,
		ctx:       ctx,
		engine:    engine,
	}

	// Pre-compile the inference graph if possible
	if err := model.compile(); err != nil {
		return nil, fmt.Errorf("compiling inference graph: %w", err)
	}

	return model, nil
}

// compile pre-compiles the inference graph for efficiency.
func (m *hfModel) compile() error {
	// Get the architecture builder
	builder := m.hfModel.Builder

	// Check if it's a BERT-like model
	bertBuilder, ok := builder.(*bert.Builder)
	if !ok {
		// For non-BERT models, we'll compile on first forward pass
		return nil
	}

	// Pre-compile the BERT forward pass
	exec, err := mlctx.NewExecAny(m.engine, m.ctx, func(ctx *mlctx.Context, inputs []*graph.Node) []*graph.Node {
		inputIDs := inputs[0]
		attentionMask := inputs[1]

		// Run BERT forward pass
		reuseCtx := ctx.Reuse()
		hidden, _ := bertBuilder.Forward(reuseCtx, inputIDs, attentionMask, nil, nil)
		return []*graph.Node{hidden}
	})
	if err != nil {
		return fmt.Errorf("creating exec: %w", err)
	}
	m.exec = exec

	return nil
}

// forward runs inference on the given inputs.
func (m *hfModel) forward(ctx context.Context, inputIDs [][]int32, attentionMask [][]int32) (*ModelOutput, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	batchSize := len(inputIDs)
	if batchSize == 0 {
		return &ModelOutput{}, nil
	}

	seqLen := len(inputIDs[0])

	// Create input tensors
	flatInputIDs := make([]int32, batchSize*seqLen)
	flatAttentionMask := make([]float32, batchSize*seqLen)
	for i := range batchSize {
		for j := range seqLen {
			flatInputIDs[i*seqLen+j] = inputIDs[i][j]
			flatAttentionMask[i*seqLen+j] = float32(attentionMask[i][j])
		}
	}

	inputIDsTensor := tensors.FromFlatDataAndDimensions(flatInputIDs, batchSize, seqLen)
	attentionMaskTensor := tensors.FromFlatDataAndDimensions(flatAttentionMask, batchSize, seqLen)

	// Run inference
	if m.exec == nil {
		return nil, fmt.Errorf("model not compiled - architecture not supported")
	}
	results, err := m.exec.Exec(inputIDsTensor, attentionMaskTensor)
	if err != nil {
		return nil, fmt.Errorf("exec failed: %w", err)
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("no output from model")
	}

	// Extract hidden states
	output := results[0]
	_ = output.Shape() // Shape is [batch, seq, hidden]

	// Get the data
	data := output.Value().([][][]float32)

	// Reshape to our format
	lastHiddenState := make([][][]float32, batchSize)
	for i := range batchSize {
		lastHiddenState[i] = data[i]
	}

	// Apply pooling
	embeddings := PoolHiddenStates(lastHiddenState, attentionMask, PoolingStrategy(m.pooling))

	// Apply normalization if requested
	if m.normalize {
		NormalizeEmbeddings(embeddings)
	}

	return &ModelOutput{
		LastHiddenState: lastHiddenState,
		Embeddings:      embeddings,
	}, nil
}

func (m *hfModel) close() error {
	return nil
}

// =============================================================================
// ONNX Model (via onnx-gomlx)
// =============================================================================

// onnxModel implements inference for ONNX models using onnx-gomlx.
// This converts ONNX graphs to GoMLX for execution.
type onnxModel struct {
	path      string
	pooling   string
	normalize bool
	onnxModel *onnx.Model
	ctx       *mlctx.Context
	engine    backends.Backend

	mu sync.Mutex
}

// newONNXModel loads an ONNX model from a file path.
func newONNXModel(onnxPath string, engine backends.Backend, pooling string, normalize bool) (*onnxModel, error) {
	// Load the ONNX model
	om, err := onnx.ReadFile(onnxPath)
	if err != nil {
		return nil, fmt.Errorf("loading ONNX model: %w", err)
	}

	// Create GoMLX context and load variables
	ctx := mlctx.New()
	if err := om.VariablesToContext(ctx); err != nil {
		return nil, fmt.Errorf("loading ONNX variables: %w", err)
	}

	return &onnxModel{
		path:      onnxPath,
		pooling:   pooling,
		normalize: normalize,
		onnxModel: om,
		ctx:       ctx,
		engine:    engine,
	}, nil
}

// forward runs inference on the given inputs.
func (m *onnxModel) forward(ctx context.Context, inputIDs [][]int32, attentionMask [][]int32) (*ModelOutput, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	batchSize := len(inputIDs)
	if batchSize == 0 {
		return &ModelOutput{}, nil
	}

	seqLen := len(inputIDs[0])

	// Create input tensors (ONNX typically expects int64)
	flatInputIDs := make([]int64, batchSize*seqLen)
	flatAttentionMask := make([]int64, batchSize*seqLen)
	for i := range batchSize {
		for j := range seqLen {
			flatInputIDs[i*seqLen+j] = int64(inputIDs[i][j])
			flatAttentionMask[i*seqLen+j] = int64(attentionMask[i][j])
		}
	}

	inputIDsTensor := tensors.FromFlatDataAndDimensions(flatInputIDs, batchSize, seqLen)
	attentionMaskTensor := tensors.FromFlatDataAndDimensions(flatAttentionMask, batchSize, seqLen)

	// Check if model requires token_type_ids (used by BERT-based models)
	var tokenTypeIdsTensor *tensors.Tensor
	inputNames, _ := m.onnxModel.Inputs()
	needsTokenTypeIds := false
	for _, name := range inputNames {
		if name == "token_type_ids" {
			needsTokenTypeIds = true
			break
		}
	}
	if needsTokenTypeIds {
		// Create zeros tensor for token_type_ids (same shape as input_ids)
		flatTokenTypeIds := make([]int64, batchSize*seqLen) // zeros by default
		tokenTypeIdsTensor = tensors.FromFlatDataAndDimensions(flatTokenTypeIds, batchSize, seqLen)
	}

	// Build the ONNX graph function
	graphFn := func(mlCtx *mlctx.Context, inputs []*graph.Node) []*graph.Node {
		inputMap := map[string]*graph.Node{
			"input_ids":      inputs[0],
			"attention_mask": inputs[1],
		}
		if len(inputs) > 2 {
			inputMap["token_type_ids"] = inputs[2]
		}
		return m.onnxModel.CallGraph(mlCtx.Reuse(), inputs[0].Graph(), inputMap)
	}

	// Execute with the appropriate inputs
	var results []*tensors.Tensor
	var err error
	if tokenTypeIdsTensor != nil {
		results, err = mlctx.ExecOnceN(m.engine, m.ctx, graphFn, inputIDsTensor, attentionMaskTensor, tokenTypeIdsTensor)
	} else {
		results, err = mlctx.ExecOnceN(m.engine, m.ctx, graphFn, inputIDsTensor, attentionMaskTensor)
	}
	if err != nil {
		return nil, fmt.Errorf("exec failed: %w", err)
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("no output from ONNX model")
	}

	// First output: either last_hidden_state [batch, seq, hidden] or logits [batch, classes]
	output := results[0]
	shape := output.Shape()

	// Handle different output shapes:
	// - 3D [batch, seq, hidden]: hidden states (encoder models, embeddings)
	// - 2D [batch, classes]: logits (classification models, rerankers)
	switch len(shape.Dimensions) {
	case 3:
		// Standard encoder output: [batch, seq, hidden]
		data := output.Value().([][][]float32)

		// Reshape to our format
		lastHiddenState := make([][][]float32, batchSize)
		for i := range batchSize {
			lastHiddenState[i] = data[i]
		}

		// Apply pooling
		embeddings := PoolHiddenStates(lastHiddenState, attentionMask, PoolingStrategy(m.pooling))

		// Apply normalization if requested
		if m.normalize {
			NormalizeEmbeddings(embeddings)
		}

		return &ModelOutput{
			LastHiddenState: lastHiddenState,
			Embeddings:      embeddings,
		}, nil

	case 2:
		// Classification/reranker output: [batch, classes] - logits
		data := output.Value().([][]float32)
		numClasses := int(shape.Dimensions[1])

		logits := make([][]float32, batchSize)
		for i := range batchSize {
			logits[i] = make([]float32, numClasses)
			copy(logits[i], data[i])
		}

		return &ModelOutput{
			Logits: logits,
		}, nil

	default:
		return nil, fmt.Errorf("unexpected output shape: %v (expected 2D or 3D)", shape.Dimensions)
	}
}

func (m *onnxModel) close() error {
	return nil
}

// =============================================================================
// SessionFactory Implementation (for seq2seq and other multi-model pipelines)
// =============================================================================

// gomlxSessionFactory creates sessions from ONNX model files using GoMLX.
type gomlxSessionFactory struct {
	backend *gomlxBackend
}

func (f *gomlxSessionFactory) CreateSession(modelPath string, opts ...SessionOption) (Session, error) {
	// Get the inference backend
	engine, err := f.backend.engineMgr.getEngine(f.backend.engineType)
	if err != nil {
		return nil, fmt.Errorf("getting GoMLX engine: %w", err)
	}

	// Load the ONNX model
	om, err := onnx.ReadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("loading ONNX model: %w", err)
	}

	// Create context and load variables
	ctx := mlctx.New()
	if err := om.VariablesToContext(ctx); err != nil {
		return nil, fmt.Errorf("loading ONNX variables: %w", err)
	}

	// Get input/output info
	inputNames, inputShapes := om.Inputs()
	outputNames, outputShapes := om.Outputs()

	inputInfo := make([]TensorInfo, len(inputNames))
	for i, name := range inputNames {
		inputInfo[i] = TensorInfo{
			Name:     name,
			Shape:    intsToInt64s(inputShapes[i].Dimensions),
			DataType: gomlxDataType(inputShapes[i].DType),
		}
	}

	outputInfo := make([]TensorInfo, len(outputNames))
	for i, name := range outputNames {
		outputInfo[i] = TensorInfo{
			Name:     name,
			Shape:    intsToInt64s(outputShapes[i].Dimensions),
			DataType: gomlxDataType(outputShapes[i].DType),
		}
	}

	return &gomlxSession{
		onnxModel:   om,
		ctx:         ctx,
		engine:      engine,
		inputInfo:   inputInfo,
		outputInfo:  outputInfo,
		inputNames:  inputNames,
		outputNames: outputNames,
	}, nil
}

func (f *gomlxSessionFactory) Backend() BackendType {
	return f.backend.backendType
}

// gomlxSession implements Session for raw tensor I/O using GoMLX.
type gomlxSession struct {
	onnxModel   *onnx.Model
	ctx         *mlctx.Context
	engine      backends.Backend
	inputInfo   []TensorInfo
	outputInfo  []TensorInfo
	inputNames  []string
	outputNames []string
	mu          sync.Mutex
}

func (s *gomlxSession) Run(inputs []NamedTensor) ([]NamedTensor, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.onnxModel == nil {
		return nil, fmt.Errorf("session is closed")
	}

	// Build a map of input name -> tensor for fast lookup
	inputMap := make(map[string]NamedTensor, len(inputs))
	for _, input := range inputs {
		inputMap[input.Name] = input
	}

	// Convert inputs to GoMLX tensors in the order expected by the model
	gomlxInputs := make([]*tensors.Tensor, len(s.inputNames))
	for i, name := range s.inputNames {
		input, ok := inputMap[name]
		if !ok {
			return nil, fmt.Errorf("missing input tensor: %s", name)
		}
		tensor, err := namedTensorToGoMLX(input)
		if err != nil {
			return nil, fmt.Errorf("converting input tensor %s: %w", name, err)
		}
		gomlxInputs[i] = tensor
	}

	// Build the ONNX graph function
	graphFn := func(mlCtx *mlctx.Context, graphInputs []*graph.Node) []*graph.Node {
		inputNodeMap := make(map[string]*graph.Node, len(s.inputNames))
		for i, name := range s.inputNames {
			inputNodeMap[name] = graphInputs[i]
		}
		return s.onnxModel.CallGraph(mlCtx.Reuse(), graphInputs[0].Graph(), inputNodeMap)
	}

	// Execute - convert []*tensors.Tensor to []any for variadic call
	args := make([]any, len(gomlxInputs))
	for i, t := range gomlxInputs {
		args[i] = t
	}
	results, err := mlctx.ExecOnceN(s.engine, s.ctx, graphFn, args...)
	if err != nil {
		return nil, fmt.Errorf("executing ONNX graph: %w", err)
	}

	// Convert outputs to NamedTensors
	outputs := make([]NamedTensor, len(results))
	for i, result := range results {
		name := ""
		if i < len(s.outputNames) {
			name = s.outputNames[i]
		}
		output, err := gomlxToNamedTensor(result, name)
		if err != nil {
			return nil, fmt.Errorf("converting output tensor %d: %w", i, err)
		}
		outputs[i] = output
	}

	return outputs, nil
}

func (s *gomlxSession) InputInfo() []TensorInfo {
	return s.inputInfo
}

func (s *gomlxSession) OutputInfo() []TensorInfo {
	return s.outputInfo
}

func (s *gomlxSession) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.onnxModel = nil
	s.ctx = nil
	return nil
}

// =============================================================================
// Helper functions for tensor conversion
// =============================================================================

// intsToInt64s converts []int to []int64.
func intsToInt64s(dims []int) []int64 {
	result := make([]int64, len(dims))
	for i, d := range dims {
		result[i] = int64(d)
	}
	return result
}

// gomlxDataType converts GoMLX DType to our DataType.
func gomlxDataType(dt dtypes.DType) DataType {
	switch dt {
	case dtypes.Float32:
		return DataTypeFloat32
	case dtypes.Float64:
		return DataTypeFloat32 // Downgrade to float32 for compatibility
	case dtypes.Float16, dtypes.BFloat16:
		return DataTypeFloat16
	case dtypes.Int64:
		return DataTypeInt64
	case dtypes.Int32:
		return DataTypeInt32
	case dtypes.Int8, dtypes.Int16:
		return DataTypeInt32 // Upgrade to int32 for compatibility
	case dtypes.Bool:
		return DataTypeBool
	default:
		return DataTypeFloat32 // Default to float32
	}
}

// namedTensorToGoMLX converts a NamedTensor to a GoMLX tensor.
func namedTensorToGoMLX(nt NamedTensor) (*tensors.Tensor, error) {
	dims := make([]int, len(nt.Shape))
	for i, d := range nt.Shape {
		dims[i] = int(d)
	}

	switch data := nt.Data.(type) {
	case []float32:
		return tensors.FromFlatDataAndDimensions(data, dims...), nil
	case []float64:
		// Convert float64 to float32
		f32 := make([]float32, len(data))
		for i, v := range data {
			f32[i] = float32(v)
		}
		return tensors.FromFlatDataAndDimensions(f32, dims...), nil
	case []int64:
		return tensors.FromFlatDataAndDimensions(data, dims...), nil
	case []int32:
		// Convert int32 to int64
		i64 := make([]int64, len(data))
		for i, v := range data {
			i64[i] = int64(v)
		}
		return tensors.FromFlatDataAndDimensions(i64, dims...), nil
	case []int:
		// Convert int to int64
		i64 := make([]int64, len(data))
		for i, v := range data {
			i64[i] = int64(v)
		}
		return tensors.FromFlatDataAndDimensions(i64, dims...), nil
	case []bool:
		return tensors.FromFlatDataAndDimensions(data, dims...), nil
	default:
		return nil, fmt.Errorf("unsupported tensor data type: %T", data)
	}
}

// gomlxToNamedTensor converts a GoMLX tensor to a NamedTensor.
func gomlxToNamedTensor(t *tensors.Tensor, name string) (NamedTensor, error) {
	shape := t.Shape()
	dims := make([]int64, shape.Rank())
	for i := range shape.Rank() {
		dims[i] = int64(shape.Dimensions[i])
	}

	// Extract data based on dtype
	var data interface{}
	switch shape.DType {
	case dtypes.Float32:
		data = extractFloat32Data(t)
	case dtypes.Float64:
		data = extractFloat64Data(t)
	case dtypes.Int64:
		data = extractInt64Data(t)
	case dtypes.Int32:
		data = extractInt32Data(t)
	case dtypes.Bool:
		data = extractBoolData(t)
	default:
		// Try float32 as default
		data = extractFloat32Data(t)
	}

	return NamedTensor{
		Name:  name,
		Shape: dims,
		Data:  data,
	}, nil
}

// extractFloat32Data extracts float32 data from a tensor as a flat slice.
func extractFloat32Data(t *tensors.Tensor) []float32 {
	val := t.Value()
	return flattenFloat32(val)
}

// extractFloat64Data extracts float64 data from a tensor as a flat slice.
func extractFloat64Data(t *tensors.Tensor) []float64 {
	val := t.Value()
	return flattenFloat64(val)
}

// extractInt64Data extracts int64 data from a tensor as a flat slice.
func extractInt64Data(t *tensors.Tensor) []int64 {
	val := t.Value()
	return flattenInt64(val)
}

// extractInt32Data extracts int32 data from a tensor as a flat slice.
func extractInt32Data(t *tensors.Tensor) []int32 {
	val := t.Value()
	return flattenInt32(val)
}

// extractBoolData extracts bool data from a tensor as a flat slice.
func extractBoolData(t *tensors.Tensor) []bool {
	val := t.Value()
	return flattenBool(val)
}

// flattenFloat32 recursively flattens multi-dimensional float32 data.
func flattenFloat32(val interface{}) []float32 {
	switch v := val.(type) {
	case []float32:
		return v
	case [][]float32:
		var result []float32
		for _, row := range v {
			result = append(result, row...)
		}
		return result
	case [][][]float32:
		var result []float32
		for _, matrix := range v {
			for _, row := range matrix {
				result = append(result, row...)
			}
		}
		return result
	case [][][][]float32:
		var result []float32
		for _, cube := range v {
			for _, matrix := range cube {
				for _, row := range matrix {
					result = append(result, row...)
				}
			}
		}
		return result
	default:
		return nil
	}
}

// flattenFloat64 recursively flattens multi-dimensional float64 data.
func flattenFloat64(val interface{}) []float64 {
	switch v := val.(type) {
	case []float64:
		return v
	case [][]float64:
		var result []float64
		for _, row := range v {
			result = append(result, row...)
		}
		return result
	case [][][]float64:
		var result []float64
		for _, matrix := range v {
			for _, row := range matrix {
				result = append(result, row...)
			}
		}
		return result
	default:
		return nil
	}
}

// flattenInt64 recursively flattens multi-dimensional int64 data.
func flattenInt64(val interface{}) []int64 {
	switch v := val.(type) {
	case []int64:
		return v
	case [][]int64:
		var result []int64
		for _, row := range v {
			result = append(result, row...)
		}
		return result
	case [][][]int64:
		var result []int64
		for _, matrix := range v {
			for _, row := range matrix {
				result = append(result, row...)
			}
		}
		return result
	default:
		return nil
	}
}

// flattenInt32 recursively flattens multi-dimensional int32 data.
func flattenInt32(val interface{}) []int32 {
	switch v := val.(type) {
	case []int32:
		return v
	case [][]int32:
		var result []int32
		for _, row := range v {
			result = append(result, row...)
		}
		return result
	case [][][]int32:
		var result []int32
		for _, matrix := range v {
			for _, row := range matrix {
				result = append(result, row...)
			}
		}
		return result
	default:
		return nil
	}
}

// flattenBool recursively flattens multi-dimensional bool data.
func flattenBool(val interface{}) []bool {
	switch v := val.(type) {
	case []bool:
		return v
	case [][]bool:
		var result []bool
		for _, row := range v {
			result = append(result, row...)
		}
		return result
	default:
		return nil
	}
}
