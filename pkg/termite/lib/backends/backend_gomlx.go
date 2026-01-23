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

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	mlctx "github.com/gomlx/gomlx/pkg/ml/context"
	hfmodels "github.com/gomlx/huggingface-gomlx"
	"github.com/gomlx/huggingface-gomlx/architectures/bert"
	"github.com/gomlx/onnx-gomlx/onnx"

	// Import simplego backend - always available
	_ "github.com/gomlx/gomlx/backends/simplego"
)

func init() {
	RegisterBackend(&gomlxBackend{})
}

// gomlxBackend implements Backend using GoMLX for inference.
//
// This backend supports two model formats:
//   - HuggingFace: SafeTensors + config.json (via huggingface-gomlx)
//   - ONNX: .onnx model files (via onnx-gomlx)
//
// And multiple inference engines:
//   - simplego: Pure Go, always available, slower
//   - xla: Hardware accelerated (CUDA, TPU, optimized CPU), requires XLA/PJRT
//
// The model format and inference engine are auto-detected by default,
// but can be overridden via LoadOptions.
type gomlxBackend struct {
	engineMgr *engineManager
}

func (b *gomlxBackend) Type() BackendType {
	return BackendGoMLX
}

func (b *gomlxBackend) Name() string {
	return "GoMLX"
}

func (b *gomlxBackend) Available() bool {
	return true
}

func (b *gomlxBackend) Priority() int {
	// Lower priority than direct ONNX Runtime, but always available
	return 30
}

func (b *gomlxBackend) Loader() ModelLoader {
	if b.engineMgr == nil {
		b.engineMgr = newEngineManager()
	}
	return &gomlxModelLoader{backend: b}
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
		return backends.NewWithConfig(backendType)
	}

	// Use cached default engine
	if m.defaultEngine != nil {
		return m.defaultEngine, nil
	}

	// Auto-detect: try xla first, fall back to simplego
	engine, err := backends.NewWithConfig("xla")
	if err != nil {
		// XLA not available, use simplego
		engine, err = backends.NewWithConfig("simplego")
		if err != nil {
			return nil, err
		}
	}

	m.defaultEngine = engine
	return engine, nil
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
		hfModel: hfModel,
		path:    path,
		config:  config,
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
		onnxModel: onnxModel,
		path:      path,
		config:    config,
	}, nil
}

func (l *gomlxModelLoader) SupportsModel(path string) bool {
	format := detectGoMLXModelFormat(path)
	return format != ModelFormatAuto
}

func (l *gomlxModelLoader) Backend() BackendType {
	return BackendGoMLX
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
	hfModel   *hfModel
	onnxModel *onnxModel
	path      string
	config    *LoadConfig
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
	return BackendGoMLX
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

	// Build the ONNX graph function
	graphFn := func(mlCtx *mlctx.Context, inputs []*graph.Node) []*graph.Node {
		inputMap := map[string]*graph.Node{
			"input_ids":      inputs[0],
			"attention_mask": inputs[1],
		}
		return m.onnxModel.CallGraph(mlCtx.Reuse(), inputs[0].Graph(), inputMap)
	}

	// Execute
	results, err := mlctx.ExecOnceN(m.engine, m.ctx, graphFn, inputIDsTensor, attentionMaskTensor)
	if err != nil {
		return nil, fmt.Errorf("exec failed: %w", err)
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("no output from ONNX model")
	}

	// First output should be last_hidden_state [batch, seq, hidden]
	output := results[0]
	shape := output.Shape()

	if len(shape.Dimensions) < 3 {
		return nil, fmt.Errorf("unexpected output shape: %v", shape.Dimensions)
	}

	// Extract data
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

func (m *onnxModel) close() error {
	return nil
}
