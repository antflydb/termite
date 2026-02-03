# 004: Use go-huggingface + huggingface-gomlx

## Summary

- **go-huggingface**: Hub download, tokenizers, safetensors parsing
- **huggingface-gomlx**: Model architectures, GoMLX inference
- **onnxruntime_go**: Direct ONNX Runtime inference (for ONNX models)

## Motivation

- go-huggingface + huggingface-gomlx gives us full control
- Cleaner separation: tokenization is decoupled from inference
- Pure Go tokenizers (no CGO required for tokenization)

## New Architecture

### Package Structure

```
lib/models/             # Shared types and interfaces
    └── models.go       # Backend interface, Model[B], PoolingStrategy, etc.

lib/backends/           # Renamed from lib/hugot/
    ├── backend.go      # Backend registry and selection
    ├── session.go      # Session interface
    ├── model.go        # Model struct, LoadModel
    ├── model_ort.go    # ORTModel implements models.Backend
    ├── model_gomlx.go  # GoMLXModel implements models.Backend
    ├── pipeline.go     # Pipeline = Tokenizer + Model
    │
    └── gomlx/          # GoMLX-specific implementations
        ├── common.go       # EngineManager, shared utilities
        ├── huggingface.go  # HFModel - loads SafeTensors via huggingface-gomlx
        └── onnx.go         # ONNXModel - loads ONNX via onnx-gomlx

lib/embeddings/         # Uses Pipeline, adds L2 normalization
lib/reranking/          # Uses Pipeline, adds score extraction
lib/chunking/           # Uses Pipeline, adds boundary detection
lib/ner/                # Uses Pipeline, adds BIO parsing
lib/classification/     # Uses Pipeline, adds zero-shot logic
lib/seq2seq/            # Uses Pipeline, adds generation loop
lib/generation/         # Uses Pipeline, adds streaming (TBD)
lib/reading/            # Uses Pipeline, adds vision (TBD)
```

### GoMLX Subpackage (`lib/backends/gomlx/`)

The GoMLX backend supports two model formats with pluggable inference engines:

**Model Formats:**
- **HuggingFace** (`huggingface.go`): Loads SafeTensors + config.json via `huggingface-gomlx`
- **ONNX** (`onnx.go`): Loads .onnx files via `onnx-gomlx`, converts to GoMLX graphs

**Inference Engines** (auto-detected or configurable):
- `simplego`: Pure Go, always available, slower
- `xla`: Hardware accelerated (CUDA, TPU, optimized CPU), requires XLA/PJRT

**Common utilities** (`common.go`):
- `EngineManager`: Manages GoMLX backend selection (xla vs simplego)
- Pooling, normalization functions shared with `lib/models/`

### Dependency Graph

```
go-huggingface (pure Go)
├── hub/              # Download from HF Hub
├── tokenizers/       # SentencePiece, WordPiece, BPE
└── models/safetensors/  # Parse weights

huggingface-gomlx (depends on GoMLX)
└── architectures/    # BERT, LLaMA, DeBERTa builders

onnxruntime_go        # ONNX Runtime CGO bindings
ortgenai              # ONNX GenAI for text generation

termite/lib/backends/
├── Uses go-huggingface for tokenization
├── Uses onnxruntime_go for ONNX inference
└── Uses huggingface-gomlx for HF model inference
```

### Pipeline Type

The core abstraction - pairs tokenizer with model:

```go
// lib/backends/pipeline.go
type Pipeline struct {
    Tokenizer tokenizers.Tokenizer  // from go-huggingface
    Model     Model                  // interface for inference
    Config    *PipelineConfig
}

// Convenience methods that use both
func (p *Pipeline) Encode(texts []string) (*EncodedBatch, error)
func (p *Pipeline) Forward(batch *EncodedBatch) (*ModelOutput, error)

// Components are accessible for direct use
func (p *Pipeline) TokenCount(text string) int {
    return len(p.Tokenizer.Encode(text))
}
```

### Model Interface

```go
// lib/backends/model.go
type Model interface {
    Forward(inputs *ModelInputs) (*ModelOutput, error)
    Close() error
}

type ModelInputs struct {
    InputIDs      [][]int32
    AttentionMask [][]int32
    TokenTypeIDs  [][]int32  // optional
}

type ModelOutput struct {
    LastHiddenState [][]float32  // [batch, seq, hidden]
    Logits          [][]float32  // [batch, classes] for classification
    Embeddings      [][]float32  // [batch, hidden] for pooled output
}
```

### Session-based Architecture (Vision2Seq / Generative)

For complex models like Vision2Seq, we use a session-based architecture where backends provide
low-level session primitives and pipelines build models from sessions:

```go
// lib/backends/session.go - Low-level tensor I/O
type Session interface {
    Run(inputs []NamedTensor) ([]NamedTensor, error)
    InputInfo() []TensorInfo
    OutputInfo() []TensorInfo
    Close() error
}

type SessionFactory interface {
    CreateSession(modelPath string, opts ...SessionOption) (Session, error)
    Backend() BackendType
}

// Optional interface backends implement to provide session factory
type SessionFactoryProvider interface {
    SessionFactory() SessionFactory
}
```

```go
// lib/backends/session_manager.go - Gets session factory respecting backend priority
factory, backendType, err := sessionManager.GetSessionFactoryForModel(modelBackends)
```

```go
// lib/pipelines/vision2seq_model.go - Builds model from sessions
model, err := pipelines.LoadVision2SeqModel(modelPath, factory)
```

This separation allows:
- **Backends**: Focus on session creation (ONNX Runtime, GoMLX, etc.)
- **Pipelines**: Build domain-specific models from sessions
- **No circular dependencies**: Backends don't import pipelines

## References

- [go-huggingface](https://github.com/gomlx/go-huggingface) - Hub, tokenizers, safetensors
- [huggingface-gomlx](https://github.com/gomlx/huggingface-gomlx) - Model architectures (using ajroetker fork)
- [onnxruntime_go](https://github.com/yalue/onnxruntime_go) - ONNX Runtime bindings
- [ortgenai](https://github.com/knights-analytics/ortgenai) - ONNX GenAI bindings for text generation
- [onnx-gomlx](https://github.com/gomlx/onnx-gomlx) - ONNX → GoMLX conversion
- hugot `feature/genai-multimodal-with-seq2seq` branch - Vision2Seq implementation

## Notes

### GoMLX Package Restructuring (v0.26+)

GoMLX upstream restructured packages in recent versions. Import paths changed:

| Old Path | New Path |
|----------|----------|
| `github.com/gomlx/gomlx/graph` | `github.com/gomlx/gomlx/pkg/core/graph` |
| `github.com/gomlx/gomlx/tensors` | `github.com/gomlx/gomlx/pkg/core/tensors` |
| `github.com/gomlx/gomlx/ml/context` | `github.com/gomlx/gomlx/pkg/ml/context` |
| `github.com/gomlx/gomlx/ml/context/ctxbuilder` | (merged into `pkg/ml/context`) |

API changes:
- `ctxbuilder.NewExec()` → `context.NewExecAny()` (returns `(*Exec, error)`)
- `exec.Call()` → `exec.Exec()` (returns `([]*tensors.Tensor, error)`)
- `context.ExecOnceN()` now returns `([]*tensors.Tensor, error)`
