# backends

The `backends` package provides a unified interface for ML inference with multiple backend support:

- **ONNX Runtime**: Fastest inference, supports CPU/CUDA/CoreML (requires `onnx,ORT` build tags)
- **GoMLX**: Pure Go inference with optional XLA acceleration (always available)
  - **simplego**: Pure Go engine, always available, no dependencies
  - **xla**: Hardware accelerated (CUDA, TPU, optimized CPU) via PJRT

## Architecture

This package replaces the `hugot` dependency with a custom stack:

- **go-huggingface**: Hub download, tokenizers (SentencePiece, WordPiece, BPE), SafeTensors parsing
- **huggingface-gomlx**: Model architectures (BERT, LLaMA, DeBERTa), GoMLX inference
- **onnx-gomlx**: ONNX model execution via GoMLX
- **onnxruntime_go**: Direct ONNX Runtime inference

## Usage

### Basic Pipeline Usage

```go
import (
    "github.com/antflydb/antfly/termite/pkg/termite/lib/backends"
    "github.com/gomlx/go-huggingface/hub"
    "github.com/gomlx/go-huggingface/tokenizers"
)

// Create tokenizer from HuggingFace repo
repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
tokenizer, err := tokenizers.New(repo)

// Get default backend and load model
backend := backends.GetDefaultBackend()
model, err := backend.Loader().Load(modelPath,
    backends.WithNormalization(true),
    backends.WithPooling("mean"))

// Create pipeline
pipeline := backends.NewPipeline(tokenizer, model, nil)
defer pipeline.Close()

// Generate embeddings
output, err := pipeline.Run(ctx, []string{"Hello world"})
```

### Using SessionManager

```go
manager := backends.NewSessionManager()
defer manager.Close()

// Configure priority
manager.SetPriority([]backends.BackendSpec{
    {Backend: backends.BackendONNX, Device: backends.DeviceCUDA},
    {Backend: backends.BackendGoMLX, Device: backends.DeviceAuto}, // auto-selects xla or simplego
    {Backend: backends.BackendONNX, Device: backends.DeviceCPU},
})

// Load model with backend selection
model, backendUsed, err := manager.LoadModel(modelPath,
    []string{"onnx", "gomlx"}, // Supported backends
    backends.WithONNXFile("model_f16.onnx"))
```

### Feature Extraction Pipeline

```go
embedder, ok := model.(backends.FeatureExtractionModel)
if ok {
    embeddings, err := embedder.EmbedBatch(ctx, inputs)
}

// Or use the typed pipeline
pipeline := backends.NewFeatureExtractionPipeline(tokenizer, embedder, nil)
embeddings, err := pipeline.Embed(ctx, []string{"Hello", "World"})
```

## Build Tags

| Tags | Backends Available | Use Case |
|------|-------------------|----------|
| (none) | GoMLX (simplego) | Development, cross-platform |
| `onnx,ORT` | GoMLX + ONNX | Production CPU/GPU |

The GoMLX backend automatically detects available engines:
- If XLA/PJRT is available, uses hardware acceleration
- Otherwise falls back to pure Go (simplego)

Build example:
```bash
go build -tags="onnx,ORT" ./cmd/termite
```

## Model Interface

All models implement the base `Model` interface:

```go
type Model interface {
    Forward(ctx context.Context, inputs *ModelInputs) (*ModelOutput, error)
    Close() error
    Name() string
    Backend() BackendType
}
```

Specialized interfaces extend this:

- `FeatureExtractionModel` - Embedding generation
- `TokenClassificationModel` - NER, chunking
- `SequenceClassificationModel` - Text classification
- `CrossEncoderModel` - Reranking
- `Seq2SeqModel` - Text generation
- `VisionModel` - Image processing
- `MultimodalModel` - Text + image

## Configuration Options

```go
// Functional options for model loading
model, err := loader.Load(path,
    backends.WithONNXFile("model_quantized.onnx"),
    backends.WithMaxLength(512),
    backends.WithNormalization(true),
    backends.WithPooling("mean"),      // "mean", "cls", "max", "none"
    backends.WithGPUMode(backends.GPUModeAuto),
    backends.WithGoMLXBackend(backends.GoMLXBackendXLA), // or GoMLXBackendSimpleGo
    backends.WithBatchSize(32),
    backends.WithNumThreads(4),
)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ONNXRUNTIME_ROOT` | ONNX Runtime library directory |
| `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` | Library search paths |
| `GOMLX_BACKEND` | GoMLX engine override (e.g., `xla`, `simplego`) |
| `PJRT_PLUGIN_LIBRARY_PATH` | PJRT plugin location for XLA |
| `TERMITE_FORCE_COREML` | Force CoreML on macOS (experimental) |
| `TERMITE_GPU` | GPU mode override (auto/cuda/tpu/coreml/off) |

## Directory Structure

```
lib/backends/
├── backend.go           # Registry and backend interface
├── backend_onnx.go      # ONNX Runtime (Linux/Windows)
├── backend_onnx_darwin.go  # ONNX Runtime (macOS/CoreML)
├── backend_gomlx.go     # GoMLX (HuggingFace + ONNX models, xla/simplego engines)
├── gpu.go               # GPU detection utilities
├── model.go             # Model interfaces, options, pooling utilities
├── pipeline.go          # Pipeline types (tokenizer + model)
├── session_manager.go   # Multi-backend session management
├── types.go             # Common types
└── README.md            # This file
```

## Migration from hugot

The package maintains similar patterns to the old `hugot` package:

| Old (hugot) | New (backends) |
|-------------|----------------|
| `hugot.NewSession()` | `backends.GetDefaultBackend().Loader().Load()` |
| `hugot.SessionManager` | `backends.SessionManager` |
| `hugot.BackendType` | `backends.BackendType` |
| `khugot.FeatureExtractionPipeline` | `backends.FeatureExtractionPipeline` |
| `options.WithOption` | `backends.LoadOption` |

Key differences:
1. Tokenizers are now separate (from go-huggingface)
2. Pipelines pair tokenizer + model explicitly
3. Models are loaded via `ModelLoader` interface
4. Functional options pattern for configuration

### Migration Status

All task packages have been migrated to use the new `pipelines.Pipeline` type:

| Package | Old File | New File | Status |
|---------|----------|----------|--------|
| `lib/embeddings` | `hugot.go` | `embedder.go` | ✅ Complete |
| `lib/reranking` | `hugot.go` | `reranker.go` | ✅ Complete |
| `lib/chunking` | `hugot.go` | `chunker.go` | ✅ Complete |
| `lib/ner` | `hugot.go` | `ner_model.go` | ✅ Complete |
| `lib/classification` | `hugot.go` | `zsc_model.go` | ✅ Complete |
| `lib/seq2seq` | `hugot.go` | `seq2seq_model.go` | ✅ Complete |

Each migrated package provides:
- `PooledXxx` type for concurrent access with semaphore-based pooling
- `PooledXxxConfig` for configuration
- Factory function `NewPooledXxx(cfg, sessionManager)` returning the model and backend type used

The old `hugot.go` files are retained for compatibility until full deprecation.
