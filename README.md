# Termite

[![Build status](https://github.com/antflydb/termite/actions/workflows/termite-go.yml/badge.svg)](https://github.com/antflydb/termite/actions)
[![Docs](https://img.shields.io/badge/docs-antfly.io-blue)](https://antfly.io/termite)

ML inference service for embeddings, chunking, reranking, classification, NER, OCR, transcription, text generation, and more — with two-tier caching (memory + singleflight).

Termite is the companion ML service for [Antfly](https://github.com/antflydb/antfly), the distributed search engine. It runs automatically in Antfly's swarm mode and can also be used standalone. If you're new, the [Antfly quickstart](https://antfly.io/docs/guides/quickstart) is the fastest way to see everything working together.

**[Documentation](https://antfly.io/termite)** | **[Discord](https://discord.gg/zrdjguy84P)**

## Features

- **Embeddings** — dense and sparse vectors, multimodal (text, images, audio)
- **Chunking** — semantic text segmentation
- **Reranking** — cross-encoder relevance scoring
- **Classification** — zero-shot text classification (NLI-based, 100+ languages)
- **Recognition (NER)** — named entity recognition, zero-shot labels, relation extraction
- **Reading (OCR)** — document understanding, OCR, visual question answering
- **Transcription** — speech-to-text (Whisper, Wav2Vec2)
- **Extraction** — schema-based structured data extraction
- **Rewriting** — paraphrasing, question generation (Seq2Seq)
- **Generation** — text generation with tool calling (OpenAI-compatible)
- **Multiple backends** — ONNX Runtime, XLA (TPU/CUDA), pure Go
- **SIMD / SME acceleration** — vector math uses hardware intrinsics via [go-highway](https://github.com/ajroetker/go-highway) on x86 and ARM
- **Native Go ML** — XLA backend powered by [GoMLX](https://github.com/gomlx/gomlx) and [GoLLMX](https://github.com/gomlx/gollmx), working toward making native Go ML/LLM inference a reality
- **Kubernetes operator** — autoscaling with TermitePool and TermiteRoute CRDs (see [antfly/pkg/termite-operator](https://github.com/antflydb/antfly/tree/main/pkg/termite-operator))

## Running

```bash
# Standalone server
go run ./cmd/termite run
```

## Inference Backends

Termite supports multiple inference backends, selected via build tags. The **omni** build includes everything so you can pick at runtime.

| Build | Tags | Description | Use Case |
|-------|------|-------------|----------|
| Pure Go | (none) | No CGO, always works | Development, testing |
| ONNX | `onnx,ORT` | Fast CPU/GPU via ONNX Runtime | Production (recommended) |
| XLA | `xla,XLA` | TPU/CUDA via GoMLX | Cloud TPU, NVIDIA GPU |
| **Omni** | `onnx,ORT,xla,XLA` | All backends | Maximum flexibility |

### Omni Build (Recommended)

Includes both ONNX and XLA backends — pick which one to use at runtime without recompiling.

```bash
# Download dependencies for all platforms
./scripts/download-onnxruntime.sh
./scripts/download-pjrt.sh

# Build omni binary
CGO_ENABLED=1 go build -tags="onnx,ORT,xla,XLA" -o termite ./pkg/termite/cmd

# Run with backend priority (tries in order until one works)
./termite run --backend-priority="onnx:cuda,xla:tpu,onnx:cpu,go"
```

### ONNX Runtime

**Dependencies:**
- [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) - download for your platform or install with homebrew
- [Tokenizers](https://github.com/daulet/tokenizers/releases/) - HuggingFace tokenizers bindings

```bash
# Download dependencies
./scripts/download-onnxruntime.sh

# Or manually (macOS with homebrew)
CGO_ENABLED=1 \
DYLD_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib \
CGO_LDFLAGS="-L$(pwd) -ltokenizers" \
go run -tags="onnx,ORT" ./pkg/termite/cmd run
```

### XLA Runtime (TPU/GPU)

For TPU or CUDA GPU acceleration via GoMLX XLA backend. Hardware is autodetected.

**Dependencies:**
- [PJRT CPU Binaries](https://github.com/gomlx/pjrt-cpu-binaries) - prebuilt XLA PJRT plugins
- [Tokenizers](https://github.com/daulet/tokenizers/releases/) - HuggingFace tokenizers bindings

```bash
# Download dependencies
./scripts/download-pjrt.sh

# Build with XLA support
go build -tags="xla,XLA" -o termite ./pkg/termite/cmd

# Run with autodetection (TPU > CUDA > CPU)
./termite run
```

**Autodetection:**
- **TPU**: Detected via `libtpu.so`, `/dev/accel*` devices, or GKE TPU node metadata
- **CUDA**: Detected via `nvidia-smi` or `libcudart.so` in library path

**Installing Additional PJRT Plugins:**

The omni and XLA builds bundle a CPU PJRT plugin that's auto-discovered from `lib/` next to the binary. For TPU or CUDA, install the right plugin:

```bash
# Install TPU plugin (for Google Cloud TPU)
go run github.com/gomlx/go-xla/cmd/pjrt_installer@latest -plugin=tpu

# Install CUDA plugin (for NVIDIA GPU)
go run github.com/gomlx/go-xla/cmd/pjrt_installer@latest -plugin=cuda

# Install to a specific location
go run github.com/gomlx/go-xla/cmd/pjrt_installer@latest -plugin=tpu -path=/usr/local/lib/go-xla
```

Installed plugins are found automatically via standard go-xla search paths. To override, set `PJRT_PLUGIN_LIBRARY_PATH`.

**Platform Availability:**

| Platform | PJRT CPU | Notes |
|----------|----------|-------|
| linux-amd64 | Yes | |
| linux-arm64 | Yes | |
| darwin-arm64 | Yes | Apple Silicon |
| darwin-amd64 | No | Intel Mac not supported upstream |

## Models

Pull from registry:

```bash
termite pull bge-small-en-v1.5
termite pull mxbai-rerank-base-v1
termite pull chonky-mmbert-small-multilingual-1

# List available models
termite list --remote
```

Models auto-discovered from `chunker_models_dir`, `embedder_models_dir`, `reranker_models_dir`.

### Available Models

#### Embedders

| Model | Size | Dims | Variants | Notes |
|-------|------|------|----------|-------|
| `bge-small-en-v1.5` | 128MB | 384 | f16, i8 | Fast English embeddings |
| `all-MiniLM-L6-v2` | 87MB | 384 | f32, f16, i8 | Fastest, good quality |
| `all-mpnet-base-v2` | 418MB | 768 | f32, f16, i8 | Best sentence-transformers accuracy |
| `nomic-embed-text-v1.5` | 548MB | 768 | f16, i8 | 8K context, Matryoshka dims |
| `bge-m3` | 2.2GB | 1024 | f16, i8 | 100+ languages, 8K context |
| `gte-Qwen2-1.5B-instruct` | 6GB | 1536 | f16 | 32K context, instruction-following |
| `snowflake-arctic-embed-l-v2.0` | 1.3GB | 1024 | f16, i8 | Retrieval-optimized, Matryoshka |
| `stella_en_1.5B_v5` | 6GB | 1024 | f16 | Premium English, top MTEB scores |
| `embeddinggemma-300m-ONNX` | 1.2GB | 768 | f16, q4, q4f16 | Multilingual, edge-optimized |
| `splade-cocondenser-ensembledistil` | — | sparse | f32 | Sparse embeddings (SPLADE) |

#### Multimodal Embedders

| Model | Size | Dims | Variants | Notes |
|-------|------|------|----------|-------|
| `clip-vit-base-patch32` | 584MB | 512 | f16, i8 | Text + image embeddings (CLIP) |
| `clipclap` | — | 512 | — | Text + image (CLIP variant) |
| `clap-htsat-unfused` | — | 512 | — | Audio + text embeddings (CLAP) |

#### Rerankers

| Model | Size | Variants |
|-------|------|----------|
| `mxbai-rerank-base-v1` | 713MB | f16, i8 |

#### Chunkers

| Model | Size | Variants |
|-------|------|----------|
| `chonky-mmbert-small-multilingual-1` | 570MB | f16, i8 |

#### Classifiers

| Model | Size | Variants | Notes |
|-------|------|----------|-------|
| `mDeBERTa-v3-base-mnli-xnli` | — | f32, f16, i8 | Zero-shot, 100+ languages |
| `bart-large-mnli` | — | f32, f16, i8 | Zero-shot, English |

#### Recognizers (NER)

| Model | Size | Variants | Capabilities |
|-------|------|----------|--------------|
| `bert-base-NER` | 413MB | f32, f16, i8 | labels |
| `bert-large-NER` | 1.3GB | f32, f16, i8 | labels |
| `gliner_small-v2.1` | 199MB | f32, f16, i8 | labels, zeroshot |
| `gliner2-base-v1` | — | f32, f16, i8 | labels, zeroshot (improved) |
| `gliner-multitask-large-v0.5` | 1.3GB | f32, f16, i8 | labels, zeroshot, relations, answers |
| `rebel-large` | 3.0GB | - | relations |

#### Readers (OCR / Document Understanding)

| Model | Size | Variants | Notes |
|-------|------|----------|-------|
| `trocr-base-printed` | — | — | Printed text OCR |
| `donut-base-finetuned-cord-v2` | — | — | Receipt/form parsing |
| `donut-base-finetuned-docvqa` | — | — | Document question answering |
| `moondream2` | — | — | General vision understanding |

#### Transcribers (Speech-to-Text)

| Model | Size | Variants | Notes |
|-------|------|----------|-------|
| `whisper-tiny.en` | — | — | OpenAI Whisper, English |

#### Extractors

| Model | Size | Variants | Notes |
|-------|------|----------|-------|
| `gliner2-base-v1` | — | f32, f16, i8 | Schema-based field extraction |

#### Rewriters

| Model | Size | Variants |
|-------|------|----------|
| `flan-t5-small-squad-qg` | 569MB | - |
| `pegasus_paraphrase` | 4.5GB | - |

#### Generators

| Model | Size | Variants |
|-------|------|----------|
| `functiongemma-270m-it` | 1.1GB | - |
| `gemma-3-1b-it` | 3.7GB | - |

### Model Variants

Models come in multiple precision variants, trading off size and speed for accuracy:

| Variant | File | Description |
|---------|------|-------------|
| (default) | `model.onnx` | FP32 baseline - highest accuracy |
| `f16` | `model_f16.onnx` | FP16 - ~50% smaller, recommended for ARM64/M-series |
| `i8` | `model_i8.onnx` | INT8 dynamic quantization - smallest, fastest CPU inference |

Pull specific variants:

```bash
# Pull using variant suffix (recommended)
termite pull bge-small-en-v1.5-i8

# Or use --variants flag
termite pull --variants i8 bge-small-en-v1.5

# Pull multiple models with same variant
termite pull bge-small-en-v1.5-i8 mxbai-rerank-base-v1-i8

# Pull multiple variants for one model
termite pull --variants f16,i8 bge-small-en-v1.5
```

Use variants in config:

```yaml
embedder:
  provider: termite
  model: bge-small-en-v1.5-f16  # Use FP16 variant
```

Termite auto-selects the best available variant if not specified.

## API

All endpoints accept JSON. See `openapi.yaml` for full schema details.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/embed` | POST | Generate dense and sparse embeddings (text, image, audio) |
| `/api/chunk` | POST | Chunk text into semantic segments |
| `/api/rerank` | POST | Rerank documents by relevance |
| `/api/classify` | POST | Zero-shot text classification |
| `/api/recognize` | POST | Named entity recognition |
| `/api/read` | POST | OCR and document understanding |
| `/api/transcribe` | POST | Speech-to-text transcription |
| `/api/extract` | POST | Schema-based structured data extraction |
| `/api/rewrite` | POST | Text rewriting (paraphrase, question generation) |
| `/api/generate` | POST | Text generation (OpenAI-compatible) |
| `/api/models` | GET | List available models |
| `/api/version` | GET | Version info |

### Multimodal Input

The `/api/embed` endpoint supports multimodal input using the OpenAI content format:

```json
{
  "model": "clip-vit-base-patch32",
  "input": [
    {
      "content": [
        {"type": "text", "text": "a photo of a cat"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  ]
}
```

## Configuration

Config via file (`termite.yaml`), flags, or environment variables (`TERMITE_` prefix):

```yaml
api_url: "http://localhost:11433"
models_dir: "./models"

# Backend priority with optional device specifiers
# Format: "backend" or "backend:device"
# Devices: auto (default), cuda, coreml, tpu, cpu
backend_priority:
  - onnx:cuda      # Try ONNX with CUDA first
  - xla:tpu        # Then XLA with TPU
  - onnx:cpu       # Fall back to ONNX CPU
  - go             # Pure Go fallback (always works)

keep_alive: "5m"
max_loaded_models: 3
allow_downloads: false
log:
  level: info
  style: terminal
```

### Backend Priority

The `backend_priority` setting controls which backends Termite tries, in order. Each entry can be:

- **Backend only**: `onnx`, `xla`, `go` - uses auto device detection
- **Backend with device**: `onnx:cuda`, `xla:tpu`, `onnx:coreml` - explicit device

**Available backends** (depend on build tags):
| Backend | Build Tags | Devices Supported |
|---------|------------|-------------------|
| `onnx` | `onnx,ORT` | `cuda`, `coreml` (macOS), `cpu` |
| `xla` | `xla,XLA` | `tpu`, `cuda`, `cpu` |
| `go` | (none) | `cpu` only |

**Example configurations:**

```yaml
# GPU-first with CPU fallback
backend_priority: ["onnx:cuda", "xla:cuda", "onnx:cpu", "go"]

# macOS with CoreML acceleration
backend_priority: ["onnx:coreml", "go"]

# Cloud TPU deployment
backend_priority: ["xla:tpu", "xla:cpu"]

# Simple auto-detection (default)
backend_priority: ["onnx", "xla", "go"]
```

## Kubernetes Operator

The Termite Kubernetes operator — `TermitePool` (autoscaling replicas, hardware selection) and `TermiteRoute` (traffic routing by model/endpoint) — lives in the antfly monorepo at [`pkg/termite-operator`](https://github.com/antflydb/antfly/tree/main/pkg/termite-operator). See that directory for CRD definitions, manifests, and deployment docs.

The [model registry protocol](specs/tla/model-registry-protocol.tla) used by the operator is formally specified in TLA+ in this repo.

## Community

[Discord](https://discord.gg/zrdjguy84P) for questions, discussion, and updates.

## License

Apache License 2.0
