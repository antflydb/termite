# Termite

[![Build status](https://github.com/antflydb/termite/actions/workflows/termite-go.yml/badge.svg)](https://github.com/antflydb/termite/actions)
[![Docs](https://img.shields.io/badge/docs-antfly.io-blue)](https://antfly.io/termite)

ML inference service for embeddings, chunking, and reranking with two-tier caching (memory + singleflight).

**[Documentation](https://antfly.io/termite)** | **[Discord](https://discord.gg/zrdjguy84P)**

## Running

```bash
# Standalone server
go run ./cmd/termite run
```

## Inference Backends

Termite supports multiple inference backends selected via build tags. The **omni** build includes all backends for maximum flexibility.

| Build | Tags | Description | Use Case |
|-------|------|-------------|----------|
| Pure Go | (none) | No CGO, always works | Development, testing |
| ONNX | `onnx,ORT` | Fast CPU/GPU via ONNX Runtime | Production (recommended) |
| XLA | `xla,XLA` | TPU/CUDA via GoMLX | Cloud TPU, NVIDIA GPU |
| **Omni** | `onnx,ORT,xla,XLA` | All backends | Maximum flexibility |

### Omni Build (Recommended)

The omni build includes both ONNX and XLA backends, enabling runtime backend selection without recompilation.

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

For ~16x faster CPU inference. See `lib/hugot/README.md` for setup.

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

The omni and XLA builds bundle a CPU PJRT plugin that's auto-discovered from `lib/` next to the binary. To use TPU or CUDA acceleration, install the appropriate plugin:

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
| linux-amd64 | ✓ | |
| linux-arm64 | ✓ | |
| darwin-arm64 | ✓ | Apple Silicon |
| darwin-amd64 | ✗ | Intel Mac not supported upstream |

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

| Model | Size | Variants |
|-------|------|----------|
| `bge-small-en-v1.5` | 128MB | f16, i8 |
| `all-MiniLM-L6-v2` | 87MB | f32, f16, i8 |
| `all-mpnet-base-v2` | 418MB | f32, f16, i8 |
| `clip-vit-base-patch32` | 584MB | f16, i8 |

#### Rerankers

| Model | Size | Variants |
|-------|------|----------|
| `mxbai-rerank-base-v1` | 713MB | f16, i8 |

#### Chunkers

| Model | Size | Variants |
|-------|------|----------|
| `chonky-mmbert-small-multilingual-1` | 570MB | f16, i8 |

#### Recognizers (NER)

| Model | Size | Variants | Capabilities |
|-------|------|----------|--------------|
| `bert-base-NER` | 413MB | f32, f16, i8 | labels |
| `bert-large-NER` | 1.3GB | f32, f16, i8 | labels |
| `gliner_small-v2.1` | 199MB | f32, f16, i8 | labels, zeroshot |
| `gliner-multitask-large-v0.5` | 1.3GB | f32, f16, i8 | labels, zeroshot, relations, answers |
| `rebel-large` | 3.0GB | - | relations |

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

Models support multiple precision variants for different performance/accuracy tradeoffs:

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

## Kubernetes Operator

Deploy on GKE with TPU support using the Termite Operator.

### Custom Resources

**TermitePool**: Manages a pool of Termite replicas with autoscaling.

```yaml
apiVersion: termite.antfly.io/v1alpha1
kind: TermitePool
metadata:
  name: embeddings-pool
spec:
  workloadType: read-heavy
  models:
    preload:
      - name: bge-small-en-v1.5
        variant: i8
        priority: high
        strategy: eager    # Always loaded, never evicted
      - name: mxbai-rerank-base-v1
        variant: i8
        priority: high
        # strategy defaults to loadingStrategy (lazy)
    loadingStrategy: lazy  # Default for models without explicit strategy
    keepAlive: 5m          # Idle timeout for lazy models
  replicas:
    min: 2
    max: 10
  hardware:
    accelerator: tpu-v5-lite-podslice
    topology: "2x2"
  autoscaling:
    enabled: true
    metrics:
      - type: queue-depth
        target: "50"
```

**TermiteRoute**: Routes traffic to pools based on model or endpoint.

### Running the Operator

```bash
# Build operator
go build -o termite-operator ./cmd/termite-operator

# Generate CRDs and RBAC manifests
make generate
```

See `pkg/operator/` for CRD definitions and controller implementation.

## API

See `openapi.yaml` for endpoints: `/api/embeddings`, `/api/chunk`, `/api/rerank`.

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
log:
  level: info
  style: terminal
```

### Backend Priority

The `backend_priority` setting controls which inference backends Termite tries, in order. Each entry can be:

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

## Community

Join our [Discord](https://discord.gg/zrdjguy84P) for support, discussion, and updates.

## License

Apache License 2.0
