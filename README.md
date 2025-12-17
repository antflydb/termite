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

### ONNX Runtime

For ~16x faster CPU inference. See `lib/hugot/README.md` for setup.

**Dependencies:**
- [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) - download for your platform or install with homebrew
- [Tokenizers](https://github.com/daulet/tokenizers/releases/) - HuggingFace tokenizers bindings

```bash
# Assuming you have the libtokenizers.a in your working directory
# and installed ONNX with homebrew
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
# Build with XLA support
go build -tags="xla,XLA" -o termite ./cmd/termite

# Run with autodetection (TPU > CUDA > CPU)
./termite run
```

**Autodetection:**
- **TPU**: Detected via `libtpu.so`, `/dev/accel*` devices, or GKE TPU node metadata
- **CUDA**: Detected via `nvidia-smi` or `libcudart.so` in library path

**Override autodetection:**
```bash
GOMLX_BACKEND="xla:tpu" ./termite run   # Force TPU
GOMLX_BACKEND="xla:cuda" ./termite run  # Force CUDA
GOMLX_BACKEND="xla:cpu" ./termite run   # Force CPU
```

**Environment Variables:**
- `GOMLX_BACKEND`: Override autodetection (`xla:tpu`, `xla:cuda`, `xla:cpu`)
- `PJRT_PLUGIN_LIBRARY_PATH`: Custom path to TPU PJRT plugin (optional)

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
gpu: "auto"  # auto, tpu, cuda, coreml, off
keep_alive: "5m"
max_loaded_models: 3
log:
  level: info
  style: terminal
```

## Community

Join our [Discord](https://discord.gg/zrdjguy84P) for support, discussion, and updates.

## License

Apache License 2.0
