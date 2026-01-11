# Termite Dashboard Reference Documentation

This document provides comprehensive reference information for building a Termite dashboard UI, covering model downloads, runtime configuration, and hardware settings.

---

## Table of Contents

1. [CLI Commands Overview](#cli-commands-overview)
2. [Model Download (Pull Command)](#model-download-pull-command)
3. [Model Registry & Management](#model-registry--management)
4. [Runtime/Backend Configuration](#runtimebackend-configuration)
5. [Hardware Configuration](#hardware-configuration)
6. [Environment Variables Reference](#environment-variables-reference)
7. [Configuration File Format](#configuration-file-format)
8. [TypeScript/JSON Schemas](#typescriptjson-schemas)
9. [Validation Rules](#validation-rules)
10. [Common Examples](#common-examples)

---

## CLI Commands Overview

Termite CLI uses Cobra framework with these primary commands:

| Command | Description | Usage |
|---------|-------------|-------|
| `termite run` | Start the Termite inference server | `termite run [flags]` |
| `termite pull` | Download models from registry or HuggingFace | `termite pull <model> [models...]` |
| `termite list` | List available models (local or remote) | `termite list [--remote] [--type <type>]` |

### Global Flags (All Commands)

| Flag | Environment Variable | Type | Default | Description |
|------|---------------------|------|---------|-------------|
| `--config` | `TERMITE_CONFIG` | string | - | Config file path |
| `--log-level` | `TERMITE_LOG_LEVEL` | string | `info` | Log level: debug, info, warn, error |
| `--log-style` | `TERMITE_LOG_STYLE` | string | `logfmt` | Output: terminal, json, noop |
| `--registry` | `TERMITE_REGISTRY` | string | `https://registry.antfly.io/v1` | Model registry URL |
| `--models-dir` | `TERMITE_MODELS_DIR` | string | `~/.termite/models` | Model storage directory |

---

## Model Download (Pull Command)

### Command Syntax

```bash
termite pull <owner/model-name> [owner/model-name...] [flags]
```

### Model Reference Formats

The pull command accepts three reference formats:

| Format | Example | Description |
|--------|---------|-------------|
| **Registry** | `BAAI/bge-small-en-v1.5` | Pull from Antfly Registry |
| **Registry + Variant** | `BAAI/bge-small-en-v1.5:i8` | Registry with inline variant |
| **HuggingFace** | `hf:BAAI/bge-small-en-v1.5` | Pull from HuggingFace Hub |

### Pull Command Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--variants` | string[] | `["f32"]` | Variant IDs to download |
| `--type` | string | auto-detect | Model type (see Model Types) |
| `--hf-token` | string | `$HF_TOKEN` | HuggingFace API token for gated models |
| `--variant` | string | - | HuggingFace ONNX variant (fp16, q4, etc.) |

### Model Variants (Quantization Options)

| Variant ID | Name | Filename | Size | Use Case |
|------------|------|----------|------|----------|
| `f32` | FP32 | `model.onnx` | Baseline | Highest accuracy (default) |
| `f16` | FP16 | `model_f16.onnx` | ~50% smaller | Balance accuracy/size |
| `bf16` | BF16 | `model_bf16.onnx` | ~50% smaller | Better range than FP16 |
| `i8` | INT8 Dynamic | `model_i8.onnx` | Smallest | Fastest CPU inference |
| `i8-st` | INT8 Static | `model_i8-st.onnx` | Small | Calibrated quantization |
| `i4` | INT4 | `model_i4.onnx` | Smallest | Extreme compression |

### Model Types

| Type ID | Description | File Structure |
|---------|-------------|----------------|
| `embedder` | Text/image embedding models | Single `model.onnx` or multimodal (CLIP) |
| `chunker` | Text chunking/segmentation | Single `model.onnx` |
| `reranker` | Document ranking/relevance | Single `model.onnx` |
| `generator` | Text generation (LLMs) | `genai_config.json` + ONNX files |
| `recognizer` | NER/entity extraction | `model.onnx` or encoder/decoder pair |
| `rewriter` | Seq2seq models | `encoder.onnx` + `decoder.onnx` |

### Model Capabilities

| Capability | Applies To | Description |
|------------|-----------|-------------|
| `multimodal` | Embedders | Can embed both images and text (CLIP-style) |
| `labels` | Recognizers | Performs entity extraction (NER) |
| `zeroshot` | Recognizers | Supports arbitrary labels at inference |
| `relations` | Recognizers | Supports relation extraction |
| `answers` | Recognizers | Supports extractive QA |

### Pull Examples

```bash
# Registry pulls
termite pull BAAI/bge-small-en-v1.5
termite pull BAAI/bge-small-en-v1.5:i8
termite pull --variants f16,i8 BAAI/bge-small-en-v1.5
termite pull sentence-transformers/all-MiniLM-L6-v2 mixedbread-ai/mxbai-rerank-base-v1

# HuggingFace pulls
termite pull hf:BAAI/bge-small-en-v1.5
termite pull hf:onnx-community/embeddinggemma-300m-ONNX --type embedder
termite pull hf:dslim/bert-base-NER --type recognizer
termite pull hf:onnxruntime/Gemma-3-ONNX

# With custom directory
termite pull --models-dir /opt/termite/models BAAI/bge-small-en-v1.5
```

---

## Model Registry & Management

### Model Storage Structure

```
~/.termite/models/
├── embedders/
│   └── <owner>/<model-name>/
│       ├── model.onnx
│       ├── model_i8.onnx
│       ├── tokenizer.json
│       └── model_manifest.json
├── chunkers/<owner>/<model-name>/
├── rerankers/<owner>/<model-name>/
├── generators/<owner>/<model-name>/
├── recognizers/<owner>/<model-name>/
└── rewriters/<owner>/<model-name>/
```

### Model Manifest Schema

```typescript
interface ModelManifest {
  schemaVersion: 1 | 2;
  name: string;                           // Model name
  source: string;                         // "owner/model" identifier
  owner: string;                          // Organization name
  type: ModelType;                        // embedder, chunker, etc.
  description?: string;
  capabilities?: string[];                // multimodal, labels, etc.
  files: ModelFile[];                     // Base model files
  variants?: Record<string, VariantEntry>; // Quantized variants
  backends?: string[];                    // Supported backends
  provenance?: ModelProvenance;           // Download metadata
}

interface ModelFile {
  name: string;                           // Filename
  digest: string;                         // SHA256 hash ("sha256:...")
  size: number;                           // Size in bytes
}

interface VariantEntry {
  files: ModelFile[];
}
```

### Registry Index Entry

```typescript
interface ModelIndexEntry {
  name: string;
  source: string;                         // "owner/model"
  owner: string;
  type: ModelType;
  description?: string;
  capabilities?: string[];
  size: number;                           // Total size in bytes
  variants?: string[];                    // Available variant IDs
}
```

---

## Runtime/Backend Configuration

### Available Backends

| Backend | Build Tags | Priority | Hardware Support | Description |
|---------|------------|----------|------------------|-------------|
| **ONNX** | `onnx,ORT` | 10 (highest) | CPU, CUDA, CoreML | Fastest inference, production recommended |
| **XLA** | `xla,XLA` | 20 | CPU, CUDA, TPU | TPU support, distributed execution |
| **Go** | (none) | 100 (lowest) | CPU only | Pure Go fallback, always available |

### Backend + Device Combinations

| Combination | Description | Use Case |
|-------------|-------------|----------|
| `onnx` | ONNX with auto device detection | General purpose |
| `onnx:cuda` | ONNX with NVIDIA GPU | GPU inference |
| `onnx:coreml` | ONNX with Apple CoreML | macOS optimization |
| `onnx:cpu` | ONNX forced CPU | Consistent performance |
| `xla` | XLA with auto detection | TPU auto-detect |
| `xla:tpu` | XLA with Google TPU | TPU inference |
| `xla:cuda` | XLA with NVIDIA GPU | GPU via XLA |
| `xla:cpu` | XLA forced CPU | CPU via XLA |
| `go` | Pure Go (CPU only) | Fallback, development |

### Device Types

| Device | Value | Description |
|--------|-------|-------------|
| Auto | `auto` | Auto-detect best available (default) |
| CUDA | `cuda` | NVIDIA CUDA GPU |
| CoreML | `coreml` | Apple CoreML (macOS only) |
| TPU | `tpu` | Google TPU |
| CPU | `cpu` | Force CPU-only |

### GPU Modes

| Mode | Value | Description |
|------|-------|-------------|
| Auto | `auto` | Auto-detect GPU availability |
| TPU | `tpu` | Force TPU usage |
| CUDA | `cuda` | Force CUDA GPU |
| CoreML | `coreml` | Force CoreML (macOS) |
| Off | `off` | CPU only, disable GPU |

### Backend Priority Configuration

```yaml
# Simple (auto device detection)
backend_priority: ["onnx", "xla", "go"]

# Explicit device preferences
backend_priority:
  - "onnx:cuda"        # Try ONNX with NVIDIA GPU first
  - "xla:tpu"          # Then XLA with TPU
  - "onnx:cpu"         # Fall back to ONNX CPU
  - "xla:cpu"          # Then XLA CPU
  - "go"               # Finally pure Go

# macOS with CoreML
backend_priority: ["onnx:coreml", "onnx:cpu", "go"]
```

### Build Commands

```bash
# Pure Go only (default, slowest)
go build -o termite ./pkg/termite/cmd

# ONNX Runtime (recommended for production)
go build -tags="onnx,ORT" -o termite ./pkg/termite/cmd

# XLA only (TPU support)
go build -tags="xla,XLA" -o termite ./pkg/termite/cmd

# All backends (maximum flexibility)
go build -tags="onnx,ORT,xla,XLA" -o termite ./pkg/termite/cmd
```

---

## Hardware Configuration

### GPU Detection Priority

**TPU Detection Order:**
1. `GOMLX_BACKEND` environment variable contains "tpu"
2. TPU libraries found: `libtpu.so` or `pjrt_plugin_tpu.so`
3. GKE TPU device files: `/dev/accel*` or `/sys/class/tpu`
4. TPU metadata endpoint (GKE nodes)

**CUDA Detection Order:**
1. `nvidia-smi` command execution
2. CUDA runtime libraries: `libcudart.so*`
3. `LD_LIBRARY_PATH` contains CUDA paths

**macOS:**
- CoreML always available on Apple Silicon/Intel Macs

### GPU Info Response

```typescript
interface GPUInfo {
  available: boolean;
  type: "cuda" | "coreml" | "tpu" | "none";
  device_name?: string;      // e.g., "NVIDIA A100 40GB"
  driver_version?: string;
  cuda_version?: string;     // CUDA compute capability
}
```

### Kubernetes Hardware Configuration (TermitePool CRD)

```typescript
interface HardwareConfig {
  // TPU accelerator type
  // Examples: "tpu-v5-lite-podslice", "tpu-v4-podslice"
  // Empty = CPU only
  accelerator?: string;

  // TPU topology (required when accelerator is set)
  // Examples: "1x1", "2x2", "2x4", "2x2x1"
  topology?: string;

  // GKE machine type
  // Examples: "n2-standard-4", "ct5lp-hightpu-4t"
  machineType?: string;

  // Enable spot/preemptible instances
  spot?: boolean;
}
```

### GKE-Specific Configuration

```typescript
interface GKEConfig {
  // Use GKE Autopilot mode
  autopilot?: boolean;

  // Compute class for Autopilot
  // Valid: "Accelerator", "Balanced", "Performance",
  //        "Scale-Out", "autopilot", "autopilot-spot"
  // Note: "Accelerator" is GPU only, NOT TPU
  autopilotComputeClass?: string;

  // Pod disruption budget configuration
  podDisruptionBudget?: PDBConfig;
}
```

### TPU Machine Types & Topologies

| Machine Type | Chips | Topology | Use Case |
|--------------|-------|----------|----------|
| `ct5lp-hightpu-1t` | 1 | `1x1` | Small models, low traffic |
| `ct5lp-hightpu-4t` | 4 | `2x2` | Medium models (default) |
| `ct5lp-hightpu-8t` | 8 | `2x4` | Large models, high throughput |

### Resource Limits for Hardware

```yaml
resources:
  limits:
    # NVIDIA GPU
    nvidia.com/gpu: "1"

    # Google Cloud GPU
    cloud.google.com/gke-gpu: "1"

    # Google TPU (chip count based on topology)
    google.com/tpu: "4"    # For 2x2 topology

    memory: "16Gi"
    cpu: "4"
```

### Kubernetes Node Selectors

```yaml
# TPU workloads
nodeSelector:
  cloud.google.com/gke-tpu-accelerator: "tpu-v5-lite-podslice"
  cloud.google.com/gke-tpu-topology: "2x2"

# Spot instances
nodeSelector:
  cloud.google.com/gke-spot: "true"
```

### Kubernetes Tolerations

```yaml
# TPU nodes
tolerations:
  - key: "google.com/tpu"
    operator: "Exists"
    effect: "NoSchedule"

# Spot nodes
tolerations:
  - key: "cloud.google.com/gke-spot"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
```

---

## Environment Variables Reference

### Core Termite Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TERMITE_CONFIG` | - | Config file path |
| `TERMITE_API_URL` | `http://localhost:11433` | API endpoint |
| `TERMITE_REGISTRY` | `https://registry.antfly.io/v1` | Model registry URL |
| `TERMITE_MODELS_DIR` | `~/.termite/models` | Model storage directory |
| `TERMITE_LOG_LEVEL` | `info` | Log level |
| `TERMITE_LOG_STYLE` | `logfmt` | Log format |
| `TERMITE_HEALTH_PORT` | `4200` | Health/metrics port |
| `TERMITE_BACKEND_PRIORITY` | `onnx,xla,go` | Comma-separated backend list |
| `TERMITE_KEEP_ALIVE` | `5m` | Model TTL (time-to-live) |
| `TERMITE_MAX_LOADED_MODELS` | `10` | Max concurrent models |
| `TERMITE_MAX_MEMORY_MB` | - | Max memory budget |
| `TERMITE_PRELOAD` | - | Comma-separated models to preload |
| `TERMITE_GPU_MODE` | `auto` | GPU mode: auto, cuda, tpu, coreml, off |

### ONNX Runtime Variables

| Variable | Platform | Description |
|----------|----------|-------------|
| `ONNXRUNTIME_ROOT` | All | Directory containing libonnxruntime |
| `LD_LIBRARY_PATH` | Linux | Library search path |
| `DYLD_LIBRARY_PATH` | macOS | Library search path |
| `ORTGENAI_DYLIB_PATH` | All | Path to libonnxruntime-genai |
| `TERMITE_FORCE_COREML` | macOS | Set to `1` to force CoreML (experimental) |
| `TERMITE_DEBUG` | macOS | Enable CoreML compute plan profiling |

### XLA/GoMLX Variables

| Variable | Description |
|----------|-------------|
| `GOMLX_BACKEND` | Override device: `xla:tpu`, `xla:cuda`, `xla:cpu` |
| `PJRT_PLUGIN_LIBRARY_PATH` | Custom PJRT plugin location |
| `XLA_FLAGS` | XLA compilation flags |

### HuggingFace Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token for gated models |

---

## Configuration File Format

### File Locations (Search Order)

1. Explicit: `--config /path/to/termite.yaml`
2. Home: `~/.termite.yaml` or `~/.termite.yml`
3. Current: `./termite.yaml` or `./termite.yml`

### Complete Configuration Schema

```yaml
# API endpoint
api_url: "http://localhost:11433"

# Model storage directory
models_dir: "~/.termite/models"

# Backend priority (format: "backend" or "backend:device")
backend_priority:
  - "onnx:cuda"
  - "xla:tpu"
  - "onnx:cpu"
  - "go"

# Model loading behavior
keep_alive: "5m"           # Model TTL, "0" for eager loading
max_loaded_models: 10      # Max concurrent models
max_memory_mb: 16384       # Max memory usage (MB)

# Models to preload at startup
preload:
  - "BAAI/bge-small-en-v1.5"
  - "mixedbread-ai/mxbai-rerank-base-v1"

# Per-model loading strategies
model_strategies:
  "BAAI/bge-small-en-v1.5": "eager"      # Never unload
  "openai/clip-vit-base-patch32": "lazy" # Unload when idle

# Logging configuration
log:
  level: "info"            # debug, info, warn, error
  style: "logfmt"          # terminal, json, noop
```

### Loading Strategies

| Strategy | Value | Description |
|----------|-------|-------------|
| Lazy | `lazy` | Load on first request, unload after `keep_alive` |
| Eager | `eager` | Load at startup, never unload |

---

## TypeScript/JSON Schemas

### Pull Request Schema

```typescript
interface PullRequest {
  models: ModelReference[];
  options?: PullOptions;
}

interface ModelReference {
  // Full model reference
  // Examples: "BAAI/bge-small-en-v1.5", "hf:sentence-transformers/all-MiniLM-L6-v2"
  ref: string;

  // Inline variant (parsed from ref if colon-separated)
  variant?: string;
}

interface PullOptions {
  // Registry URL (default: https://registry.antfly.io/v1)
  registry?: string;

  // Model storage directory
  modelsDir?: string;

  // Variants to download
  variants?: VariantID[];

  // Model type (for HuggingFace when auto-detect fails)
  type?: ModelType;

  // HuggingFace API token
  hfToken?: string;

  // HuggingFace ONNX variant
  hfVariant?: string;
}

type VariantID = "f32" | "f16" | "bf16" | "i8" | "i8-st" | "i4";
type ModelType = "embedder" | "chunker" | "reranker" | "generator" | "recognizer" | "rewriter";
```

### Runtime Configuration Schema

```typescript
interface RuntimeConfig {
  // Backend priority with optional device
  backendPriority: BackendSpec[];

  // Model loading behavior
  keepAlive?: string;           // Duration string, e.g., "5m"
  maxLoadedModels?: number;
  maxMemoryMb?: number;

  // Models to preload
  preload?: string[];

  // Per-model strategies
  modelStrategies?: Record<string, LoadingStrategy>;
}

interface BackendSpec {
  backend: BackendType;
  device?: DeviceType;
}

type BackendType = "onnx" | "xla" | "go";
type DeviceType = "auto" | "cuda" | "coreml" | "tpu" | "cpu";
type LoadingStrategy = "lazy" | "eager";
```

### Hardware Configuration Schema

```typescript
interface HardwareConfig {
  // TPU configuration
  accelerator?: TPUAccelerator;
  topology?: TPUTopology;

  // Machine configuration
  machineType?: string;
  spot?: boolean;

  // GPU mode
  gpuMode?: GPUMode;
}

type TPUAccelerator = "tpu-v5-lite-podslice" | "tpu-v4-podslice" | string;
type TPUTopology = "1x1" | "2x2" | "2x4" | "2x2x1" | string;
type GPUMode = "auto" | "cuda" | "coreml" | "tpu" | "off";
```

### Full Config Schema

```typescript
interface TermiteConfig {
  // API configuration
  apiUrl?: string;

  // Registry configuration
  registry?: string;
  modelsDir?: string;

  // Runtime configuration
  backendPriority?: string[];    // "backend" or "backend:device"
  keepAlive?: string;
  maxLoadedModels?: number;
  maxMemoryMb?: number;
  preload?: string[];
  modelStrategies?: Record<string, string>;

  // Logging configuration
  log?: {
    level?: "debug" | "info" | "warn" | "error";
    style?: "terminal" | "json" | "noop";
  };

  // Server configuration
  healthPort?: number;
}
```

---

## Validation Rules

### Model Reference Validation

- Must match format: `[hf:]owner/model-name[:variant]`
- Owner and model name: alphanumeric, hyphens, underscores
- HuggingFace prefix: `hf:` (optional)
- Variant suffix: valid variant ID after colon

### Backend Priority Validation

- Must be array of valid backend specs
- Format: `"backend"` or `"backend:device"`
- Valid backends: `onnx`, `xla`, `go`
- Valid devices: `auto`, `cuda`, `coreml`, `tpu`, `cpu`
- `coreml` only valid on macOS
- `tpu` only valid with `xla` backend

### Hardware Configuration Validation (Kubernetes)

1. **TPU Configuration:**
   - If `accelerator` is set, `topology` is required
   - `topology` must match accelerator (e.g., 2x2 requires 4-chip accelerator)

2. **Autopilot + Spot Conflict:**
   - `hardware.spot=true` conflicts with `gke.autopilot=true`
   - Use `autopilotComputeClass: autopilot-spot` instead

3. **Accelerator Compute Class:**
   - `Accelerator` class requires GPU resources
   - Do NOT use for TPU workloads

4. **Immutable Fields:**
   - `gke.autopilot` and `gke.autopilotComputeClass` are immutable after deployment

---

## Common Examples

### Dashboard: Download Model Form

```typescript
// Form fields for model download
interface DownloadModelForm {
  // Model reference (required)
  modelRef: string;           // e.g., "BAAI/bge-small-en-v1.5"

  // Source selection
  source: "registry" | "huggingface";

  // Variants to download (multi-select)
  variants: VariantID[];      // ["f32", "i8"]

  // Model type (required for some HuggingFace models)
  type?: ModelType;

  // HuggingFace token (for gated models)
  hfToken?: string;
}

// Convert to CLI command
function toCliCommand(form: DownloadModelForm): string {
  const parts = ["termite", "pull"];

  if (form.source === "huggingface") {
    parts.push(`hf:${form.modelRef}`);
  } else {
    parts.push(form.modelRef);
  }

  if (form.variants.length > 0 && form.source === "registry") {
    parts.push(`--variants ${form.variants.join(",")}`);
  }

  if (form.type) {
    parts.push(`--type ${form.type}`);
  }

  if (form.hfToken) {
    parts.push(`--hf-token ${form.hfToken}`);
  }

  return parts.join(" ");
}
```

### Dashboard: Runtime Configuration Form

```typescript
interface RuntimeConfigForm {
  // Backend priority (drag-and-drop list)
  backends: Array<{
    backend: BackendType;
    device: DeviceType;
    enabled: boolean;
  }>;

  // Model loading settings
  keepAlive: string;          // "5m", "10m", "0" (eager)
  maxLoadedModels: number;
  maxMemoryMb?: number;

  // Preload models (multi-select from installed models)
  preload: string[];
}

// Convert to config object
function toConfig(form: RuntimeConfigForm): Partial<TermiteConfig> {
  return {
    backendPriority: form.backends
      .filter(b => b.enabled)
      .map(b => b.device === "auto" ? b.backend : `${b.backend}:${b.device}`),
    keepAlive: form.keepAlive,
    maxLoadedModels: form.maxLoadedModels,
    maxMemoryMb: form.maxMemoryMb,
    preload: form.preload,
  };
}
```

### Dashboard: Hardware Configuration Form

```typescript
interface HardwareConfigForm {
  // Compute type selection
  computeType: "cpu" | "gpu" | "tpu";

  // GPU options
  gpuMode?: GPUMode;

  // TPU options
  tpuAccelerator?: string;
  tpuTopology?: string;

  // Instance options
  useSpot?: boolean;
  machineType?: string;
}

// UI helpers
const TPU_OPTIONS = [
  { accelerator: "tpu-v5-lite-podslice", topologies: ["1x1", "2x2", "2x4"] },
  { accelerator: "tpu-v4-podslice", topologies: ["2x2", "2x2x1", "2x2x2"] },
];

const GPU_MODES = [
  { value: "auto", label: "Auto-detect" },
  { value: "cuda", label: "NVIDIA CUDA" },
  { value: "coreml", label: "Apple CoreML (macOS only)" },
  { value: "off", label: "Disabled (CPU only)" },
];

const BACKEND_OPTIONS = [
  { value: "onnx", label: "ONNX Runtime", description: "Fastest CPU/GPU inference" },
  { value: "xla", label: "XLA (GoMLX)", description: "TPU and distributed support" },
  { value: "go", label: "Pure Go", description: "Fallback, always available" },
];

const DEVICE_OPTIONS = [
  { value: "auto", label: "Auto-detect" },
  { value: "cuda", label: "NVIDIA CUDA GPU" },
  { value: "coreml", label: "Apple CoreML" },
  { value: "tpu", label: "Google TPU" },
  { value: "cpu", label: "CPU Only" },
];
```

### Example: Complete Config Generation

```typescript
function generateTermiteConfig(
  runtime: RuntimeConfigForm,
  hardware: HardwareConfigForm
): TermiteConfig {
  const config: TermiteConfig = {
    backendPriority: [],
    keepAlive: runtime.keepAlive,
    maxLoadedModels: runtime.maxLoadedModels,
    preload: runtime.preload,
  };

  // Build backend priority based on hardware
  if (hardware.computeType === "tpu") {
    config.backendPriority = ["xla:tpu", "xla:cpu", "go"];
  } else if (hardware.computeType === "gpu") {
    config.backendPriority = [
      `onnx:${hardware.gpuMode || "cuda"}`,
      "onnx:cpu",
      "go"
    ];
  } else {
    config.backendPriority = ["onnx:cpu", "xla:cpu", "go"];
  }

  // Override with user's custom priority if set
  if (runtime.backends.length > 0) {
    config.backendPriority = runtime.backends
      .filter(b => b.enabled)
      .map(b => b.device === "auto" ? b.backend : `${b.backend}:${b.device}`);
  }

  return config;
}
```

---

## API Endpoints (for reference)

The Termite server exposes these relevant endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List loaded models |
| GET | `/api/config` | Get current configuration |
| GET | `/api/backends` | List available backends |
| GET | `/api/gpu` | Get GPU info |
| POST | `/api/embed` | Generate embeddings |
| POST | `/api/rerank` | Rerank documents |
| POST | `/api/chunk` | Chunk text |

---

## Key Source Files

| File | Description |
|------|-------------|
| `pkg/termite/cmd/cmd/pull.go` | Pull command implementation |
| `pkg/termite/cmd/cmd/list.go` | List command implementation |
| `pkg/termite/cmd/cmd/run.go` | Run command implementation |
| `pkg/termite/lib/cli/pull.go` | Pull logic (registry + HuggingFace) |
| `lib/modelregistry/manifest.go` | Model manifest schema |
| `lib/modelregistry/client.go` | Registry client |
| `lib/modelregistry/huggingface.go` | HuggingFace client |
| `lib/hugot/backend.go` | Backend interface and types |
| `lib/hugot/backend_onnx.go` | ONNX backend |
| `lib/hugot/backend_xla.go` | XLA backend |
| `lib/hugot/gpu.go` | GPU detection |
| `lib/hugot/session_manager.go` | Session management |
| `pkg/operator/api/v1alpha1/termitepool_types.go` | K8s CRD types |
| `pkg/termite/api.gen.go` | API and config types |
