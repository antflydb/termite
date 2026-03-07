# Plan: Traditional ML Model Inference in Termite

## Executive Summary

Timber demonstrates a compelling approach: **ahead-of-time compilation** of traditional ML models (XGBoost, LightGBM, sklearn, CatBoost) into native code with zero dependencies. Their key innovations are a framework-agnostic IR, multi-pass optimization, and C99 codegen.

We can do this better. Instead of compiling to C and loading via ctypes (fragile, Python-centric), we'll:

1. **Adopt Timber's IR concept** — define a Go-native IR for tree ensembles, linear models, and preprocessing pipelines
2. **Build a Python conversion CLI** (reuse/adapt Timber's frontends) that exports models to our IR format
3. **Implement a SIMD-accelerated Go inference engine** using go-highway — no CGO, no ONNX Runtime, pure Go with hardware-accelerated tree traversal
4. **Integrate as a first-class Termite model type** with registry, lazy loading, API endpoints, and operator support

This gives us microsecond-latency inference for traditional ML in the same service that already handles embeddings, reranking, and NER — no separate process, no Python, no C compiler needed at runtime.

---

## Part 1: Why Not ONNX for Traditional ML?

ONNX *can* represent tree ensembles (via ML opset), but:
- **ONNX Runtime tree traversal is generic** — no model-specific optimizations (dead leaf elimination, branch sorting)
- **80-150us latency** per Timber's benchmarks vs **~2us** for compiled trees
- **Overhead**: session creation, tensor allocation, memory copies for what's essentially array lookups
- **No preprocessing fusion** — scalers/encoders are separate ONNX graphs

A Go-native engine with go-highway SIMD can match or beat Timber's C99 performance while staying in-process with zero FFI overhead.

---

## Part 2: Architecture Overview

```
                    +-------------------------------------+
                    |         Model Conversion             |
                    |  (Python CLI -- offline, one-time)    |
                    |                                       |
                    |  XGBoost --+                          |
                    |  LightGBM -+-> Parser -> IR -> Optim  |
                    |  sklearn  -+        |                 |
                    |  CatBoost -+   tabular_model.json     |
                    +------------------+------------------+
                                       | upload to registry
                                       v
+--------------------------------------------------------------+
|                        Termite                                |
|                                                               |
|  +--------------+    +-----------------+    +--------------+  |
|  | Model Registry|--->| PredictorRegistry|--->| Go-Highway   | |
|  | (discovery +  |    | (lazy loading,   |    | Inference    | |
|  |  manifest)    |    |  TTL cache,      |    | Engine       | |
|  |              |    |  ref counting)   |    |              | |
|  +--------------+    +-----------------+    | TreeEngine   | |
|                                              | LinearEngine | |
|  POST /api/predict ---------------------->  | SVMEngine    | |
|  POST /api/predict/batch ---------------->  | Pipeline     | |
|  GET  /api/models ----------------------->  +--------------+ |
+--------------------------------------------------------------+
```

---

## Part 3: IR Design (Go-native)

Inspired by Timber's `TimberIR`, but designed for Go serialization and go-highway consumption.

### 3.1 File Format: `tabular_model.json`

```jsonc
{
  "schema_version": 1,
  "metadata": {
    "name": "fraud-detector-v2",
    "source_framework": "xgboost",
    "source_version": "2.0.3",
    "task": "binary_classification",  // regression | binary_classification | multiclass | ranking
    "num_features": 30,
    "num_classes": 2,
    "feature_names": ["amount", "merchant_category", ...],
    "feature_types": ["float32", "int32", ...],
    "created_at": "2026-03-06T..."
  },
  "pipeline": [
    // Preprocessing stages (executed in order)
    {
      "type": "scaler",
      "method": "standard",        // standard | minmax | robust | maxabs
      "mean": [0.5, 1.2, ...],
      "scale": [0.1, 0.3, ...]
    },
    {
      "type": "imputer",
      "strategy": "median",        // mean | median | most_frequent | constant
      "fill_values": [0.5, 1.0, ...]
    },
    // The model stage
    {
      "type": "tree_ensemble",
      "objective": "binary_logistic",  // binary_logistic | squared_error | softmax | lambdarank
      "base_score": 0.0,
      "num_trees": 200,
      "num_features": 30,
      "max_depth": 8,
      // Flat array layout for cache-friendly traversal
      "nodes": {
        "feature_index": [2, 5, -1, 0, ...],   // -1 = leaf
        "threshold":     [0.5, 1.2, 0, 3.7, ...],
        "left_child":    [1, 3, -1, 7, ...],    // -1 = none
        "right_child":   [2, 4, -1, 8, ...],
        "leaf_value":    [0, 0, 0.23, 0, ...],  // non-zero only for leaves
        "default_left":  [true, false, ...],     // NaN handling
        "tree_starts":   [0, 15, 31, ...]       // index of each tree's root
      },
      // Optimization annotations (from converter)
      "annotations": {
        "threshold_precision": ["f32", "i8", "f16", ...],  // per-feature
        "dead_leaves_eliminated": 47,
        "branch_order": "frequency_sorted"
      }
    }
  ],
  // Post-processing
  "output": {
    "activation": "sigmoid",  // sigmoid | softmax | identity | exp
    "num_outputs": 1
  }
}
```

### 3.2 Go Types

```
lib/tabular/
  ir.go              # IR types: TabularModel, Stage, TreeEnsemble, LinearModel, etc.
  ir_test.go         # Serialization round-trip tests
  loader.go          # Load from tabular_model.json
  loader_test.go
```

Key types:
- `TabularModel` -- top-level container with metadata, pipeline stages, output config
- `TreeEnsemble` -- flat-array tree representation optimized for go-highway vectorized traversal
- `LinearModel` -- weights/biases with activation
- `SVMModel` -- support vectors, dual coefficients, kernel params
- `ScalerStage`, `ImputerStage`, `EncoderStage` -- preprocessing

The flat-array layout (parallel arrays of `feature_index`, `threshold`, `left_child`, `right_child`, `leaf_value`) is critical -- it enables SIMD comparison of multiple trees' thresholds simultaneously and is cache-line friendly.

---

## Part 4: Python Conversion CLI

### 4.1 `termite-convert` Tool

A lightweight Python package (separate repo or in `tools/termite-convert/`) that converts framework models to our IR:

```bash
# Install
pip install termite-convert

# Convert
termite-convert xgboost model.json -o ./my-model/
termite-convert lightgbm model.txt -o ./my-model/
termite-convert sklearn model.pkl -o ./my-model/
termite-convert catboost model.cbm -o ./my-model/
termite-convert onnx model.onnx -o ./my-model/  # ML opset only

# With optimization
termite-convert xgboost model.json -o ./my-model/ \
  --optimize \
  --dead-leaf-threshold 0.001 \
  --calibration-data train.csv  # for branch sorting
```

Output structure:
```
my-model/
  tabular_model.json          # The IR
  model_manifest.json         # Termite manifest (type: "predictor")
  config.json                 # Optional metadata
```

### 4.2 Conversion Pipeline

Directly inspired by Timber's architecture but simplified:

1. **Parse** -- framework-specific parsers (can adapt Timber's frontends under Apache 2.0)
2. **Normalize** -- convert to our IR format
3. **Optimize** -- run optimization passes:
   - Dead leaf elimination (prune negligible leaves)
   - Threshold quantization annotation (identify int8/f16-safe thresholds)
   - Branch sorting by frequency (requires calibration data)
   - Constant feature detection and elimination
4. **Emit** -- write `tabular_model.json` + `model_manifest.json`

### 4.3 Registry Integration

Converted models can be:
- Placed directly in `~/.termite/models/predictors/<name>/`
- Uploaded to the Antfly model registry (`registry.antfly.io`)
- Pulled via `termite pull <model-ref>` just like ONNX models

---

## Part 5: Go-Highway Inference Engine

This is the core innovation -- a pure Go inference engine that uses go-highway SIMD for tree traversal.

### 5.1 File Structure

```
lib/tabular/
  ir.go                  # IR types
  loader.go              # JSON loading
  engine.go              # Predictor interface + dispatch
  tree_engine.go         # Tree ensemble inference (go-highway SIMD)
  tree_engine_test.go    # Correctness tests vs reference implementations
  linear_engine.go       # Linear model inference
  svm_engine.go          # SVM inference (RBF/linear/poly kernels)
  preprocess.go          # Scaler, imputer, encoder execution
  preprocess_test.go
  optimizer/
    dead_leaf.go       # Dead leaf elimination
    quantize.go        # Threshold quantization
    branch_sort.go     # Branch reordering
    optimizer.go       # Pipeline orchestrator
  benchmark_test.go      # Performance benchmarks
```

### 5.2 Tree Traversal with go-highway

The key performance insight from Timber: tree ensembles are embarrassingly parallelizable across trees. Each tree is independent. With SIMD, we can traverse multiple trees simultaneously:

```go
// Conceptual approach -- traverse N trees in parallel using SIMD
//
// For each depth level:
//   1. Gather feature values for current nodes across N trees (SIMD load)
//   2. Gather thresholds for current nodes across N trees (SIMD load)
//   3. Compare features vs thresholds (SIMD compare) -> N branch decisions
//   4. Select left_child or right_child based on comparison (SIMD blend)
//   5. Repeat until all trees reach leaves
//   6. Sum leaf values (SIMD horizontal add)

func (e *TreeEngine) PredictBatch(features [][]float32) []float32 {
    // For each sample:
    //   - Process trees in SIMD-width chunks (8 trees at a time on AVX2)
    //   - Accumulate leaf values
    //   - Apply output activation (sigmoid/softmax)
}
```

go-highway provides the primitives we need:
- `vec.Load` / `vec.Store` -- gather feature/threshold values
- `algo.Compare` -- vectorized threshold comparison
- `nn.Sigmoid`, `nn.Softmax` -- output activations
- `vec.Sum` -- horizontal reduction for leaf accumulation

### 5.3 Predictor Interface

```go
// lib/tabular/engine.go

type Predictor interface {
    // Predict runs inference on a batch of samples.
    // features: [batch_size][num_features]float32
    // Returns: [batch_size][num_outputs]float32
    Predict(ctx context.Context, features [][]float32) ([][]float32, error)

    // PredictSingle runs inference on a single sample (optimized path).
    PredictSingle(features []float32) ([]float32, error)

    // NumFeatures returns the expected input dimension.
    NumFeatures() int

    // NumOutputs returns the output dimension.
    NumOutputs() int

    // Metadata returns model metadata (feature names, task type, etc.)
    Metadata() *ModelMetadata

    Close() error
}

// LoadPredictor loads a tabular model from a directory.
func LoadPredictor(modelDir string) (Predictor, error) {
    // 1. Read tabular_model.json
    // 2. Build preprocessing pipeline
    // 3. Initialize engine (tree/linear/SVM) with go-highway
    // 4. Return composite Predictor
}
```

### 5.4 Performance Targets

Based on Timber's benchmarks and go-highway's SIMD capabilities:

| Metric | Target | Timber (C99) | ONNX Runtime |
|--------|--------|-------------|--------------|
| Single sample latency | <5us | ~2us | ~80-150us |
| Throughput (50 trees, 30 features) | >200k samples/sec | 500k/sec | ~10k/sec |
| Memory per model | <10MB typical | Similar | Higher (runtime overhead) |

Even if we're 2-3x slower than compiled C (Go function call overhead, GC), we'll still be 20-50x faster than ONNX Runtime for tree models.

---

## Part 6: Termite Integration

### 6.1 New Model Type and Capability

In `lib/modelregistry/manifest.go`:
```go
const ModelTypePredictor ModelType = "predictor"

// New capabilities for tabular models
const (
    CapabilityTreeEnsemble Capability = "tree_ensemble"
    CapabilityLinear       Capability = "linear"
    CapabilitySVM          Capability = "svm"
    CapabilityTabular      Capability = "tabular"  // generic
)
```

### 6.2 PredictorRegistry

Following the exact pattern of `EmbedderRegistry` and `NERRegistry`:

```
pkg/termite/
  predictor_registry.go       # PredictorRegistry (lazy load, TTL cache, ref counting)
  registry_interfaces.go      # + PredictorRegistryInterface
```

```go
type PredictorRegistryInterface interface {
    Acquire(ctx context.Context, name string) (tabular.Predictor, error)
    Release(name string)
    List() []PredictorModelInfo
}
```

Model directory: `~/.termite/models/predictors/<owner>/<name>/`

### 6.3 API Endpoint

New endpoint in `openapi.yaml`:

```yaml
/api/predict:
  post:
    summary: Run tabular prediction
    requestBody:
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/PredictRequest'
    responses:
      '200':
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/PredictResponse'

components:
  schemas:
    PredictRequest:
      type: object
      required: [model, inputs]
      properties:
        model:
          type: string
          description: Model name (e.g., "fraud-detector-v2")
        inputs:
          type: array
          items:
            type: array
            items:
              type: number
          description: "Batch of feature vectors [[f1, f2, ...], ...]"
        feature_names:
          type: array
          items:
            type: string
          description: "Optional: named features as map keys instead of positional array"

    PredictResponse:
      type: object
      properties:
        model:
          type: string
        predictions:
          type: array
          items:
            type: array
            items:
              type: number
          description: "[batch_size][num_outputs] predictions"
        labels:
          type: array
          items:
            type: string
          description: "Class labels for classification tasks"
        task:
          type: string
          enum: [regression, binary_classification, multiclass, ranking]
```

### 6.4 Handler

In `pkg/termite/termite.go`:
```go
func (n *TermiteNode) handleApiPredict(ctx context.Context, req PredictRequest) (*PredictResponse, error) {
    predictor, err := n.predictors.Acquire(ctx, req.Model)
    if err != nil { return nil, err }
    defer n.predictors.Release(req.Model)

    results, err := predictor.Predict(ctx, req.Inputs)
    if err != nil { return nil, err }

    return &PredictResponse{
        Model:       req.Model,
        Predictions: results,
        Task:        predictor.Metadata().Task,
    }, nil
}
```

### 6.5 Caching

Reuse the existing two-tier caching pattern (memory + singleflight). For tabular predictions, cache keys would be `hash(model_name + input_features)`. This is particularly effective for lookup-heavy workloads (e.g., recommendation scoring where the same user features are scored against multiple items).

---

## Part 7: Optimization Passes (Go-native)

Implement key Timber optimizations in Go, applied either at conversion time (Python) or load time (Go):

### 7.1 Dead Leaf Elimination
Prune leaves whose `|value| < threshold * max_leaf_value`. Collapse subtrees where both children are pruned. This reduces tree size by 10-30% typically.

### 7.2 Threshold Quantization
Annotate thresholds with minimum required precision (int8, float16, float32). At runtime, use narrower SIMD lanes where possible (2x throughput with int16 vs float32 on same vector width).

### 7.3 Branch Sorting
Given calibration data, reorder tree children so the more likely branch is the "fall-through" path. Improves branch predictor hit rate. Requires a calibration dataset at conversion time.

### 7.4 Tree Pruning / Fusion (Future)
- Merge trees with identical structure
- Fuse consecutive preprocessing stages
- Constant-fold features that never vary

---

## Part 8: Implementation Phases

### Phase 1: Foundation (Week 1-2)
1. Define IR types in `lib/tabular/ir.go`
2. Implement JSON loader in `lib/tabular/loader.go`
3. Build basic tree engine without SIMD in `lib/tabular/tree_engine.go`
4. Add linear model engine
5. Add preprocessing stages (scaler, imputer)
6. Unit tests with hand-crafted models
7. Write `termite-convert` Python CLI -- XGBoost parser first

### Phase 2: SIMD Acceleration (Week 2-3)
1. Implement go-highway vectorized tree traversal
2. Benchmark against naive Go implementation and ONNX Runtime
3. Optimize memory layout for cache efficiency
4. Add batch processing path

### Phase 3: Termite Integration (Week 3-4)
1. Add `ModelTypePredictor` and capabilities to manifest system
2. Implement `PredictorRegistry` (lazy loading, TTL, ref counting)
3. Add `/api/predict` endpoint to OpenAPI spec
4. Run `oapi-codegen` for server + client
5. Implement `handleApiPredict` handler
6. Add predictor model discovery to `TermiteNode`
7. Wire into CLI (`termite run`, `termite pull`, `termite list`)

### Phase 4: Converters and Optimization (Week 4-5)
1. Add LightGBM, sklearn, CatBoost parsers to `termite-convert`
2. Implement dead leaf elimination pass
3. Implement threshold quantization annotation
4. Add ONNX ML opset parser (tree ensembles, linear, SVM)
5. Add branch sorting (requires calibration data CLI flag)

### Phase 5: Production Hardening (Week 5-6)
1. E2E tests with real models (download XGBoost/LightGBM models, convert, serve, validate)
2. Registry integration (upload converted models to `registry.antfly.io`)
3. Operator support (`TermitePool` for predictor workloads)
4. Dashboard updates (show predictor models, metrics)
5. Documentation and examples

---

## Part 9: Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Runtime** | Pure Go + go-highway | No CGO, no external deps, works everywhere Termite works. Go-highway provides AVX2/NEON/fallback automatically. |
| **IR format** | JSON | Human-readable, debuggable, versionable. Models are small (few MB) so parsing overhead is negligible vs inference. |
| **Conversion** | Separate Python tool | Framework SDKs (xgboost, sklearn) are Python-only. Conversion is offline/one-time. Avoids Python dependency in Termite. |
| **Not using Timber's runtime** | Correct | Timber's ctypes+gcc approach is fragile. We get the same perf from go-highway SIMD without the complexity. |
| **Not using ONNX for trees** | Correct | 40-75x slower. ONNX is great for neural nets, poor for tree ensembles. |
| **Model type** | "predictor" (not "classifier"/"regressor") | Covers all tabular tasks. Consistent with Timber's unified approach. |
| **Optimization at convert time** | Yes, with load-time fallback | Most optimizations only need to run once. Store optimized IR. |
| **SVM support** | Phase 4 | Lower priority than trees. Include in IR from day 1 for forward compat. |

---

## Part 10: Risk and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| go-highway tree traversal slower than C99 | 2-5x slower than Timber claims | Still 20-50x faster than ONNX. Benchmark early (Phase 2). Can add CGO C backend behind build tag if needed. |
| XGBoost binary format not supported | Some users can't export JSON | XGBoost `save_model("model.json")` is well-documented. Add binary parser later. |
| sklearn pickle security | Arbitrary code execution | Document risk clearly. Recommend ONNX export path for sklearn. |
| NaN handling edge cases | Wrong predictions | Timber's `default_left` approach is proven. Test exhaustively with NaN inputs. |
| Calibration data format | UX friction for branch sorting | Make it optional. CSV with header row. Auto-detect feature columns. |

---

## Appendix: Comparison with Timber

| Aspect | Timber | Termite (proposed) |
|--------|--------|-------------------|
| **Language** | Python + C99 codegen | Go + go-highway SIMD |
| **Runtime deps** | gcc/clang (compile step) | None (pure Go) |
| **Deployment** | Standalone binary or ctypes | Part of existing Termite service |
| **Model management** | `~/.timber/registry.json` | Antfly model registry (SHA256-verified) |
| **Scaling** | Single process | K8s operator (TermitePool) |
| **API compat** | Ollama-style | Termite API + Ollama compat |
| **Multimodal** | No | Yes (same service handles embeddings, NER, etc.) |
| **Differential privacy** | Yes (Laplace/Gaussian noise) | Not planned (niche feature) |
| **MISRA-C / WASM** | Yes | Not needed (not targeting embedded) |
| **Preprocessing** | Full pipeline in C | Full pipeline in Go |
| **Caching** | None | Two-tier (memory + singleflight) |
