# CoreML Shared Weight Storage & Parallel Pre-compilation

## Problem

CoreML compilation per unique input shape takes ~11.3s total:
| Step | Time | % |
|------|------|---|
| `model.ToModel()` | ~2µs | 0% |
| `model.SaveMLPackageWithBlobs()` | ~1.73s | 15% |
| `bridge.CompileModel()` | ~214ms | 2% |
| `bridge.LoadModel()` | ~9.34s | 83% |

`SaveMLPackageWithBlobs` writes ~1.3GB of weight.bin data for EVERY shape, even though the weights are identical across shapes (same model, different input dimensions). For 9 bucket shapes, that's 9 × 1.3GB = ~11.7GB of redundant writes.

`bridge.LoadModel()` calls Apple's `MLModel(contentsOf:configuration:)` — ANE kernel JIT, memory alloc. Unavoidable per-process per-shape.

With 9 shapes compiled lazily and sequentially, cold start is **~102s**.

## Design: Shared weight.bin

The `extractTensorsToBlob` function walks the MIL program deterministically (functions → blocks → operations → inputs). For the same model with different input shapes, the weight constants appear in the same order with the same data. Therefore **weight.bin is byte-identical across shapes**.

We can write weight.bin once and symlink it into subsequent .mlpackage directories. This drops `SaveMLPackageWithBlobs` from ~1.73s to ~0s for shapes 2-9.

## Changes

### 1. Add shared weights support to `model.SaveMLPackageWithBlobs` (go-coreml)

**File:** `go-coreml/model/serialize_blob.go`

Add `SharedWeightsPath` option:

```go
type BlobSerializeOptions struct {
    SerializeOptions
    UseBlobStorage    bool
    BlobThreshold     int64
    SharedWeightsPath string // If set, symlink to this weight.bin instead of writing new one
}
```

**New function `SaveMLPackageWithSharedWeights`** (or modify existing):
- When `SharedWeightsPath` is set:
  1. Create directory structure as normal
  2. Run `extractTensorsToBlob` with a **null blob writer** that computes offsets but doesn't write data. This mutates the protobuf to use `BlobFileValue` references with correct offsets.
  3. Symlink `weights/weight.bin → SharedWeightsPath`
  4. Write model.mlmodel + Manifest.json as normal

**New `blob.NewNullWriter`:**

```go
// NewNullWriter creates a writer that computes offsets without writing to disk.
// Used when sharing weight.bin across multiple compilations.
func NewNullWriter() *Writer {
    return &Writer{
        offset:  DefaultAlignment,
        entries: nil,
        // file is nil — AddBlob tracks offsets but doesn't write
    }
}
```

Update `AddBlob` and `Close` to handle nil file (skip I/O, return offsets).

**File:** `go-coreml/blob/writer.go`

### 2. Add shared weight caching to `runtime.Runtime` (go-coreml)

**File:** `go-coreml/runtime/runtime.go`

```go
type Runtime struct {
    cacheDir     string
    computeUnits bridge.ComputeUnits

    // Shared weight storage — first compilation writes weight.bin,
    // subsequent compilations symlink to it.
    mu               sync.Mutex
    sharedWeightsPath string // Path to reusable weight.bin
    sharedWeightsDir  string // Temp dir holding the shared weights (cleaned up on Close)
}
```

Add `Close()` method to Runtime for cleanup:
```go
func (r *Runtime) Close() {
    if r.sharedWeightsDir != "" {
        os.RemoveAll(r.sharedWeightsDir)
    }
}
```

**Modified `CompileProgram` flow:**

```go
func (r *Runtime) CompileProgram(program, inputs, outputs) (*Executable, error) {
    tempDir := os.MkdirTemp(r.cacheDir, "gocoreml-")

    coremlModel := model.ToModel(program, inputs, outputs, model.DefaultOptions())

    packagePath := filepath.Join(tempDir, "model.mlpackage")
    blobOpts := model.DefaultBlobOptions()

    // Check if we can reuse shared weights
    r.mu.Lock()
    sharedPath := r.sharedWeightsPath
    r.mu.Unlock()

    if sharedPath != "" {
        blobOpts.SharedWeightsPath = sharedPath
    }

    model.SaveMLPackageWithBlobs(coremlModel, packagePath, blobOpts)

    // First compilation: save weight.bin path for reuse
    if sharedPath == "" {
        weightsPath := filepath.Join(packagePath, "Data", "com.apple.CoreML", "weights", "weight.bin")

        // Move weight.bin to a dedicated shared dir (survives individual Executable.Close())
        sharedDir := os.MkdirTemp(r.cacheDir, "gocoreml-weights-")
        os.MkdirAll(filepath.Join(sharedDir, "weights"), 0755)
        sharedWeightPath := filepath.Join(sharedDir, "weights", "weight.bin")
        os.Rename(weightsPath, sharedWeightPath)

        // Symlink back into the current .mlpackage so CompileModel still works
        os.Symlink(sharedWeightPath, weightsPath)

        r.mu.Lock()
        r.sharedWeightsPath = sharedWeightPath
        r.sharedWeightsDir = sharedDir
        r.mu.Unlock()
    }

    compiledPath := bridge.CompileModel(packagePath, tempDir)
    coremlBridge := bridge.LoadModel(compiledPath)

    return &Executable{model: coremlBridge, tempDir: tempDir, ...}, nil
}
```

**Savings per subsequent shape:** ~1.73s (skip weight.bin I/O). 8 shapes × 1.73s = **~13.8s saved**.

### 3. Parallel pre-compilation of bucket shapes (termite)

**File:** `termite/pkg/termite/lib/backends/backend_gomlx.go`

After creating `mlctx.Exec`, pre-compile all bucket shapes concurrently using `exec.PreCompile()`:

```go
func (m *onnxModel) compile(buckets bucketConfig) error {
    // ... existing exec creation ...

    if buckets.enabled {
        var eg errgroup.Group
        for _, b := range buckets.batchBuckets {
            for _, s := range buckets.sequenceBuckets {
                b, s := b, s
                eg.Go(func() error {
                    ids := tensors.FromFlatDataAndDimensions(make([]int64, b*s), b, s)
                    mask := tensors.FromFlatDataAndDimensions(make([]int64, b*s), b, s)
                    if m.hasTokenTypeIds {
                        tids := tensors.FromFlatDataAndDimensions(make([]int64, b*s), b, s)
                        return m.exec.PreCompile(ids, mask, tids)
                    }
                    return m.exec.PreCompile(ids, mask)
                })
            }
        }
        if err := eg.Wait(); err != nil {
            return fmt.Errorf("pre-compiling bucket shapes: %w", err)
        }
    }
    return nil
}
```

Thread `bucketConfig` from `loadONNX()`/`loadHuggingFace()` → `newONNXModel()`/`newHFModel()` → `compile()`.

Same change for `hfModel.compile()`.

**Impact:** 9 shapes compile in parallel instead of lazily one-at-a-time. Wall clock drops from 9 × 11.3s = ~102s to ~11.3s (limited by slowest single shape).

### 4. Cleanup: remove debug timing from runtime.go

Remove `time.Now()` / `fmt.Fprintf(os.Stderr, "[coreml] ...")` lines.

**File:** `go-coreml/runtime/runtime.go`

## Implementation Order

1. `blob.NewNullWriter` + `BlobSerializeOptions.SharedWeightsPath` (go-coreml)
2. Shared weight cache in `runtime.Runtime` (go-coreml)
3. Parallel pre-compilation (termite)
4. Remove debug timing (go-coreml)

## Expected Results

| Scenario | Before | After |
|----------|--------|-------|
| Cold start (first process run) | ~102s sequential | ~11.3s parallel |
| Steady-state inference | ~35ms/batch | ~35ms/batch (unchanged) |
| Disk writes per model load | 9 × 1.3GB = 11.7GB | 1 × 1.3GB = 1.3GB |

## Verification

```bash
cd /Users/ajroetker/go/src/github.com/antflydb/antfly/termite
go test -v -run TestCompareBackendEmbeddings -tags "xla XLA" ./pkg/termite/ -timeout 300s
```

- CoreML iter 0 should drop from ~75s → ~11s
- Steady-state inference unchanged at ~35ms/batch
- Only 1 weight.bin written (check disk I/O with `fs_usage` or stderr logging)
