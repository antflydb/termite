# Plan: Add Shape Caching/Bucketing to GoMLX Backend

## Problem

The GoMLX backend in termite compiles a new XLA program for every unique `(batchSize, seqLen)` input shape. Each compilation costs milliseconds and ~100MB memory. With varying input sizes, this causes unbounded memory growth and repeated
compilation overhead. Additionally, `gomlxSession` uses `ExecOnceN` which recompiles on **every call**.

## Solution

Add shape bucketing: quantize input dimensions to coarse buckets so only a small fixed set of shapes ever reaches the JIT compiler. Skip bucketing for the pure Go backend (no JIT cost).

## Files to Modify

1. **`pkg/termite/lib/backends/model.go`** -- Add config fields + load options
2. **`pkg/termite/lib/backends/backend_gomlx.go`** -- All bucketing logic, session caching fix

## New File

3. **`pkg/termite/lib/backends/bucketing_test.go`** -- Unit tests

---

## Step 1: Add bucket config fields to `LoadConfig` (model.go)

Add three fields after `NumThreads` (line 215):

```go
BatchBuckets    []int // Bucket sizes for batch dimension padding (default: [1, 8, 32] for XLA/CoreML)
SequenceBuckets []int // Bucket sizes for sequence length padding (default: [32, 128, 512] for XLA/CoreML)
MaxCacheSize    int   // Max cached compiled graphs (0 = auto-calculate from buckets, -1 = unlimited)
```

Add three `LoadOption` functions after the existing options:

```go
func WithBatchBuckets(buckets []int) LoadOption
func WithSequenceBuckets(buckets []int) LoadOption
func WithMaxCacheSize(size int) LoadOption
```

No defaults in `DefaultLoadConfig()` -- defaults are backend-dependent and resolved in Step 2.

## Step 2: Add bucketing utilities (backend_gomlx.go)

Add after `init()`:

- **Default buckets**: `defaultBatchBuckets = []int{1, 8, 32}`, `defaultSequenceBuckets = []int{32, 128, 512}`
- **`shapeBucket(n int, buckets []int) (int, error)`**: returns smallest bucket >= n, error if exceeds max
- **`bucketConfig` struct**: `batchBuckets []int`, `sequenceBuckets []int`, `maxCacheSize int`, `enabled bool`
- **`resolveBucketConfig(backendType, *LoadConfig) bucketConfig`**:
- `BackendGo` without explicit buckets: `enabled=false, maxCacheSize=-1`
- `BackendGo` with explicit buckets: `enabled=true, maxCacheSize=len(batch)*len(seq)`
- `BackendXLA`/`BackendCoreML`: `enabled=true`, use defaults or user-provided, `maxCacheSize=len(batch)*len(seq)`

## Step 3: Add `buckets` field to `gomlxModelWrapper`

```go
type gomlxModelWrapper struct {
hfModel     *hfModel
onnxModel   *onnxModel
path        string
config      *LoadConfig
backendType BackendType
buckets     bucketConfig  // NEW

```

## Step 4: Set `SetMaxCache` on Exec during `compile()`

- Add `maxCacheSize int` parameter to `newHFModel()` and `newONNXModel()`
- Store in struct, call `m.exec.SetMaxCache(m.maxCacheSize)` after `NewExecAny` in both `compile()` methods
- Only call if `maxCacheSize != 0` (0 means keep GoMLX default of 32)

Update `loadHuggingFace()` and `loadONNX()` to:
1. Call `resolveBucketConfig()`
2. Pass `buckets.maxCacheSize` to `newHFModel`/`newONNXModel`
3. Store `buckets` in the wrapper

## Step 5: Implement padding/trimming in `gomlxModelWrapper.Forward()`

Refactor `Forward()` into:

- **`Forward()`**: Checks `m.buckets.enabled`. If disabled, delegates to `forwardDirect()`. If enabled:
1. Record original `batchSize` and `seqLen`
2. Call `shapeBucket()` for both dimensions
3. Call `padModelInputs()` to pad to bucketed dimensions
4. Call `forwardDirect()` with padded inputs
5. Call `trimModelOutput()` to trim back to original dimensions

- **`forwardDirect()`**: Current `Forward()` logic (delegates to hfModel/onnxModel)

- **`padModelInputs(inputs, batchSize, seqLen) *ModelInputs`**:
- If already at target size, return as-is (no allocation)
- Allocate new `[][]int32` slices at bucketed dimensions
- Copy original data, leave padding as zeros
- Pad `InputIDs`, `AttentionMask`, and `TokenTypeIDs` (if present)
- Attention mask zeros for padded positions ensure pooling correctness

- **`trimModelOutput(output, origBatch, origSeq) *ModelOutput`**:
- `LastHiddenState [batch, seq, hidden]`: slice both batch and seq
- `Embeddings [batch, hidden]`: slice batch only
- `Logits [batch, classes]`: slice batch only
- Pass through `EncoderOutput` and `PastKeyValues` unchanged

## Step 6: Fix `gomlxSession` to cache compiled graph

Current: `gomlxSession.Run()` calls `mlctx.ExecOnceN()` every time (recompiles every call).

Fix:
1. Add `exec *mlctx.Exec` field to `gomlxSession`
2. Add `compile(backendType)` method that creates `mlctx.NewExecAny` and calls `SetMaxCache`:
- Go backend: `-1` (unlimited)
- XLA/CoreML: default 32 (GoMLX default, sufficient for sessions)
3. Call `compile()` from `gomlxSessionFactory.CreateSession()` after constructing the session
4. Update `Run()` to use `s.exec.Exec()` instead of `ExecOnceN`, with fallback for safety
5. Update `Close()` to call `s.exec.Finalize()`

## Step 7: Unit tests (bucketing_test.go)

- `TestShapeBucket`: boundary values, exact match, exceeds max
- `TestResolveBucketConfig`: Go/XLA/CoreML defaults, custom overrides, Go with explicit buckets
- `TestPadModelInputs`: no-op case, batch padding, seq padding, both, TokenTypeIDs
- `TestTrimModelOutput`: each output field type, nil fields stay nil

---

## Verification

```bash
# Unit tests (pure Go, no build tags needed)
cd /Users/ajroetker/go/src/github.com/antflydb/antfly/termite
GOEXPERIMENT=simd go1.26rc2 test ./pkg/termite/lib/backends/...

# Build check
GOEXPERIMENT=simd go1.26rc2 build ./pkg/termite/lib/backends/...

# E2E tests (if models available)
make e2e E2E_TEST=TestBackendComparison
```
