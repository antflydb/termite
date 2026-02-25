# WS1: Termite SPLADE Sparse Embedding Pipeline

## Context

Antfly's hybrid search needs sparse (SPLADE) vector support alongside dense embeddings and BM25. SPLADE models use the same BERT tokenizer and ONNX runtime as dense embedders -- the only difference is output processing: instead of pooling hidden states into a fixed-dimension dense vector, SPLADE applies `max(0, log(1+exp(x)))` activation over the vocab dimension and sparsifies to top-k entries.

This plan covers the Termite-side work (WS1 from `work-log/planned/sparse-embeddings.md`): adding a sparse embedding pipeline that reuses the existing `/embed` endpoint (Pinecone-style), where the model's "sparse" capability determines whether the response contains `dense embeddings` or `sparse_embeddings`.

## Design Decisions

- **Same `/embed` endpoint** -- extend request with optional `top_k`/`min_weight`, extend response with optional `sparse_embeddings` field
- **Same model directory** -- sparse models live in `models/embedders/` alongside dense models, differentiated by "sparse" capability in manifest
- **Single `EmbedderRegistry`** -- extended with `AcquireSparse()` method for sparse models
- **Separate Go interface** -- `SparseEmbedder` in `libaf/embeddings/` (return type is `[]SparseVector` not `[][]float32`)
- **New pipeline** -- `SparseEmbeddingPipeline` shares tokenizer/backend, different post-processing

## Files to Create

### 1. `antfly-go/libaf/embeddings/sparse.go` -- SparseEmbedder interface

```go
type SparseVector struct {
    Indices []int32
    Values  []float32
}

type SparseEmbedder interface {
    SparseEmbed(ctx context.Context, texts []string) ([]SparseVector, error)
}
```

Text-only interface (no multimodal SPLADE models exist). Separate from `Embedder` because the return type is fundamentally different.

### 2. `termite/pkg/termite/lib/pipelines/sparse_embedding.go` -- Pipeline

```go
type SparseEmbeddingPipelineConfig struct {
    MaxLength int     // default 512
    TopK      int     // default 256
    MinWeight float32 // default 0.0
}

type SparseEmbeddingPipeline struct {
    Model     backends.Model
    Tokenizer tokenizers.Tokenizer
    Config    *SparseEmbeddingPipelineConfig
}

func (p *SparseEmbeddingPipeline) Embed(ctx, texts) ([]SparseVector, error)
func (p *SparseEmbeddingPipeline) EmbedBatch(ctx, inputs) ([]SparseVector, error)
func (p *SparseEmbeddingPipeline) Close() error

func LoadSparseEmbeddingPipeline(modelPath, sessionManager, backends, opts...) (*SparseEmbeddingPipeline, BackendType, error)
```

Composition, not inheritance -- doesn't embed `EmbeddingPipeline` because it has different output processing and doesn't need `ImageProcessor`/`AudioProcessor`/`Projector`. Directly holds the same `backends.Model` and `tokenizers.Tokenizer`.

**Shared tokenization** -- extract `TokenizeTexts(tokenizer, texts, maxLength) (*ModelInputs, error)` into `pipelines/tokenize.go` so both `EmbeddingPipeline.Embed` and `SparseEmbeddingPipeline.Embed` use it.

**SPLADE post-processing** in `EmbedBatch`:
1. `Model.Forward(ctx, inputs)` → `ModelOutput`
2. If `output.Logits` exists → use directly `[batch, vocab]`
3. Else if `output.LastHiddenState` → max-pool over sequence dim (SIMD via `hwy.Max`)
4. Apply `log(1+exp(x))` element-wise via `activation.Softplus` from go-highway (already implemented with SIMD + numerical stability threshold at 20.0). Softplus output is always >= 0, so no separate ReLU/max(0,x) step needed.
5. Threshold at `MinWeight`, take top-k by value → `[]SparseVector`

### 3. `termite/pkg/termite/lib/pipelines/sparse_embedding_test.go` -- Tests

- SPLADE activation function correctness (known input/output pairs)
- Sparsification: top-k selection, min_weight threshold
- Padding mask handling (masked positions excluded from max-pool)
- Batch processing (multiple texts)
- Empty input handling

### 4. `termite/pkg/termite/lib/pipelines/tokenize.go` -- Shared tokenization

Extract from `EmbeddingPipeline.Embed` (lines 566-609 of `embedding.go`):

```go
func TokenizeTexts(tokenizer tokenizers.Tokenizer, texts []string, maxLength int) (*backends.ModelInputs, error)
```

Both `EmbeddingPipeline.Embed` and `SparseEmbeddingPipeline.Embed` call this.

### 5. `termite/pkg/termite/lib/embeddings/sparse_embedder.go` -- PooledSparseEmbedder

Mirrors `PooledEmbedder` (`embedder.go:87`) -- semaphore + round-robin over N `SparseEmbeddingPipeline` instances:

```go
type PooledSparseEmbedder struct {
    pipelines    []*pipelines.SparseEmbeddingPipeline
    sem          *semaphore.Weighted
    nextPipeline atomic.Uint64
    poolSize     int
    batchSize    int
    backendType  backends.BackendType
}

func NewPooledSparseEmbedder(cfg, sessionManager) (*PooledSparseEmbedder, BackendType, error)
func (p *PooledSparseEmbedder) SparseEmbed(ctx, texts) ([]SparseVector, error)
func (p *PooledSparseEmbedder) Close() error
```

### 6. `termite/pkg/termite/sparse_embedding_cache.go` -- Caching

Mirrors `embedding_cache.go` but for `[]SparseVector`. Cannot reuse existing cache because it's typed `[][]float32`.

```go
type SparseEmbeddingCache struct { ... }
type CachedSparseEmbedder struct { ... }
```

Same pattern: TTL cache (2 min) + singleflight deduplication. Cache key uses xxhash of model + text content (simpler than multimodal -- text only).

## Files to Modify

### 7. `termite/pkg/termite/lib/modelregistry/manifest.go`

Add capability constant:

```go
CapabilitySparse Capability = "sparse"
```

### 8. `termite/pkg/termite/openapi.yaml`

Add `SparseVector` schema:

```yaml
SparseVector:
  type: object
  required: [indices, values]
  properties:
    indices:
      type: array
      items: { type: integer, format: int32 }
    values:
      type: array
      items: { type: number, format: float }
```

Extend `EmbedRequest` -- add optional fields:

```yaml
top_k:
  type: integer
  description: "Max non-zero entries per sparse vector (only for sparse models, default 256)"
min_weight:
  type: number
  format: float
  description: "Min weight threshold for sparse vectors (only for sparse models, default 0.0)"
```

Extend `EmbedResponse` -- add optional field:

```yaml
sparse_embeddings:
  type: array
  items:
    $ref: '#/components/schemas/SparseVector'
  description: "Sparse embedding vectors (populated when model has sparse capability, mutually exclusive with embeddings)"
```

Run `make generate` after changes.

### 9. `termite/pkg/termite/embedder_registry.go`

Add `AcquireSparse(modelName) (SparseEmbedder, error)` method. When a model has `CapabilitySparse`:
- Load creates `PooledSparseEmbedder` instead of `PooledEmbedder`
- Store in the same TTL cache (use any type or a wrapper interface)
- Same ref-counting and eviction logic

Alternative: separate sparse TTL cache field alongside the existing cache (simpler typing).

### 10. `termite/pkg/termite/api.go`

Extend `handleApiEmbed` (line 226):
1. After acquiring model, check if it has "sparse" capability
2. If sparse: acquire via `AcquireSparse()`, wrap in `CachedSparseEmbedder`, call `SparseEmbed`
3. Response: populate `sparse_embeddings` field instead of `embeddings`
4. Binary format: use sparse serialization (see below)

### 11. `termite/pkg/termite/codec.go`

Add sparse binary serialization/deserialization. Use Content-Type to distinguish formats: dense returns `application/octet-stream` (unchanged), sparse returns `application/x-sparse-vectors`. No format tags needed.

Sparse binary format:

```
[uint64 num_vectors]
For each vector:
  [uint32 nnz]              // number of non-zero entries
  [int32  * nnz indices]    // sorted ascending
  [float32 * nnz values]    // corresponding weights
```

All values little-endian. Typical wire size for top_k=256: `8 + N*(4 + 256*4 + 256*4) = 8 + N*2052` bytes.

```go
func SerializeSparseVectors(w io.Writer, vecs []SparseVector) error
func DeserializeSparseVectors(r io.Reader) ([]SparseVector, error)
```

### 12. `termite/pkg/client/client.go`

Add client methods:

```go
func (c *TermiteClient) SparseEmbed(ctx, model, input) ([]SparseVector, error)
func (c *TermiteClient) SparseEmbedWithConfig(ctx, model, input, topK, minWeight) ([]SparseVector, error)
```

These call the same `/embed` endpoint but set `top_k`/`min_weight` in the request and parse `sparse_embeddings` from the response.

Also add `deserializeSparseVectors` for binary format handling.

### 13. `termite/pkg/termite/termite.go`

Initialize `SparseEmbeddingCache` alongside existing `EmbeddingCache`.

### 14. Models response

Extend `/api/models` response to indicate which models have sparse capability, so clients can discover sparse-capable models.

## Implementation Order

1. Define types: `SparseVector`, `SparseEmbedder` interface (`libaf/embeddings/sparse.go`)
2. Extract shared tokenization helper (`pipelines/tokenize.go`), refactor `EmbeddingPipeline.Embed` to use it
3. Implement `SparseEmbeddingPipeline` with SPLADE activation (`pipelines/sparse_embedding.go`) + tests
4. Implement `PooledSparseEmbedder` (`embeddings/sparse_embedder.go`)
5. Add `CapabilitySparse` to manifest (`modelregistry/manifest.go`)
6. Update OpenAPI spec (`openapi.yaml`) + `make generate`
7. Extend `EmbedderRegistry` with `AcquireSparse()`
8. Add sparse binary codec (`codec.go`)
9. Implement sparse cache (`sparse_embedding_cache.go`)
10. Extend `handleApiEmbed` handler (`api.go`)
11. Add client methods (`client.go`)
12. Wire up in `termite.go`

## Verification

```bash
# Unit tests for SPLADE pipeline
GOEXPERIMENT=simd go test ./termite/pkg/termite/lib/pipelines/... -run Sparse -v

# Unit tests for pooled sparse embedder
GOEXPERIMENT=simd go test ./termite/pkg/termite/lib/embeddings/... -run Sparse -v

# Build verification
GOEXPERIMENT=simd go build ./termite/...

# After exporting a SPLADE model:
# scripts/export_model_to_registry.py --model naver/splade-cocondenser-ensembledistil --type embedder --capabilities sparse
# Test end-to-end with swarm mode
```
