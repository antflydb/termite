# Termite

Termite is a standalone ML inference service for embeddings, chunking, and reranking.

## Key Directories

- `pkg/termite/` - Core service logic, API handlers, caching, per-task registries
- `pkg/termite/cmd/` - CLI entrypoint (cobra-based: `run`, `pull`, `list` subcommands)
- `lib/backends/` - Backend interface, SessionManager, ONNX/GoMLX/XLA/CoreML implementations
- `lib/pipelines/` - Task-specific pipelines (embedding, seq2seq, vision2seq, speech2seq, NER, etc.)
- `lib/embeddings/` - Embedding model wrappers with pooling
- `lib/chunking/` - Text chunking implementations
- `lib/reranking/` - Reranker implementations
- `lib/ner/` - Named entity recognition
- `lib/seq2seq/` - Seq2Seq utilities
- `lib/generation/` - Text generation infrastructure
- `lib/reading/` - Document reading (OCR)
- `lib/transcribing/` - Speech transcription
- `lib/classification/` - Text classification
- `lib/modelregistry/` - Model download/management

## Build Tags

- Default: Pure Go inference (slow, no CGO)
- `onnx,ORT`: ONNX Runtime backend (fast CPU, includes CLIP multimodal)
- `xla,XLA`: GoMLX XLA backend (TPU/CUDA/CPU)

## Patterns

**Backend abstraction** (`lib/backends/`): Backends implement a `Backend` interface with self-registration via `init()`. `SessionManager` handles backend selection by priority. Pipelines depend only on abstract `backends.Model` interface, never on concrete backend types.

**Encoder-decoder pipeline** (`lib/pipelines/encoder_decoder.go`): Shared base for Seq2Seq, Vision2Seq, and Speech2Seq. Manages encoder execution, autoregressive decoding, and KV-cache.

**Lazy model loading**: Models loaded on first request, configurable via `keep_alive` and `max_loaded_models`.

**Two-tier caching**: Memory cache + singleflight for deduplication.

## Testing

```bash
go test ./...                           # Unit tests
go test -tags="onnx,ORT" ./...          # With ONNX backend
make test                               # Full test suite
```

**E2E tests** with ONNX+XLA (downloads deps and models on first run):

```bash
make e2e                            # Run all E2E tests
make e2e E2E_TEST=TestName          # Run specific test
make e2e E2E_TIMEOUT=15m            # Custom timeout (default: 15m)
```

## Release Tags

Tags follow Go module conventions and trigger CI:

- `v*` — root module release + container build

The Termite operator and proxy now live in the antfly monorepo (`pkg/termite-operator`, `pkg/termite-proxy`); their release tags (`pkg/termite-operator/v*`, `pkg/termite-proxy/v*`) are managed there.

## Code Generation

```bash
make generate    # CRDs, DeepCopy, RBAC manifests
```
