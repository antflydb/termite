# Termite

Termite is a standalone ML inference service for embeddings, chunking, and reranking.

## Key Directories

- `cmd/termite/` - CLI entrypoint (cobra-based: `run`, `pull`, `list` subcommands)
- `cmd/termite-operator/` - Kubernetes operator entrypoint
- `cmd/termite-proxy/` - Load-balancing proxy entrypoint
- `pkg/termite/` - Core service logic, API handlers, caching
- `pkg/operator/` - Kubernetes operator (CRDs, controllers)
- `pkg/proxy/` - Request routing and K8s service discovery
- `lib/hugot/` - Inference session abstraction (ONNX, XLA, pure Go backends)
- `lib/embeddings/` - Embedding model implementations
- `lib/chunking/` - Text chunking implementations
- `lib/reranking/` - Reranker implementations
- `lib/modelregistry/` - Model download/management

## Build Tags

- Default: Pure Go inference (slow, no CGO)
- `onnx,ORT`: ONNX Runtime backend (fast CPU, includes CLIP multimodal)
- `xla,XLA`: GoMLX XLA backend (TPU/CUDA/CPU)

## Patterns

**Session abstraction** (`lib/hugot/session*.go`): Build tags select the backend implementation. `session.go` has the interface, `session_go.go`, `session_onnx.go`, and `session_xla.go` provide implementations.

**Lazy model loading**: Models loaded on first request, configurable via `keep_alive` and `max_loaded_models`.

**Two-tier caching**: Memory cache + singleflight for deduplication.

**Operator CRDs**: `TermitePool` (scaling/hardware) and `TermiteRoute` (traffic routing) in `pkg/operator/api/v1alpha1/`.

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

## Code Generation

```bash
make generate    # CRDs, DeepCopy, RBAC manifests
```
