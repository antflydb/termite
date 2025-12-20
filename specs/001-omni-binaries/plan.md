# Multi-Backend Runtime Selection for Termite

## Goal

Enable a single Termite binary to include both ONNX and XLA backends with:
- **Model-specific backends**: Models declare preferred backend in manifest
- **Fallback capability**: If preferred backend unavailable, try alternatives
- **Runtime detection**: Detect available backends/hardware at startup
- **Single binary**: One build works everywhere
- **All model types**: Embedders, chunkers, and rerankers all support multi-backend
- **Configurable priority**: Backend fallback order configurable via config

## Current State

Build tags are mutually exclusive - only one backend per build:
- `session_go.go`: `!(onnx && ORT) && !(xla && XLA)`
- `session_onnx.go`: `onnx && ORT`
- `session_xla.go`: `xla && XLA`

## Architecture Changes

### 1. Backend Registry Pattern

Create a registry where backends self-register via `init()`:

```
pkg/termite/lib/hugot/
├── backend.go              # Backend interface + registry (no build tags)
├── backend_go.go           # Pure Go backend (no build tags, always available)
├── backend_onnx.go         # ONNX backend (build tag: onnx && ORT && !darwin)
├── backend_onnx_darwin.go  # ONNX+CoreML backend (build tag: onnx && ORT && darwin)
├── backend_xla.go          # XLA backend (build tag: xla && XLA)
├── session_manager.go      # Manages sessions per backend (no build tags)
└── gpu.go                  # GPU detection (existing, no changes)
```

**Delete after migration:**
- `session_go.go`
- `session_onnx.go`
- `session_onnx_darwin.go`
- `session_xla.go`

### 2. Backend Interface

```go
// backend.go
type BackendType string

const (
    BackendGo   BackendType = "go"
    BackendONNX BackendType = "onnx"
    BackendXLA  BackendType = "xla"
)

type Backend interface {
    Type() BackendType
    Name() string                    // e.g., "ONNX Runtime (CUDA)"
    Available() bool                 // Can this backend be used?
    CreateSession(opts ...options.WithOption) (*hugot.Session, error)
}

var (
    registry = make(map[BackendType]Backend)
    defaultPriority = []BackendType{BackendONNX, BackendXLA, BackendGo}
)

func RegisterBackend(b Backend)
func GetBackend(t BackendType) (Backend, bool)
func ListAvailable() []Backend
func GetDefaultBackend(priority []BackendType) Backend  // First available in priority order
func SetPriority(priority []BackendType)  // Configure fallback order
```

### 3. Session Manager

Holds one session per backend type (lazy-created):

```go
// session_manager.go
type SessionManager struct {
    sessions map[BackendType]*hugot.Session
    mu       sync.RWMutex
}

func (sm *SessionManager) GetSession(backend BackendType) (*hugot.Session, error)
func (sm *SessionManager) GetSessionWithFallback(preferred BackendType) (*hugot.Session, BackendType, error)
func (sm *SessionManager) Close() error
```

### 4. Model Manifest Extension

Add optional `backends` field to `ModelManifest`:

```go
// pkg/termite/lib/modelregistry/manifest.go
type ModelManifest struct {
    // ... existing fields ...

    // Backends lists supported inference backends: ["onnx", "xla", "go"]
    // Empty or omitted = ALL backends supported (default)
    // Use to restrict a model to specific backends if needed
    Backends []string `json:"backends,omitempty"`
}

// SupportsBackend returns true if this model supports the given backend
// Empty Backends slice means ALL backends are supported
func (m *ModelManifest) SupportsBackend(backend BackendType) bool {
    if len(m.Backends) == 0 {
        return true // All backends supported by default
    }
    for _, b := range m.Backends {
        if BackendType(b) == backend {
            return true
        }
    }
    return false
}
```

### 5. Build Tags

New coexistence-friendly tags:

| File | Build Tag | When Included |
|------|-----------|---------------|
| `backend.go` | (none) | Always |
| `backend_go.go` | (none) | Always |
| `backend_onnx.go` | `onnx && ORT && !darwin` | With ONNX on Linux/Windows |
| `backend_onnx_darwin.go` | `onnx && ORT && darwin` | With ONNX+CoreML on macOS |
| `backend_xla.go` | `xla && XLA` | With XLA |

**Combined build:** `go build -tags="onnx,ORT,xla,XLA"`

## Files to Modify

| File | Change |
|------|--------|
| `pkg/termite/lib/hugot/backend.go` | **NEW** - Backend interface, registry, priority |
| `pkg/termite/lib/hugot/backend_go.go` | **NEW** - Pure Go backend impl |
| `pkg/termite/lib/hugot/backend_onnx.go` | **NEW** - ONNX backend impl (from session_onnx.go) |
| `pkg/termite/lib/hugot/backend_onnx_darwin.go` | **NEW** - ONNX+CoreML backend impl (from session_onnx_darwin.go) |
| `pkg/termite/lib/hugot/backend_xla.go` | **NEW** - XLA backend impl (from session_xla.go) |
| `pkg/termite/lib/hugot/session_manager.go` | **NEW** - Multi-backend session management |
| `pkg/termite/lib/hugot/session.go` | **MODIFY** - Keep backward-compat API, delegate to registry |
| `pkg/termite/lib/hugot/session_go.go` | **DELETE** |
| `pkg/termite/lib/hugot/session_onnx.go` | **DELETE** |
| `pkg/termite/lib/hugot/session_onnx_darwin.go` | **DELETE** |
| `pkg/termite/lib/hugot/session_xla.go` | **DELETE** |
| `pkg/termite/lib/modelregistry/manifest.go` | **MODIFY** - Add Backends field + SupportsBackend() |
| `pkg/termite/termite.go` | **MODIFY** - Use SessionManager instead of single session |
| `pkg/termite/model_registry.go` | **MODIFY** - Check model backend support, use fallback |
| `pkg/termite/lazy_registry.go` | **MODIFY** - Check model backend support, use fallback |
| `pkg/termite/lib/embeddings/hugot.go` | **MODIFY** - Accept SessionManager + backend selection |
| `pkg/termite/lib/chunking/hugot.go` | **MODIFY** - Accept SessionManager + backend selection |
| `pkg/termite/lib/reranking/hugot.go` | **MODIFY** - Accept SessionManager + backend selection |
| `pkg/termite/reranker.go` | **MODIFY** - Use SessionManager with backend fallback |
| `pkg/termite/chunking_cache.go` | **MODIFY** - Use SessionManager with backend fallback |
| `pkg/termite/config.go` | **MODIFY** - Add backend_priority config option |

## Implementation Order

1. **Create backend abstraction** (`backend.go`, `backend_go.go`)
2. **Create ONNX backend** (`backend_onnx.go`, `backend_onnx_darwin.go`) - move logic from session_onnx*.go
3. **Create XLA backend** (`backend_xla.go`) - move logic from `session_xla.go`
4. **Create SessionManager** (`session_manager.go`)
5. **Update session.go** for backward compatibility
6. **Update config.go** - add backend_priority option
7. **Update manifest.go** - add Backends field + SupportsBackend()
8. **Update termite.go** - use SessionManager, load priority from config
9. **Update model registries** - check backend support, use fallback
10. **Update embeddings/hugot.go** - use SessionManager
11. **Update chunking/hugot.go** - use SessionManager
12. **Update reranking/hugot.go** - use SessionManager
13. **Delete old session_*.go files**
14. **Test all build configurations**

## Backward Compatibility

- `hugot.NewSession()` continues to work (uses default backend)
- `hugot.BackendName()` continues to work
- Models without `backends` field support ALL backends (default behavior)
- Single-backend builds work unchanged

## Example Usage

**Config (termite.yaml):**
```yaml
# Backend priority - first available backend in this order wins
# Default: [onnx, xla, go]
backend_priority: [onnx, xla, go]

# GPU/accelerator selection
gpu: auto  # or cuda, coreml, tpu, off
```

**Model manifest.json (supports all backends - default):**
```json
{
  "name": "bge-small-en-v1.5",
  "type": "embedder",
  "files": [...]
}
```

**Model manifest.json (restricted to specific backends):**
```json
{
  "name": "custom-xla-model",
  "type": "embedder",
  "backends": ["xla"],
  "files": [...]
}
```

**Startup logs:**
```
INFO  Available backends: [ONNX Runtime (CoreML), GoMLX XLA (CPU), goMLX (Pure Go)]
INFO  Default backend: ONNX Runtime (CoreML)
INFO  Loading model bge-small-en-v1.5 with backend ONNX Runtime (CoreML)
INFO  Loading model custom-xla-model with backend GoMLX XLA (CPU) (model requires: [xla])
INFO  Loading model broken-model with backend goMLX (Pure Go) (fallback: onnx unavailable)
```

## Build Commands

```bash
# Pure Go only (always works, slowest)
go build ./cmd/termite

# ONNX only (current production default)
go build -tags="onnx,ORT" ./cmd/termite

# XLA only
go build -tags="xla,XLA" ./cmd/termite

# Both backends (NEW - maximum flexibility)
go build -tags="onnx,ORT,xla,XLA" ./cmd/termite
```

## Distribution Strategy

**Design principle:** Maximum fallback capability with reasonable binary/image sizes.

### Binary Releases (goreleaser)

| Archive | Build Tags | Bundled Libraries | Size |
|---------|------------|-------------------|------|
| `termite` | (none) | None | ~15MB |
| `termite-omni` | `onnx,ORT,xla,XLA` | ONNX Runtime + PJRT CPU (static) | ~150MB |

**PJRT CPU binaries** from [gomlx/pjrt-cpu-binaries](https://github.com/gomlx/pjrt-cpu-binaries) are small (~50MB) - bundled for XLA CPU fallback.

**Fallback chain in `termite-omni`:**
```
ONNX (CUDA/CoreML/CPU) → XLA (CUDA/TPU/CPU) → Pure Go
```

**What works out of the box in binaries:**
- ONNX CPU: Bundled ONNX Runtime
- XLA CPU: Static linking (no external libs needed)
- Pure Go: Always available

**What requires external libs (auto-detected at runtime):**
- ONNX CUDA: Requires CUDA libs
- ONNX CoreML: macOS only, uses system frameworks
- XLA CUDA: Requires PJRT CUDA plugin + CUDA libs
- XLA TPU: Requires `libtpu.so`

### Docker Images

| Image | Build Tags | Libraries | Use Case |
|-------|------------|-----------|----------|
| `termite:latest` | (none) | None | Development |
| `termite:omni` | `onnx,ORT,xla,XLA` | ONNX + PJRT CPU | Production (default) |

**Runtime detection on specialized hardware:**
- GKE TPU nodes: `libtpu.so` mounted from host → XLA uses TPU
- NVIDIA GPU nodes: CUDA libs from nvidia runtime → ONNX uses CUDA
- CPU nodes: Uses bundled ONNX CPU or PJRT CPU

### PJRT Library Installation

Use gomlx's `pjrt_installer` tool instead of manual download scripts:

```bash
# Install all plugins at build time
go run github.com/gomlx/go-xla/cmd/pjrt_installer@latest \
    -plugin=cpu -plugin=cuda -plugin=tpu \
    -path=/usr/local/lib/go-xla
```

**Options:**
1. **Build-time install** (Docker) - run pjrt_installer in Dockerfile
2. **Static CPU linking** - use `github.com/gomlx/gopjrt/pjrt/cpu/static` build tag (no external CPU lib needed)
3. **Runtime auto-install** - call `installer.AutoInstall()` at startup for development

**Static Linking Availability:**
| Plugin | Static Linking | Notes |
|--------|----------------|-------|
| CPU | ✅ Yes | `github.com/gomlx/gopjrt/pjrt/cpu/static` build tag |
| CUDA | ❌ No | Requires dynamic CUDA libraries from nvidia runtime |
| TPU | ❌ No | Requires `libtpu.so` mounted from GKE TPU host |

**Recommended approach for `termite-omni`:**
- **Binaries**: Static CPU linking + bundled ONNX (XLA CPU fallback works without external libs)
- **Docker**: pjrt_installer at build time for CPU plugin; CUDA/TPU detected dynamically from host mounts

### Updated Dockerfile Example

```dockerfile
# Dockerfile.termite-omni
FROM golang:1.23 AS builder

# Install PJRT plugins (CPU, CUDA, TPU)
RUN go run github.com/gomlx/go-xla/cmd/pjrt_installer@latest \
    -plugin=cpu -plugin=cuda -plugin=tpu \
    -path=/usr/local/lib/go-xla

# Build with both backends
RUN CGO_ENABLED=1 go build -tags="onnx,ORT,xla,XLA" -o /termite ./cmd/termite
```

### Files to Update

| File | Purpose |
|------|---------|
| `.goreleaser.yaml` | Add `termite-omni` builds with static CPU linking |
| `Dockerfile.termite-omni` | Use pjrt_installer for PJRT plugins |
| `scripts/download-onnxruntime.sh` | Keep for ONNX libs (already exists) |
