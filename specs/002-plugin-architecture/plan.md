# Termite Plugin Command System

Add `termite plugin` command for managing backend runtime plugins (XLA and ONNX), with auto-install on startup.

> **First step**: Write this spec to `specs/002-plugin-architecture/README.md` in the termite repo.

## Key Insight: Dynamic Loading

Both ONNX and XLA use **dynamic loading** (dlopen) at runtime:
- **libonnxruntime**: Loaded via `ort.SetSharedLibraryPath()` + `InitializeEnvironment()`
- **PJRT plugins**: Loaded via go-xla's dlopen wrapper
- **libtokenizers**: Statically linked at compile time (already in binary)

This means the binary can start without GPU libraries and download them on-demand.

## Distribution Strategy

**Omni builds bundle CPU fallbacks only:**
- libonnxruntime.so/dylib (CPU execution provider)
- pjrt_c_api_cpu_plugin.so

**GPU plugins downloaded on-demand:**
- `xla:cuda` - PJRT CUDA plugin + NVIDIA libs
- `xla:tpu` - PJRT TPU plugin

## Command Structure

```
termite plugin list                    # Show installed/available plugins
termite plugin install xla:cuda        # Install PJRT CUDA plugin + NVIDIA libs
termite plugin install xla:tpu         # Install PJRT TPU plugin
termite plugin install onnx:cuda       # Install ONNX CUDA execution provider (if needed)
```

## Configuration

```yaml
# termite.yaml
plugins_dir: ~/.termite/plugins        # Default location
plugins:
  auto_install: true                   # Auto-download GPU plugins on startup if hardware detected
```

## Plugin Directory Layout

```
~/.termite/plugins/
  xla/
    cuda/pjrt_c_api_cuda_plugin.so + nvidia libs
    tpu/pjrt_c_api_tpu_plugin.so
  onnx/
    cuda/...  # CUDA execution provider libs if separate
```

---

## Implementation Steps

### 1. Add paths helper
**File:** `pkg/termite/lib/paths/paths.go`
- Add `DefaultPluginsDir()` returning `~/.termite/plugins/`

### 2. Create plugins package
**New files in `pkg/termite/lib/plugins/`:**

| File | Purpose |
|------|---------|
| `plugin.go` | PluginSpec type, ParsePluginSpec("xla:cuda") |
| `registry.go` | Discover installed plugins, list available/bundled |
| `installer.go` | Download/extract logic, progress callbacks |
| `installer_xla.go` | XLA GPU install via `github.com/gomlx/go-xla/pkg/installer` (build-tagged) |
| `installer_xla_stub.go` | Stub for non-XLA builds |
| `autoinstall.go` | GPU detection + auto-install on startup |

Note: ONNX CPU is bundled; no separate installer needed for base functionality.

### 3. Add CLI commands
**New files in `pkg/termite/cmd/cmd/`:**

| File | Purpose |
|------|---------|
| `plugin.go` | Plugin command group |
| `plugin_list.go` | List subcommand with tabular output |
| `plugin_install.go` | Install subcommand with progress display |

**Modify:** `root.go` - Add plugin command, config bindings for `plugins_dir` and `plugins.auto_install`

### 4. Integrate with backends
**Modify:** `pkg/termite/lib/hugot/backend_xla.go`
- In `init()`, check `~/.termite/plugins/xla/{device}/` before bundled/standard paths
- Update `findUserPlugin()` to use new paths

**Modify:** `pkg/termite/lib/hugot/backend_onnx.go` and `backend_onnx_darwin.go`
- Add `~/.termite/plugins/onnx/{platform}/lib/` to library search order
- Check user plugins before ONNXRUNTIME_ROOT

### 5. Add auto-install on startup
**Modify:** `pkg/termite/termite.go` (in `RunAsTermite`)
- After config parsing, before session manager init:
```go
if config.Plugins.AutoInstall {
    autoInstaller := plugins.NewAutoInstaller(...)
    if err := autoInstaller.DetectAndInstall(ctx); err != nil {
        zl.Warn("Auto-install failed", zap.Error(err))
    }
}
```

### 6. Update config schema
**Modify:** `pkg/termite/openapi.yaml`
- Add `plugins_dir` and `plugins.auto_install` to Config schema

---

## Key Files to Modify

- `pkg/termite/lib/paths/paths.go` - Add DefaultPluginsDir
- `pkg/termite/cmd/cmd/root.go` - Add plugin cmd, config bindings
- `pkg/termite/lib/hugot/backend_xla.go` - User plugin discovery
- `pkg/termite/lib/hugot/backend_onnx.go` - User plugin discovery
- `pkg/termite/lib/hugot/backend_onnx_darwin.go` - User plugin discovery
- `pkg/termite/termite.go` - Auto-install on startup
- `pkg/termite/openapi.yaml` - Config schema

## Key Files to Create

- `pkg/termite/lib/plugins/` - New package (6-7 files)
- `pkg/termite/cmd/cmd/plugin.go` - Command group
- `pkg/termite/cmd/cmd/plugin_list.go` - List subcommand
- `pkg/termite/cmd/cmd/plugin_install.go` - Install subcommand

## Dependencies

- `github.com/gomlx/go-xla/pkg/installer` - Already transitive dep (v0.1.4)
- ONNX download logic ported from `scripts/download-onnxruntime.sh`

## Design Decisions

1. **Location: `~/.termite/plugins/`** - Consistent with models dir, no sudo needed
2. **Bundle CPU, download GPU** - Works offline for CPU, auto-downloads GPU plugins when hardware detected
3. **Auto-install: Synchronous at startup** - Server is fully ready when it starts; configurable via `plugins.auto_install: false`
4. **Failure handling: Graceful degradation** - Log warning, continue with CPU fallback
5. **XLA integration: Use go-xla installer** - Leverage existing installer for CUDA/TPU with build-tag guards
