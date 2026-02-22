# Termite on GKE TPU - Minimal Deployment

Deploy Termite ML inference service on Google Kubernetes Engine with TPU v5e acceleration.

## Overview

This deployment uses:
- **GoMLX XLA backend** for hardware-accelerated inference
- **TPU v5e** for cost-efficient ML inference (up to 2.5x perf/$ vs v4)
- **R2-backed model registry** for free egress model downloads
- **StatefulSet** for stable pod identity and ordered deployment

## Architecture

```
+-------------------------------------------------------------+
| GKE Cluster                                                  |
| +-----------------------------------------------------------+|
| | TPU Node Pool (ct5lp-hightpu-4t, 2x2 topology)            ||
| | +-------------------------------------------------------+ ||
| | | Termite Pod                                            | ||
| | | +---------------+  +--------------------------------+  | ||
| | | | Init:         |  | Container:                     |  | ||
| | | | model-puller  |->| termite (GoMLX XLA -> TPU)     |  | ||
| | | |               |  |                                |  | ||
| | | | /termite pull |  | GOMLX_BACKEND=xla:tpu          |  | ||
| | | | from R2       |  | /models (emptyDir)             |  | ||
| | | +---------------+  +--------------------------------+  | ||
| | +-------------------------------------------------------+ ||
| +-----------------------------------------------------------+|
+-------------------------------------------------------------+
```

## Quick Start

### 1. Create GKE Cluster with TPU

```bash
# Set your project
export PROJECT_ID=your-project-id

# Create cluster and TPU node pool
./cluster-setup.sh $PROJECT_ID termite-tpu us-central1-a
```

### 2. Build and Push Images

From the `termite/` directory:

```bash
# Build base image (used for model pulling)
docker build -f Dockerfile.termite -t ghcr.io/antflydb/termite:latest .

# Build omni image (ONNX + XLA, used for inference)
docker build -f Dockerfile.termite-omni -t ghcr.io/antflydb/termite:omni .

# Push to registry
docker push ghcr.io/antflydb/termite:latest
docker push ghcr.io/antflydb/termite:omni
```

### 3. Deploy Termite

```bash
# Apply manifests
kubectl apply -f termite-statefulset.yaml

# Watch deployment
kubectl -n termite get pods -w

# Check logs
kubectl -n termite logs -f termite-tpu-0 -c termite
```

### 4. Test Endpoint

```bash
# Port forward for local testing
kubectl -n termite port-forward svc/termite-tpu-lb 8080:80

# List models
curl http://localhost:8080/api/models

# Generate embeddings
curl -X POST http://localhost:8080/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-small-en-v1.5", "input": "Hello, world!"}'
```

## Configuration

### Model Selection

Edit the ConfigMap to change which models are pulled:

```yaml
# termite-statefulset.yaml
data:
  TERMITE_MODELS: "bge-small-en-v1.5,mxbai-rerank-base-v1,chonky-mmbert-small-multilingual-1"
  TERMITE_VARIANTS: "i8"  # Optional: pull i8 variants for all models
```

To use a variant in your application config, append the variant suffix:
```yaml
embedder:
  model: bge-small-en-v1.5-i8  # Use INT8 quantized variant
```

### TPU Topology

Adjust based on model size and throughput needs:

| Machine Type | Chips | Topology | Use Case |
|--------------|-------|----------|----------|
| ct5lp-hightpu-1t | 1 | 1x1 | Small models, low traffic |
| ct5lp-hightpu-4t | 4 | 2x2 | Medium models (default) |
| ct5lp-hightpu-8t | 8 | 2x4 | Large models, high throughput |

Update `nodeSelector` and resource requests:

```yaml
nodeSelector:
  cloud.google.com/gke-tpu-topology: "2x4"  # Match your pool
resources:
  requests:
    google.com/tpu: 8  # Must match chip count
  limits:
    google.com/tpu: 8
```

### Backend Selection

The `GOMLX_BACKEND` environment variable controls hardware:

| Value | Hardware | Use Case |
|-------|----------|----------|
| `xla:tpu` | TPU | GKE TPU nodes (default) |
| `xla:cuda` | NVIDIA GPU | GPU nodes |
| `xla:cpu` | CPU | Fallback/testing |

## Cost Optimization

### Spot VMs

The cluster setup uses `--spot` for TPU nodes. Remove for production:

```bash
# In cluster-setup.sh, remove --spot flag
gcloud container node-pools create ... # Remove --spot
```

### Scale to Zero

Set `min-nodes=0` to scale down when idle:

```yaml
# HPA in termite-statefulset.yaml
spec:
  minReplicas: 0  # Warning: cold start takes 2-5 minutes
```

### Right-size TPU

For smaller models (< 500M params), use single-chip nodes:

```bash
gcloud container node-pools create termite-tpu-pool \
  --machine-type=ct5lp-hightpu-1t \
  --tpu-topology=1x1 \
  ...
```

## Troubleshooting

### Pod stuck in Pending

```bash
# Check TPU node availability
kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu-v5-lite-podslice

# Check events
kubectl -n termite describe pod termite-tpu-0
```

Common causes:
- TPU quota exceeded
- No TPU nodes available (check autoscaling)
- Resource request mismatch (must match node's chip count exactly)

### XLA Compilation Slow

First request triggers JIT compilation (30-120 seconds). Subsequent requests are fast.

```yaml
# Increase startup probe timeout if needed
startupProbe:
  failureThreshold: 60  # Allow 10 minutes
```

### Model Pull Failures

```bash
# Check init container logs
kubectl -n termite logs termite-tpu-0 -c model-puller

# Verify registry access
curl -I https://registry.antfly.ai/v1/index.json
```

### libtpu.so Not Found

On non-GKE environments, libtpu.so may not be available:

```bash
# Check library path
kubectl -n termite exec termite-tpu-0 -- ls -la /usr/local/lib/

# Fall back to CPU for testing
kubectl -n termite set env statefulset/termite-tpu GOMLX_BACKEND=xla:cpu
```

## Files

| File | Description |
|------|-------------|
| `termite-statefulset.yaml` | Main Kubernetes manifests |
| `cluster-setup.sh` | GKE cluster and TPU pool creation |

## Docker Images

| Image | Dockerfile | Purpose |
|-------|------------|---------|
| `ghcr.io/antflydb/termite:latest` | `Dockerfile.termite` | Base image for model pulling |
| `ghcr.io/antflydb/termite:omni` | `Dockerfile.termite-omni` | ONNX + XLA inference |

## See Also

- [Comprehensive Deployment](../comprehensive/) - Multi-pool routing with proxy and operator
- [Termite API](../../pkg/termite/openapi.yaml) - OpenAPI specification
