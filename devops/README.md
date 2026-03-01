# Termite DevOps

Kubernetes deployment examples for Termite ML inference service on GKE with TPU acceleration.

## Deployment Options

| Option | Use Case | Components |
|--------|----------|------------|
| [**Minimal**](./minimal/) | Development, simple production | Single StatefulSet + HPA |
| [**Comprehensive**](./comprehensive/) | Production with routing | Operator + Proxy + Multiple pools + KEDA |

## Quick Comparison

### Minimal Deployment

Best for:
- Development and testing
- Simple production workloads
- Single model serving
- Teams new to Termite

Features:
- Single StatefulSet with TPU
- Basic HPA autoscaling
- Init container for model pulling
- Simple to understand and modify

```bash
cd minimal/
./cluster-setup.sh my-project termite us-central1-a
kubectl apply -f termite-statefulset.yaml
```

### Comprehensive Deployment

Best for:
- Production with mixed workloads
- Multi-model serving with different SLAs
- Cost optimization (spot instances, scale-to-zero)
- Teams needing advanced routing

Features:
- Termite Operator for CRD-based configuration
- Termite Proxy for intelligent routing
- Multiple specialized pools (read-heavy, write-heavy, burst)
- KEDA integration for custom metrics autoscaling
- Prometheus/Grafana monitoring

```bash
cd comprehensive/
kubectl apply -f ../../config/crd/bases/
kubectl apply -f .
```

## Building Docker Images

All images are built from the `termite/` directory:

```bash
cd /path/to/antfly/termite

# Base Termite (for model pulling, CPU-only)
docker build -f Dockerfile.termite -t ghcr.io/antflydb/termite:latest .

# Termite Omni (ONNX + XLA, runtime backend selection)
docker build -f Dockerfile.termite-omni -t ghcr.io/antflydb/termite:omni .

# Routing Proxy
docker build -f Dockerfile.proxy -t ghcr.io/antflydb/termite-proxy:latest .

# Kubernetes Operator
docker build -f Dockerfile.operator -t ghcr.io/antflydb/termite-operator:latest .
```

### Multi-arch Builds

For production, build multi-architecture images:

```bash
# Create builder (one-time)
docker buildx create --name multiarch --driver docker-container --use

# Build and push
docker buildx build --platform linux/amd64,linux/arm64 \
  -f Dockerfile.termite -t ghcr.io/antflydb/termite:latest --push .

docker buildx build --platform linux/amd64,linux/arm64 \
  -f Dockerfile.termite-omni -t ghcr.io/antflydb/termite:omni --push .
```

## Image Reference

| Image | Dockerfile | Purpose | Architecture |
|-------|------------|---------|--------------|
| `termite:latest` | `Dockerfile.termite` | Base image, model pulling | amd64, arm64 |
| `termite:omni` | `Dockerfile.termite-omni` | ONNX + XLA inference | amd64, arm64 |
| `termite-proxy:latest` | `Dockerfile.proxy` | Routing proxy | amd64, arm64 |
| `termite-operator:latest` | `Dockerfile.operator` | Kubernetes operator | amd64, arm64 |

## GKE TPU Setup

Both deployment options use the same GKE cluster setup:

```bash
# Using the provided script
./minimal/cluster-setup.sh PROJECT_ID CLUSTER_NAME ZONE

# Or manually
gcloud container clusters create termite \
  --zone=us-central1-a \
  --machine-type=e2-standard-4

gcloud container node-pools create termite-tpu-pool \
  --cluster=termite \
  --zone=us-central1-a \
  --machine-type=ct5lp-hightpu-4t \
  --tpu-topology=2x2 \
  --spot
```

### TPU Machine Types

| Machine | Chips | Topology | Use Case |
|---------|-------|----------|----------|
| ct5lp-hightpu-1t | 1 | 1x1 | Small models, burst pool |
| ct5lp-hightpu-4t | 4 | 2x2 | Medium models (default) |
| ct5lp-hightpu-8t | 8 | 2x4 | Large models, high throughput |

## Environment Variables

### Termite

| Variable | Default | Description |
|----------|---------|-------------|
| `GOMLX_BACKEND` | `xla:tpu` | Backend: `xla:tpu`, `xla:cuda`, `xla:cpu` |
| `TERMITE_EMBEDDER_MODELS_DIR` | `/models/embedders` | Embedder models directory |
| `TERMITE_CHUNKER_MODELS_DIR` | `/models/chunkers` | Chunker models directory |
| `TERMITE_RERANKER_MODELS_DIR` | `/models/rerankers` | Reranker models directory |
| `ANTFLY_REGISTRY_URL` | `https://registry.antfly.ai/v1` | Model registry URL |

### Termite Proxy

| Variable | Default | Description |
|----------|---------|-------------|
| `TERMITE_PROXY_LISTEN` | `:8080` | API listen address |
| `TERMITE_PROXY_HEALTH_PORT` | `4200` | Health/metrics port |
| `TERMITE_PROXY_DEFAULT_POOL` | `default` | Default routing pool |
| `TERMITE_PROXY_REFRESH_INTERVAL` | `10s` | Endpoint refresh interval |
| `TERMITE_PROXY_NAMESPACE` | `` | Namespace to watch (empty = all) |
| `TERMITE_PROXY_SELECTOR` | `app.kubernetes.io/name=termite` | Pod label selector |

### Termite Operator

| Variable | Default | Description |
|----------|---------|-------------|
| `TERMITE_OPERATOR_METRICS_BIND_ADDRESS` | `:8080` | Metrics endpoint |
| `TERMITE_OPERATOR_HEALTH_PROBE_BIND_ADDRESS` | `:8081` | Health probes |
| `TERMITE_OPERATOR_LEADER_ELECT` | `false` | Enable leader election |
| `TERMITE_OPERATOR_TERMITE_IMAGE` | `antfly/termite:latest` | Default Termite image |
| `TERMITE_OPERATOR_DEBUG` | `false` | Debug logging |

## Testing

After deployment, test the endpoint:

```bash
# Port forward
kubectl -n termite port-forward svc/termite-tpu-lb 8080:80  # minimal
kubectl -n termite port-forward svc/termite-proxy-lb 8080:11433  # comprehensive

# List models
curl http://localhost:8080/api/models

# Generate embeddings
curl -X POST http://localhost:8080/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-small-en-v1.5", "input": "Hello, world!"}'

# Chunk text
curl -X POST http://localhost:8080/api/chunk \
  -H "Content-Type: application/json" \
  -d '{"model": "chonky-mmbert", "text": "Your long document here..."}'

# Rerank
curl -X POST http://localhost:8080/api/rerank \
  -H "Content-Type: application/json" \
  -d '{"model": "mxbai-rerank-base-v2", "query": "search query", "documents": ["doc1", "doc2"]}'
```

## Related Documentation

- [Termite CLAUDE.md](../CLAUDE.md) - Termite development guide
- [CRD Definitions](../config/crd/bases/) - TermitePool and TermiteRoute specs
- [Termite API](../pkg/termite/openapi.yaml) - OpenAPI specification
- [GKE TPU Docs](https://cloud.google.com/kubernetes-engine/docs/concepts/tpus)
