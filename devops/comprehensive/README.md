# Termite on GKE TPU - Optional Components

This directory contains optional components for production deployments:
- **KEDA Integration**: Queue-depth and latency-based autoscaling
- **Prometheus/Grafana**: Full observability

For core deployment, use `make deploy` which installs from `deploy/install.yaml`.

## Quick Start

```bash
# 1. Deploy the operator and proxy
make deploy

# 2. Deploy sample pools
make deploy-samples

# 3. (Optional) Enable KEDA autoscaling
kubectl apply -f devops/comprehensive/05-autoscaling.yaml

# 4. (Optional) Enable monitoring
kubectl apply -f devops/comprehensive/06-monitoring.yaml

# 5. Watch pool status
kubectl -n termite-operator-namespace get termitepools -w
```

## Components

### KEDA Autoscaling (`05-autoscaling.yaml`)

Provides more flexible autoscaling than HPA, including:
- Scale to zero
- Custom Prometheus metrics
- Multiple trigger types

**Prerequisites:**
- [KEDA](https://keda.sh/docs/latest/deploy/) installed
- Prometheus installed and scraping termite-proxy metrics

### Prometheus/Grafana (`06-monitoring.yaml`)

Provides:
- ServiceMonitor for Prometheus scraping
- Grafana dashboard ConfigMap (auto-imported by sidecar)

**Prerequisites:**
- Prometheus Operator installed
- Grafana with sidecar enabled

## Architecture

```
                              +------------------+
                              |  LoadBalancer    |
                              |  (termite-proxy) |
                              +--------+---------+
                                       |
                              +--------v---------+
                              |  Termite Proxy   |
                              |  (3 replicas)    |
                              |  - Model routing |
                              |  - Load balance  |
                              +--------+---------+
                                       |
          +----------------------------+----------------------------+
          |                            |                            |
+---------v---------+      +-----------v-----------+     +----------v----------+
| Read-Heavy Pool   |      | Write-Heavy Pool      |     | Burst Pool          |
| (2-8 replicas)    |      | (1-15 replicas)       |     | (0-20 replicas)     |
|                   |      |                       |     |                     |
| Models:           |      | Models:               |     | Models:             |
| - bge-small       |      | - bge-small           |     | - bge-small         |
| - mxbai-rerank    |      | - chonky-mmbert       |     |                     |
| - clip-vit        |      |                       |     |                     |
|                   |      |                       |     |                     |
| TPU: 2x2          |      | TPU: 2x4              |     | TPU: 1x1            |
| Spot: No          |      | Spot: Yes             |     | Spot: Yes           |
+-------------------+      +-----------------------+     +---------------------+
```

## Workload Pools

Three specialized pools for different workload patterns:

| Pool | Use Case | Min/Max | TPU | Spot |
|------|----------|---------|-----|------|
| `read-heavy-embedders` | Search queries | 2/8 | 2x2 | No |
| `write-heavy-indexers` | Bulk indexing | 1/15 | 2x4 | Yes |
| `burst-pool` | Overflow handling | 0/20 | 1x1 | Yes |

## Monitoring

### View Pool Status

```bash
# List all pools
kubectl -n termite-operator-namespace get termitepools

# Detailed pool status
kubectl -n termite-operator-namespace describe termitepool read-heavy-embedders

# Watch pods
kubectl -n termite-operator-namespace get pods -l app.kubernetes.io/name=termite -w
```

### Proxy Metrics

```bash
# Port forward to proxy
kubectl -n termite-operator-namespace port-forward svc/termite-proxy 4200:4200

# Get metrics
curl http://localhost:4200/metrics
```

Key metrics:
- `termite_proxy_requests_total`: Request count by pool
- `termite_proxy_queue_depth`: Current queue depth
- `termite_proxy_request_duration_seconds`: Latency histogram
- `termite_proxy_endpoint_healthy`: Healthy endpoint count

### Grafana Dashboard

If using the provided dashboard (`06-monitoring.yaml`), access via:
```bash
kubectl -n monitoring port-forward svc/grafana 3000:3000
```
Dashboard is auto-imported as "Termite TPU Routing".

## Troubleshooting

### Operator Not Creating Pods

```bash
# Check operator logs
kubectl -n termite-operator-namespace logs -l app.kubernetes.io/name=termite-operator

# Check CRDs are installed
kubectl get crd termitepools.antfly.io
```

### Proxy Not Routing

```bash
# Check proxy logs
kubectl -n termite-operator-namespace logs -l app.kubernetes.io/name=termite-proxy

# Verify endpoints discovered
kubectl -n termite-operator-namespace get endpoints
```

### Pools Not Scaling

```bash
# Check KEDA ScaledObject status
kubectl -n termite-operator-namespace get scaledobject

# Check Prometheus metrics availability
kubectl -n termite-operator-namespace port-forward svc/termite-proxy 4200:4200
curl http://localhost:4200/metrics | grep queue_depth
```

## Files

| File | Description |
|------|-------------|
| `05-autoscaling.yaml` | KEDA ScaledObjects (optional) |
| `06-monitoring.yaml` | ServiceMonitor + Grafana dashboard (optional) |

## Docker Images

| Image | Dockerfile | Purpose |
|-------|------------|---------|
| `ghcr.io/antflydb/termite:latest` | `Dockerfile.termite` | Base image |
| `ghcr.io/antflydb/termite:omni` | `Dockerfile.termite-omni` | ONNX + XLA inference |
| `ghcr.io/antflydb/termite-proxy:latest` | `Dockerfile.proxy` | Routing proxy |
| `ghcr.io/antflydb/termite-operator:latest` | `Dockerfile.operator` | Kubernetes operator |

## See Also

- [Core Configuration](../../config/) - Namespace, RBAC, deployments
- [Sample Pools](../../config/samples/) - Example TermitePool configurations
- [CRD Reference](../../config/crd/bases/) - TermitePool and TermiteRoute definitions
- [Minimal Deployment](../minimal/) - Simple single-StatefulSet deployment
