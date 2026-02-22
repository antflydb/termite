#!/bin/bash
# GKE Cluster Setup for Termite TPU
#
# This script creates a GKE cluster with TPU v5e node pool optimized for
# ML inference workloads. TPU v5e provides up to 2.5x inference performance
# per dollar compared to TPU v4.
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - Project with TPU quota enabled
#   - Billing enabled
#
# Usage:
#   ./cluster-setup.sh [PROJECT_ID] [CLUSTER_NAME] [ZONE]
#
# Example:
#   ./cluster-setup.sh my-project termite-tpu us-central1-a

set -euo pipefail

# Configuration
PROJECT_ID="${1:-$(gcloud config get-value project)}"
CLUSTER_NAME="${2:-termite-tpu}"
ZONE="${3:-us-central1-a}"
REGION="${ZONE%-*}"

# TPU configuration
TPU_TYPE="tpu-v5-lite-podslice"  # v5e for inference
TPU_TOPOLOGY="2x2"               # 4 chips per node
TPU_MACHINE_TYPE="ct5lp-hightpu-4t"

# Node pool configuration
TPU_POOL_NAME="termite-tpu-pool"
TPU_MIN_NODES=0
TPU_MAX_NODES=3

echo "=== GKE TPU Cluster Setup ==="
echo "Project:  $PROJECT_ID"
echo "Cluster:  $CLUSTER_NAME"
echo "Zone:     $ZONE"
echo "TPU Type: $TPU_TYPE ($TPU_TOPOLOGY)"
echo ""

# Confirm before proceeding
read -p "Proceed with cluster creation? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Set project
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
    container.googleapis.com \
    tpu.googleapis.com \
    --quiet

# Check if cluster exists
if gcloud container clusters describe "$CLUSTER_NAME" --zone="$ZONE" &>/dev/null; then
    echo "Cluster $CLUSTER_NAME already exists. Adding TPU node pool..."
else
    echo "Creating GKE cluster: $CLUSTER_NAME"
    gcloud container clusters create "$CLUSTER_NAME" \
        --zone="$ZONE" \
        --machine-type=e2-standard-4 \
        --num-nodes=1 \
        --enable-ip-alias \
        --enable-autoscaling \
        --min-nodes=1 \
        --max-nodes=3 \
        --workload-pool="${PROJECT_ID}.svc.id.goog" \
        --logging=SYSTEM,WORKLOAD \
        --monitoring=SYSTEM
fi

# Check if TPU node pool exists
if gcloud container node-pools describe "$TPU_POOL_NAME" \
    --cluster="$CLUSTER_NAME" --zone="$ZONE" &>/dev/null; then
    echo "TPU node pool $TPU_POOL_NAME already exists."
else
    echo "Creating TPU v5e node pool: $TPU_POOL_NAME"
    gcloud container node-pools create "$TPU_POOL_NAME" \
        --cluster="$CLUSTER_NAME" \
        --zone="$ZONE" \
        --machine-type="$TPU_MACHINE_TYPE" \
        --tpu-topology="$TPU_TOPOLOGY" \
        --num-nodes=1 \
        --enable-autoscaling \
        --min-nodes="$TPU_MIN_NODES" \
        --max-nodes="$TPU_MAX_NODES" \
        --spot  # Use Spot VMs for cost savings (remove for production)
fi

# Get cluster credentials
echo "Fetching cluster credentials..."
gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="$ZONE"

# Verify TPU nodes
echo ""
echo "=== Cluster Status ==="
kubectl get nodes -l cloud.google.com/gke-tpu-accelerator="$TPU_TYPE"

echo ""
echo "=== TPU Node Details ==="
kubectl get nodes -l cloud.google.com/gke-tpu-accelerator="$TPU_TYPE" \
    -o custom-columns=\
NAME:.metadata.name,\
TPU:.metadata.labels.cloud\\.google\\.com/gke-tpu-accelerator,\
TOPOLOGY:.metadata.labels.cloud\\.google\\.com/gke-tpu-topology,\
STATUS:.status.conditions[-1].type

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Build Termite images (from termite/ directory):"
echo "     docker build -f Dockerfile.termite -t ghcr.io/antflydb/termite:latest ."
echo "     docker build -f Dockerfile.termite-omni -t ghcr.io/antflydb/termite:omni ."
echo "  2. Push to registry:"
echo "     docker push ghcr.io/antflydb/termite:latest"
echo "     docker push ghcr.io/antflydb/termite:omni"
echo "  3. Deploy Termite:"
echo "     kubectl apply -f termite-statefulset.yaml"
echo "  4. Check status:"
echo "     kubectl -n termite get pods -w"
echo ""
echo "Cost optimization tips:"
echo "  - Use --spot flag for non-critical workloads (up to 90% savings)"
echo "  - Set min-nodes=0 to scale to zero when idle"
echo "  - Use ct5lp-hightpu-1t for smaller models (1 chip instead of 4)"
