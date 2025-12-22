#!/bin/bash
#
# Pull NER models for Termite
#
# This script exports HuggingFace NER models to ONNX format and pulls them
# into the Termite models directory.
#
# Usage:
#   ./scripts/pull_ner_models.sh                    # Pull default models
#   ./scripts/pull_ner_models.sh dslim/bert-base-NER  # Pull specific model
#   MODELS_DIR=/path/to/models ./scripts/pull_ner_models.sh  # Custom dir
#

set -e

# Default models directory
MODELS_DIR="${MODELS_DIR:-$HOME/.termite/models}"

# Default NER models to pull
DEFAULT_MODELS=(
    "dslim/bert-base-NER"
    "dslim/bert-large-NER"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_python_deps() {
    log_info "Checking Python dependencies..."

    python3 -c "import torch" 2>/dev/null || {
        log_error "PyTorch not installed. Install with: pip install torch"
        exit 1
    }

    python3 -c "import transformers" 2>/dev/null || {
        log_error "Transformers not installed. Install with: pip install transformers"
        exit 1
    }

    python3 -c "import onnx" 2>/dev/null || {
        log_error "ONNX not installed. Install with: pip install onnx"
        exit 1
    }

    python3 -c "import onnxruntime" 2>/dev/null || {
        log_error "ONNX Runtime not installed. Install with: pip install onnxruntime"
        exit 1
    }

    log_info "All Python dependencies satisfied"
}

check_termite() {
    if ! command -v termite &> /dev/null; then
        log_warn "termite command not found in PATH"
        log_warn "Make sure termite is installed or use 'go run ./cmd/termite/...'"
        return 1
    fi
    return 0
}

export_and_pull() {
    local model_name="$1"
    local model_short_name=$(basename "$model_name")
    local output_dir="${MODELS_DIR}/ner/${model_short_name}"

    log_info "Processing: $model_name"
    log_info "Output directory: $output_dir"

    # Export to ONNX
    log_info "Exporting model to ONNX..."
    python3 "$(dirname "$0")/export_ner.py" "$model_name" \
        --output "$output_dir" \
        --quantize \
        --verify

    log_info "Model exported successfully to: $output_dir"

    # Verify files exist
    if [ -f "$output_dir/model.onnx" ] && [ -f "$output_dir/config.json" ]; then
        log_info "Export verified: model.onnx and config.json present"
    else
        log_error "Export failed: missing required files"
        exit 1
    fi

    echo ""
}

main() {
    echo "========================================"
    echo "  Termite NER Model Setup"
    echo "========================================"
    echo ""

    check_python_deps

    # Determine which models to process
    if [ $# -gt 0 ]; then
        MODELS=("$@")
    else
        MODELS=("${DEFAULT_MODELS[@]}")
        log_info "Using default models: ${MODELS[*]}"
    fi

    echo ""
    log_info "Models directory: $MODELS_DIR"
    mkdir -p "$MODELS_DIR/ner"

    # Process each model
    for model in "${MODELS[@]}"; do
        echo ""
        echo "----------------------------------------"
        export_and_pull "$model"
    done

    echo ""
    echo "========================================"
    echo "  Setup Complete!"
    echo "========================================"
    echo ""
    echo "Models installed to: $MODELS_DIR/ner/"
    echo ""
    echo "To run Termite with these models:"
    echo "  termite run --models-dir $MODELS_DIR"
    echo ""
    echo "To test NER:"
    echo '  curl -X POST http://localhost:8080/api/ner \'
    echo '    -H "Content-Type: application/json" \'
    echo '    -d '"'"'{"model": "bert-base-NER", "texts": ["John Smith works at Google."]}'"'"
    echo ""
}

main "$@"
