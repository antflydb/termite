# T5Gemma-2 Model Support

T5Gemma-2 is Google's multimodal encoder-decoder model from the Gemma 3 family, trained with the UL2 objective. This document explains how to export and run T5Gemma-2 models in Termite.

## Features

- **Text Embeddings**: 640-dimensional vectors from the encoder
- **Image Embeddings**: 640-dimensional vectors from SigLIP vision encoder
- **Text Generation**: Seq2seq generation for summarization, Q&A, rewriting
- **Multimodal**: Combined text + image understanding

## Prerequisites

### 1. Install uv (Python Package Manager)

The export script uses [uv](https://github.com/astral-sh/uv) for hermetic, reproducible Python environments:

```bash
# macOS
brew install uv

# Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. About optimum-onnx (Optional)

**The export script is self-contained** - it applies ONNX compatibility patches manually and does NOT require optimum-onnx.

If you prefer using `optimum-cli export onnx` instead of our script, we have a fork with T5Gemma-2 support:

```bash
# Only needed if using optimum-cli (not our export script)
git clone https://github.com/timkaye11/optimum-onnx.git
cd optimum-onnx
git checkout add-t5gemma2-support
pip install -e .

# Then export with optimum-cli
optimum-cli export onnx --model google/t5gemma-2-270m-270m ./output
```

The fork adds:
- `T5Gemma2OnnxConfig` in `optimum/exporters/onnx/model_configs.py`
- Removes transformers version upper bound (T5Gemma-2 requires transformers >= 4.58.0)

### 3. HuggingFace Access

T5Gemma-2 models require accepting Google's license on HuggingFace:

1. Go to https://huggingface.co/google/t5gemma-2-270m-270m
2. Accept the license agreement
3. Get a HuggingFace token from https://huggingface.co/settings/tokens

```bash
# Login to HuggingFace
huggingface-cli login
```

## Exporting Models

### Quick Start

```bash
cd termite

# Export the 270M model (recommended for development, ~6.7GB)
./scripts/export_t5gemma2.py \
  --model google/t5gemma-2-270m-270m \
  --output ~/.termite/models/rewriters/google/t5gemma-2-270m

# Verify the export with a test
./scripts/export_t5gemma2.py \
  --model google/t5gemma-2-270m-270m \
  --output ~/.termite/models/rewriters/google/t5gemma-2-270m \
  --test
```

### Available Models

| Model | Total Params | Size | Use Case |
|-------|--------------|------|----------|
| `google/t5gemma-2-270m-270m` | ~800M | ~6.7GB | Development, testing |
| `google/t5gemma-2-1b-1b` | ~2B | ~15GB | Balanced quality/speed |
| `google/t5gemma-2-4b-4b` | ~8B | ~50GB | Best quality |

### Export Options

```bash
./scripts/export_t5gemma2.py --help

Options:
  --model MODEL       HuggingFace model ID (required)
  --output DIR        Output directory for ONNX files (required)
  --test              Run validation tests after export
  --test-input TEXT   Custom test input for generation
  --skip-encoder      Skip encoder export (if already exported)
  --skip-decoder      Skip decoder export (if already exported)
  --skip-vision       Skip vision encoder export
```

### Exported Files

After export, the output directory contains:

```
~/.termite/models/rewriters/google/t5gemma-2-270m/
├── encoder.onnx           # Text encoder (1.07 GB)
├── decoder.onnx           # Decoder with KV cache (1.74 GB)
├── decoder-init.onnx      # Decoder for first token (1.74 GB)
├── vision_encoder.onnx    # SigLIP vision encoder (1.67 GB)
├── embedding_layer.onnx   # Word embeddings fallback (671 MB)
├── config.json            # HuggingFace model config
├── t5gemma2_config.json   # Termite-specific config
├── tokenizer.json         # SentencePiece tokenizer (33 MB)
├── tokenizer_config.json  # Tokenizer settings
└── preprocessor_config.json # Vision preprocessing
```

## Running in Termite

### Configuration

Set the rewriters directory in your termite config:

```yaml
# config.yaml
rewriters_dir: ~/.termite/models/rewriters
keep_alive: 5m
max_loaded_models: 2
```

Or via environment variable:

```bash
export TERMITE_REWRITERS_DIR=~/.termite/models/rewriters
```

### Starting the Server

```bash
# Build with ONNX support
CGO_ENABLED=1 go build -tags="onnx,ORT" -o termite ./pkg/termite/cmd

# Run
./termite run --config config.yaml
```

### Verifying Model Discovery

```bash
# List available models
curl http://localhost:8080/api/models | jq

# Should show:
# {
#   "rewriters": ["google/t5gemma-2-270m"],
#   ...
# }
```

## API Usage

### Text Embeddings

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/t5gemma-2-270m",
    "input": ["What is machine learning?", "How do neural networks work?"]
  }'
```

Response:
```json
{
  "data": [
    {"embedding": [0.123, -0.456, ...], "index": 0},
    {"embedding": [0.789, -0.012, ...], "index": 1}
  ],
  "model": "google/t5gemma-2-270m",
  "usage": {"prompt_tokens": 12, "total_tokens": 12}
}
```

### Image Embeddings

```bash
# Base64-encoded image
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/t5gemma-2-270m",
    "input": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]
  }'
```

### Text Generation (Rewriting)

```bash
curl -X POST http://localhost:8080/api/rewrite \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/t5gemma-2-270m",
    "texts": ["Summarize: The quick brown fox jumps over the lazy dog."]
  }'
```

Response:
```json
{
  "outputs": [
    {"text": "A fox jumps over a dog.", "tokens": [...]}
  ],
  "model": "google/t5gemma-2-270m"
}
```

## Troubleshooting

### Export Fails with "transformers version" Error

Ensure you're using the forked optimum-onnx:
```bash
cd /path/to/optimum-onnx
git checkout add-t5gemma2-support
pip install -e .
```

### "Model not found" in Termite

Check the directory structure:
```bash
ls -la ~/.termite/models/rewriters/google/t5gemma-2-270m/

# Must contain:
# - encoder.onnx
# - decoder.onnx
# - decoder-init.onnx
# - config.json with "model_type": "t5gemma2"
```

### Out of Memory During Export

T5Gemma-2 requires significant RAM for export:
- 270M model: ~8GB RAM
- 1B model: ~16GB RAM
- 4B model: ~32GB RAM

Use `--skip-vision` if you only need text capabilities.

### Slow Inference

Ensure you're using the ONNX build:
```bash
# Check build tags
./termite version

# Should show: Backend: onnx
```

## Technical Details

### Model Architecture

- **Encoder**: 18 layers, 4 attention heads (GQA: 1 KV head), 640 hidden size
- **Decoder**: 18 layers, merged self/cross attention, alternating sliding window
- **Vision**: SigLIP with 27 layers, 896x896 input, 256 tokens per image
- **Context**: 128K encoder tokens, 32K decoder tokens

### ONNX Export Patches

The export script applies several patches for ONNX compatibility:
1. Disables torch.dynamo (incompatible with tracing)
2. Replaces vmap-based attention masks with broadcasting
3. Sets sliding window to large value during export
4. Forces eager attention mode

### Known Limitations

1. **GenerateMultimodal**: Currently validates images but doesn't inject vision embeddings into generation (text generation only uses `<image>` placeholder tokens)
2. **KV Cache**: decoder.onnx currently copies decoder-init.onnx (full KV cache optimization is complex)
3. **Fused Embeddings**: Text and image embeddings are separate (no fused text+image vectors yet)

## References

- [HuggingFace Model Card](https://huggingface.co/google/t5gemma-2-270m-270m)
- [Transformers Documentation](https://huggingface.co/docs/transformers/model_doc/t5gemma2)
- [Google Blog Post](https://blog.google/technology/developers/t5gemma-2/)
- [optimum-onnx Fork](https://github.com/timkaye11/optimum-onnx/tree/add-t5gemma2-support)
