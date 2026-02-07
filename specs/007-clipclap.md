# Plan: CLIP+CLAP Unified Model (`antflydb/clipclap`)

Train a projection layer mapping CLAP audio embeddings into CLIP embedding space, package as a unified `antflydb/clipclap` model supporting text, image, and audio search in a single embedding space.

## Architecture

Both CLIP and CLAP produce 512-dim embeddings in different spaces. We train a linear projection (512→512) using **text as a bridge**: encode the same text with both models' text encoders, then learn the mapping CLAP_space → CLIP_space. At inference:

- **Text** → CLIP text encoder → 512-dim (CLIP space)
- **Image** → CLIP visual encoder + visual_projection → 512-dim (CLIP space)
- **Audio** → CLAP audio encoder + **trained projection** → 512-dim (CLIP space)

## Files to Create

### 1. Training Script: `scripts/train_clipclap_projection.py`
- Uses PEP 723 `uv` inline script metadata (same pattern as `export_model_to_registry.py`):
  ```python
  #!/usr/bin/env -S uv run
  # /// script
  # requires-python = ">=3.10"
  # dependencies = [
  #     "transformers",
  #     "torch",
  #     "onnx",
  #     "datasets",
  #     "huggingface_hub",
  # ]
  # ///
  ```
- Loads CLIP (`openai/clip-vit-base-patch32`) and CLAP (`laion/larger_clap_music_and_speech`) from HuggingFace
- Downloads COCO Captions via `datasets` library (~600k diverse captions)
- Encodes text with both CLIP and CLAP text encoders → paired (CLIP_embed, CLAP_embed)
- Trains a `nn.Linear(512, 512)` projection via cosine similarity loss
- Exports to `audio_projection.onnx` (input: `[batch, 512]`, output: `[batch, 512]`)
- CLI args: `--clip-model`, `--clap-model`, `--output-dir`, `--epochs`, `--push-to-hub`
- `--push-to-hub antflydb/clipclap` uploads the assembled model to HuggingFace via `huggingface_hub` (uses `HF_TOKEN` env var)
- Runnable as: `uv run scripts/train_clipclap_projection.py --output-dir /tmp/clipclap`
- With upload: `uv run scripts/train_clipclap_projection.py --output-dir /tmp/clipclap --push-to-hub antflydb/clipclap`

### 2. Exporter: `scripts/exporters/clipclap.py`
- Registered as `@register_exporter("embedder", capability="image,audio")`
- Assembles the combined model from source CLIP + CLAP + trained projection:
  - `text_model.onnx` from CLIP text encoder
  - `visual_model.onnx` from CLIP visual encoder
  - `visual_projection.onnx` from CLIP
  - `text_projection.onnx` from CLIP
  - `audio_model.onnx` from CLAP audio encoder
  - `audio_projection.onnx` — the trained CLAP→CLIP projection
  - `tokenizer.json`, `preprocessor_config.json` from CLIP
  - Combined `clip_config.json` with vision, text, AND audio config sections
- Falls back to training the projection on-the-fly if not pre-computed

### 3. Go Embedder: `pkg/termite/lib/embeddings/clipclap.go`
- `CLIPCLAPEmbedder` struct with `textPipeline`, `visualPipeline`, `audioPipeline`
- Constructor calls `pipelines.LoadEmbeddingPipelines()` which returns all 3 pipelines
  - The audio pipeline automatically picks up `audio_projection.onnx` as its `Projector` (existing mechanism at `embedding.go:1047-1084`)
- `Embed()` routes: text→textPipeline, image→visualPipeline, audio→audioPipeline
- Capabilities: `text/plain`, `image/*`, `audio/*`
- Pattern follows existing `CLIPEmbedder` and `CLAPEmbedder` exactly

### 4. Model Discovery: `pkg/termite/model_registry.go`
- Add `isMultimodalCLIPCLAPModel(modelPath)` — checks for `visual_model.onnx` AND `audio_model.onnx` AND `text_model.onnx`

### 5. Registry Loading: `pkg/termite/embedder_registry.go`
- In `discoverModels()`: check for CLIPCLAP **before** CLIP/CLAP (since CLIPCLAP matches both patterns)
- In `loadModel()`: add `case "clipclap"` → `NewCLIPCLAPEmbedder(...)`
- In `HasCapability()`: clipclap has both `CapabilityImage` and `CapabilityAudio`

### 6. Exporter Registration: `scripts/exporters/__init__.py`
- Add `from . import clipclap` to trigger registration

### 7. E2E Test: `e2e/clipclap_test.go`
- Tests text, image, AND audio embedding with the combined model
- Verifies all 3 modalities produce 512-dim embeddings
- Checks cross-modal similarity (all pairs share the same space)

## Key Design Decisions

1. **Reuse existing `Projector` mechanism**: The `EmbeddingPipeline` already supports loading `audio_projection.onnx` and applying it post-inference (`embedding.go:1047-1084`, `embedding.go:521-527`). We name our trained projection `audio_projection.onnx` and it gets picked up automatically.

2. **Single config file**: Create a `clip_config.json` with `model_type: "clipclap"` containing all 3 modality configs (vision, text, audio). The raw config loader already supports these nested fields.

3. **Discovery order**: Check for CLIPCLAP (has all 3 encoder files) before CLIP (has visual+text) or CLAP (has audio+text) to prevent misclassification.

4. **Text-bridged training**: Only requires text data (no paired audio-image data). A linear projection trains in minutes. ~5k diverse captions is sufficient.

## Existing Code to Reuse

| Component | File | Pattern |
|-----------|------|---------|
| Projection loading | `lib/pipelines/embedding.go:1047-1084` | `loadAudioEmbeddingPipeline` auto-loads `audio_projection.onnx` |
| Projection application | `lib/pipelines/embedding.go:539-561` | `applyProjection()` runs embeddings through Projector |
| CLIP embedder pattern | `lib/embeddings/clip.go` | Struct layout, constructor, Embed(), extractContent() |
| CLAP embedder pattern | `lib/embeddings/clap.go` | Audio content extraction, EmbedAudio convenience methods |
| Model discovery | `model_registry.go:55-94` | `isMultimodalModel()` / `isMultimodalAudioModel()` |
| Registry loading | `embedder_registry.go:456-492` | `loadModel()` switch cases |
| E2E test patterns | `e2e/clip_test.go`, `e2e/clap_test.go` | Server setup, embedImage(), embedAudio(), createTestAudio() |

## Implementation Order

1. `scripts/train_clipclap_projection.py` — training script (Python)
2. `scripts/exporters/clipclap.py` — exporter (Python)
3. `scripts/exporters/__init__.py` — register the new exporter
4. `pkg/termite/lib/embeddings/clipclap.go` — Go embedder
5. `pkg/termite/model_registry.go` — add `isMultimodalCLIPCLAPModel()`
6. `pkg/termite/embedder_registry.go` — discovery + loading + capability
7. `e2e/clipclap_test.go` — E2E test

## Verification

1. **Training**: `uv run scripts/train_clipclap_projection.py --output-dir /tmp/clipclap_projection` → produces `audio_projection.onnx`
2. **Export**: `uv run scripts/export_model_to_registry.py embedder antflydb/clipclap --capabilities image,audio --output-dir /tmp/clipclap` → assembles full model
3. **Unit tests**: `GOEXPERIMENT=simd go1.26rc2 test ./pkg/termite/lib/embeddings/... -run CLIPCLAP`
4. **E2E**: `make e2e E2E_TEST=TestCLIPCLAPE2E` — tests all 3 modalities + cross-modal similarity
