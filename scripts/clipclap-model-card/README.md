---
license: mit
library_name: onnxruntime
tags:
  - onnx
  - multimodal
  - clip
  - clap
  - audio
  - image
  - text
  - embeddings
  - feature-extraction
  - antfly
  - termite
pipeline_tag: feature-extraction
datasets:
  - OpenSound/AudioCaps
---

# CLIPCLAP — Unified Text + Image + Audio Embeddings

CLIPCLAP is a unified multimodal embedding model that maps **text**, **images**, and **audio** into a shared 512-dimensional vector space. It combines OpenAI's [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) (text + image) with LAION's [CLAP](https://huggingface.co/laion/larger_clap_music_and_speech) (audio) through a trained linear projection.

Built by [antflydb](https://github.com/antflydb) for use with [Termite](https://github.com/antflydb/antfly/tree/main/termite), a standalone ML inference service for embeddings, chunking, and reranking.

## Architecture

```
Text  ──→ CLIP text encoder  ──→ text_projection  ──→ 512-dim (CLIP space)
Image ──→ CLIP visual encoder ──→ visual_projection ──→ 512-dim (CLIP space)
Audio ──→ CLAP audio encoder  ──→ audio_projection  ──→ 512-dim (CLIP space)
```

- **Text & Image**: Standard CLIP ViT-B/32 encoders and projections (unchanged from `openai/clip-vit-base-patch32`).
- **Audio**: CLAP HTSAT audio encoder from `laion/larger_clap_music_and_speech`. The audio projection combines CLAP's native audio projection (1024→512) with a trained 512→512 linear layer that maps CLAP audio space into CLIP space.

All three modalities produce **512-dimensional L2-normalized embeddings** that are directly comparable via cosine similarity.

## Intended Uses

- Multimodal search (text↔image↔audio)
- Building unified media indexes with [Antfly](https://github.com/antflydb/antfly)
- Cross-modal retrieval (find images from audio queries, audio from text, etc.)
- Audio-visual content discovery

## How to Use with Termite

```bash
# Pull and run the model
termite pull clipclap
termite run

# Embed text
curl -X POST http://localhost:8082/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "clipclap",
    "input": [
      {"type": "text", "text": "a cat sitting on a windowsill"},
      {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
      {"type": "audio_url", "audio_url": {"url": "https://example.com/cat-purring.wav"}}
    ]
  }'
```

## Training Details

### Audio Projection

The audio projection layer bridges CLAP and CLIP embedding spaces. Training procedure:

1. Load audio-caption pairs from [OpenSound/AudioCaps](https://huggingface.co/datasets/OpenSound/AudioCaps)
2. Encode audio through CLAP: audio encoder → audio_projection → L2 normalize
3. Encode captions through CLIP: text encoder → text_projection → L2 normalize
4. Train a 512→512 linear projection (CLAP audio → CLIP text) using CLIP-style contrastive loss (InfoNCE)

The contrastive loss pushes matching audio-text pairs together while pushing non-matching pairs apart within each batch, preserving content discrimination.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Training dataset | OpenSound/AudioCaps |
| Samples | 5000 audio-caption pairs |
| Epochs | 20 |
| Batch size | 256 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Loss | Symmetric InfoNCE (temperature=0.07) |
| Train/val split | 90/10 |

### Source Models

| Component | Model |
|-----------|-------|
| CLIP | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) |
| CLAP | [laion/larger_clap_music_and_speech](https://huggingface.co/laion/larger_clap_music_and_speech) |

## ONNX Files

| File | Description | Size |
|------|-------------|------|
| `text_model.onnx` | CLIP text encoder | ~254 MB |
| `visual_model.onnx` | CLIP visual encoder | ~330 MB |
| `text_projection.onnx` | CLIP text projection (512→512) | ~4 KB |
| `visual_projection.onnx` | CLIP visual projection (768→512) | ~6 KB |
| `audio_model.onnx` | CLAP HTSAT audio encoder | ~590 MB |
| `audio_projection.onnx` | Combined CLAP→CLIP projection (1024→512) | ~8 KB |

Additional files: `clip_config.json`, `tokenizer.json`, `preprocessor_config.json`, `projection_training_metadata.json`.

## Limitations

- **Audio duration**: Audio is truncated to ~10 seconds (inherited from CLAP)
- **Language**: Primarily English text support
- **Audio-visual alignment**: The projection is trained via caption similarity (audio↔text↔image), not direct audio-image pairs. Audio-to-image retrieval may be less precise than text-to-image.
- **CLIP limitations**: Inherits CLIP's weaknesses in fine-grained visual classification, object counting, and abstract concepts
- **Training data**: Audio projection trained on AudioCaps which covers common environmental sounds and may underperform on niche audio domains

## Citation

If you use CLIPCLAP, please cite the underlying models:

```bibtex
@inproceedings{radford2021clip,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle={ICML},
  year={2021}
}

@inproceedings{wu2023clap,
  title={Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  author={Wu, Yusong and Chen, Ke and Zhang, Tianyu and others},
  booktitle={ICASSP},
  year={2023}
}
```
