#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers",
#     "torch",
#     "onnx",
#     "onnxscript",
#     "datasets",
#     "huggingface_hub",
#     "pillow",
# ]
# ///
"""
Build the antflydb/clipclap unified multimodal model.

Loads CLIP (text+image) and CLAP (audio), trains a linear projection mapping
CLAP audio embeddings into CLIP space using text as a bridge, then exports
all ONNX files and configs needed for a single model that embeds text, images,
and audio into a shared 512-dim space.

At inference time:
  - Text  → CLIP text encoder → 512-dim (CLIP space)
  - Image → CLIP visual encoder + visual_projection → 512-dim (CLIP space)
  - Audio → CLAP audio encoder + trained projection → 512-dim (CLIP space)

Output directory will contain:
  - text_model.onnx           (CLIP text encoder)
  - visual_model.onnx         (CLIP visual encoder)
  - text_projection.onnx      (CLIP text projection)
  - visual_projection.onnx    (CLIP visual projection)
  - audio_model.onnx          (CLAP audio encoder)
  - audio_projection.onnx     (trained CLAP→CLIP projection)
  - clip_config.json           (combined config with vision, text, audio sections)
  - tokenizer.json, preprocessor_config.json, etc.

Usage:
    # Build locally
    uv run scripts/build_clipclap.py --output-dir /tmp/clipclap

    # Build and push to HuggingFace
    uv run scripts/build_clipclap.py --output-dir /tmp/clipclap --push-to-hub antflydb/clipclap
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the antflydb/clipclap unified multimodal model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model ID on HuggingFace (default: openai/clip-vit-base-patch32)",
    )
    parser.add_argument(
        "--clap-model",
        default="laion/larger_clap_music_and_speech",
        help="CLAP model ID on HuggingFace (default: laion/larger_clap_music_and_speech)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the complete model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Projection training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Projection training batch size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Projection training learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--num-captions",
        type=int,
        default=5000,
        help="Number of COCO captions for projection training (default: 5000)",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="HuggingFace repo to push model to (e.g., antflydb/clipclap). Uses HF_TOKEN env var.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Projection training
# ---------------------------------------------------------------------------

def load_captions(num_captions: int) -> list[str]:
    """Load diverse captions from Conceptual Captions dataset."""
    from datasets import load_dataset

    logger.info(f"Loading {num_captions} captions from Conceptual Captions...")
    ds = load_dataset(
        "google-research-datasets/conceptual_captions",
        split="train",
        streaming=True,
    )

    captions: list[str] = []
    seen: set[str] = set()
    for example in ds:
        caption = example.get("caption", "")
        if caption and caption not in seen:
            seen.add(caption)
            captions.append(caption)
            if len(captions) >= num_captions:
                break

    logger.info(f"Collected {len(captions)} unique captions")
    return captions


@torch.no_grad()
def encode_texts_clip(model, tokenizer, texts: list[str], batch_size: int = 64, device: str = "cpu") -> torch.Tensor:
    """Encode texts with CLIP text encoder, return normalized embeddings."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        pooled_output = text_outputs[1]
        text_features = model.text_projection(pooled_output)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        all_embeddings.append(text_features.cpu())
    return torch.cat(all_embeddings, dim=0)


@torch.no_grad()
def encode_texts_clap(model, processor, texts: list[str], batch_size: int = 64, device: str = "cpu") -> torch.Tensor:
    """Encode texts with CLAP text encoder, return normalized embeddings."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        pooled_output = text_outputs[1]
        text_features = model.text_projection(pooled_output)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        all_embeddings.append(text_features.cpu())
    return torch.cat(all_embeddings, dim=0)


def train_projection(
    clip_embeddings: torch.Tensor,
    clap_embeddings: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
) -> nn.Linear:
    """Train a linear projection from CLAP space to CLIP space."""
    assert clip_embeddings.shape == clap_embeddings.shape
    embed_dim = clip_embeddings.shape[1]

    logger.info(f"Training projection: {embed_dim}→{embed_dim}, {len(clip_embeddings)} samples")

    n = len(clip_embeddings)
    n_train = int(n * 0.9)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_dataset = TensorDataset(clap_embeddings[train_idx], clip_embeddings[train_idx])
    val_dataset = TensorDataset(clap_embeddings[val_idx], clip_embeddings[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    projection = nn.Linear(embed_dim, embed_dim, bias=False)
    nn.init.eye_(projection.weight)

    optimizer = optim.Adam(projection.parameters(), lr=lr)
    cos_loss = nn.CosineEmbeddingLoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        projection.train()
        train_losses = []
        for clap_batch, clip_batch in train_loader:
            optimizer.zero_grad()
            projected = projection(clap_batch)
            loss = cos_loss(projected, clip_batch, torch.ones(len(clap_batch)))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        projection.eval()
        val_losses = []
        with torch.no_grad():
            for clap_batch, clip_batch in val_loader:
                projected = projection(clap_batch)
                loss = cos_loss(projected, clip_batch, torch.ones(len(clap_batch)))
                val_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        with torch.no_grad():
            all_proj = projection(clap_embeddings[val_idx])
            cos_sim = nn.functional.cosine_similarity(all_proj, clip_embeddings[val_idx]).mean().item()

        logger.info(
            f"  Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_cos_sim={cos_sim:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in projection.state_dict().items()}

    if best_state is not None:
        projection.load_state_dict(best_state)

    return projection


# ---------------------------------------------------------------------------
# ONNX export helpers
# ---------------------------------------------------------------------------

def export_onnx(module, dummy_input, path: Path, input_names, output_names, dynamic_axes):
    """Export a PyTorch module to ONNX and validate."""
    import onnx

    torch.onnx.export(
        module,
        dummy_input,
        str(path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    logger.info(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast, ClapModel, ClapProcessor

    # ── Step 1: Load source models ────────────────────────────────────────
    logger.info(f"\n[1/5] Loading source models")

    logger.info(f"  Loading CLIP: {args.clip_model}")
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_tokenizer = CLIPTokenizerFast.from_pretrained(args.clip_model)
    clip_model.eval()

    logger.info(f"  Loading CLAP: {args.clap_model}")
    clap_model = ClapModel.from_pretrained(args.clap_model).to(device)
    clap_processor = ClapProcessor.from_pretrained(args.clap_model)
    clap_model.eval()

    clip_dim = clip_model.config.projection_dim
    clap_dim = clap_model.config.projection_dim
    logger.info(f"  CLIP projection dim: {clip_dim}, CLAP projection dim: {clap_dim}")
    if clip_dim != clap_dim:
        logger.error(f"Dimension mismatch: CLIP={clip_dim}, CLAP={clap_dim}")
        sys.exit(1)

    # ── Step 2: Train projection ──────────────────────────────────────────
    logger.info(f"\n[2/5] Training CLAP→CLIP audio projection")

    captions = load_captions(args.num_captions)

    logger.info("  Encoding captions with CLIP text encoder...")
    clip_embeddings = encode_texts_clip(clip_model, clip_tokenizer, captions, device=device)
    logger.info(f"    Shape: {clip_embeddings.shape}")

    logger.info("  Encoding captions with CLAP text encoder...")
    clap_embeddings = encode_texts_clap(clap_model, clap_processor, captions, device=device)
    logger.info(f"    Shape: {clap_embeddings.shape}")

    projection = train_projection(
        clip_embeddings,
        clap_embeddings,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # ── Step 3: Export ONNX models ────────────────────────────────────────
    logger.info(f"\n[3/5] Exporting ONNX models to {out}")

    # Move models to CPU for export
    clip_model = clip_model.cpu()
    clap_model = clap_model.cpu()

    # Image size from processor
    image_size = clip_processor.image_processor.size.get("shortest_edge", 224)
    if isinstance(image_size, dict):
        image_size = image_size.get("height", 224)

    # Visual encoder
    logger.info("  Exporting CLIP visual encoder...")
    export_onnx(
        clip_model.vision_model,
        (torch.randn(1, 3, image_size, image_size),),
        out / "visual_model.onnx",
        input_names=["pixel_values"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
            "pooler_output": {0: "batch_size"},
        },
    )

    # Text encoder
    logger.info("  Exporting CLIP text encoder...")
    text_inputs = clip_tokenizer(
        ["a photo of a cat"], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    export_onnx(
        clip_model.text_model,
        (text_inputs["input_ids"], text_inputs["attention_mask"]),
        out / "text_model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            "pooler_output": {0: "batch_size"},
        },
    )

    # Visual projection
    logger.info("  Exporting CLIP visual projection...")
    export_onnx(
        clip_model.visual_projection,
        torch.randn(1, clip_model.config.vision_config.hidden_size),
        out / "visual_projection.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Text projection
    logger.info("  Exporting CLIP text projection...")
    export_onnx(
        clip_model.text_projection,
        torch.randn(1, clip_model.config.text_config.hidden_size),
        out / "text_projection.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Audio encoder
    logger.info("  Exporting CLAP audio encoder...")
    sample_rate = clap_processor.feature_extractor.sampling_rate
    max_length_s = clap_processor.feature_extractor.max_length_s
    num_samples = int(sample_rate * max_length_s)
    dummy_audio = torch.randn(1, num_samples)
    audio_inputs = clap_processor(
        audio=dummy_audio.numpy(), return_tensors="pt", sampling_rate=sample_rate
    )
    export_onnx(
        clap_model.audio_model,
        (audio_inputs["input_features"],),
        out / "audio_model.onnx",
        input_names=["input_features"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_features": {0: "batch_size", 2: "time"},
            "last_hidden_state": {0: "batch_size"},
            "pooler_output": {0: "batch_size"},
        },
    )

    # Combined audio projection: CLAP audio_projection (1024→512) + trained projection (512→512)
    # The CLAP audio encoder outputs 1024-dim, which needs CLAP's own audio_projection first,
    # then our trained CLAP→CLIP projection. We combine both into a single ONNX model.
    logger.info("  Exporting combined audio projection (1024→512)...")

    class CombinedAudioProjection(nn.Module):
        def __init__(self, clap_audio_proj, trained_proj):
            super().__init__()
            self.clap_audio_proj = clap_audio_proj
            self.trained_proj = trained_proj

        def forward(self, x):
            x = self.clap_audio_proj(x)
            x = self.trained_proj(x)
            return x

    combined_proj = CombinedAudioProjection(clap_model.audio_projection, projection)
    combined_proj.eval()
    clap_audio_hidden = clap_model.config.audio_config.hidden_size
    export_onnx(
        combined_proj,
        torch.randn(1, clap_audio_hidden),
        out / "audio_projection.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # ── Step 4: Save configs and tokenizer ────────────────────────────────
    logger.info(f"\n[4/5] Saving configs and tokenizer")

    clip_processor.save_pretrained(out)
    clip_tokenizer.save_pretrained(out)

    clipclap_config = {
        "model_type": "clipclap",
        "vision_config": {
            "hidden_size": clip_model.config.vision_config.hidden_size,
            "image_size": clip_model.config.vision_config.image_size,
            "patch_size": clip_model.config.vision_config.patch_size,
            "projection_dim": clip_model.config.projection_dim,
        },
        "text_config": {
            "hidden_size": clip_model.config.text_config.hidden_size,
            "max_position_embeddings": clip_model.config.text_config.max_position_embeddings,
            "projection_dim": clip_model.config.projection_dim,
        },
        "audio_config": {
            "hidden_size": clap_model.config.audio_config.hidden_size,
            "sample_rate": sample_rate,
            "max_length_s": max_length_s,
            "projection_dim": clap_model.config.projection_dim,
        },
        "projection_dim": clip_model.config.projection_dim,
    }
    with open(out / "clip_config.json", "w") as f:
        json.dump(clipclap_config, f, indent=2)
    logger.info(f"  Saved clip_config.json")

    # Training metadata
    metadata = {
        "clip_model": args.clip_model,
        "clap_model": args.clap_model,
        "embed_dim": clip_dim,
        "num_captions": len(captions),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    with open(out / "projection_training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved projection_training_metadata.json")

    # ── Step 5: Push to hub ───────────────────────────────────────────────
    if args.push_to_hub:
        logger.info(f"\n[5/5] Pushing to HuggingFace Hub: {args.push_to_hub}")
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        if not token:
            logger.error("HF_TOKEN environment variable not set. Cannot push to hub.")
            sys.exit(1)

        api = HfApi(token=token)
        api.create_repo(args.push_to_hub, exist_ok=True)
        api.upload_folder(
            folder_path=str(out),
            repo_id=args.push_to_hub,
            commit_message="Update CLIPCLAP model with trained audio projection",
        )
        logger.info(f"  Pushed to https://huggingface.co/{args.push_to_hub}")
    else:
        logger.info(f"\n[5/5] Skipping hub push (use --push-to-hub to upload)")

    # ── Done ──────────────────────────────────────────────────────────────
    logger.info(f"\nDone! Model saved to {out}")
    logger.info(f"Files:")
    for p in sorted(out.iterdir()):
        size = p.stat().st_size
        if size > 1024 * 1024:
            logger.info(f"  {p.name:40s} {size / 1024 / 1024:.1f} MB")
        else:
            logger.info(f"  {p.name:40s} {size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
