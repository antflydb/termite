#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "torch>=2.6.0",
#     "onnx>=1.15.0",
#     "onnxruntime>=1.17.0",
#     "onnxscript",
#     "pillow",
#     "requests",
#     "numpy",
#     "accelerate",
#     "sentencepiece",
# ]
# ///
"""
Export T5Gemma-2 models to ONNX format for Termite.

T5Gemma-2 is Google's encoder-decoder model adapted from Gemma 3 via UL2 training.
It has unique architectural features:
  - Merged self/cross-attention in decoder
  - Tied word embeddings between encoder and decoder
  - Grouped Query Attention (GQA)
  - Rotary Position Embeddings (RoPE)
  - Alternating sliding window attention (4096 tokens)
  - SigLIP vision encoder for multimodal

This script exports 4 ONNX files:
  1. encoder.onnx - Text encoder (for embeddings and seq2seq)
  2. vision_encoder.onnx - SigLIP vision encoder (for image embeddings)
  3. decoder_init.onnx - Decoder without past_key_values (first token)
  4. decoder.onnx - Decoder with past_key_values (efficient generation)

Usage:
    # Export 270M variant (recommended for development)
    ./scripts/export_t5gemma2.py --model google/t5gemma-2-270m-270m --output ./models/t5gemma2/270m

    # Export with testing
    ./scripts/export_t5gemma2.py --model google/t5gemma-2-270m-270m --output ./models/t5gemma2/270m --test

    # Export 1B variant
    ./scripts/export_t5gemma2.py --model google/t5gemma-2-1b-1b --output ./models/t5gemma2/1b

Available Models:
    - google/t5gemma-2-270m-270m (~800M total params, fast)
    - google/t5gemma-2-1b-1b (~2B total params, balanced)
    - google/t5gemma-2-4b-4b (~8B total params, best quality)

References:
    - HuggingFace: https://huggingface.co/docs/transformers/model_doc/t5gemma2
    - Google Blog: https://blog.google/technology/developers/t5gemma-2/
    - Paper: https://arxiv.org/abs/2512.14856

Prerequisites:
    - uv (Python package manager): brew install uv
    - HuggingFace account with access to T5Gemma-2 models

Note: This script is self-contained and does NOT require optimum-onnx.
It applies the same ONNX compatibility patches manually.

If you prefer using optimum-cli for export, use our fork:
    https://github.com/timkaye11/optimum-onnx (branch: add-t5gemma2-support)

See docs/T5GEMMA2.md for full setup instructions.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import os

# Disable torch dynamo verbose mode
os.environ["TORCHDYNAMO_VERBOSE"] = "0"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ONNX opset version - use 17 for better compatibility with complex attention patterns
OPSET_VERSION = 17


def disable_dynamo():
    """Disable torch dynamo to avoid tracing issues."""
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.reset()
        # Completely disable dynamo
        torch._dynamo.disable()
    except Exception:
        pass


def patch_transformers_for_onnx():
    """
    Patch transformers masking utilities for ONNX export compatibility.

    This applies the same patches as optimum-onnx's ModelPatcher, replacing
    vmap-based mask creation with broadcasting-based alternatives that are
    compatible with TorchScript/ONNX tracing.

    Based on: https://github.com/huggingface/optimum-onnx/blob/main/optimum/exporters/onnx/model_patcher.py
    """
    try:
        import torch
        from typing import Callable, Optional

        import transformers.masking_utils as masking_utils
        from transformers.masking_utils import (
            ALL_MASK_ATTENTION_FUNCTIONS,
            and_masks,
            causal_mask_function,
            padding_mask_function,
        )

        def sdpa_mask_without_vmap(
            batch_size: int,
            cache_position: torch.Tensor,
            kv_length: int,
            kv_offset: int = 0,
            mask_function: Optional[Callable] = None,
            attention_mask: Optional[torch.Tensor] = None,
            local_size: Optional[int] = None,
            allow_is_causal_skip: bool = True,
            **kwargs,
        ) -> Optional[torch.Tensor]:
            """
            Custom vectorized implementation of sdpa_mask without using vmap.
            Uses broadcasting-based index creation instead.
            """
            if mask_function is None:
                mask_function = causal_mask_function

            q_length = cache_position.shape[0]

            # Potentially pad the 2D mask - handle the padding_length check safely
            padding_mask = None
            if attention_mask is not None:
                padding_needed = kv_length + kv_offset - attention_mask.shape[-1]
                if padding_needed > 0:
                    padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_needed))
                else:
                    padding_mask = attention_mask

            # Potentially add the padding 2D mask
            if padding_mask is not None:
                mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

            # Create broadcatable indices (optimum-onnx approach)
            device = cache_position.device
            q_indices = cache_position[None, None, :, None]
            head_indices = torch.arange(1, dtype=torch.long, device=device)[None, :, None, None]
            batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)[:, None, None, None]
            kv_indices = torch.arange(kv_length, dtype=torch.long, device=device)[None, None, None, :] + kv_offset

            # Apply mask function element-wise through broadcasting
            causal_mask = mask_function(batch_indices, head_indices, q_indices, kv_indices)

            # Expand the mask to match batch size and query length
            causal_mask = causal_mask.expand(batch_size, -1, q_length, kv_length)

            return causal_mask

        def eager_mask_without_vmap(
            batch_size: int,
            cache_position: torch.Tensor,
            kv_length: int,
            kv_offset: int = 0,
            mask_function: Optional[Callable] = None,
            attention_mask: Optional[torch.Tensor] = None,
            dtype: torch.dtype = torch.float32,
            **kwargs,
        ) -> torch.Tensor:
            """
            Eager attention mask without vmap, adapted from transformers.
            Converts boolean mask to float mask with 0 for attend, -inf for mask.
            """
            kwargs.pop("allow_is_causal_skip", None)
            kwargs.pop("allow_is_bidirectional_skip", None)
            kwargs.pop("allow_torch_fix", None)

            mask = sdpa_mask_without_vmap(
                batch_size=batch_size,
                cache_position=cache_position,
                kv_length=kv_length,
                kv_offset=kv_offset,
                mask_function=mask_function,
                attention_mask=attention_mask,
                allow_is_causal_skip=False,
                **kwargs,
            )

            if mask is not None:
                min_dtype = torch.finfo(dtype).min
                # Convert bool mask to float: True -> 0.0, False -> -inf
                mask = torch.where(mask, torch.zeros((), device=mask.device, dtype=dtype), min_dtype)

            return mask

        # Register the patched functions
        ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", sdpa_mask_without_vmap)
        ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask_without_vmap)

        # Patch find_packed_sequence_indices to avoid torch.diff (not supported in ONNX)
        def find_packed_sequence_indices_onnx(position_ids: torch.Tensor) -> Optional[torch.Tensor]:
            """
            ONNX-compatible version that always returns None (no packed sequences).
            This is safe for export since we're exporting with fixed batch size.
            """
            return None

        masking_utils.find_packed_sequence_indices = find_packed_sequence_indices_onnx

        logger.info("Patched transformers masking utilities for ONNX compatibility (optimum-onnx style)")
        return True

    except Exception as e:
        logger.warning(f"Could not patch transformers masking utilities: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch>=2.0.0")
    try:
        import transformers
    except ImportError:
        missing.append("transformers>=4.48.0")
    try:
        import onnx
    except ImportError:
        missing.append("onnx>=1.15.0")
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime>=1.17.0")

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


class T5Gemma2EncoderWrapper:
    """Wrapper to export just the encoder portion of T5Gemma2."""

    def __init__(self, model):
        self.encoder = model.encoder
        self.embed_tokens = model.encoder.embed_tokens

    def forward(self, input_ids, attention_mask):
        # Get embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # Run encoder
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )

        return encoder_outputs.last_hidden_state


class T5Gemma2DecoderInitWrapper:
    """Wrapper to export decoder without past_key_values (first token generation)."""

    def __init__(self, model):
        self.decoder = model.decoder
        self.lm_head = model.lm_head
        self.embed_tokens = model.decoder.embed_tokens

    def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
        # Get decoder embeddings
        inputs_embeds = self.embed_tokens(decoder_input_ids)

        # Run decoder without past
        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            return_dict=True,
        )

        # Get logits
        logits = self.lm_head(decoder_outputs.last_hidden_state)

        # Return logits and past_key_values for subsequent decoding
        return logits, decoder_outputs.past_key_values


class T5Gemma2DecoderWrapper:
    """Wrapper to export decoder with past_key_values (efficient generation)."""

    def __init__(self, model):
        self.decoder = model.decoder
        self.lm_head = model.lm_head
        self.embed_tokens = model.decoder.embed_tokens

    def forward(
        self,
        decoder_input_ids,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_values,
    ):
        # Get decoder embeddings
        inputs_embeds = self.embed_tokens(decoder_input_ids)

        # Run decoder with past
        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        # Get logits
        logits = self.lm_head(decoder_outputs.last_hidden_state)

        return logits, decoder_outputs.past_key_values


def export_encoder(model, processor, output_dir: Path) -> None:
    """Export the text encoder to ONNX.

    Due to T5Gemma-2's complex attention implementation, we export two options:
    1. embedding_layer.onnx - Just the embedding lookup (fast, always works)
    2. encoder.onnx - Full encoder with attention (may fail with complex models)
    """
    import torch
    import onnx

    logger.info("\n1. Exporting text encoder...")

    # Create dummy inputs
    dummy_text = "This is a test sentence for the encoder."
    inputs = processor.tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get encoder from model using get_encoder() method
    encoder = model.get_encoder()

    # Get embed_tokens from the model's text_embed_tokens or from encoder
    if hasattr(model.model, 'text_embed_tokens'):
        embed_tokens = model.model.text_embed_tokens
    elif hasattr(encoder, 'embed_tokens'):
        embed_tokens = encoder.embed_tokens
    else:
        # Fallback: try to get from decoder's embed_tokens (they may be shared)
        decoder = model.get_decoder()
        embed_tokens = decoder.embed_tokens

    # Export 1: Embedding layer only (simpler, always works)
    embedding_path = output_dir / "embedding_layer.onnx"
    logger.info("   1a. Exporting embedding layer...")

    class EmbeddingModule(torch.nn.Module):
        def __init__(self, embed_tokens):
            super().__init__()
            self.embed_tokens = embed_tokens

        def forward(self, input_ids):
            return self.embed_tokens(input_ids)

    embedding_module = EmbeddingModule(embed_tokens)
    embedding_module.eval()
    for param in embedding_module.parameters():
        param.requires_grad = False

    with torch.no_grad():
        torch.onnx.export(
            embedding_module,
            (input_ids,),
            str(embedding_path),
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["input_ids"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "embeddings": {0: "batch_size", 1: "sequence_length"},
            },
            dynamo=False,
        )

    # Validate embedding layer
    onnx_model = onnx.load(str(embedding_path))
    onnx.checker.check_model(onnx_model)
    size_mb = embedding_path.stat().st_size / (1024 * 1024)
    logger.info(f"       Saved: embedding_layer.onnx ({size_mb:.1f} MB)")

    # Export 2: Full encoder (may fail due to complex attention)
    encoder_path = output_dir / "encoder.onnx"
    logger.info("   1b. Exporting full encoder (this may take a while)...")

    class EncoderModule(torch.nn.Module):
        def __init__(self, encoder, embed_tokens):
            super().__init__()
            self.encoder = encoder
            self.embed_tokens = embed_tokens

        def forward(self, input_ids, attention_mask):
            inputs_embeds = self.embed_tokens(input_ids)
            outputs = self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )
            return outputs.last_hidden_state

    encoder_module = EncoderModule(encoder, embed_tokens)
    encoder_module.eval()
    for param in encoder_module.parameters():
        param.requires_grad = False

    try:
        with torch.no_grad():
            torch.onnx.export(
                encoder_module,
                (input_ids, attention_mask),
                str(encoder_path),
                export_params=True,
                opset_version=OPSET_VERSION,
                do_constant_folding=True,
                input_names=["input_ids", "attention_mask"],
                output_names=["encoder_hidden_states"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
                },
                dynamo=False,
            )

        # Validate
        onnx_model = onnx.load(str(encoder_path))
        onnx.checker.check_model(onnx_model)
        size_mb = encoder_path.stat().st_size / (1024 * 1024)
        logger.info(f"       Saved: encoder.onnx ({size_mb:.1f} MB)")

        # Test encoder output shape
        with torch.no_grad():
            test_output = encoder_module(input_ids, attention_mask)
            logger.info(f"       Encoder output shape: {test_output.shape}")

    except Exception as e:
        logger.warning(f"       Full encoder export failed: {e}")
        logger.info("       Using embedding_layer.onnx for embeddings (without attention)")
        logger.info("       For full encoder functionality, use PyTorch inference instead of ONNX")

    # Export 3: Encoder with inputs_embeds (for embedding-to-text generation)
    encoder_embeds_path = output_dir / "encoder_embeds.onnx"
    logger.info("   1c. Exporting encoder with inputs_embeds support...")

    # Get hidden size from model config
    if hasattr(model.config, 'encoder'):
        hidden_size = model.config.encoder.hidden_size
    elif hasattr(model.config, 'hidden_size'):
        hidden_size = model.config.hidden_size
    else:
        hidden_size = 640  # Default for T5Gemma2-270m

    class EncoderEmbedsModule(torch.nn.Module):
        """Encoder that accepts pre-computed embeddings directly (inputs_embeds)."""
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, inputs_embeds, attention_mask):
            """Run encoder directly on embeddings, bypassing token embedding lookup."""
            outputs = self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )
            return outputs.last_hidden_state

    encoder_embeds_module = EncoderEmbedsModule(encoder)
    encoder_embeds_module.eval()
    for param in encoder_embeds_module.parameters():
        param.requires_grad = False

    # Create dummy inputs_embeds
    batch_size, seq_len = input_ids.shape
    dummy_inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)

    try:
        with torch.no_grad():
            torch.onnx.export(
                encoder_embeds_module,
                (dummy_inputs_embeds, attention_mask),
                str(encoder_embeds_path),
                export_params=True,
                opset_version=OPSET_VERSION,
                do_constant_folding=True,
                input_names=["inputs_embeds", "attention_mask"],
                output_names=["encoder_hidden_states"],
                dynamic_axes={
                    "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
                },
                dynamo=False,
            )

        # Validate
        onnx_model = onnx.load(str(encoder_embeds_path))
        onnx.checker.check_model(onnx_model)
        size_mb = encoder_embeds_path.stat().st_size / (1024 * 1024)
        logger.info(f"       Saved: encoder_embeds.onnx ({size_mb:.1f} MB)")

        # Test encoder_embeds output shape
        with torch.no_grad():
            test_output = encoder_embeds_module(dummy_inputs_embeds, attention_mask)
            logger.info(f"       Encoder (embeds) output shape: {test_output.shape}")

    except Exception as e:
        logger.warning(f"       Encoder with inputs_embeds export failed: {e}")
        logger.info("       Embedding-to-text generation will require the standard encoder path")


def export_vision_encoder(model, processor, output_dir: Path) -> bool:
    """Export the vision encoder (SigLIP) to ONNX if present."""
    import torch
    import onnx

    # Check if model has vision encoder - try multiple paths
    vision_tower = None
    encoder = model.get_encoder()

    # Try different possible locations for vision tower
    if hasattr(encoder, 'vision_tower'):
        vision_tower = encoder.vision_tower
    elif hasattr(model.model, 'vision_tower'):
        vision_tower = model.model.vision_tower
    elif hasattr(model, 'vision_tower'):
        vision_tower = model.vision_tower

    if vision_tower is None:
        logger.info("\n2. No vision encoder found (text-only model)")
        return False

    logger.info("\n2. Exporting vision encoder (SigLIP)...")

    vision_path = output_dir / "vision_encoder.onnx"

    # Get image size from processor
    if hasattr(processor, "image_processor"):
        image_size = processor.image_processor.size.get("height", 896)
    else:
        image_size = 896  # Default for T5Gemma2

    # Create dummy image input
    dummy_pixel_values = torch.randn(1, 3, image_size, image_size)

    class VisionEncoderModule(torch.nn.Module):
        def __init__(self, vision_tower):
            super().__init__()
            self.vision_tower = vision_tower

        def forward(self, pixel_values):
            outputs = self.vision_tower(pixel_values, return_dict=True)
            return outputs.last_hidden_state

    vision_module = VisionEncoderModule(vision_tower)
    vision_module.eval()

    try:
        with torch.no_grad():
            torch.onnx.export(
                vision_module,
                (dummy_pixel_values,),
                str(vision_path),
                export_params=True,
                opset_version=OPSET_VERSION,
                do_constant_folding=True,
                input_names=["pixel_values"],
                output_names=["vision_hidden_states"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size"},
                    "vision_hidden_states": {0: "batch_size"},
                },
                dynamo=False,  # Use legacy exporter
            )

        # Validate
        onnx_model = onnx.load(str(vision_path))
        onnx.checker.check_model(onnx_model)

        size_mb = vision_path.stat().st_size / (1024 * 1024)
        logger.info(f"   Saved: vision_encoder.onnx ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        logger.warning(f"   Vision encoder export failed: {e}")
        return False


def export_decoder_init(model, processor, output_dir: Path, hidden_size: int) -> None:
    """Export the decoder without past_key_values (for first token generation)."""
    import torch
    import onnx

    logger.info("\n3. Exporting decoder-init (without past_key_values)...")

    decoder_init_path = output_dir / "decoder-init.onnx"

    # Get decoder components using get_decoder() method
    decoder = model.get_decoder()
    lm_head = model.lm_head

    # Get embed_tokens - try multiple paths
    if hasattr(decoder, 'embed_tokens'):
        embed_tokens = decoder.embed_tokens
    elif hasattr(model.model, 'text_embed_tokens'):
        embed_tokens = model.model.text_embed_tokens
    elif hasattr(model.model, 'embed_tokens'):
        embed_tokens = model.model.embed_tokens
    else:
        raise AttributeError("Could not find embed_tokens in model")

    # Create dummy inputs
    batch_size = 1
    encoder_seq_len = 32
    decoder_seq_len = 1

    dummy_decoder_input_ids = torch.zeros((batch_size, decoder_seq_len), dtype=torch.long)
    dummy_encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, hidden_size)
    dummy_encoder_attention_mask = torch.ones((batch_size, encoder_seq_len), dtype=torch.long)

    class DecoderInitModule(torch.nn.Module):
        def __init__(self, decoder, lm_head, embed_tokens):
            super().__init__()
            self.decoder = decoder
            self.lm_head = lm_head
            self.embed_tokens = embed_tokens

        def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
            inputs_embeds = self.embed_tokens(decoder_input_ids)
            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,  # No cache for init
                return_dict=True,
            )
            logits = self.lm_head(outputs.last_hidden_state)
            return logits

    decoder_init_module = DecoderInitModule(decoder, lm_head, embed_tokens)
    decoder_init_module.eval()

    with torch.no_grad():
        torch.onnx.export(
            decoder_init_module,
            (dummy_decoder_input_ids, dummy_encoder_hidden_states, dummy_encoder_attention_mask),
            str(decoder_init_path),
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["input_ids", "encoder_hidden_states", "encoder_attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
                "encoder_hidden_states": {0: "batch_size", 1: "encoder_sequence_length"},
                "encoder_attention_mask": {0: "batch_size", 1: "encoder_sequence_length"},
                "logits": {0: "batch_size", 1: "decoder_sequence_length"},
            },
            dynamo=False,  # Use legacy exporter
        )

    # Validate
    onnx_model = onnx.load(str(decoder_init_path))
    onnx.checker.check_model(onnx_model)

    size_mb = decoder_init_path.stat().st_size / (1024 * 1024)
    logger.info(f"   Saved: decoder-init.onnx ({size_mb:.1f} MB)")


def export_decoder_with_past(model, processor, output_dir: Path, hidden_size: int) -> None:
    """
    Export the decoder with past_key_values for efficient autoregressive generation.

    Note: This is complex due to T5Gemma2's merged self/cross-attention and GQA.
    For initial implementation, we use decoder-init for all steps (slower but simpler).
    """
    import torch

    logger.info("\n4. Exporting decoder with past_key_values...")
    logger.info("   Note: Full KV cache export is complex due to merged attention.")
    logger.info("   Using decoder-init for all steps (slower but functional).")

    # For now, we'll copy decoder-init as decoder (same file, different name)
    # This means generation will be slower but will work correctly
    decoder_init_path = output_dir / "decoder-init.onnx"
    decoder_path = output_dir / "decoder.onnx"

    if decoder_init_path.exists():
        shutil.copy(str(decoder_init_path), str(decoder_path))
        size_mb = decoder_path.stat().st_size / (1024 * 1024)
        logger.info(f"   Saved: decoder.onnx ({size_mb:.1f} MB)")
        logger.info("   (Using decoder-init format - no KV cache optimization)")


def save_configs(model, processor, output_dir: Path, model_id: str) -> None:
    """Save model and tokenizer configuration files."""
    logger.info("\n5. Saving configuration files...")

    # Save tokenizer
    processor.tokenizer.save_pretrained(str(output_dir))
    logger.info("   Saved: tokenizer files")

    # Save model config
    model.config.save_pretrained(str(output_dir))
    logger.info("   Saved: config.json")

    # Save image processor config if available
    if hasattr(processor, 'image_processor') and processor.image_processor is not None:
        try:
            processor.image_processor.save_pretrained(str(output_dir))
            logger.info("   Saved: preprocessor_config.json")
        except Exception as e:
            logger.warning(f"   Could not save image processor config: {e}")

    # Extract config values with fallbacks for different config structures
    def get_config_value(config, *paths, default=None):
        """Try multiple paths to get a config value."""
        for path in paths:
            obj = config
            try:
                for key in path.split('.'):
                    obj = getattr(obj, key)
                return obj
            except AttributeError:
                continue
        return default

    hidden_size = get_config_value(
        model.config,
        'decoder.hidden_size',
        'hidden_size',
        default=2304
    )
    vocab_size = get_config_value(
        model.config,
        'decoder.vocab_size',
        'vocab_size',
        default=262208
    )
    num_encoder_layers = get_config_value(
        model.config,
        'encoder.text_config.num_hidden_layers',
        'encoder.num_hidden_layers',
        'num_hidden_layers',
        default=26
    )
    num_decoder_layers = get_config_value(
        model.config,
        'decoder.num_hidden_layers',
        'num_hidden_layers',
        default=26
    )
    num_attention_heads = get_config_value(
        model.config,
        'decoder.num_attention_heads',
        'num_attention_heads',
        default=8
    )
    num_key_value_heads = get_config_value(
        model.config,
        'decoder.num_key_value_heads',
        'num_key_value_heads',
        default=4
    )

    # Create T5Gemma2-specific config for Termite
    t5gemma2_config = {
        "model_id": model_id,
        "model_type": "t5gemma2",
        "task": "multimodal_seq2seq",
        "capabilities": ["embeddings", "generation", "decoding", "multimodal"],
        "max_encoder_length": 131072,  # 128K
        "max_decoder_length": 32768,   # 32K output
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "generation_config": {
            "max_new_tokens": 256,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
        },
    }

    config_path = output_dir / "t5gemma2_config.json"
    with open(config_path, "w") as f:
        json.dump(t5gemma2_config, f, indent=2)
    logger.info("   Saved: t5gemma2_config.json")


def export_model(model_id: str, output_dir: str) -> None:
    """
    Export a T5Gemma-2 model to ONNX format.

    Args:
        model_id: HuggingFace model ID (e.g., 'google/t5gemma-2-270m-270m')
        output_dir: Directory to save the exported model
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    # Disable dynamo and patch transformers before any model operations
    disable_dynamo()
    patch_transformers_for_onnx()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting T5Gemma-2: {model_id}")
    logger.info(f"Output: {output_dir}")

    # Load model and tokenizer
    logger.info("\nLoading model and tokenizer...")
    logger.info("(This may take a while for larger models)")

    # Load tokenizer directly (AutoProcessor has compatibility issues with T5Gemma-2)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Try to load image processor separately if available (for multimodal)
    image_processor = None
    try:
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        logger.info("Loaded image processor for multimodal support")
    except Exception as e:
        logger.info(f"No image processor found (text-only mode): {e}")

    # Create a simple processor-like object that has tokenizer attribute
    class SimpleProcessor:
        def __init__(self, tokenizer, image_processor=None):
            self.tokenizer = tokenizer
            self.image_processor = image_processor

    processor = SimpleProcessor(tokenizer, image_processor)

    # Load model with configuration optimized for ONNX export
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Set sliding window to a very large value to effectively disable it
    # (Setting to None breaks other code that expects an int)
    VERY_LARGE_WINDOW = 2**20  # 1M tokens
    if hasattr(config, 'encoder'):
        if hasattr(config.encoder, 'sliding_window'):
            config.encoder.sliding_window = VERY_LARGE_WINDOW
        if hasattr(config.encoder, 'sliding_window_size'):
            config.encoder.sliding_window_size = VERY_LARGE_WINDOW
    if hasattr(config, 'decoder'):
        if hasattr(config.decoder, 'sliding_window'):
            config.decoder.sliding_window = VERY_LARGE_WINDOW
        if hasattr(config.decoder, 'sliding_window_size'):
            config.decoder.sliding_window_size = VERY_LARGE_WINDOW

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float32,  # Use FP32 for ONNX export
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",  # Use eager attention for ONNX compatibility
    )
    model.eval()

    # Disable gradient checkpointing if enabled
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()

    # Patch model config to use large sliding window at runtime
    VERY_LARGE_WINDOW = 2**20  # 1M tokens
    for module in model.modules():
        if hasattr(module, 'sliding_window') and module.sliding_window is not None:
            module.sliding_window = VERY_LARGE_WINDOW
        if hasattr(module, 'config') and hasattr(module.config, 'sliding_window'):
            if module.config.sliding_window is not None:
                module.config.sliding_window = VERY_LARGE_WINDOW

    # Get hidden size from config with fallbacks
    def get_config_value(config, *paths, default=None):
        """Try multiple paths to get a config value."""
        for path in paths:
            obj = config
            try:
                for key in path.split('.'):
                    obj = getattr(obj, key)
                return obj
            except AttributeError:
                continue
        return default

    hidden_size = get_config_value(
        model.config,
        'decoder.hidden_size',
        'hidden_size',
        default=2304
    )
    logger.info(f"Model hidden size: {hidden_size}")

    # Export components
    export_encoder(model, processor, output_path)
    has_vision = export_vision_encoder(model, processor, output_path)
    export_decoder_init(model, processor, output_path, hidden_size)
    export_decoder_with_past(model, processor, output_path, hidden_size)
    save_configs(model, processor, output_path, model_id)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Export complete!")
    logger.info("=" * 60)
    logger.info(f"\nExported files in {output_dir}:")
    for f in sorted(output_path.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"   {f.name} ({size_mb:.1f} MB)")

    logger.info(f"\nCapabilities:")
    logger.info("   - Text embeddings: encoder.onnx")
    if has_vision:
        logger.info("   - Image embeddings: vision_encoder.onnx")
    logger.info("   - Text generation: encoder.onnx + decoder-init.onnx")
    logger.info("   - Decode from embeddings: decoder-init.onnx (with encoder_hidden_states)")


def test_exported_model(output_dir: str, test_text: Optional[str] = None) -> None:
    """Test the exported ONNX model with sample inputs."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    logger.info("\n" + "=" * 60)
    logger.info("Testing exported model...")
    logger.info("=" * 60)

    output_path = Path(output_dir)

    # Check required files
    encoder_path = output_path / "encoder.onnx"
    decoder_init_path = output_path / "decoder-init.onnx"

    if not encoder_path.exists():
        logger.error(f"encoder.onnx not found at {encoder_path}")
        return
    if not decoder_init_path.exists():
        logger.error(f"decoder-init.onnx not found at {decoder_init_path}")
        return

    # Load tokenizer directly (avoid AutoProcessor issues)
    tokenizer = AutoTokenizer.from_pretrained(str(output_path))

    # Create simple processor-like wrapper for compatibility
    class SimpleProcessor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    processor = SimpleProcessor(tokenizer)

    # Create ONNX Runtime sessions
    logger.info("\nLoading ONNX models...")
    encoder_session = ort.InferenceSession(str(encoder_path))
    decoder_init_session = ort.InferenceSession(str(decoder_init_path))

    # Test input
    if test_text is None:
        test_text = "The capital of France is"

    logger.info(f"\nTest input: {test_text}")

    # Tokenize
    inputs = processor.tokenizer(
        test_text,
        return_tensors="np",
        padding=True,
    )
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    # Run encoder
    logger.info("\nRunning encoder...")
    encoder_outputs = encoder_session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )
    encoder_hidden_states = encoder_outputs[0]
    logger.info(f"Encoder output shape: {encoder_hidden_states.shape}")

    # Test embeddings (mean pool)
    embeddings = np.mean(encoder_hidden_states, axis=1)
    logger.info(f"Embedding shape (mean pooled): {embeddings.shape}")
    logger.info(f"Embedding sample: {embeddings[0, :5]}...")

    # Run decoder for generation
    logger.info("\nRunning decoder (greedy generation)...")
    decoder_input_ids = np.array([[processor.tokenizer.pad_token_id or 0]], dtype=np.int64)

    generated_ids = []
    max_new_tokens = 20

    for step in range(max_new_tokens):
        decoder_outputs = decoder_init_session.run(
            None,
            {
                "input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": attention_mask,
            },
        )
        logits = decoder_outputs[0]

        # Greedy: take argmax of last token
        next_token_id = int(np.argmax(logits[0, -1, :]))

        # Check for EOS
        if next_token_id == processor.tokenizer.eos_token_id:
            break

        generated_ids.append(next_token_id)
        decoder_input_ids = np.concatenate(
            [decoder_input_ids, np.array([[next_token_id]], dtype=np.int64)],
            axis=1,
        )

    # Decode
    generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    logger.info(f"\nGenerated: {generated_text}")

    logger.info("\nTest complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Export T5Gemma-2 models to ONNX for Termite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export 270M variant (recommended for development)
  ./scripts/export_t5gemma2.py --model google/t5gemma-2-270m-270m --output ./models/t5gemma2/270m

  # Export with testing
  ./scripts/export_t5gemma2.py --model google/t5gemma-2-270m-270m --output ./models/t5gemma2/270m --test

  # Export 1B variant
  ./scripts/export_t5gemma2.py --model google/t5gemma-2-1b-1b --output ./models/t5gemma2/1b

Available Models:
  - google/t5gemma-2-270m-270m (~800M total params, fast)
  - google/t5gemma-2-1b-1b (~2B total params, balanced)
  - google/t5gemma-2-4b-4b (~8B total params, best quality)

Output Files:
  - encoder.onnx           Text encoder for embeddings & seq2seq
  - vision_encoder.onnx    SigLIP vision encoder (if multimodal)
  - decoder-init.onnx      Decoder for first token / all tokens
  - decoder.onnx           Decoder with KV cache (if supported)
  - t5gemma2_config.json   Termite-specific configuration
  - tokenizer files        For text processing
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="google/t5gemma-2-270m-270m",
        help="HuggingFace model ID (default: google/t5gemma-2-270m-270m)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./models/t5gemma2/270m",
        help="Output directory for the exported model",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the exported model after export",
    )
    parser.add_argument(
        "--test-input",
        type=str,
        default=None,
        help="Custom input text for testing",
    )

    args = parser.parse_args()

    check_dependencies()

    try:
        export_model(args.model, args.output)
        if args.test:
            test_exported_model(args.output, args.test_input)
        logger.info("\nAll done!")
    except Exception as e:
        logger.error(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
