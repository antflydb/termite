"""Moondream VLM model exporter.

Moondream is a vision-language model with:
- SigLIP vision encoder (378x378 images)
- Projection MLP (vision features -> text embedding space)
- Phi-2 text decoder (1.8B params)

The model uses a custom HuggingFace architecture that requires separate export of each component.

Requirements:
- torch, torchvision, einops, pillow
- transformers>=4.40,<4.50

Note: This exporter uses the '2024-08-26' revision of moondream2 which is compatible
with standard Python environments. Later revisions require pyvips.
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

from . import register_exporter
from .base import BaseExporter

logger = logging.getLogger(__name__)

# Use this revision for compatibility (doesn't require pyvips)
MOONDREAM_REVISION = "2024-08-26"


def _patch_transformers_compatibility():
    """Patch transformers PreTrainedModel for compatibility with Moondream's custom code.

    Newer transformers versions expect 'all_tied_weights_keys' attribute which
    Moondream's HfMoondream class doesn't provide.
    """
    from transformers import PreTrainedModel

    original_init = PreTrainedModel.__init__

    def patched_init(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        if not hasattr(self, 'all_tied_weights_keys'):
            self.all_tied_weights_keys = set()

    PreTrainedModel.__init__ = patched_init


def detect_moondream_version(model_id: str) -> str:
    """Detect Moondream version from model ID.

    Returns:
        Version string: "2" or "3"
    """
    model_id_lower = model_id.lower()
    if "moondream3" in model_id_lower or "moondream-3" in model_id_lower:
        return "3"
    return "2"


@register_exporter("reader")
class MoondreamExporter(BaseExporter):
    """Exporter for Moondream VLM models (exported as Reader type).

    Exports three ONNX files:
      - vision_encoder.onnx: SigLIP vision encoder
      - projection.onnx: Vision-to-text projection MLP
      - decoder_model.onnx: Phi-2 decoder (without KV-cache for simplicity)

    Model structure (for vikhyatk/moondream2):
      - model.model.vision: Vision encoder (SigLIP)
        - patch_emb: Patch embedding linear
        - pos_emb: Positional embedding parameter
        - blocks: Transformer blocks
        - post_ln: Post layer norm
        - proj_mlp: Projection MLP (fc1, fc2)
      - model.model.text: Text decoder (Phi-2)
        - wte: Word token embedding
        - blocks: Transformer blocks
        - post_ln: Post layer norm
        - lm_head: Language model head

    Note: Uses revision '2024-08-26' for compatibility (later versions require pyvips).
    """

    def __init__(
        self,
        model_id: str,
        output_dir: Path,
        variants: list[str] | None = None,
        trust_remote_code: bool = True,
        revision: str = MOONDREAM_REVISION,
    ):
        super().__init__(model_id, output_dir, variants)
        self.trust_remote_code = trust_remote_code
        self.revision = revision

    def export(self) -> Path:
        from transformers import AutoModelForCausalLM

        # Apply compatibility patch before loading
        _patch_transformers_compatibility()

        logger.info(f"Exporting Moondream VLM model: {self.model_id}")
        logger.info(f"Using revision: {self.revision}")
        logger.info(f"Output: {self.output_dir}")

        # Detect version
        version = detect_moondream_version(self.model_id)
        logger.info(f"Detected Moondream version: {version}")

        # Load the model with trust_remote_code (required for moondream)
        # Keep in float16 for memory efficiency - individual export functions
        # handle type conversions as needed
        logger.info("Loading Moondream model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.float16,
            revision=self.revision,
            low_cpu_mem_usage=True,
        )
        model.eval()

        # Detect model structure based on revision
        # 2024-08-26 revision: Moondream class with vision_encoder, text_model
        # 2025-01-09+ revision: HfMoondream class with model.vision, model.text
        logger.info(f"Model class: {type(model).__name__}")

        if hasattr(model, 'model') and hasattr(model.model, 'vision'):
            # Newer revision (2025-01-09+)
            inner_model = model.model
            vision_encoder = inner_model.vision
            projection = inner_model.vision.proj_mlp
            text_model = inner_model.text
            tokenizer = inner_model.tokenizer
            model_structure = "new"
        elif hasattr(model, 'vision_encoder'):
            # Older revision (2024-08-26)
            vision_encoder = model.vision_encoder.encoder
            projection = model.vision_encoder.projection
            text_model = model.text_model
            tokenizer = None  # Will use HF tokenizer
            model_structure = "old"
        else:
            raise ValueError(f"Unknown model structure for {type(model).__name__}")

        logger.info(f"Detected model structure: {model_structure}")

        # Export vision encoder
        logger.info("Exporting vision encoder...")
        self._export_vision_encoder_v2(vision_encoder, model_structure)

        # Export projection layer
        logger.info("Exporting projection layer...")
        self._export_projection_v2(projection, model_structure)

        # Export decoder
        logger.info("Exporting decoder...")
        self._export_decoder_v2(text_model, model_structure)

        # Save tokenizer
        logger.info("Saving tokenizer...")
        self._save_tokenizer_v2(model, tokenizer, model_structure)

        # Create image processor config
        logger.info("Creating image processor config...")
        self._create_image_processor_config(version)

        # Create config.json (HuggingFace-style model config)
        # This is required by the VLM pipeline for model configuration
        logger.info("Creating config.json...")
        config = {
            "model_type": "moondream",
            "architectures": ["MoondreamForConditionalGeneration"],
            # Text decoder (Phi-2) config
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "intermediate_size": 8192,
            "vocab_size": 51200,
            # Vision encoder (SigLIP) config
            "vision_hidden_size": 1152,
            "image_size": 378,
            "patch_size": 14,
            "num_channels": 3,
            # Token IDs (Phi-2 uses GPT-2 tokenizer)
            "bos_token_id": 50256,
            "eos_token_id": 50256,
            "pad_token_id": 50256,
            # Generation config
            "max_length": 2048,
        }

        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Create termite_metadata.json
        logger.info("Creating termite_metadata.json...")
        metadata = {
            "model_type": "moondream",
            "version": version,
            "source_model": self.model_id,
            "source_revision": self.revision,
            "export_format": "onnx",
            "framework": "pytorch",
            "architecture": "vision-language",
            "components": {
                "vision_encoder": "siglip",
                "decoder": "phi-2",
            },
        }

        metadata_path = self.output_dir / "termite_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Create model manifest for Termite registry
        logger.info("Creating manifest.json...")
        manifest = {
            "model_type": "reader",
            "capabilities": ["description", "structured_output"],
            "architecture": "moondream",
            "version": version,
        }
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Log exported files
        logger.info("\nExported files:")
        for f in sorted(self.output_dir.iterdir()):
            if f.is_file():
                size = f.stat().st_size / (1024 * 1024)
                logger.info(f"  {f.name}: {size:.1f} MB")

        return self.output_dir

    def _export_vision_encoder(self, inner_model):
        """Export the SigLIP vision encoder to ONNX.

        The vision encoder is at inner_model.vision and includes:
        - patch_emb: Linear layer for patch embedding
        - pos_emb: Positional embedding parameter
        - blocks: List of transformer blocks
        - post_ln: Post layer normalization
        """
        vision = inner_model.vision

        class VisionEncoderWrapper(nn.Module):
            """Wrapper for the vision encoder that handles the full forward pass."""
            def __init__(self, vision_model, config):
                super().__init__()
                self.vision = vision_model
                self.config = config

            def forward(self, pixel_values):
                # pixel_values: [B, C, H, W]
                # Based on vision.py: vision_encoder function
                from einops import rearrange

                # Get config values (default moondream2 values)
                patch_size = getattr(self.config, 'enc_patch_size', 14)
                n_heads = getattr(self.config, 'enc_n_heads', 16)

                # Patch embedding: B,C,H,W -> B,T,D
                x = rearrange(
                    pixel_values,
                    "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                    p1=patch_size,
                    p2=patch_size,
                )

                x = torch.nn.functional.linear(x, self.vision.patch_emb.weight, self.vision.patch_emb.bias)
                x = x + self.vision.pos_emb

                # Transformer blocks
                for block in self.vision.blocks:
                    # Self-attention
                    ln1_out = torch.nn.functional.layer_norm(
                        x, block.ln1.normalized_shape, block.ln1.weight, block.ln1.bias
                    )
                    # QKV projection
                    qkv = torch.nn.functional.linear(ln1_out, block.attn.qkv.weight, block.attn.qkv.bias)
                    q, k, v = qkv.chunk(3, dim=-1)

                    # Reshape for multi-head attention
                    B, T, D = q.shape
                    head_dim = D // n_heads
                    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
                    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
                    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)

                    # Scaled dot-product attention
                    attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                    attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
                    attn_out = torch.nn.functional.linear(attn_out, block.attn.proj.weight, block.attn.proj.bias)

                    # Residual
                    x = x + attn_out

                    # MLP
                    ln2_out = torch.nn.functional.layer_norm(
                        x, block.ln2.normalized_shape, block.ln2.weight, block.ln2.bias
                    )
                    mlp_out = torch.nn.functional.linear(ln2_out, block.mlp.fc1.weight, block.mlp.fc1.bias)
                    mlp_out = torch.nn.functional.gelu(mlp_out, approximate='tanh')
                    mlp_out = torch.nn.functional.linear(mlp_out, block.mlp.fc2.weight, block.mlp.fc2.bias)

                    # Residual
                    x = x + mlp_out

                # Post layer norm
                x = torch.nn.functional.layer_norm(
                    x, self.vision.post_ln.normalized_shape,
                    self.vision.post_ln.weight, self.vision.post_ln.bias
                )

                return x

        # Get config from inner model if available
        config = getattr(inner_model, 'config', None)
        if config:
            config = getattr(config, 'vision', config)

        wrapper = VisionEncoderWrapper(vision, config)
        wrapper.eval()

        # Moondream2 uses 378x378 images (SigLIP)
        img_size = 378

        # Create dummy input
        dummy_input = torch.randn(1, 3, img_size, img_size)

        # Export to ONNX
        output_path = self.output_dir / "vision_encoder.onnx"
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            input_names=["pixel_values"],
            output_names=["image_features"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_features": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        logger.info(f"  Saved: {output_path.name}")

    def _export_projection(self, inner_model):
        """Export the vision-to-text projection MLP to ONNX.

        The projection is at inner_model.vision.proj_mlp and has:
        - fc1: Linear layer
        - fc2: Linear layer
        """
        proj_mlp = inner_model.vision.proj_mlp

        class ProjectionWrapper(nn.Module):
            def __init__(self, proj):
                super().__init__()
                self.proj = proj

            def forward(self, image_features):
                # Simple 2-layer MLP with GELU
                x = torch.nn.functional.linear(
                    image_features, self.proj.fc1.weight, self.proj.fc1.bias
                )
                x = torch.nn.functional.gelu(x, approximate='tanh')
                x = torch.nn.functional.linear(x, self.proj.fc2.weight, self.proj.fc2.bias)
                return x

        wrapper = ProjectionWrapper(proj_mlp)
        wrapper.eval()

        # Get the input dimension from fc1
        # The projection takes concatenated global + local features
        # For moondream2: enc_dim * 2 = 1152 * 2 = 2304
        in_features = proj_mlp.fc1.in_features

        # SigLIP outputs 729 tokens (27x27 patches from 378/14)
        dummy_input = torch.randn(1, 729, in_features)

        # Export to ONNX
        output_path = self.output_dir / "projection.onnx"
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            input_names=["image_features"],
            output_names=["projected_features"],
            dynamic_axes={
                "image_features": {0: "batch_size", 1: "num_patches"},
                "projected_features": {0: "batch_size", 1: "num_patches"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        logger.info(f"  Saved: {output_path.name}")

    def _export_decoder(self, inner_model):
        """Export the Phi-2 decoder to ONNX.

        The text decoder is at inner_model.text and has:
        - wte: Word token embedding
        - blocks: Transformer decoder blocks
        - post_ln: Post layer norm
        - lm_head: Language model head (may be shared with wte)

        For simplicity, we export without KV-cache support.
        The Go runtime will handle caching at the pipeline level.
        """
        text = inner_model.text

        class DecoderWrapper(nn.Module):
            def __init__(self, text_model):
                super().__init__()
                self.text = text_model

            def forward(self, input_ids, attention_mask=None):
                # Get embeddings
                hidden = torch.nn.functional.embedding(input_ids, self.text.wte)

                # For simplicity in ONNX export, use a basic causal mask
                seq_len = input_ids.size(1)
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
                    diagonal=1
                )
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
                attn_mask = causal_mask.float() * -1e9

                # Process through decoder blocks
                for block in self.text.blocks:
                    # Layer norm
                    ln_out = torch.nn.functional.layer_norm(
                        hidden, block.ln.normalized_shape, block.ln.weight, block.ln.bias
                    )

                    # Self-attention
                    qkv = torch.nn.functional.linear(ln_out, block.attn.qkv.weight, block.attn.qkv.bias)
                    q, k, v = qkv.chunk(3, dim=-1)

                    B, T, D = q.shape
                    n_heads = 32  # Phi-2 default
                    head_dim = D // n_heads

                    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
                    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
                    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)

                    # Note: Phi-2 uses rotary embeddings, but for initial export we skip them
                    # The Go runtime should handle this properly
                    attn_out = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask[:, :, :T, :T]
                    )
                    attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
                    attn_out = torch.nn.functional.linear(attn_out, block.attn.proj.weight, block.attn.proj.bias)

                    # MLP
                    mlp_out = torch.nn.functional.linear(ln_out, block.mlp.fc1.weight, block.mlp.fc1.bias)
                    mlp_out = torch.nn.functional.gelu(mlp_out, approximate='tanh')
                    mlp_out = torch.nn.functional.linear(mlp_out, block.mlp.fc2.weight, block.mlp.fc2.bias)

                    # Residual (parallel attention + MLP)
                    hidden = hidden + attn_out + mlp_out

                # Final layer norm
                hidden = torch.nn.functional.layer_norm(
                    hidden, self.text.post_ln.normalized_shape,
                    self.text.post_ln.weight, self.text.post_ln.bias
                )

                # LM head - may be tied to wte
                if hasattr(self.text, 'lm_head'):
                    logits = torch.nn.functional.linear(hidden, self.text.lm_head.weight, self.text.lm_head.bias)
                else:
                    # Tied weights
                    logits = torch.nn.functional.linear(hidden, self.text.wte, None)

                return logits

        wrapper = DecoderWrapper(text)
        wrapper.eval()

        # Create dummy inputs
        seq_len = 32
        dummy_input_ids = torch.ones(1, seq_len, dtype=torch.long)

        # Export to ONNX
        output_path = self.output_dir / "decoder_model.onnx"
        torch.onnx.export(
            wrapper,
            dummy_input_ids,
            str(output_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        logger.info(f"  Saved: {output_path.name}")

    def _save_tokenizer(self, inner_model):
        """Save the tokenizer.

        Moondream uses tokenizers.Tokenizer (not HuggingFace AutoTokenizer).
        We save it in a format compatible with our Go tokenizer.
        """
        try:
            # The tokenizer is a tokenizers.Tokenizer instance
            tokenizer = inner_model.tokenizer

            # Save the tokenizer JSON
            tokenizer_json = tokenizer.to_str()
            tokenizer_path = self.output_dir / "tokenizer.json"
            with open(tokenizer_path, "w") as f:
                f.write(tokenizer_json)
            logger.info(f"  Saved: tokenizer.json")

            # Create tokenizer_config.json for compatibility
            config = {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 2048,
                "pad_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "bos_token": "<|endoftext|>",
            }
            config_path = self.output_dir / "tokenizer_config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"  Saved: tokenizer_config.json")

        except Exception as e:
            logger.warning(f"  Could not save tokenizer: {e}")
            logger.warning("  Downloading tokenizer from HuggingFace instead...")

            # Fall back to downloading from HuggingFace
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                revision=self.revision,
            )
            tokenizer.save_pretrained(str(self.output_dir))
            logger.info(f"  Saved tokenizer files")

    def _export_vision_encoder_v2(self, vision_encoder, model_structure: str):
        """Export vision encoder to ONNX (handles both old and new structures).

        Converts to float32 for export to ensure type consistency in the ONNX graph.
        Vision encoder is ~300M params so fits comfortably in RAM.
        """
        # Convert vision encoder to float32 for consistent ONNX types
        vision_encoder = vision_encoder.float()

        class VisionEncoderWrapper(nn.Module):
            def __init__(self, encoder, structure):
                super().__init__()
                self.encoder = encoder
                self.structure = structure

            def forward(self, pixel_values):
                if self.structure == "old":
                    # Old structure: EncoderWrapper with __call__ that handles everything
                    return self.encoder(pixel_values)
                else:
                    # New structure: Manual forward pass through vision encoder
                    from einops import rearrange

                    patch_size = 14
                    n_heads = 16

                    x = rearrange(
                        pixel_values,
                        "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                        p1=patch_size,
                        p2=patch_size,
                    )

                    x = torch.nn.functional.linear(x, self.encoder.patch_emb.weight, self.encoder.patch_emb.bias)
                    x = x + self.encoder.pos_emb

                    for block in self.encoder.blocks:
                        ln1_out = torch.nn.functional.layer_norm(
                            x, block.ln1.normalized_shape, block.ln1.weight, block.ln1.bias
                        )
                        qkv = torch.nn.functional.linear(ln1_out, block.attn.qkv.weight, block.attn.qkv.bias)
                        q, k, v = qkv.chunk(3, dim=-1)

                        B, T, D = q.shape
                        head_dim = D // n_heads
                        q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
                        k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
                        v = v.view(B, T, n_heads, head_dim).transpose(1, 2)

                        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
                        attn_out = torch.nn.functional.linear(attn_out, block.attn.proj.weight, block.attn.proj.bias)

                        x = x + attn_out

                        ln2_out = torch.nn.functional.layer_norm(
                            x, block.ln2.normalized_shape, block.ln2.weight, block.ln2.bias
                        )
                        mlp_out = torch.nn.functional.linear(ln2_out, block.mlp.fc1.weight, block.mlp.fc1.bias)
                        mlp_out = torch.nn.functional.gelu(mlp_out, approximate='tanh')
                        mlp_out = torch.nn.functional.linear(mlp_out, block.mlp.fc2.weight, block.mlp.fc2.bias)

                        x = x + mlp_out

                    x = torch.nn.functional.layer_norm(
                        x, self.encoder.post_ln.normalized_shape,
                        self.encoder.post_ln.weight, self.encoder.post_ln.bias
                    )

                    return x

        wrapper = VisionEncoderWrapper(vision_encoder, model_structure)
        wrapper.eval()

        # Moondream2 uses 378x378 images
        img_size = 378
        dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

        output_path = self.output_dir / "vision_encoder.onnx"
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            input_names=["pixel_values"],
            output_names=["image_features"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_features": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        logger.info(f"  Saved: {output_path.name}")

    def _export_projection_v2(self, projection, model_structure: str):
        """Export projection layer to ONNX (handles both old and new structures).

        Converts to float32 for export to ensure type consistency in the ONNX graph.
        Projection is tiny (~10M params) so fits easily in RAM.
        """
        # Convert projection to float32 for consistent ONNX types
        projection = projection.float()

        class ProjectionWrapper(nn.Module):
            def __init__(self, proj, structure):
                super().__init__()
                self.proj = proj
                self.structure = structure

            def forward(self, image_features):
                if self.structure == "old":
                    # Old structure: VisionProjection class
                    return self.proj(image_features)
                else:
                    # New structure: proj_mlp ModuleDict
                    x = torch.nn.functional.linear(
                        image_features, self.proj.fc1.weight, self.proj.fc1.bias
                    )
                    x = torch.nn.functional.gelu(x, approximate='tanh')
                    x = torch.nn.functional.linear(x, self.proj.fc2.weight, self.proj.fc2.bias)
                    return x

        wrapper = ProjectionWrapper(projection, model_structure)
        wrapper.eval()

        # Get input dimensions - try to infer from model
        if model_structure == "old":
            # Old structure projection expects concatenated global + local features
            # Try to get from the projection layer itself
            if hasattr(projection, 'proj') and hasattr(projection.proj, 'in_features'):
                in_features = projection.proj.in_features
            elif hasattr(projection, 'in_features'):
                in_features = projection.in_features
            else:
                # Default: 2304 = 1152 * 2 (global + local)
                in_features = 2304
        else:
            # New structure has fc1.in_features
            in_features = projection.fc1.in_features

        # 729 tokens from 27x27 patches
        dummy_input = torch.randn(1, 729, in_features, dtype=torch.float32)

        output_path = self.output_dir / "projection.onnx"
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            input_names=["image_features"],
            output_names=["projected_features"],
            dynamic_axes={
                "image_features": {0: "batch_size", 1: "num_patches"},
                "projected_features": {0: "batch_size", 1: "num_patches"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        logger.info(f"  Saved: {output_path.name}")

    def _export_decoder_v2(self, text_model, model_structure: str):
        """Export text decoder to ONNX (handles both old and new structures)."""
        # Skip Optimum for custom models - it doesn't handle trust_remote_code well
        # and tends to hang or crash silently. Use manual export instead.
        logger.info("  Using manual decoder export...")
        self._export_decoder_manual(text_model)

    def _export_decoder_manual(self, text_model):
        """Manual decoder export using legacy ONNX export (more compatible).

        Uses float16 to reduce memory requirements, with explicit type handling
        to avoid mixed-type issues in the ONNX graph.
        """
        import gc

        class DecoderWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids):
                outputs = self.model(input_ids=input_ids)
                return outputs.logits

        # Ensure decoder is in float16 for memory efficiency
        text_model = text_model.half()
        wrapper = DecoderWrapper(text_model)
        wrapper.eval()

        # Clear any cached memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        seq_len = 32
        dummy_input_ids = torch.ones(1, seq_len, dtype=torch.long)

        output_path = self.output_dir / "decoder_model.onnx"

        logger.info("  Exporting decoder in float16 mode...")

        # Use legacy export (dynamo=False) for complex models like Phi-2
        # Export in float16 - the Go runtime will handle any necessary conversions
        torch.onnx.export(
            wrapper,
            dummy_input_ids,
            str(output_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=14,  # Use older opset for better compatibility
            do_constant_folding=True,
            dynamo=False,  # Use legacy TorchScript-based export
        )
        logger.info(f"  Saved: {output_path.name}")

    def _save_tokenizer_v2(self, model, tokenizer, model_structure: str):
        """Save tokenizer (handles both old and new structures)."""
        try:
            if model_structure == "new" and tokenizer is not None:
                # New structure: tokenizers.Tokenizer instance
                tokenizer_json = tokenizer.to_str()
                tokenizer_path = self.output_dir / "tokenizer.json"
                with open(tokenizer_path, "w") as f:
                    f.write(tokenizer_json)
                logger.info(f"  Saved: tokenizer.json")
            else:
                # Old structure or fallback: use HuggingFace
                from transformers import AutoTokenizer
                hf_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    revision=self.revision,
                )
                hf_tokenizer.save_pretrained(str(self.output_dir))
                logger.info(f"  Saved tokenizer files")

            # Create tokenizer_config.json for compatibility
            config = {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 2048,
                "pad_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "bos_token": "<|endoftext|>",
            }
            config_path = self.output_dir / "tokenizer_config.json"
            if not config_path.exists():
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

        except Exception as e:
            logger.warning(f"  Could not save tokenizer: {e}")

    def _create_image_processor_config(self, version: str):
        """Create preprocessor_config.json for image processing."""
        # Moondream2 uses SigLIP's image preprocessing
        config = {
            "do_resize": True,
            "size": {"height": 378, "width": 378},
            "do_rescale": True,
            "rescale_factor": 0.00392156862745098,  # 1/255
            "do_normalize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "do_convert_rgb": True,
            "processor_class": "SiglipImageProcessor",
        }

        config_path = self.output_dir / "preprocessor_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"  Saved: {config_path.name}")

    def test(self) -> bool:
        import onnxruntime as ort
        import numpy as np

        try:
            # Test vision encoder
            encoder_path = self.output_dir / "vision_encoder.onnx"
            if encoder_path.exists():
                logger.info("Testing vision encoder...")
                session = ort.InferenceSession(
                    str(encoder_path), providers=["CPUExecutionProvider"]
                )
                dummy_input = np.random.randn(1, 3, 378, 378).astype(np.float32)
                outputs = session.run(None, {"pixel_values": dummy_input})
                logger.info(f"  Vision encoder output shape: {outputs[0].shape}")
            else:
                logger.warning("vision_encoder.onnx not found")

            # Test projection
            projection_path = self.output_dir / "projection.onnx"
            if projection_path.exists():
                logger.info("Testing projection layer...")
                session = ort.InferenceSession(
                    str(projection_path), providers=["CPUExecutionProvider"]
                )
                # Get input shape from model
                input_info = session.get_inputs()[0]
                in_features = input_info.shape[-1] if input_info.shape[-1] else 2304
                dummy_input = np.random.randn(1, 729, in_features).astype(np.float32)
                outputs = session.run(None, {"image_features": dummy_input})
                logger.info(f"  Projection output shape: {outputs[0].shape}")
            else:
                logger.warning("projection.onnx not found")

            # Test decoder
            decoder_path = self.output_dir / "decoder_model.onnx"
            if decoder_path.exists():
                logger.info(f"Testing decoder_model.onnx...")
                session = ort.InferenceSession(
                    str(decoder_path), providers=["CPUExecutionProvider"]
                )
                inputs = session.get_inputs()
                logger.info(f"  Decoder inputs: {[i.name for i in inputs]}")

                # Test with dummy input
                dummy_ids = np.ones((1, 16), dtype=np.int64)
                outputs = session.run(None, {"input_ids": dummy_ids})
                logger.info(f"  Decoder output shape: {outputs[0].shape}")
            else:
                logger.warning("decoder_model.onnx not found")

            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
