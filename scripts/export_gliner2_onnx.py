#!/usr/bin/env python3
"""
GLiNER2 ONNX Export Script

This script exports GLiNER2 models to ONNX format for use with Termite.
GLiNER2 is a unified multi-task model supporting NER, classification,
structured extraction, and relation extraction.

Unlike GLiNER v1 which has a built-in export_to_onnx() method, GLiNER2
requires manual ONNX export. This script creates a wrapper module that
matches the GLiNER v1 ONNX interface for compatibility with Termite's
existing GLiNER pipeline.

Usage:
    python export_gliner2_onnx.py fastino/gliner2-base-v1 ./output_dir
    python export_gliner2_onnx.py fastino/gliner2-large-v1 ./output_dir --variants f16 i8

Requirements:
    pip install gliner2 torch onnx onnxruntime transformers
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GLiNER2ONNXWrapper(nn.Module):
    """
    ONNX-exportable wrapper for GLiNER2 models.

    This wrapper adapts GLiNER2's forward pass to match the GLiNER v1 ONNX
    interface expected by Termite's GLiNER pipeline:

    Inputs:
        - input_ids: [batch, seq_len] - Token IDs
        - attention_mask: [batch, seq_len] - Attention mask
        - words_mask: [batch, seq_len] - Word boundary tracking
        - text_lengths: [batch, 1] - Number of text tokens
        - span_idx: [batch, num_spans, 2] - Span start/end positions
        - span_mask: [batch, num_spans] - Valid span mask

    Outputs:
        - logits: [batch, num_tokens, max_width, num_labels] - Span scores
    """

    def __init__(self, model, max_width: int = 12):
        super().__init__()
        self.model = model
        self.max_width = max_width

        # Extract the encoder and span classifier from GLiNER2
        # GLiNER2 typically has these components:
        # - encoder: DeBERTa or similar transformer encoder
        # - span_rep_layer: Span representation layer
        # - entity_classifier: Classification head

        # Access model components (adapt based on actual GLiNER2 structure)
        if hasattr(model, 'model'):
            # GLiNER2 wraps the actual model
            self.encoder = model.model.encoder if hasattr(model.model, 'encoder') else model.model
            self.span_rep_layer = getattr(model.model, 'span_rep_layer', None)
            self.entity_classifier = getattr(model.model, 'entity_classifier', None)
        else:
            self.encoder = getattr(model, 'encoder', model)
            self.span_rep_layer = getattr(model, 'span_rep_layer', None)
            self.entity_classifier = getattr(model, 'entity_classifier', None)

        # Get hidden size from encoder config
        if hasattr(self.encoder, 'config'):
            self.hidden_size = self.encoder.config.hidden_size
        else:
            self.hidden_size = 768  # Default for DeBERTa-base

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        words_mask: torch.Tensor,
        text_lengths: torch.Tensor,
        span_idx: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass matching GLiNER v1 ONNX interface.

        Args:
            input_ids: [batch, seq_len] Token IDs
            attention_mask: [batch, seq_len] Attention mask
            words_mask: [batch, seq_len] Word boundary tracking
            text_lengths: [batch, 1] Number of text tokens
            span_idx: [batch, num_spans, 2] Span positions (start, end)
            span_mask: [batch, num_spans] Valid span mask

        Returns:
            logits: [batch, num_tokens, max_width, num_labels] Span classification scores
        """
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        num_spans = span_idx.shape[1]

        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get hidden states
        if hasattr(encoder_outputs, 'last_hidden_state'):
            hidden_states = encoder_outputs.last_hidden_state
        else:
            hidden_states = encoder_outputs[0]

        # Build span representations from hidden states
        # For each span, concatenate start and end token representations
        span_start_idx = span_idx[:, :, 0]  # [batch, num_spans]
        span_end_idx = span_idx[:, :, 1]    # [batch, num_spans]

        # Gather start and end representations
        # Expand indices for gathering: [batch, num_spans, hidden_size]
        span_start_idx_expanded = span_start_idx.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        span_end_idx_expanded = span_end_idx.unsqueeze(-1).expand(-1, -1, self.hidden_size)

        start_reps = torch.gather(hidden_states, 1, span_start_idx_expanded)
        end_reps = torch.gather(hidden_states, 1, span_end_idx_expanded)

        # Combine span representations (concatenation or other method)
        if self.span_rep_layer is not None:
            span_reps = self.span_rep_layer(start_reps, end_reps)
        else:
            # Simple concatenation followed by projection
            span_reps = torch.cat([start_reps, end_reps], dim=-1)

        # Apply entity classifier
        if self.entity_classifier is not None:
            logits = self.entity_classifier(span_reps)
        else:
            # Fallback: simple linear projection
            # This should be replaced with actual classifier from model
            logits = span_reps  # Placeholder

        # Reshape logits to expected format: [batch, num_tokens, max_width, num_labels]
        # The num_spans = num_tokens * max_width
        num_tokens = text_lengths[0, 0].item() if text_lengths.numel() > 0 else num_spans // self.max_width
        num_labels = logits.shape[-1] if len(logits.shape) > 2 else 1

        # Reshape: [batch, num_spans, num_labels] -> [batch, num_tokens, max_width, num_labels]
        logits = logits.view(batch_size, num_tokens, self.max_width, num_labels)

        return logits


class GLiNER2DirectWrapper(nn.Module):
    """
    Direct wrapper that uses GLiNER2's internal forward pass.

    This wrapper preserves GLiNER2's exact inference logic while adapting
    the interface for ONNX export.
    """

    def __init__(self, model, max_width: int = 12):
        super().__init__()
        self.gliner2_model = model
        self.max_width = max_width

        # Get the underlying PyTorch model
        if hasattr(model, 'model'):
            self.model = model.model
        else:
            self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        words_mask: torch.Tensor,
        text_lengths: torch.Tensor,
        span_idx: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass using GLiNER2's internal model.
        """
        batch_size = input_ids.shape[0]

        # Call GLiNER2's model forward
        # The exact signature depends on GLiNER2's implementation
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            words_mask=words_mask,
            text_lengths=text_lengths,
            span_idx=span_idx,
            span_mask=span_mask,
        )

        # Extract logits from outputs
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('span_logits', outputs))
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        return logits


def analyze_gliner2_model(model_id: str) -> dict:
    """
    Analyze a GLiNER2 model to understand its architecture.

    Returns:
        dict with model architecture information
    """
    try:
        from gliner2 import GLiNER2
    except ImportError:
        logger.error("gliner2 package not installed. Run: pip install gliner2")
        sys.exit(1)

    logger.info(f"Loading GLiNER2 model: {model_id}")
    model = GLiNER2.from_pretrained(model_id)

    info = {
        "model_id": model_id,
        "type": type(model).__name__,
        "attributes": [],
        "methods": [],
        "model_attributes": [],
    }

    # Analyze top-level attributes
    for attr in dir(model):
        if not attr.startswith('_'):
            obj = getattr(model, attr, None)
            if callable(obj):
                info["methods"].append(attr)
            else:
                info["attributes"].append(attr)

    # Check for internal model
    if hasattr(model, 'model'):
        inner_model = model.model
        info["has_inner_model"] = True
        info["inner_model_type"] = type(inner_model).__name__

        for attr in dir(inner_model):
            if not attr.startswith('_'):
                obj = getattr(inner_model, attr, None)
                if isinstance(obj, nn.Module):
                    info["model_attributes"].append(f"{attr}: {type(obj).__name__}")

    # Check for encoder
    if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        encoder = model.model.encoder
        if hasattr(encoder, 'config'):
            info["encoder_config"] = {
                "hidden_size": getattr(encoder.config, 'hidden_size', None),
                "num_layers": getattr(encoder.config, 'num_hidden_layers', None),
                "vocab_size": getattr(encoder.config, 'vocab_size', None),
            }

    return info, model


def export_gliner2_to_onnx(
    model_id: str,
    output_dir: Path,
    variants: Optional[list[str]] = None,
    max_width: int = 12,
    max_seq_len: int = 512,
    opset_version: int = 14,
) -> Path:
    """
    Export a GLiNER2 model to ONNX format.

    Args:
        model_id: HuggingFace model ID (e.g., fastino/gliner2-base-v1)
        output_dir: Directory to save the exported model
        variants: List of variant types (f16, i8)
        max_width: Maximum span width (default: 12)
        max_seq_len: Maximum sequence length (default: 512)
        opset_version: ONNX opset version (default: 14)

    Returns:
        Path to the output directory
    """
    import onnx
    from onnx import numpy_helper

    try:
        from gliner2 import GLiNER2
    except ImportError:
        logger.error("gliner2 package not installed. Run: pip install gliner2")
        sys.exit(1)

    variants = variants or []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting GLiNER2 model: {model_id}")
    logger.info(f"Output directory: {output_dir}")

    # Load the model
    logger.info("Loading GLiNER2 model...")
    model = GLiNER2.from_pretrained(model_id)

    # Analyze model structure
    logger.info("Analyzing model architecture...")
    info, _ = analyze_gliner2_model(model_id)
    logger.info(f"Model type: {info['type']}")
    if 'encoder_config' in info:
        logger.info(f"Encoder config: {info['encoder_config']}")

    # Create ONNX wrapper
    logger.info("Creating ONNX wrapper...")

    # Try direct wrapper first, fall back to manual wrapper
    try:
        wrapper = GLiNER2DirectWrapper(model, max_width=max_width)
        wrapper.eval()

        # Test forward pass
        test_batch = 1
        test_seq_len = 64
        test_num_tokens = 10
        test_num_spans = test_num_tokens * max_width

        dummy_inputs = {
            'input_ids': torch.randint(0, 30000, (test_batch, test_seq_len)),
            'attention_mask': torch.ones(test_batch, test_seq_len, dtype=torch.long),
            'words_mask': torch.zeros(test_batch, test_seq_len, dtype=torch.long),
            'text_lengths': torch.tensor([[test_num_tokens]], dtype=torch.long),
            'span_idx': torch.zeros(test_batch, test_num_spans, 2, dtype=torch.long),
            'span_mask': torch.ones(test_batch, test_num_spans, dtype=torch.bool),
        }

        # Build valid span indices
        for t in range(test_num_tokens):
            for w in range(max_width):
                idx = t * max_width + w
                dummy_inputs['span_idx'][0, idx, 0] = t
                dummy_inputs['span_idx'][0, idx, 1] = min(t + w, test_num_tokens - 1)

        with torch.no_grad():
            test_output = wrapper(**dummy_inputs)
        logger.info(f"Direct wrapper test passed. Output shape: {test_output.shape}")

    except Exception as e:
        logger.warning(f"Direct wrapper failed: {e}")
        logger.info("Falling back to manual wrapper...")
        wrapper = GLiNER2ONNXWrapper(model, max_width=max_width)
        wrapper.eval()

    # Prepare dummy inputs for export
    batch_size = 1
    seq_len = max_seq_len
    num_tokens = 50  # Example number of text tokens
    num_spans = num_tokens * max_width

    dummy_input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    dummy_words_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    dummy_text_lengths = torch.tensor([[num_tokens]], dtype=torch.long)
    dummy_span_idx = torch.zeros(batch_size, num_spans, 2, dtype=torch.long)
    dummy_span_mask = torch.ones(batch_size, num_spans, dtype=torch.bool)

    # Build valid span indices
    for t in range(num_tokens):
        for w in range(max_width):
            idx = t * max_width + w
            dummy_span_idx[0, idx, 0] = t
            dummy_span_idx[0, idx, 1] = min(t + w, num_tokens - 1)

    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    logger.info(f"Exporting to ONNX: {onnx_path}")

    dynamic_axes = {
        'input_ids': {0: 'batch', 1: 'seq_len'},
        'attention_mask': {0: 'batch', 1: 'seq_len'},
        'words_mask': {0: 'batch', 1: 'seq_len'},
        'text_lengths': {0: 'batch'},
        'span_idx': {0: 'batch', 1: 'num_spans'},
        'span_mask': {0: 'batch', 1: 'num_spans'},
        'logits': {0: 'batch', 1: 'num_tokens', 2: 'max_width', 3: 'num_labels'},
    }

    input_names = ['input_ids', 'attention_mask', 'words_mask', 'text_lengths', 'span_idx', 'span_mask']
    output_names = ['logits']

    try:
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask, dummy_words_mask,
             dummy_text_lengths, dummy_span_idx, dummy_span_mask),
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
        )
        logger.info(f"  Saved: model.onnx")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        logger.info("Trying with dynamo=True (PyTorch 2.1+)...")

        try:
            torch.onnx.export(
                wrapper,
                (dummy_input_ids, dummy_attention_mask, dummy_words_mask,
                 dummy_text_lengths, dummy_span_idx, dummy_span_mask),
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                dynamo=True,
            )
            logger.info(f"  Saved: model.onnx (with dynamo)")
        except Exception as e2:
            logger.error(f"ONNX export with dynamo also failed: {e2}")
            raise

    # Verify the exported model
    logger.info("Verifying exported ONNX model...")
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("  ONNX model verification passed")
    except Exception as e:
        logger.warning(f"  ONNX verification warning: {e}")

    # Apply FP16 conversion if requested
    if "f16" in variants:
        logger.info("Converting to FP16...")
        try:
            from onnxconverter_common import float16
            fp16_path = output_dir / "model_f16.onnx"
            onnx_model = onnx.load(str(onnx_path))
            fp16_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
            onnx.save(fp16_model, str(fp16_path))
            logger.info(f"  Saved: model_f16.onnx")
        except ImportError:
            logger.warning("  onnxconverter-common not installed. Skipping FP16 conversion.")
        except Exception as e:
            logger.warning(f"  FP16 conversion failed: {e}")

    # Apply INT8 quantization if requested
    if "i8" in variants:
        logger.info("Applying INT8 quantization...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            i8_path = output_dir / "model_i8.onnx"
            quantize_dynamic(
                str(onnx_path),
                str(i8_path),
                weight_type=QuantType.QInt8,
            )
            logger.info(f"  Saved: model_i8.onnx")
        except ImportError:
            logger.warning("  onnxruntime not installed. Skipping INT8 quantization.")
        except Exception as e:
            logger.warning(f"  INT8 quantization failed: {e}")

    # Save tokenizer
    logger.info("Saving tokenizer...")
    try:
        from transformers import AutoTokenizer

        # Try to get tokenizer from model
        tokenizer = None
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        elif hasattr(model, 'data_processor') and hasattr(model.data_processor, 'tokenizer'):
            tokenizer = model.data_processor.tokenizer

        if tokenizer is None:
            # Fall back to DeBERTa tokenizer (common base for GLiNER models)
            logger.info("  Loading default DeBERTa tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

        tokenizer.save_pretrained(str(output_dir))
        logger.info("  Saved: tokenizer files")
    except Exception as e:
        logger.warning(f"  Could not save tokenizer: {e}")

    # Create GLiNER config for Termite
    logger.info("Creating gliner_config.json...")
    gliner_config = {
        "max_width": max_width,
        "max_len": max_seq_len,
        "default_labels": ["person", "organization", "location", "date", "product"],
        "threshold": 0.5,
        "flat_ner": True,
        "multi_label": False,
        "model_type": "gliner2",
        "model_id": model_id,
        "capabilities": ["ner", "classification", "relations"],
    }

    config_path = output_dir / "gliner_config.json"
    with open(config_path, "w") as f:
        json.dump(gliner_config, f, indent=2)
    logger.info("  Saved: gliner_config.json")

    logger.info(f"\nExport complete! Model saved to: {output_dir}")
    return output_dir


def test_onnx_model(output_dir: Path, max_width: int = 12):
    """
    Test the exported ONNX model with ONNX Runtime.
    """
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed. Skipping ONNX test.")
        return

    onnx_path = output_dir / "model.onnx"
    if not onnx_path.exists():
        logger.warning(f"ONNX model not found: {onnx_path}")
        return

    logger.info(f"\nTesting ONNX model: {onnx_path}")

    # Load ONNX model
    session = ort.InferenceSession(str(onnx_path))

    # Print input/output info
    logger.info("Model inputs:")
    for inp in session.get_inputs():
        logger.info(f"  {inp.name}: {inp.shape} ({inp.type})")

    logger.info("Model outputs:")
    for out in session.get_outputs():
        logger.info(f"  {out.name}: {out.shape} ({out.type})")

    # Create test inputs
    batch_size = 1
    seq_len = 64
    num_tokens = 10
    num_spans = num_tokens * max_width

    inputs = {
        'input_ids': np.random.randint(0, 30000, (batch_size, seq_len)).astype(np.int64),
        'attention_mask': np.ones((batch_size, seq_len), dtype=np.int64),
        'words_mask': np.zeros((batch_size, seq_len), dtype=np.int64),
        'text_lengths': np.array([[num_tokens]], dtype=np.int64),
        'span_idx': np.zeros((batch_size, num_spans, 2), dtype=np.int64),
        'span_mask': np.ones((batch_size, num_spans), dtype=np.bool_),
    }

    # Build valid span indices
    for t in range(num_tokens):
        for w in range(max_width):
            idx = t * max_width + w
            inputs['span_idx'][0, idx, 0] = t
            inputs['span_idx'][0, idx, 1] = min(t + w, num_tokens - 1)

    # Run inference
    try:
        outputs = session.run(None, inputs)
        logger.info(f"Inference successful!")
        logger.info(f"Output shape: {outputs[0].shape}")
        logger.info(f"Output dtype: {outputs[0].dtype}")
        logger.info(f"Output range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
    except Exception as e:
        logger.error(f"Inference failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Export GLiNER2 models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export GLiNER2 base model
  python export_gliner2_onnx.py fastino/gliner2-base-v1 ./models/gliner2-base

  # Export with FP16 and INT8 variants
  python export_gliner2_onnx.py fastino/gliner2-large-v1 ./models/gliner2-large --variants f16 i8

  # Analyze model architecture without exporting
  python export_gliner2_onnx.py fastino/gliner2-base-v1 --analyze-only
        """,
    )

    parser.add_argument("model_id", help="HuggingFace model ID (e.g., fastino/gliner2-base-v1)")
    parser.add_argument("output_dir", nargs="?", help="Output directory for exported model")
    parser.add_argument("--variants", nargs="+", choices=["f16", "i8"],
                        help="Additional variants to create (f16=FP16, i8=INT8)")
    parser.add_argument("--max-width", type=int, default=12,
                        help="Maximum span width (default: 12)")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze model architecture, don't export")
    parser.add_argument("--test", action="store_true",
                        help="Test the exported ONNX model after export")

    args = parser.parse_args()

    if args.analyze_only:
        info, _ = analyze_gliner2_model(args.model_id)
        print("\nModel Analysis:")
        print(json.dumps(info, indent=2))
        return

    if not args.output_dir:
        parser.error("output_dir is required unless --analyze-only is specified")

    output_dir = Path(args.output_dir)

    # Export the model
    export_gliner2_to_onnx(
        model_id=args.model_id,
        output_dir=output_dir,
        variants=args.variants,
        max_width=args.max_width,
        max_seq_len=args.max_seq_len,
    )

    # Test if requested
    if args.test:
        test_onnx_model(output_dir, max_width=args.max_width)


if __name__ == "__main__":
    main()
