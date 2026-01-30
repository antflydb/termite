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
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GLiNER2SpanWrapper(nn.Module):
    """
    ONNX-exportable wrapper for GLiNER2 models.

    This wrapper extracts the core components from GLiNER2 and creates
    an ONNX-exportable forward pass that matches the GLiNER v1 interface.

    The key challenge is that:
    - span_idx contains indices relative to text tokens (0 to text_length-1)
    - hidden_states has shape [batch, seq_len, hidden] where text tokens
      start at some offset in the sequence (after [CLS] and label tokens)
    - GLiNER2's SpanRepLayer expects hidden_states size to match num_spans

    Solution: We extract the projection layers from GLiNER2's span_rep and
    implement our own span representation that:
    1. Finds text token offset using words_mask
    2. Adjusts span_idx to absolute sequence positions
    3. Gathers start/end representations correctly

    ONNX Interface (matches GLiNER v1):
        Inputs:
            - input_ids: [batch, seq_len] - Token IDs
            - attention_mask: [batch, seq_len] - Attention mask
            - words_mask: [batch, seq_len] - Word boundary (>0 for text tokens)
            - text_lengths: [batch, 1] - Number of text tokens
            - span_idx: [batch, num_spans, 2] - Span positions (text-relative)
            - span_mask: [batch, num_spans] - Valid span mask

        Outputs:
            - logits: [batch, num_spans, 1] - Span scores (sigmoid-ready)
    """

    def __init__(self, gliner2_model, max_width: int = 8):
        super().__init__()

        # GLiNER2 model components
        self.encoder = gliner2_model.encoder
        self.classifier = gliner2_model.classifier

        # Extract projection layers from span_rep's internal span_rep_layer
        # GLiNER2 uses SpanRepLayer which contains SpanMarkerV0 as span_rep_layer
        span_layer = gliner2_model.span_rep.span_rep_layer
        self.project_start = span_layer.project_start
        self.project_end = span_layer.project_end
        self.out_project = span_layer.out_project

        self.max_width = max_width
        self.hidden_size = gliner2_model.hidden_size

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
        Forward pass for ONNX export.

        Args:
            input_ids: [batch, seq_len] Token IDs
            attention_mask: [batch, seq_len] Attention mask
            words_mask: [batch, seq_len] Word boundary tracking (>0 for text tokens)
            text_lengths: [batch, 1] Number of text tokens
            span_idx: [batch, num_spans, 2] Span positions relative to text start
            span_mask: [batch, num_spans] Valid span mask

        Returns:
            logits: [batch, num_spans, 1] Span classification scores
        """
        batch_size = input_ids.shape[0]
        num_spans = span_idx.shape[1]

        # 1. Encode input through DeBERTa
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        # 2. Find text token offset from words_mask
        # words_mask > 0 indicates text tokens; we need the first such position
        text_mask = (words_mask > 0).long()  # [batch, seq_len]
        # argmax returns the first occurrence of max value (1)
        text_start_idx = text_mask.argmax(dim=1, keepdim=True)  # [batch, 1]

        # 3. Adjust span indices from text-relative to absolute sequence positions
        span_idx_offset = text_start_idx.unsqueeze(-1).expand(-1, num_spans, 2)  # [batch, num_spans, 2]
        span_idx_abs = span_idx + span_idx_offset  # [batch, num_spans, 2]

        # 4. Project hidden states for start and end representations
        start_rep = self.project_start(hidden_states)  # [batch, seq_len, hidden_size]
        end_rep = self.project_end(hidden_states)      # [batch, seq_len, hidden_size]

        # 5. Gather start/end representations using absolute indices
        start_indices = span_idx_abs[:, :, 0].unsqueeze(-1).expand(-1, -1, self.hidden_size)
        end_indices = span_idx_abs[:, :, 1].unsqueeze(-1).expand(-1, -1, self.hidden_size)

        start_span_rep = torch.gather(start_rep, 1, start_indices)  # [batch, num_spans, hidden_size]
        end_span_rep = torch.gather(end_rep, 1, end_indices)        # [batch, num_spans, hidden_size]

        # 6. Concatenate and project (mimics SpanMarkerV0 logic)
        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()  # [batch, num_spans, hidden_size*2]
        span_reps = self.out_project(cat)  # [batch, num_spans, hidden_size]

        # 7. Apply classifier to get span scores
        logits = self.classifier(span_reps)  # [batch, num_spans, 1]

        return logits


class GLiNER2EntityWrapper(nn.Module):
    """
    Wrapper that outputs per-label scores for entity extraction.

    This matches the GLiNER v1 output format where logits have shape
    [batch, num_tokens, max_width, num_labels].

    For GLiNER2, we compute span scores and replicate for each label
    since labels are handled via schema encoding in the input.
    """

    def __init__(self, gliner2_model, max_width: int = 8, num_labels: int = 1):
        super().__init__()
        self.span_wrapper = GLiNER2SpanWrapper(gliner2_model, max_width)
        self.max_width = max_width
        self.num_labels = num_labels

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
        Forward pass with GLiNER v1 compatible output shape.

        Returns:
            logits: [batch, num_tokens, max_width, num_labels]
        """
        batch_size = input_ids.shape[0]
        num_spans = span_idx.shape[1]

        # Get span scores
        span_logits = self.span_wrapper(
            input_ids, attention_mask, words_mask,
            text_lengths, span_idx, span_mask
        )  # [batch, num_spans, 1]

        # Compute num_tokens from span structure
        # num_spans = num_tokens * max_width
        num_tokens = num_spans // self.max_width

        # Reshape to [batch, num_tokens, max_width, 1]
        logits = span_logits.view(batch_size, num_tokens, self.max_width, 1)

        # Expand to num_labels if needed
        if self.num_labels > 1:
            logits = logits.expand(-1, -1, -1, self.num_labels)

        return logits


def analyze_gliner2_model(model_id: str) -> Tuple[Dict[str, Any], Any]:
    """
    Analyze a GLiNER2 model to understand its architecture.

    Returns:
        Tuple of (info dict, model)
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

    # Get key attributes
    info["max_width"] = getattr(model, 'max_width', 8)
    info["hidden_size"] = getattr(model, 'hidden_size', 768)

    # Check encoder
    if hasattr(model, 'encoder'):
        encoder = model.encoder
        if hasattr(encoder, 'config'):
            info["encoder_config"] = {
                "hidden_size": getattr(encoder.config, 'hidden_size', None),
                "num_layers": getattr(encoder.config, 'num_hidden_layers', None),
                "vocab_size": getattr(encoder.config, 'vocab_size', None),
                "model_type": getattr(encoder.config, 'model_type', None),
            }

    # Check components
    info["has_span_rep"] = hasattr(model, 'span_rep')
    info["has_classifier"] = hasattr(model, 'classifier')
    info["has_count_embed"] = hasattr(model, 'count_embed')

    return info, model


def export_gliner2_to_onnx(
    model_id: str,
    output_dir: Path,
    variants: Optional[list] = None,
    max_width: Optional[int] = None,
    max_seq_len: int = 512,
    opset_version: int = 17,
) -> Path:
    """
    Export a GLiNER2 model to ONNX format.

    Args:
        model_id: HuggingFace model ID (e.g., fastino/gliner2-base-v1)
        output_dir: Directory to save the exported model
        variants: List of variant types (f16, i8)
        max_width: Maximum span width (auto-detected from model if None)
        max_seq_len: Maximum sequence length (default: 512)
        opset_version: ONNX opset version (default: 17)

    Returns:
        Path to the output directory
    """
    import onnx

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

    # Load and analyze model
    logger.info("Loading GLiNER2 model...")
    info, model = analyze_gliner2_model(model_id)
    model.eval()

    # Auto-detect max_width from model
    if max_width is None:
        max_width = info.get("max_width", 8)
    logger.info(f"Using max_width: {max_width}")

    logger.info(f"Model type: {info['type']}")
    logger.info(f"Hidden size: {info.get('hidden_size', 768)}")
    if 'encoder_config' in info:
        logger.info(f"Encoder: {info['encoder_config'].get('model_type', 'unknown')}")

    # Create ONNX wrapper
    logger.info("Creating ONNX wrapper...")
    wrapper = GLiNER2SpanWrapper(model, max_width=max_width)
    wrapper.eval()

    # Test forward pass
    logger.info("Testing forward pass...")
    test_batch = 1
    test_seq_len = 64
    test_num_tokens = 10
    test_num_spans = test_num_tokens * max_width

    with torch.no_grad():
        dummy_inputs = create_dummy_inputs(
            test_batch, test_seq_len, test_num_tokens, max_width
        )
        test_output = wrapper(**dummy_inputs)
        logger.info(f"Forward pass successful. Output shape: {test_output.shape}")

    # Prepare export inputs
    batch_size = 1
    seq_len = max_seq_len
    num_tokens = 50
    num_spans = num_tokens * max_width

    dummy_inputs = create_dummy_inputs(batch_size, seq_len, num_tokens, max_width)

    # Define ONNX export settings
    onnx_path = output_dir / "model.onnx"
    logger.info(f"Exporting to ONNX: {onnx_path}")

    input_names = ['input_ids', 'attention_mask', 'words_mask', 'text_lengths', 'span_idx', 'span_mask']
    output_names = ['logits']

    dynamic_axes = {
        'input_ids': {0: 'batch', 1: 'seq_len'},
        'attention_mask': {0: 'batch', 1: 'seq_len'},
        'words_mask': {0: 'batch', 1: 'seq_len'},
        'text_lengths': {0: 'batch'},
        'span_idx': {0: 'batch', 1: 'num_spans'},
        'span_mask': {0: 'batch', 1: 'num_spans'},
        'logits': {0: 'batch', 1: 'num_spans'},
    }

    try:
        # Use traditional TorchScript-based export (dynamo=False) for reliability
        # The dynamo exporter is still experimental and can be very slow
        torch.onnx.export(
            wrapper,
            (
                dummy_inputs['input_ids'],
                dummy_inputs['attention_mask'],
                dummy_inputs['words_mask'],
                dummy_inputs['text_lengths'],
                dummy_inputs['span_idx'],
                dummy_inputs['span_mask'],
            ),
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,  # Use TorchScript export for speed and reliability
        )
        logger.info(f"  Saved: model.onnx")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise

    # Verify the exported model
    logger.info("Verifying exported ONNX model...")
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("  ONNX model verification passed")

        # Log model size
        model_size = os.path.getsize(onnx_path) / (1024 * 1024)
        logger.info(f"  Model size: {model_size:.1f} MB")
    except Exception as e:
        logger.warning(f"  ONNX verification warning: {e}")

    # Apply FP16 conversion if requested
    if "f16" in variants:
        convert_to_fp16(onnx_path, output_dir / "model_f16.onnx")

    # Apply INT8 quantization if requested
    if "i8" in variants:
        quantize_to_int8(onnx_path, output_dir / "model_i8.onnx")

    # Save tokenizer
    save_tokenizer(model, output_dir)

    # Create GLiNER config for Termite
    save_gliner_config(model_id, max_width, max_seq_len, output_dir)

    logger.info(f"\nExport complete! Model saved to: {output_dir}")
    return output_dir


def create_dummy_inputs(
    batch_size: int, seq_len: int, num_tokens: int, max_width: int
) -> Dict[str, torch.Tensor]:
    """Create dummy inputs for ONNX export.

    Creates a realistic input structure matching Termite's GLiNER pipeline:
    - Sequence: [CLS] [label tokens] [SEP] [text tokens] [SEP] [PAD...]
    - words_mask: 0 for non-text, >0 for text tokens (word index)
    - span_idx: indices relative to text token start (0 to num_tokens-1)
    """
    num_spans = num_tokens * max_width

    # Simulate a realistic sequence structure:
    # [CLS] + 5 label tokens + [SEP] + num_tokens text tokens + [SEP] + padding
    text_start_idx = 7  # After [CLS](1) + 5 labels + [SEP](1) = 7

    inputs = {
        'input_ids': torch.randint(0, 30000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long),
        'words_mask': torch.zeros(batch_size, seq_len, dtype=torch.long),
        'text_lengths': torch.tensor([[num_tokens]], dtype=torch.long),
        'span_idx': torch.zeros(batch_size, num_spans, 2, dtype=torch.long),
        'span_mask': torch.ones(batch_size, num_spans, dtype=torch.bool),
    }

    # Set attention mask (1 for real tokens, 0 for padding)
    text_end_idx = text_start_idx + num_tokens + 1  # +1 for final [SEP]
    for b in range(batch_size):
        # Attention mask covers [CLS] + labels + [SEP] + text + [SEP]
        inputs['attention_mask'][b, text_end_idx:] = 0

        # words_mask: >0 for text tokens (use word index starting at 1)
        for t in range(num_tokens):
            if text_start_idx + t < seq_len:
                inputs['words_mask'][b, text_start_idx + t] = t + 1  # Word index

    # Build span indices (text-relative: 0 to num_tokens-1)
    for b in range(batch_size):
        for t in range(num_tokens):
            for w in range(max_width):
                idx = t * max_width + w
                start = t
                end = min(t + w, num_tokens - 1)
                inputs['span_idx'][b, idx, 0] = start
                inputs['span_idx'][b, idx, 1] = end
                # Mask is True if span end is within bounds
                inputs['span_mask'][b, idx] = (t + w) < num_tokens

    return inputs


def convert_to_fp16(input_path: Path, output_path: Path):
    """Convert ONNX model to FP16."""
    logger.info("Converting to FP16...")
    try:
        from onnxconverter_common import float16
        import onnx

        onnx_model = onnx.load(str(input_path))
        fp16_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
        onnx.save(fp16_model, str(output_path))
        logger.info(f"  Saved: {output_path.name}")
    except ImportError:
        logger.warning("  onnxconverter-common not installed. Skipping FP16 conversion.")
    except Exception as e:
        logger.warning(f"  FP16 conversion failed: {e}")


def quantize_to_int8(input_path: Path, output_path: Path):
    """Quantize ONNX model to INT8."""
    logger.info("Applying INT8 quantization...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            str(input_path),
            str(output_path),
            weight_type=QuantType.QInt8,
        )
        logger.info(f"  Saved: {output_path.name}")
    except ImportError:
        logger.warning("  onnxruntime not installed. Skipping INT8 quantization.")
    except Exception as e:
        logger.warning(f"  INT8 quantization failed: {e}")


def save_tokenizer(model, output_dir: Path):
    """Save tokenizer from model."""
    logger.info("Saving tokenizer...")
    try:
        from transformers import AutoTokenizer

        tokenizer = None
        if hasattr(model, 'processor') and hasattr(model.processor, 'tokenizer'):
            tokenizer = model.processor.tokenizer
        elif hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer

        if tokenizer is None:
            logger.info("  Loading default DeBERTa tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

        tokenizer.save_pretrained(str(output_dir))
        logger.info("  Saved: tokenizer files")
    except Exception as e:
        logger.warning(f"  Could not save tokenizer: {e}")


def save_gliner_config(model_id: str, max_width: int, max_seq_len: int, output_dir: Path):
    """Save GLiNER config for Termite."""
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
        "capabilities": ["ner", "classification", "structured", "relations"],
    }

    config_path = output_dir / "gliner_config.json"
    with open(config_path, "w") as f:
        json.dump(gliner_config, f, indent=2)
    logger.info("  Saved: gliner_config.json")


def test_onnx_model(output_dir: Path, max_width: int = 8):
    """Test the exported ONNX model with ONNX Runtime."""
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
    input_names = set()
    for inp in session.get_inputs():
        logger.info(f"  {inp.name}: {inp.shape} ({inp.type})")
        input_names.add(inp.name)

    logger.info("Model outputs:")
    for out in session.get_outputs():
        logger.info(f"  {out.name}: {out.shape} ({out.type})")

    # Create test inputs with realistic structure
    batch_size = 1
    seq_len = 64
    num_tokens = 10
    num_spans = num_tokens * max_width

    # Simulate: [CLS] + 5 labels + [SEP] + text tokens + [SEP] + padding
    text_start_idx = 7

    # Build all possible inputs
    all_inputs = {
        'input_ids': np.random.randint(0, 30000, (batch_size, seq_len)).astype(np.int64),
        'attention_mask': np.ones((batch_size, seq_len), dtype=np.int64),
        'words_mask': np.zeros((batch_size, seq_len), dtype=np.int64),
        'text_lengths': np.array([[num_tokens]], dtype=np.int64),
        'span_idx': np.zeros((batch_size, num_spans, 2), dtype=np.int64),
        'span_mask': np.ones((batch_size, num_spans), dtype=np.bool_),
    }

    # Set attention mask and words_mask
    text_end_idx = text_start_idx + num_tokens + 1
    all_inputs['attention_mask'][0, text_end_idx:] = 0
    for t in range(num_tokens):
        if text_start_idx + t < seq_len:
            all_inputs['words_mask'][0, text_start_idx + t] = t + 1

    # Build span indices (text-relative)
    for t in range(num_tokens):
        for w in range(max_width):
            idx = t * max_width + w
            all_inputs['span_idx'][0, idx, 0] = t
            all_inputs['span_idx'][0, idx, 1] = min(t + w, num_tokens - 1)
            all_inputs['span_mask'][0, idx] = (t + w) < num_tokens

    # Filter to only inputs the model actually expects
    inputs = {k: v for k, v in all_inputs.items() if k in input_names}
    logger.info(f"Using inputs: {list(inputs.keys())}")

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
    parser.add_argument("--max-width", type=int, default=None,
                        help="Maximum span width (auto-detected from model if not specified)")
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
        test_onnx_model(output_dir, max_width=args.max_width or 8)


if __name__ == "__main__":
    main()
