"""GLiNER2 multi-task NER model exporter."""

import json
import logging
import os
from pathlib import Path
from typing import Any

from . import register_exporter
from .base import BaseExporter
from .embedder import convert_to_fp16

logger = logging.getLogger(__name__)


def _create_dummy_inputs(
    batch_size: int, seq_len: int, num_tokens: int, max_width: int
) -> dict:
    """Create dummy inputs for GLiNER2 ONNX export.

    Creates a realistic input structure matching Termite's GLiNER pipeline:
    - Sequence: [CLS] [label tokens] [SEP] [text tokens] [SEP] [PAD...]
    - words_mask: 0 for non-text, >0 for text tokens (word index)
    - span_idx: indices relative to text token start (0 to num_tokens-1)
    """
    import torch

    num_spans = num_tokens * max_width

    # Simulate a realistic sequence structure:
    # [CLS] + 5 label tokens + [SEP] + num_tokens text tokens + [SEP] + padding
    text_start_idx = 7  # After [CLS](1) + 5 labels + [SEP](1) = 7

    inputs = {
        "input_ids": torch.randint(0, 30000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "words_mask": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "text_lengths": torch.tensor([[num_tokens]], dtype=torch.long),
        "span_idx": torch.zeros(batch_size, num_spans, 2, dtype=torch.long),
        "span_mask": torch.ones(batch_size, num_spans, dtype=torch.bool),
    }

    # Set attention mask (1 for real tokens, 0 for padding)
    text_end_idx = text_start_idx + num_tokens + 1  # +1 for final [SEP]
    for b in range(batch_size):
        # Attention mask covers [CLS] + labels + [SEP] + text + [SEP]
        inputs["attention_mask"][b, text_end_idx:] = 0

        # words_mask: >0 for text tokens (use word index starting at 1)
        for t in range(num_tokens):
            if text_start_idx + t < seq_len:
                inputs["words_mask"][b, text_start_idx + t] = t + 1  # Word index

    # Build span indices (text-relative: 0 to num_tokens-1)
    for b in range(batch_size):
        for t in range(num_tokens):
            for w in range(max_width):
                idx = t * max_width + w
                start = t
                end = min(t + w, num_tokens - 1)
                inputs["span_idx"][b, idx, 0] = start
                inputs["span_idx"][b, idx, 1] = end
                # Mask is True if span end is within bounds
                inputs["span_mask"][b, idx] = (t + w) < num_tokens

    return inputs


def _analyze_model(model_id: str) -> tuple[dict[str, Any], Any]:
    """Analyze a GLiNER2 model to understand its architecture."""
    from gliner2 import GLiNER2

    logger.info(f"Loading GLiNER2 model: {model_id}")
    model = GLiNER2.from_pretrained(model_id)

    info = {
        "model_id": model_id,
        "type": type(model).__name__,
        "max_width": getattr(model, "max_width", 8),
        "hidden_size": getattr(model, "hidden_size", 768),
        "has_span_rep": hasattr(model, "span_rep"),
        "has_classifier": hasattr(model, "classifier"),
    }

    # Check encoder
    if hasattr(model, "encoder") and hasattr(model.encoder, "config"):
        info["encoder_config"] = {
            "hidden_size": getattr(model.encoder.config, "hidden_size", None),
            "num_layers": getattr(model.encoder.config, "num_hidden_layers", None),
            "vocab_size": getattr(model.encoder.config, "vocab_size", None),
            "model_type": getattr(model.encoder.config, "model_type", None),
        }

    return info, model


def _save_config(
    model_id: str, max_width: int, max_seq_len: int, output_dir: Path
) -> None:
    """Save GLiNER2 config for Termite."""
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
        "capabilities": ["ner", "zeroshot", "classification", "relations"],
        "tasks": {
            "ner": {
                "model_file": "model.onnx",
                "threshold": 0.5,
                "flat_ner": True,
                "prompt_format": "<<ENT>>{label}<<SEP>>",
            },
            "relations": {
                "model_file": "model.onnx",
                "threshold": 0.3,
                "default_entity_labels": ["person", "organization", "location"],
                "default_relation_labels": ["works_for", "located_in", "founded"],
                "prompt_format": "<<REL>>{entity}::{relation}<<SEP>>",
            },
            "classification": {
                "model_file": "model.onnx",
                "threshold": 0.5,
                "multi_label": True,
                "prompt_format": "<<CLS>>{label}<<SEP>>",
            },
        },
        "relation_labels": [
            "works_for",
            "located_in",
            "founded",
            "part_of",
            "affiliated_with",
        ],
        "relation_threshold": 0.3,
    }

    config_path = output_dir / "gliner_config.json"
    with open(config_path, "w") as f:
        json.dump(gliner_config, f, indent=2)
    logger.info("  Saved: gliner_config.json")


def _save_tokenizer(model: Any, output_dir: Path) -> None:
    """Save tokenizer from GLiNER2 model."""
    from transformers import AutoTokenizer

    logger.info("Saving tokenizer...")
    try:
        tokenizer = None
        if hasattr(model, "processor") and hasattr(model.processor, "tokenizer"):
            tokenizer = model.processor.tokenizer
        elif hasattr(model, "tokenizer"):
            tokenizer = model.tokenizer

        if tokenizer is None:
            logger.info("  Loading default DeBERTa tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

        tokenizer.save_pretrained(str(output_dir))
        logger.info("  Saved: tokenizer files")
    except Exception as e:
        logger.warning(f"  Could not save tokenizer: {e}")


def _quantize_to_int8(input_path: Path, output_path: Path) -> None:
    """Quantize ONNX model to INT8."""
    logger.info("Applying INT8 quantization...")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

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


class _SpanWrapper:
    """ONNX-exportable wrapper for GLiNER2 models.

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
    """

    def __new__(cls, gliner2_model: Any, max_width: int = 8):
        import torch
        import torch.nn as nn

        class SpanWrapperModule(nn.Module):
            def __init__(self, model, max_width: int):
                super().__init__()
                self.encoder = model.encoder
                self.classifier = model.classifier

                # Extract projection layers from span_rep's internal span_rep_layer
                span_layer = model.span_rep.span_rep_layer
                self.project_start = span_layer.project_start
                self.project_end = span_layer.project_end
                self.out_project = span_layer.out_project

                self.max_width = max_width
                self.hidden_size = model.hidden_size

            def forward(
                self,
                input_ids,
                attention_mask,
                words_mask,
                text_lengths,
                span_idx,
                span_mask,
            ):
                num_spans = span_idx.shape[1]

                # 1. Encode input through DeBERTa
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden_states = encoder_outputs.last_hidden_state

                # 2. Find text token offset from words_mask
                text_mask = (words_mask > 0).long()
                text_start_idx = text_mask.argmax(dim=1, keepdim=True)

                # 3. Adjust span indices from text-relative to absolute positions
                span_idx_offset = text_start_idx.unsqueeze(-1).expand(-1, num_spans, 2)
                span_idx_abs = span_idx + span_idx_offset

                # 4. Clamp indices to valid range
                seq_len = hidden_states.shape[1]
                span_idx_abs = span_idx_abs.clamp(0, seq_len - 1)

                # 5. Project hidden states
                start_rep = self.project_start(hidden_states)
                end_rep = self.project_end(hidden_states)

                # 6. Gather span representations
                start_indices = span_idx_abs[:, :, 0].unsqueeze(-1).expand(
                    -1, -1, self.hidden_size
                )
                end_indices = span_idx_abs[:, :, 1].unsqueeze(-1).expand(
                    -1, -1, self.hidden_size
                )

                start_span_rep = torch.gather(start_rep, 1, start_indices)
                end_span_rep = torch.gather(end_rep, 1, end_indices)

                # 7. Combine and project
                cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()
                span_reps = self.out_project(cat)

                # 8. Classify spans
                logits = self.classifier(span_reps)

                return logits

        return SpanWrapperModule(gliner2_model, max_width)


@register_exporter("recognizer", capability="labels-v2")
class GLiNER2Exporter(BaseExporter):
    """Exporter for GLiNER2 multi-task models.

    GLiNER2 is a unified model supporting NER, classification, structured
    extraction, and relation extraction. Unlike GLiNER v1, it requires
    manual ONNX export since there's no built-in export_to_onnx() method.

    This exporter creates an ONNX-exportable wrapper that matches the
    GLiNER v1 interface for compatibility with Termite's GLiNER pipeline.
    """

    capabilities = ["labels-v2"]

    def __init__(
        self,
        model_id: str,
        output_dir: Path,
        variants: list[str] | None = None,
        max_width: int | None = None,
        max_seq_len: int = 512,
        opset_version: int = 17,
    ):
        super().__init__(model_id, output_dir, variants)
        self.max_width = max_width
        self.max_seq_len = max_seq_len
        self.opset_version = opset_version

    def export(self) -> Path:
        import onnx
        import torch

        logger.info(f"Exporting GLiNER2 model: {self.model_id}")
        logger.info(f"Output: {self.output_dir}")

        # Load and analyze model
        logger.info("Loading GLiNER2 model...")
        info, model = _analyze_model(self.model_id)
        model.eval()

        # Auto-detect max_width from model
        max_width = self.max_width or info.get("max_width", 8)
        logger.info(f"Using max_width: {max_width}")
        logger.info(f"Model type: {info['type']}")
        logger.info(f"Hidden size: {info.get('hidden_size', 768)}")

        # Create ONNX wrapper
        logger.info("Creating ONNX wrapper...")
        wrapper = _SpanWrapper(model, max_width=max_width)
        wrapper.eval()

        # Test forward pass
        logger.info("Testing forward pass...")
        test_batch = 1
        test_seq_len = 64
        test_num_tokens = 10

        with torch.no_grad():
            dummy_inputs = _create_dummy_inputs(
                test_batch, test_seq_len, test_num_tokens, max_width
            )
            test_output = wrapper(**dummy_inputs)
            logger.info(f"Forward pass successful. Output shape: {test_output.shape}")

        # Prepare export inputs
        batch_size = 1
        seq_len = self.max_seq_len
        num_tokens = 50

        dummy_inputs = _create_dummy_inputs(batch_size, seq_len, num_tokens, max_width)

        # Define ONNX export settings
        onnx_path = self.output_dir / "model.onnx"
        logger.info(f"Exporting to ONNX: {onnx_path}")

        input_names = [
            "input_ids",
            "attention_mask",
            "words_mask",
            "text_lengths",
            "span_idx",
            "span_mask",
        ]
        output_names = ["logits"]

        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "words_mask": {0: "batch", 1: "seq_len"},
            "text_lengths": {0: "batch"},
            "span_idx": {0: "batch", 1: "num_spans"},
            "span_mask": {0: "batch", 1: "num_spans"},
            "logits": {0: "batch", 1: "num_spans"},
        }

        try:
            torch.onnx.export(
                wrapper,
                (
                    dummy_inputs["input_ids"],
                    dummy_inputs["attention_mask"],
                    dummy_inputs["words_mask"],
                    dummy_inputs["text_lengths"],
                    dummy_inputs["span_idx"],
                    dummy_inputs["span_mask"],
                ),
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=self.opset_version,
                do_constant_folding=True,
                export_params=True,
                dynamo=False,  # Use TorchScript export for speed and reliability
            )
            logger.info("  Saved: model.onnx")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise

        # Verify the exported model
        logger.info("Verifying exported ONNX model...")
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info("  ONNX model verification passed")

            model_size = os.path.getsize(onnx_path) / (1024 * 1024)
            logger.info(f"  Model size: {model_size:.1f} MB")
        except Exception as e:
            logger.warning(f"  ONNX verification warning: {e}")

        # Apply FP16 conversion if requested
        if "f16" in self.variants:
            convert_to_fp16(onnx_path, self.output_dir / "model_f16.onnx")

        # Apply INT8 quantization if requested
        if "i8" in self.variants:
            _quantize_to_int8(onnx_path, self.output_dir / "model_i8.onnx")

        # Save tokenizer and config
        _save_tokenizer(model, self.output_dir)
        _save_config(self.model_id, max_width, self.max_seq_len, self.output_dir)

        logger.info(f"\nExport complete! Model saved to: {self.output_dir}")
        return self.output_dir

    def test(self) -> bool:
        import numpy as np
        import onnxruntime as ort

        try:
            onnx_path = self.output_dir / "model.onnx"
            if not onnx_path.exists():
                logger.error(f"model.onnx not found in {self.output_dir}")
                return False

            # Load config for max_width
            config_path = self.output_dir / "gliner_config.json"
            max_width = 8
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    max_width = config.get("max_width", 8)

            logger.info("Loading GLiNER2 ONNX model...")
            session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )

            # Print input/output info
            logger.info("Model inputs:")
            input_names = set()
            for inp in session.get_inputs():
                logger.info(f"  {inp.name}: {inp.shape} ({inp.type})")
                input_names.add(inp.name)

            logger.info("Model outputs:")
            for out in session.get_outputs():
                logger.info(f"  {out.name}: {out.shape} ({out.type})")

            # Create test inputs
            batch_size = 1
            seq_len = 64
            num_tokens = 10
            num_spans = num_tokens * max_width
            text_start_idx = 7

            all_inputs = {
                "input_ids": np.random.randint(0, 30000, (batch_size, seq_len)).astype(
                    np.int64
                ),
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "words_mask": np.zeros((batch_size, seq_len), dtype=np.int64),
                "text_lengths": np.array([[num_tokens]], dtype=np.int64),
                "span_idx": np.zeros((batch_size, num_spans, 2), dtype=np.int64),
                "span_mask": np.ones((batch_size, num_spans), dtype=np.bool_),
            }

            # Set attention mask and words_mask
            text_end_idx = text_start_idx + num_tokens + 1
            all_inputs["attention_mask"][0, text_end_idx:] = 0
            for t in range(num_tokens):
                if text_start_idx + t < seq_len:
                    all_inputs["words_mask"][0, text_start_idx + t] = t + 1

            # Build span indices (text-relative)
            for t in range(num_tokens):
                for w in range(max_width):
                    idx = t * max_width + w
                    all_inputs["span_idx"][0, idx, 0] = t
                    all_inputs["span_idx"][0, idx, 1] = min(t + w, num_tokens - 1)
                    all_inputs["span_mask"][0, idx] = (t + w) < num_tokens

            # Filter to only inputs the model expects
            inputs = {k: v for k, v in all_inputs.items() if k in input_names}
            logger.info(f"Using inputs: {list(inputs.keys())}")

            # Run inference
            outputs = session.run(None, inputs)
            logger.info("Inference successful!")
            logger.info(f"Output shape: {outputs[0].shape}")
            logger.info(f"Output dtype: {outputs[0].dtype}")
            logger.info(f"Output range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
