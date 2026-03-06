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

# GLiNER2 special token IDs (DeBERTa-v3 with added tokens)
_E_TOKEN_ID = 128005  # [E] entity marker
_P_TOKEN_ID = 128003  # [P] prompt/schema marker
_SEP_TEXT_ID = 128002  # [SEP_TEXT] separator


def _create_dummy_inputs(
    batch_size: int, seq_len: int, num_words: int, num_labels: int, max_width: int
) -> dict:
    """Create dummy inputs for GLiNER2 ONNX export.

    Simulates the structured schema prompt format:
    ( [P] entities ( [E] label1 [E] label2 [E] label3 ) ) [SEP_TEXT] word1 word2 ...

    The input_ids must contain the actual special token IDs so the wrapper
    can find [E] and [P] positions.
    """
    import torch

    num_spans = num_words * max_width

    # Build a realistic input_ids with proper special tokens
    # ( [P] entities ( [E] label1 [E] label2 [E] label3 ) ) [SEP_TEXT] w1 w2 ... wN [PAD...]
    schema_tokens = [287, _P_TOKEN_ID, 6967, 287]  # ( [P] entities (
    for i in range(num_labels):
        schema_tokens.extend([_E_TOKEN_ID, 604 + i])  # [E] labelN
    schema_tokens.extend([1263, 1263, _SEP_TEXT_ID])  # ) ) [SEP_TEXT]

    # Text tokens (one sub-token per word for simplicity)
    text_tokens = [300 + i for i in range(num_words)]

    all_tokens = schema_tokens + text_tokens
    # Pad to seq_len
    while len(all_tokens) < seq_len:
        all_tokens.append(0)
    all_tokens = all_tokens[:seq_len]

    text_start = len(schema_tokens)

    input_ids = torch.tensor([all_tokens], dtype=torch.long)
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    attention_mask[0, : text_start + num_words] = 1

    words_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for w in range(num_words):
        pos = text_start + w
        if pos < seq_len:
            words_mask[0, pos] = w + 1  # 1-indexed

    # Word-level span indices
    span_idx = torch.zeros(batch_size, num_spans, 2, dtype=torch.long)
    span_mask = torch.zeros(batch_size, num_spans, dtype=torch.bool)
    for w in range(num_words):
        for wi in range(max_width):
            si = w * max_width + wi
            end_word = w + wi
            if end_word < num_words:
                span_idx[0, si, 0] = w
                span_idx[0, si, 1] = end_word
                span_mask[0, si] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "words_mask": words_mask,
        "span_idx": span_idx,
        "span_mask": span_mask,
    }


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
        "multi_label": True,
        "model_type": "gliner2",
        "model_id": model_id,
        "capabilities": ["ner", "zeroshot", "classification", "relations"],
        "tasks": {
            "ner": {
                "model_file": "model.onnx",
                "threshold": 0.5,
                "flat_ner": True,
            },
            "relations": {
                "model_file": "model.onnx",
                "threshold": 0.3,
                "default_entity_labels": ["person", "organization", "location"],
                "default_relation_labels": ["works_for", "located_in", "founded"],
            },
            "classification": {
                "model_file": "model.onnx",
                "threshold": 0.5,
                "multi_label": True,
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

    GLiNER2's inference pipeline:
    1. Encode structured schema prompt + text through DeBERTa
    2. Extract [P] embedding and [E] embeddings from schema positions
    3. Extract word-level text embeddings (first sub-token per word)
    4. Compute span representations from text embeddings
    5. Predict entity count from [P] embedding (hardcoded to 1 for ONNX)
    6. Project schema embeddings through count_embed
    7. Score spans via dot product: einsum(span_rep, projected_schema)

    The Go code must build the full structured prompt:
      ( [P] entities ( [E] label1 [E] label2 ... ) ) [SEP_TEXT] text_tokens...

    Output: [batch, num_words, max_width, num_labels] logits
    """

    def __new__(cls, gliner2_model: Any, max_width: int = 8):
        import torch
        import torch.nn as nn

        class SpanWrapperModule(nn.Module):
            def __init__(self, model, max_width: int):
                super().__init__()
                self.encoder = model.encoder
                self.span_rep = model.span_rep
                self.count_embed = model.count_embed
                self.max_width = max_width
                self.hidden_size = model.hidden_size
                self.e_token_id = _E_TOKEN_ID
                self.p_token_id = _P_TOKEN_ID

            def forward(
                self,
                input_ids,      # [1, seq_len]
                attention_mask,  # [1, seq_len]
                words_mask,      # [1, seq_len] — >0 for text tokens (word index)
                span_idx,        # [1, num_spans, 2] — word-level span indices
                span_mask,       # [1, num_spans]
            ):
                # 1. Encode through DeBERTa
                hidden = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state  # [1, seq_len, hidden]

                # 2. Find [P] and [E] positions from input_ids
                # [P] token embedding — used for count prediction
                p_mask = (input_ids[0] == self.p_token_id)
                # [E] token embeddings — one per label
                e_mask = (input_ids[0] == self.e_token_id)

                # Extract [P] embedding (first [P] token)
                p_positions = p_mask.nonzero(as_tuple=True)[0]
                p_emb = hidden[0, p_positions[0]]  # [hidden]

                # Extract [E] embeddings
                e_positions = e_mask.nonzero(as_tuple=True)[0]
                e_embs = hidden[0, e_positions]  # [num_labels, hidden]
                num_labels = e_embs.shape[0]

                # 3. Extract word-level text embeddings (first sub-token per word)
                # words_mask > 0 marks text tokens, value = word index (1-indexed)
                text_mask = (words_mask[0] > 0)
                text_positions = text_mask.nonzero(as_tuple=True)[0]

                # Group by word: take first occurrence of each word index
                word_indices = words_mask[0, text_positions]  # word IDs for each text token
                # Find first token for each word by detecting where word index changes
                # word_indices is monotonically non-decreasing, so we detect transitions
                n = word_indices.shape[0]
                first_token_mask = torch.ones(n, dtype=torch.bool, device=input_ids.device)
                first_token_mask[1:] = word_indices[1:] != word_indices[:-1]

                first_positions = text_positions[first_token_mask]
                word_embs = hidden[0, first_positions]  # [num_words, hidden]
                num_words = word_embs.shape[0]

                # 4. Compute span representations
                # span_idx contains word-level indices (0 to num_words-1)
                # span_rep expects [batch, num_words, hidden] and [batch, num_spans, 2]
                span_rep = self.span_rep(
                    word_embs.unsqueeze(0), span_idx
                )  # [1, num_words, max_width, hidden]

                # 5. Count prediction — hardcoded to 1 for ONNX compatibility.
                # The count_embed GRU uses dynamic sequence length based on count,
                # which doesn't trace well. For NER, count is almost always 1.
                pred_count = 1

                # 6. Project schema embeddings through count_embed
                struct_proj = self.count_embed(
                    e_embs, pred_count
                )  # [count, num_labels, hidden]

                # 7. Score via dot product
                # span_rep: [1, num_words, max_width, hidden] -> squeeze to [num_words, max_width, hidden]
                # struct_proj: [count, num_labels, hidden]
                # einsum: "lkd,bpd->bplk" where l=num_words, k=max_width, b=count, p=num_labels
                span_rep_sq = span_rep.squeeze(0)  # [num_words, max_width, hidden]
                logits = torch.einsum(
                    "lkd,bpd->bplk", span_rep_sq, struct_proj
                )  # [count, num_labels, num_words, max_width]

                # With count=1, squeeze the count dimension
                logits = logits.squeeze(0)  # [num_labels, num_words, max_width]

                # Reshape: [num_labels, num_words, max_width] -> [1, num_words, max_width, num_labels]
                logits = logits.permute(1, 2, 0).unsqueeze(0)

                return logits

        return SpanWrapperModule(gliner2_model, max_width)


@register_exporter("recognizer", capability="labels-v2")
class GLiNER2Exporter(BaseExporter):
    """Exporter for GLiNER2 multi-task models.

    GLiNER2 is a unified model supporting NER, classification, structured
    extraction, and relation extraction. Unlike GLiNER v1, it requires
    manual ONNX export since there's no built-in export_to_onnx() method.

    This exporter creates an ONNX-exportable wrapper that includes the full
    inference pipeline: encoder, span representations, count prediction,
    count embedding, and dot-product scoring.

    Output shape: [1, num_words, max_width, num_labels]
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
        num_words = 10
        num_labels = 3

        with torch.no_grad():
            dummy = _create_dummy_inputs(1, 64, num_words, num_labels, max_width)
            test_output = wrapper(
                dummy["input_ids"],
                dummy["attention_mask"],
                dummy["words_mask"],
                dummy["span_idx"],
                dummy["span_mask"],
            )
            logger.info(f"Forward pass successful. Output shape: {test_output.shape}")
            assert test_output.shape == (1, num_words, max_width, num_labels), (
                f"Expected (1, {num_words}, {max_width}, {num_labels}), got {test_output.shape}"
            )

        # Prepare export inputs with larger dimensions
        seq_len = self.max_seq_len
        num_words_export = 50
        num_labels_export = 5

        dummy = _create_dummy_inputs(1, seq_len, num_words_export, num_labels_export, max_width)

        # ONNX export
        onnx_path = self.output_dir / "model.onnx"
        logger.info(f"Exporting to ONNX: {onnx_path}")

        input_names = [
            "input_ids",
            "attention_mask",
            "words_mask",
            "span_idx",
            "span_mask",
        ]
        output_names = ["logits"]

        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "words_mask": {0: "batch", 1: "seq_len"},
            "span_idx": {0: "batch", 1: "num_spans"},
            "span_mask": {0: "batch", 1: "num_spans"},
            "logits": {0: "batch", 1: "num_words", 3: "num_labels"},
        }

        try:
            torch.onnx.export(
                wrapper,
                (
                    dummy["input_ids"],
                    dummy["attention_mask"],
                    dummy["words_mask"],
                    dummy["span_idx"],
                    dummy["span_mask"],
                ),
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=self.opset_version,
                do_constant_folding=True,
                export_params=True,
                dynamo=False,
            )
            logger.info("  Saved: model.onnx")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise

        # Verify
        logger.info("Verifying exported ONNX model...")
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info("  ONNX model verification passed")

            model_size = os.path.getsize(onnx_path) / (1024 * 1024)
            logger.info(f"  Model size: {model_size:.1f} MB")
        except Exception as e:
            logger.warning(f"  ONNX verification warning: {e}")

        if "f16" in self.variants:
            convert_to_fp16(onnx_path, self.output_dir / "model_f16.onnx")

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

            logger.info("Model inputs:")
            input_names = set()
            for inp in session.get_inputs():
                logger.info(f"  {inp.name}: {inp.shape} ({inp.type})")
                input_names.add(inp.name)

            logger.info("Model outputs:")
            for out in session.get_outputs():
                logger.info(f"  {out.name}: {out.shape} ({out.type})")

            # Create test inputs matching the structured schema format
            num_words = 10
            num_labels = 3
            num_spans = num_words * max_width

            # Build schema tokens
            schema_tokens = [287, _P_TOKEN_ID, 6967, 287]  # ( [P] entities (
            for i in range(num_labels):
                schema_tokens.extend([_E_TOKEN_ID, 604 + i])
            schema_tokens.extend([1263, 1263, _SEP_TEXT_ID])

            text_tokens = [300 + i for i in range(num_words)]
            all_tokens = schema_tokens + text_tokens
            seq_len = 64
            while len(all_tokens) < seq_len:
                all_tokens.append(0)
            all_tokens = all_tokens[:seq_len]

            text_start = len(schema_tokens)

            all_inputs = {
                "input_ids": np.array([all_tokens], dtype=np.int64),
                "attention_mask": np.zeros((1, seq_len), dtype=np.int64),
                "words_mask": np.zeros((1, seq_len), dtype=np.int64),
                "span_idx": np.zeros((1, num_spans, 2), dtype=np.int64),
                "span_mask": np.zeros((1, num_spans), dtype=np.bool_),
            }

            all_inputs["attention_mask"][0, : text_start + num_words] = 1
            for w in range(num_words):
                pos = text_start + w
                if pos < seq_len:
                    all_inputs["words_mask"][0, pos] = w + 1

            for w in range(num_words):
                for wi in range(max_width):
                    si = w * max_width + wi
                    end_word = w + wi
                    if end_word < num_words:
                        all_inputs["span_idx"][0, si, 0] = w
                        all_inputs["span_idx"][0, si, 1] = end_word
                        all_inputs["span_mask"][0, si] = True

            inputs = {k: v for k, v in all_inputs.items() if k in input_names}
            logger.info(f"Using inputs: {list(inputs.keys())}")

            outputs = session.run(None, inputs)
            logger.info("Inference successful!")
            logger.info(f"Output shape: {outputs[0].shape}")
            logger.info(f"Output dtype: {outputs[0].dtype}")
            logger.info(f"Output range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
