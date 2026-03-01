"""Sparse embedder (SPLADE) exporter using ORTModelForMaskedLM."""

import logging
from pathlib import Path

from . import register_exporter
from .base import BaseExporter
from .embedder import convert_to_fp16

logger = logging.getLogger(__name__)


@register_exporter("embedder", "sparse")
class SparseEmbedderExporter(BaseExporter):
    """Exporter for sparse (SPLADE) embedding models.

    SPLADE models are BertForMaskedLM architectures that output logits over the
    full vocabulary. The logits are post-processed with softplus activation and
    top-k sparsification to produce sparse embedding vectors.

    Unlike standard embedders which use ORTModelForFeatureExtraction (hidden states),
    sparse models must use ORTModelForMaskedLM to preserve the MLM head that maps
    hidden states to vocabulary-dimension logits.
    """

    def export(self) -> Path:
        from transformers import AutoTokenizer
        from optimum.onnxruntime import ORTModelForMaskedLM

        logger.info(f"Exporting sparse embedder: {self.model_id}")
        logger.info(f"Output: {self.output_dir}")

        logger.info("Converting to ONNX format (MaskedLM)...")
        ort_model = ORTModelForMaskedLM.from_pretrained(self.model_id, export=True)
        ort_model.save_pretrained(self.output_dir)

        logger.info("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.save_pretrained(self.output_dir)

        # Create i8 variant (must be done BEFORE fp16)
        if "i8" in self.variants:
            logger.info("Applying dynamic quantization (int8)...")
            try:
                from optimum.onnxruntime import ORTQuantizer
                from optimum.onnxruntime.configuration import AutoQuantizationConfig

                quantizer = ORTQuantizer.from_pretrained(self.output_dir)
                dqconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
                quantizer.quantize(save_dir=self.output_dir, quantization_config=dqconfig)
                old_quantized = self.output_dir / "model_quantized.onnx"
                new_quantized = self.output_dir / "model_i8.onnx"
                if old_quantized.exists():
                    old_quantized.rename(new_quantized)
                logger.info("Quantization complete")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")

        # Create FP16 variant
        if "f16" in self.variants:
            logger.info("Creating FP16 variant...")
            try:
                fp32_path = self.output_dir / "model.onnx"
                fp16_path = self.output_dir / "model_f16.onnx"
                convert_to_fp16(fp32_path, fp16_path)
                logger.info("FP16 conversion complete")
            except Exception as e:
                logger.warning(f"FP16 conversion failed: {e}")

        return self.output_dir

    def test(self) -> bool:
        try:
            from transformers import AutoTokenizer
            from optimum.onnxruntime import ORTModelForMaskedLM
            import torch

            logger.info(f"Testing sparse embedder: {self.output_dir}")

            model = ORTModelForMaskedLM.from_pretrained(self.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.output_dir)

            test_text = "Machine learning is a subset of artificial intelligence."
            inputs = tokenizer(
                [test_text],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = model(**inputs)

            assert outputs.logits is not None, "Expected logits output from MaskedLM model"
            batch_size, seq_len, vocab_size = outputs.logits.shape
            logger.info(f"  Logits shape: [{batch_size}, {seq_len}, {vocab_size}]")

            assert batch_size == 1, f"Expected batch_size=1, got {batch_size}"
            assert vocab_size > 1000, f"Expected vocab_size > 1000, got {vocab_size}"

            # Verify SPLADE activation produces reasonable sparse vectors:
            # max-pool over sequence, apply softplus, take top-k
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            pooled = logits.max(dim=0).values  # [vocab_size]
            activated = torch.log1p(torch.exp(pooled))  # softplus
            top_k = min(256, vocab_size)
            topk_values, topk_indices = torch.topk(activated, top_k)

            non_zero = (topk_values > 0).sum().item()
            logger.info(f"  Top-{top_k} non-zero entries: {non_zero}")
            logger.info(f"  Max activation: {topk_values[0].item():.4f}")
            logger.info(f"  Min top-k activation: {topk_values[-1].item():.4f}")

            assert non_zero > 0, "Expected at least some non-zero activations"

            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
