"""Pix2Struct reader model exporter."""

import json
import logging
from pathlib import Path

from . import register_exporter
from .base import BaseExporter

logger = logging.getLogger(__name__)

# Pix2Struct model variants
PIX2STRUCT_MODELS = {
    "google/pix2struct-docvqa-base": "docvqa",
    "google/pix2struct-ocrvqa-base": "ocrvqa",
    "google/pix2struct-infographics-vqa-base": "infographics",
    "google/pix2struct-chartqa-base": "chartqa",
    "google/pix2struct-docvqa-large": "docvqa",
    "google/pix2struct-ocrvqa-large": "ocrvqa",
    "google/pix2struct-infographics-vqa-large": "infographics",
    "google/pix2struct-chartqa-large": "chartqa",
}


def detect_pix2struct_variant(model_id: str) -> str:
    """Detect Pix2Struct variant from model ID.

    Returns:
        Variant: "docvqa", "ocrvqa", "infographics", "chartqa", or "generic"
    """
    if model_id in PIX2STRUCT_MODELS:
        return PIX2STRUCT_MODELS[model_id]

    model_id_lower = model_id.lower()
    if "docvqa" in model_id_lower:
        return "docvqa"
    if "ocrvqa" in model_id_lower:
        return "ocrvqa"
    if "infographics" in model_id_lower:
        return "infographics"
    if "chartqa" in model_id_lower:
        return "chartqa"

    return "generic"


@register_exporter("reader", "pix2struct")
class Pix2StructExporter(BaseExporter):
    """Exporter for Pix2Struct models (DocVQA, ChartQA, OCR-VQA, InfographicsVQA).

    Uses Hugging Face's Optimum library to create ONNX files for the
    encoder-decoder architecture:
      - encoder_model.onnx
      - decoder_model.onnx
      - decoder_with_past_model.onnx
    """

    def export(self) -> Path:
        from optimum.onnxruntime import ORTModelForVision2Seq
        from transformers import AutoProcessor

        logger.info(f"Exporting Pix2Struct model: {self.model_id}")
        logger.info(f"Output: {self.output_dir}")

        # Detect variant
        variant = detect_pix2struct_variant(self.model_id)
        logger.info(f"Detected variant: {variant}")

        # Export to ONNX using Optimum
        logger.info("Exporting to ONNX format...")
        ort_model = ORTModelForVision2Seq.from_pretrained(
            self.model_id,
            export=True,
        )
        ort_model.save_pretrained(str(self.output_dir))

        # Save processor (tokenizer + image processor)
        logger.info("Saving processor...")
        try:
            processor = AutoProcessor.from_pretrained(self.model_id)
            processor.save_pretrained(str(self.output_dir))
        except Exception as e:
            logger.warning(f"Could not save processor: {e}")

        # Create termite_metadata.json
        logger.info("Creating termite_metadata.json...")
        metadata = {
            "model_type": "pix2struct",
            "source_model": self.model_id,
            "export_format": "onnx",
            "framework": "optimum",
            "output_format": "text",
            "variant": variant,
        }

        metadata_path = self.output_dir / "termite_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("  Saved: termite_metadata.json")

        # Log exported files
        logger.info("\nExported files:")
        for f in sorted(self.output_dir.iterdir()):
            if f.is_file():
                size = f.stat().st_size / (1024 * 1024)
                logger.info(f"  {f.name}: {size:.1f} MB")

        return self.output_dir

    def test(self) -> bool:
        import onnxruntime as ort
        import numpy as np

        try:
            encoder_path = self.output_dir / "encoder_model.onnx"
            if not encoder_path.exists():
                logger.error("encoder_model.onnx not found")
                return False

            logger.info("Testing encoder model...")
            encoder_session = ort.InferenceSession(
                str(encoder_path), providers=["CPUExecutionProvider"]
            )

            encoder_inputs = encoder_session.get_inputs()
            logger.info(f"  Encoder inputs: {[i.name for i in encoder_inputs]}")

            # Pix2Struct uses variable-size patches via flattened_patches input
            for inp in encoder_inputs:
                if inp.name == "flattened_patches":
                    # Shape: [batch, num_patches, patch_size]
                    shape = [1 if isinstance(d, str) else d for d in inp.shape]
                    # Default to reasonable patch count if dynamic
                    if isinstance(inp.shape[1], str):
                        shape[1] = 512
                    dummy_input = np.random.randn(*shape).astype(np.float32)
                    attention_mask = np.ones(
                        (shape[0], shape[1]), dtype=np.int64
                    )
                    outputs = encoder_session.run(
                        None,
                        {
                            "flattened_patches": dummy_input,
                            "attention_mask": attention_mask,
                        },
                    )
                    logger.info(
                        f"  Encoder output shapes: {[o.shape for o in outputs]}"
                    )
                    logger.info("Test passed!")
                    return True
                elif inp.name == "pixel_values":
                    shape = [1 if isinstance(d, str) else d for d in inp.shape]
                    dummy_input = np.random.randn(*shape).astype(np.float32)
                    outputs = encoder_session.run(
                        None, {"pixel_values": dummy_input}
                    )
                    logger.info(
                        f"  Encoder output shapes: {[o.shape for o in outputs]}"
                    )
                    logger.info("Test passed!")
                    return True

            logger.warning("Could not find expected input")
            return False

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
