"""Vision2Seq reader model exporter (TrOCR, Donut, Nougat, Florence-2)."""

import json
import logging
from pathlib import Path

from . import register_exporter
from .base import BaseExporter

logger = logging.getLogger(__name__)

# Reader model type patterns for auto-detection
READER_MODEL_PATTERNS = {
    "trocr": ["trocr", "TrOCR"],
    "donut": ["donut", "Donut"],
    "nougat": ["nougat", "Nougat"],
    "florence": ["florence", "Florence"],
}


def detect_reader_type(model_id: str) -> str:
    """Detect reader model type from model ID.

    Returns:
        Model type: "trocr", "donut", "nougat", "florence", or "generic"
    """
    model_id_lower = model_id.lower()

    for model_type, patterns in READER_MODEL_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in model_id_lower:
                return model_type

    return "generic"


def get_reader_output_format(reader_type: str) -> str:
    """Get the output format for a reader model type."""
    formats = {
        "donut": "json",
        "nougat": "markdown",
        "trocr": "text",
        "florence": "text",
        "generic": "text",
    }
    return formats.get(reader_type, "text")


@register_exporter("reader")
class ReaderExporter(BaseExporter):
    """Exporter for Vision2Seq models (TrOCR, Donut, Nougat, Florence-2).

    Uses Hugging Face's Optimum library to create ONNX files for vision-encoder-decoder models:
      - encoder_model.onnx
      - decoder_model.onnx
      - decoder_with_past_model.onnx
    """

    def __init__(
        self,
        model_id: str,
        output_dir: Path,
        variants: list[str] | None = None,
        trust_remote_code: bool = False,
    ):
        super().__init__(model_id, output_dir, variants)
        self.trust_remote_code = trust_remote_code

    def export(self) -> Path:
        from optimum.onnxruntime import ORTModelForVision2Seq
        from transformers import AutoConfig, AutoProcessor

        logger.info(f"Exporting Vision2Seq reader model: {self.model_id}")
        logger.info(f"Output: {self.output_dir}")

        # Detect model type
        reader_type = detect_reader_type(self.model_id)
        output_format = get_reader_output_format(reader_type)
        logger.info(f"Detected model type: {reader_type}")
        logger.info(f"Output format: {output_format}")

        # Load config for info
        logger.info("Loading model configuration...")
        config = AutoConfig.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )

        # Log model info if available
        if hasattr(config, "encoder"):
            enc = config.encoder
            if hasattr(enc, "image_size"):
                img_size = enc.image_size
                if isinstance(img_size, (list, tuple)):
                    logger.info(f"Image size: {img_size[0]}x{img_size[1]}")
                else:
                    logger.info(f"Image size: {img_size}x{img_size}")
            if hasattr(enc, "model_type"):
                logger.info(f"Encoder type: {enc.model_type}")

        # Export to ONNX using Optimum
        logger.info("Exporting to ONNX format (this may take a few minutes)...")
        logger.info(
            "This will create encoder_model.onnx, decoder_model.onnx, and decoder_with_past_model.onnx"
        )

        ort_model = ORTModelForVision2Seq.from_pretrained(
            self.model_id,
            export=True,
            trust_remote_code=self.trust_remote_code,
        )
        ort_model.save_pretrained(str(self.output_dir))

        # Save processor (tokenizer + image processor)
        logger.info("Saving processor (tokenizer + image processor)...")
        try:
            processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code
            )
            processor.save_pretrained(str(self.output_dir))
        except Exception as e:
            logger.warning(f"Could not save processor: {e}")
            logger.warning("You may need to copy tokenizer files manually.")

        # Create termite_metadata.json for model type detection
        logger.info("Creating termite_metadata.json...")
        metadata = {
            "model_type": reader_type,
            "source_model": self.model_id,
            "export_format": "onnx",
            "framework": "optimum",
            "output_format": output_format,
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
            decoder_path = self.output_dir / "decoder_model.onnx"

            if not encoder_path.exists():
                logger.error("encoder_model.onnx not found")
                return False

            logger.info("Testing encoder model...")
            encoder_session = ort.InferenceSession(
                str(encoder_path), providers=["CPUExecutionProvider"]
            )

            # Get expected input shape from encoder
            encoder_inputs = encoder_session.get_inputs()
            logger.info(f"  Encoder inputs: {[i.name for i in encoder_inputs]}")

            # Create dummy input matching expected shape
            for inp in encoder_inputs:
                if inp.name == "pixel_values":
                    shape = inp.shape
                    # Replace dynamic dims with reasonable values
                    shape = [1 if isinstance(d, str) else d for d in shape]
                    dummy_input = np.random.randn(*shape).astype(np.float32)
                    break
            else:
                logger.warning("Could not find pixel_values input")
                dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

            outputs = encoder_session.run(None, {"pixel_values": dummy_input})
            logger.info(f"  Encoder output shapes: {[o.shape for o in outputs]}")

            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
