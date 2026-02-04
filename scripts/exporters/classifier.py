"""Zero-shot classifier model exporter."""

import json
import logging
from pathlib import Path

from . import register_exporter
from .base import BaseExporter

logger = logging.getLogger(__name__)


@register_exporter("classifier")
class ClassifierExporter(BaseExporter):
    """Exporter for zero-shot classification models (NLI-based).

    Uses Hugging Face's Optimum library to export models like mDeBERTa-mnli-xnli.
    Creates a zsc_config.json file with classification-specific settings.
    """

    def export(self) -> Path:
        from transformers import AutoTokenizer, AutoConfig
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        logger.info(f"Exporting zero-shot classifier: {self.model_id}")
        logger.info(f"Output: {self.output_dir}")

        # Load model config first to check label mapping
        logger.info("Loading model configuration...")
        config = AutoConfig.from_pretrained(self.model_id)

        # Export to ONNX
        logger.info("Converting to ONNX format...")
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            self.model_id, export=True
        )
        ort_model.save_pretrained(self.output_dir)

        # Save tokenizer
        logger.info("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.save_pretrained(self.output_dir)

        # Create int8 quantized variant if requested
        if "i8" in self.variants:
            logger.info("Applying dynamic quantization (int8)...")
            try:
                quantizer = ORTQuantizer.from_pretrained(self.output_dir)
                dqconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
                quantizer.quantize(save_dir=self.output_dir, quantization_config=dqconfig)
                # Rename the quantized file
                old_quantized = self.output_dir / "model_quantized.onnx"
                new_quantized = self.output_dir / "model_i8.onnx"
                if old_quantized.exists():
                    old_quantized.rename(new_quantized)
                logger.info("Quantization complete")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")

        # Create FP16 variant if requested
        if "f16" in self.variants:
            logger.info("Creating FP16 variant...")
            try:
                from .embedder import convert_to_fp16

                fp32_path = self.output_dir / "model.onnx"
                fp16_path = self.output_dir / "model_f16.onnx"
                convert_to_fp16(fp32_path, fp16_path)
                logger.info("FP16 conversion complete")
            except Exception as e:
                logger.warning(f"FP16 conversion failed: {e}")

        # Create ZSC config with label mapping
        zsc_config = {
            "model_type": "zero-shot-classification",
            "hypothesis_template": "This example is {}.",
            "multi_label": False,
        }

        # Add label mapping if available
        if hasattr(config, "id2label") and config.id2label:
            zsc_config["id2label"] = config.id2label
            zsc_config["label2id"] = config.label2id
            logger.info(f"Label mapping: {config.id2label}")

        config_path = self.output_dir / "zsc_config.json"
        with open(config_path, "w") as f:
            json.dump(zsc_config, f, indent=2)
        logger.info(f"Created: {config_path}")

        return self.output_dir

    def test(self) -> bool:
        from transformers import AutoTokenizer
        from optimum.onnxruntime import ORTModelForSequenceClassification

        try:
            logger.info("Testing zero-shot classifier...")

            model = ORTModelForSequenceClassification.from_pretrained(self.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.output_dir)

            # Test with NLI-style input
            premise = "I love playing soccer"
            hypothesis = "This example is about sports."
            inputs = tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
            )

            outputs = model(**inputs)
            logger.info(f"  Logits shape: {outputs.logits.shape}")
            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
