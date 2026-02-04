"""GLiNER zero-shot NER model exporter (v1)."""

import json
import logging
import shutil
from pathlib import Path

from . import register_exporter
from .base import BaseExporter
from .embedder import convert_to_fp16

logger = logging.getLogger(__name__)


@register_exporter("recognizer", capability="labels")
class GLiNERExporter(BaseExporter):
    """Exporter for GLiNER zero-shot NER models.

    GLiNER models can extract any entity type without retraining - just specify
    the entity labels at inference time.
    """

    capabilities = ["labels"]

    def export(self) -> Path:
        from gliner import GLiNER

        logger.info(f"Exporting GLiNER model: {self.model_id}")
        logger.info(f"Output: {self.output_dir}")

        # Load the GLiNER model
        logger.info("Loading GLiNER model...")
        model = GLiNER.from_pretrained(self.model_id)

        # Export to ONNX using GLiNER's built-in method
        logger.info("Exporting to ONNX format...")
        temp_onnx_dir = self.output_dir / "temp_onnx"
        temp_onnx_dir.mkdir(exist_ok=True)

        try:
            # Export using GLiNER's built-in method
            use_int8 = "i8" in self.variants
            model.export_to_onnx(str(temp_onnx_dir), quantize=use_int8)

            # Move the exported files to the final location
            for file in temp_onnx_dir.iterdir():
                if file.is_file():
                    dest = self.output_dir / file.name
                    shutil.move(str(file), str(dest))
                    logger.info(f"  Saved: {file.name}")
        finally:
            if temp_onnx_dir.exists():
                shutil.rmtree(temp_onnx_dir)

        # Rename quantized file if created
        quantized_file = self.output_dir / "model_quantized.onnx"
        if quantized_file.exists():
            quantized_file.rename(self.output_dir / "model_i8.onnx")
            logger.info("  Renamed: model_quantized.onnx -> model_i8.onnx")

        # Apply FP16 conversion if requested
        if "f16" in self.variants:
            onnx_file = self.output_dir / "model.onnx"
            if onnx_file.exists():
                fp16_file = self.output_dir / "model_f16.onnx"
                convert_to_fp16(onnx_file, fp16_file)

        # Create GLiNER config file for Termite
        gliner_config = {
            "max_width": 12,
            "default_labels": ["person", "organization", "location", "date", "product"],
            "threshold": 0.5,
            "flat_ner": True,
            "multi_label": False,
            "model_id": self.model_id,
        }

        config_path = self.output_dir / "gliner_config.json"
        with open(config_path, "w") as f:
            json.dump(gliner_config, f, indent=2)
        logger.info("  Saved: gliner_config.json")

        # Save tokenizer
        try:
            if (
                hasattr(model, "data_processor")
                and hasattr(model.data_processor, "transformer_tokenizer")
            ):
                tokenizer = model.data_processor.transformer_tokenizer
                tokenizer.save_pretrained(str(self.output_dir))
                logger.info("  Saved: tokenizer files")
            elif hasattr(model, "tokenizer"):
                model.tokenizer.save_pretrained(str(self.output_dir))
                logger.info("  Saved: tokenizer files")
            else:
                from transformers import AutoTokenizer

                base_model = "microsoft/deberta-v3-small"
                logger.info(f"  Loading tokenizer from base model: {base_model}")
                tokenizer = AutoTokenizer.from_pretrained(base_model)
                tokenizer.save_pretrained(str(self.output_dir))
                logger.info("  Saved: tokenizer files")
        except Exception as e:
            logger.warning(f"  Could not save tokenizer: {e}")

        return self.output_dir

    def test(self) -> bool:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        try:
            onnx_path = self.output_dir / "model.onnx"
            if not onnx_path.exists():
                logger.error(f"model.onnx not found in {self.output_dir}")
                return False

            logger.info("Loading GLiNER ONNX model...")
            session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )

            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(str(self.output_dir))

            # Test with sample input
            test_text = "Tim Cook is the CEO of Apple Inc. in Cupertino."
            inputs = tokenizer(
                test_text,
                return_tensors="np",
                padding="max_length",
                max_length=128,
                truncation=True,
            )

            logger.info(f'Test input: "{test_text}"')
            logger.info(f"Input shape: {inputs['input_ids'].shape}")
            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
