"""Standard embedder/reranker/chunker exporter using Optimum."""

import logging
from pathlib import Path

from . import register_exporter
from .base import BaseExporter

logger = logging.getLogger(__name__)

# Models that require trust_remote_code=True
TRUST_REMOTE_CODE_MODELS = {
    "nomic-ai/nomic-embed-text-v1.5",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "dunzhang/stella_en_1.5B_v5",
    "google/embeddinggemma-300m",
}


def get_ort_model_class(model_type: str):
    """Get the appropriate ORT model class for the model type."""
    from optimum.onnxruntime import (
        ORTModelForFeatureExtraction,
        ORTModelForSequenceClassification,
        ORTModelForTokenClassification,
    )

    classes = {
        "embedder": ORTModelForFeatureExtraction,
        "reranker": ORTModelForSequenceClassification,
        "chunker": ORTModelForTokenClassification,
    }
    return classes[model_type]


def convert_to_fp16(input_path: Path, output_path: Path) -> None:
    """Convert an ONNX model from FP32 to FP16 precision."""
    import onnx
    from onnxconverter_common import float16

    logger.info(f"Converting to FP16: {input_path.name} -> {output_path.name}")

    model = onnx.load(str(input_path))

    # Check for large initializers that may need external data
    has_large_initializers = any(
        len(init.raw_data) > 100_000_000  # 100MB threshold
        for init in model.graph.initializer
    )

    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,
    )

    if has_large_initializers:
        onnx.save(
            model_fp16,
            str(output_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{output_path.name}_data",
        )
    else:
        onnx.save(model_fp16, str(output_path))


class OptimumExporter(BaseExporter):
    """Base class for Optimum-based exporters."""

    ort_class_name: str = "ORTModelForFeatureExtraction"

    def export(self) -> Path:
        from transformers import AutoTokenizer
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        logger.info(f"Exporting {self.model_type}: {self.model_id}")
        logger.info(f"Output: {self.output_dir}")

        trust_remote_code = self.model_id in TRUST_REMOTE_CODE_MODELS
        if trust_remote_code:
            logger.info("Model requires trust_remote_code=True")

        ort_class = get_ort_model_class(self.model_type)

        logger.info("Converting to ONNX format...")
        ort_model = ort_class.from_pretrained(
            self.model_id, export=True, trust_remote_code=trust_remote_code
        )
        ort_model.save_pretrained(self.output_dir)

        logger.info("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=trust_remote_code
        )
        tokenizer.save_pretrained(self.output_dir)

        # Create i8 variant (must be done BEFORE fp16 to avoid multi-file issue)
        if "i8" in self.variants:
            logger.info("Applying dynamic quantization (int8)...")
            try:
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

            ort_class = get_ort_model_class(self.model_type)
            model = ort_class.from_pretrained(self.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.output_dir)

            test_text = self._get_test_text()
            inputs = tokenizer(
                [test_text],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = model(**inputs)

            self._validate_outputs(outputs)
            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False

    def _get_test_text(self) -> str:
        return "Test sentence for embedding."

    def _validate_outputs(self, outputs):
        assert outputs.last_hidden_state is not None
        logger.info(f"  Embedding shape: {outputs.last_hidden_state.shape}")


@register_exporter("embedder")
class EmbedderExporter(OptimumExporter):
    """Exporter for text embedding models."""

    def _get_test_text(self) -> str:
        return "Test sentence for embedding."

    def _validate_outputs(self, outputs):
        assert outputs.last_hidden_state is not None
        logger.info(f"  Embedding shape: {outputs.last_hidden_state.shape}")


@register_exporter("reranker")
class RerankerExporter(OptimumExporter):
    """Exporter for reranker models."""

    def test(self) -> bool:
        try:
            from transformers import AutoTokenizer

            ort_class = get_ort_model_class(self.model_type)
            model = ort_class.from_pretrained(self.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.output_dir)

            inputs = tokenizer(
                [["Query", "Document to rerank"]],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = model(**inputs)

            assert outputs.logits is not None
            logger.info(f"  Logits shape: {outputs.logits.shape}")
            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False


@register_exporter("chunker")
class ChunkerExporter(OptimumExporter):
    """Exporter for text chunking models."""

    def _get_test_text(self) -> str:
        return "Test text for chunking."

    def _validate_outputs(self, outputs):
        assert outputs.logits is not None
        logger.info(f"  Logits shape: {outputs.logits.shape}")
