"""CLIP visual embedding model exporter."""

import json
import logging
from pathlib import Path

from . import register_exporter
from .base import BaseExporter

logger = logging.getLogger(__name__)


@register_exporter("embedder", capability="image")
class CLIPExporter(BaseExporter):
    """Exporter for CLIP image+text embedding models.

    CLIP models have separate visual and text encoders that produce embeddings
    in a shared space, enabling cross-modal retrieval.
    """

    capabilities = ["image"]

    def export(self) -> Path:
        import torch
        import onnx
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

        logger.info(f"Exporting CLIP image model: {self.model_id}")
        logger.info(f"Output: {self.output_dir}")

        # Load model and processor
        logger.info("Loading CLIP model...")
        model = CLIPModel.from_pretrained(self.model_id)
        processor = CLIPProcessor.from_pretrained(self.model_id)
        tokenizer = CLIPTokenizerFast.from_pretrained(self.model_id)

        model.eval()

        # Get image size from processor
        image_size = processor.image_processor.size.get("shortest_edge", 224)
        if isinstance(image_size, dict):
            image_size = image_size.get("height", 224)

        # Export visual encoder
        logger.info("Exporting visual encoder...")
        visual_path = self.output_dir / "visual_model.onnx"
        dummy_pixel_values = torch.randn(1, 3, image_size, image_size)

        torch.onnx.export(
            model.vision_model,
            (dummy_pixel_values,),
            str(visual_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "last_hidden_state": {0: "batch_size"},
                "pooler_output": {0: "batch_size"},
            },
        )
        onnx_model = onnx.load(str(visual_path))
        onnx.checker.check_model(onnx_model)
        logger.info(f"  Visual encoder saved: {visual_path}")

        # Export text encoder
        logger.info("Exporting text encoder...")
        text_path = self.output_dir / "text_model.onnx"
        dummy_text = ["a photo of a cat"]
        inputs = tokenizer(
            dummy_text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        torch.onnx.export(
            model.text_model,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(text_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
                "pooler_output": {0: "batch_size"},
            },
        )
        onnx_model = onnx.load(str(text_path))
        onnx.checker.check_model(onnx_model)
        logger.info(f"  Text encoder saved: {text_path}")

        # Save configs
        logger.info("Saving configuration files...")
        model.config.save_pretrained(self.output_dir)
        processor.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        # Create CLIP-specific config
        clip_config = {
            "model_type": "clip",
            "vision_config": {
                "hidden_size": model.config.vision_config.hidden_size,
                "image_size": model.config.vision_config.image_size,
                "patch_size": model.config.vision_config.patch_size,
                "projection_dim": model.config.projection_dim,
            },
            "text_config": {
                "hidden_size": model.config.text_config.hidden_size,
                "max_position_embeddings": model.config.text_config.max_position_embeddings,
                "projection_dim": model.config.projection_dim,
            },
            "projection_dim": model.config.projection_dim,
        }
        with open(self.output_dir / "clip_config.json", "w") as f:
            json.dump(clip_config, f, indent=2)

        # Export projection layers as ONNX
        logger.info("Exporting projection layers...")

        # Visual projection
        visual_proj_path = self.output_dir / "visual_projection.onnx"
        dummy_visual = torch.randn(1, model.config.vision_config.hidden_size)
        torch.onnx.export(
            model.visual_projection,
            dummy_visual,
            str(visual_proj_path),
            export_params=True,
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        logger.info(f"  visual_projection.onnx: {model.visual_projection.weight.shape}")

        # Text projection
        text_proj_path = self.output_dir / "text_projection.onnx"
        dummy_text = torch.randn(1, model.config.text_config.hidden_size)
        torch.onnx.export(
            model.text_projection,
            dummy_text,
            str(text_proj_path),
            export_params=True,
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        logger.info(f"  text_projection.onnx: {model.text_projection.weight.shape}")

        # Create FP16 variants if requested
        if "f16" in self.variants:
            logger.info("Creating FP16 variants...")
            try:
                from .embedder import convert_to_fp16

                convert_to_fp16(visual_path, self.output_dir / "visual_model_f16.onnx")
                logger.info("  Visual encoder converted to FP16")

                convert_to_fp16(text_path, self.output_dir / "text_model_f16.onnx")
                logger.info("  Text encoder converted to FP16")
            except Exception as e:
                logger.warning(f"FP16 conversion failed: {e}")

        # Create int8 quantized variants if requested
        if "i8" in self.variants:
            logger.info("Applying dynamic quantization (int8)...")
            try:
                from onnxruntime.quantization import QuantType, quantize_dynamic

                quantize_dynamic(
                    model_input=str(visual_path),
                    model_output=str(self.output_dir / "visual_model_i8.onnx"),
                    weight_type=QuantType.QUInt8,
                )
                logger.info("  Visual encoder quantized")

                quantize_dynamic(
                    model_input=str(text_path),
                    model_output=str(self.output_dir / "text_model_i8.onnx"),
                    weight_type=QuantType.QUInt8,
                )
                logger.info("  Text encoder quantized")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")

        return self.output_dir

    def test(self) -> bool:
        import onnxruntime as ort
        import numpy as np
        from PIL import Image
        from transformers import CLIPProcessor, CLIPTokenizerFast

        try:
            visual_path = self.output_dir / "visual_model.onnx"
            text_path = self.output_dir / "text_model.onnx"
            clip_config_path = self.output_dir / "clip_config.json"

            # Load CLIP config for expected dimensions
            with open(clip_config_path) as f:
                clip_config = json.load(f)

            expected_visual_dim = clip_config["vision_config"]["hidden_size"]
            expected_text_dim = clip_config["text_config"]["hidden_size"]

            # Load processor and tokenizer
            processor = CLIPProcessor.from_pretrained(self.output_dir)
            tokenizer = CLIPTokenizerFast.from_pretrained(self.output_dir)

            # Test visual encoder
            logger.info("Testing visual encoder...")
            visual_session = ort.InferenceSession(
                str(visual_path), providers=["CPUExecutionProvider"]
            )

            image_size = processor.image_processor.size.get("shortest_edge", 224)
            if isinstance(image_size, dict):
                image_size = image_size.get("height", 224)

            dummy_image = Image.new("RGB", (image_size, image_size), color="red")
            pixel_values = processor(images=dummy_image, return_tensors="np")["pixel_values"]

            visual_outputs = visual_session.run(None, {"pixel_values": pixel_values})
            pooler_output = visual_outputs[1]
            logger.info(f"  Visual embedding shape: {pooler_output.shape}")

            if pooler_output.shape[-1] != expected_visual_dim:
                logger.error(
                    f"Visual dim {pooler_output.shape[-1]} != expected {expected_visual_dim}"
                )
                return False

            # Test text encoder
            logger.info("Testing text encoder...")
            text_session = ort.InferenceSession(
                str(text_path), providers=["CPUExecutionProvider"]
            )

            text_inputs = tokenizer(
                ["a photo of a cat"],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="np",
            )

            text_outputs = text_session.run(
                None,
                {
                    "input_ids": text_inputs["input_ids"],
                    "attention_mask": text_inputs["attention_mask"],
                },
            )
            text_pooler = text_outputs[1]
            logger.info(f"  Text embedding shape: {text_pooler.shape}")

            if text_pooler.shape[-1] != expected_text_dim:
                logger.error(
                    f"Text dim {text_pooler.shape[-1]} != expected {expected_text_dim}"
                )
                return False

            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
