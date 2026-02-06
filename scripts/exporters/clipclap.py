"""CLIPCLAP unified image+audio embedding model exporter.

Downloads a pre-built combined model from HuggingFace that was assembled by
build_clipclap.py from CLIP (text+image) and CLAP (audio) with a trained
projection layer mapping CLAP audio embeddings into CLIP embedding space.
This enables text, image, and audio search in a single 512-dim embedding space.

The build_clipclap.py script handles:
  - Loading CLIP and CLAP source models
  - Training the audio projection on thousands of captions
  - Exporting all ONNX files
  - Pushing the assembled model to HuggingFace

This exporter just downloads the pre-built ONNX files and validates them.
"""

import json
import logging
from pathlib import Path

from . import register_exporter
from .base import BaseExporter

logger = logging.getLogger(__name__)


@register_exporter("embedder", capability="audio,image")
class CLIPCLAPExporter(BaseExporter):
    """Exporter for combined CLIP+CLAP (CLIPCLAP) embedding models.

    Downloads a pre-built unified model supporting text, image, and audio
    embedding in a single shared space. The model is built and published
    by build_clipclap.py.

    The model directory contains:
      - text_model.onnx (from CLIP text encoder)
      - visual_model.onnx (from CLIP visual encoder)
      - visual_projection.onnx (from CLIP)
      - text_projection.onnx (from CLIP)
      - audio_model.onnx (from CLAP audio encoder)
      - audio_projection.onnx (trained CLAP→CLIP projection)
      - tokenizer.json, preprocessor_config.json (from CLIP)
      - clip_config.json (combined config with vision, text, and audio sections)
    """

    capabilities = ["image", "audio"]

    def export(self) -> Path:
        """Download pre-built CLIPCLAP ONNX model from HuggingFace.

        CLIPCLAP models are assembled and trained by build_clipclap.py, not
        exported from a single PyTorch checkpoint. This delegates to
        download_onnx() which pulls the pre-built files.
        """
        return self.download_onnx()

    def test(self) -> bool:
        import onnxruntime as ort
        import numpy as np
        from transformers import CLIPProcessor, CLIPTokenizerFast

        try:
            visual_path = self.output_dir / "visual_model.onnx"
            text_path = self.output_dir / "text_model.onnx"
            audio_path = self.output_dir / "audio_model.onnx"
            audio_proj_path = self.output_dir / "audio_projection.onnx"
            config_path = self.output_dir / "clip_config.json"

            # Verify all files exist
            required = [visual_path, text_path, audio_path, audio_proj_path, config_path]
            for p in required:
                if not p.exists():
                    logger.error(f"Missing required file: {p}")
                    return False

            with open(config_path) as f:
                config = json.load(f)

            # Test visual encoder
            logger.info("Testing visual encoder...")
            visual_session = ort.InferenceSession(str(visual_path), providers=["CPUExecutionProvider"])
            processor = CLIPProcessor.from_pretrained(self.output_dir)
            image_size = processor.image_processor.size.get("shortest_edge", 224)
            if isinstance(image_size, dict):
                image_size = image_size.get("height", 224)

            from PIL import Image
            dummy_image = Image.new("RGB", (image_size, image_size), color="red")
            pixel_values = processor(images=dummy_image, return_tensors="np")["pixel_values"]
            visual_outputs = visual_session.run(None, {"pixel_values": pixel_values})
            logger.info(f"  Visual encoder output shape: {visual_outputs[1].shape}")

            # Test text encoder
            logger.info("Testing text encoder...")
            text_session = ort.InferenceSession(str(text_path), providers=["CPUExecutionProvider"])
            tokenizer = CLIPTokenizerFast.from_pretrained(self.output_dir)
            text_inputs = tokenizer(
                ["a test caption"],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="np",
            )
            text_outputs = text_session.run(
                None,
                {"input_ids": text_inputs["input_ids"], "attention_mask": text_inputs["attention_mask"]},
            )
            logger.info(f"  Text encoder output shape: {text_outputs[1].shape}")

            # Test audio encoder
            logger.info("Testing audio encoder...")
            audio_session = ort.InferenceSession(str(audio_path), providers=["CPUExecutionProvider"])
            audio_hidden = config["audio_config"]["hidden_size"]
            dummy_audio_features = np.random.randn(1, 1, 1001, 64).astype(np.float32)
            audio_outputs = audio_session.run(None, {"input_features": dummy_audio_features})
            logger.info(f"  Audio encoder output shape: {audio_outputs[1].shape}")

            # Test audio projection
            logger.info("Testing audio projection...")
            proj_session = ort.InferenceSession(str(audio_proj_path), providers=["CPUExecutionProvider"])
            dummy_proj_input = np.random.randn(1, audio_hidden).astype(np.float32)
            proj_output = proj_session.run(None, {"input": dummy_proj_input})
            logger.info(f"  Audio projection output shape: {proj_output[0].shape}")

            expected_dim = config["projection_dim"]
            if proj_output[0].shape[-1] != expected_dim:
                logger.error(f"Projection dim {proj_output[0].shape[-1]} != expected {expected_dim}")
                return False

            logger.info("All tests passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
