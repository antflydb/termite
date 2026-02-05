"""CLIPCLAP unified image+audio embedding model exporter.

Assembles a combined model from CLIP (text+image) and CLAP (audio) with a trained
projection layer that maps CLAP audio embeddings into CLIP embedding space.
This enables text, image, and audio search in a single 512-dim embedding space.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

from . import register_exporter
from .base import BaseExporter

logger = logging.getLogger(__name__)


@register_exporter("embedder", capability="image,audio")
class CLIPCLAPExporter(BaseExporter):
    """Exporter for combined CLIP+CLAP (CLIPCLAP) embedding models.

    Assembles a unified model supporting text, image, and audio embedding
    in a single shared space. Uses CLIP for text+image and a trained projection
    to map CLAP audio embeddings into CLIP space.

    The assembled model directory contains:
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
        import torch
        import onnx
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast, ClapModel, ClapProcessor

        clip_model_id = "openai/clip-vit-base-patch32"
        clap_model_id = "laion/larger_clap_music_and_speech"

        logger.info(f"Exporting CLIPCLAP model: {self.model_id}")
        logger.info(f"  CLIP source: {clip_model_id}")
        logger.info(f"  CLAP source: {clap_model_id}")
        logger.info(f"  Output: {self.output_dir}")

        # Load CLIP
        logger.info("Loading CLIP model...")
        clip_model = CLIPModel.from_pretrained(clip_model_id)
        clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        clip_tokenizer = CLIPTokenizerFast.from_pretrained(clip_model_id)
        clip_model.eval()

        # Load CLAP
        logger.info("Loading CLAP model...")
        clap_model = ClapModel.from_pretrained(clap_model_id)
        clap_processor = ClapProcessor.from_pretrained(clap_model_id)
        clap_model.eval()

        image_size = clip_processor.image_processor.size.get("shortest_edge", 224)
        if isinstance(image_size, dict):
            image_size = image_size.get("height", 224)

        # === Export CLIP visual encoder ===
        logger.info("Exporting CLIP visual encoder...")
        visual_path = self.output_dir / "visual_model.onnx"
        dummy_pixel_values = torch.randn(1, 3, image_size, image_size)
        torch.onnx.export(
            clip_model.vision_model,
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
        onnx.checker.check_model(onnx.load(str(visual_path)))
        logger.info(f"  visual_model.onnx saved")

        # === Export CLIP text encoder ===
        logger.info("Exporting CLIP text encoder...")
        text_path = self.output_dir / "text_model.onnx"
        inputs = clip_tokenizer(
            ["a photo of a cat"],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        torch.onnx.export(
            clip_model.text_model,
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
        onnx.checker.check_model(onnx.load(str(text_path)))
        logger.info(f"  text_model.onnx saved")

        # === Export CLIP visual projection ===
        logger.info("Exporting CLIP visual projection...")
        visual_proj_path = self.output_dir / "visual_projection.onnx"
        dummy_visual = torch.randn(1, clip_model.config.vision_config.hidden_size)
        torch.onnx.export(
            clip_model.visual_projection,
            dummy_visual,
            str(visual_proj_path),
            export_params=True,
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logger.info(f"  visual_projection.onnx saved")

        # === Export CLIP text projection ===
        logger.info("Exporting CLIP text projection...")
        text_proj_path = self.output_dir / "text_projection.onnx"
        dummy_text = torch.randn(1, clip_model.config.text_config.hidden_size)
        torch.onnx.export(
            clip_model.text_projection,
            dummy_text,
            str(text_proj_path),
            export_params=True,
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logger.info(f"  text_projection.onnx saved")

        # === Export CLAP audio encoder ===
        logger.info("Exporting CLAP audio encoder...")
        audio_path = self.output_dir / "audio_model.onnx"
        sample_rate = clap_processor.feature_extractor.sampling_rate
        max_length = clap_processor.feature_extractor.max_length_s
        num_samples = int(sample_rate * max_length)
        dummy_audio = torch.randn(1, num_samples)
        audio_inputs = clap_processor(
            audio=dummy_audio.numpy(), return_tensors="pt", sampling_rate=sample_rate
        )
        torch.onnx.export(
            clap_model.audio_model,
            (audio_inputs["input_features"],),
            str(audio_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_features"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_features": {0: "batch_size", 2: "time"},
                "last_hidden_state": {0: "batch_size"},
                "pooler_output": {0: "batch_size"},
            },
        )
        onnx.checker.check_model(onnx.load(str(audio_path)))
        logger.info(f"  audio_model.onnx saved")

        # === Train or locate audio projection ===
        audio_proj_path = self.output_dir / "audio_projection.onnx"
        if not audio_proj_path.exists():
            logger.info("Training audio projection (CLAP→CLIP)...")
            self._train_projection(clip_model, clip_tokenizer, clap_model, clap_processor, audio_proj_path)
        else:
            logger.info(f"  audio_projection.onnx already exists, skipping training")

        # === Save configs and processors ===
        logger.info("Saving configuration files...")
        clip_processor.save_pretrained(self.output_dir)
        clip_tokenizer.save_pretrained(self.output_dir)

        # Combined config with all three modality sections
        clipclap_config = {
            "model_type": "clipclap",
            "vision_config": {
                "hidden_size": clip_model.config.vision_config.hidden_size,
                "image_size": clip_model.config.vision_config.image_size,
                "patch_size": clip_model.config.vision_config.patch_size,
                "projection_dim": clip_model.config.projection_dim,
            },
            "text_config": {
                "hidden_size": clip_model.config.text_config.hidden_size,
                "max_position_embeddings": clip_model.config.text_config.max_position_embeddings,
                "projection_dim": clip_model.config.projection_dim,
            },
            "audio_config": {
                "hidden_size": clap_model.config.audio_config.hidden_size,
                "sample_rate": sample_rate,
                "max_length_s": max_length,
                "projection_dim": clap_model.config.projection_dim,
            },
            "projection_dim": clip_model.config.projection_dim,
        }
        with open(self.output_dir / "clip_config.json", "w") as f:
            json.dump(clipclap_config, f, indent=2)

        # Create int8 quantized variants if requested
        if "i8" in self.variants:
            logger.info("Applying dynamic quantization (int8)...")
            try:
                from onnxruntime.quantization import QuantType, quantize_dynamic

                for name in ["visual_model", "text_model", "audio_model"]:
                    src = self.output_dir / f"{name}.onnx"
                    dst = self.output_dir / f"{name}_quantized.onnx"
                    quantize_dynamic(
                        model_input=str(src),
                        model_output=str(dst),
                        weight_type=QuantType.QUInt8,
                    )
                    logger.info(f"  {name} quantized")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")

        return self.output_dir

    def _train_projection(self, clip_model, clip_tokenizer, clap_model, clap_processor, output_path: Path):
        """Train the audio projection layer on-the-fly using text bridging."""
        import torch
        import torch.nn as nn
        import onnx

        device = "cpu"
        clip_model = clip_model.to(device)
        clap_model = clap_model.to(device)

        # Use a small set of diverse captions for training
        captions = [
            "a photo of a cat", "a dog playing in the park", "a sunset over the ocean",
            "a person riding a bicycle", "a city skyline at night", "a bird flying in the sky",
            "a car driving on a highway", "a child playing with toys", "a mountain landscape",
            "a forest with tall trees", "a river flowing through a valley",
            "a group of people at a concert", "a piano being played", "drums and guitar music",
            "a thunderstorm with lightning", "waves crashing on a beach",
            "a crowd cheering at a sports game", "wind blowing through leaves",
            "a helicopter flying overhead", "birds chirping in the morning",
            "a train passing by", "a baby laughing", "fireworks in the night sky",
            "a waterfall in a tropical forest", "rain falling on a tin roof",
            "a dog barking loudly", "a cat meowing", "an alarm clock ringing",
            "footsteps on gravel", "a door creaking open",
            "a busy street with traffic", "a quiet library", "a construction site",
            "a restaurant kitchen", "an airplane taking off",
        ]

        # Encode with both models using lower-level API
        with torch.no_grad():
            clip_inputs = clip_tokenizer(captions, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            clip_text_outputs = clip_model.text_model(
                input_ids=clip_inputs["input_ids"],
                attention_mask=clip_inputs["attention_mask"],
            )
            clip_embs = clip_model.text_projection(clip_text_outputs[1])
            clip_embs = clip_embs / clip_embs.norm(dim=-1, keepdim=True)

            clap_inputs = clap_processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)
            clap_text_outputs = clap_model.text_model(
                input_ids=clap_inputs["input_ids"],
                attention_mask=clap_inputs.get("attention_mask"),
            )
            clap_embs = clap_model.text_projection(clap_text_outputs[1])
            clap_embs = clap_embs / clap_embs.norm(dim=-1, keepdim=True)

        embed_dim = clip_embs.shape[1]

        # Train linear projection
        projection = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(projection.weight)
        optimizer = torch.optim.Adam(projection.parameters(), lr=1e-3)
        cos_loss = nn.CosineEmbeddingLoss()

        for epoch in range(50):
            projection.train()
            optimizer.zero_grad()
            projected = projection(clap_embs.detach())
            target = torch.ones(len(captions))
            loss = cos_loss(projected, clip_embs.detach(), target)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    sim = nn.functional.cosine_similarity(projection(clap_embs), clip_embs).mean().item()
                logger.info(f"    Epoch {epoch + 1}/50: loss={loss.item():.4f}, cos_sim={sim:.4f}")

        # Combine CLAP's audio_projection (1024→512) with trained projection (512→512)
        # The CLAP audio encoder outputs hidden_size-dim, which needs CLAP's own
        # audio_projection first, then our trained CLAP→CLIP projection.
        class CombinedAudioProjection(nn.Module):
            def __init__(self, clap_audio_proj, trained_proj):
                super().__init__()
                self.clap_audio_proj = clap_audio_proj
                self.trained_proj = trained_proj

            def forward(self, x):
                x = self.clap_audio_proj(x)
                x = self.trained_proj(x)
                return x

        combined = CombinedAudioProjection(clap_model.audio_projection, projection)
        combined.eval()
        clap_hidden = clap_model.config.audio_config.hidden_size
        dummy = torch.randn(1, clap_hidden)
        torch.onnx.export(
            combined,
            dummy,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        onnx.checker.check_model(onnx.load(str(output_path)))
        logger.info(f"    audio_projection.onnx saved")

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
            for p in [visual_path, text_path, audio_path, audio_proj_path, config_path]:
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
            sample_rate = config["audio_config"]["sample_rate"]
            max_length_s = config["audio_config"]["max_length_s"]
            num_samples = int(sample_rate * max_length_s)
            dummy_audio = np.random.randn(num_samples).astype(np.float32)

            from transformers import ClapProcessor
            clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
            audio_inputs = clap_processor(audio=dummy_audio, return_tensors="np", sampling_rate=sample_rate)
            audio_outputs = audio_session.run(None, {"input_features": audio_inputs["input_features"]})
            logger.info(f"  Audio encoder output shape: {audio_outputs[1].shape}")

            # Test audio projection
            logger.info("Testing audio projection...")
            proj_session = ort.InferenceSession(str(audio_proj_path), providers=["CPUExecutionProvider"])
            dummy_proj_input = np.random.randn(1, config["projection_dim"]).astype(np.float32)
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
