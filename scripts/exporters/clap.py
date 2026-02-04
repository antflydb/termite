"""CLAP audio embedding model exporter."""

import json
import logging
from pathlib import Path

from . import register_exporter
from .base import BaseExporter

logger = logging.getLogger(__name__)


@register_exporter("embedder", capability="audio")
class CLAPExporter(BaseExporter):
    """Exporter for CLAP audio+text embedding models.

    CLAP models have separate audio and text encoders that produce embeddings
    in a shared space, enabling cross-modal retrieval.
    """

    capabilities = ["audio"]

    def export(self) -> Path:
        import torch
        import onnx
        from transformers import ClapModel, ClapProcessor

        logger.info(f"Exporting CLAP audio model: {self.model_id}")
        logger.info(f"Output: {self.output_dir}")

        # Load model and processor
        logger.info("Loading CLAP model...")
        model = ClapModel.from_pretrained(self.model_id)
        processor = ClapProcessor.from_pretrained(self.model_id)

        model.eval()

        # Export audio encoder
        logger.info("Exporting audio encoder...")
        audio_path = self.output_dir / "audio_model.onnx"
        sample_rate = processor.feature_extractor.sampling_rate
        max_length = processor.feature_extractor.max_length_s
        num_samples = int(sample_rate * max_length)
        dummy_audio = torch.randn(1, num_samples)

        audio_inputs = processor(
            audios=dummy_audio.numpy(), return_tensors="pt", sampling_rate=sample_rate
        )

        torch.onnx.export(
            model.audio_model,
            (audio_inputs["input_features"],),
            str(audio_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_features"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_features": {0: "batch_size"},
                "last_hidden_state": {0: "batch_size"},
                "pooler_output": {0: "batch_size"},
            },
        )
        onnx_model = onnx.load(str(audio_path))
        onnx.checker.check_model(onnx_model)
        logger.info(f"  Audio encoder saved: {audio_path}")

        # Export text encoder
        logger.info("Exporting text encoder...")
        text_path = self.output_dir / "text_model.onnx"
        dummy_text = ["a sound of a dog barking"]
        text_inputs = processor(text=dummy_text, return_tensors="pt", padding=True)

        torch.onnx.export(
            model.text_model,
            (text_inputs["input_ids"], text_inputs["attention_mask"]),
            str(text_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "last_hidden_state": {0: "batch_size", 1: "sequence"},
                "pooler_output": {0: "batch_size"},
            },
        )
        onnx_model = onnx.load(str(text_path))
        onnx.checker.check_model(onnx_model)
        logger.info(f"  Text encoder saved: {text_path}")

        # Save processor and config files
        processor.save_pretrained(self.output_dir)

        # Create CLAP-specific config
        clap_config = {
            "model_type": "clap",
            "audio_config": {
                "hidden_size": model.config.audio_config.hidden_size,
                "sample_rate": sample_rate,
                "max_length_s": max_length,
            },
            "text_config": {
                "hidden_size": model.config.text_config.hidden_size,
            },
            "projection_dim": model.config.projection_dim,
        }
        with open(self.output_dir / "clap_config.json", "w") as f:
            json.dump(clap_config, f, indent=2)

        # Create int8 quantized variants if requested
        if "i8" in self.variants:
            logger.info("Applying dynamic quantization (int8)...")
            try:
                from onnxruntime.quantization import QuantType, quantize_dynamic

                quantize_dynamic(
                    model_input=str(audio_path),
                    model_output=str(self.output_dir / "audio_model_i8.onnx"),
                    weight_type=QuantType.QUInt8,
                )
                logger.info("  Audio encoder quantized")

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
        from transformers import ClapProcessor

        try:
            audio_path = self.output_dir / "audio_model.onnx"
            text_path = self.output_dir / "text_model.onnx"
            clap_config_path = self.output_dir / "clap_config.json"
            config_path = self.output_dir / "config.json"

            # Load config - try clap_config.json first, fall back to config.json (Xenova)
            if clap_config_path.exists():
                with open(clap_config_path) as f:
                    clap_config = json.load(f)
                expected_audio_dim = clap_config["audio_config"]["hidden_size"]
                expected_text_dim = clap_config["text_config"]["hidden_size"]
                projection_dim = clap_config["projection_dim"]
            elif config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                expected_audio_dim = config.get("audio_config", {}).get("hidden_size", 768)
                expected_text_dim = config.get("text_config", {}).get("hidden_size", 512)
                projection_dim = config.get("projection_dim", 512)
            else:
                raise FileNotFoundError("No CLAP config found")

            processor = ClapProcessor.from_pretrained(self.output_dir)

            # Test audio encoder
            logger.info("Testing audio encoder...")
            audio_session = ort.InferenceSession(
                str(audio_path), providers=["CPUExecutionProvider"]
            )

            sample_rate = processor.feature_extractor.sampling_rate
            max_length_s = getattr(processor.feature_extractor, "max_length_s", 10)
            num_samples = int(sample_rate * max_length_s)
            dummy_audio = np.random.randn(num_samples).astype(np.float32)

            audio_inputs = processor(
                audio=dummy_audio, return_tensors="np", sampling_rate=sample_rate
            )
            audio_outputs = audio_session.run(
                None, {"input_features": audio_inputs["input_features"]}
            )
            audio_pooler = audio_outputs[1] if len(audio_outputs) > 1 else audio_outputs[0]
            logger.info(f"  Audio outputs: {len(audio_outputs)} tensors")
            logger.info(f"  Audio embedding shape: {audio_pooler.shape}")

            # Test text encoder
            logger.info("Testing text encoder...")
            text_session = ort.InferenceSession(
                str(text_path), providers=["CPUExecutionProvider"]
            )

            text_inputs = processor(
                text=["a sound of a dog barking"], return_tensors="np", padding=True
            )

            input_names = {inp.name for inp in text_session.get_inputs()}
            text_feed = {"input_ids": text_inputs["input_ids"]}
            if "attention_mask" in input_names:
                text_feed["attention_mask"] = text_inputs["attention_mask"]

            text_outputs = text_session.run(None, text_feed)
            text_pooler = text_outputs[1] if len(text_outputs) > 1 else text_outputs[0]
            logger.info(f"  Text outputs: {len(text_outputs)} tensors")
            logger.info(f"  Text embedding shape: {text_pooler.shape}")

            logger.info(f"  Audio hidden size: {expected_audio_dim}")
            logger.info(f"  Text hidden size: {expected_text_dim}")
            logger.info(f"  Projection dimension: {projection_dim}")

            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
