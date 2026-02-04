"""Seq2Seq model exporter (T5, FLAN-T5)."""

import json
import logging
import shutil
from pathlib import Path

from . import register_exporter
from .base import BaseExporter

logger = logging.getLogger(__name__)


@register_exporter("seq2seq")
class Seq2SeqExporter(BaseExporter):
    """Exporter for T5/FLAN-T5/seq2seq models for text generation.

    Uses Hugging Face's Optimum library to create three ONNX files:
      - encoder.onnx
      - decoder-init.onnx (initial decoder, no past_key_values)
      - decoder.onnx (decoder with past_key_values)
    """

    def export(self) -> Path:
        from optimum.exporters.onnx import main_export
        from transformers import AutoTokenizer, AutoConfig

        logger.info(f"Exporting seq2seq model: {self.model_id}")
        logger.info(f"Output: {self.output_dir}")

        # Load model config first to get model info
        logger.info("Loading model configuration...")
        config = AutoConfig.from_pretrained(self.model_id)

        # Export to ONNX using Optimum
        logger.info("Exporting to ONNX format (this may take a few minutes)...")
        main_export(
            model_name_or_path=self.model_id,
            output=self.output_dir,
            task="text2text-generation-with-past",
            opset=14,
            device="cpu",
        )

        # Rename files to match expected naming convention for Hugot
        logger.info("Renaming ONNX files to match Hugot conventions...")
        rename_map = {
            "encoder_model.onnx": "encoder.onnx",
            "decoder_model.onnx": "decoder-init.onnx",
            "decoder_with_past_model.onnx": "decoder.onnx",
        }

        for old_name, new_name in rename_map.items():
            old_path = self.output_dir / old_name
            new_path = self.output_dir / new_name
            if old_path.exists():
                shutil.move(str(old_path), str(new_path))
                logger.info(f"  Renamed {old_name} -> {new_name}")

        # Ensure tokenizer files are present
        logger.info("Saving tokenizer files...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.save_pretrained(str(self.output_dir))

        # Create seq2seq config
        seq2seq_config = {
            "model_type": "seq2seq",
            "source_model": self.model_id,
            "encoder_file": "encoder.onnx",
            "decoder_init_file": "decoder-init.onnx",
            "decoder_file": "decoder.onnx",
        }

        if hasattr(config, "d_model"):
            seq2seq_config["hidden_size"] = config.d_model
        if hasattr(config, "vocab_size"):
            seq2seq_config["vocab_size"] = config.vocab_size

        config_path = self.output_dir / "seq2seq_config.json"
        with open(config_path, "w") as f:
            json.dump(seq2seq_config, f, indent=2)
        logger.info(f"  Saved: seq2seq_config.json")

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
        from transformers import AutoTokenizer

        try:
            encoder_path = self.output_dir / "encoder.onnx"
            decoder_init_path = self.output_dir / "decoder-init.onnx"

            if not encoder_path.exists():
                logger.error("encoder.onnx not found")
                return False

            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.output_dir)

            # Test encoder
            logger.info("Testing encoder...")
            encoder_session = ort.InferenceSession(
                str(encoder_path), providers=["CPUExecutionProvider"]
            )

            test_input = "Translate to French: Hello, how are you?"
            inputs = tokenizer(test_input, return_tensors="np", padding=True)

            encoder_outputs = encoder_session.run(
                None,
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
            )
            logger.info(f"  Encoder output shape: {encoder_outputs[0].shape}")

            # Test decoder-init
            if decoder_init_path.exists():
                logger.info("Testing decoder-init...")
                decoder_session = ort.InferenceSession(
                    str(decoder_init_path), providers=["CPUExecutionProvider"]
                )

                decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
                decoder_inputs = {
                    "input_ids": decoder_input_ids,
                    "encoder_hidden_states": encoder_outputs[0],
                }

                # Add encoder attention mask if needed
                input_names = {inp.name for inp in decoder_session.get_inputs()}
                if "encoder_attention_mask" in input_names:
                    decoder_inputs["encoder_attention_mask"] = inputs["attention_mask"]

                decoder_outputs = decoder_session.run(None, decoder_inputs)
                logger.info(f"  Decoder output shape: {decoder_outputs[0].shape}")

            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
