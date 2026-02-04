"""REBEL relation extraction model exporter."""

import json
import logging
import shutil
from pathlib import Path

from . import register_exporter
from .base import BaseExporter

logger = logging.getLogger(__name__)


@register_exporter("recognizer", capability="relations")
class REBELExporter(BaseExporter):
    """Exporter for REBEL relation extraction models.

    REBEL (Relation Extraction By End-to-end Language generation) is a seq2seq model
    based on BART that extracts relation triplets from text.
    """

    capabilities = ["relations"]

    def export(self) -> Path:
        from optimum.exporters.onnx import main_export
        from transformers import AutoTokenizer, AutoConfig

        logger.info(f"Exporting REBEL model: {self.model_id}")
        logger.info(f"Output: {self.output_dir}")

        # Load model config
        logger.info("Loading model configuration...")
        config = AutoConfig.from_pretrained(self.model_id)

        # Export to ONNX using Optimum (REBEL is based on BART, so use seq2seq task)
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

        # Save tokenizer
        logger.info("Saving tokenizer files...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.save_pretrained(str(self.output_dir))

        # Create REBEL config file for Termite
        is_multilingual = "mrebel" in self.model_id.lower()
        rebel_config = {
            "model_type": "rebel",
            "model_id": self.model_id,
            "max_length": 256,
            "num_beams": 3,
            "task": "relation_extraction",
            "multilingual": is_multilingual,
            # Special tokens used by REBEL for parsing output
            "triplet_token": "<triplet>",
            "subject_token": "<subj>",
            "object_token": "<obj>",
        }

        config_path = self.output_dir / "rebel_config.json"
        with open(config_path, "w") as f:
            json.dump(rebel_config, f, indent=2)
        logger.info("  Saved: rebel_config.json")

        return self.output_dir

    def test(self) -> bool:
        import onnxruntime as ort
        import numpy as np
        from transformers import AutoTokenizer

        try:
            encoder_path = self.output_dir / "encoder.onnx"
            if not encoder_path.exists():
                logger.error("encoder.onnx not found")
                return False

            logger.info("Testing REBEL model...")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.output_dir)

            # Load encoder
            logger.info("Loading encoder...")
            encoder_session = ort.InferenceSession(
                str(encoder_path), providers=["CPUExecutionProvider"]
            )

            # Test with sample text
            test_text = (
                "Punta Cana is a resort town in the municipality of Higuey, "
                "in La Altagracia Province, the easternmost province of the Dominican Republic."
            )
            inputs = tokenizer(test_text, return_tensors="np", padding=True)

            encoder_outputs = encoder_session.run(
                None,
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
            )

            logger.info(f"  Input: {test_text[:50]}...")
            logger.info(f"  Encoder output shape: {encoder_outputs[0].shape}")

            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
