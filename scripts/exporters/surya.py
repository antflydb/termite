"""Surya OCR exporter (detection, recognition, layout, reading order)."""

import json
import logging
import shutil
from pathlib import Path

from . import register_exporter
from .base import BaseExporter

logger = logging.getLogger(__name__)

# Surya model repos on HuggingFace
SURYA_REPOS = {
    "detection": "vikp/surya_det3",
    "recognition": "vikp/surya_rec2",
    "layout": "vikp/surya_layout3",
    "order": "vikp/surya_order",
}

# Default Surya input sizes
DET_INPUT_SIZE = (1024, 1024)
REC_INPUT_SIZE = (256, 896)
LAYOUT_INPUT_SIZE = (1024, 1024)


@register_exporter("reader", "surya")
class SuryaExporter(BaseExporter):
    """Exporter for Surya OCR models (detection, recognition, layout, order).

    Surya uses custom architectures that require torch.onnx.export:
    - Detection: Modified EfficientViT → heatmap (segmentation)
    - Recognition: Modified Donut (GQA/MoE) → encoder-decoder with Vision2Seq pipeline
    - Layout: Segmentation → class heatmaps
    - Order: Bbox positions → reading order indices

    License note: Surya model weights use a restrictive license
    (free for research/personal/startups <$2M revenue, code is GPL-3.0).
    """

    def export(self) -> Path:
        """Export Surya OCR models to ONNX format."""
        logger.info(f"Exporting Surya OCR models: {self.model_id}")
        logger.info(f"Output: {self.output_dir}")

        # Determine which stages to export based on model_id
        stages = self._determine_stages()
        logger.info(f"Stages to export: {list(stages.keys())}")

        exported_stages = {}

        # Export detection
        if "detection" in stages:
            det_meta = self._export_detection(stages["detection"])
            if det_meta:
                exported_stages["detection"] = det_meta

        # Export recognition (encoder-decoder, uses Vision2Seq pipeline)
        if "recognition" in stages:
            rec_meta = self._export_recognition(stages["recognition"])
            if rec_meta:
                exported_stages["recognition"] = rec_meta

        # Export layout
        if "layout" in stages:
            layout_meta = self._export_layout(stages["layout"])
            if layout_meta:
                exported_stages["layout"] = layout_meta

        # Export order
        if "order" in stages:
            order_meta = self._export_order(stages["order"])
            if order_meta:
                exported_stages["order"] = order_meta

        if not exported_stages:
            raise RuntimeError("No stages were successfully exported")

        self._create_metadata(exported_stages)
        return self.output_dir

    def _determine_stages(self) -> dict[str, str]:
        """Determine which stages to export based on model_id.

        If model_id is a specific Surya model (e.g., vikp/surya_det3),
        export only that stage. If model_id is 'surya' or 'vikp/surya',
        export all stages using default repos.
        """
        model_lower = self.model_id.lower()

        # Check for specific stage model
        for stage, repo in SURYA_REPOS.items():
            if self.model_id == repo:
                return {stage: repo}

        # Check for stage keywords in model_id
        if "det" in model_lower and "layout" not in model_lower:
            return {"detection": self.model_id}
        if "rec" in model_lower:
            return {"recognition": self.model_id}
        if "layout" in model_lower:
            return {"layout": self.model_id}
        if "order" in model_lower:
            return {"order": self.model_id}

        # Default: export all stages
        return dict(SURYA_REPOS)

    def _export_detection(self, repo_id: str) -> dict | None:
        """Export Surya detection model to ONNX."""
        import torch

        logger.info(f"Exporting detection model from {repo_id}...")

        try:
            from surya.model.detection.model import load_model as load_det_model
            from surya.model.detection.processor import load_processor as load_det_processor

            model = load_det_model()
            processor = load_det_processor()
            model.eval()

            # Create dummy input
            dummy_input = torch.randn(1, 3, DET_INPUT_SIZE[0], DET_INPUT_SIZE[1])

            # Export to ONNX
            onnx_path = str(self.output_dir / "detection.onnx")
            logger.info("  Running torch.onnx.export for detection...")
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                opset_version=14,
                input_names=["pixel_values"],
                output_names=["heatmap"],
                dynamic_axes={
                    "pixel_values": {0: "batch", 2: "height", 3: "width"},
                    "heatmap": {0: "batch", 2: "height", 3: "width"},
                },
            )
            logger.info(f"  Saved: detection.onnx")

            # Save processor config
            if hasattr(processor, "save_pretrained"):
                processor.save_pretrained(str(self.output_dir / "detection_processor"))

            return {
                "model_file": "detection.onnx",
                "post_processor": "heatmap",
            }

        except ImportError:
            logger.warning(
                "surya-ocr package not installed. "
                "Trying to download pre-exported ONNX..."
            )
            return self._try_download_stage(repo_id, "detection")

        except Exception as e:
            logger.error(f"Failed to export detection model: {e}")
            return self._try_download_stage(repo_id, "detection")

    def _export_recognition(self, repo_id: str) -> dict | None:
        """Export Surya recognition model to ONNX.

        Surya rec is a modified Donut (encoder-decoder), so we export
        encoder and decoder separately to match the Vision2Seq pipeline format.
        """
        import torch

        logger.info(f"Exporting recognition model from {repo_id}...")

        try:
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            from surya.model.recognition.tokenizer import load_tokenizer

            model = load_rec_model()
            processor = load_rec_processor()
            model.eval()

            # Export encoder
            logger.info("  Exporting recognition encoder...")
            encoder = model.encoder if hasattr(model, "encoder") else model.get_encoder()
            dummy_pixel_values = torch.randn(1, 3, REC_INPUT_SIZE[0], REC_INPUT_SIZE[1])

            encoder_path = str(self.output_dir / "encoder_model.onnx")
            torch.onnx.export(
                encoder,
                dummy_pixel_values,
                encoder_path,
                opset_version=14,
                input_names=["pixel_values"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "pixel_values": {0: "batch", 2: "height", 3: "width"},
                    "last_hidden_state": {0: "batch", 1: "sequence"},
                },
            )
            logger.info("  Saved: encoder_model.onnx")

            # Export decoder
            logger.info("  Exporting recognition decoder...")
            decoder = model.decoder if hasattr(model, "decoder") else model.get_decoder()

            # Decoder inputs: input_ids, encoder_hidden_states
            encoder_output = encoder(dummy_pixel_values)
            if isinstance(encoder_output, tuple):
                encoder_hidden = encoder_output[0]
            else:
                encoder_hidden = encoder_output.last_hidden_state if hasattr(encoder_output, "last_hidden_state") else encoder_output

            dummy_input_ids = torch.zeros(1, 1, dtype=torch.long)

            decoder_path = str(self.output_dir / "decoder_model.onnx")
            try:
                torch.onnx.export(
                    decoder,
                    (dummy_input_ids, encoder_hidden),
                    decoder_path,
                    opset_version=14,
                    input_names=["input_ids", "encoder_hidden_states"],
                    output_names=["logits"],
                    dynamic_axes={
                        "input_ids": {0: "batch", 1: "sequence"},
                        "encoder_hidden_states": {0: "batch", 1: "encoder_sequence"},
                        "logits": {0: "batch", 1: "sequence"},
                    },
                )
                logger.info("  Saved: decoder_model.onnx")
            except Exception as e:
                logger.warning(f"  Decoder export failed (complex architecture): {e}")
                logger.info("  Attempting full model export as fallback...")
                return self._export_recognition_full(model, processor, dummy_pixel_values)

            # Save tokenizer and processor
            try:
                tokenizer = load_tokenizer()
                tokenizer.save_pretrained(str(self.output_dir))
            except Exception as e:
                logger.warning(f"  Could not save tokenizer: {e}")

            if hasattr(processor, "save_pretrained"):
                processor.save_pretrained(str(self.output_dir))

            return {
                "type": "vision2seq",
                "encoder_file": "encoder_model.onnx",
                "decoder_file": "decoder_model.onnx",
            }

        except ImportError:
            logger.warning(
                "surya-ocr package not installed. "
                "Trying to download pre-exported ONNX..."
            )
            return self._try_download_stage(repo_id, "recognition")

        except Exception as e:
            logger.error(f"Failed to export recognition model: {e}")
            return self._try_download_stage(repo_id, "recognition")

    def _export_recognition_full(self, model, processor, dummy_pixel_values) -> dict | None:
        """Fallback: export the full recognition model if encoder/decoder split fails."""
        import torch

        try:
            dummy_input_ids = torch.zeros(1, 1, dtype=torch.long)

            full_path = str(self.output_dir / "rec_model.onnx")
            torch.onnx.export(
                model,
                (dummy_pixel_values, dummy_input_ids),
                full_path,
                opset_version=14,
                input_names=["pixel_values", "input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "pixel_values": {0: "batch", 2: "height", 3: "width"},
                    "input_ids": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"},
                },
            )
            logger.info("  Saved: rec_model.onnx (full model)")

            return {
                "type": "vision2seq",
                "model_file": "rec_model.onnx",
            }
        except Exception as e:
            logger.error(f"  Full model export also failed: {e}")
            return None

    def _export_layout(self, repo_id: str) -> dict | None:
        """Export Surya layout model to ONNX."""
        import torch

        logger.info(f"Exporting layout model from {repo_id}...")

        try:
            from surya.model.layout.model import load_model as load_layout_model
            from surya.model.layout.processor import load_processor as load_layout_processor

            model = load_layout_model()
            processor = load_layout_processor()
            model.eval()

            # Create dummy input
            dummy_input = torch.randn(1, 3, LAYOUT_INPUT_SIZE[0], LAYOUT_INPUT_SIZE[1])

            # Export to ONNX
            onnx_path = str(self.output_dir / "layout.onnx")
            logger.info("  Running torch.onnx.export for layout...")
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                opset_version=14,
                input_names=["pixel_values"],
                output_names=["class_heatmaps"],
                dynamic_axes={
                    "pixel_values": {0: "batch", 2: "height", 3: "width"},
                    "class_heatmaps": {0: "batch"},
                },
            )
            logger.info("  Saved: layout.onnx")

            # Save processor config
            if hasattr(processor, "save_pretrained"):
                processor.save_pretrained(str(self.output_dir / "layout_processor"))

            return {
                "model_file": "layout.onnx",
            }

        except ImportError:
            logger.warning("surya-ocr package not installed for layout export")
            return self._try_download_stage(repo_id, "layout")

        except Exception as e:
            logger.error(f"Failed to export layout model: {e}")
            return self._try_download_stage(repo_id, "layout")

    def _export_order(self, repo_id: str) -> dict | None:
        """Export Surya reading order model to ONNX."""
        import torch

        logger.info(f"Exporting order model from {repo_id}...")

        try:
            from surya.model.ordering.model import load_model as load_order_model
            from surya.model.ordering.processor import load_processor as load_order_processor

            model = load_order_model()
            processor = load_order_processor()
            model.eval()

            # Order model takes bounding box coordinates
            # Typical input: [batch, num_boxes, 4] (x1,y1,x2,y2 normalized)
            dummy_bboxes = torch.randn(1, 20, 4)

            # Export to ONNX
            onnx_path = str(self.output_dir / "order.onnx")
            logger.info("  Running torch.onnx.export for order...")
            torch.onnx.export(
                model,
                dummy_bboxes,
                onnx_path,
                opset_version=14,
                input_names=["bboxes"],
                output_names=["order_indices"],
                dynamic_axes={
                    "bboxes": {0: "batch", 1: "num_boxes"},
                    "order_indices": {0: "batch", 1: "num_boxes"},
                },
            )
            logger.info("  Saved: order.onnx")

            return {
                "model_file": "order.onnx",
            }

        except ImportError:
            logger.warning("surya-ocr package not installed for order export")
            return self._try_download_stage(repo_id, "order")

        except Exception as e:
            logger.error(f"Failed to export order model: {e}")
            return self._try_download_stage(repo_id, "order")

    def _try_download_stage(self, repo_id: str, stage_name: str) -> dict | None:
        """Try to download pre-exported ONNX files for a stage from HuggingFace."""
        from huggingface_hub import hf_hub_download, list_repo_files

        try:
            repo_files = list_repo_files(repo_id)
        except Exception:
            logger.warning(f"Could not list files for {repo_id}")
            return None

        onnx_files = [f for f in repo_files if f.endswith(".onnx")]
        if not onnx_files:
            logger.warning(f"No ONNX files found in {repo_id}")
            return None

        logger.info(f"  Downloading pre-exported files from {repo_id}...")
        for filename in repo_files:
            if filename.endswith((".safetensors", ".bin", ".h5", ".pdparams")):
                continue
            if filename.startswith("."):
                continue

            logger.info(f"    Downloading: {filename}")
            hf_hub_download(repo_id, filename, local_dir=self.output_dir)

        # Return metadata based on stage type
        if stage_name == "detection":
            det_file = next((f for f in onnx_files if "det" in f), onnx_files[0])
            return {"model_file": det_file, "post_processor": "heatmap"}
        elif stage_name == "recognition":
            encoder_file = next((f for f in onnx_files if "encoder" in f), None)
            decoder_file = next((f for f in onnx_files if "decoder" in f), None)
            if encoder_file and decoder_file:
                return {
                    "type": "vision2seq",
                    "encoder_file": encoder_file,
                    "decoder_file": decoder_file,
                }
            rec_file = next((f for f in onnx_files if "rec" in f), onnx_files[0])
            return {"type": "vision2seq", "model_file": rec_file}
        elif stage_name == "layout":
            layout_file = next((f for f in onnx_files if "layout" in f), onnx_files[0])
            return {"model_file": layout_file}
        elif stage_name == "order":
            order_file = next((f for f in onnx_files if "order" in f), onnx_files[0])
            return {"model_file": order_file}

        return None

    def _create_metadata(self, stages: dict):
        """Create termite_metadata.json for multi-stage OCR pipeline."""
        metadata = {
            "model_type": "surya",
            "source_model": self.model_id,
            "export_format": "onnx",
            "framework": "torch",
            "pipeline_type": "multistage_ocr",
            "stages": stages,
        }

        metadata_path = self.output_dir / "termite_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("  Saved: termite_metadata.json")

    def test(self) -> bool:
        import onnxruntime as ort
        import numpy as np

        try:
            # Test detection model
            det_path = self.output_dir / "detection.onnx"
            if det_path.exists():
                logger.info("Testing detection model...")
                session = ort.InferenceSession(
                    str(det_path), providers=["CPUExecutionProvider"]
                )
                inputs = session.get_inputs()
                logger.info(f"  Detection inputs: {[i.name for i in inputs]}")

                shape = [1, 3, DET_INPUT_SIZE[0], DET_INPUT_SIZE[1]]
                dummy = np.random.randn(*shape).astype(np.float32)
                outputs = session.run(None, {inputs[0].name: dummy})
                logger.info(
                    f"  Detection output shapes: {[o.shape for o in outputs]}"
                )

            # Test recognition encoder
            encoder_path = self.output_dir / "encoder_model.onnx"
            if encoder_path.exists():
                logger.info("Testing recognition encoder...")
                session = ort.InferenceSession(
                    str(encoder_path), providers=["CPUExecutionProvider"]
                )
                inputs = session.get_inputs()
                logger.info(f"  Encoder inputs: {[i.name for i in inputs]}")

                shape = [1, 3, REC_INPUT_SIZE[0], REC_INPUT_SIZE[1]]
                dummy = np.random.randn(*shape).astype(np.float32)
                outputs = session.run(None, {inputs[0].name: dummy})
                logger.info(
                    f"  Encoder output shapes: {[o.shape for o in outputs]}"
                )

            # Test layout model
            layout_path = self.output_dir / "layout.onnx"
            if layout_path.exists():
                logger.info("Testing layout model...")
                session = ort.InferenceSession(
                    str(layout_path), providers=["CPUExecutionProvider"]
                )
                inputs = session.get_inputs()
                logger.info(f"  Layout inputs: {[i.name for i in inputs]}")

                shape = [1, 3, LAYOUT_INPUT_SIZE[0], LAYOUT_INPUT_SIZE[1]]
                dummy = np.random.randn(*shape).astype(np.float32)
                outputs = session.run(None, {inputs[0].name: dummy})
                logger.info(
                    f"  Layout output shapes: {[o.shape for o in outputs]}"
                )

            # Test order model
            order_path = self.output_dir / "order.onnx"
            if order_path.exists():
                logger.info("Testing order model...")
                session = ort.InferenceSession(
                    str(order_path), providers=["CPUExecutionProvider"]
                )
                inputs = session.get_inputs()
                logger.info(f"  Order inputs: {[i.name for i in inputs]}")

                dummy = np.random.randn(1, 20, 4).astype(np.float32)
                outputs = session.run(None, {inputs[0].name: dummy})
                logger.info(
                    f"  Order output shapes: {[o.shape for o in outputs]}"
                )

            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
