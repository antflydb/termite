"""PaddleOCR PP-OCRv4 exporter (detection + CTC recognition)."""

import json
import logging
import shutil
from pathlib import Path

from . import register_exporter
from .base import BaseExporter

logger = logging.getLogger(__name__)

# Known PaddleOCR ONNX model repos on HuggingFace
PADDLEOCR_REPOS = {
    # Community-maintained ONNX exports
    "monkt/paddleocr-onnx": {
        "detection": "det.onnx",
        "recognition": "rec.onnx",
        "char_dict": "ppocr_keys_v1.txt",
    },
}

# Default PaddleOCR detection/recognition input sizes
DET_INPUT_SIZE = (960, 960)
REC_INPUT_SIZE = (48, 320)


@register_exporter("reader", "paddleocr")
class PaddleOCRExporter(BaseExporter):
    """Exporter for PaddleOCR PP-OCRv4 models (DBNet detection + SVTR/CTC recognition).

    PaddleOCR uses a different pipeline from Vision2Seq models:
    - Detection: DBNet outputs a probability map, post-processed with DB algorithm
    - Recognition: SVTR with CTC decoding (argmax → collapse → remove blanks)

    Two export strategies:
    1. Download pre-exported ONNX from HuggingFace community repos
    2. Download Paddle format models + convert via paddle2onnx
    """

    def export(self) -> Path:
        """Export PaddleOCR models to ONNX format."""
        logger.info(f"Exporting PaddleOCR model: {self.model_id}")
        logger.info(f"Output: {self.output_dir}")

        # Try to download pre-exported ONNX first
        if self._try_download_preexported():
            return self.output_dir

        # Fall back to paddle2onnx conversion
        return self._export_via_paddle2onnx()

    def _try_download_preexported(self) -> bool:
        """Try to download pre-exported ONNX models from HuggingFace."""
        from huggingface_hub import hf_hub_download, list_repo_files

        try:
            repo_files = list_repo_files(self.model_id)
        except Exception:
            return False

        # Check if repo has ONNX files
        has_det = any("det" in f and f.endswith(".onnx") for f in repo_files)
        has_rec = any("rec" in f and f.endswith(".onnx") for f in repo_files)

        if not (has_det and has_rec):
            logger.info("No pre-exported ONNX found, will convert from Paddle format")
            return False

        logger.info("Found pre-exported ONNX models, downloading...")

        for filename in repo_files:
            # Skip large non-ONNX files
            if filename.endswith((".safetensors", ".bin", ".h5", ".pdparams")):
                continue
            if filename.startswith("."):
                continue

            logger.info(f"  Downloading: {filename}")
            hf_hub_download(self.model_id, filename, local_dir=self.output_dir)

        self._create_metadata("preexported")
        return True

    def _export_via_paddle2onnx(self) -> Path:
        """Convert PaddleOCR models from Paddle format to ONNX."""
        try:
            import paddle2onnx  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "paddle2onnx is required for PaddleOCR export. "
                "Install with: pip install paddle2onnx paddlepaddle"
            )

        from huggingface_hub import hf_hub_download

        logger.info("Converting PaddleOCR models from Paddle format to ONNX...")

        # Download Paddle inference models
        for component in ["det", "rec"]:
            model_file = f"{component}_model.pdmodel"
            params_file = f"{component}_model.pdiparams"

            for filename in [model_file, params_file]:
                logger.info(f"  Downloading: {filename}")
                hf_hub_download(self.model_id, filename, local_dir=self.output_dir)

            # Convert to ONNX
            input_model = str(self.output_dir / model_file)
            input_params = str(self.output_dir / params_file)
            output_onnx = str(self.output_dir / f"{component}.onnx")

            logger.info(f"  Converting {component} to ONNX...")
            import paddle2onnx

            paddle2onnx.export(
                input_model,
                input_params,
                output_onnx,
                opset_version=11,
                enable_onnx_checker=True,
            )

        # Download character dictionary
        try:
            hf_hub_download(
                self.model_id, "ppocr_keys_v1.txt", local_dir=self.output_dir
            )
        except Exception:
            logger.warning("Could not download character dictionary, using default")
            self._create_default_char_dict()

        self._create_metadata("paddle2onnx")
        return self.output_dir

    def _create_default_char_dict(self):
        """Create a basic character dictionary if none is available."""
        # Minimal ASCII character dictionary as fallback
        dict_path = self.output_dir / "ppocr_keys_v1.txt"
        if not dict_path.exists():
            chars = list(
                " !\"#$%&'()*+,-./0123456789:;<=>?@"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
                "abcdefghijklmnopqrstuvwxyz{|}~"
            )
            with open(dict_path, "w") as f:
                for c in chars:
                    f.write(c + "\n")

    def _create_metadata(self, framework: str):
        """Create termite_metadata.json for multi-stage OCR pipeline."""
        # Find the actual ONNX file names
        det_file = self._find_onnx_file("det")
        rec_file = self._find_onnx_file("rec")
        dict_file = "ppocr_keys_v1.txt"

        metadata = {
            "model_type": "paddleocr",
            "source_model": self.model_id,
            "export_format": "onnx",
            "framework": framework,
            "pipeline_type": "multistage_ocr",
            "stages": {
                "detection": {
                    "model_file": det_file,
                    "post_processor": "db",
                },
                "recognition": {
                    "type": "ctc",
                    "model_file": rec_file,
                    "char_dict_file": dict_file,
                },
            },
        }

        metadata_path = self.output_dir / "termite_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("  Saved: termite_metadata.json")

    def _find_onnx_file(self, prefix: str) -> str:
        """Find an ONNX file matching a prefix in the output directory."""
        for f in self.output_dir.iterdir():
            if f.name.startswith(prefix) and f.name.endswith(".onnx"):
                return f.name
        return f"{prefix}.onnx"

    def test(self) -> bool:
        import onnxruntime as ort
        import numpy as np

        try:
            # Test detection model
            det_path = self.output_dir / self._find_onnx_file("det")
            if det_path.exists():
                logger.info("Testing detection model...")
                session = ort.InferenceSession(
                    str(det_path), providers=["CPUExecutionProvider"]
                )
                inputs = session.get_inputs()
                logger.info(f"  Detection inputs: {[i.name for i in inputs]}")

                # Create dummy input
                shape = [1, 3, DET_INPUT_SIZE[0], DET_INPUT_SIZE[1]]
                dummy = np.random.randn(*shape).astype(np.float32)
                outputs = session.run(None, {inputs[0].name: dummy})
                logger.info(
                    f"  Detection output shapes: {[o.shape for o in outputs]}"
                )

            # Test recognition model
            rec_path = self.output_dir / self._find_onnx_file("rec")
            if rec_path.exists():
                logger.info("Testing recognition model...")
                session = ort.InferenceSession(
                    str(rec_path), providers=["CPUExecutionProvider"]
                )
                inputs = session.get_inputs()
                logger.info(f"  Recognition inputs: {[i.name for i in inputs]}")

                shape = [1, 3, REC_INPUT_SIZE[0], REC_INPUT_SIZE[1]]
                dummy = np.random.randn(*shape).astype(np.float32)
                outputs = session.run(None, {inputs[0].name: dummy})
                logger.info(
                    f"  Recognition output shapes: {[o.shape for o in outputs]}"
                )

            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
