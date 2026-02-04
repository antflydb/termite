"""Base exporter class and utilities for model export."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseExporter(ABC):
    """Base class for all model exporters.

    Subclasses should define:
    - capabilities: list of capabilities this exporter handles (e.g., ["audio"])
    - model_type: the model type this exporter handles (e.g., "embedder")
    """

    capabilities: list[str] = []
    model_type: str | None = None

    def __init__(self, model_id: str, output_dir: Path, variants: list[str] | None = None):
        self.model_id = model_id
        self.output_dir = output_dir
        self.variants = variants or []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, from_onnx: bool = False) -> Path:
        """Run the export process.

        Args:
            from_onnx: If True, download pre-exported ONNX files instead of converting

        Returns:
            Path to the exported model directory
        """
        if from_onnx:
            return self.download_onnx()
        return self.export()

    def download_onnx(self) -> Path:
        """Download a pre-exported ONNX model from HuggingFace.

        Downloads all relevant files (ONNX models, configs, tokenizers) while
        skipping large original model files (safetensors, bin, h5, msgpack).
        Files in onnx/ subdirectory are flattened to the root.
        """
        from huggingface_hub import hf_hub_download, list_repo_files
        import shutil

        logger.info(f"Downloading pre-exported ONNX model: {self.model_id}")
        logger.info(f"Output: {self.output_dir}")

        # Download all files from the HuggingFace repo
        repo_files = list_repo_files(self.model_id)
        logger.info(f"Found {len(repo_files)} files in repo")

        for filename in repo_files:
            # Skip large original model files
            if filename.endswith(('.safetensors', '.bin', '.h5', '.msgpack')):
                logger.info(f"  Skipping: {filename}")
                continue
            # Skip hidden files and directories
            if filename.startswith('.'):
                continue

            logger.info(f"  Downloading: {filename}")
            local_path = hf_hub_download(self.model_id, filename, local_dir=self.output_dir)

            # Flatten onnx/ subdirectory to root
            if filename.startswith("onnx/"):
                flat_name = filename.replace("onnx/", "")
                dest_path = self.output_dir / flat_name
                if not dest_path.exists():
                    shutil.move(local_path, dest_path)
                    logger.info(f"    -> Moved to: {flat_name}")

        # Clean up empty onnx directory if it exists
        onnx_dir = self.output_dir / "onnx"
        if onnx_dir.exists() and onnx_dir.is_dir():
            try:
                onnx_dir.rmdir()
            except OSError:
                pass  # Directory not empty, that's fine

        return self.output_dir

    @abstractmethod
    def export(self) -> Path:
        """Export the model to ONNX format.

        Returns:
            Path to the exported model directory
        """
        pass

    @abstractmethod
    def test(self) -> bool:
        """Test the exported model.

        Returns:
            True if test passed, False otherwise
        """
        pass

    def create_variants(self, model_path: Path, variant_types: list[str]) -> dict[str, Path]:
        """Create quantized/precision variants of a model.

        Args:
            model_path: Path to the base ONNX model
            variant_types: List of variant types to create (e.g., ["f16", "i8"])

        Returns:
            Dict mapping variant type to output path
        """
        from onnxruntime.quantization import QuantType, quantize_dynamic

        variants = {}

        for variant_type in variant_types:
            if variant_type == "i8":
                output_path = model_path.parent / model_path.name.replace(".onnx", "_i8.onnx")
                try:
                    quantize_dynamic(
                        model_input=str(model_path),
                        model_output=str(output_path),
                        weight_type=QuantType.QUInt8,
                    )
                    variants["i8"] = output_path
                    logger.info(f"  Created i8 variant: {output_path.name}")
                except Exception as e:
                    logger.warning(f"  Failed to create i8 variant: {e}")

            elif variant_type == "f16":
                output_path = model_path.parent / model_path.name.replace(".onnx", "_f16.onnx")
                try:
                    from onnxconverter_common import float16
                    import onnx

                    model = onnx.load(str(model_path))
                    model_fp16 = float16.convert_float_to_float16(model)
                    onnx.save(model_fp16, str(output_path))
                    variants["f16"] = output_path
                    logger.info(f"  Created f16 variant: {output_path.name}")
                except Exception as e:
                    logger.warning(f"  Failed to create f16 variant: {e}")

        return variants
