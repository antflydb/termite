"""Generative LLM model exporter using ONNX Runtime GenAI."""

import json
import logging
import os
import subprocess
from pathlib import Path

from . import register_exporter
from .base import BaseExporter

logger = logging.getLogger(__name__)

# Map variant names to builder arguments
VARIANT_CONFIG = {
    "f32": {"precision": "fp32", "execution_provider": "cpu"},
    "f16": {"precision": "fp16", "execution_provider": "cpu"},
    "i4": {"precision": "int4", "execution_provider": "cpu"},
    "i4-cuda": {"precision": "int4", "execution_provider": "cuda"},
    "i4-dml": {"precision": "int4", "execution_provider": "dml"},
}


def detect_tool_call_format(model_dir: Path) -> str | None:
    """Detect the tool calling format based on tokenizer config files.

    Checks special_tokens_map.json and tokenizer_config.json for known
    tool calling token patterns.

    Returns:
        Tool call format name (e.g., "functiongemma") or None if not detected.
    """
    functiongemma_tokens = [
        "start_function_declaration",
        "end_function_declaration",
        "start_function_call",
        "end_function_call",
    ]

    # Check special_tokens_map.json
    special_tokens_path = model_dir / "special_tokens_map.json"
    if special_tokens_path.exists():
        try:
            with open(special_tokens_path) as f:
                special_tokens = json.load(f)
                content = json.dumps(special_tokens).lower()
                if all(token in content for token in functiongemma_tokens):
                    return "functiongemma"
        except (json.JSONDecodeError, IOError):
            pass

    # Check tokenizer_config.json
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        try:
            with open(tokenizer_config_path) as f:
                tokenizer_config = json.load(f)
                content = json.dumps(tokenizer_config).lower()
                if all(token in content for token in functiongemma_tokens):
                    return "functiongemma"
        except (json.JSONDecodeError, IOError):
            pass

    return None


def detect_and_add_tool_call_format(model_dir: Path) -> None:
    """Detect tool calling format and add it to genai_config.json.

    This modifies the genai_config.json in place to add the tool_call_format
    field if a known tool calling format is detected.
    """
    tool_format = detect_tool_call_format(model_dir)
    if not tool_format:
        return

    genai_config_path = model_dir / "genai_config.json"
    if not genai_config_path.exists():
        logger.warning(
            f"genai_config.json not found in {model_dir}, skipping tool format detection"
        )
        return

    try:
        with open(genai_config_path) as f:
            config = json.load(f)

        config["tool_call_format"] = tool_format

        with open(genai_config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Added tool_call_format: {tool_format} to genai_config.json")
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to update genai_config.json: {e}")


@register_exporter("generator")
class GeneratorExporter(BaseExporter):
    """Exporter for generative LLMs using ONNX Runtime GenAI model builder.

    Creates optimized ONNX models for text generation with support for
    different precisions and hardware targets.

    Variants:
        - "f32": FP32 precision (default)
        - "f16": FP16 precision (works on CPU and CUDA)
        - "i4": INT4 quantized for CPU
        - "i4-cuda": INT4 quantized for CUDA
        - "i4-dml": INT4 quantized for DirectML (Windows)
    """

    def __init__(
        self,
        model_id: str,
        output_dir: Path,
        variants: list[str] | None = None,
        hf_token: str | None = None,
    ):
        super().__init__(model_id, output_dir, variants or ["f32"])
        self.hf_token = hf_token

    def export(self) -> Path:
        logger.info(f"Exporting generator model: {self.model_id}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Variants: {', '.join(self.variants)}")

        # Export each variant
        for variant in self.variants:
            if variant not in VARIANT_CONFIG:
                logger.warning(f"Unknown variant: {variant}, skipping")
                continue

            config = VARIANT_CONFIG[variant]
            precision = config["precision"]
            exec_provider = config["execution_provider"]

            # Determine output path for this variant
            if variant in ("f32", "f16"):
                variant_dir = self.output_dir
            else:
                variant_dir = self.output_dir / variant
                variant_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"\nExporting variant: {variant}")
            logger.info(f"  Precision: {precision}")
            logger.info(f"  Execution provider: {exec_provider}")
            logger.info(f"  Output: {variant_dir}")

            # Build the model builder command
            cmd = [
                "python",
                "-m",
                "onnxruntime_genai.models.builder",
                "-m",
                self.model_id,
                "-o",
                str(variant_dir),
                "-p",
                precision,
                "-e",
                exec_provider,
            ]

            logger.info(f"  Running: {' '.join(cmd)}")

            try:
                # Prepare environment with HF_TOKEN if provided
                env = os.environ.copy()
                if self.hf_token:
                    env["HF_TOKEN"] = self.hf_token

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    env=env,
                )
                if result.stdout:
                    for line in result.stdout.strip().split("\n"):
                        logger.info(f"  {line}")
                logger.info(f"  Variant {variant} exported successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"  Failed to export variant {variant}")
                if e.stdout:
                    logger.error(f"  stdout: {e.stdout}")
                if e.stderr:
                    logger.error(f"  stderr: {e.stderr}")
                raise RuntimeError(
                    f"Model builder failed for variant {variant}: {e.stderr}"
                )
            except FileNotFoundError:
                logger.error("  onnxruntime-genai not installed")
                logger.error("  Install with: pip install onnxruntime-genai")
                raise RuntimeError("onnxruntime-genai package not found")

        # Log exported files
        logger.info("\nExported files:")
        for f in sorted(self.output_dir.rglob("*")):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                rel_path = f.relative_to(self.output_dir)
                logger.info(f"  {rel_path}: {size_mb:.2f} MB")

        # Detect and add tool calling format to genai_config.json
        detect_and_add_tool_call_format(self.output_dir)

        return self.output_dir

    def test(self) -> bool:
        try:
            genai_config = self.output_dir / "genai_config.json"
            model_onnx = self.output_dir / "model.onnx"

            if not genai_config.exists():
                logger.error("genai_config.json not found")
                return False

            if not model_onnx.exists():
                logger.error("model.onnx not found")
                return False

            logger.info("Testing generator model...")
            with open(genai_config) as f:
                config = json.load(f)

            logger.info(f"  Model config loaded")
            if "model" in config:
                logger.info(f"  Model type: {config['model'].get('type', 'unknown')}")

            logger.info("Test passed!")
            return True

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
