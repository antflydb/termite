#!/usr/bin/env python3
"""
Export GLiNER models to ONNX format for zero-shot Named Entity Recognition.

GLiNER models can extract any entity type without retraining - just specify
the entity labels at inference time.

Usage:
    python scripts/export_gliner.py --model urchade/gliner_small-v2.1 --output ./models/ner/gliner_small
    python scripts/export_gliner.py --model urchade/gliner_medium-v2.1 --output ./models/ner/gliner_medium --quantize int8
    python scripts/export_gliner.py --model urchade/gliner_medium-v2.1 --output ./models/ner/gliner_medium --quantize fp16

Available Models:
    - urchade/gliner_small-v2.1  (~166M params, fast inference)
    - urchade/gliner_medium-v2.1 (~209M params, balanced)
    - urchade/gliner_large-v2.1  (~304M params, most accurate)

After export, use with termite:
    termite run --models-dir ./models
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    try:
        import gliner
    except ImportError:
        missing.append("gliner")
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")
    try:
        import onnxconverter_common
    except ImportError:
        missing.append("onnxconverter-common")

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def convert_to_fp16(input_path: Path, output_path: Path):
    """Convert an ONNX model to FP16 precision."""
    import onnx
    from onnxconverter_common import float16

    logger.info("  Converting to FP16...")
    model = onnx.load(str(input_path))
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,  # Keep inputs/outputs as float32 for compatibility
    )
    onnx.save(model_fp16, str(output_path))
    logger.info(f"  Saved FP16 model: {output_path.name}")


def export_gliner_model(model_id: str, output_dir: str, quantize: str | None = None):
    """
    Export a GLiNER model to ONNX format.

    Args:
        model_id: HuggingFace model ID (e.g., urchade/gliner_small-v2.1)
        output_dir: Directory to save the exported model
        quantize: Quantization type: 'int8', 'fp16', or None for no quantization
    """
    from gliner import GLiNER

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading GLiNER model: {model_id}")
    model = GLiNER.from_pretrained(model_id)

    logger.info(f"Exporting to ONNX format...")

    # GLiNER has built-in ONNX export
    temp_onnx_dir = output_path / "temp_onnx"
    temp_onnx_dir.mkdir(exist_ok=True)

    try:
        # Export using GLiNER's built-in method
        use_int8 = quantize == "int8"
        model.export_to_onnx(
            str(temp_onnx_dir),
            quantize=use_int8
        )

        # Move the exported files to the final location
        for file in temp_onnx_dir.iterdir():
            if file.is_file():
                dest = output_path / file.name
                shutil.move(str(file), str(dest))
                logger.info(f"  Saved: {file.name}")

        # Apply FP16 conversion if requested
        if quantize == "fp16":
            onnx_file = output_path / "model.onnx"
            if not onnx_file.exists():
                onnx_files = list(output_path.glob("*.onnx"))
                if onnx_files:
                    onnx_file = onnx_files[0]
            if onnx_file.exists():
                fp16_file = output_path / "model_fp16.onnx"
                convert_to_fp16(onnx_file, fp16_file)
                onnx_file.unlink()
                fp16_file.rename(output_path / "model.onnx")
    finally:
        if temp_onnx_dir.exists():
            shutil.rmtree(temp_onnx_dir)

    # Create GLiNER config file for Termite
    gliner_config = {
        "max_width": 12,
        "default_labels": ["person", "organization", "location", "date", "product"],
        "threshold": 0.5,
        "flat_ner": True,
        "multi_label": False,
        "model_id": model_id,
    }

    config_path = output_path / "gliner_config.json"
    with open(config_path, "w") as f:
        json.dump(gliner_config, f, indent=2)
    logger.info(f"  Saved: gliner_config.json")

    # Save tokenizer
    try:
        if hasattr(model, 'data_processor') and hasattr(model.data_processor, 'transformer_tokenizer'):
            tokenizer = model.data_processor.transformer_tokenizer
            tokenizer.save_pretrained(str(output_path))
            logger.info("  Saved: tokenizer files")
        elif hasattr(model, 'tokenizer'):
            model.tokenizer.save_pretrained(str(output_path))
            logger.info("  Saved: tokenizer files")
        else:
            from transformers import AutoTokenizer
            base_model = "microsoft/deberta-v3-small"
            logger.info(f"  Loading tokenizer from base model: {base_model}")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            tokenizer.save_pretrained(str(output_path))
            logger.info("  Saved: tokenizer files")
    except Exception as e:
        logger.warning(f"  Could not save tokenizer: {e}")

    # Rename model file to match Termite conventions
    onnx_files = list(output_path.glob("*.onnx"))
    for onnx_file in onnx_files:
        if onnx_file.name == "model.onnx":
            continue
        if quantize == "int8" and "quantized" not in onnx_file.name:
            new_name = output_path / "model_quantized.onnx"
        else:
            new_name = output_path / "model.onnx"
        if onnx_file != new_name:
            onnx_file.rename(new_name)
            logger.info(f"  Renamed: {onnx_file.name} -> {new_name.name}")

    logger.info(f"\nExport complete!")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"\nUsage:")
    logger.info(f"  termite run --models-dir {output_path.parent.parent}")


def main():
    parser = argparse.ArgumentParser(
        description="Export GLiNER models to ONNX for zero-shot NER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export small GLiNER model (no quantization)
  python scripts/export_gliner.py --model urchade/gliner_small-v2.1 --output ./models/ner/gliner_small

  # Export with INT8 quantization
  python scripts/export_gliner.py --model urchade/gliner_medium-v2.1 --output ./models/ner/gliner_medium --quantize int8

  # Export with FP16 quantization
  python scripts/export_gliner.py --model urchade/gliner_medium-v2.1 --output ./models/ner/gliner_medium_fp16 --quantize fp16

Available Models:
  - urchade/gliner_small-v2.1  (~166M params, ~330MB)
  - urchade/gliner_medium-v2.1 (~209M params, ~420MB)
  - urchade/gliner_large-v2.1  (~304M params, ~610MB)
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="urchade/gliner_small-v2.1",
        help="HuggingFace model ID (default: urchade/gliner_small-v2.1)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./models/ner/gliner_small",
        help="Output directory for the exported model",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["int8", "fp16"],
        default=None,
        help="Quantization type: 'int8' (smallest) or 'fp16' (balanced)",
    )

    args = parser.parse_args()

    check_dependencies()

    try:
        export_gliner_model(args.model, args.output, args.quantize)
        logger.info("\nAll done!")
    except Exception as e:
        logger.error(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
