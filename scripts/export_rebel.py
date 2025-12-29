#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "optimum[onnxruntime]",
#     "transformers",
#     "onnx",
# ]
# ///
"""
Export REBEL models to ONNX format for relation extraction.

REBEL (Relation Extraction By End-to-end Language generation) is a seq2seq model
based on BART that extracts relation triplets from text. It supports 200+ relation
types out of the box.

Usage:
    # Export REBEL large model
    uv run scripts/export_rebel.py --output ~/.termite/models/rel/rebel-large

    # Export multilingual mREBEL
    uv run scripts/export_rebel.py --model Babelscape/mrebel-large --output ~/.termite/models/rel/mrebel-large

Available Models:
    - Babelscape/rebel-large      (English, 200+ relation types)
    - Babelscape/mrebel-large     (Multilingual, more relation types)
    - Babelscape/mrebel-large-32  (Multilingual, 32 languages)

After export, use with termite:
    termite run --models-dir ~/.termite/models
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def export_rebel_model(model_id: str, output_dir: str):
    """
    Export a REBEL model to ONNX format using Optimum.

    Args:
        model_id: HuggingFace model ID (e.g., Babelscape/rebel-large)
        output_dir: Directory to save the exported model
    """
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading REBEL model: {model_id}")
    logger.info("This may take a few minutes for the initial download...")

    # Export using Optimum
    logger.info("Exporting to ONNX format...")
    model = ORTModelForSeq2SeqLM.from_pretrained(
        model_id,
        export=True,
        provider="CPUExecutionProvider"
    )

    # Save the exported model
    logger.info(f"Saving to {output_path}...")
    model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_path)
    logger.info("  Saved: tokenizer files")

    # Rename ONNX files to match Hugot's expected naming convention
    # Optimum exports: encoder_model.onnx, decoder_model.onnx, decoder_with_past_model.onnx
    # Hugot expects: encoder.onnx, decoder-init.onnx, decoder.onnx
    rename_map = {
        "encoder_model.onnx": "encoder.onnx",
        "decoder_model.onnx": "decoder-init.onnx",
        "decoder_with_past_model.onnx": "decoder.onnx",
    }
    for old_name, new_name in rename_map.items():
        old_path = output_path / old_name
        new_path = output_path / new_name
        if old_path.exists():
            old_path.rename(new_path)
            logger.info(f"  Renamed: {old_name} -> {new_name}")

    # Create REBEL config file for Termite
    rebel_config = {
        "model_type": "rebel",
        "model_id": model_id,
        "max_length": 256,
        "num_beams": 3,
        "task": "relation_extraction",
        # Special tokens used by REBEL
        "triplet_token": "<triplet>",
        "subject_token": "<subj>",
        "object_token": "<obj>",
    }

    # Check if multilingual
    if "mrebel" in model_id.lower():
        rebel_config["multilingual"] = True

    config_path = output_path / "rebel_config.json"
    with open(config_path, "w") as f:
        json.dump(rebel_config, f, indent=2)
    logger.info("  Saved: rebel_config.json")

    # List exported files
    logger.info("\nExported files:")
    total_size = 0
    for f in sorted(output_path.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            logger.info(f"  {f.name}: {size/1024/1024:.1f} MB")

    logger.info(f"\nTotal size: {total_size/1024/1024:.1f} MB")
    logger.info(f"\nExport complete!")
    logger.info(f"Output directory: {output_path}")

    # Test inference
    logger.info("\nTesting ONNX inference...")
    test_text = "Steve Jobs founded Apple in Cupertino."
    inputs = tokenizer(test_text, return_tensors="pt", max_length=256, truncation=True)
    outputs = model.generate(**inputs, max_length=256, num_beams=3)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

    logger.info(f"  Input: {test_text}")
    logger.info(f"  Output: {decoded}")

    # Parse and display triplets
    triplets = parse_rebel_output(decoded)
    logger.info(f"\n  Extracted triplets:")
    for t in triplets:
        logger.info(f"    {t['subject']} --[{t['relation']}]--> {t['object']}")


def parse_rebel_output(text: str) -> list[dict]:
    """
    Parse REBEL output into structured triplets.

    REBEL output format:
    <s><triplet> Subject <subj> Object <obj> relation <triplet> ...

    Returns list of dicts with 'subject', 'object', 'relation' keys.
    """
    triplets = []

    # Remove start/end tokens
    text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()

    # Split by triplet token
    parts = text.split("<triplet>")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Parse subject, object, relation
        # Format: "Subject <subj> Object <obj> relation"
        if "<subj>" in part and "<obj>" in part:
            try:
                # Split by <subj> first
                subj_split = part.split("<subj>")
                subject = subj_split[0].strip()

                # The rest contains object and relation
                rest = subj_split[1] if len(subj_split) > 1 else ""

                # Split by <obj>
                obj_split = rest.split("<obj>")
                obj = obj_split[0].strip()
                relation = obj_split[1].strip() if len(obj_split) > 1 else ""

                if subject and obj and relation:
                    triplets.append({
                        "subject": subject,
                        "object": obj,
                        "relation": relation
                    })
            except Exception:
                continue

    return triplets


def main():
    parser = argparse.ArgumentParser(
        description="Export REBEL models to ONNX for relation extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export REBEL large (English)
  uv run scripts/export_rebel.py --output ~/.termite/models/rel/rebel-large

  # Export multilingual mREBEL
  uv run scripts/export_rebel.py --model Babelscape/mrebel-large --output ~/.termite/models/rel/mrebel-large

Available Models:
  - Babelscape/rebel-large      (~3GB, English, 200+ relations)
  - Babelscape/mrebel-large     (~3GB, Multilingual)
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Babelscape/rebel-large",
        help="HuggingFace model ID (default: Babelscape/rebel-large)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./models/rel/rebel-large",
        help="Output directory for the exported model",
    )

    args = parser.parse_args()

    try:
        export_rebel_model(args.model, args.output)
        logger.info("\nAll done!")
    except Exception as e:
        logger.error(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
