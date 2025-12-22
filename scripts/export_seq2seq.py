#!/usr/bin/env python3
"""
Export T5/FLAN-T5/doc2query models to ONNX format for text generation in Termite.

This script exports T5-based encoder-decoder models using Hugging Face's Optimum library,
which creates three separate ONNX files:
  - encoder.onnx (encoder model)
  - decoder-init.onnx (initial decoder, no past_key_values)
  - decoder.onnx (decoder with past_key_values for efficient generation)

Usage:
    python scripts/export_seq2seq.py --model lmqg/flan-t5-small-squad-qg --output ./models/generators/flan-t5-small-qg
    python scripts/export_seq2seq.py --model doc2query/msmarco-t5-small-v1 --output ./models/generators/doc2query-small

Available Models:
    Question Generation (LMQG):
      - lmqg/flan-t5-small-squad-qg  (~77M params, fast)
      - lmqg/flan-t5-base-squad-qg   (~250M params, balanced)
      - lmqg/flan-t5-large-squad-qg  (~780M params, best quality)

    Query Generation (doc2query):
      - doc2query/msmarco-t5-small-v1  (~60M params, fast)
      - doc2query/msmarco-t5-base-v1   (~220M params, balanced)

Input Format (LMQG question generation):
    "generate question: <hl> {answer} <hl> {context_with_answer}"

    Example:
    "generate question: <hl> Python <hl> Python is a high-level programming language."
    -> "What is a high-level programming language?"

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
        import optimum
    except ImportError:
        missing.append("optimum[onnxruntime]")
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import torch
    except ImportError:
        missing.append("torch")

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def export_model(model_name: str, output_dir: str) -> None:
    """
    Export a T5 model to ONNX format using Optimum.

    Args:
        model_name: HuggingFace model name (e.g., 'lmqg/flan-t5-small-squad-qg')
        output_dir: Directory to save the exported model
    """
    from optimum.exporters.onnx import main_export
    from transformers import AutoTokenizer, AutoConfig

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting {model_name} to {output_dir}")

    # Load model config first to get model info
    logger.info("\n1. Loading model configuration...")
    config = AutoConfig.from_pretrained(model_name)

    # Export to ONNX using Optimum
    logger.info("\n2. Exporting to ONNX format...")
    logger.info("   This may take a few minutes...")

    main_export(
        model_name_or_path=model_name,
        output=output_path,
        task="text2text-generation-with-past",
        opset=14,
        device="cpu",
    )

    # Rename files to match expected naming convention for Hugot
    logger.info("\n3. Renaming ONNX files to match Hugot conventions...")
    rename_map = {
        "encoder_model.onnx": "encoder.onnx",
        "decoder_model.onnx": "decoder-init.onnx",
        "decoder_with_past_model.onnx": "decoder.onnx",
    }

    for old_name, new_name in rename_map.items():
        old_path = output_path / old_name
        new_path = output_path / new_name
        if old_path.exists():
            shutil.move(str(old_path), str(new_path))
            logger.info(f"   Renamed {old_name} -> {new_name}")

    # Copy tokenizer files
    logger.info("\n4. Ensuring tokenizer files are present...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(output_path))

    # Create/update config.json with seq2seq-specific settings
    logger.info("\n5. Updating config.json with seq2seq settings...")
    config_dict = config.to_dict()

    if "decoder_start_token_id" not in config_dict:
        config_dict["decoder_start_token_id"] = config_dict.get("pad_token_id", 0)
    if "eos_token_id" not in config_dict:
        config_dict["eos_token_id"] = 1
    if "pad_token_id" not in config_dict:
        config_dict["pad_token_id"] = 0
    if "num_decoder_layers" not in config_dict:
        config_dict["num_decoder_layers"] = config_dict.get("num_layers", 6)

    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Create seq2seq_config.json for Termite
    seq2seq_config = {
        "model_id": model_name,
        "task": "question_generation" if "qg" in model_name.lower() else "query_generation",
        "max_length": 64,
        "num_beams": 1,
        "num_return_sequences": 1,
        "input_format": "generate question: <hl> {answer} <hl> {context}" if "qg" in model_name.lower() else "{document}",
    }
    seq2seq_config_path = output_path / "seq2seq_config.json"
    with open(seq2seq_config_path, "w") as f:
        json.dump(seq2seq_config, f, indent=2)
    logger.info("  Saved: seq2seq_config.json")

    # List exported files
    logger.info("\n6. Export complete!")
    logger.info(f"\nExported files in {output_dir}:")
    for f in sorted(output_path.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"   {f.name} ({size_mb:.1f} MB)")

    logger.info(f"\nUsage:")
    logger.info(f"  termite run --models-dir {output_path.parent.parent}")


def test_exported_model(output_dir: str, test_input: str = None) -> None:
    """Test the exported model with a sample input."""
    from transformers import AutoTokenizer
    import onnxruntime as ort
    import numpy as np

    logger.info("\n" + "=" * 60)
    logger.info("Testing exported model...")
    logger.info("=" * 60)

    output_path = Path(output_dir)

    encoder_file = output_path / "encoder.onnx"
    decoder_init_file = output_path / "decoder-init.onnx"

    if not encoder_file.exists():
        logger.warning(f"Warning: encoder.onnx not found at {encoder_file}")
        return
    if not decoder_init_file.exists():
        logger.warning(f"Warning: decoder-init.onnx not found at {decoder_init_file}")
        return

    tokenizer = AutoTokenizer.from_pretrained(str(output_path))

    logger.info("\nLoading ONNX models...")
    encoder_session = ort.InferenceSession(str(encoder_file))
    decoder_init_session = ort.InferenceSession(str(decoder_init_file))

    if test_input is None:
        test_input = "generate question: <hl> Python <hl> Python is an interpreted, high-level programming language."
    logger.info(f"\nInput: {test_input}")

    inputs = tokenizer(test_input, return_tensors="np", padding=True)
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    logger.info("\nRunning encoder...")
    encoder_outputs = encoder_session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )
    encoder_hidden_states = encoder_outputs[0]
    logger.info(f"Encoder output shape: {encoder_hidden_states.shape}")

    logger.info("\nRunning decoder-init...")
    decoder_input_ids = np.array([[0]], dtype=np.int64)

    decoder_init_inputs = {inp.name for inp in decoder_init_session.get_inputs()}
    logger.info(f"Decoder-init expects inputs: {decoder_init_inputs}")

    feed_dict = {"input_ids": decoder_input_ids}
    if "encoder_hidden_states" in decoder_init_inputs:
        feed_dict["encoder_hidden_states"] = encoder_hidden_states
    if "encoder_attention_mask" in decoder_init_inputs:
        feed_dict["encoder_attention_mask"] = attention_mask

    decoder_outputs = decoder_init_session.run(None, feed_dict)
    logits = decoder_outputs[0]
    logger.info(f"Decoder logits shape: {logits.shape}")

    logger.info("\nGenerating (greedy, max 32 tokens)...")
    generated_ids = [0]

    for _ in range(32):
        decoder_input_ids = np.array([generated_ids], dtype=np.int64)
        feed_dict = {"input_ids": decoder_input_ids}
        if "encoder_hidden_states" in decoder_init_inputs:
            feed_dict["encoder_hidden_states"] = encoder_hidden_states
        if "encoder_attention_mask" in decoder_init_inputs:
            feed_dict["encoder_attention_mask"] = attention_mask

        decoder_outputs = decoder_init_session.run(None, feed_dict)
        logits = decoder_outputs[0]
        next_token = int(np.argmax(logits[0, -1, :]))

        if next_token == 1:
            break
        generated_ids.append(next_token)

    result = tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
    logger.info(f"\nGenerated: {result}")


def main():
    parser = argparse.ArgumentParser(
        description="Export T5/FLAN-T5/doc2query models to ONNX for Termite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export LMQG FLAN-T5 question generation model
  python scripts/export_seq2seq.py --model lmqg/flan-t5-small-squad-qg --output ./models/generators/flan-t5-small-qg

  # Export with testing
  python scripts/export_seq2seq.py --model lmqg/flan-t5-small-squad-qg --output ./models/generators/flan-t5-small-qg --test

  # Export doc2query for query generation
  python scripts/export_seq2seq.py --model doc2query/msmarco-t5-small-v1 --output ./models/generators/doc2query-small

Available Models:
  Question Generation (LMQG - SQuAD trained):
    - lmqg/flan-t5-small-squad-qg  (~77M params)
    - lmqg/flan-t5-base-squad-qg   (~250M params)
    - lmqg/flan-t5-large-squad-qg  (~780M params)

  Query Generation (doc2query - MSMARCO trained):
    - doc2query/msmarco-t5-small-v1  (~60M params)
    - doc2query/msmarco-t5-base-v1   (~220M params)
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="lmqg/flan-t5-small-squad-qg",
        help="HuggingFace model ID (default: lmqg/flan-t5-small-squad-qg)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./models/generators/flan-t5-small-qg",
        help="Output directory for the exported model",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the exported model after export",
    )
    parser.add_argument(
        "--test-input",
        type=str,
        default=None,
        help="Custom input text for testing",
    )

    args = parser.parse_args()

    check_dependencies()

    try:
        export_model(args.model, args.output)
        if args.test:
            test_exported_model(args.output, args.test_input)
        logger.info("\nAll done!")
    except Exception as e:
        logger.error(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
