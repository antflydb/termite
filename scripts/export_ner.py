#!/usr/bin/env python3
"""
Export HuggingFace NER models to ONNX format for use with Termite.

Usage:
    python scripts/export_ner.py dslim/bert-base-NER
    python scripts/export_ner.py dslim/bert-base-NER --quantize
    python scripts/export_ner.py dslim/bert-base-NER --output ./my-models/

After export, pull with termite:
    termite pull hf:dslim/bert-base-NER --type ner
"""

import argparse
import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

def export_ner_model(
    model_name: str,
    output_dir: str | None = None,
    quantize: bool = False,
    opset_version: int = 14,
):
    """Export a HuggingFace NER model to ONNX format."""
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    print(f"Loading model: {model_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.eval()

    # Determine output directory
    if output_dir is None:
        # Default: ~/.cache/huggingface/hub/models--{org}--{name}/onnx/
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        safe_name = model_name.replace("/", "--")
        output_dir = cache_dir / f"models--{safe_name}" / "onnx"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create dummy input
    dummy_text = "John Smith works at Google in New York."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True,
    )

    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    print(f"Exporting to ONNX (opset {opset_version})...")

    # Define dynamic axes for variable sequence length
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size", 1: "sequence"},
    }

    # Add token_type_ids if model uses it
    if "token_type_ids" in inputs:
        dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence"}
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        export_inputs = (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
    else:
        input_names = ["input_ids", "attention_mask"]
        export_inputs = (inputs["input_ids"], inputs["attention_mask"])

    torch.onnx.export(
        model,
        export_inputs,
        str(onnx_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"Exported: {onnx_path}")

    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model validated successfully")

    # Quantize if requested
    if quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantized_path = output_dir / "model_i8.onnx"
        print(f"Quantizing to INT8...")

        quantize_dynamic(
            str(onnx_path),
            str(quantized_path),
            weight_type=QuantType.QInt8,
        )
        print(f"Quantized: {quantized_path}")

    # Save tokenizer files
    print("Saving tokenizer files...")
    tokenizer.save_pretrained(str(output_dir))

    # Copy config.json (contains id2label mapping needed for NER)
    config_path = output_dir / "config.json"
    if not config_path.exists():
        model.config.save_pretrained(str(output_dir))

    print(f"\nExport complete!")
    print(f"Files in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")

    # Print termite pull command
    print(f"\nTo use with termite:")
    print(f"  termite pull hf:{model_name} --type ner")

    return output_dir


def verify_export(output_dir: Path, model_name: str):
    """Verify the exported model works correctly."""
    import onnxruntime as ort
    from transformers import AutoTokenizer
    import numpy as np

    print(f"\nVerifying export...")

    tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    session = ort.InferenceSession(str(output_dir / "model.onnx"))

    # Test inference
    test_text = "Barack Obama was born in Hawaii."
    inputs = tokenizer(
        test_text,
        return_tensors="np",
        padding="max_length",
        max_length=128,
        truncation=True,
    )

    # Build input dict for ONNX
    ort_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    if "token_type_ids" in inputs:
        ort_inputs["token_type_ids"] = inputs["token_type_ids"]

    outputs = session.run(None, ort_inputs)
    logits = outputs[0]

    predictions = np.argmax(logits, axis=-1)[0]

    # Load label mapping from config
    import json
    with open(output_dir / "config.json") as f:
        config = json.load(f)

    id2label = config.get("id2label", {})

    # Decode tokens and show entities
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    print(f"\nTest input: \"{test_text}\"")
    print("Detected entities:")

    for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        label = id2label.get(str(pred_id), f"LABEL_{pred_id}")
        if label != "O":
            print(f"  {token}: {label}")

    print("\nVerification complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace NER models to ONNX for Termite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export bert-base-NER
  python scripts/export_ner.py dslim/bert-base-NER

  # Export with INT8 quantization
  python scripts/export_ner.py dslim/bert-base-NER --quantize

  # Export distilbert NER
  python scripts/export_ner.py elastic/distilbert-base-cased-finetuned-conll03-english

  # Custom output directory
  python scripts/export_ner.py dslim/bert-base-NER --output ./models/ner/bert-base-NER/

After export, use with termite:
  termite pull hf:dslim/bert-base-NER --type ner
  termite run --models-dir ./models
        """,
    )
    parser.add_argument("model_name", help="HuggingFace model name (e.g., dslim/bert-base-NER)")
    parser.add_argument("--output", "-o", help="Output directory (default: HF cache)")
    parser.add_argument("--quantize", "-q", action="store_true", help="Also create INT8 quantized version")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version (default: 14)")
    parser.add_argument("--verify", "-v", action="store_true", help="Verify export with test inference")

    args = parser.parse_args()

    check_dependencies()

    output_dir = export_ner_model(
        args.model_name,
        output_dir=args.output,
        quantize=args.quantize,
        opset_version=args.opset,
    )

    if args.verify:
        verify_export(output_dir, args.model_name)


if __name__ == "__main__":
    main()
