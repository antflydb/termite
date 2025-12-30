#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "optimum[onnxruntime]",
#     "transformers",
#     "torch",
#     "boto3",
#     "pillow",
#     "onnx",
#     "onnxscript",
#     "onnxconverter-common",
#     "gliner",
#     "onnxruntime-genai",
# ]
# ///
"""
Export HuggingFace models to ONNX format and prepare for Antfly model registry.

This unified script handles embedders, rerankers, chunkers, recognizers (NER/extraction),
rewriters (seq2seq), generators (LLMs), and multimodal models, generating
the manifest files needed for the R2-hosted registry.

Usage:
    # Export an embedder (FP32 only)
    ./export_model_to_registry.py embedder BAAI/bge-small-en-v1.5

    # Export a reranker with int8 quantization
    ./export_model_to_registry.py reranker mixedbread-ai/mxbai-rerank-base-v1 --variants i8

    # Export a reranker with FP16 precision (recommended for ARM64/M-series)
    ./export_model_to_registry.py reranker mixedbread-ai/mxbai-rerank-base-v1 --variants f16

    # Export with both i8 and f16 variants
    ./export_model_to_registry.py reranker mixedbread-ai/mxbai-rerank-base-v1 --variants f16 i8

    # Export a chunker
    ./export_model_to_registry.py chunker mirth/chonky_mmbert_small_multilingual_1

    # Export a CLIP multimodal model with variants
    ./export_model_to_registry.py embedder openai/clip-vit-base-patch32 --capabilities multimodal --backends onnx --variants f16 i8

    # Export a traditional NER model (BERT-based TokenClassification)
    ./export_model_to_registry.py recognizer dslim/bert-base-NER --capabilities labels

    # Export a GLiNER zero-shot NER model
    ./export_model_to_registry.py recognizer urchade/gliner_small-v2.1 --capabilities labels,zeroshot

    # Export a GLiNER multitask model (NER + relations + QA)
    ./export_model_to_registry.py recognizer knowledgator/gliner-multitask-large-v0.5 --capabilities labels,zeroshot,relations,answers

    # Export a REBEL relation extraction model
    ./export_model_to_registry.py recognizer Babelscape/rebel-large --capabilities relations

    # Export a seq2seq rewriter model (e.g., question generation)
    ./export_model_to_registry.py rewriter lmqg/flan-t5-small-squad-qg

    # Export a generative LLM (default FP16)
    ./export_model_to_registry.py generator google/gemma-3-1b-it

    # Export a generative LLM with INT4 quantization
    ./export_model_to_registry.py generator google/gemma-3-1b-it --variants f16 i4

    # Export a generative LLM with CUDA-optimized INT4
    ./export_model_to_registry.py generator google/gemma-3-1b-it --variants i4-cuda

    # Export and upload to R2
    ./export_model_to_registry.py embedder BAAI/bge-small-en-v1.5 --upload

    # Specify custom output name
    ./export_model_to_registry.py embedder BAAI/bge-small-en-v1.5 --name bge-small-en-v1.5

Recognizer Capabilities:
    labels    - Entity extraction (NER) - extracts labeled spans from text
    zeroshot  - Supports arbitrary labels at inference time (GLiNER models)
    relations - Relation extraction between entities (GLiNER multitask, REBEL)
    answers   - Extractive question answering (GLiNER multitask)

Output Structure:
    output/
    ├── manifests/
    │   └── bge-small-en-v1.5.json    # Registry manifest
    ├── blobs/
    │   ├── sha256:abc123...          # model.onnx
    │   ├── sha256:def456...          # tokenizer.json
    │   └── sha256:ghi789...          # config.json
    └── models/
        └── embedders/
            └── bge-small-en-v1.5/    # Local model directory
                ├── model.onnx
                ├── tokenizer.json
                └── config.json
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Literal

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)

ModelType = Literal["embedder", "reranker", "chunker", "recognizer", "rewriter", "generator"]

# Recognizer capabilities - these describe what extraction tasks the model supports
# Used in manifest to advertise model capabilities to Termite
RECOGNIZER_CAPABILITIES = {
    "labels",     # Entity extraction (NER) - extracts labeled spans
    "zeroshot",   # Supports arbitrary labels at inference time (GLiNER)
    "relations",  # Relation extraction between entities (GLiNER multitask, REBEL)
    "answers",    # Extractive question answering (GLiNER multitask)
}

# Model type to ORT class mapping
MODEL_TYPE_CONFIG = {
    "embedder": {
        "ort_class": "ORTModelForFeatureExtraction",
        "default_model": "BAAI/bge-small-en-v1.5",
        "dir_name": "embedders",
    },
    "reranker": {
        "ort_class": "ORTModelForSequenceClassification",
        "default_model": "mixedbread-ai/mxbai-rerank-base-v1",
        "dir_name": "rerankers",
    },
    "chunker": {
        "ort_class": "ORTModelForTokenClassification",
        "default_model": "mirth/chonky_mmbert_small_multilingual_1",
        "dir_name": "chunkers",
    },
    "recognizer": {
        "ort_class": "ORTModelForTokenClassification",
        "default_model": "dslim/bert-base-NER",
        "dir_name": "recognizers",
    },
    "rewriter": {
        "ort_class": None,  # Uses Optimum seq2seq export
        "default_model": "lmqg/flan-t5-small-squad-qg",
        "dir_name": "rewriters",
    },
    "generator": {
        "ort_class": None,  # Uses onnxruntime-genai model builder
        "default_model": "google/gemma-3-1b-it",
        "dir_name": "generators",
    },
}

# Files to include in the manifest (in order of importance)
MANIFEST_FILES = [
    "model.onnx",
    "model.onnx_data",  # External data for large ONNX models
    "tokenizer.json",
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]

# Additional files for multimodal models
MULTIMODAL_MANIFEST_FILES = [
    "visual_model.onnx",
    "visual_model.onnx.data",  # External data for large visual encoder
    "text_model.onnx",
    "text_model.onnx.data",  # External data for large text encoder
    "visual_projection.onnx",
    "visual_projection.onnx.data",  # External data for projection layer
    "text_projection.onnx",
    "text_projection.onnx.data",  # External data for projection layer
    "tokenizer.json",
    "config.json",
    "clip_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
]

# Files for seq2seq/rewriter models (T5, FLAN-T5, etc.)
SEQ2SEQ_MANIFEST_FILES = [
    "encoder.onnx",
    "encoder.onnx_data",
    "decoder-init.onnx",
    "decoder-init.onnx_data",
    "decoder.onnx",
    "decoder.onnx_data",
    "tokenizer.json",
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",  # SentencePiece model for T5
    "seq2seq_config.json",
]

# Files for GLiNER models
GLINER_MANIFEST_FILES = [
    "model.onnx",
    "model.onnx_data",
    "tokenizer.json",
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "gliner_config.json",
]

# Files for REBEL relation extraction models (seq2seq architecture)
REBEL_MANIFEST_FILES = [
    "encoder.onnx",
    "encoder.onnx_data",
    "decoder-init.onnx",
    "decoder-init.onnx_data",
    "decoder.onnx",
    "decoder.onnx_data",
    "tokenizer.json",
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "rebel_config.json",
]

# Files for generator models (ONNX Runtime GenAI)
GENERATOR_MANIFEST_FILES = [
    "genai_config.json",
    "model.onnx",
    "model.onnx.data",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "generation_config.json",
]


def detect_recognizer_type(model_id: str) -> tuple[str, list[str]]:
    """
    Detect recognizer architecture and capabilities from model ID.

    Returns:
        tuple of (architecture, capabilities):
        - architecture: "gliner", "rebel", or "ner"
        - capabilities: list of detected capabilities
    """
    model_id_lower = model_id.lower()

    # GLiNER models
    if "gliner" in model_id_lower:
        capabilities = ["labels", "zeroshot"]
        # Multitask GLiNER models support relations and QA
        if "multitask" in model_id_lower or "relex" in model_id_lower:
            capabilities.extend(["relations", "answers"])
        return "gliner", capabilities

    # REBEL relation extraction models
    if "rebel" in model_id_lower or "mrebel" in model_id_lower:
        return "rebel", ["relations"]

    # Traditional NER models (BERT-based TokenClassification)
    return "ner", ["labels"]


def is_gliner_model(model_id: str) -> bool:
    """Check if model ID is a GLiNER model."""
    return "gliner" in model_id.lower()


def is_rebel_model(model_id: str) -> bool:
    """Check if model ID is a REBEL relation extraction model."""
    model_id_lower = model_id.lower()
    return "rebel" in model_id_lower or "mrebel" in model_id_lower


def convert_to_fp16(input_path: Path, output_path: Path) -> None:
    """
    Convert an ONNX model from FP32 to FP16 precision.

    FP16 advantages over int8 quantization for modern ARM processors:
    - Native FP16 hardware support (M-series chips)
    - No quantization/dequantization overhead
    - Better accuracy than int8
    - 2x memory reduction
    - Faster inference on ARM64 with FP16 SIMD

    Args:
        input_path: Path to input FP32 ONNX model
        output_path: Path to output FP16 ONNX model
    """
    import onnx
    from onnxconverter_common import float16

    logger.info(f"Converting to FP16: {input_path.name} -> {output_path.name}")

    # Load FP32 model
    model_fp32 = onnx.load(str(input_path))

    # Convert to FP16
    # keep_io_types=True preserves input/output as FP32 for easier integration
    model_fp16 = float16.convert_float_to_float16(
        model_fp32,
        keep_io_types=True,  # Keep inputs/outputs as FP32
        disable_shape_infer=False,  # Run shape inference
    )

    # Save FP16 model
    onnx.save(model_fp16, str(output_path))

    # Log size reduction
    fp32_size = input_path.stat().st_size / (1024 * 1024)
    fp16_size = output_path.stat().st_size / (1024 * 1024)
    reduction = (1 - fp16_size / fp32_size) * 100
    logger.info(f"  FP16 size: {fp16_size:.2f} MB (was {fp32_size:.2f} MB, {reduction:.1f}% reduction)")


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return f"sha256:{sha256_hash.hexdigest()}"


def get_ort_model_class(model_type: ModelType):
    """Get the appropriate ORT model class for the model type."""
    if model_type == "multimodal":
        return None  # CLIP uses custom export

    from optimum.onnxruntime import (
        ORTModelForFeatureExtraction,
        ORTModelForSequenceClassification,
        ORTModelForTokenClassification,
    )

    classes = {
        "embedder": ORTModelForFeatureExtraction,
        "reranker": ORTModelForSequenceClassification,
        "chunker": ORTModelForTokenClassification,
    }
    return classes[model_type]


def export_model(
    model_type: ModelType,
    model_id: str,
    output_dir: Path,
    variants: list[str] | None = None,
    capabilities: list[str] | None = None,
) -> Path:
    """
    Export a HuggingFace model to ONNX format.

    Args:
        model_type: Type of model (embedder, reranker, chunker)
        model_id: HuggingFace model ID
        output_dir: Directory to save the model
        variants: List of variant types to create (e.g., ["f16", "i8"])
        capabilities: List of capabilities (e.g., ["multimodal"])

    Returns the path to the exported model directory.
    """
    variants = variants or []
    capabilities = capabilities or []

    # Multimodal models (e.g., CLIP) use a special export path
    if "multimodal" in capabilities:
        return export_multimodal_model(model_id, output_dir, variants)

    from transformers import AutoTokenizer
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting {model_type}: {model_id}")
    logger.info(f"Output: {output_dir}")

    # Get the appropriate ORT class
    ort_class = get_ort_model_class(model_type)

    # Load and export
    logger.info("Converting to ONNX format...")
    ort_model = ort_class.from_pretrained(model_id, export=True)
    ort_model.save_pretrained(output_dir)

    # Save tokenizer
    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)

    # Create int8 quantized variant if requested (must be done BEFORE fp16 to avoid multi-file issue)
    if "i8" in variants:
        logger.info("Applying dynamic quantization (int8)...")
        try:
            quantizer = ORTQuantizer.from_pretrained(output_dir)
            dqconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
            quantizer.quantize(save_dir=output_dir, quantization_config=dqconfig)
            # Rename the quantized file to use new naming convention
            old_quantized = output_dir / "model_quantized.onnx"
            new_quantized = output_dir / "model_i8.onnx"
            if old_quantized.exists():
                old_quantized.rename(new_quantized)
            logger.info("Quantization complete")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")

    # Create FP16 variant if requested (done after i8 to avoid multi-file issue with ORTQuantizer)
    if "f16" in variants:
        logger.info("Creating FP16 variant...")
        try:
            fp32_path = output_dir / "model.onnx"
            fp16_path = output_dir / "model_f16.onnx"
            convert_to_fp16(fp32_path, fp16_path)
            logger.info("FP16 conversion complete")
        except Exception as e:
            logger.warning(f"FP16 conversion failed: {e}")

    return output_dir


def export_multimodal_model(
    model_id: str,
    output_dir: Path,
    variants: list[str] | None = None,
) -> Path:
    """
    Export a CLIP model to ONNX format.

    CLIP models have separate visual and text encoders that are exported
    as separate ONNX files.

    Args:
        model_id: HuggingFace model ID
        output_dir: Directory to save the model
        variants: List of variant types to create (e.g., ["f16", "i8"])
    """
    variants = variants or []
    import torch
    import onnx
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting multimodal (CLIP): {model_id}")
    logger.info(f"Output: {output_dir}")

    # Load model and processor
    logger.info("Loading CLIP model...")
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)

    model.eval()

    # Get image size from processor
    image_size = processor.image_processor.size.get("shortest_edge", 224)
    if isinstance(image_size, dict):
        image_size = image_size.get("height", 224)

    # Export visual encoder
    logger.info("Exporting visual encoder...")
    visual_path = output_dir / "visual_model.onnx"
    dummy_pixel_values = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model.vision_model,
        (dummy_pixel_values,),
        str(visual_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
            "pooler_output": {0: "batch_size"},
        },
    )
    onnx_model = onnx.load(str(visual_path))
    onnx.checker.check_model(onnx_model)
    logger.info(f"  Visual encoder saved: {visual_path}")

    # Export text encoder
    logger.info("Exporting text encoder...")
    text_path = output_dir / "text_model.onnx"
    dummy_text = ["a photo of a cat"]
    inputs = tokenizer(
        dummy_text,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    torch.onnx.export(
        model.text_model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(text_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            "pooler_output": {0: "batch_size"},
        },
    )
    onnx_model = onnx.load(str(text_path))
    onnx.checker.check_model(onnx_model)
    logger.info(f"  Text encoder saved: {text_path}")

    # Save configs
    logger.info("Saving configuration files...")
    model.config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Create CLIP-specific config
    clip_config = {
        "model_type": "clip",
        "vision_config": {
            "hidden_size": model.config.vision_config.hidden_size,
            "image_size": model.config.vision_config.image_size,
            "patch_size": model.config.vision_config.patch_size,
            "projection_dim": model.config.projection_dim,
        },
        "text_config": {
            "hidden_size": model.config.text_config.hidden_size,
            "max_position_embeddings": model.config.text_config.max_position_embeddings,
            "projection_dim": model.config.projection_dim,
        },
        "projection_dim": model.config.projection_dim,
    }
    with open(output_dir / "clip_config.json", "w") as f:
        json.dump(clip_config, f, indent=2)

    # Export projection layers as ONNX for consistent runtime loading
    logger.info("Exporting projection layers...")

    # Visual projection: [batch, 768] -> [batch, 512]
    visual_proj_path = output_dir / "visual_projection.onnx"
    dummy_visual = torch.randn(1, model.config.vision_config.hidden_size)
    torch.onnx.export(
        model.visual_projection,
        dummy_visual,
        str(visual_proj_path),
        export_params=True,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    logger.info(f"  visual_projection.onnx: {model.visual_projection.weight.shape}")

    # Text projection: [batch, 512] -> [batch, 512]
    text_proj_path = output_dir / "text_projection.onnx"
    dummy_text = torch.randn(1, model.config.text_config.hidden_size)
    torch.onnx.export(
        model.text_projection,
        dummy_text,
        str(text_proj_path),
        export_params=True,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    logger.info(f"  text_projection.onnx: {model.text_projection.weight.shape}")

    # Create FP16 variants if requested
    if "f16" in variants:
        logger.info("Creating FP16 variants...")
        try:
            convert_to_fp16(visual_path, output_dir / "visual_model_f16.onnx")
            logger.info("  Visual encoder converted to FP16")

            convert_to_fp16(text_path, output_dir / "text_model_f16.onnx")
            logger.info("  Text encoder converted to FP16")
        except Exception as e:
            logger.warning(f"FP16 conversion failed: {e}")

    # Create int8 quantized variants if requested
    if "i8" in variants:
        logger.info("Applying dynamic quantization (int8)...")
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(
                model_input=str(visual_path),
                model_output=str(output_dir / "visual_model_i8.onnx"),
                weight_type=QuantType.QUInt8,
            )
            logger.info("  Visual encoder quantized")

            quantize_dynamic(
                model_input=str(text_path),
                model_output=str(output_dir / "text_model_i8.onnx"),
                weight_type=QuantType.QUInt8,
            )
            logger.info("  Text encoder quantized")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")

    return output_dir


def export_gliner_model(
    model_id: str,
    output_dir: Path,
    variants: list[str] | None = None,
) -> Path:
    """
    Export a GLiNER zero-shot NER model to ONNX format.

    GLiNER models can extract any entity type without retraining - just specify
    the entity labels at inference time.

    Args:
        model_id: HuggingFace model ID (e.g., urchade/gliner_small-v2.1)
        output_dir: Directory to save the model
        variants: List of variant types to create (e.g., ["f16", "i8"])
    """
    import shutil
    from gliner import GLiNER

    variants = variants or []
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting GLiNER model: {model_id}")
    logger.info(f"Output: {output_dir}")

    # Load the GLiNER model
    logger.info("Loading GLiNER model...")
    model = GLiNER.from_pretrained(model_id)

    # Export to ONNX using GLiNER's built-in method
    logger.info("Exporting to ONNX format...")
    temp_onnx_dir = output_dir / "temp_onnx"
    temp_onnx_dir.mkdir(exist_ok=True)

    try:
        # Export using GLiNER's built-in method
        use_int8 = "i8" in variants
        model.export_to_onnx(str(temp_onnx_dir), quantize=use_int8)

        # Move the exported files to the final location
        for file in temp_onnx_dir.iterdir():
            if file.is_file():
                dest = output_dir / file.name
                shutil.move(str(file), str(dest))
                logger.info(f"  Saved: {file.name}")
    finally:
        if temp_onnx_dir.exists():
            shutil.rmtree(temp_onnx_dir)

    # Rename quantized file if created
    quantized_file = output_dir / "model_quantized.onnx"
    if quantized_file.exists():
        quantized_file.rename(output_dir / "model_i8.onnx")
        logger.info("  Renamed: model_quantized.onnx -> model_i8.onnx")

    # Apply FP16 conversion if requested
    if "f16" in variants:
        onnx_file = output_dir / "model.onnx"
        if onnx_file.exists():
            fp16_file = output_dir / "model_f16.onnx"
            convert_to_fp16(onnx_file, fp16_file)

    # Create GLiNER config file for Termite
    gliner_config = {
        "max_width": 12,
        "default_labels": ["person", "organization", "location", "date", "product"],
        "threshold": 0.5,
        "flat_ner": True,
        "multi_label": False,
        "model_id": model_id,
    }

    config_path = output_dir / "gliner_config.json"
    with open(config_path, "w") as f:
        json.dump(gliner_config, f, indent=2)
    logger.info("  Saved: gliner_config.json")

    # Save tokenizer
    try:
        if hasattr(model, 'data_processor') and hasattr(model.data_processor, 'transformer_tokenizer'):
            tokenizer = model.data_processor.transformer_tokenizer
            tokenizer.save_pretrained(str(output_dir))
            logger.info("  Saved: tokenizer files")
        elif hasattr(model, 'tokenizer'):
            model.tokenizer.save_pretrained(str(output_dir))
            logger.info("  Saved: tokenizer files")
        else:
            from transformers import AutoTokenizer
            base_model = "microsoft/deberta-v3-small"
            logger.info(f"  Loading tokenizer from base model: {base_model}")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            tokenizer.save_pretrained(str(output_dir))
            logger.info("  Saved: tokenizer files")
    except Exception as e:
        logger.warning(f"  Could not save tokenizer: {e}")

    return output_dir


def export_rebel_model(
    model_id: str,
    output_dir: Path,
    variants: list[str] | None = None,
) -> Path:
    """
    Export a REBEL relation extraction model to ONNX format.

    REBEL (Relation Extraction By End-to-end Language generation) is a seq2seq model
    based on BART that extracts relation triplets from text.

    Args:
        model_id: HuggingFace model ID (e.g., Babelscape/rebel-large)
        output_dir: Directory to save the model
        variants: List of variant types (not used for REBEL currently)
    """
    import shutil
    from optimum.exporters.onnx import main_export
    from transformers import AutoTokenizer, AutoConfig

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting REBEL model: {model_id}")
    logger.info(f"Output: {output_dir}")

    # Load model config
    logger.info("Loading model configuration...")
    config = AutoConfig.from_pretrained(model_id)

    # Export to ONNX using Optimum (REBEL is based on BART, so use seq2seq task)
    logger.info("Exporting to ONNX format (this may take a few minutes)...")
    main_export(
        model_name_or_path=model_id,
        output=output_dir,
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
        old_path = output_dir / old_name
        new_path = output_dir / new_name
        if old_path.exists():
            shutil.move(str(old_path), str(new_path))
            logger.info(f"  Renamed {old_name} -> {new_name}")

    # Save tokenizer
    logger.info("Saving tokenizer files...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(str(output_dir))

    # Create REBEL config file for Termite
    is_multilingual = "mrebel" in model_id.lower()
    rebel_config = {
        "model_type": "rebel",
        "model_id": model_id,
        "max_length": 256,
        "num_beams": 3,
        "task": "relation_extraction",
        "multilingual": is_multilingual,
        # Special tokens used by REBEL for parsing output
        "triplet_token": "<triplet>",
        "subject_token": "<subj>",
        "object_token": "<obj>",
    }

    config_path = output_dir / "rebel_config.json"
    with open(config_path, "w") as f:
        json.dump(rebel_config, f, indent=2)
    logger.info("  Saved: rebel_config.json")

    return output_dir


def export_seq2seq_model(
    model_id: str,
    output_dir: Path,
    variants: list[str] | None = None,
) -> Path:
    """
    Export a T5/FLAN-T5/seq2seq model to ONNX format for text generation.

    Uses Hugging Face's Optimum library to create three ONNX files:
      - encoder.onnx
      - decoder-init.onnx (initial decoder, no past_key_values)
      - decoder.onnx (decoder with past_key_values)

    Args:
        model_id: HuggingFace model ID (e.g., lmqg/flan-t5-small-squad-qg)
        output_dir: Directory to save the model
        variants: List of variant types (not used for seq2seq currently)
    """
    import shutil
    from optimum.exporters.onnx import main_export
    from transformers import AutoTokenizer, AutoConfig

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting seq2seq model: {model_id}")
    logger.info(f"Output: {output_dir}")

    # Load model config first to get model info
    logger.info("Loading model configuration...")
    config = AutoConfig.from_pretrained(model_id)

    # Export to ONNX using Optimum
    logger.info("Exporting to ONNX format (this may take a few minutes)...")
    main_export(
        model_name_or_path=model_id,
        output=output_dir,
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
        old_path = output_dir / old_name
        new_path = output_dir / new_name
        if old_path.exists():
            shutil.move(str(old_path), str(new_path))
            logger.info(f"  Renamed {old_name} -> {new_name}")

    # Ensure tokenizer files are present
    logger.info("Saving tokenizer files...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(str(output_dir))

    # Create/update config.json with seq2seq-specific settings
    logger.info("Updating config.json with seq2seq settings...")
    config_dict = config.to_dict()

    if "decoder_start_token_id" not in config_dict:
        config_dict["decoder_start_token_id"] = config_dict.get("pad_token_id", 0)
    if "eos_token_id" not in config_dict:
        config_dict["eos_token_id"] = 1
    if "pad_token_id" not in config_dict:
        config_dict["pad_token_id"] = 0
    if "num_decoder_layers" not in config_dict:
        config_dict["num_decoder_layers"] = config_dict.get("num_layers", 6)

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Create seq2seq_config.json for Termite
    seq2seq_config = {
        "model_id": model_id,
        "task": "question_generation" if "qg" in model_id.lower() else "query_generation",
        "max_length": 64,
        "num_beams": 1,
        "num_return_sequences": 1,
        "input_format": "generate question: <hl> {answer} <hl> {context}" if "qg" in model_id.lower() else "{document}",
    }
    seq2seq_config_path = output_dir / "seq2seq_config.json"
    with open(seq2seq_config_path, "w") as f:
        json.dump(seq2seq_config, f, indent=2)
    logger.info("  Saved: seq2seq_config.json")

    return output_dir


def export_generator_model(
    model_id: str,
    output_dir: Path,
    variants: list[str] | None = None,
) -> Path:
    """
    Export a generative LLM to ONNX format using ONNX Runtime GenAI model builder.

    Uses the onnxruntime-genai model builder to create optimized ONNX models
    for text generation with support for different precisions and hardware targets.

    Args:
        model_id: HuggingFace model ID (e.g., google/gemma-3-1b-it)
        output_dir: Directory to save the model
        variants: List of variant types to create:
            - "f16": FP16 precision (works on CPU and CUDA)
            - "i4": INT4 quantized for CPU
            - "i4-cuda": INT4 quantized for CUDA
            - "i4-dml": INT4 quantized for DirectML (Windows)
    """
    import subprocess

    variants = variants or ["f16"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting generator model: {model_id}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Variants: {', '.join(variants)}")

    # Map variant names to builder arguments
    VARIANT_CONFIG = {
        "f16": {"precision": "fp16", "execution_provider": "cpu"},
        "i4": {"precision": "int4", "execution_provider": "cpu"},
        "i4-cuda": {"precision": "int4", "execution_provider": "cuda"},
        "i4-dml": {"precision": "int4", "execution_provider": "dml"},
    }

    # Export each variant
    for variant in variants:
        if variant not in VARIANT_CONFIG:
            logger.warning(f"Unknown variant: {variant}, skipping")
            continue

        config = VARIANT_CONFIG[variant]
        precision = config["precision"]
        exec_provider = config["execution_provider"]

        # Determine output path for this variant
        if variant == "f16":
            # Base variant goes directly in output_dir
            variant_dir = output_dir
        else:
            # Other variants go in subdirectories named by variant
            variant_dir = output_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nExporting variant: {variant}")
        logger.info(f"  Precision: {precision}")
        logger.info(f"  Execution provider: {exec_provider}")
        logger.info(f"  Output: {variant_dir}")

        # Build the model builder command
        cmd = [
            "python", "-m", "onnxruntime_genai.models.builder",
            "-m", model_id,
            "-o", str(variant_dir),
            "-p", precision,
            "-e", exec_provider,
        ]

        logger.info(f"  Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    logger.info(f"  {line}")
            logger.info(f"  Variant {variant} exported successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"  Failed to export variant {variant}")
            if e.stdout:
                logger.error(f"  stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"  stderr: {e.stderr}")
            raise RuntimeError(f"Model builder failed for variant {variant}: {e.stderr}")
        except FileNotFoundError:
            logger.error("  onnxruntime-genai not installed")
            logger.error("  Install with: pip install onnxruntime-genai")
            raise RuntimeError("onnxruntime-genai package not found")

    # Log exported files
    logger.info("\nExported files:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            rel_path = f.relative_to(output_dir)
            logger.info(f"  {rel_path}: {size_mb:.2f} MB")

    return output_dir


def generate_manifest(
    model_type: ModelType,
    model_name: str,
    model_dir: Path,
    description: str = "",
    source: str = "",
    owner: str = "",
    capabilities: list[str] | None = None,
    backends: list[str] | None = None,
    recognizer_arch: str | None = None,
) -> dict:
    """Generate a registry manifest for the exported model.

    Args:
        model_type: Type of model (embedder, reranker, chunker, recognizer, etc.)
        model_name: Name for the model in the registry
        model_dir: Directory containing the exported model files
        description: Human-readable description
        source: Source model ID (e.g., HuggingFace model ID)
        owner: Model owner/organization (e.g., "BAAI", "sentence-transformers")
        capabilities: List of capabilities (e.g., ["labels", "zeroshot", "relations"])
        backends: List of supported backends (e.g., ["onnx"])
        recognizer_arch: For recognizers, the architecture type: "gliner", "rebel", or "ner"
    """
    files = []
    variants = {}

    # Use appropriate file list based on model type and capabilities
    is_multimodal = capabilities and "multimodal" in capabilities
    is_generator = model_type == "generator"
    is_gliner = recognizer_arch == "gliner"
    is_rebel = recognizer_arch == "rebel"

    if is_multimodal:
        file_list = MULTIMODAL_MANIFEST_FILES
    elif model_type == "rewriter":
        file_list = SEQ2SEQ_MANIFEST_FILES
    elif is_gliner:
        file_list = GLINER_MANIFEST_FILES
    elif is_rebel:
        file_list = REBEL_MANIFEST_FILES
    elif is_generator:
        file_list = GENERATOR_MANIFEST_FILES
    else:
        file_list = MANIFEST_FILES

    for filename in file_list:
        filepath = model_dir / filename
        if filepath.exists():
            digest = compute_sha256(filepath)
            size = filepath.stat().st_size
            files.append({
                "name": filename,
                "digest": digest,
                "size": size,
            })
            logger.info(f"  {filename}: {size:,} bytes ({digest[:20]}...)")

    # Check for variant models using the new naming convention
    # Variant ID -> filename mapping
    VARIANT_FILENAMES = {
        "f16": "model_f16.onnx",
        "bf16": "model_bf16.onnx",
        "i8": "model_i8.onnx",
        "i8-st": "model_i8-st.onnx",
        "i4": "model_i4.onnx",
    }

    if is_multimodal:
        # CLIP has separate visual and text encoders for each variant
        # Check for f16 variants
        visual_f16_path = model_dir / "visual_model_f16.onnx"
        text_f16_path = model_dir / "text_model_f16.onnx"
        if visual_f16_path.exists() and text_f16_path.exists():
            variants["f16"] = []
            for vt_path in [visual_f16_path, text_f16_path]:
                digest = compute_sha256(vt_path)
                size = vt_path.stat().st_size
                variants["f16"].append({
                    "name": vt_path.name,
                    "digest": digest,
                    "size": size,
                })
                logger.info(f"  {vt_path.name}: {size:,} bytes ({digest[:20]}...)")

        # Check for i8 variants
        visual_i8_path = model_dir / "visual_model_i8.onnx"
        text_i8_path = model_dir / "text_model_i8.onnx"
        if visual_i8_path.exists() and text_i8_path.exists():
            variants["i8"] = []
            for qt_path in [visual_i8_path, text_i8_path]:
                digest = compute_sha256(qt_path)
                size = qt_path.stat().st_size
                variants["i8"].append({
                    "name": qt_path.name,
                    "digest": digest,
                    "size": size,
                })
                logger.info(f"  {qt_path.name}: {size:,} bytes ({digest[:20]}...)")
    elif is_generator:
        # Generator variants are stored in subdirectories (i4/, i4-cuda/, i4-dml/)
        GENERATOR_VARIANTS = ["i4", "i4-cuda", "i4-dml"]
        for variant_id in GENERATOR_VARIANTS:
            variant_dir = model_dir / variant_id
            if variant_dir.exists() and variant_dir.is_dir():
                variant_files = []
                for filename in GENERATOR_MANIFEST_FILES:
                    filepath = variant_dir / filename
                    if filepath.exists():
                        digest = compute_sha256(filepath)
                        size = filepath.stat().st_size
                        variant_files.append({
                            "name": f"{variant_id}/{filename}",
                            "digest": digest,
                            "size": size,
                        })
                        logger.info(f"  {variant_id}/{filename}: {size:,} bytes ({digest[:20]}...)")
                if variant_files:
                    variants[variant_id] = variant_files
    else:
        # Check for all known variant files
        for variant_id, filename in VARIANT_FILENAMES.items():
            variant_path = model_dir / filename
            if variant_path.exists():
                digest = compute_sha256(variant_path)
                size = variant_path.stat().st_size
                variant_files = [{
                    "name": filename,
                    "digest": digest,
                    "size": size,
                }]
                logger.info(f"  {filename}: {size:,} bytes ({digest[:20]}...)")

                # Check for external data file for this variant
                data_filename = filename.replace(".onnx", ".onnx_data")
                data_path = model_dir / data_filename
                if data_path.exists():
                    data_digest = compute_sha256(data_path)
                    data_size = data_path.stat().st_size
                    variant_files.append({
                        "name": data_filename,
                        "digest": data_digest,
                        "size": data_size,
                    })
                    logger.info(f"  {data_filename}: {data_size:,} bytes ({data_digest[:20]}...)")

                # Store as list if multiple files, single dict if just one
                variants[variant_id] = variant_files if len(variant_files) > 1 else variant_files[0]

    manifest = {
        "schemaVersion": 2,
        "name": model_name,
        "type": model_type,
        "description": description,
        "source": source,
        "files": files,
    }

    # Add owner if specified
    if owner:
        manifest["owner"] = owner

    # Add provenance information
    from datetime import datetime, timezone
    manifest["provenance"] = {
        "downloadedFrom": f"https://huggingface.co/{source}" if source else "",
        "downloadedAt": datetime.now(timezone.utc).isoformat(),
    }

    # Add capabilities and backends if specified
    if capabilities:
        manifest["capabilities"] = capabilities
    if backends:
        manifest["backends"] = backends

    if variants:
        manifest["variants"] = variants

    return manifest


def prepare_registry_files(
    manifest: dict,
    model_dir: Path,
    output_dir: Path,
) -> None:
    """
    Prepare files for registry upload.

    Creates:
    - manifests/<owner>/<name>.json (or manifests/<name>.json if no owner)
    - blobs/sha256:...
    """
    manifests_dir = output_dir / "manifests"
    blobs_dir = output_dir / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    # Write manifest with owner structure if available
    if manifest.get("owner"):
        owner_dir = manifests_dir / manifest["owner"]
        owner_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = owner_dir / f"{manifest['name']}.json"
    else:
        manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifests_dir / f"{manifest['name']}.json"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Created manifest: {manifest_path}")

    # Copy files to blobs directory (content-addressable)
    all_files = manifest["files"].copy()
    if manifest.get("variants"):
        # Handle variants map
        for variant_id, variant_info in manifest["variants"].items():
            if isinstance(variant_info, list):
                # Multimodal has list of files per variant
                all_files.extend(variant_info)
            else:
                all_files.append(variant_info)

    for file_info in all_files:
        src = model_dir / file_info["name"]
        dst = blobs_dir / file_info["digest"]
        if not dst.exists():
            shutil.copy2(src, dst)
            logger.info(f"Created blob: {file_info['digest'][:30]}... ({file_info['name']})")
        else:
            logger.info(f"Blob exists: {file_info['digest'][:30]}... ({file_info['name']})")


def upload_to_r2(
    output_dir: Path,
    bucket: str,
    endpoint_url: str,
    prefix: str = "v1",
) -> None:
    """Upload registry files to R2."""
    import boto3

    logger.info(f"Uploading to R2 bucket: {bucket}")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    # Upload manifests (including owner subdirectories)
    manifests_dir = output_dir / "manifests"
    for manifest_file in manifests_dir.rglob("*.json"):
        # Get relative path to preserve owner/name structure
        rel_path = manifest_file.relative_to(manifests_dir)
        key = f"{prefix}/manifests/{rel_path}"
        logger.info(f"Uploading {key}...")
        s3.upload_file(
            str(manifest_file),
            bucket,
            key,
            ExtraArgs={"ContentType": "application/json"},
        )

    # Upload blobs
    blobs_dir = output_dir / "blobs"
    for blob_file in blobs_dir.iterdir():
        if blob_file.is_file():
            key = f"{prefix}/blobs/{blob_file.name}"
            # Check if blob already exists
            try:
                s3.head_object(Bucket=bucket, Key=key)
                logger.info(f"Blob exists, skipping: {key}")
                continue
            except s3.exceptions.ClientError:
                pass

            logger.info(f"Uploading {key}...")
            s3.upload_file(
                str(blob_file),
                bucket,
                key,
                ExtraArgs={"ContentType": "application/octet-stream"},
            )

    # Upload index.json
    index_file = output_dir / "index.json"
    if index_file.exists():
        key = f"{prefix}/index.json"
        logger.info(f"Uploading {key}...")
        s3.upload_file(
            str(index_file),
            bucket,
            key,
            ExtraArgs={"ContentType": "application/json"},
        )

    logger.info("Upload complete!")


def fetch_remote_index(
    bucket: str,
    endpoint_url: str,
    prefix: str = "v1",
) -> dict | None:
    """Fetch the existing index.json from R2."""
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    try:
        key = f"{prefix}/index.json"
        response = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise


def update_registry_index(
    output_dir: Path,
    manifest: dict,
    remote_index: dict | None = None,
) -> None:
    """Update or create the registry index.json.

    If remote_index is provided, it will be used as the base instead of
    the local index.json file. This ensures we don't overwrite entries
    that exist in the remote registry but not locally.
    """
    index_path = output_dir / "index.json"

    # Use remote index if provided, otherwise load local or create new
    if remote_index is not None:
        index = remote_index
        logger.info("Using remote index as base")
    elif index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        index = {"schemaVersion": 2, "models": []}

    # Upgrade schema version if needed
    if index.get("schemaVersion", 1) < 2:
        index["schemaVersion"] = 2

    # Calculate total size (base files only, variants are optional downloads)
    total_size = sum(f["size"] for f in manifest["files"])

    # Collect variant IDs
    variant_ids = []
    if manifest.get("variants"):
        variant_ids = list(manifest["variants"].keys())

    # Build full name for matching (owner/name if owner present)
    owner = manifest.get("owner", "")
    full_name = f"{owner}/{manifest['name']}" if owner else manifest["name"]

    # Create index entry
    entry = {
        "name": manifest["name"],
        "type": manifest["type"],
        "description": manifest.get("description", ""),
        "source": manifest.get("source", ""),
        "size": total_size,
        "variants": variant_ids,
    }

    # Add owner if present
    if owner:
        entry["owner"] = owner

    # Add capabilities and backends if present
    if manifest.get("capabilities"):
        entry["capabilities"] = manifest["capabilities"]
    if manifest.get("backends"):
        entry["backends"] = manifest["backends"]

    # Helper to get full name from entry
    def get_full_name(m):
        e_owner = m.get("owner", "")
        return f"{e_owner}/{m['name']}" if e_owner else m["name"]

    # Update or add entry (match by full owner/name)
    existing_idx = next(
        (i for i, m in enumerate(index["models"]) if get_full_name(m) == full_name),
        None
    )
    if existing_idx is not None:
        index["models"][existing_idx] = entry
    else:
        index["models"].append(entry)

    # Sort by full name (owner/name)
    index["models"].sort(key=get_full_name)

    # Write index
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    logger.info(f"Updated index: {index_path}")


def list_r2_objects(
    s3_client,
    bucket: str,
    prefix: str,
) -> list[dict]:
    """List all objects in R2 with given prefix."""
    objects = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append(obj)
    return objects


def fetch_manifest_from_r2(
    s3_client,
    bucket: str,
    key: str,
) -> dict | None:
    """Fetch a manifest from R2."""
    from botocore.exceptions import ClientError

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise


def collect_digests_from_manifest(manifest: dict) -> set[str]:
    """Collect all blob digests referenced by a manifest."""
    digests = set()

    # Base files
    for f in manifest.get("files", []):
        digests.add(f["digest"])

    # Variant files
    for variant_info in manifest.get("variants", {}).values():
        if isinstance(variant_info, list):
            # Multimodal: list of files per variant
            for f in variant_info:
                digests.add(f["digest"])
        else:
            # Single file variant
            digests.add(variant_info["digest"])

    return digests


def garbage_collect(
    bucket: str,
    endpoint_url: str,
    prefix: str = "v1",
    dry_run: bool = True,
    local_index_path: Path | None = None,
) -> None:
    """Garbage collect orphaned manifests and blobs from R2.

    This function uses the local index.json as the source of truth:
    1. Uploads local index.json to R2
    2. Finds orphaned manifests (in R2 but not in local index) and deletes them
    3. Identifies orphaned blobs (not referenced by remaining manifests)
    4. Deletes orphaned blobs (unless dry_run=True)
    """
    import boto3

    logger.info("=" * 60)
    logger.info("Registry Garbage Collection")
    logger.info("=" * 60)
    logger.info(f"Bucket:   {bucket}")
    logger.info(f"Prefix:   {prefix}")
    logger.info(f"Dry run:  {dry_run}")
    logger.info("=" * 60)

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    # Step 1: Load local index as source of truth
    logger.info("\n[1/6] Loading local index.json as source of truth...")
    if local_index_path is None:
        local_index_path = Path("./registry/index.json")

    if not local_index_path.exists():
        logger.error(f"Local index not found: {local_index_path}")
        logger.error("Run an export first or create index.json manually")
        return

    with open(local_index_path) as f:
        local_index = json.load(f)

    indexed_models = {m["name"] for m in local_index.get("models", [])}
    logger.info(f"Local index contains {len(indexed_models)} models:")
    for name in sorted(indexed_models):
        logger.info(f"  - {name}")

    # Step 2: List all manifests in R2
    logger.info("\n[2/6] Listing manifests in R2...")
    manifest_objects = list_r2_objects(s3, bucket, f"{prefix}/manifests/")
    r2_manifest_keys = {obj["Key"] for obj in manifest_objects}
    logger.info(f"Found {len(r2_manifest_keys)} manifest files in R2")

    # Extract model names from R2 manifests
    r2_model_names = set()
    for key in r2_manifest_keys:
        filename = key.split("/")[-1]
        if filename.endswith(".json"):
            model_name = filename[:-5]  # Remove .json
            r2_model_names.add(model_name)

    # Step 3: Find orphaned manifests (in R2 but not in local index)
    logger.info("\n[3/6] Finding orphaned manifests...")
    orphaned_manifests = r2_model_names - indexed_models
    if orphaned_manifests:
        logger.info(f"Found {len(orphaned_manifests)} orphaned manifests (in R2 but not in local index):")
        for name in sorted(orphaned_manifests):
            logger.info(f"  - {name}")
    else:
        logger.info("No orphaned manifests found")

    # Step 4: Find missing manifests (in local index but not in R2)
    logger.info("\n[4/7] Checking for missing manifests...")
    missing_manifests = indexed_models - r2_model_names
    if missing_manifests:
        logger.warning(f"Found {len(missing_manifests)} models in index without manifests in R2:")
        for name in sorted(missing_manifests):
            logger.warning(f"  - {name} (will be removed from index)")
        # Remove missing entries from local index before uploading
        local_index["models"] = [m for m in local_index["models"] if m["name"] not in missing_manifests]
    else:
        logger.info("All indexed models have manifests in R2")

    # Step 5: Check for updated manifests (local differs from R2)
    logger.info("\n[5/7] Checking for updated manifests...")
    local_manifests_dir = local_index_path.parent / "manifests"
    updated_manifests = set()

    for model_name in indexed_models & r2_model_names:  # Models in both local and R2
        local_manifest_path = local_manifests_dir / f"{model_name}.json"
        if not local_manifest_path.exists():
            continue

        # Load local manifest and compute hash
        with open(local_manifest_path, "rb") as f:
            local_content = f.read()
        local_hash = hashlib.sha256(local_content).hexdigest()

        # Fetch R2 manifest and compute hash
        r2_key = f"{prefix}/manifests/{model_name}.json"
        r2_manifest = fetch_manifest_from_r2(s3, bucket, r2_key)
        if r2_manifest:
            r2_content = json.dumps(r2_manifest, indent=2).encode("utf-8")
            r2_hash = hashlib.sha256(r2_content).hexdigest()

            if local_hash != r2_hash:
                updated_manifests.add(model_name)
                logger.info(f"  {model_name}: changed (will sync)")

    if updated_manifests:
        logger.info(f"Found {len(updated_manifests)} manifests to update")
    else:
        logger.info("All manifests are up to date")

    # Step 5b: Collect digests from updated local manifests to check for missing blobs
    updated_manifest_digests = set()
    for model_name in updated_manifests:
        local_manifest_path = local_manifests_dir / f"{model_name}.json"
        if local_manifest_path.exists():
            with open(local_manifest_path) as f:
                local_manifest = json.load(f)
            digests = collect_digests_from_manifest(local_manifest)
            updated_manifest_digests.update(digests)

    # Step 6: Collect referenced blobs from remaining valid manifests
    logger.info("\n[6/7] Collecting referenced blobs from valid manifests...")
    referenced_digests = set()
    valid_model_names = indexed_models - missing_manifests  # Models that are in index AND have manifests

    for key in r2_manifest_keys:
        filename = key.split("/")[-1]
        model_name = filename[:-5] if filename.endswith(".json") else filename

        # Only collect blobs from manifests that will be kept
        if model_name in valid_model_names:
            manifest = fetch_manifest_from_r2(s3, bucket, key)
            if manifest:
                digests = collect_digests_from_manifest(manifest)
                referenced_digests.update(digests)
                logger.info(f"  {model_name}: {len(digests)} blobs (keeping)")
        elif model_name in orphaned_manifests:
            logger.info(f"  {model_name}: (orphaned, will delete)")

    logger.info(f"Total referenced blobs: {len(referenced_digests)}")

    # Step 6: Find orphaned blobs
    logger.info("\n[7/7] Finding orphaned blobs...")
    blob_objects = list_r2_objects(s3, bucket, f"{prefix}/blobs/")
    all_blob_digests = set()
    blob_sizes = {}

    for obj in blob_objects:
        digest = obj["Key"].split("/")[-1]
        all_blob_digests.add(digest)
        blob_sizes[digest] = obj["Size"]

    orphaned_blobs = all_blob_digests - referenced_digests
    orphaned_size = sum(blob_sizes.get(d, 0) for d in orphaned_blobs)

    logger.info(f"Total blobs in R2: {len(all_blob_digests)}")
    logger.info(f"Orphaned blobs: {len(orphaned_blobs)} ({orphaned_size / 1024 / 1024:.1f} MB)")

    if orphaned_blobs:
        logger.info("\nOrphaned blobs:")
        for digest in sorted(orphaned_blobs)[:10]:
            size = blob_sizes.get(digest, 0)
            logger.info(f"  - {digest[:20]}... ({size / 1024 / 1024:.1f} MB)")
        if len(orphaned_blobs) > 10:
            logger.info(f"  ... and {len(orphaned_blobs) - 10} more")

    # Check for missing blobs from updated manifests
    missing_blobs = updated_manifest_digests - all_blob_digests
    local_blobs_dir = local_index_path.parent / "blobs"
    uploadable_blobs = set()
    missing_blob_sizes = {}

    for digest in missing_blobs:
        blob_path = local_blobs_dir / digest
        if blob_path.exists():
            uploadable_blobs.add(digest)
            missing_blob_sizes[digest] = blob_path.stat().st_size

    missing_size = sum(missing_blob_sizes.get(d, 0) for d in uploadable_blobs)

    if uploadable_blobs:
        logger.info(f"\nMissing blobs to upload: {len(uploadable_blobs)} ({missing_size / 1024 / 1024:.1f} MB)")
        for digest in sorted(uploadable_blobs)[:10]:
            size = missing_blob_sizes.get(digest, 0)
            logger.info(f"  - {digest[:20]}... ({size / 1024 / 1024:.1f} MB)")
        if len(uploadable_blobs) > 10:
            logger.info(f"  ... and {len(uploadable_blobs) - 10} more")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Manifests to update: {len(updated_manifests)}")
    logger.info(f"Missing blobs to upload: {len(uploadable_blobs)} ({missing_size / 1024 / 1024:.1f} MB)")
    logger.info(f"Orphaned manifests to delete: {len(orphaned_manifests)}")
    logger.info(f"Orphaned blobs to delete: {len(orphaned_blobs)} ({orphaned_size / 1024 / 1024:.1f} MB)")

    if dry_run:
        logger.info("\nDry run - no changes made. Run with --no-dry-run to apply.")
        return

    # Apply changes
    # Upload missing blobs first (before manifests that reference them)
    if uploadable_blobs:
        logger.info("\nUploading missing blobs to R2...")
        uploaded = 0
        for digest in sorted(uploadable_blobs):
            blob_path = local_blobs_dir / digest
            key = f"{prefix}/blobs/{digest}"
            size_mb = blob_path.stat().st_size / 1024 / 1024
            logger.info(f"  Uploading {digest[:20]}... ({size_mb:.1f} MB)")
            s3.upload_file(
                str(blob_path),
                bucket,
                key,
                ExtraArgs={"ContentType": "application/octet-stream"},
            )
            uploaded += 1
        logger.info(f"Uploaded {uploaded} blobs")

    # Upload updated manifests to R2
    if updated_manifests:
        logger.info("\nUploading updated manifests to R2...")
        for model_name in sorted(updated_manifests):
            local_manifest_path = local_manifests_dir / f"{model_name}.json"
            with open(local_manifest_path, "rb") as f:
                manifest_content = f.read()
            s3.put_object(
                Bucket=bucket,
                Key=f"{prefix}/manifests/{model_name}.json",
                Body=manifest_content,
                ContentType="application/json",
            )
            logger.info(f"  Updated {model_name}.json")
        logger.info(f"Updated {len(updated_manifests)} manifests")

    # Upload local index to R2
    logger.info("\nUploading local index.json to R2...")
    local_index["models"].sort(key=lambda m: m["name"])
    index_json = json.dumps(local_index, indent=2)
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/index.json",
        Body=index_json.encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(f"Updated R2 index.json ({len(local_index['models'])} models)")

    # Delete orphaned manifests
    if orphaned_manifests:
        logger.info("\nDeleting orphaned manifests...")
        for name in orphaned_manifests:
            key = f"{prefix}/manifests/{name}.json"
            s3.delete_object(Bucket=bucket, Key=key)
            logger.info(f"  Deleted {name}.json")
        logger.info(f"Deleted {len(orphaned_manifests)} orphaned manifests")

    # Delete orphaned blobs
    if orphaned_blobs:
        logger.info("\nDeleting orphaned blobs...")
        deleted = 0
        for digest in orphaned_blobs:
            key = f"{prefix}/blobs/{digest}"
            s3.delete_object(Bucket=bucket, Key=key)
            deleted += 1
            if deleted % 10 == 0:
                logger.info(f"  Deleted {deleted}/{len(orphaned_blobs)} blobs...")
        logger.info(f"Deleted {deleted} orphaned blobs")

    logger.info("\nGarbage collection complete!")


def test_model(
    model_type: ModelType,
    model_dir: Path,
    capabilities: list[str] | None = None,
    recognizer_arch: str | None = None,
) -> bool:
    """Test the exported model."""
    logger.info("Testing exported model...")
    capabilities = capabilities or []

    if "multimodal" in capabilities:
        return test_multimodal_model(model_dir)

    if model_type == "rewriter":
        return test_seq2seq_model(model_dir)

    if model_type == "generator":
        return test_generator_model(model_dir)

    if recognizer_arch == "gliner":
        return test_gliner_model(model_dir)

    if recognizer_arch == "rebel":
        return test_rebel_model(model_dir)

    try:
        from transformers import AutoTokenizer

        ort_class = get_ort_model_class(model_type)
        model = ort_class.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Simple test based on model type
        if model_type == "embedder":
            inputs = tokenizer(
                ["Test sentence for embedding."],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = model(**inputs)
            assert outputs.last_hidden_state is not None
            logger.info(f"  Embedding shape: {outputs.last_hidden_state.shape}")

        elif model_type == "reranker":
            inputs = tokenizer(
                [["Query", "Document to rerank"]],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = model(**inputs)
            assert outputs.logits is not None
            logger.info(f"  Logits shape: {outputs.logits.shape}")

        elif model_type == "chunker" or model_type == "recognizer":
            inputs = tokenizer(
                ["Test text for chunking." if model_type == "chunker" else "John Smith works at Google in New York."],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = model(**inputs)
            assert outputs.logits is not None
            logger.info(f"  Logits shape: {outputs.logits.shape}")

        logger.info("Test passed!")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


def test_gliner_model(model_dir: Path) -> bool:
    """Test a GLiNER model."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    try:
        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            logger.error(f"model.onnx not found in {model_dir}")
            return False

        logger.info("Loading GLiNER ONNX model...")
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # Test with sample input
        test_text = "Tim Cook is the CEO of Apple Inc. in Cupertino."
        inputs = tokenizer(
            test_text,
            return_tensors="np",
            padding="max_length",
            max_length=128,
            truncation=True,
        )

        logger.info(f"Test input: \"{test_text}\"")
        logger.info(f"Input shape: {inputs['input_ids'].shape}")
        logger.info("Test passed!")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


def test_rebel_model(model_dir: Path) -> bool:
    """Test a REBEL relation extraction model."""
    import onnxruntime as ort
    import numpy as np
    from transformers import AutoTokenizer

    try:
        encoder_path = model_dir / "encoder.onnx"
        decoder_init_path = model_dir / "decoder-init.onnx"

        if not encoder_path.exists():
            logger.error(f"encoder.onnx not found in {model_dir}")
            return False
        if not decoder_init_path.exists():
            logger.error(f"decoder-init.onnx not found in {model_dir}")
            return False

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        logger.info("Loading encoder...")
        encoder_session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])

        logger.info("Loading decoder-init...")
        decoder_init_session = ort.InferenceSession(str(decoder_init_path), providers=["CPUExecutionProvider"])

        # Test with relation extraction input
        test_input = "Tim Cook is the CEO of Apple Inc. Apple is headquartered in Cupertino."
        logger.info(f"Test input: \"{test_input}\"")

        inputs = tokenizer(test_input, return_tensors="np", padding=True)
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        logger.info("Running encoder...")
        encoder_outputs = encoder_session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        encoder_hidden_states = encoder_outputs[0]
        logger.info(f"Encoder output shape: {encoder_hidden_states.shape}")

        logger.info("Running decoder-init...")
        decoder_input_ids = np.array([[0]], dtype=np.int64)

        decoder_init_inputs = {inp.name for inp in decoder_init_session.get_inputs()}
        feed_dict = {"input_ids": decoder_input_ids}
        if "encoder_hidden_states" in decoder_init_inputs:
            feed_dict["encoder_hidden_states"] = encoder_hidden_states
        if "encoder_attention_mask" in decoder_init_inputs:
            feed_dict["encoder_attention_mask"] = attention_mask

        decoder_outputs = decoder_init_session.run(None, feed_dict)
        logits = decoder_outputs[0]
        logger.info(f"Decoder logits shape: {logits.shape}")

        logger.info("REBEL relation extraction test passed!")
        return True

    except Exception as e:
        logger.error(f"REBEL test failed: {e}")
        return False


def test_seq2seq_model(model_dir: Path) -> bool:
    """Test a seq2seq/rewriter model."""
    import onnxruntime as ort
    import numpy as np
    from transformers import AutoTokenizer

    try:
        encoder_path = model_dir / "encoder.onnx"
        decoder_init_path = model_dir / "decoder-init.onnx"

        if not encoder_path.exists():
            logger.error(f"encoder.onnx not found in {model_dir}")
            return False
        if not decoder_init_path.exists():
            logger.error(f"decoder-init.onnx not found in {model_dir}")
            return False

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        logger.info("Loading encoder...")
        encoder_session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])

        logger.info("Loading decoder-init...")
        decoder_init_session = ort.InferenceSession(str(decoder_init_path), providers=["CPUExecutionProvider"])

        # Test with sample input
        test_input = "generate question: <hl> Python <hl> Python is an interpreted, high-level programming language."
        logger.info(f"Test input: \"{test_input}\"")

        inputs = tokenizer(test_input, return_tensors="np", padding=True)
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        logger.info("Running encoder...")
        encoder_outputs = encoder_session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        encoder_hidden_states = encoder_outputs[0]
        logger.info(f"Encoder output shape: {encoder_hidden_states.shape}")

        logger.info("Running decoder-init...")
        decoder_input_ids = np.array([[0]], dtype=np.int64)

        decoder_init_inputs = {inp.name for inp in decoder_init_session.get_inputs()}
        feed_dict = {"input_ids": decoder_input_ids}
        if "encoder_hidden_states" in decoder_init_inputs:
            feed_dict["encoder_hidden_states"] = encoder_hidden_states
        if "encoder_attention_mask" in decoder_init_inputs:
            feed_dict["encoder_attention_mask"] = attention_mask

        decoder_outputs = decoder_init_session.run(None, feed_dict)
        logits = decoder_outputs[0]
        logger.info(f"Decoder logits shape: {logits.shape}")

        logger.info("Test passed!")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


def test_multimodal_model(model_dir: Path) -> bool:
    """Test the exported CLIP model."""
    import onnxruntime as ort
    import numpy as np
    from PIL import Image
    from transformers import CLIPProcessor, CLIPTokenizerFast

    try:
        visual_path = model_dir / "visual_model.onnx"
        text_path = model_dir / "text_model.onnx"
        clip_config_path = model_dir / "clip_config.json"

        # Load CLIP config for expected dimensions
        with open(clip_config_path) as f:
            clip_config = json.load(f)

        expected_visual_dim = clip_config["vision_config"]["hidden_size"]
        expected_text_dim = clip_config["text_config"]["hidden_size"]
        projection_dim = clip_config["projection_dim"]

        # Load processor and tokenizer
        processor = CLIPProcessor.from_pretrained(model_dir)
        tokenizer = CLIPTokenizerFast.from_pretrained(model_dir)

        # Test visual encoder
        logger.info("Testing visual encoder...")
        visual_session = ort.InferenceSession(str(visual_path), providers=["CPUExecutionProvider"])

        image_size = processor.image_processor.size.get("shortest_edge", 224)
        if isinstance(image_size, dict):
            image_size = image_size.get("height", 224)

        dummy_image = Image.new("RGB", (image_size, image_size), color="red")
        pixel_values = processor(images=dummy_image, return_tensors="np")["pixel_values"]

        visual_outputs = visual_session.run(None, {"pixel_values": pixel_values})
        pooler_output = visual_outputs[1]
        logger.info(f"  Visual embedding shape: {pooler_output.shape}")

        # Verify visual encoder output dimension
        if pooler_output.shape[-1] != expected_visual_dim:
            logger.error(f"Test failed: Visual dim {pooler_output.shape[-1]} != expected {expected_visual_dim}")
            return False

        # Test text encoder
        logger.info("Testing text encoder...")
        text_session = ort.InferenceSession(str(text_path), providers=["CPUExecutionProvider"])

        text_inputs = tokenizer(
            ["a photo of a cat"],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )

        text_outputs = text_session.run(
            None,
            {
                "input_ids": text_inputs["input_ids"],
                "attention_mask": text_inputs["attention_mask"],
            },
        )
        text_pooler = text_outputs[1]
        logger.info(f"  Text embedding shape: {text_pooler.shape}")

        # Verify text encoder output dimension
        if text_pooler.shape[-1] != expected_text_dim:
            logger.error(f"Test failed: Text dim {text_pooler.shape[-1]} != expected {expected_text_dim}")
            return False

        logger.info(f"  Visual hidden size: {expected_visual_dim}")
        logger.info(f"  Text hidden size: {expected_text_dim}")
        logger.info(f"  Projection dimension: {projection_dim}")

        # Test projection layers
        logger.info("Testing projection layers...")
        visual_proj_path = model_dir / "visual_projection.onnx"
        text_proj_path = model_dir / "text_projection.onnx"

        if not visual_proj_path.exists() or not text_proj_path.exists():
            logger.error("Test failed: Projection layer files not found")
            return False

        visual_proj_session = ort.InferenceSession(str(visual_proj_path), providers=["CPUExecutionProvider"])
        text_proj_session = ort.InferenceSession(str(text_proj_path), providers=["CPUExecutionProvider"])

        # Apply projections using ONNX runtime
        visual_projected = visual_proj_session.run(None, {"input": pooler_output.astype(np.float32)})[0]
        text_projected = text_proj_session.run(None, {"input": text_pooler.astype(np.float32)})[0]

        logger.info(f"  Projected visual shape: {visual_projected.shape}")
        logger.info(f"  Projected text shape: {text_projected.shape}")

        if visual_projected.shape[-1] != text_projected.shape[-1]:
            logger.error("Test failed: Projected dimensions don't match!")
            return False

        if visual_projected.shape[-1] != projection_dim:
            logger.error(f"Test failed: Projected dim {visual_projected.shape[-1]} != expected {projection_dim}")
            return False

        logger.info(f"  Final embedding dimension: {projection_dim}")
        logger.info("Test passed!")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


def test_generator_model(model_dir: Path) -> bool:
    """Test an ONNX Runtime GenAI generator model."""
    try:
        genai_config_path = model_dir / "genai_config.json"
        if not genai_config_path.exists():
            logger.error(f"genai_config.json not found in {model_dir}")
            return False

        # Load genai config
        logger.info("Loading genai_config.json...")
        with open(genai_config_path) as f:
            genai_config = json.load(f)

        logger.info(f"  Model type: {genai_config.get('model', {}).get('type', 'unknown')}")
        logger.info(f"  Vocab size: {genai_config.get('model', {}).get('vocab_size', 'unknown')}")

        # Check for model.onnx
        model_path = model_dir / "model.onnx"
        if not model_path.exists():
            logger.error(f"model.onnx not found in {model_dir}")
            return False

        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Model size: {model_size_mb:.2f} MB")

        # Try to load with onnxruntime-genai if available
        try:
            import onnxruntime_genai as og

            logger.info("Loading model with onnxruntime-genai...")
            model = og.Model(str(model_dir))
            tokenizer = og.Tokenizer(model)

            # Simple generation test
            test_prompt = "Hello, my name is"
            logger.info(f"Test prompt: \"{test_prompt}\"")

            tokens = tokenizer.encode(test_prompt)
            params = og.GeneratorParams(model)
            params.set_search_options(max_length=20, do_sample=False)
            params.input_ids = tokens

            generator = og.Generator(model, params)

            # Generate a few tokens
            generated_tokens = []
            for _ in range(5):
                if generator.is_done():
                    break
                generator.compute_logits()
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                generated_tokens.append(new_token)

            output_text = tokenizer.decode(generated_tokens)
            logger.info(f"  Generated: \"{output_text}\"")
            logger.info("Test passed!")
            return True

        except ImportError:
            logger.warning("  onnxruntime-genai not installed, skipping inference test")
            logger.info("  Install with: pip install onnxruntime-genai")
            logger.info("  Basic file validation passed")
            return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


def cmd_gc(args):
    """Handle the gc subcommand."""
    endpoint = args.r2_endpoint or os.environ.get("AWS_ENDPOINT_URL")
    if not endpoint:
        logger.error("Endpoint not specified. Set --r2-endpoint or AWS_ENDPOINT_URL")
        sys.exit(1)

    garbage_collect(
        bucket=args.r2_bucket,
        endpoint_url=endpoint,
        dry_run=not args.no_dry_run,
        local_index_path=args.index_path,
    )


def cmd_export(args):
    """Handle the export subcommand."""
    # Get default model if not specified
    model_id = args.model_id or MODEL_TYPE_CONFIG[args.model_type]["default_model"]

    # Auto-detect recognizer architecture and capabilities
    recognizer_arch = None
    capabilities = list(args.capabilities) if args.capabilities else []

    if args.model_type == "recognizer":
        recognizer_arch, detected_capabilities = detect_recognizer_type(model_id)
        # Merge detected capabilities with user-specified ones (user can override)
        if not capabilities:
            capabilities = detected_capabilities
        logger.info(f"Detected recognizer architecture: {recognizer_arch}")
        logger.info(f"Detected capabilities: {', '.join(detected_capabilities)}")

    # Parse owner and model name from model_id
    parts = model_id.split("/")
    if len(parts) == 2:
        model_owner = parts[0]
        model_name = args.name or parts[1].replace("_", "-").lower()
    else:
        model_owner = "unknown"
        model_name = args.name or model_id.replace("_", "-").lower()

    # Setup paths with owner/model structure
    config = MODEL_TYPE_CONFIG[args.model_type]
    model_dir = args.output_dir / "models" / config["dir_name"] / model_owner / model_name

    logger.info("=" * 60)
    logger.info("Antfly Model Registry Export")
    logger.info("=" * 60)
    logger.info(f"Type:        {args.model_type}")
    if recognizer_arch:
        logger.info(f"Architecture:{recognizer_arch}")
    logger.info(f"Source:      {model_id}")
    logger.info(f"Owner:       {model_owner}")
    logger.info(f"Name:        {model_name}")
    logger.info(f"Output:      {args.output_dir}")
    logger.info(f"Variants:    {', '.join(args.variants) if args.variants else 'none'}")
    logger.info(f"Capabilities:{', '.join(capabilities) if capabilities else 'none'}")
    logger.info(f"Backends:    {', '.join(args.backends) if args.backends else 'none'}")
    logger.info("=" * 60)

    # Export model using appropriate function
    logger.info("\n[1/4] Exporting model to ONNX...")
    if args.model_type == "rewriter":
        export_seq2seq_model(model_id, model_dir, args.variants)
    elif args.model_type == "generator":
        export_generator_model(model_id, model_dir, args.variants)
    elif recognizer_arch == "gliner":
        export_gliner_model(model_id, model_dir, args.variants)
    elif recognizer_arch == "rebel":
        export_rebel_model(model_id, model_dir, args.variants)
    else:
        export_model(args.model_type, model_id, model_dir, args.variants, capabilities)

    # Test model
    if not args.no_test:
        logger.info("\n[2/4] Testing exported model...")
        if not test_model(args.model_type, model_dir, capabilities, recognizer_arch=recognizer_arch):
            logger.error("Model test failed, aborting")
            sys.exit(1)
    else:
        logger.info("\n[2/4] Skipping model test")

    # Generate manifest
    logger.info("\n[3/4] Generating registry manifest...")
    manifest = generate_manifest(
        args.model_type,
        model_name,
        model_dir,
        args.description,
        source=model_id,
        owner=model_owner,
        capabilities=capabilities if capabilities else None,
        backends=args.backends if args.backends else None,
        recognizer_arch=recognizer_arch,
    )

    # Save local manifest to model directory
    local_manifest_path = model_dir / "model_manifest.json"
    with open(local_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved local manifest: {local_manifest_path}")

    # Prepare registry files
    prepare_registry_files(manifest, model_dir, args.output_dir)

    # Fetch remote index if uploading to avoid overwriting existing entries
    remote_index = None
    endpoint = args.r2_endpoint or os.environ.get("AWS_ENDPOINT_URL")
    if args.upload:
        if not endpoint:
            logger.error("Endpoint not specified. Set --r2-endpoint or AWS_ENDPOINT_URL")
            sys.exit(1)
        logger.info("Fetching existing index from R2...")
        remote_index = fetch_remote_index(args.r2_bucket, endpoint)
        if remote_index:
            logger.info(f"Found {len(remote_index.get('models', []))} existing models in remote index")
        else:
            logger.info("No existing index found in R2, creating new one")

    update_registry_index(args.output_dir, manifest, remote_index)

    # Upload to R2
    if args.upload:
        logger.info("\n[4/4] Uploading to R2...")
        upload_to_r2(args.output_dir, args.r2_bucket, endpoint)
    else:
        logger.info("\n[4/4] Skipping R2 upload (use --upload to enable)")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Export complete!")
    logger.info("=" * 60)
    logger.info(f"\nLocal model:  {model_dir}")
    logger.info(f"Manifest:     {args.output_dir}/manifests/{model_name}.json")
    logger.info(f"Index:        {args.output_dir}/index.json")
    logger.info(f"\nTo use locally:")
    logger.info(f"  cp -r {model_dir} ./models/{config['dir_name']}/")
    if not args.upload:
        logger.info(f"\nTo upload to R2:")
        logger.info(f"  {sys.argv[0]} {args.model_type} {model_id} --upload --r2-endpoint <endpoint>")


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace models to ONNX for Antfly registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export BGE embedder (FP32 only)
  %(prog)s embedder BAAI/bge-small-en-v1.5

  # Export reranker with FP16 variant
  %(prog)s reranker mixedbread-ai/mxbai-rerank-base-v1 --variants f16

  # Export reranker with int8 and FP16 variants
  %(prog)s reranker mixedbread-ai/mxbai-rerank-base-v1 --variants f16 i8

  # Export CLIP multimodal model
  %(prog)s embedder openai/clip-vit-base-patch32 --capabilities multimodal --backends onnx --variants f16 i8

  # Export and upload to R2
  %(prog)s embedder BAAI/bge-small-en-v1.5 --upload --r2-bucket my-bucket

  # Custom output name
  %(prog)s embedder BAAI/bge-small-en-v1.5 --name bge-small-en

  # Export a traditional NER model (BERT-based TokenClassification)
  %(prog)s recognizer dslim/bert-base-NER

  # Export a GLiNER zero-shot NER model (auto-detected)
  %(prog)s recognizer urchade/gliner_small-v2.1

  # Export a GLiNER multitask model (NER + relations + QA)
  %(prog)s recognizer knowledgator/gliner-multitask-large-v0.5

  # Export a REBEL relation extraction model (auto-detected)
  %(prog)s recognizer Babelscape/rebel-large

  # Export a seq2seq rewriter model (e.g., question generation)
  %(prog)s rewriter lmqg/flan-t5-small-squad-qg

  # Export a generative LLM (default: FP16)
  %(prog)s generator google/gemma-3-1b-it

  # Export a generative LLM with INT4 quantization
  %(prog)s generator google/gemma-3-1b-it --variants f16 i4

  # Export a generative LLM with CUDA-optimized INT4
  %(prog)s generator google/gemma-3-1b-it --variants i4-cuda

  # Garbage collect orphaned blobs and index entries (dry run)
  %(prog)s gc

  # Garbage collect and apply changes
  %(prog)s gc --no-dry-run

Variant Types:
  f16       FP16 precision (recommended for ARM64/M-series chips)
  i8        INT8 dynamic quantization
  i8-st     INT8 static quantization (not yet implemented)
  i4        INT4 quantization for CPU (generators only)
  i4-cuda   INT4 quantization for CUDA (generators only)
  i4-dml    INT4 quantization for DirectML/Windows (generators only)

Environment Variables (for --upload and gc):
  AWS_ACCESS_KEY_ID      Access key (standard AWS env var, works with R2)
  AWS_SECRET_ACCESS_KEY  Secret key (standard AWS env var, works with R2)
  AWS_ENDPOINT_URL       S3-compatible endpoint (e.g., https://<account>.r2.cloudflarestorage.com)
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # GC subcommand
    gc_parser = subparsers.add_parser(
        "gc",
        help="Garbage collect orphaned manifests and blobs from R2",
        description="Remove orphaned manifests and blobs from R2. Uses local index.json as source of truth.",
    )
    gc_parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually delete orphaned items (default is dry-run)",
    )
    gc_parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("./registry/index.json"),
        help="Path to local index.json (source of truth)",
    )
    gc_parser.add_argument(
        "--r2-bucket",
        default="antfly-model-registry",
        help="R2 bucket name",
    )
    gc_parser.add_argument(
        "--r2-endpoint",
        help="S3-compatible endpoint URL (or set AWS_ENDPOINT_URL env var)",
    )

    # Export subcommands (one for each model type)
    for model_type in ["embedder", "reranker", "chunker", "recognizer", "rewriter", "generator"]:
        export_parser = subparsers.add_parser(
            model_type,
            help=f"Export a {model_type} model to ONNX",
        )
        export_parser.add_argument(
            "model_id",
            nargs="?",
            help="HuggingFace model ID (default: type-specific default)",
        )
        export_parser.add_argument(
            "--name",
            help="Model name for registry (default: derived from model_id)",
        )
        export_parser.add_argument(
            "--description",
            default="",
            help="Model description for registry",
        )
        export_parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("./registry"),
            help="Output directory for registry files",
        )
        export_parser.add_argument(
            "--variants",
            type=str,
            nargs="*",
            default=[],
            help="Variant types to create: f16 (FP16), i8 (int8 dynamic), i8-st (int8 static), "
                 "i4 (int4 CPU, generators only), i4-cuda (int4 CUDA, generators only). "
                 "Example: --variants f16 i8",
        )
        export_parser.add_argument(
            "--capabilities",
            type=str,
            nargs="*",
            default=[],
            help=(
                "Model capabilities. For embedders: multimodal. "
                "For recognizers: labels, zeroshot, relations, answers. "
                "Capabilities are auto-detected for recognizers but can be overridden. "
                "Example: --capabilities multimodal or --capabilities labels zeroshot"
            ),
        )
        export_parser.add_argument(
            "--backends",
            type=str,
            nargs="*",
            default=[],
            help="Required backends: onnx, xla, go. Example: --backends onnx",
        )
        export_parser.add_argument(
            "--no-test",
            action="store_true",
            help="Skip model testing",
        )
        export_parser.add_argument(
            "--upload",
            action="store_true",
            help="Upload to R2 after export",
        )
        export_parser.add_argument(
            "--r2-bucket",
            default="antfly-model-registry",
            help="R2 bucket name",
        )
        export_parser.add_argument(
            "--r2-endpoint",
            help="S3-compatible endpoint URL (or set AWS_ENDPOINT_URL env var)",
        )
        # Store the model type for later
        export_parser.set_defaults(model_type=model_type)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "gc":
        cmd_gc(args)
    else:
        # Export command (embedder, reranker, chunker, recognizer, rewriter, generator)
        cmd_export(args)


if __name__ == "__main__":
    main()
