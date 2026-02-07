"""Model exporter registry and factory."""

from pathlib import Path
from typing import Type

from .base import BaseExporter

# Registry of exporters by (model_type, capability) tuple
_EXPORTER_REGISTRY: dict[tuple[str, str | None], Type[BaseExporter]] = {}


def register_exporter(model_type: str, capability: str | None = None):
    """Decorator to register an exporter class.

    Args:
        model_type: The model type (e.g., "embedder", "reranker")
        capability: Optional capability (e.g., "audio", "image")
    """
    def decorator(cls: Type[BaseExporter]) -> Type[BaseExporter]:
        cls.model_type = model_type
        if capability:
            cls.capabilities = [capability]
        _EXPORTER_REGISTRY[(model_type, capability)] = cls
        return cls
    return decorator


def get_exporter(
    model_type: str,
    model_id: str,
    output_dir: Path,
    variants: list[str] | None = None,
    capabilities: list[str] | None = None,
    **kwargs,
) -> BaseExporter:
    """Get the appropriate exporter for a model type and capabilities.

    Args:
        model_type: The model type (e.g., "embedder", "reranker")
        model_id: HuggingFace model ID
        output_dir: Directory to save the exported model
        variants: List of variant types to create
        capabilities: List of capabilities (e.g., ["audio", "image"])
        **kwargs: Additional arguments passed to exporter constructor

    Returns:
        An instance of the appropriate exporter
    """
    capabilities = capabilities or []

    # Try combined capabilities first (e.g., "image,audio" for CLIPCLAP)
    if len(capabilities) > 1:
        combined = ",".join(sorted(capabilities))
        key = (model_type, combined)
        if key in _EXPORTER_REGISTRY:
            return _EXPORTER_REGISTRY[key](model_id, output_dir, variants, **kwargs)

    # Try individual capability-specific exporters
    for cap in capabilities:
        key = (model_type, cap)
        if key in _EXPORTER_REGISTRY:
            return _EXPORTER_REGISTRY[key](model_id, output_dir, variants, **kwargs)

    # Fall back to model-type-only exporter
    key = (model_type, None)
    if key in _EXPORTER_REGISTRY:
        return _EXPORTER_REGISTRY[key](model_id, output_dir, variants, **kwargs)

    raise ValueError(f"No exporter registered for model_type={model_type}, capabilities={capabilities}")


def list_exporters() -> list[tuple[str, str | None, Type[BaseExporter]]]:
    """List all registered exporters.

    Returns:
        List of (model_type, capability, exporter_class) tuples
    """
    return [(k[0], k[1], v) for k, v in _EXPORTER_REGISTRY.items()]


# Import all exporters to trigger registration
from . import embedder
from . import clip
from . import clap
from . import clipclap
from . import classifier
from . import reader
from . import seq2seq
from . import generator
from . import gliner
from . import gliner2
from . import rebel
