# Copyright 2025 Antfly, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IR types matching the Go tabular model format."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class TreeNodes:
    feature_index: list[int]
    threshold: list[float]
    left_child: list[int]
    right_child: list[int]
    leaf_value: list[float]
    default_left: list[bool]
    tree_starts: list[int]


@dataclass
class TreeAnnotations:
    threshold_precision: list[str] = field(default_factory=list)
    dead_leaves_eliminated: int = 0
    branch_order: str = ""


@dataclass
class TreeEnsemble:
    objective: str
    base_score: float
    num_trees: int
    num_features: int
    max_depth: int
    nodes: TreeNodes
    annotations: TreeAnnotations = field(default_factory=TreeAnnotations)


@dataclass
class LinearModel:
    weights: list[list[float]]
    biases: list[float]
    activation: str = "identity"


@dataclass
class Scaler:
    method: str
    center: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    min: list[float] = field(default_factory=list)
    max: list[float] = field(default_factory=list)


@dataclass
class Imputer:
    strategy: str
    fill_values: list[float] = field(default_factory=list)


@dataclass
class SVMModel:
    kernel: str  # rbf, linear, poly, sigmoid
    support_vectors: list[list[float]]
    dual_coefficients: list[list[float]]
    intercept: list[float]
    gamma: float = 0.0
    coef0: float = 0.0
    degree: int = 0


@dataclass
class Stage:
    type: str
    tree_ensemble: Optional[TreeEnsemble] = None
    linear: Optional[LinearModel] = None
    svm: Optional[SVMModel] = None
    scaler: Optional[Scaler] = None
    imputer: Optional[Imputer] = None


@dataclass
class Metadata:
    name: str
    source_framework: str
    task: str
    num_features: int
    source_version: str = ""
    num_classes: int = 0
    feature_names: list[str] = field(default_factory=list)
    feature_types: list[str] = field(default_factory=list)
    created_at: str = ""


@dataclass
class OutputCfg:
    activation: str
    num_outputs: int


@dataclass
class TabularModel:
    schema_version: int
    metadata: Metadata
    pipeline: list[Stage]
    output: OutputCfg

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict, omitting empty optional fields."""
        d = _clean_dict(asdict(self))
        return d

    def save(self, path: Path):
        """Write tabular_model.json to the given directory."""
        path.mkdir(parents=True, exist_ok=True)
        model_file = path / "tabular_model.json"
        with open(model_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_manifest(self, path: Path):
        """Write model_manifest.json for Termite discovery."""
        manifest = {
            "model_type": "predictor",
            "capabilities": _capabilities_for_model(self),
            "metadata": {
                "name": self.metadata.name,
                "source_framework": self.metadata.source_framework,
                "task": self.metadata.task,
            },
        }
        manifest_file = path / "model_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)


def _capabilities_for_model(m: TabularModel) -> list[str]:
    caps = ["tabular"]
    for stage in m.pipeline:
        if stage.type == "tree_ensemble":
            caps.append("tree_ensemble")
        elif stage.type == "linear":
            caps.append("linear_model")
        elif stage.type == "svm":
            caps.append("svm")
    return caps


def _clean_dict(d):
    """Recursively remove None values and empty optional fields."""
    if isinstance(d, dict):
        cleaned = {}
        for k, v in d.items():
            v = _clean_dict(v)
            # Skip None and empty strings/lists for optional fields
            if v is None:
                continue
            if isinstance(v, str) and v == "" and k in (
                "source_version", "created_at", "branch_order",
            ):
                continue
            if isinstance(v, list) and len(v) == 0 and k in (
                "feature_names", "feature_types", "threshold_precision",
                "center", "scale", "min", "max", "fill_values",
            ):
                continue
            if isinstance(v, int) and v == 0 and k in (
                "num_classes", "dead_leaves_eliminated",
            ):
                continue
            cleaned[k] = v
        return cleaned
    elif isinstance(d, list):
        return [_clean_dict(item) for item in d]
    return d
