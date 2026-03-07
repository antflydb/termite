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

"""CatBoost model parser.

Supports:
- CatBoost JSON model files (model.save_model("model.json", format="json"))
- CatBoostClassifier / CatBoostRegressor objects (in-memory)

CatBoost JSON format stores oblivious decision trees. Each tree has a fixed
structure where all nodes at the same depth split on the same feature with
the same threshold (oblivious trees). Leaf values are stored as a flat array
of size 2^depth per tree.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

from termite_convert.ir import (
    TabularModel, Metadata, Stage, TreeEnsemble, TreeNodes,
    OutputCfg,
)


def parse_catboost(
    source: Union[str, Path, Any],
    name: str = "",
) -> TabularModel:
    """Parse a CatBoost model into Termite's tabular IR.

    Args:
        source: Path to CatBoost JSON model file, or a CatBoost model object.
        name: Model name override.

    Returns:
        A TabularModel ready for serialization.
    """
    # Handle model objects
    if hasattr(source, "save_model"):
        return _parse_catboost_object(source, name)

    path = Path(source)
    if not name:
        name = path.stem

    with open(path) as f:
        data = json.load(f)

    return _parse_catboost_json(data, name)


def _parse_catboost_object(model: Any, name: str) -> TabularModel:
    """Parse a CatBoost model object by dumping to JSON."""
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        tmp_path = f.name
        model.save_model(tmp_path, format="json")
    try:
        with open(tmp_path) as fh:
            data = json.load(fh)
        return _parse_catboost_json(data, name or "catboost-model")
    finally:
        os.unlink(tmp_path)


def _parse_catboost_json(data: dict, name: str) -> TabularModel:
    """Parse CatBoost JSON format."""
    model_info = data.get("model_info", {})
    oblivious_trees = data.get("oblivious_trees", [])

    if not oblivious_trees:
        raise ValueError("No trees found in CatBoost model. "
                         "Ensure the model was saved with format='json'.")

    # Extract model parameters
    params = model_info.get("params", {})
    loss_function = params.get("loss_function", {})
    loss_type = loss_function.get("type", "RMSE") if isinstance(loss_function, dict) else str(loss_function)

    # Feature count from the first tree's splits
    num_features = _count_features(data, oblivious_trees)

    task, activation = _loss_to_task(loss_type)

    # Scale and bias
    scale_and_bias = data.get("scale_and_bias", [[1.0], [0.0]])
    if len(scale_and_bias) >= 2:
        scale = scale_and_bias[0][0] if isinstance(scale_and_bias[0], list) else scale_and_bias[0]
        bias = scale_and_bias[1][0] if isinstance(scale_and_bias[1], list) else scale_and_bias[1]
    else:
        scale = 1.0
        bias = 0.0

    # Convert oblivious trees to standard binary trees
    feature_index = []
    threshold = []
    left_child = []
    right_child = []
    leaf_value = []
    default_left = []
    tree_starts = []
    max_depth = 0

    for tree in oblivious_trees:
        tree_starts.append(len(feature_index))
        depth = _convert_oblivious_tree(
            tree, scale,
            feature_index, threshold,
            left_child, right_child,
            leaf_value, default_left,
        )
        max_depth = max(max_depth, depth)

    num_trees = len(oblivious_trees)

    # Determine num_classes
    num_classes = 0
    if task == "multiclass":
        # CatBoost stores class count in model_info
        class_params = model_info.get("class_params", {})
        num_classes = int(class_params.get("class_count", 0))

    num_outputs = max(num_classes, 1)
    if task == "binary_classification":
        num_outputs = 1

    return TabularModel(
        schema_version=1,
        metadata=Metadata(
            name=name,
            source_framework="catboost",
            source_version=model_info.get("catboost_version_info", ""),
            task=task,
            num_features=num_features,
            num_classes=num_classes if num_classes > 2 else 0,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        pipeline=[
            Stage(
                type="tree_ensemble",
                tree_ensemble=TreeEnsemble(
                    objective=loss_type,
                    base_score=bias,
                    num_trees=num_trees,
                    num_features=num_features,
                    max_depth=max_depth,
                    nodes=TreeNodes(
                        feature_index=feature_index,
                        threshold=threshold,
                        left_child=left_child,
                        right_child=right_child,
                        leaf_value=leaf_value,
                        default_left=default_left,
                        tree_starts=tree_starts,
                    ),
                ),
            ),
        ],
        output=OutputCfg(
            activation=activation,
            num_outputs=num_outputs,
        ),
    )


def _convert_oblivious_tree(
    tree: dict,
    scale: float,
    feature_index: list[int],
    threshold: list[float],
    left_child: list[int],
    right_child: list[int],
    leaf_value: list[float],
    default_left: list[bool],
) -> int:
    """Convert a CatBoost oblivious tree to standard binary tree format.

    Oblivious trees: all nodes at the same depth use the same split.
    splits[0] is the root split, splits[1] is depth-1, etc.
    leaf_values has 2^depth entries.
    """
    splits = tree.get("splits", [])
    leaf_values = tree.get("leaf_values", [])
    leaf_weights = tree.get("leaf_weights", [])

    depth = len(splits)
    if depth == 0:
        # Single leaf tree
        base = len(feature_index)
        lv = leaf_values[0] * scale if leaf_values else 0.0
        feature_index.append(-1)
        threshold.append(0.0)
        left_child.append(-1)
        right_child.append(-1)
        leaf_value.append(lv)
        default_left.append(False)
        return 0

    # Build a complete binary tree of depth `depth`
    # Total nodes = 2^(depth+1) - 1
    num_internal = (1 << depth) - 1
    num_leaves = 1 << depth
    total_nodes = num_internal + num_leaves
    base = len(feature_index)

    # Pre-allocate all nodes
    for _ in range(total_nodes):
        feature_index.append(-1)
        threshold.append(0.0)
        left_child.append(-1)
        right_child.append(-1)
        leaf_value.append(0.0)
        default_left.append(False)

    # Fill internal nodes level by level
    # In oblivious trees, splits are ordered from root (index 0) to deepest
    # But CatBoost stores splits in reverse: splits[0] is the deepest level
    for level in range(depth):
        split_idx = depth - 1 - level  # CatBoost reverses the order
        split = splits[split_idx]

        feat = int(split.get("float_feature_index", split.get("feature_index", 0)))
        thresh = float(split.get("border", split.get("threshold", 0.0)))

        # Nodes at this level: indices [2^level - 1, 2^(level+1) - 2]
        level_start = (1 << level) - 1
        level_end = (1 << (level + 1)) - 1

        for i in range(level_start, level_end):
            idx = base + i
            feature_index[idx] = feat
            threshold[idx] = thresh
            left_child[idx] = base + 2 * i + 1
            right_child[idx] = base + 2 * i + 2
            default_left[idx] = True  # CatBoost default: left for missing

    # Fill leaf nodes
    for i in range(num_leaves):
        idx = base + num_internal + i
        if i < len(leaf_values):
            leaf_value[idx] = leaf_values[i] * scale
        feature_index[idx] = -1
        left_child[idx] = -1
        right_child[idx] = -1

    return depth


def _count_features(data: dict, trees: list[dict]) -> int:
    """Determine feature count from model data."""
    # Try model_info first
    model_info = data.get("model_info", {})
    params = model_info.get("params", {})

    # Try from features_info
    features_info = data.get("features_info", {})
    float_features = features_info.get("float_features", [])
    if float_features:
        max_idx = max(f.get("flat_feature_index", 0) for f in float_features)
        return max_idx + 1

    # Infer from splits
    max_feat = 0
    for tree in trees:
        for split in tree.get("splits", []):
            fi = split.get("float_feature_index", split.get("feature_index", 0))
            max_feat = max(max_feat, fi)

    return max_feat + 1


def _loss_to_task(loss_type: str) -> tuple[str, str]:
    """Map CatBoost loss function to task type and activation."""
    loss = loss_type.upper()

    if loss in ("LOGLOSS", "CROSSENTROPY"):
        return "binary_classification", "sigmoid"
    if loss in ("MULTICLASS", "MULTICLASSONEHOT"):
        return "multiclass", "softmax"
    if loss in ("QUERYRMSE", "PAIRLOGIT", "PAIRLOGITPAIRWISE",
                "YETIARANKPAIRWISE", "QUERYSOFTMAX"):
        return "ranking", "identity"
    if loss in ("POISSON",):
        return "regression", "exp"
    # RMSE, MAE, Quantile, etc.
    return "regression", "identity"
