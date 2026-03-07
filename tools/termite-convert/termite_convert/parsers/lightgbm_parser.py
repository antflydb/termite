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

"""LightGBM model parser.

Supports:
- Text model files (lgb.save_model("model.txt"))
- JSON model files (lgb.dump_model())

LightGBM text format uses a flat array per tree with:
- split_feature, threshold, decision_type, left_child, right_child
- leaf_value for leaf outputs
- Leaf indices are negative (leaf_index = -(node_index + 1))
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from termite_convert.ir import (
    TabularModel, Metadata, Stage, TreeEnsemble, TreeNodes,
    OutputCfg,
)


def parse_lightgbm(
    source: Union[str, Path],
    name: str = "",
) -> TabularModel:
    """Parse a LightGBM model into Termite's tabular IR.

    Args:
        source: Path to a LightGBM model file (text or JSON format).
        name: Model name override.

    Returns:
        A TabularModel ready for serialization.
    """
    path = Path(source)
    if not name:
        name = path.stem

    content = path.read_text()

    # Detect format
    if content.lstrip().startswith("{"):
        return _parse_json(json.loads(content), name)
    else:
        return _parse_text(content, name)


def _parse_json(data: dict, name: str) -> TabularModel:
    """Parse LightGBM JSON dump format."""
    num_features = data.get("max_feature_idx", 0) + 1
    num_class = data.get("num_class", 1)
    objective = data.get("objective", "regression")
    feature_names = data.get("feature_names", [])
    trees_data = data.get("tree_info", [])

    if not trees_data:
        raise ValueError("No trees found in LightGBM model")

    task, activation = _objective_to_task(objective, num_class)

    feature_index = []
    threshold = []
    left_child = []
    right_child = []
    leaf_value = []
    default_left = []
    tree_starts = []
    max_depth = 0

    for tree_info in trees_data:
        tree = tree_info.get("tree_structure", tree_info)
        tree_starts.append(len(feature_index))
        nodes = []
        _collect_lgb_nodes(tree, nodes)

        base = len(feature_index)
        # Create node ID mapping
        node_map = {}
        for i, node in enumerate(nodes):
            node_map[id(node)] = i

        for node in nodes:
            idx = base + node_map[id(node)]
            if "leaf_value" in node:
                feature_index.append(-1)
                threshold.append(0.0)
                left_child.append(-1)
                right_child.append(-1)
                leaf_value.append(float(node["leaf_value"]))
                default_left.append(False)
            else:
                feature_index.append(int(node.get("split_feature", 0)))
                threshold.append(float(node.get("threshold", 0.0)))

                left_node = node.get("left_child", {})
                right_node = node.get("right_child", {})
                left_idx = base + node_map[id(left_node)] if id(left_node) in node_map else -1
                right_idx = base + node_map[id(right_node)] if id(right_node) in node_map else -1

                left_child.append(left_idx)
                right_child.append(right_idx)
                leaf_value.append(0.0)

                dl = node.get("default_left", True)
                default_left.append(bool(dl))

                depth = node.get("depth", 0)
                if depth > max_depth:
                    max_depth = depth

    num_trees = len(trees_data)
    num_outputs = max(num_class, 1)
    if task == "binary_classification":
        num_outputs = 1

    return TabularModel(
        schema_version=1,
        metadata=Metadata(
            name=name,
            source_framework="lightgbm",
            task=task,
            num_features=num_features,
            num_classes=num_class if num_class > 2 else 0,
            feature_names=feature_names,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        pipeline=[
            Stage(
                type="tree_ensemble",
                tree_ensemble=TreeEnsemble(
                    objective=objective,
                    base_score=0.0,
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


def _collect_lgb_nodes(node: dict, nodes: list):
    """Recursively collect LightGBM tree nodes in BFS order."""
    nodes.append(node)
    if "left_child" in node:
        _collect_lgb_nodes(node["left_child"], nodes)
    if "right_child" in node:
        _collect_lgb_nodes(node["right_child"], nodes)


def _parse_text(content: str, name: str) -> TabularModel:
    """Parse LightGBM text model format.

    The text format has sections like:
    - 'max_feature_idx=N'
    - 'num_class=N'
    - 'objective=...'
    - 'Tree=N' sections with split arrays
    """
    lines = content.split("\n")

    # Parse header parameters
    params = {}
    for line in lines:
        if "=" in line and not line.startswith("Tree"):
            key, _, val = line.partition("=")
            params[key.strip()] = val.strip()

    num_features = int(params.get("max_feature_idx", "0")) + 1
    num_class = int(params.get("num_class", "1"))
    objective = params.get("objective", "regression")
    feature_names = params.get("feature_names", "").split()

    task, activation = _objective_to_task(objective, num_class)

    # Find tree sections
    feature_index = []
    threshold_list = []
    left_child = []
    right_child = []
    leaf_value_list = []
    default_left = []
    tree_starts = []
    max_depth = 0

    tree_sections = _split_tree_sections(lines)
    for section in tree_sections:
        tree_starts.append(len(feature_index))
        depth = _parse_text_tree(
            section, feature_index, threshold_list,
            left_child, right_child, leaf_value_list, default_left,
        )
        max_depth = max(max_depth, depth)

    num_trees = len(tree_sections)
    num_outputs = max(num_class, 1)
    if task == "binary_classification":
        num_outputs = 1

    return TabularModel(
        schema_version=1,
        metadata=Metadata(
            name=name,
            source_framework="lightgbm",
            task=task,
            num_features=num_features,
            num_classes=num_class if num_class > 2 else 0,
            feature_names=feature_names if feature_names != [""] else [],
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        pipeline=[
            Stage(
                type="tree_ensemble",
                tree_ensemble=TreeEnsemble(
                    objective=objective,
                    base_score=0.0,
                    num_trees=num_trees,
                    num_features=num_features,
                    max_depth=max_depth,
                    nodes=TreeNodes(
                        feature_index=feature_index,
                        threshold=threshold_list,
                        left_child=left_child,
                        right_child=right_child,
                        leaf_value=leaf_value_list,
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


def _split_tree_sections(lines: list[str]) -> list[list[str]]:
    """Split model text into per-tree sections."""
    sections = []
    current = []
    in_tree = False
    for line in lines:
        if re.match(r"^Tree=\d+$", line):
            if current and in_tree:
                sections.append(current)
            current = []
            in_tree = True
        elif line.startswith("end of trees"):
            if current and in_tree:
                sections.append(current)
            break
        elif in_tree:
            current.append(line)
    return sections


def _parse_text_tree(
    section: list[str],
    feature_index: list[int],
    threshold: list[float],
    left_child: list[int],
    right_child: list[int],
    leaf_value: list[float],
    default_left: list[bool],
) -> int:
    """Parse a single tree from LightGBM text format.

    Each tree section contains arrays like:
      split_feature=0 2 1 ...
      threshold=0.5 1.2 0.8 ...
      left_child=1 3 5 ...
      right_child=2 4 6 ...
      leaf_value=0.1 -0.2 ...

    Negative child indices mean leaf: leaf_index = ~child_index
    """
    arrays = {}
    for line in section:
        if "=" in line:
            key, _, val = line.partition("=")
            arrays[key.strip()] = val.strip()

    num_leaves = int(arrays.get("num_leaves", "0"))
    num_internal = num_leaves - 1  # binary tree property
    if num_internal <= 0:
        # Single-leaf tree
        lv = float(arrays.get("leaf_value", "0"))
        feature_index.append(-1)
        threshold.append(0.0)
        left_child.append(-1)
        right_child.append(-1)
        leaf_value.append(lv)
        default_left.append(False)
        return 0

    base = len(feature_index)
    total_nodes = num_internal + num_leaves

    # Parse internal node arrays
    splits = [int(x) for x in arrays.get("split_feature", "").split()]
    thresholds = [float(x) for x in arrays.get("threshold", "").split()]
    lefts = [int(x) for x in arrays.get("left_child", "").split()]
    rights = [int(x) for x in arrays.get("right_child", "").split()]
    defaults = arrays.get("decision_type", "").split()
    leaf_values = [float(x) for x in arrays.get("leaf_value", "").split()]

    # Internal nodes first (indices 0..num_internal-1)
    for i in range(num_internal):
        feature_index.append(splits[i] if i < len(splits) else -1)
        threshold.append(thresholds[i] if i < len(thresholds) else 0.0)

        # Map child indices: negative means leaf
        left_raw = lefts[i] if i < len(lefts) else -1
        right_raw = rights[i] if i < len(rights) else -1

        if left_raw < 0:
            left_child.append(base + num_internal + (~left_raw))
        else:
            left_child.append(base + left_raw)

        if right_raw < 0:
            right_child.append(base + num_internal + (~right_raw))
        else:
            right_child.append(base + right_raw)

        leaf_value.append(0.0)

        # Default left from decision_type bitmask
        dl = True
        if i < len(defaults):
            dt = int(defaults[i])
            dl = bool(dt & 2)  # bit 1 = default left
        default_left.append(dl)

    # Leaf nodes (indices num_internal..total_nodes-1)
    for i in range(num_leaves):
        feature_index.append(-1)
        threshold.append(0.0)
        left_child.append(-1)
        right_child.append(-1)
        leaf_value.append(leaf_values[i] if i < len(leaf_values) else 0.0)
        default_left.append(False)

    # Compute depth
    max_depth = int(arrays.get("max_depth", "0"))
    return max_depth


def _objective_to_task(objective: str, num_class: int) -> tuple[str, str]:
    """Map LightGBM objective to task type and activation."""
    obj = objective.lower().split()[0] if objective else "regression"

    if obj in ("binary", "cross_entropy", "cross_entropy_lambda"):
        return "binary_classification", "sigmoid"
    if obj in ("multiclass", "multiclassova", "softmax", "multiclass_ova"):
        return "multiclass", "softmax"
    if obj in ("lambdarank", "rank_xendcg", "rank_pairwise"):
        return "ranking", "identity"
    if obj in ("poisson", "gamma", "tweedie"):
        return "regression", "exp"
    return "regression", "identity"
