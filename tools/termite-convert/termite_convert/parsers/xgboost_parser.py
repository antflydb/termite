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

"""XGBoost model parser.

Supports:
- JSON model files (xgb.save_model("model.json"))
- Booster objects (in-memory)

XGBoost JSON format stores trees in a nested structure. Each tree node has:
- split_feature_id, split_threshold, yes (left child), no (right child)
- leaf nodes have "leaf" value instead of "split"
- default_left indicates NaN handling direction
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np

from termite_convert.ir import (
    TabularModel, Metadata, Stage, TreeEnsemble, TreeNodes,
    TreeAnnotations, OutputCfg,
)


def parse_xgboost(
    source: Union[str, Path, dict],
    name: str = "",
) -> TabularModel:
    """Parse an XGBoost model into Termite's tabular IR.

    Args:
        source: Path to XGBoost JSON model file, or a dict from json.load().
        name: Model name override. Defaults to filename stem.

    Returns:
        A TabularModel ready for serialization.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not name:
            name = path.stem
        with open(path) as f:
            model_data = json.load(f)
    else:
        model_data = source
        if not name:
            name = "xgboost-model"

    learner = model_data.get("learner", model_data)
    learner_params = learner.get("learner_model_param", {})
    gradient_booster = learner.get("gradient_booster", {})
    gbtree_params = gradient_booster.get("model", {})

    # Extract parameters
    num_features = int(learner_params.get("num_feature", 0))
    base_score = _parse_xgb_scalar(learner_params.get("base_score", 0.5))
    num_class = int(learner_params.get("num_class", 0))

    # Determine objective and task
    objective_data = learner.get("objective", {})
    objective_name = objective_data.get("name", "reg:squarederror")
    task, activation = _objective_to_task(objective_name, num_class)

    # Parse trees
    trees_data = gbtree_params.get("trees", [])
    if not trees_data:
        raise ValueError("No trees found in XGBoost model")

    # Flatten all trees into SoA layout
    feature_index = []
    threshold = []
    left_child = []
    right_child = []
    leaf_value = []
    default_left = []
    tree_starts = []
    max_depth = 0

    for tree_data in trees_data:
        tree_starts.append(len(feature_index))
        depth = _flatten_xgb_tree(
            tree_data, feature_index, threshold,
            left_child, right_child, leaf_value, default_left,
        )
        max_depth = max(max_depth, depth)

    num_trees = len(trees_data)

    # Feature names
    feature_names = []
    feature_map = learner.get("feature_names", [])
    if feature_map:
        feature_names = feature_map

    # Adjust num_outputs for multiclass
    num_outputs = max(num_class, 1)
    if task == "binary_classification":
        num_outputs = 1

    return TabularModel(
        schema_version=1,
        metadata=Metadata(
            name=name,
            source_framework="xgboost",
            source_version=learner_params.get("version", ""),
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
                    objective=objective_name,
                    base_score=base_score,
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


def _flatten_xgb_tree(
    tree_data: dict,
    feature_index: list[int],
    threshold: list[float],
    left_child: list[int],
    right_child: list[int],
    leaf_value: list[float],
    default_left: list[bool],
) -> int:
    """Flatten an XGBoost tree from nested format into parallel arrays.

    XGBoost JSON trees have nodes stored in arrays with indices:
    - tree_data["nodes"] is an array of node objects
    - Each node has: "nodeid", "split_feature_id"/"leaf", "yes", "no", "missing"

    Returns the tree depth.
    """
    base = len(feature_index)

    # XGBoost stores nodes in a flat array indexed by nodeid
    nodes = tree_data.get("nodes", [])
    if not nodes:
        # Try the legacy format where tree_data itself is the root node
        nodes = _collect_nodes_recursive(tree_data)

    num_nodes = len(nodes)
    # Pre-allocate
    for _ in range(num_nodes):
        feature_index.append(-1)
        threshold.append(0.0)
        left_child.append(-1)
        right_child.append(-1)
        leaf_value.append(0.0)
        default_left.append(False)

    max_depth = 0
    for node in nodes:
        nid = node.get("nodeid", 0)
        idx = base + nid

        if "leaf" in node:
            # Leaf node
            feature_index[idx] = -1
            leaf_value[idx] = float(node["leaf"])
            left_child[idx] = -1
            right_child[idx] = -1
        else:
            # Internal node
            feature_index[idx] = int(node.get("split_feature_id", node.get("split", 0)))
            threshold[idx] = float(node.get("split_condition", node.get("threshold", 0.0)))
            yes = int(node.get("yes", -1))
            no = int(node.get("no", -1))
            left_child[idx] = base + yes if yes >= 0 else -1
            right_child[idx] = base + no if no >= 0 else -1

            # default_left: "missing" points to "yes" (left)
            missing = node.get("missing", yes)
            default_left[idx] = (missing == yes)

            depth = node.get("depth", 0)
            max_depth = max(max_depth, depth)

    return max_depth


def _collect_nodes_recursive(node: dict, nodes: list | None = None) -> list:
    """Collect nodes from legacy nested XGBoost format into a flat list."""
    if nodes is None:
        nodes = []
    nodes.append(node)
    for child in node.get("children", []):
        _collect_nodes_recursive(child, nodes)
    return nodes


def _parse_xgb_scalar(value) -> float:
    """Parse an XGBoost scalar value which may be bracketed scientific notation.

    XGBoost 3.x stores base_score as e.g. '[4.82E-1]' instead of '0.482'.
    """
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip("[] \t\n")
    return float(s)


def _objective_to_task(objective: str, num_class: int) -> tuple[str, str]:
    """Map XGBoost objective to task type and activation."""
    if objective in ("binary:logistic", "binary:logitraw"):
        return "binary_classification", "sigmoid"
    if objective in ("binary:hinge",):
        return "binary_classification", "identity"
    if objective in ("multi:softmax", "multi:softprob"):
        return "multiclass", "softmax"
    if objective in ("rank:pairwise", "rank:ndcg", "rank:map"):
        return "ranking", "identity"
    if objective in ("count:poisson", "survival:cox"):
        return "regression", "exp"
    # Default: regression
    return "regression", "identity"
