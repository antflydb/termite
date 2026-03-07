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

"""ONNX ML opset model parser.

Supports:
- TreeEnsembleClassifier / TreeEnsembleRegressor (ML opset)
- LinearClassifier / LinearRegressor (ML opset)
- SVMClassifier / SVMRegressor (ML opset)

Requires: onnx (pip install onnx)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from termite_convert.ir import (
    TabularModel, Metadata, Stage, TreeEnsemble, TreeNodes,
    TreeAnnotations, LinearModel, SVMModel, OutputCfg, Scaler,
)


def parse_onnx(
    source: Union[str, Path],
    name: str = "",
) -> TabularModel:
    """Parse an ONNX model with ML opset operators into Termite's tabular IR.

    Args:
        source: Path to ONNX model file (.onnx).
        name: Model name override. Defaults to filename stem.

    Returns:
        A TabularModel ready for serialization.
    """
    import onnx

    path = Path(source)
    if not name:
        name = path.stem

    model = onnx.load(str(path))
    graph = model.graph

    # Find ML opset operators
    ml_nodes = []
    for node in graph.node:
        if node.domain == "ai.onnx.ml":
            ml_nodes.append(node)

    if not ml_nodes:
        raise ValueError(
            "No ONNX ML opset operators found. "
            "This parser only supports tree ensembles, linear models, "
            "and SVMs from the ONNX ML opset."
        )

    # Determine input features
    num_features = 0
    if graph.input:
        input_shape = graph.input[0].type.tensor_type.shape
        if input_shape.dim and len(input_shape.dim) >= 2:
            dim = input_shape.dim[-1]
            if dim.dim_value > 0:
                num_features = dim.dim_value

    pipeline = []
    task = "regression"
    activation = "identity"
    num_outputs = 1

    for node in ml_nodes:
        op = node.op_type

        if op in ("TreeEnsembleClassifier", "TreeEnsembleRegressor"):
            te, t, act, n_out = _parse_tree_ensemble(node, num_features)
            pipeline.append(Stage(type="tree_ensemble", tree_ensemble=te))
            task, activation, num_outputs = t, act, n_out

        elif op in ("LinearClassifier", "LinearRegressor"):
            lm, t, act, n_out = _parse_linear(node, num_features)
            pipeline.append(Stage(type="linear", linear=lm))
            task, activation, num_outputs = t, act, n_out

        elif op in ("SVMClassifier", "SVMRegressor"):
            svm, t, act, n_out = _parse_svm(node, num_features)
            pipeline.append(Stage(type="svm", svm=svm))
            task, activation, num_outputs = t, act, n_out

        elif op == "Scaler":
            scaler = _parse_scaler(node)
            pipeline.append(Stage(type="scaler", scaler=scaler))

    if not any(s.type in ("tree_ensemble", "linear", "svm") for s in pipeline):
        raise ValueError("No supported model operator found in ONNX graph")

    return TabularModel(
        schema_version=1,
        metadata=Metadata(
            name=name,
            source_framework="onnx",
            task=task,
            num_features=num_features,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        pipeline=pipeline,
        output=OutputCfg(
            activation=activation,
            num_outputs=num_outputs,
        ),
    )


def _get_attr(node, name, default=None):
    """Get an attribute from an ONNX node by name."""
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == 1:  # FLOAT
                return attr.f
            elif attr.type == 2:  # INT
                return attr.i
            elif attr.type == 3:  # STRING
                return attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
            elif attr.type == 6:  # FLOATS
                return list(attr.floats)
            elif attr.type == 7:  # INTS
                return list(attr.ints)
            elif attr.type == 8:  # STRINGS
                return [s.decode("utf-8") if isinstance(s, bytes) else s for s in attr.strings]
    return default


def _parse_tree_ensemble(node, num_features: int):
    """Parse TreeEnsembleClassifier or TreeEnsembleRegressor."""
    op = node.op_type
    is_classifier = op == "TreeEnsembleClassifier"

    # Get node arrays
    nodes_featureids = _get_attr(node, "nodes_featureids", [])
    nodes_values = _get_attr(node, "nodes_values", [])  # thresholds
    nodes_treeids = _get_attr(node, "nodes_treeids", [])
    nodes_nodeids = _get_attr(node, "nodes_nodeids", [])
    nodes_modes = _get_attr(node, "nodes_modes", [])
    nodes_truenodeids = _get_attr(node, "nodes_truenodeids", [])
    nodes_falsenodeids = _get_attr(node, "nodes_falsenodeids", [])
    nodes_missing_value_tracks_true = _get_attr(node, "nodes_missing_value_tracks_true", [])

    # Get leaf values
    if is_classifier:
        target_ids = _get_attr(node, "class_ids", [])
        target_nodeids = _get_attr(node, "class_nodeids", [])
        target_treeids = _get_attr(node, "class_treeids", [])
        target_weights = _get_attr(node, "class_weights", [])
    else:
        target_ids = _get_attr(node, "target_ids", [])
        target_nodeids = _get_attr(node, "target_nodeids", [])
        target_treeids = _get_attr(node, "target_treeids", [])
        target_weights = _get_attr(node, "target_weights", [])

    base_values = _get_attr(node, "base_values", [0.0])
    base_score = base_values[0] if base_values else 0.0

    # Build leaf value lookup: (tree_id, node_id) -> value
    leaf_values = {}
    for i in range(len(target_nodeids)):
        key = (target_treeids[i], target_nodeids[i])
        leaf_values[key] = target_weights[i]

    # Determine tree structure
    tree_ids = sorted(set(nodes_treeids))
    num_trees = len(tree_ids)

    # Group nodes by tree
    tree_nodes = {}
    for i in range(len(nodes_treeids)):
        tid = nodes_treeids[i]
        if tid not in tree_nodes:
            tree_nodes[tid] = []
        tree_nodes[tid].append(i)

    # Flatten into SoA layout
    feature_index = []
    threshold = []
    left_child = []
    right_child = []
    leaf_value = []
    default_left = []
    tree_starts = []
    max_depth = 0

    for tid in tree_ids:
        indices = tree_nodes[tid]
        base = len(feature_index)
        tree_starts.append(base)

        # Build node_id -> position mapping for this tree
        node_id_to_pos = {}
        for idx in indices:
            nid = nodes_nodeids[idx]
            node_id_to_pos[nid] = base + len(node_id_to_pos)

        n_nodes = len(indices)
        for _ in range(n_nodes):
            feature_index.append(-1)
            threshold.append(0.0)
            left_child.append(-1)
            right_child.append(-1)
            leaf_value.append(0.0)
            default_left.append(False)

        for idx in indices:
            nid = nodes_nodeids[idx]
            pos = node_id_to_pos[nid]
            mode = nodes_modes[idx] if idx < len(nodes_modes) else "LEAF"

            if isinstance(mode, bytes):
                mode = mode.decode("utf-8")

            if mode == "LEAF":
                feature_index[pos] = -1
                lv = leaf_values.get((tid, nid), 0.0)
                leaf_value[pos] = float(lv)
            else:
                feature_index[pos] = int(nodes_featureids[idx])
                threshold[pos] = float(nodes_values[idx])

                true_nid = nodes_truenodeids[idx]
                false_nid = nodes_falsenodeids[idx]
                left_child[pos] = node_id_to_pos.get(true_nid, -1)
                right_child[pos] = node_id_to_pos.get(false_nid, -1)

                if idx < len(nodes_missing_value_tracks_true):
                    default_left[pos] = bool(nodes_missing_value_tracks_true[idx])

        # Estimate depth
        depth = _estimate_depth(n_nodes)
        max_depth = max(max_depth, depth)

    # Infer num_features from tree splits if not known
    if num_features == 0 and feature_index:
        num_features = max(fi for fi in feature_index if fi >= 0) + 1

    # Determine task/activation
    if is_classifier:
        class_labels = _get_attr(node, "classlabels_int64s", [])
        if not class_labels:
            class_labels = _get_attr(node, "classlabels_strings", [])
        n_classes = len(class_labels) if class_labels else 2

        post_transform = _get_attr(node, "post_transform", "NONE")
        if n_classes == 2:
            task = "binary_classification"
            activation = "sigmoid" if post_transform in ("LOGISTIC", "NONE") else "identity"
            num_outputs = 1
        else:
            task = "multiclass"
            activation = "softmax" if post_transform in ("SOFTMAX", "NONE") else "identity"
            num_outputs = n_classes
    else:
        post_transform = _get_attr(node, "post_transform", "NONE")
        task = "regression"
        activation = "identity"
        num_outputs = 1
        if post_transform == "LOGISTIC":
            activation = "sigmoid"

    te = TreeEnsemble(
        objective=f"onnx:{op}",
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
    )

    return te, task, activation, num_outputs


def _parse_linear(node, num_features: int):
    """Parse LinearClassifier or LinearRegressor."""
    op = node.op_type
    is_classifier = op == "LinearClassifier"

    coefficients = _get_attr(node, "coefficients", [])
    intercepts = _get_attr(node, "intercepts", [])

    if is_classifier:
        class_labels = _get_attr(node, "classlabels_int64s", [])
        if not class_labels:
            class_labels = _get_attr(node, "classlabels_strings", [])
        n_classes = len(class_labels) if class_labels else 2
        post_transform = _get_attr(node, "post_transform", "NONE")

        if n_classes == 2 and len(intercepts) == 1:
            # Binary: single set of coefficients
            weights = [coefficients[:num_features]]
            biases = list(intercepts)
            task = "binary_classification"
            activation = "sigmoid" if post_transform in ("LOGISTIC", "NONE") else "identity"
            num_outputs = 1
        else:
            # Multiclass: coefficients are flattened [n_classes * n_features]
            weights = []
            for c in range(n_classes):
                start = c * num_features
                weights.append([float(v) for v in coefficients[start:start + num_features]])
            biases = [float(v) for v in intercepts]
            task = "multiclass"
            activation = "softmax" if post_transform in ("SOFTMAX", "NONE") else "identity"
            num_outputs = n_classes
    else:
        n_targets = _get_attr(node, "targets", 1)
        if n_targets == 1:
            weights = [[float(v) for v in coefficients[:num_features]]]
            biases = [float(intercepts[0])] if intercepts else [0.0]
        else:
            weights = []
            for t in range(n_targets):
                start = t * num_features
                weights.append([float(v) for v in coefficients[start:start + num_features]])
            biases = [float(v) for v in intercepts]

        post_transform = _get_attr(node, "post_transform", "NONE")
        task = "regression"
        activation = "identity"
        num_outputs = n_targets if n_targets > 1 else 1

    lm = LinearModel(
        weights=weights,
        biases=biases,
        activation="identity",
    )

    return lm, task, activation, num_outputs


def _parse_svm(node, num_features: int):
    """Parse SVMClassifier or SVMRegressor."""
    op = node.op_type
    is_classifier = op == "SVMClassifier"

    kernel = _get_attr(node, "kernel_type", "LINEAR")
    if isinstance(kernel, bytes):
        kernel = kernel.decode("utf-8")
    kernel = kernel.lower()

    sv = _get_attr(node, "support_vectors", [])
    coefficients = _get_attr(node, "coefficients", [])
    rho = _get_attr(node, "rho", [0.0])
    gamma = _get_attr(node, "gamma", 0.0)
    coef0 = _get_attr(node, "coef0", 0.0)
    degree = _get_attr(node, "degree", 3)

    # Reshape support vectors: flat list -> [n_sv][n_features]
    if num_features == 0 and sv:
        # Try to infer from coefficients length
        n_sv = len(coefficients) if coefficients else 1
        if n_sv > 0 and len(sv) % n_sv == 0:
            num_features = len(sv) // n_sv

    if num_features > 0 and sv:
        n_sv = len(sv) // num_features
        support_vectors = []
        for i in range(n_sv):
            support_vectors.append([float(v) for v in sv[i * num_features:(i + 1) * num_features]])
    else:
        support_vectors = []

    # Dual coefficients: for binary SVM, single row
    dual_coefs = [[float(v) for v in coefficients]]
    intercept = [-float(v) for v in rho]  # ONNX uses rho (negated intercept)

    if is_classifier:
        class_labels = _get_attr(node, "classlabels_int64s", [])
        if not class_labels:
            class_labels = _get_attr(node, "classlabels_strings", [])
        n_classes = len(class_labels) if class_labels else 2

        post_transform = _get_attr(node, "post_transform", "NONE")
        if n_classes == 2:
            task = "binary_classification"
            activation = "sigmoid" if post_transform in ("LOGISTIC",) else "identity"
            num_outputs = 1
        else:
            task = "multiclass"
            activation = "identity"
            num_outputs = n_classes
    else:
        task = "regression"
        activation = "identity"
        num_outputs = 1

    svm_model = SVMModel(
        kernel=kernel,
        support_vectors=support_vectors,
        dual_coefficients=dual_coefs,
        intercept=intercept,
        gamma=float(gamma),
        coef0=float(coef0),
        degree=int(degree),
    )

    return svm_model, task, activation, num_outputs


def _parse_scaler(node):
    """Parse ONNX ML Scaler operator."""
    offset = _get_attr(node, "offset", [])
    scale = _get_attr(node, "scale", [])

    return Scaler(
        method="standard",
        center=[float(v) for v in offset],
        scale=[float(v) for v in scale],
    )


def _estimate_depth(n_nodes: int) -> int:
    """Estimate tree depth from node count (complete binary tree approximation)."""
    if n_nodes <= 1:
        return 0
    depth = 0
    nodes = 1
    while nodes < n_nodes:
        depth += 1
        nodes = 2 * nodes + 1
    return depth
