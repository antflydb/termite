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

"""Python-side optimization passes for tabular models.

These run at conversion time to produce an optimized IR before saving.
The Go runtime also has load-time optimization passes (see lib/tabular/optimizer/).
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from termite_convert.ir import TabularModel, TreeAnnotations


def optimize_model(
    model: TabularModel,
    dead_leaf_threshold: float = 0.001,
    calibration_data: Optional[np.ndarray] = None,
) -> TabularModel:
    """Run all optimization passes on a model.

    Args:
        model: The model to optimize.
        dead_leaf_threshold: Fraction of max leaf value below which leaves are pruned.
        calibration_data: Optional [n_samples, n_features] array for branch sorting.

    Returns:
        An optimized copy of the model.
    """
    model = copy.deepcopy(model)

    for stage in model.pipeline:
        if stage.tree_ensemble is not None:
            eliminated = dead_leaf_elimination(stage.tree_ensemble, dead_leaf_threshold)
            annotate_threshold_precision(stage.tree_ensemble)
            if stage.tree_ensemble.annotations is None:
                stage.tree_ensemble.annotations = TreeAnnotations()
            stage.tree_ensemble.annotations.dead_leaves_eliminated = eliminated

            if calibration_data is not None:
                branch_sort(stage.tree_ensemble, calibration_data)
                stage.tree_ensemble.annotations.branch_order = "frequency_sorted"

    return model


def dead_leaf_elimination(te, threshold_fraction: float) -> int:
    """Prune leaves whose |value| < threshold_fraction * max|leaf_value|.

    When both children of an internal node become dead leaves, collapse the
    subtree into a single leaf with zero value.

    Returns the number of leaves eliminated.
    """
    nodes = te.nodes
    n = len(nodes.feature_index)
    if n == 0:
        return 0

    # Find max absolute leaf value
    max_val = 0.0
    for i in range(n):
        if nodes.feature_index[i] < 0:  # leaf
            v = abs(nodes.leaf_value[i])
            if v > max_val:
                max_val = v

    if max_val == 0:
        return 0

    abs_threshold = threshold_fraction * max_val
    eliminated = 0

    # Mark dead leaves
    is_dead = [False] * n
    for i in range(n):
        if nodes.feature_index[i] < 0 and abs(nodes.leaf_value[i]) < abs_threshold:
            is_dead[i] = True
            eliminated += 1

    # Bottom-up: collapse internal nodes where both children are dead
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if nodes.feature_index[i] >= 0 and not is_dead[i]:
                left = nodes.left_child[i]
                right = nodes.right_child[i]
                if left >= 0 and right >= 0 and is_dead[left] and is_dead[right]:
                    # Collapse to leaf with zero value
                    nodes.feature_index[i] = -1
                    nodes.left_child[i] = -1
                    nodes.right_child[i] = -1
                    nodes.leaf_value[i] = 0.0
                    is_dead[i] = True
                    eliminated += 1  # This node becomes a dead leaf too
                    changed = True

    return eliminated


def annotate_threshold_precision(te) -> None:
    """Annotate each feature's thresholds with minimum required precision.

    Checks if thresholds for a feature can be safely represented as:
    - "i8": all thresholds are integers in [-128, 127]
    - "i16": all thresholds are integers in [-32768, 32767]
    - "f16": all thresholds fit in float16 without loss
    - "f32": default
    """
    nodes = te.nodes
    n = len(nodes.feature_index)

    # Collect thresholds per feature
    feature_thresholds: dict[int, list[float]] = {}
    for i in range(n):
        fi = nodes.feature_index[i]
        if fi >= 0:
            feature_thresholds.setdefault(fi, []).append(nodes.threshold[i])

    precision = ["f32"] * te.num_features
    for fi, thresholds in feature_thresholds.items():
        if fi < 0 or fi >= te.num_features:
            continue
        precision[fi] = _min_precision(thresholds)

    if not hasattr(te, 'annotations') or te.annotations is None:
        te.annotations = TreeAnnotations()
    te.annotations.threshold_precision = precision


def _min_precision(values: list[float]) -> str:
    """Determine minimum precision needed for a set of threshold values."""
    all_int = all(v == int(v) for v in values)

    if all_int:
        min_v = min(values)
        max_v = max(values)
        if -128 <= min_v and max_v <= 127:
            return "i8"
        if -32768 <= min_v and max_v <= 32767:
            return "i16"

    # Check if float16 is sufficient (no precision loss)
    arr = np.array(values, dtype=np.float64)
    arr16 = arr.astype(np.float16).astype(np.float64)
    if np.allclose(arr, arr16, rtol=1e-3, atol=1e-6):
        return "f16"

    return "f32"


def branch_sort(te, calibration_data: np.ndarray) -> None:
    """Reorder tree children so the more likely branch is the fall-through (left) path.

    For each internal node, run calibration data through the tree and count
    how many samples go left vs right. If right is more frequent, swap the
    children so the common path is left (fall-through for branch prediction).

    Args:
        te: TreeEnsemble to modify in-place.
        calibration_data: [n_samples, n_features] numpy array.
    """
    nodes = te.nodes
    n = len(nodes.feature_index)
    n_samples = len(calibration_data)

    if n == 0 or n_samples == 0:
        return

    # Count left/right traversals per node
    left_count = [0] * n
    right_count = [0] * n

    for tree_idx in range(te.num_trees):
        start = nodes.tree_starts[tree_idx]
        # Determine tree end
        if tree_idx + 1 < te.num_trees:
            end = nodes.tree_starts[tree_idx + 1]
        else:
            end = n

        for sample in calibration_data:
            node_idx = start
            max_iter = te.max_depth + 2

            for _ in range(max_iter):
                if node_idx < start or node_idx >= end:
                    break

                fi = nodes.feature_index[node_idx]
                if fi < 0:  # leaf
                    break

                fv = float(sample[fi]) if fi < len(sample) else float('nan')

                if fv != fv:  # NaN
                    if nodes.default_left[node_idx]:
                        left_count[node_idx] += 1
                        node_idx = nodes.left_child[node_idx]
                    else:
                        right_count[node_idx] += 1
                        node_idx = nodes.right_child[node_idx]
                elif fv <= nodes.threshold[node_idx]:
                    left_count[node_idx] += 1
                    node_idx = nodes.left_child[node_idx]
                else:
                    right_count[node_idx] += 1
                    node_idx = nodes.right_child[node_idx]

    # Swap children where right is more frequent
    swaps = 0
    for i in range(n):
        fi = nodes.feature_index[i]
        if fi < 0:  # leaf
            continue

        if right_count[i] > left_count[i]:
            # Swap children
            nodes.left_child[i], nodes.right_child[i] = nodes.right_child[i], nodes.left_child[i]
            # Flip threshold comparison: swap means we negate the condition.
            # For the "go left if <= threshold" semantic, swapping children is
            # equivalent to "go left if > threshold". We achieve this by
            # flipping default_left and keeping threshold as-is -- the Go engine
            # checks fv <= threshold ? left : right, so after swap the frequent
            # path (formerly right) is now left.
            nodes.default_left[i] = not nodes.default_left[i]
            swaps += 1
