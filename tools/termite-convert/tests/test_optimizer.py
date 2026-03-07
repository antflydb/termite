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

"""Tests for optimizer passes."""

import numpy as np

from termite_convert.ir import (
    TabularModel, Metadata, Stage, TreeEnsemble, TreeNodes,
    TreeAnnotations, OutputCfg,
)
from termite_convert.optimizer import (
    dead_leaf_elimination, annotate_threshold_precision, optimize_model,
    branch_sort,
)


def _make_simple_tree():
    return TreeEnsemble(
        objective="binary_logistic",
        base_score=0.0,
        num_trees=1,
        num_features=2,
        max_depth=1,
        nodes=TreeNodes(
            feature_index=[0, -1, -1],
            threshold=[0.5, 0, 0],
            left_child=[1, -1, -1],
            right_child=[2, -1, -1],
            leaf_value=[0, 1.0, 0.00001],
            default_left=[True, False, False],
            tree_starts=[0],
        ),
    )


def test_dead_leaf_elimination():
    te = _make_simple_tree()
    eliminated = dead_leaf_elimination(te, 0.001)
    assert eliminated == 1  # leaf with value 0.00001 is dead


def test_dead_leaf_collapse():
    te = TreeEnsemble(
        objective="binary_logistic",
        base_score=0.0,
        num_trees=1,
        num_features=1,
        max_depth=2,
        nodes=TreeNodes(
            feature_index=[0, -1, 1, -1, -1],
            threshold=[0.5, 0, 0.3, 0, 0],
            left_child=[1, -1, 3, -1, -1],
            right_child=[2, -1, 4, -1, -1],
            leaf_value=[0, 1.0, 0, 0.000001, 0.000002],
            default_left=[True, False, True, False, False],
            tree_starts=[0],
        ),
    )
    eliminated = dead_leaf_elimination(te, 0.001)
    assert eliminated == 3  # two dead leaves + collapsed parent


def test_annotate_threshold_precision():
    te = _make_simple_tree()
    # Threshold is 0.5 (float, not int, not f16-safe without numpy)
    annotate_threshold_precision(te)
    assert len(te.annotations.threshold_precision) == 2


def test_optimize_model():
    model = TabularModel(
        schema_version=1,
        metadata=Metadata(
            name="test", source_framework="xgboost",
            task="binary_classification", num_features=2,
        ),
        pipeline=[Stage(type="tree_ensemble", tree_ensemble=_make_simple_tree())],
        output=OutputCfg(activation="sigmoid", num_outputs=1),
    )
    optimized = optimize_model(model, dead_leaf_threshold=0.001)
    te = optimized.pipeline[0].tree_ensemble
    assert te.annotations.dead_leaves_eliminated == 1


def test_branch_sort():
    """Branch sorting swaps children when right branch is more frequent."""
    # Tree: split on feature 0 at threshold 0.5
    # If most calibration data has feature 0 > 0.5, right branch is more frequent
    te = TreeEnsemble(
        objective="binary_logistic",
        base_score=0.0,
        num_trees=1,
        num_features=1,
        max_depth=1,
        nodes=TreeNodes(
            feature_index=[0, -1, -1],
            threshold=[0.5, 0, 0],
            left_child=[1, -1, -1],
            right_child=[2, -1, -1],
            leaf_value=[0, 0.3, 0.7],
            default_left=[True, False, False],
            tree_starts=[0],
        ),
    )

    # All samples have feature > 0.5, so right branch is always taken
    calibration = np.array([[0.9], [0.8], [0.7], [0.6]])

    original_left = te.nodes.left_child[0]
    original_right = te.nodes.right_child[0]

    branch_sort(te, calibration)

    # Children should be swapped (right was more frequent -> now left)
    assert te.nodes.left_child[0] == original_right
    assert te.nodes.right_child[0] == original_left


def test_branch_sort_no_swap():
    """Branch sorting does not swap when left is already more frequent."""
    te = TreeEnsemble(
        objective="binary_logistic",
        base_score=0.0,
        num_trees=1,
        num_features=1,
        max_depth=1,
        nodes=TreeNodes(
            feature_index=[0, -1, -1],
            threshold=[0.5, 0, 0],
            left_child=[1, -1, -1],
            right_child=[2, -1, -1],
            leaf_value=[0, 0.3, 0.7],
            default_left=[True, False, False],
            tree_starts=[0],
        ),
    )

    # All samples have feature <= 0.5, so left branch is always taken
    calibration = np.array([[0.1], [0.2], [0.3], [0.4]])

    original_left = te.nodes.left_child[0]
    original_right = te.nodes.right_child[0]

    branch_sort(te, calibration)

    # Children should NOT be swapped
    assert te.nodes.left_child[0] == original_left
    assert te.nodes.right_child[0] == original_right


def test_optimize_model_with_calibration():
    """optimize_model with calibration data enables branch sorting."""
    model = TabularModel(
        schema_version=1,
        metadata=Metadata(
            name="test", source_framework="xgboost",
            task="binary_classification", num_features=2,
        ),
        pipeline=[Stage(type="tree_ensemble", tree_ensemble=_make_simple_tree())],
        output=OutputCfg(activation="sigmoid", num_outputs=1),
    )
    calibration = np.array([[0.9, 0.1], [0.8, 0.2]])
    optimized = optimize_model(model, calibration_data=calibration)
    te = optimized.pipeline[0].tree_ensemble
    assert te.annotations.branch_order == "frequency_sorted"
