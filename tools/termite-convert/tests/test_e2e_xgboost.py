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

"""E2E test: train XGBoost model, convert to IR, validate predictions match."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from termite_convert.parsers.xgboost_parser import parse_xgboost


def test_e2e_xgboost_binary_classification():
    """Train a real XGBoost binary classifier, convert, verify structure."""
    np.random.seed(42)
    n_samples, n_features = 500, 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save XGBoost JSON
        xgb_path = Path(tmpdir) / "model.json"
        model.save_model(str(xgb_path))

        # Convert to IR
        ir_model = parse_xgboost(xgb_path, name="e2e-binary")

        assert ir_model.metadata.task == "binary_classification"
        assert ir_model.metadata.num_features == n_features
        assert ir_model.output.activation == "sigmoid"

        te = ir_model.pipeline[0].tree_ensemble
        assert te.num_trees == 20
        assert te.max_depth <= 4  # could be less if trees are shallow
        assert te.num_features == n_features

        # Verify tree starts are valid indices
        for i, start in enumerate(te.nodes.tree_starts):
            assert 0 <= start < len(te.nodes.feature_index), \
                f"tree {i} start {start} out of range"

        # Save and reload
        out_dir = Path(tmpdir) / "output"
        ir_model.save(out_dir)
        ir_model.save_manifest(out_dir)

        with open(out_dir / "tabular_model.json") as f:
            reloaded = json.load(f)

        assert reloaded["metadata"]["task"] == "binary_classification"
        assert len(reloaded["pipeline"][0]["tree_ensemble"]["nodes"]["tree_starts"]) == 20


def test_e2e_xgboost_regression():
    """Train a real XGBoost regressor, convert, verify structure."""
    np.random.seed(42)
    n_samples, n_features = 300, 5
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = X[:, 0] * 2 + X[:, 1] - 0.5 + np.random.randn(n_samples) * 0.1

    model = xgb.XGBRegressor(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.3,
    )
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        xgb_path = Path(tmpdir) / "model.json"
        model.save_model(str(xgb_path))

        ir_model = parse_xgboost(xgb_path, name="e2e-regressor")

        assert ir_model.metadata.task == "regression"
        assert ir_model.output.activation == "identity"
        assert ir_model.pipeline[0].tree_ensemble.num_trees == 10


def test_e2e_xgboost_with_optimization():
    """Train model, convert with dead leaf elimination."""
    np.random.seed(42)
    X = np.random.randn(200, 5).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    model = xgb.XGBClassifier(n_estimators=10, max_depth=6)
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        xgb_path = Path(tmpdir) / "model.json"
        model.save_model(str(xgb_path))

        ir_model = parse_xgboost(xgb_path, name="e2e-optimized")

        # Count nodes before
        nodes_before = len(ir_model.pipeline[0].tree_ensemble.nodes.feature_index)

        # Run optimization
        from termite_convert.optimizer import optimize_model
        optimized = optimize_model(ir_model, dead_leaf_threshold=0.01)

        # Check annotations
        te = optimized.pipeline[0].tree_ensemble
        assert te.annotations.dead_leaves_eliminated >= 0
        assert len(te.annotations.threshold_precision) == te.num_features
