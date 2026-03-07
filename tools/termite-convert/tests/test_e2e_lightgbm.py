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

"""E2E test: train LightGBM model, convert to IR, validate."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from termite_convert.parsers.lightgbm_parser import parse_lightgbm


def test_e2e_lightgbm_binary_classification():
    """Train a real LightGBM binary classifier, convert via JSON dump."""
    np.random.seed(42)
    n_samples, n_features = 500, 8
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=15,
        max_depth=4,
        num_leaves=10,
        verbose=-1,
    )
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save as text format
        txt_path = Path(tmpdir) / "model.txt"
        model.booster_.save_model(str(txt_path))

        ir_model = parse_lightgbm(txt_path, name="e2e-lgb-binary")

        assert ir_model.metadata.task == "binary_classification"
        assert ir_model.metadata.num_features == n_features
        assert ir_model.output.activation == "sigmoid"

        te = ir_model.pipeline[0].tree_ensemble
        assert te.num_trees == 15
        assert te.num_features == n_features

        # Save and verify
        out_dir = Path(tmpdir) / "output"
        ir_model.save(out_dir)
        ir_model.save_manifest(out_dir)

        assert (out_dir / "tabular_model.json").exists()
        assert (out_dir / "model_manifest.json").exists()


def test_e2e_lightgbm_json_dump():
    """Test parsing from LightGBM JSON dump format."""
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = X[:, 0] * 2 + np.random.randn(200) * 0.1

    model = lgb.LGBMRegressor(n_estimators=10, max_depth=3, verbose=-1)
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Dump as JSON
        json_path = Path(tmpdir) / "model.json"
        dump = model.booster_.dump_model()
        with open(json_path, "w") as f:
            json.dump(dump, f)

        ir_model = parse_lightgbm(json_path, name="e2e-lgb-regressor")

        assert ir_model.metadata.task == "regression"
        assert ir_model.output.activation == "identity"
        assert ir_model.pipeline[0].tree_ensemble.num_trees == 10
