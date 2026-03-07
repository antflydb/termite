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

"""Tests for XGBoost parser."""

import json
import tempfile
from pathlib import Path

from termite_convert.parsers.xgboost_parser import parse_xgboost


def _make_xgb_json(num_features=3, objective="binary:logistic"):
    """Create a minimal XGBoost JSON model for testing."""
    return {
        "learner": {
            "learner_model_param": {
                "num_feature": str(num_features),
                "base_score": "0.5",
                "num_class": "0",
            },
            "objective": {"name": objective},
            "gradient_booster": {
                "model": {
                    "trees": [
                        {
                            "nodes": [
                                {
                                    "nodeid": 0,
                                    "split_feature_id": 0,
                                    "split_condition": 0.5,
                                    "yes": 1,
                                    "no": 2,
                                    "missing": 1,
                                },
                                {
                                    "nodeid": 1,
                                    "leaf": 0.3,
                                },
                                {
                                    "nodeid": 2,
                                    "leaf": -0.2,
                                },
                            ]
                        },
                        {
                            "nodes": [
                                {
                                    "nodeid": 0,
                                    "split_feature_id": 1,
                                    "split_condition": 1.0,
                                    "yes": 1,
                                    "no": 2,
                                    "missing": 2,
                                },
                                {
                                    "nodeid": 1,
                                    "leaf": 0.1,
                                },
                                {
                                    "nodeid": 2,
                                    "leaf": -0.1,
                                },
                            ]
                        },
                    ]
                }
            },
        }
    }


def test_parse_xgboost_from_dict():
    data = _make_xgb_json()
    model = parse_xgboost(data, name="test-model")

    assert model.schema_version == 1
    assert model.metadata.name == "test-model"
    assert model.metadata.source_framework == "xgboost"
    assert model.metadata.task == "binary_classification"
    assert model.metadata.num_features == 3
    assert model.output.activation == "sigmoid"
    assert model.output.num_outputs == 1

    # Should have one tree ensemble stage
    assert len(model.pipeline) == 1
    te = model.pipeline[0].tree_ensemble
    assert te is not None
    assert te.num_trees == 2
    assert te.base_score == 0.5

    # Check tree structure: 2 trees * 3 nodes each = 6 total nodes
    assert len(te.nodes.feature_index) == 6
    assert te.nodes.tree_starts == [0, 3]


def test_parse_xgboost_from_file():
    data = _make_xgb_json()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        model = parse_xgboost(f.name)

    assert model.metadata.source_framework == "xgboost"
    assert model.pipeline[0].tree_ensemble.num_trees == 2


def test_parse_xgboost_regression():
    data = _make_xgb_json(objective="reg:squarederror")
    model = parse_xgboost(data, name="regressor")

    assert model.metadata.task == "regression"
    assert model.output.activation == "identity"


def test_save_roundtrip():
    data = _make_xgb_json()
    model = parse_xgboost(data, name="roundtrip-test")

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(Path(tmpdir))
        model.save_manifest(Path(tmpdir))

        # Verify files exist
        assert (Path(tmpdir) / "tabular_model.json").exists()
        assert (Path(tmpdir) / "model_manifest.json").exists()

        # Verify JSON is valid
        with open(Path(tmpdir) / "tabular_model.json") as f:
            loaded = json.load(f)
        assert loaded["schema_version"] == 1
        assert loaded["metadata"]["name"] == "roundtrip-test"

        with open(Path(tmpdir) / "model_manifest.json") as f:
            manifest = json.load(f)
        assert manifest["model_type"] == "predictor"
        assert "tree_ensemble" in manifest["capabilities"]


def test_node_structure():
    """Verify the flat SoA layout matches what Go expects."""
    data = _make_xgb_json()
    model = parse_xgboost(data, name="structure-test")
    te = model.pipeline[0].tree_ensemble

    # Tree 0: root splits on feature 0 at 0.5, left=leaf(0.3), right=leaf(-0.2)
    assert te.nodes.feature_index[0] == 0   # root: split on feature 0
    assert te.nodes.threshold[0] == 0.5
    assert te.nodes.feature_index[1] == -1  # left leaf
    assert te.nodes.leaf_value[1] == 0.3
    assert te.nodes.feature_index[2] == -1  # right leaf
    assert te.nodes.leaf_value[2] == -0.2
    assert te.nodes.default_left[0] is True  # missing=yes(1) -> left

    # Tree 1 starts at index 3
    assert te.nodes.feature_index[3] == 1   # root: split on feature 1
    assert te.nodes.threshold[3] == 1.0
    assert te.nodes.default_left[3] is False  # missing=no(2) -> right
