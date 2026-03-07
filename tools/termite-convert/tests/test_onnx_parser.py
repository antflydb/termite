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

"""Tests for the ONNX ML opset parser."""

import pytest

try:
    import onnx
    from onnx import TensorProto, helper
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

pytestmark = pytest.mark.skipif(not HAS_ONNX, reason="onnx not installed")


def _make_tree_ensemble_classifier_onnx(tmp_path):
    """Create a minimal ONNX model with TreeEnsembleClassifier."""
    # Tree: split on feature 0 at 0.5
    #   left leaf -> class 0 (weight 1.0)
    #   right leaf -> class 1 (weight 1.0)
    tree_node = helper.make_node(
        "TreeEnsembleClassifier",
        inputs=["X"],
        outputs=["Y", "Z"],
        domain="ai.onnx.ml",
        nodes_treeids=[0, 0, 0],
        nodes_nodeids=[0, 1, 2],
        nodes_featureids=[0, 0, 0],
        nodes_values=[0.5, 0.0, 0.0],
        nodes_hitrates=[1.0, 1.0, 1.0],
        nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF"],
        nodes_truenodeids=[1, 0, 0],
        nodes_falsenodeids=[2, 0, 0],
        nodes_missing_value_tracks_true=[1, 0, 0],
        class_treeids=[0, 0],
        class_nodeids=[1, 2],
        class_ids=[0, 1],
        class_weights=[1.0, 1.0],
        classlabels_int64s=[0, 1],
        post_transform="LOGISTIC",
    )

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [None])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [None, 2])

    graph = helper.make_graph([tree_node], "test_tree", [X], [Y, Z])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 11),
        helper.make_opsetid("ai.onnx.ml", 1),
    ])

    model_path = tmp_path / "tree_classifier.onnx"
    onnx.save(model, str(model_path))
    return model_path


def _make_linear_regressor_onnx(tmp_path):
    """Create a minimal ONNX model with LinearRegressor."""
    linear_node = helper.make_node(
        "LinearRegressor",
        inputs=["X"],
        outputs=["Y"],
        domain="ai.onnx.ml",
        coefficients=[1.5, -0.5, 2.0],
        intercepts=[0.1],
        targets=1,
    )

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1])

    graph = helper.make_graph([linear_node], "test_linear", [X], [Y])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 11),
        helper.make_opsetid("ai.onnx.ml", 1),
    ])

    model_path = tmp_path / "linear_regressor.onnx"
    onnx.save(model, str(model_path))
    return model_path


def _make_svm_classifier_onnx(tmp_path):
    """Create a minimal ONNX model with SVMClassifier."""
    svm_node = helper.make_node(
        "SVMClassifier",
        inputs=["X"],
        outputs=["Y", "Z"],
        domain="ai.onnx.ml",
        kernel_type="LINEAR",
        support_vectors=[1.0, 0.0, 0.0, 1.0],
        coefficients=[0.5, -0.5],
        rho=[0.1],
        classlabels_int64s=[0, 1],
    )

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 2])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [None])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [None, 2])

    graph = helper.make_graph([svm_node], "test_svm", [X], [Y, Z])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 11),
        helper.make_opsetid("ai.onnx.ml", 1),
    ])

    model_path = tmp_path / "svm_classifier.onnx"
    onnx.save(model, str(model_path))
    return model_path


def test_parse_tree_ensemble_classifier(tmp_path):
    from termite_convert.parsers.onnx_parser import parse_onnx

    model_path = _make_tree_ensemble_classifier_onnx(tmp_path)
    model = parse_onnx(model_path, name="test-tree")

    assert model.metadata.name == "test-tree"
    assert model.metadata.source_framework == "onnx"
    assert model.metadata.task == "binary_classification"
    assert model.output.activation == "sigmoid"
    assert model.output.num_outputs == 1

    assert len(model.pipeline) == 1
    stage = model.pipeline[0]
    assert stage.type == "tree_ensemble"
    assert stage.tree_ensemble is not None
    assert stage.tree_ensemble.num_trees == 1
    assert len(stage.tree_ensemble.nodes.tree_starts) == 1


def test_parse_linear_regressor(tmp_path):
    from termite_convert.parsers.onnx_parser import parse_onnx

    model_path = _make_linear_regressor_onnx(tmp_path)
    model = parse_onnx(model_path, name="test-linear")

    assert model.metadata.name == "test-linear"
    assert model.metadata.task == "regression"
    assert model.output.activation == "identity"
    assert model.output.num_outputs == 1

    stage = model.pipeline[0]
    assert stage.type == "linear"
    assert stage.linear is not None
    assert len(stage.linear.weights) == 1
    assert len(stage.linear.weights[0]) == 3
    assert stage.linear.biases == [pytest.approx(0.1)]


def test_parse_svm_classifier(tmp_path):
    from termite_convert.parsers.onnx_parser import parse_onnx

    model_path = _make_svm_classifier_onnx(tmp_path)
    model = parse_onnx(model_path, name="test-svm")

    assert model.metadata.name == "test-svm"
    assert model.metadata.task == "binary_classification"

    stage = model.pipeline[0]
    assert stage.type == "svm"
    assert stage.svm is not None
    assert stage.svm.kernel == "linear"
    assert len(stage.svm.support_vectors) == 2
    assert len(stage.svm.dual_coefficients) == 1


def test_parse_onnx_no_ml_opset(tmp_path):
    """Parsing a model without ML opset should raise ValueError."""
    from termite_convert.parsers.onnx_parser import parse_onnx

    # Create a simple identity model with no ML ops
    identity = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 3])
    graph = helper.make_graph([identity], "no_ml", [X], [Y])
    model = helper.make_model(graph)

    model_path = tmp_path / "no_ml.onnx"
    onnx.save(model, str(model_path))

    with pytest.raises(ValueError, match="No ONNX ML opset operators found"):
        parse_onnx(model_path)
