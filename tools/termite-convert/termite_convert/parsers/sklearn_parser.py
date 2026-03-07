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

"""sklearn model parser.

Supports:
- GradientBoostingClassifier / GradientBoostingRegressor
- RandomForestClassifier / RandomForestRegressor
- LinearRegression / LogisticRegression / Ridge / Lasso / SGDClassifier
- Pipelines with StandardScaler, MinMaxScaler, RobustScaler

Models must be loaded via joblib/pickle before passing to this parser.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np

from termite_convert.ir import (
    TabularModel, Metadata, Stage, TreeEnsemble, TreeNodes,
    LinearModel, Scaler, OutputCfg,
)


def parse_sklearn(
    model: Any,
    name: str = "sklearn-model",
    feature_names: list[str] | None = None,
) -> TabularModel:
    """Parse a sklearn model into Termite's tabular IR.

    Args:
        model: A fitted sklearn estimator or Pipeline.
        name: Model name.
        feature_names: Optional feature names.

    Returns:
        A TabularModel ready for serialization.
    """
    # Handle Pipeline
    pipeline_stages = []
    estimator = model

    if _is_pipeline(model):
        for step_name, step in model.steps[:-1]:
            stage = _parse_preprocessing_step(step)
            if stage is not None:
                pipeline_stages.append(stage)
        estimator = model.steps[-1][1]

    # Parse the estimator
    if _is_tree_ensemble(estimator):
        model_stage, task, activation, num_features, num_outputs = _parse_tree_ensemble(estimator)
    elif _is_linear_model(estimator):
        model_stage, task, activation, num_features, num_outputs = _parse_linear_model(estimator)
    else:
        raise ValueError(
            f"Unsupported sklearn estimator type: {type(estimator).__name__}. "
            "Supported: GradientBoosting*, RandomForest*, LinearRegression, "
            "LogisticRegression, Ridge, Lasso, SGDClassifier, Pipeline."
        )

    pipeline_stages.append(model_stage)

    return TabularModel(
        schema_version=1,
        metadata=Metadata(
            name=name,
            source_framework="sklearn",
            task=task,
            num_features=num_features,
            feature_names=feature_names or [],
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        pipeline=pipeline_stages,
        output=OutputCfg(
            activation=activation,
            num_outputs=num_outputs,
        ),
    )


def _is_pipeline(model: Any) -> bool:
    try:
        from sklearn.pipeline import Pipeline
        return isinstance(model, Pipeline)
    except ImportError:
        return hasattr(model, "steps")


def _is_tree_ensemble(model: Any) -> bool:
    class_name = type(model).__name__
    return class_name in (
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "RandomForestClassifier", "RandomForestRegressor",
        "ExtraTreesClassifier", "ExtraTreesRegressor",
        "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
    )


def _is_linear_model(model: Any) -> bool:
    class_name = type(model).__name__
    return class_name in (
        "LinearRegression", "LogisticRegression", "Ridge", "Lasso",
        "ElasticNet", "SGDClassifier", "SGDRegressor",
    )


def _parse_preprocessing_step(step: Any) -> Stage | None:
    """Convert a sklearn preprocessing step into an IR Stage."""
    class_name = type(step).__name__

    if class_name == "StandardScaler":
        mean = step.mean_.tolist() if hasattr(step, "mean_") and step.mean_ is not None else []
        scale = step.scale_.tolist() if hasattr(step, "scale_") and step.scale_ is not None else []
        return Stage(
            type="scaler",
            scaler=Scaler(method="standard", center=mean, scale=scale),
        )
    elif class_name == "MinMaxScaler":
        return Stage(
            type="scaler",
            scaler=Scaler(
                method="minmax",
                min=step.data_min_.tolist(),
                max=step.data_max_.tolist(),
                scale=step.scale_.tolist(),
                center=step.min_.tolist(),
            ),
        )
    elif class_name == "RobustScaler":
        center = step.center_.tolist() if hasattr(step, "center_") and step.center_ is not None else []
        scale = step.scale_.tolist() if hasattr(step, "scale_") and step.scale_ is not None else []
        return Stage(
            type="scaler",
            scaler=Scaler(method="robust", center=center, scale=scale),
        )
    elif class_name == "MaxAbsScaler":
        return Stage(
            type="scaler",
            scaler=Scaler(
                method="maxabs",
                scale=step.max_abs_.tolist(),
            ),
        )
    else:
        warnings.warn(f"Unsupported preprocessing step: {class_name}, skipping")
        return None


def _parse_tree_ensemble(
    estimator: Any,
) -> tuple[Stage, str, str, int, int]:
    """Parse a sklearn tree ensemble into a TreeEnsemble stage."""
    class_name = type(estimator).__name__
    is_classifier = "Classifier" in class_name
    is_gradient_boosting = "GradientBoosting" in class_name

    # Determine task
    if is_classifier:
        n_classes = estimator.n_classes_ if hasattr(estimator, "n_classes_") else 2
        if n_classes > 2:
            task = "multiclass"
            activation = "softmax"
        else:
            task = "binary_classification"
            activation = "sigmoid"
    else:
        task = "regression"
        activation = "identity"
        n_classes = 0

    # Collect decision trees
    if is_gradient_boosting:
        # GradientBoosting stores trees in estimators_ as a 2D array [n_iterations, n_classes]
        all_trees = []
        for iteration in estimator.estimators_:
            if hasattr(iteration, "__iter__"):
                for tree in iteration:
                    all_trees.append(tree)
            else:
                all_trees.append(iteration)
        learning_rate = estimator.learning_rate
    else:
        # RandomForest stores trees in estimators_ as a flat list
        all_trees = estimator.estimators_
        learning_rate = 1.0

    num_features = estimator.n_features_in_

    # Flatten trees into SoA layout
    feature_index = []
    threshold_list = []
    left_child_list = []
    right_child_list = []
    leaf_value_list = []
    default_left_list = []
    tree_starts_list = []
    max_depth = 0

    for tree_obj in all_trees:
        tree = tree_obj.tree_ if hasattr(tree_obj, "tree_") else tree_obj
        tree_starts_list.append(len(feature_index))
        depth = _flatten_sklearn_tree(
            tree, learning_rate,
            feature_index, threshold_list,
            left_child_list, right_child_list,
            leaf_value_list, default_left_list,
        )
        max_depth = max(max_depth, depth)

    num_trees = len(all_trees)

    # Base score for gradient boosting
    base_score = 0.0
    if is_gradient_boosting and hasattr(estimator, "init_"):
        init = estimator.init_
        if hasattr(init, "constant_"):
            base_score = float(np.mean(init.constant_))

    num_outputs = max(n_classes, 1) if is_classifier else 1

    stage = Stage(
        type="tree_ensemble",
        tree_ensemble=TreeEnsemble(
            objective=_sklearn_objective(class_name),
            base_score=base_score,
            num_trees=num_trees,
            num_features=num_features,
            max_depth=max_depth,
            nodes=TreeNodes(
                feature_index=feature_index,
                threshold=threshold_list,
                left_child=left_child_list,
                right_child=right_child_list,
                leaf_value=leaf_value_list,
                default_left=default_left_list,
                tree_starts=tree_starts_list,
            ),
        ),
    )

    return stage, task, activation, num_features, num_outputs


def _flatten_sklearn_tree(
    tree: Any,
    learning_rate: float,
    feature_index: list[int],
    threshold: list[float],
    left_child: list[int],
    right_child: list[int],
    leaf_value: list[float],
    default_left: list[bool],
) -> int:
    """Flatten a sklearn DecisionTree into parallel arrays."""
    base = len(feature_index)
    n_nodes = tree.node_count

    for i in range(n_nodes):
        feat = int(tree.feature[i])
        if feat < 0:
            # Leaf node (TREE_UNDEFINED = -2 in sklearn)
            feature_index.append(-1)
            threshold.append(0.0)
            left_child.append(-1)
            right_child.append(-1)
            # For regression trees, value is [n_outputs, n_classes]
            val = tree.value[i].flatten()
            # For classification, normalize by sample count; for regression, use raw value
            if len(val) == 1:
                leaf_value.append(float(val[0]) * learning_rate)
            else:
                # For gradient boosting, the value is already the raw leaf weight
                leaf_value.append(float(val[0]) * learning_rate)
            default_left.append(True)  # sklearn sends missing left
        else:
            feature_index.append(feat)
            threshold.append(float(tree.threshold[i]))
            left_child.append(base + int(tree.children_left[i]))
            right_child.append(base + int(tree.children_right[i]))
            leaf_value.append(0.0)
            default_left.append(True)

    return int(tree.max_depth)


def _parse_linear_model(
    estimator: Any,
) -> tuple[Stage, str, str, int, int]:
    """Parse a sklearn linear model."""
    class_name = type(estimator).__name__
    is_classifier = class_name in ("LogisticRegression", "SGDClassifier")

    coef = np.atleast_2d(estimator.coef_)
    intercept = np.atleast_1d(estimator.intercept_)

    num_features = coef.shape[1]
    num_outputs = coef.shape[0]

    if is_classifier:
        if num_outputs <= 2:
            task = "binary_classification"
            activation = "sigmoid"
            num_outputs = 1
            # For binary, sklearn stores coef as (1, n_features) or (2, n_features)
            coef = coef[:1]
            intercept = intercept[:1]
        else:
            task = "multiclass"
            activation = "softmax"
    else:
        task = "regression"
        activation = "identity"

    stage = Stage(
        type="linear",
        linear=LinearModel(
            weights=coef.tolist(),
            biases=intercept.tolist(),
            activation="identity",
        ),
    )

    return stage, task, activation, num_features, num_outputs


def _sklearn_objective(class_name: str) -> str:
    mapping = {
        "GradientBoostingClassifier": "binary_logistic",
        "GradientBoostingRegressor": "squared_error",
        "RandomForestClassifier": "binary_logistic",
        "RandomForestRegressor": "squared_error",
        "ExtraTreesClassifier": "binary_logistic",
        "ExtraTreesRegressor": "squared_error",
    }
    return mapping.get(class_name, "squared_error")
