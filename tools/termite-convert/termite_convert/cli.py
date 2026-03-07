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

"""CLI for converting ML models to Termite's tabular IR format."""

from __future__ import annotations

import sys
from pathlib import Path

import click


@click.group()
@click.version_option()
def main():
    """Convert traditional ML models to Termite's tabular IR format."""
    pass


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory")
@click.option("--name", default="", help="Model name override")
@click.option("--optimize", is_flag=True, help="Run optimization passes")
@click.option("--dead-leaf-threshold", type=float, default=0.001,
              help="Threshold for dead leaf elimination (fraction of max leaf value)")
@click.option("--calibration-data", default="", type=click.Path(),
              help="CSV file with calibration data for branch sorting")
def xgboost(model_path: str, output: str, name: str, optimize: bool, dead_leaf_threshold: float, calibration_data: str):
    """Convert an XGBoost JSON model."""
    from termite_convert.parsers.xgboost_parser import parse_xgboost

    model = parse_xgboost(model_path, name=name)

    if optimize:
        model = _run_optimizations(model, dead_leaf_threshold, calibration_data)

    _save_model(model, output)


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory")
@click.option("--name", default="", help="Model name override")
@click.option("--optimize", is_flag=True, help="Run optimization passes")
@click.option("--dead-leaf-threshold", type=float, default=0.001,
              help="Threshold for dead leaf elimination (fraction of max leaf value)")
@click.option("--calibration-data", default="", type=click.Path(),
              help="CSV file with calibration data for branch sorting")
def lightgbm(model_path: str, output: str, name: str, optimize: bool, dead_leaf_threshold: float, calibration_data: str):
    """Convert a LightGBM model (text or JSON format)."""
    from termite_convert.parsers.lightgbm_parser import parse_lightgbm

    model = parse_lightgbm(model_path, name=name)

    if optimize:
        model = _run_optimizations(model, dead_leaf_threshold, calibration_data)

    _save_model(model, output)


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory")
@click.option("--name", default="", help="Model name override")
@click.option("--optimize", is_flag=True, help="Run optimization passes")
@click.option("--dead-leaf-threshold", type=float, default=0.001,
              help="Threshold for dead leaf elimination (fraction of max leaf value)")
@click.option("--calibration-data", default="", type=click.Path(),
              help="CSV file with calibration data for branch sorting")
def sklearn(model_path: str, output: str, name: str, optimize: bool, dead_leaf_threshold: float, calibration_data: str):
    """Convert a sklearn model (joblib/pickle format).

    Supports: GradientBoosting*, RandomForest*, LinearRegression,
    LogisticRegression, Ridge, Lasso, Pipeline with scalers.
    """
    try:
        import joblib
    except ImportError:
        click.echo("Error: joblib is required for sklearn models. Install with: pip install joblib", err=True)
        sys.exit(1)

    from termite_convert.parsers.sklearn_parser import parse_sklearn

    estimator = joblib.load(model_path)
    model = parse_sklearn(estimator, name=name or Path(model_path).stem)

    if optimize:
        model = _run_optimizations(model, dead_leaf_threshold, calibration_data)

    _save_model(model, output)


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory")
@click.option("--name", default="", help="Model name override")
@click.option("--optimize", is_flag=True, help="Run optimization passes")
@click.option("--dead-leaf-threshold", type=float, default=0.001,
              help="Threshold for dead leaf elimination (fraction of max leaf value)")
@click.option("--calibration-data", default="", type=click.Path(),
              help="CSV file with calibration data for branch sorting")
def catboost(model_path: str, output: str, name: str, optimize: bool, dead_leaf_threshold: float, calibration_data: str):
    """Convert a CatBoost model (JSON format).

    Export from CatBoost with: model.save_model("model.json", format="json")
    """
    from termite_convert.parsers.catboost_parser import parse_catboost

    model = parse_catboost(model_path, name=name)

    if optimize:
        model = _run_optimizations(model, dead_leaf_threshold, calibration_data)

    _save_model(model, output)


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory")
@click.option("--name", default="", help="Model name override")
@click.option("--optimize", is_flag=True, help="Run optimization passes")
@click.option("--dead-leaf-threshold", type=float, default=0.001,
              help="Threshold for dead leaf elimination (fraction of max leaf value)")
@click.option("--calibration-data", default="", type=click.Path(),
              help="CSV file with calibration data for branch sorting")
def onnx(model_path: str, output: str, name: str, optimize: bool, dead_leaf_threshold: float, calibration_data: str):
    """Convert an ONNX model with ML opset operators.

    Supports: TreeEnsembleClassifier/Regressor, LinearClassifier/Regressor,
    SVMClassifier/Regressor from the ONNX ML opset.
    """
    try:
        import onnx  # noqa: F401
    except ImportError:
        click.echo("Error: onnx is required for ONNX models. Install with: pip install onnx", err=True)
        sys.exit(1)

    from termite_convert.parsers.onnx_parser import parse_onnx

    model = parse_onnx(model_path, name=name)

    if optimize:
        model = _run_optimizations(model, dead_leaf_threshold, calibration_data)

    _save_model(model, output)


def _run_optimizations(model, dead_leaf_threshold: float, calibration_data_path: str = ""):
    """Run optimization passes on the model."""
    import numpy as np
    from termite_convert.optimizer import optimize_model

    calibration_data = None
    if calibration_data_path:
        calibration_data = np.loadtxt(calibration_data_path, delimiter=",", skiprows=1)
        click.echo(f"  Loaded calibration data: {calibration_data.shape[0]} samples")

    return optimize_model(
        model,
        dead_leaf_threshold=dead_leaf_threshold,
        calibration_data=calibration_data,
    )


def _save_model(model, output: str):
    """Save model and manifest to the output directory."""
    out_path = Path(output)
    model.save(out_path)
    model.save_manifest(out_path)

    click.echo(f"Model saved to {out_path}/")
    click.echo(f"  tabular_model.json  ({model.metadata.source_framework}, "
               f"{model.metadata.task})")

    # Print stats
    for stage in model.pipeline:
        if stage.tree_ensemble is not None:
            te = stage.tree_ensemble
            total_nodes = len(te.nodes.feature_index)
            click.echo(f"  Trees: {te.num_trees}, Nodes: {total_nodes}, "
                       f"Max depth: {te.max_depth}, Features: {te.num_features}")
        elif stage.linear is not None:
            lm = stage.linear
            click.echo(f"  Linear: {len(lm.weights)}x{len(lm.weights[0])}")

    click.echo(f"  model_manifest.json (type: predictor)")


if __name__ == "__main__":
    main()
