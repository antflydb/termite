#!/usr/bin/env python3
"""Cross-validation test: ensures Go and Python converters produce identical IR.

Trains small XGBoost and LightGBM models, converts them with both tools,
and compares the resulting tabular_model.json field-by-field.

Usage:
    python3 tools/cross_validate_converters.py [--termite-binary ./termite]
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def train_xgboost_model(model_path: str):
    """Train a small XGBoost binary classifier and save as JSON."""
    import xgboost as xgb

    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

    dtrain = xgb.DMatrix(X, label=y, feature_names=[f"f{i}" for i in range(5)])
    params = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "eta": 0.3,
        "seed": 42,
    }
    bst = xgb.train(params, dtrain, num_boost_round=5)
    bst.save_model(model_path)
    print(f"  XGBoost model saved to {model_path}")
    return bst


def train_lightgbm_model_json(model_path: str):
    """Train a small LightGBM binary classifier and save as JSON dump."""
    import lightgbm as lgb

    np.random.seed(42)
    X = np.random.randn(100, 4).astype(np.float32)
    y = (X[:, 0] * X[:, 1] > 0).astype(np.float32)

    dtrain = lgb.Dataset(X, label=y, feature_name=[f"feat{i}" for i in range(4)])
    params = {
        "objective": "binary",
        "num_leaves": 8,
        "learning_rate": 0.1,
        "seed": 42,
        "verbose": -1,
    }
    bst = lgb.train(params, dtrain, num_boost_round=5)

    # Save as JSON (lgb.dump_model format)
    model_json = bst.dump_model()
    with open(model_path, "w") as f:
        json.dump(model_json, f)
    print(f"  LightGBM JSON model saved to {model_path}")
    return bst


def train_lightgbm_model_text(model_path: str):
    """Train a small LightGBM binary classifier and save as text."""
    import lightgbm as lgb

    np.random.seed(42)
    X = np.random.randn(100, 4).astype(np.float32)
    y = (X[:, 0] * X[:, 1] > 0).astype(np.float32)

    dtrain = lgb.Dataset(X, label=y, feature_name=[f"feat{i}" for i in range(4)])
    params = {
        "objective": "binary",
        "num_leaves": 8,
        "learning_rate": 0.1,
        "seed": 42,
        "verbose": -1,
    }
    bst = lgb.train(params, dtrain, num_boost_round=5)
    bst.save_model(model_path)
    print(f"  LightGBM text model saved to {model_path}")
    return bst


def convert_with_go(termite_binary: str, framework: str, model_path: str, output_dir: str):
    """Convert using the Go termite CLI."""
    cmd = [termite_binary, "convert", framework, model_path, "-o", output_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Go converter failed: {result.stderr}")
        return None
    ir_path = os.path.join(output_dir, "tabular_model.json")
    with open(ir_path) as f:
        return json.load(f)


def convert_with_python(framework: str, model_path: str, output_dir: str):
    """Convert using the Python termite-convert tool."""
    if framework == "xgboost":
        from termite_convert.parsers.xgboost_parser import parse_xgboost
        model = parse_xgboost(model_path)
    elif framework == "lightgbm":
        from termite_convert.parsers.lightgbm_parser import parse_lightgbm
        model = parse_lightgbm(model_path)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    ir_path = os.path.join(output_dir, "tabular_model.json")
    with open(ir_path, "w") as f:
        json.dump(model.to_dict(), f, indent=2)
    return model.to_dict()


def compare_ir(go_ir: dict, py_ir: dict, label: str) -> list[str]:
    """Compare two IR dicts field-by-field. Returns list of differences."""
    diffs = []

    # Compare metadata (skip created_at and source_version which will differ)
    for key in ["name", "source_framework", "task", "num_features", "num_classes"]:
        go_val = go_ir.get("metadata", {}).get(key)
        py_val = py_ir.get("metadata", {}).get(key)
        if go_val != py_val:
            diffs.append(f"[{label}] metadata.{key}: Go={go_val} vs Python={py_val}")

    # Compare feature names
    go_fn = go_ir.get("metadata", {}).get("feature_names", [])
    py_fn = py_ir.get("metadata", {}).get("feature_names", [])
    if go_fn != py_fn:
        diffs.append(f"[{label}] metadata.feature_names: Go={go_fn} vs Python={py_fn}")

    # Compare output config
    go_out = go_ir.get("output", {})
    py_out = py_ir.get("output", {})
    for key in ["activation", "num_outputs"]:
        if go_out.get(key) != py_out.get(key):
            diffs.append(f"[{label}] output.{key}: Go={go_out.get(key)} vs Python={py_out.get(key)}")

    # Compare pipeline structure
    go_pipeline = go_ir.get("pipeline", [])
    py_pipeline = py_ir.get("pipeline", [])
    if len(go_pipeline) != len(py_pipeline):
        diffs.append(f"[{label}] pipeline length: Go={len(go_pipeline)} vs Python={len(py_pipeline)}")
        return diffs

    for stage_idx, (go_stage, py_stage) in enumerate(zip(go_pipeline, py_pipeline)):
        if go_stage.get("type") != py_stage.get("type"):
            diffs.append(f"[{label}] pipeline[{stage_idx}].type: Go={go_stage.get('type')} vs Python={py_stage.get('type')}")
            continue

        if go_stage.get("type") == "tree_ensemble":
            go_te = go_stage.get("tree_ensemble", {})
            py_te = py_stage.get("tree_ensemble", {})

            for key in ["objective", "num_trees", "num_features", "max_depth"]:
                if go_te.get(key) != py_te.get(key):
                    diffs.append(f"[{label}] tree_ensemble.{key}: Go={go_te.get(key)} vs Python={py_te.get(key)}")

            # Compare base_score with tolerance
            go_bs = go_te.get("base_score", 0)
            py_bs = py_te.get("base_score", 0)
            if abs(go_bs - py_bs) > 1e-6:
                diffs.append(f"[{label}] tree_ensemble.base_score: Go={go_bs} vs Python={py_bs}")

            # Compare node arrays
            go_nodes = go_te.get("nodes", {})
            py_nodes = py_te.get("nodes", {})

            for arr_key in ["feature_index", "left_child", "right_child", "tree_starts", "default_left"]:
                go_arr = go_nodes.get(arr_key, [])
                py_arr = py_nodes.get(arr_key, [])
                if go_arr != py_arr:
                    # Show first difference
                    for i, (g, p) in enumerate(zip(go_arr, py_arr)):
                        if g != p:
                            diffs.append(
                                f"[{label}] nodes.{arr_key}[{i}]: Go={g} vs Python={p} "
                                f"(array lengths: Go={len(go_arr)} vs Python={len(py_arr)})"
                            )
                            break
                    else:
                        if len(go_arr) != len(py_arr):
                            diffs.append(f"[{label}] nodes.{arr_key} length: Go={len(go_arr)} vs Python={len(py_arr)}")

            # Compare float arrays with tolerance
            for arr_key in ["threshold", "leaf_value"]:
                go_arr = go_nodes.get(arr_key, [])
                py_arr = py_nodes.get(arr_key, [])
                if len(go_arr) != len(py_arr):
                    diffs.append(f"[{label}] nodes.{arr_key} length: Go={len(go_arr)} vs Python={len(py_arr)}")
                else:
                    max_diff = 0.0
                    max_idx = -1
                    for i, (g, p) in enumerate(zip(go_arr, py_arr)):
                        d = abs(g - p)
                        if d > max_diff:
                            max_diff = d
                            max_idx = i
                    if max_diff > 1e-10:
                        diffs.append(
                            f"[{label}] nodes.{arr_key}: max diff={max_diff:.2e} at index {max_idx} "
                            f"(Go={go_arr[max_idx]}, Python={py_arr[max_idx]})"
                        )

    return diffs


def find_termite_binary():
    """Find the termite binary — build if needed."""
    # Check if already built
    binary = os.path.join(os.path.dirname(__file__), "..", "termite")
    if os.path.exists(binary):
        return os.path.abspath(binary)

    # Try to build
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Building termite binary...")
    result = subprocess.run(
        ["go", "build", "-o", os.path.join(repo_root, "termite"), "./cmd/termite/"],
        capture_output=True, text=True, cwd=repo_root,
    )
    if result.returncode != 0:
        print(f"Failed to build termite: {result.stderr}")
        sys.exit(1)
    return os.path.join(repo_root, "termite")


def main():
    termite_binary = find_termite_binary()
    print(f"Using termite binary: {termite_binary}")

    all_diffs = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # === XGBoost ===
        print("\n--- XGBoost Binary Classifier ---")
        xgb_model_path = os.path.join(tmpdir, "xgb_model.json")
        train_xgboost_model(xgb_model_path)

        go_out = os.path.join(tmpdir, "xgb_go")
        py_out = os.path.join(tmpdir, "xgb_py")
        os.makedirs(go_out, exist_ok=True)
        os.makedirs(py_out, exist_ok=True)

        go_ir = convert_with_go(termite_binary, "xgboost", xgb_model_path, go_out)
        py_ir = convert_with_python("xgboost", xgb_model_path, py_out)

        if go_ir and py_ir:
            diffs = compare_ir(go_ir, py_ir, "xgboost")
            all_diffs.extend(diffs)
            if diffs:
                print(f"  DIFFERENCES FOUND ({len(diffs)}):")
                for d in diffs:
                    print(f"    {d}")
            else:
                print("  MATCH: Go and Python converters produce identical IR")

        # === LightGBM JSON ===
        print("\n--- LightGBM Binary Classifier (JSON) ---")
        lgb_json_path = os.path.join(tmpdir, "lgb_model.json")
        train_lightgbm_model_json(lgb_json_path)

        go_out = os.path.join(tmpdir, "lgb_json_go")
        py_out = os.path.join(tmpdir, "lgb_json_py")
        os.makedirs(go_out, exist_ok=True)
        os.makedirs(py_out, exist_ok=True)

        go_ir = convert_with_go(termite_binary, "lightgbm", lgb_json_path, go_out)
        py_ir = convert_with_python("lightgbm", lgb_json_path, py_out)

        if go_ir and py_ir:
            diffs = compare_ir(go_ir, py_ir, "lightgbm_json")
            all_diffs.extend(diffs)
            if diffs:
                print(f"  DIFFERENCES FOUND ({len(diffs)}):")
                for d in diffs:
                    print(f"    {d}")
            else:
                print("  MATCH: Go and Python converters produce identical IR")

        # === LightGBM Text ===
        print("\n--- LightGBM Binary Classifier (Text) ---")
        lgb_txt_path = os.path.join(tmpdir, "lgb_model.txt")
        train_lightgbm_model_text(lgb_txt_path)

        go_out = os.path.join(tmpdir, "lgb_txt_go")
        py_out = os.path.join(tmpdir, "lgb_txt_py")
        os.makedirs(go_out, exist_ok=True)
        os.makedirs(py_out, exist_ok=True)

        go_ir = convert_with_go(termite_binary, "lightgbm", lgb_txt_path, go_out)
        py_ir = convert_with_python("lightgbm", lgb_txt_path, py_out)

        if go_ir and py_ir:
            diffs = compare_ir(go_ir, py_ir, "lightgbm_text")
            all_diffs.extend(diffs)
            if diffs:
                print(f"  DIFFERENCES FOUND ({len(diffs)}):")
                for d in diffs:
                    print(f"    {d}")
            else:
                print("  MATCH: Go and Python converters produce identical IR")

    # Summary
    print("\n" + "=" * 60)
    if all_diffs:
        print(f"FAIL: {len(all_diffs)} differences found across all models")
        for d in all_diffs:
            print(f"  {d}")
        sys.exit(1)
    else:
        print("PASS: All converters produce identical IR output")
        sys.exit(0)


if __name__ == "__main__":
    main()
