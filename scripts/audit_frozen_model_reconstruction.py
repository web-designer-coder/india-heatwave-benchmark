"""Compare the canonical frozen onset bundle with a separate reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_differences(left: Any, right: Any, path: str = "state") -> list[str]:
    """Return paths whose fitted values differ, including array-level values."""
    if type(left) is not type(right):
        return [path + ":type"]
    if isinstance(left, np.ndarray):
        try:
            same = np.array_equal(left, right, equal_nan=True)
        except TypeError:
            same = np.array_equal(left, right)
        return [] if same else [path + ":array"]
    if isinstance(left, pd.DataFrame):
        return [] if left.equals(right) else [path + ":dataframe"]
    if isinstance(left, dict):
        paths = []
        if set(left) != set(right):
            paths.append(path + ":keys")
        for key in sorted(set(left) & set(right), key=repr):
            paths.extend(state_differences(left[key], right[key], f"{path}.{key}"))
        return paths
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return [path + ":length"]
        return [diff for i, (a, b) in enumerate(zip(left, right))
                for diff in state_differences(a, b, f"{path}[{i}]")]
    if hasattr(left, "__dict__"):
        return state_differences(vars(left), vars(right), path)
    try:
        same = bool(left == right)
    except (TypeError, ValueError):
        same = repr(left) == repr(right)
    if isinstance(left, float) and np.isnan(left) and np.isnan(right):
        same = True
    return [] if same else [path + ":value"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("reconstruction", type=Path)
    parser.add_argument("--original-predictions", type=Path)
    parser.add_argument("--reconstructed-predictions", type=Path)
    parser.add_argument("--original-2026-predictions", type=Path)
    parser.add_argument("--reconstructed-2026-predictions", type=Path)
    parser.add_argument("--original-2026-grid-scores", type=Path)
    parser.add_argument("--reconstructed-2026-grid-scores", type=Path)
    parser.add_argument("--original-2026-metrics", type=Path)
    parser.add_argument("--reconstructed-2026-metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    original = joblib.load(args.original)
    rebuilt = joblib.load(args.reconstruction)
    keys = sorted(set(original) | set(rebuilt))
    original_estimator = original["estimator"]
    rebuilt_estimator = rebuilt["estimator"]
    original_calibrator = original["calibrator"]
    rebuilt_calibrator = rebuilt["calibrator"]
    report: dict[str, Any] = {
        "original_file_sha256": sha256_file(args.original),
        "reconstruction_file_sha256": sha256_file(args.reconstruction),
        "bundle_keys_equal": set(original) == set(rebuilt),
        "bundle_keys_original_only": sorted(set(original) - set(rebuilt)),
        "bundle_keys_reconstruction_only": sorted(set(rebuilt) - set(original)),
        "feature_columns_equal": original["feature_columns"] == rebuilt["feature_columns"],
        "feature_count": len(original["feature_columns"]),
        "temperature_normals_equal": original["temperature_normals"].equals(
            rebuilt["temperature_normals"]
        ),
        "temperature_normals_shape": list(original["temperature_normals"].shape),
        "selected_parameters_equal": original["selected_parameters"]
        == rebuilt["selected_parameters"],
        "selected_parameters": original["selected_parameters"],
        "threshold_equal": original["probability_threshold"]
        == rebuilt["probability_threshold"],
        "threshold_original": original["probability_threshold"],
        "threshold_reconstruction": rebuilt["probability_threshold"],
        "training_years_equal": original["training_years"] == rebuilt["training_years"],
        "calibration_and_threshold_years_equal": original[
            "calibration_and_threshold_years"
        ] == rebuilt["calibration_and_threshold_years"],
        "normal_fit_years_equal": original["normal_fit_years"]
        == rebuilt["normal_fit_years"],
        "predictor_snapshot_equal": original["predictor_snapshot_id"]
        == rebuilt["predictor_snapshot_id"],
        "estimator_class_equal": type(original_estimator) is type(rebuilt_estimator),
        "estimator_config_equal": original_estimator.get_params(deep=False)
        == rebuilt_estimator.get_params(deep=False),
        "estimator_fitted_state_difference_paths": state_differences(
            original_estimator.__dict__, rebuilt_estimator.__dict__, "estimator"
        )[:100],
        "estimator_n_features_in": [
            int(original_estimator.n_features_in_), int(rebuilt_estimator.n_features_in_)
        ],
        "estimator_n_iter": [int(original_estimator.n_iter_), int(rebuilt_estimator.n_iter_)],
        "calibrator_class_equal": type(original_calibrator) is type(rebuilt_calibrator),
        "calibrator_config_equal": original_calibrator.get_params(deep=False)
        == rebuilt_calibrator.get_params(deep=False),
        "calibrator_coefficients_equal": np.array_equal(
            original_calibrator.coef_, rebuilt_calibrator.coef_
        )
        and np.array_equal(original_calibrator.intercept_, rebuilt_calibrator.intercept_),
        "calibrator_coefficients": {
            "original": original_calibrator.coef_.tolist(),
            "reconstruction": rebuilt_calibrator.coef_.tolist(),
            "original_intercept": original_calibrator.intercept_.tolist(),
            "reconstruction_intercept": rebuilt_calibrator.intercept_.tolist(),
        },
    }
    if args.original_predictions and args.reconstructed_predictions:
        left = pd.read_csv(args.original_predictions, parse_dates=["date"])
        right = pd.read_csv(args.reconstructed_predictions, parse_dates=["date"])
        report["saved_predictions_equal"] = left.equals(right)
        report["saved_prediction_shapes"] = [list(left.shape), list(right.shape)]
        report["saved_prediction_columns_equal"] = list(left.columns) == list(right.columns)
        report["saved_prediction_row_count_equal"] = len(left) == len(right)
    for label, left_path, right_path in (
        (
            "independent_2026_onset_predictions",
            args.original_2026_predictions,
            args.reconstructed_2026_predictions,
        ),
        (
            "independent_2026_all_grid_scores",
            args.original_2026_grid_scores,
            args.reconstructed_2026_grid_scores,
        ),
    ):
        if left_path and right_path:
            left = pd.read_csv(left_path, parse_dates=["date"])
            right = pd.read_csv(right_path, parse_dates=["date"])
            report[label] = {
                "equal": left.equals(right),
                "rows": int(len(left)),
                "columns": int(len(left.columns)),
                "shapes_equal": left.shape == right.shape,
                "maximum_numeric_absolute_difference": float(
                    left.select_dtypes(include="number")
                    .subtract(right.select_dtypes(include="number"))
                    .abs()
                    .to_numpy()
                    .max(initial=0.0)
                )
                if left.shape == right.shape
                else None,
            }
    if args.original_2026_metrics and args.reconstructed_2026_metrics:
        left_metrics = json.loads(args.original_2026_metrics.read_text(encoding="utf-8"))
        right_metrics = json.loads(args.reconstructed_2026_metrics.read_text(encoding="utf-8"))
        left_metrics.pop("frozen_artifact_sha256", None)
        right_metrics.pop("frozen_artifact_sha256", None)
        report["independent_2026_metrics_equal_except_artifact_hash"] = (
            left_metrics == right_metrics
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
