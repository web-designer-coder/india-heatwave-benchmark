"""Evaluate next-day heatwave skill separately for onset and active-event cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from heatwave_benchmark.onset import state_conditioned_metrics  # noqa: E402
from heatwave_benchmark.artifacts import (  # noqa: E402
    benchmark_feature_frame_path,
    benchmark_output_directory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--predictor-source", choices=("nasa_power", "era5"), default="nasa_power"
    )
    args = parser.parse_args()

    default_reports = benchmark_output_directory(args.predictor_source)
    predictions_path = args.predictions or default_reports / "test_predictions.csv.gz"
    features_path = args.features or benchmark_feature_frame_path(args.predictor_source)
    input_summary_path = args.summary or default_reports / "benchmark_summary.json"
    output_directory = args.output_directory or default_reports

    missing = [path for path in (predictions_path, features_path, input_summary_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required benchmark inputs are missing: {missing}")

    summary = json.loads(input_summary_path.read_text(encoding="utf-8"))
    if summary["data"]["predictor_source"] != args.predictor_source:
        raise ValueError("selected predictor source does not match the benchmark summary")
    prediction_header = pd.read_csv(predictions_path, nrows=0).columns.tolist()
    probability_columns = sorted(
        column for column in prediction_header if column.endswith("_calibrated_probability")
    )
    if not probability_columns:
        raise ValueError("prediction file contains no calibrated model probabilities")
    models = [column.removesuffix("_calibrated_probability") for column in probability_columns]
    required_prediction_columns = {
        "date",
        "grid_id",
        "target",
        "scenario",
        "horizon_days",
        "spatial_fold",
        *probability_columns,
        *(f"{model}_threshold_calibrated" for model in models),
    }
    missing_prediction_columns = required_prediction_columns - set(prediction_header)
    if missing_prediction_columns:
        raise ValueError(
            f"prediction file is missing columns: {sorted(missing_prediction_columns)}"
        )

    predictions = pd.read_csv(
        predictions_path,
        usecols=sorted(required_prediction_columns),
        parse_dates=["date"],
    )
    predictions = predictions.loc[
        predictions["horizon_days"].eq(1)
        & predictions["scenario"].isin(("temporal", "spatial_transfer"))
    ].copy()
    if predictions.empty:
        raise ValueError("prediction file has no temporal or spatial-transfer Day-1 test rows")

    features = pd.read_csv(
        features_path,
        usecols=["date", "grid_id", "current_heatwave"],
        parse_dates=["date"],
    )
    if features.duplicated(["date", "grid_id"]).any():
        raise ValueError("feature frame must have unique date and grid_id rows")
    predictions = predictions.merge(
        features,
        on=["date", "grid_id"],
        how="left",
        validate="many_to_one",
    )
    unmatched = int(predictions["current_heatwave"].isna().sum())
    before_state_filter = len(predictions)
    predictions = predictions.loc[
        predictions["target"].notna() & predictions["current_heatwave"].notna()
    ].copy()
    if predictions.empty:
        raise ValueError("no test rows have both a known target and issue-day heatwave state")

    output_rows: list[dict[str, object]] = []

    def append_metrics(
        frame: pd.DataFrame,
        scenario: str,
        model: str,
        spatial_fold: int | None,
    ) -> None:
        values = frame[["target", "current_heatwave", "probability", "threshold"]].copy()
        scored = state_conditioned_metrics(values, "probability", "threshold")
        for result in scored.to_dict(orient="records"):
            output_rows.append(
                {
                    "scenario": scenario,
                    "horizon_days": 1,
                    "model": model,
                    "spatial_fold": spatial_fold,
                    **result,
                }
            )

    temporal = predictions.loc[predictions["scenario"].eq("temporal")].copy()
    temporal_selected = str(summary["temporal_h1"]["best_validation_model"])
    for model in models:
        temporal_values = temporal[["target", "current_heatwave"]].copy()
        temporal_values["probability"] = temporal[f"{model}_calibrated_probability"]
        temporal_values["threshold"] = temporal[f"{model}_threshold_calibrated"]
        append_metrics(temporal_values, "temporal", model, None)
        if model == temporal_selected:
            append_metrics(temporal_values, "temporal", "validation_selected", None)

    spatial = predictions.loc[predictions["scenario"].eq("spatial_transfer")].copy()
    selected_by_fold = {
        int(record["fold"]): str(record["best_validation_model"])
        for record in summary["spatial_transfer_h1"]
    }
    selected_parts: list[pd.DataFrame] = []
    for model in models:
        model_values = spatial[["target", "current_heatwave", "spatial_fold"]].copy()
        model_values["probability"] = spatial[f"{model}_calibrated_probability"]
        model_values["threshold"] = spatial[f"{model}_threshold_calibrated"]
        append_metrics(model_values, "spatial_transfer", model, None)
        for fold, fold_values in model_values.groupby("spatial_fold", sort=True):
            append_metrics(fold_values, "spatial_transfer", model, int(fold))

    for fold, selected_model in selected_by_fold.items():
        fold_values = spatial.loc[spatial["spatial_fold"].eq(fold)].copy()
        if fold_values.empty:
            continue
        part = fold_values[["target", "current_heatwave", "spatial_fold"]].copy()
        part["probability"] = fold_values[f"{selected_model}_calibrated_probability"]
        part["threshold"] = fold_values[f"{selected_model}_threshold_calibrated"]
        selected_parts.append(part)
        append_metrics(part, "spatial_transfer", "validation_selected_per_fold", fold)
    if selected_parts:
        append_metrics(
            pd.concat(selected_parts, ignore_index=True),
            "spatial_transfer",
            "validation_selected_per_fold",
            None,
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = output_directory / "onset_diagnostic_metrics.csv"
    output_summary_path = output_directory / "onset_diagnostic_summary.json"
    metrics = pd.DataFrame(output_rows).sort_values(
        ["scenario", "model", "spatial_fold", "issue_state"],
        na_position="first",
        kind="stable",
    )
    metrics.to_csv(metrics_path, index=False)
    analysis_summary = {
        "analysis": "day-1 event transition diagnostic",
        "issue_state_definition": {
            "no_active_heatwave": "IMD-aligned issue-day label is 0; Day-1 positive labels are 0-to-1 transitions",
            "active_heatwave": "IMD-aligned issue-day label is 1; this group includes both continuation and cessation outcomes",
        },
        "target_period": summary["config"]["test_years"],
        "horizon_days": 1,
        "predictor_source": summary["data"]["predictor_source"],
        "n_grid_cells": summary["data"]["n_grid_cells"],
        "rows_before_state_filter": before_state_filter,
        "rows_dropped_unknown_target_or_issue_state": before_state_filter - len(predictions),
        "rows_without_matching_issue_state": unmatched,
        "temporal_validation_selected_model": temporal_selected,
        "spatial_validation_selected_model_by_fold": selected_by_fold,
        "threshold_policy": "Carry the validation-fitted calibrated decision threshold into each subset; do not retune on test data.",
        "scope_note": (
            "Issue-day event labels are used only to stratify test metrics. They are not included "
            f"as model predictors. {('NASA POWER' if args.predictor_source == 'nasa_power' else 'ERA5')} "
            "inputs are retrospective daily analyses, so results do not demonstrate operational forecast skill."
        ),
        "metrics_csv": str(metrics_path.relative_to(PROJECT_ROOT)),
        "n_metric_rows": len(metrics),
    }
    output_summary_path.write_text(
        json.dumps(analysis_summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved onset diagnostics: {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved diagnostic summary: {output_summary_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
