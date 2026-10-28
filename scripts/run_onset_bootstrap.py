"""Add year- and block-resampled uncertainty to the Day-1 onset diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from heatwave_benchmark.metrics import bootstrap_group_metric_means  # noqa: E402
from heatwave_benchmark.onset import state_conditioned_group_metrics  # noqa: E402
from heatwave_benchmark.artifacts import (  # noqa: E402
    benchmark_feature_frame_path,
    benchmark_output_directory,
)


METRICS = ("average_precision", "brier_score", "event_recall", "false_alarm_rate")


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
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261024)
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
    models = [column.removesuffix("_calibrated_probability") for column in probability_columns]
    required = {
        "date",
        "grid_id",
        "target",
        "scenario",
        "horizon_days",
        "spatial_fold",
        *probability_columns,
        *(f"{model}_threshold_calibrated" for model in models),
    }
    missing_columns = required - set(prediction_header)
    if missing_columns:
        raise ValueError(f"prediction file is missing columns: {sorted(missing_columns)}")

    predictions = pd.read_csv(
        predictions_path,
        usecols=sorted(required),
        parse_dates=["date"],
    )
    predictions = predictions.loc[
        predictions["horizon_days"].eq(1)
        & predictions["scenario"].isin(("temporal", "spatial_transfer"))
    ].copy()
    if predictions.empty:
        raise ValueError("prediction file has no temporal or spatial-transfer Day-1 test rows")
    predictions["target_year"] = predictions["date"].dt.year

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
    predictions = predictions.loc[
        predictions["target"].notna() & predictions["current_heatwave"].notna()
    ].copy()
    if predictions.empty:
        raise ValueError("no test rows have both a known target and issue-day heatwave state")

    group_frames: list[pd.DataFrame] = []

    def add_group_metrics(
        frame: pd.DataFrame,
        scenario: str,
        model: str,
        group_col: str,
        group_unit: str,
    ) -> None:
        if frame.empty:
            return
        scored = state_conditioned_group_metrics(
            frame,
            group_col=group_col,
            probability_col="probability",
            threshold_col="threshold",
        )
        scored = scored.rename(columns={group_col: "resample_group"})
        scored.insert(0, "scenario", scenario)
        scored.insert(1, "model", model)
        scored.insert(2, "group_unit", group_unit)
        group_frames.append(scored)

    temporal = predictions.loc[predictions["scenario"].eq("temporal")].copy()
    temporal_selected = str(summary["temporal_h1"]["best_validation_model"])
    for model in models:
        values = temporal[["target_year", "target", "current_heatwave"]].copy()
        values["probability"] = temporal[f"{model}_calibrated_probability"]
        values["threshold"] = temporal[f"{model}_threshold_calibrated"]
        add_group_metrics(values, "temporal", model, "target_year", "year")
        if model == temporal_selected:
            add_group_metrics(
                values,
                "temporal",
                "validation_selected",
                "target_year",
                "year",
            )

    spatial = predictions.loc[predictions["scenario"].eq("spatial_transfer")].copy()
    selected_by_fold = {
        int(record["fold"]): str(record["best_validation_model"])
        for record in summary["spatial_transfer_h1"]
    }
    selected_parts: list[pd.DataFrame] = []
    for model in models:
        values = spatial[["spatial_fold", "target", "current_heatwave"]].copy()
        values["probability"] = spatial[f"{model}_calibrated_probability"]
        values["threshold"] = spatial[f"{model}_threshold_calibrated"]
        add_group_metrics(values, "spatial_transfer", model, "spatial_fold", "spatial_fold")
    for fold, selected_model in selected_by_fold.items():
        fold_values = spatial.loc[spatial["spatial_fold"].eq(fold)].copy()
        if fold_values.empty:
            continue
        part = fold_values[["spatial_fold", "target", "current_heatwave"]].copy()
        part["probability"] = fold_values[f"{selected_model}_calibrated_probability"]
        part["threshold"] = fold_values[f"{selected_model}_threshold_calibrated"]
        selected_parts.append(part)
    if selected_parts:
        add_group_metrics(
            pd.concat(selected_parts, ignore_index=True),
            "spatial_transfer",
            "validation_selected_per_fold",
            "spatial_fold",
            "spatial_fold",
        )

    if not group_frames:
        raise ValueError("no grouped onset metrics were produced")
    grouped = pd.concat(group_frames, ignore_index=True)
    intervals: list[pd.DataFrame] = []
    seed_rng = np.random.default_rng(args.seed)
    for keys, values in grouped.groupby(
        ["scenario", "model", "group_unit", "issue_state"], sort=True
    ):
        scenario, model, group_unit, issue_state = keys
        for metric in METRICS:
            result = bootstrap_group_metric_means(
                values,
                [metric],
                n_bootstrap=args.n_bootstrap,
                seed=int(seed_rng.integers(0, np.iinfo(np.int32).max)),
            )
            result.insert(0, "issue_state", issue_state)
            result.insert(0, "group_unit", group_unit)
            result.insert(0, "model", model)
            result.insert(0, "scenario", scenario)
            intervals.append(result)

    output_directory.mkdir(parents=True, exist_ok=True)
    group_path = output_directory / "onset_group_metrics.csv"
    interval_path = output_directory / "onset_bootstrap_intervals.csv"
    output_summary_path = output_directory / "onset_bootstrap_summary.json"
    grouped.to_csv(group_path, index=False)
    pd.concat(intervals, ignore_index=True).to_csv(interval_path, index=False)
    analysis_summary = {
        "analysis": "Day-1 state-conditioned macro-metric bootstrap",
        "bootstrap": {
            "method": "percentile bootstrap of equal-weight year or spatial-fold metrics",
            "replicates": args.n_bootstrap,
            "seed": args.seed,
            "undefined_group_metrics": "non-finite group values are excluded metric-by-metric",
        },
        "resampling_units": {
            "temporal": "test year",
            "spatial_transfer": "held-out 5-degree spatial fold",
        },
        "group_counts": {
            "temporal_years": int(temporal["target_year"].nunique()),
            "spatial_folds": int(spatial["spatial_fold"].nunique()),
        },
        "limitations": "Spatial uncertainty uses only five folds; percentile intervals are descriptive and should not be interpreted as precise population-level confidence intervals. AP and recall groups without positive cases are undefined and omitted for those metrics.",
        "predictor_source": summary["data"]["predictor_source"],
        "scope_note": "Issue-day event labels only stratify evaluation. Predictor inputs are retrospective analyses and do not establish operational forecast skill.",
        "group_metrics_csv": str(group_path.relative_to(PROJECT_ROOT)),
        "intervals_csv": str(interval_path.relative_to(PROJECT_ROOT)),
        "n_group_metric_rows": int(len(grouped)),
        "n_interval_rows": int(sum(len(item) for item in intervals)),
    }
    output_summary_path.write_text(
        json.dumps(analysis_summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved onset group metrics: {group_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved onset bootstrap intervals: {interval_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved onset bootstrap summary: {output_summary_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
