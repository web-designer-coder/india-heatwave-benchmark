"""Run state-conditioned Day-1 transition tests on a saved benchmark frame."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from heatwave_benchmark.config import ExperimentConfig  # noqa: E402
from heatwave_benchmark.experiments import select_model_features  # noqa: E402
from heatwave_benchmark.features import (  # noqa: E402
    add_temperature_anomaly_features,
    fit_calendar_temperature_normals,
)
from heatwave_benchmark.artifacts import (  # noqa: E402
    benchmark_feature_frame_path,
    transition_output_directory,
)
from heatwave_benchmark.metrics import (  # noqa: E402
    bootstrap_group_metric_means,
    bootstrap_paired_group_differences,
)
from heatwave_benchmark.onset import state_conditioned_group_metrics  # noqa: E402
from heatwave_benchmark.splits import (  # noqa: E402
    buffered_spatiotemporal_split,
    make_spatial_block_folds,
    temporal_split,
)
from heatwave_benchmark.transitions import (  # noqa: E402
    InsufficientTransitionDataError,
    fit_onset_model_search,
    run_state_transition_comparison,
)


_MODEL_NAMES = (
    "pooled_hist_gradient_boosting",
    "pooled_hist_gradient_boosting_with_issue_state",
    "state_transition_hist_gradient_boosting",
    "climatology",
    "persistence",
)
_PROBABILITY_TYPES = ("raw", "platt_calibrated")
_BOOTSTRAP_METRICS = (
    "average_precision",
    "brier_score",
    "event_recall",
    "false_alarm_rate",
)


def default_output_directory(predictor_source: str) -> Path:
    """Return a source-isolated destination for provisional or primary results."""
    return transition_output_directory(predictor_source)


def _add_result(
    result,
    scenario: str,
    fold_index: int | None,
    held_out_blocks: Sequence[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    metrics = result.metrics.copy()
    predictions = result.test_predictions.copy()
    metrics["scenario"] = scenario
    metrics["horizon_days"] = 1
    predictions["scenario"] = scenario
    predictions["horizon_days"] = 1
    if fold_index is not None:
        metrics["spatial_fold"] = fold_index
        predictions["spatial_fold"] = fold_index
    record: dict[str, object] = {
        "status": "complete",
        "scenario": scenario,
        "fold": fold_index,
        "held_out_blocks": list(held_out_blocks) if held_out_blocks is not None else [],
        "train_rows": int(result.summary["training_rows_by_state"][0] + result.summary["training_rows_by_state"][1]),
        "validation_rows": int(result.summary["validation_rows_by_state"][0] + result.summary["validation_rows_by_state"][1]),
        "test_rows": int(result.summary["test_rows_by_state"][0] + result.summary["test_rows_by_state"][1]),
        **result.summary,
    }
    return metrics, predictions, record


def _grouped_test_metrics(
    predictions: pd.DataFrame,
    scenario: str,
    group_col: str,
) -> pd.DataFrame:
    working = predictions.copy()
    if group_col == "test_year":
        working[group_col] = pd.to_datetime(working["date"], errors="raise").dt.year
    rows = []
    for model_name in _MODEL_NAMES:
        for probability_type in _PROBABILITY_TYPES:
            probability_column = (
                f"{model_name}_raw_probability"
                if probability_type == "raw"
                else f"{model_name}_calibrated_probability"
            )
            threshold_suffix = "raw" if probability_type == "raw" else "calibrated"
            threshold_column = f"{model_name}_threshold_{threshold_suffix}"
            if probability_column not in working or threshold_column not in working:
                continue
            group_metrics = state_conditioned_group_metrics(
                working,
                group_col=group_col,
                probability_col=probability_column,
                threshold_col=threshold_column,
            )
            group_metrics["model"] = model_name
            group_metrics["probability_type"] = probability_type
            group_metrics["scenario"] = scenario
            group_metrics["resampling_unit"] = group_col
            rows.append(group_metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _bootstrap_intervals(group_metrics: pd.DataFrame, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    interval_rows = []
    paired_rows = []
    group_columns = {"test_year": "test_year", "spatial_fold": "spatial_fold"}
    for index, ((scenario, resampling_unit, model, probability_type, issue_state), values) in enumerate(
        group_metrics.groupby(
            ["scenario", "resampling_unit", "model", "probability_type", "issue_state"],
            sort=True,
        )
    ):
        group_column = group_columns[resampling_unit]
        interval = bootstrap_group_metric_means(
            values,
            _BOOTSTRAP_METRICS,
            n_bootstrap=2000,
            seed=random_seed + index,
        )
        interval["scenario"] = scenario
        interval["resampling_unit"] = resampling_unit
        interval["model"] = model
        interval["probability_type"] = probability_type
        interval["issue_state"] = issue_state
        interval_rows.append(interval)

    for index, ((scenario, resampling_unit, probability_type, issue_state), values) in enumerate(
        group_metrics.groupby(
            ["scenario", "resampling_unit", "probability_type", "issue_state"],
            sort=True,
        )
    ):
        group_column = group_columns[resampling_unit]
        for baseline_index, baseline_model in enumerate(
            (
                "pooled_hist_gradient_boosting",
                "pooled_hist_gradient_boosting_with_issue_state",
            )
        ):
            paired = bootstrap_paired_group_differences(
                values,
                group_col=group_column,
                metric_columns=("average_precision", "brier_score"),
                baseline_model=baseline_model,
                n_bootstrap=2000,
                seed=random_seed + 10_000 + index * 2 + baseline_index,
            )
            paired["scenario"] = scenario
            paired["resampling_unit"] = resampling_unit
            paired["probability_type"] = probability_type
            paired["issue_state"] = issue_state
            paired_rows.append(paired)
    return pd.concat(interval_rows, ignore_index=True), pd.concat(paired_rows, ignore_index=True)


def _run_benchmark(
    frame: pd.DataFrame,
    predictor_source: str,
    max_training_rows: int,
    spatial_folds: int,
    skip_spatial: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required = {"date", "grid_id", "spatial_block", "current_heatwave", "target_h1"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"feature frame is missing required columns: {sorted(missing)}")
    original_rows = len(frame)
    target = pd.to_numeric(frame["target_h1"], errors="coerce")
    state = pd.to_numeric(frame["current_heatwave"], errors="coerce")
    usable = frame.loc[target.notna() & state.notna()].copy()
    usable["target"] = target.loc[usable.index].astype(np.int8)
    usable["current_heatwave"] = state.loc[usable.index].astype(np.int8)
    if not np.isin(usable["target"].to_numpy(), (0, 1)).all():
        raise ValueError("known target_h1 values must be binary")
    if not np.isin(usable["current_heatwave"].to_numpy(), (0, 1)).all():
        raise ValueError("known current_heatwave values must be binary")
    if usable.empty:
        raise ValueError("no records have both a known Day-1 target and issue-day state")

    config = ExperimentConfig(
        label_normal_years=(1981, 2010),
        train_years=(1981, 2007),
        validation_years=(2008, 2013),
        test_years=(2014, 2025),
    )
    metrics_frames = []
    prediction_frames = []
    summary_records = []
    temporal_partitions = temporal_split(usable, config)
    temporal_result = run_state_transition_comparison(
        temporal_partitions["train"],
        temporal_partitions["validation"],
        temporal_partitions["test"],
        target_col="target",
        state_col="current_heatwave",
        max_training_rows=max_training_rows,
        max_false_alarm_rate=0.1,
        random_state=config.random_seed,
    )
    metrics, predictions, record = _add_result(temporal_result, "temporal", None, None)
    metrics_frames.append(metrics)
    prediction_frames.append(predictions)
    summary_records.append(record)

    if not skip_spatial:
        n_folds = min(spatial_folds, usable["spatial_block"].nunique())
        if n_folds < 2:
            raise ValueError("spatial evaluation requires at least two observed spatial blocks")
        folds = make_spatial_block_folds(usable, n_splits=n_folds, seed=config.random_seed)
        for fold_index, held_out_blocks in enumerate(folds, start=1):
            try:
                partitions = buffered_spatiotemporal_split(
                    usable,
                    config,
                    held_out_blocks=held_out_blocks,
                    buffer_degrees=1.0,
                )
            except ValueError as exc:
                if "partition is empty after applying time and spatial filters" not in str(exc):
                    raise
                summary_records.append(
                    {
                        "status": "not_evaluable",
                        "scenario": "spatial_transfer",
                        "fold": fold_index,
                        "held_out_blocks": list(held_out_blocks),
                        "reason": str(exc),
                    }
                )
                continue
            try:
                result = run_state_transition_comparison(
                    partitions["train"],
                    partitions["validation"],
                    partitions["test"],
                    target_col="target",
                    state_col="current_heatwave",
                    max_training_rows=max_training_rows,
                    max_false_alarm_rate=0.1,
                    random_state=config.random_seed + fold_index,
                )
            except InsufficientTransitionDataError as exc:
                summary_records.append(
                    {
                        "status": "not_evaluable",
                        "scenario": "spatial_transfer",
                        "fold": fold_index,
                        "held_out_blocks": list(held_out_blocks),
                        "reason": str(exc),
                    }
                )
                continue
            metrics, predictions, record = _add_result(
                result, "spatial_transfer", fold_index, held_out_blocks
            )
            metrics_frames.append(metrics)
            prediction_frames.append(predictions)
            summary_records.append(record)

    metrics = pd.concat(metrics_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    grouped_frames = []
    temporal_predictions = predictions.loc[predictions["scenario"].eq("temporal")]
    if not temporal_predictions.empty:
        grouped_frames.append(_grouped_test_metrics(temporal_predictions, "temporal", "test_year"))
    spatial_predictions = predictions.loc[predictions["scenario"].eq("spatial_transfer")]
    if not spatial_predictions.empty:
        grouped_frames.append(_grouped_test_metrics(spatial_predictions, "spatial_transfer", "spatial_fold"))
    group_metrics = pd.concat(grouped_frames, ignore_index=True) if grouped_frames else pd.DataFrame()
    intervals, paired = _bootstrap_intervals(group_metrics, config.random_seed)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "predictor_source": predictor_source,
        "target": "IMD-aligned target_h1",
        "evaluation": {
            "train_years": list(config.train_years),
            "validation_years": list(config.validation_years),
            "test_years": list(config.test_years),
            "spatial_block_degrees": 5.0,
            "spatial_buffer_degrees": 1.0,
            "max_validation_false_alarm_rate": 0.1,
            "max_training_rows": max_training_rows,
            "random_seed": config.random_seed,
        },
        "input_rows": int(original_rows),
        "rows_dropped_for_unknown_target_or_issue_state": int(original_rows - len(usable)),
        "complete_case_rows": int(len(usable)),
        "temporal_and_spatial_results": summary_records,
        "resampling": {
            "temporal": "equal-weight test-year bootstrap",
            "spatial_transfer": "equal-weight held-out-fold bootstrap",
            "replicates": 2000,
            "spatial_fold_count": int(predictions["spatial_fold"].nunique())
            if "spatial_fold" in predictions
            else 0,
            "note": "Spatial intervals are descriptive and based on at most five folds.",
        },
        "scope_note": (
            "The transition method is Day-1 only. Spatial-transfer scores are conditional on the observed issue-day "
            "heatwave state being available at held-out cells. Reanalysis inputs are retrospective and do not establish "
            "operational forecast skill. No algorithm-novelty claim is made."
        ),
    }
    return metrics, predictions, group_metrics, intervals, paired, summary


def _run_onset_model(
    frame: pd.DataFrame, era5_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    """Select an onset classifier on expanding blocked folds, then score the prior holdout."""
    required = {"date", "grid_id", "spatial_block", "current_heatwave", "target_h1", "t2m_max_lag_0", "t2m_max_lag_7"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"feature frame is missing onset-model columns: {sorted(missing)}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    target = pd.to_numeric(data["target_h1"], errors="coerce")
    state = pd.to_numeric(data["current_heatwave"], errors="coerce")
    usable = data.loc[target.notna() & state.notna()].copy()
    usable["target"] = target.loc[usable.index].astype(np.int8)
    usable["current_heatwave"] = state.loc[usable.index].astype(np.int8)
    if not np.isin(usable["target"], (0, 1)).all() or not np.isin(usable["current_heatwave"], (0, 1)).all():
        raise ValueError("onset model requires known binary targets and issue-day states")

    with xr.open_dataset(era5_path) as dataset:
        if "t2m_max" not in dataset:
            raise ValueError("ERA5 daily dataset must contain t2m_max")
        weather_data = dataset["t2m_max"]
        if not {"time", "grid_id"}.issubset(weather_data.dims):
            raise ValueError("ERA5 t2m_max must have time and grid_id dimensions")

        fold_specs = ((1981, 1995, 1996, 2001), (1981, 2001, 2002, 2007), (1981, 2007, 2008, 2013))
        candidate_rows: list[pd.DataFrame] = []
        anomaly_columns = [
            "t2m_max_anom_lag_0", "t2m_max_anom_lag_3", "t2m_max_anom_lag_7",
            "t2m_max_anom_mean_3d", "t2m_max_anom_mean_7d", "t2m_max_anom_change_0_3",
        ]
        for fold_index, (train_start, train_end, valid_start, valid_end) in enumerate(fold_specs, start=1):
            weather = weather_data.sel(time=slice(f"{train_start}-01-01", f"{train_end}-12-31"))
            weather_frame = weather.to_dataframe(name="t2m_max").reset_index()
            weather_frame = weather_frame.rename(columns={"time": "date"})
            weather_frame = weather_frame.dropna(subset=["t2m_max"])
            if weather_frame.empty:
                raise ValueError(f"ERA5 t2m_max training weather is empty for fold {fold_index}")
            normals = fit_calendar_temperature_normals(weather_frame)
            train = usable.loc[usable["date"].dt.year.between(train_start, train_end)].copy()
            validation = usable.loc[usable["date"].dt.year.between(valid_start, valid_end)].copy()
            train = add_temperature_anomaly_features(train, normals)
            validation = add_temperature_anomaly_features(validation, normals)
            features = list(dict.fromkeys([*select_model_features(train), *anomaly_columns]))
            result = fit_onset_model_search(
                train, validation, validation,
                feature_columns=features,
                random_state=42 + fold_index,
            )
            scores = result.candidate_scores.copy()
            scores["fold"] = fold_index
            scores["train_years"] = f"{train_start}-{train_end}"
            scores["validation_years"] = f"{valid_start}-{valid_end}"
            scores["normal_fit_years"] = f"{train_start}-{train_end}"
            candidate_rows.append(scores)

        candidate_scores = pd.concat(candidate_rows, ignore_index=True)
        mean_scores = candidate_scores.groupby(
            ["max_leaf_nodes", "min_samples_leaf", "l2_regularization"], as_index=False
        ).agg(
            mean_validation_average_precision=("validation_average_precision", "mean"),
            mean_validation_brier_score=("validation_brier_score", "mean"),
            validation_folds=("fold", "nunique"),
        )
        selected_row = mean_scores.sort_values(
            ["mean_validation_average_precision", "mean_validation_brier_score"],
            ascending=[False, True], kind="stable",
        ).iloc[0]
        selected_params = {
            "max_leaf_nodes": int(selected_row["max_leaf_nodes"]),
            "min_samples_leaf": int(selected_row["min_samples_leaf"]),
            "l2_regularization": float(selected_row["l2_regularization"]),
        }

        final_normals_weather = weather_data.sel(time=slice("1981-01-01", "2007-12-31"))
        final_weather_frame = final_normals_weather.to_dataframe(name="t2m_max").reset_index()
        final_weather_frame = final_weather_frame.rename(columns={"time": "date"}).dropna(subset=["t2m_max"])
        final_normals = fit_calendar_temperature_normals(final_weather_frame)
        train = usable.loc[usable["date"].dt.year.between(1981, 2007)].copy()
        validation = usable.loc[usable["date"].dt.year.between(2008, 2013)].copy()
        test = usable.loc[usable["date"].dt.year.between(2014, 2025)].copy()
        train = add_temperature_anomaly_features(train, final_normals)
        validation = add_temperature_anomaly_features(validation, final_normals)
        test = add_temperature_anomaly_features(test, final_normals)
        features = list(dict.fromkeys([*select_model_features(train), *anomaly_columns]))
        final_result = fit_onset_model_search(
            train, validation, test, feature_columns=features,
            parameter_grid=[selected_params], random_state=42,
        )

        config = ExperimentConfig(
            label_normal_years=(1981, 2010), train_years=(1981, 2007),
            validation_years=(2008, 2013), test_years=(2014, 2025),
        )
        spatial_rows: list[dict[str, object]] = []
        spatial_prediction_frames: list[pd.DataFrame] = []
        fold_count = min(5, int(usable["spatial_block"].nunique()))
        if fold_count >= 2:
            spatial_folds = make_spatial_block_folds(usable, n_splits=fold_count, seed=config.random_seed)
            for fold_index, held_out_blocks in enumerate(spatial_folds, start=1):
                try:
                    partitions = buffered_spatiotemporal_split(
                        usable, config, held_out_blocks=held_out_blocks, buffer_degrees=1.0
                    )
                    spatial_train = add_temperature_anomaly_features(partitions["train"], final_normals)
                    spatial_validation = add_temperature_anomaly_features(partitions["validation"], final_normals)
                    spatial_test = add_temperature_anomaly_features(partitions["test"], final_normals)
                    spatial_features = list(dict.fromkeys([*select_model_features(spatial_train), *anomaly_columns]))
                    spatial_result = fit_onset_model_search(
                        spatial_train, spatial_validation, spatial_test,
                        feature_columns=spatial_features,
                        parameter_grid=[selected_params],
                        random_state=config.random_seed + fold_index,
                    )
                except ValueError as exc:
                    spatial_rows.append({
                        "fold": fold_index,
                        "status": "not_evaluable",
                        "held_out_blocks": list(held_out_blocks),
                        "reason": str(exc),
                    })
                    continue
                spatial_record: dict[str, object] = {
                    "fold": fold_index,
                    "status": "complete",
                    "held_out_blocks": list(held_out_blocks),
                    "train_rows": int(len(partitions["train"])),
                    "validation_rows": int(len(partitions["validation"])),
                    "test_rows": int(len(partitions["test"])),
                    **spatial_result.metrics,
                }
                spatial_rows.append(spatial_record)
                spatial_predictions = spatial_result.test_predictions.copy()
                spatial_predictions["scenario"] = "spatial_transfer_exploratory"
                spatial_predictions["spatial_fold"] = fold_index
                spatial_predictions["horizon_days"] = 1
                spatial_prediction_frames.append(spatial_predictions)

    summary = {
        "primary_selection_metric": "mean validation average precision for next-day onsets among issue-state-zero rows",
        "selection_folds": [
            {"train_years": [a, b], "validation_years": [c, d], "normal_fit_years": [a, b]}
            for a, b, c, d in fold_specs
        ],
        "selected_parameters": selected_params,
        "selected_mean_validation_average_precision": float(selected_row["mean_validation_average_precision"]),
        "selected_mean_validation_brier_score": float(selected_row["mean_validation_brier_score"]),
        "final_fit_years": [1981, 2007],
        "calibration_and_threshold_years": [2008, 2013],
        "exploratory_test_years": [2014, 2025],
        "test_status": "exploratory; previous baseline holdout results were inspected before this redesign",
        "spatial_evaluation": {
            "method": "five-fold 5-degree block holdout with 1-degree buffer; prior 2014-2025 test period",
            "normal_fit_years": [1981, 2007],
            "normal_scope": "all study cells; held-out-cell ERA5 predictor history used without target labels",
            "results": spatial_rows,
            "status": "exploratory because earlier spatial holdouts were inspected before redesign",
        },
        "normal_method": "grid-cell-specific circular 31-day day-of-year ERA5 t2m_max mean, fit to training-period predictor records only",
        "feature_columns": features,
        "candidate_selection_table": mean_scores.to_dict(orient="records"),
        "final_model": final_result.summary,
        "final_test_metrics": final_result.metrics,
    }
    predictions = final_result.test_predictions.copy()
    predictions["scenario"] = "temporal_onset_exploratory"
    predictions["horizon_days"] = 1
    if spatial_prediction_frames:
        predictions = pd.concat([predictions, *spatial_prediction_frames], ignore_index=True, sort=False)
    frozen_bundle = {
        "format_version": 1,
        "estimator": final_result.model,
        "calibrator": final_result.calibrator,
        "temperature_normals": final_normals,
        "feature_columns": features,
        "selected_parameters": selected_params,
        "probability_threshold": float(final_result.metrics["validation_threshold"]),
        "target_definition": "next-day heatwave label equals 1, conditional on current_heatwave=0",
        "training_years": [1981, 2007],
        "calibration_and_threshold_years": [2008, 2013],
        "normal_fit_years": [1981, 2007],
        "predictor_snapshot_id": "TZ2KX6A24CJ5DG17T7MG",
    }
    return candidate_scores, predictions, summary, frozen_bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictor-source",
        choices=("nasa_power", "era5"),
        default="nasa_power",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--max-training-rows", type=int, default=200_000)
    parser.add_argument("--spatial-folds", type=int, default=5)
    parser.add_argument("--skip-spatial", action="store_true")
    parser.add_argument("--onset-model", action="store_true", help="run the expanding-window onset-focused ERA5 experiment")
    parser.add_argument("--era5-daily", type=Path, default=PROJECT_ROOT / "data" / "interim" / "era5_1981_2025.nc")
    args = parser.parse_args(argv)
    if args.max_training_rows < 4:
        parser.error("--max-training-rows must be at least 4")
    if args.spatial_folds < 2:
        parser.error("--spatial-folds must be at least 2")
    feature_path = args.features or benchmark_feature_frame_path(args.predictor_source)
    if not feature_path.is_file():
        raise FileNotFoundError(f"feature frame not found: {feature_path}")

    print(f"Loading Day-1 feature frame from {feature_path}", flush=True)
    frame = pd.read_csv(feature_path, parse_dates=["date"])
    if args.onset_model:
        if args.predictor_source != "era5":
            parser.error("--onset-model currently requires --predictor-source era5")
        if not args.era5_daily.is_file():
            raise FileNotFoundError(f"ERA5 daily dataset not found: {args.era5_daily}")
        candidate_scores, onset_predictions, onset_summary, frozen_bundle = _run_onset_model(frame, args.era5_daily)
        output_directory = args.output_directory or PROJECT_ROOT / "reports" / "onset_model_era5"
        output_directory.mkdir(parents=True, exist_ok=True)
        candidate_scores.to_csv(output_directory / "onset_cv_candidate_scores.csv", index=False)
        joblib.dump(frozen_bundle, output_directory / "onset_model_frozen.joblib")
        onset_predictions.to_csv(
            output_directory / "onset_test_predictions.csv.gz", index=False, compression="gzip"
        )
        (output_directory / "onset_summary.json").write_text(
            json.dumps(onset_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"Saved onset-model results under {output_directory}", flush=True)
        return 0
    metrics, predictions, group_metrics, intervals, paired, summary = _run_benchmark(
        frame,
        predictor_source=args.predictor_source,
        max_training_rows=args.max_training_rows,
        spatial_folds=args.spatial_folds,
        skip_spatial=args.skip_spatial,
    )
    output_directory = args.output_directory or default_output_directory(args.predictor_source)
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_directory / "transition_metrics.csv", index=False)
    predictions.to_csv(
        output_directory / "transition_test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    group_metrics.to_csv(output_directory / "transition_group_metrics.csv", index=False)
    intervals.to_csv(output_directory / "transition_macro_bootstrap_intervals.csv", index=False)
    paired.to_csv(output_directory / "transition_paired_intervals.csv", index=False)
    output_paths = {
        "metrics": "transition_metrics.csv",
        "predictions": "transition_test_predictions.csv.gz",
        "group_metrics": "transition_group_metrics.csv",
        "macro_bootstrap_intervals": "transition_macro_bootstrap_intervals.csv",
        "paired_intervals": "transition_paired_intervals.csv",
    }
    summary["output_files"] = output_paths
    (output_directory / "transition_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved source-specific transition results under {output_directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
