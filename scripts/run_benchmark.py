"""Run the frozen temporal and buffered spatial heatwave benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xarray as xr
from sklearn.metrics import PrecisionRecallDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from heatwave_benchmark.config import ExperimentConfig  # noqa: E402
from heatwave_benchmark.artifacts import (  # noqa: E402
    benchmark_feature_frame_path,
    benchmark_output_directory,
)
from heatwave_benchmark.experiments import run_model_comparison  # noqa: E402
from heatwave_benchmark.labels import compute_daily_normals  # noqa: E402
from heatwave_benchmark.metrics import (  # noqa: E402
    bootstrap_group_metric_means,
    bootstrap_paired_group_differences,
    calibration_bins,
    grouped_metric_summary,
)
from heatwave_benchmark.pipeline import build_benchmark_frame  # noqa: E402
from heatwave_benchmark.splits import (  # noqa: E402
    buffered_spatiotemporal_split,
    make_spatial_block_folds,
    temporal_split,
)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _calibration_axis_limit(probabilities: object) -> float:
    values = np.asarray(probabilities, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.1
    maximum = float(np.clip(finite.max(), 0.0, 1.0))
    return min(1.0, max(0.1, float(np.ceil(maximum / 0.05) * 0.05)))


def _plot_temporal_test(predictions: pd.DataFrame, destination: Path) -> None:
    model_names = sorted(
        column.removesuffix("_raw_probability")
        for column in predictions.columns
        if column.endswith("_raw_probability")
    )
    colors = {
        "climatology": "#7a7a7a",
        "persistence": "#d97706",
        "logistic_regression": "#2563eb",
        "random_forest": "#059669",
        "hist_gradient_boosting": "#7c3aed",
        "xgboost": "#0891b2",
        "hist_gradient_boosting_block_balanced": "#db2777",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    calibration_probabilities: list[float] = []
    for model in model_names:
        raw_column = f"{model}_raw_probability"
        PrecisionRecallDisplay.from_predictions(
            predictions["target"],
            predictions[raw_column],
            name=model.replace("_", " "),
            ax=axes[0],
            color=colors.get(model),
        )
        calibrated = predictions[f"{model}_calibrated_probability"].to_numpy()
        bins = calibration_bins(predictions["target"], calibrated, n_bins=10)
        calibration_probabilities.extend(bins["mean_probability"].dropna().tolist())
        axes[1].plot(
            bins["mean_probability"],
            bins["observed_frequency"],
            marker="o",
            linewidth=1.5,
            label=model.replace("_", " "),
            color=colors.get(model),
        )
    axes[0].set_title("Temporal test: precision–recall")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    calibration_limit = _calibration_axis_limit(calibration_probabilities)
    axes[1].plot(
        [0, calibration_limit],
        [0, calibration_limit],
        linestyle="--",
        color="#9ca3af",
        linewidth=1,
    )
    axes[1].set_title("Temporal test: calibration (zoomed to populated range)")
    axes[1].set_xlabel("Mean predicted probability (zoomed)")
    axes[1].set_ylabel("Observed event frequency")
    axes[1].set_xlim(0, calibration_limit)
    axes[1].set_ylim(0, calibration_limit)
    axes[1].legend(fontsize=7, frameon=False, loc="upper left")
    figure.suptitle("India grid-cell heatwave benchmark, one-day lead", fontsize=13, weight="bold")
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1981)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--max-training-rows", type=int, default=200_000)
    parser.add_argument("--spatial-folds", type=int, default=5)
    parser.add_argument(
        "--predictor-source",
        choices=("nasa_power", "era5"),
        default="nasa_power",
        help="Retrospective daily predictor archive (default: nasa_power)",
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--skip-spatial", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig(
        label_normal_years=(1981, 2010),
        train_years=(1981, 2007),
        validation_years=(2008, 2013),
        test_years=(2014, 2025),
    )
    if args.start_year != 1981 or args.end_year != 2025:
        parser.error("the frozen evaluation config currently expects the full 1981-2025 archive")

    interim = PROJECT_ROOT / "data" / "interim"
    processed = PROJECT_ROOT / "data" / "processed"
    reports = args.output_directory or benchmark_output_directory(args.predictor_source)
    reports.mkdir(parents=True, exist_ok=True)
    tmax_path = interim / "imd_tmax_1981_2025.nc"
    predictor_filename = {
        "nasa_power": "nasa_power_1981_2025.nc",
        "era5": "era5_1981_2025.nc",
    }[args.predictor_source]
    predictor_path = interim / predictor_filename
    if not tmax_path.exists() or not predictor_path.exists():
        raise FileNotFoundError("run scripts/download_public_data.py before the benchmark")

    print(f"Loading validated IMD and {args.predictor_source} grids...", flush=True)
    tmax = xr.load_dataset(tmax_path, engine="scipy")["tmax"]
    meteorology = xr.load_dataset(predictor_path, engine="scipy")
    from heatwave_benchmark.data import load_imd_terrain_mask  # noqa: PLC0415

    terrain = load_imd_terrain_mask()
    normal_tmax = tmax.sel(time=slice("1981-01-01", "2010-12-31"))
    normals = compute_daily_normals(
        normal_tmax,
        *config.label_normal_years,
        window_days=15,
        min_years=10,
    )
    print("Computing fixed-normal labels and calendar-safe lag features...", flush=True)
    frame = build_benchmark_frame(
        meteorology,
        tmax,
        terrain,
        normals,
        forecast_horizons=config.forecast_horizons,
        max_lag_days=config.max_lag_days,
    )
    feature_path = benchmark_feature_frame_path(args.predictor_source)
    frame.to_csv(feature_path, index=False, compression="gzip")
    print(f"Prepared {len(frame):,} March–June issue rows across {frame.grid_id.nunique()} grid cells", flush=True)

    metric_frames = []
    prediction_frames = []
    group_frames = []
    summary: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "label_normal_years": config.label_normal_years,
            "train_years": config.train_years,
            "validation_years": config.validation_years,
            "test_years": config.test_years,
            "forecast_horizons": config.forecast_horizons,
            "max_lag_days": config.max_lag_days,
            "issue_months": [3, 4, 5, 6],
            "spatial_block_degrees": 5.0,
            "spatial_buffer_degrees": 1.0,
            "threshold_max_validation_false_alarm_rate": 0.1,
            "max_training_rows": args.max_training_rows,
            "random_seed": config.random_seed,
        },
        "data": {
            "imd_netcdf": str(tmax_path.relative_to(PROJECT_ROOT)),
            "predictor_source": args.predictor_source,
            "predictor_netcdf": str(predictor_path.relative_to(PROJECT_ROOT)),
            "feature_frame": str(feature_path.relative_to(PROJECT_ROOT)),
            "predictor_metadata": {
                key: str(value) for key, value in meteorology.attrs.items()
            },
            "n_grid_cells": int(frame.grid_id.nunique()),
            "n_issue_rows": int(len(frame)),
            "forecast_lead_interpretation": "target is IMD Tmax heatwave label exactly 1, 2, or 3 calendar days after the issue date; issue-day inputs are retrospective daily summaries and do not represent operational forecast products",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "xarray": _package_version("xarray"),
            "scikit_learn": sklearn.__version__,
            "xgboost": _package_version("xgboost"),
        },
    }

    for horizon in config.forecast_horizons:
        target_col = f"target_h{horizon}"
        usable = frame.loc[frame[target_col].notna()].copy()
        partitions = temporal_split(usable, config)
        result = run_model_comparison(
            partitions["train"].rename(columns={target_col: "target"}),
            partitions["validation"].rename(columns={target_col: "target"}),
            partitions["test"].rename(columns={target_col: "target"}),
            target_col="target",
            max_training_rows=args.max_training_rows,
            random_state=config.random_seed,
            include_block_balanced_boosting=(horizon == 1),
        )
        metrics = result.metrics.copy()
        metrics["scenario"] = "temporal"
        metrics["horizon_days"] = horizon
        prediction = result.test_predictions.copy()
        prediction["scenario"] = "temporal"
        prediction["horizon_days"] = horizon
        metric_frames.append(metrics)
        prediction_frames.append(prediction)
        summary[f"temporal_h{horizon}"] = result.summary
        if horizon == 1:
            for method in result.summary["thresholds_calibrated"]:
                table = grouped_metric_summary(
                    prediction,
                    group_col="spatial_block",
                    label_col="target",
                    probability_col=f"{method}_calibrated_probability",
                    threshold=result.summary["thresholds_calibrated"][method],
                )
                table["model"] = method
                table["scenario"] = "temporal"
                table["grouping"] = "spatial_block"
                group_frames.append(table)
                year_prediction = prediction.assign(year=pd.to_datetime(prediction.date).dt.year)
                by_year = grouped_metric_summary(
                    year_prediction,
                    group_col="year",
                    label_col="target",
                    probability_col=f"{method}_calibrated_probability",
                    threshold=result.summary["thresholds_calibrated"][method],
                )
                by_year["model"] = method
                by_year["scenario"] = "temporal"
                by_year["grouping"] = "year"
                group_frames.append(by_year)
        print(f"Temporal horizon {horizon}: selected {result.summary['best_validation_model']} on validation", flush=True)

    if not args.skip_spatial:
        target_col = "target_h1"
        usable = frame.loc[frame[target_col].notna()].copy()
        available_blocks = usable["spatial_block"].nunique()
        n_folds = min(args.spatial_folds, available_blocks)
        folds = make_spatial_block_folds(usable, n_splits=n_folds, seed=config.random_seed)
        spatial_summary = []
        for fold_index, held_out in enumerate(folds, start=1):
            partitions = buffered_spatiotemporal_split(
                usable,
                config,
                held_out_blocks=held_out,
                buffer_degrees=1.0,
            )
            result = run_model_comparison(
                partitions["train"].rename(columns={target_col: "target"}),
                partitions["validation"].rename(columns={target_col: "target"}),
                partitions["test"].rename(columns={target_col: "target"}),
                target_col="target",
                max_training_rows=args.max_training_rows,
                random_state=config.random_seed + fold_index,
                include_block_balanced_boosting=True,
            )
            metrics = result.metrics.copy()
            metrics["scenario"] = "spatial_transfer"
            metrics["horizon_days"] = 1
            metrics["spatial_fold"] = fold_index
            metrics["held_out_blocks"] = ";".join(held_out)
            metric_frames.append(metrics)
            prediction = result.test_predictions.copy()
            prediction["scenario"] = "spatial_transfer"
            prediction["horizon_days"] = 1
            prediction["spatial_fold"] = fold_index
            prediction_frames.append(prediction)
            for method, threshold in result.summary["thresholds_calibrated"].items():
                block_metrics = grouped_metric_summary(
                    prediction,
                    group_col="spatial_block",
                    label_col="target",
                    probability_col=f"{method}_calibrated_probability",
                    threshold=threshold,
                )
                block_metrics["model"] = method
                block_metrics["scenario"] = "spatial_transfer"
                block_metrics["grouping"] = "spatial_block"
                block_metrics["spatial_fold"] = fold_index
                group_frames.append(block_metrics)
            spatial_summary.append(
                {
                    "fold": fold_index,
                    "held_out_blocks": list(held_out),
                    "train_rows": len(partitions["train"]),
                    "validation_rows": len(partitions["validation"]),
                    "test_rows": len(partitions["test"]),
                    **result.summary,
                }
            )
            print(
                f"Spatial fold {fold_index}/{n_folds}: {len(partitions['test']):,} held-out test rows; "
                f"validation selected {result.summary['best_validation_model']}",
                flush=True,
            )
        summary["spatial_transfer_h1"] = spatial_summary

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_metrics.to_csv(reports / "metrics.csv", index=False)
    all_predictions.to_csv(reports / "test_predictions.csv.gz", index=False, compression="gzip")
    if group_frames:
        all_groups = pd.concat(group_frames, ignore_index=True)
        all_groups.to_csv(reports / "group_metrics.csv", index=False)
        intervals = []
        for (scenario, grouping, model), values in all_groups.groupby(
            ["scenario", "grouping", "model"], sort=True
        ):
            interval = bootstrap_group_metric_means(
                values,
                ["average_precision", "brier_score", "event_recall", "false_alarm_rate"],
                n_bootstrap=2000,
                seed=config.random_seed,
            )
            interval["scenario"] = scenario
            interval["grouping"] = grouping
            interval["model"] = model
            intervals.append(interval)
        pd.concat(intervals, ignore_index=True).to_csv(
            reports / "group_macro_bootstrap_intervals.csv", index=False
        )
        paired_intervals = []
        for (scenario, grouping), values in all_groups.groupby(
            ["scenario", "grouping"], sort=True
        ):
            paired = bootstrap_paired_group_differences(
                values,
                group_col=grouping,
                metric_columns=["average_precision", "brier_score", "event_recall", "false_alarm_rate"],
                baseline_model="climatology",
                n_bootstrap=2000,
                seed=config.random_seed,
            )
            paired["scenario"] = scenario
            paired["grouping"] = grouping
            paired_intervals.append(paired)
        pd.concat(paired_intervals, ignore_index=True).to_csv(
            reports / "paired_model_vs_climatology_bootstrap_intervals.csv", index=False
        )
    temporal_h1_predictions = all_predictions.loc[
        (all_predictions.scenario == "temporal") & (all_predictions.horizon_days == 1)
    ]
    _plot_temporal_test(
        temporal_h1_predictions,
        reports / "temporal_test_precision_recall_calibration.png",
    )
    (reports / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved benchmark results under {reports}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
