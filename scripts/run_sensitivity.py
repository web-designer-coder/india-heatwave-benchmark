"""Run predeclared label-normal and spatial-split sensitivity analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from heatwave_benchmark.config import ExperimentConfig  # noqa: E402
from heatwave_benchmark.artifacts import (  # noqa: E402
    benchmark_feature_frame_path,
    benchmark_output_directory,
)
from heatwave_benchmark.data import load_imd_terrain_mask  # noqa: E402
from heatwave_benchmark.experiments import run_model_comparison  # noqa: E402
from heatwave_benchmark.labels import compute_daily_normals  # noqa: E402
from heatwave_benchmark.pipeline import build_benchmark_frame  # noqa: E402
from heatwave_benchmark.splits import (  # noqa: E402
    add_spatial_block_ids,
    buffered_spatiotemporal_split,
    make_spatial_block_folds,
    temporal_split,
)


def _run_comparison(
    partitions: dict[str, pd.DataFrame],
    estimator_names: tuple[str, ...],
    max_training_rows: int,
    random_state: int,
    max_false_alarm_rate: float,
) -> object:
    renamed = {
        name: partition.rename(columns={"target_h1": "target"})
        for name, partition in partitions.items()
    }
    return run_model_comparison(
        renamed["train"],
        renamed["validation"],
        renamed["test"],
        target_col="target",
        max_training_rows=max_training_rows,
        max_false_alarm_rate=max_false_alarm_rate,
        random_state=random_state,
        estimator_names=estimator_names,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-training-rows", type=int, default=None)
    parser.add_argument("--spatial-folds", type=int, default=5)
    parser.add_argument(
        "--predictor-source", choices=("nasa_power", "era5"), default="nasa_power"
    )
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()

    benchmark_reports = benchmark_output_directory(args.predictor_source)
    reports = args.output_directory or benchmark_reports
    reports.mkdir(parents=True, exist_ok=True)
    summary_path = benchmark_reports / "benchmark_summary.json"
    primary_frame_path = benchmark_feature_frame_path(args.predictor_source)
    tmax_path = PROJECT_ROOT / "data" / "interim" / "imd_tmax_1981_2025.nc"
    missing = [path for path in (summary_path, primary_frame_path, tmax_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "run scripts/download_public_data.py and scripts/run_benchmark.py first; "
            f"missing: {[str(path.relative_to(PROJECT_ROOT)) for path in missing]}"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["data"]["predictor_source"] != args.predictor_source:
        raise ValueError("selected predictor source does not match the benchmark summary")
    predictor_path = PROJECT_ROOT / summary["data"]["predictor_netcdf"]
    if not predictor_path.is_file():
        raise FileNotFoundError(f"primary predictor file is missing: {predictor_path}")
    selected_model = str(summary["temporal_h1"]["best_validation_model"])
    if selected_model == "hist_gradient_boosting_block_balanced":
        selected_model = "hist_gradient_boosting"
    supported_estimators = {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "xgboost",
    }
    estimator_names = (selected_model,) if selected_model in supported_estimators else ()
    max_training_rows = args.max_training_rows or int(summary["config"]["max_training_rows"])
    max_false_alarm_rate = float(summary["config"]["threshold_max_validation_false_alarm_rate"])
    config = ExperimentConfig(
        label_normal_years=(1981, 2010),
        train_years=(1981, 2007),
        validation_years=(2008, 2013),
        test_years=(2014, 2025),
    )

    print(f"Sensitivity classifier selected on temporal validation: {selected_model}", flush=True)
    primary_frame = pd.read_csv(primary_frame_path, parse_dates=["date"])
    usable_primary = primary_frame.loc[primary_frame["target_h1"].notna()].copy()
    annual = usable_primary.assign(year=usable_primary.date.dt.year).groupby("year").agg(
        labeled_issue_rows=("target_h1", "size"),
        observed_grid_cells=("grid_id", "nunique"),
        events=("target_h1", "sum"),
        event_prevalence=("target_h1", "mean"),
    ).reset_index()
    annual["imd_coverage_change_2008"] = annual["year"].ge(2008)
    annual.to_csv(reports / "annual_target_coverage.csv", index=False)

    tmax = xr.load_dataset(tmax_path, engine="scipy")["tmax"]
    meteorology = xr.load_dataset(predictor_path, engine="scipy")
    terrain = load_imd_terrain_mask()
    metric_frames: list[pd.DataFrame] = []
    run_records: list[dict[str, object]] = []

    for normal_start, normal_end in ((1981, 2000), (1991, 2010)):
        normals = compute_daily_normals(
            tmax.sel(time=slice(f"{normal_start}-01-01", f"{normal_end}-12-31")),
            normal_start,
            normal_end,
            window_days=15,
            min_years=10,
        )
        frame = build_benchmark_frame(
            meteorology,
            tmax,
            terrain,
            normals,
            forecast_horizons=(1,),
            max_lag_days=7,
        )
        partitions = temporal_split(
            frame.loc[frame["target_h1"].notna()].copy(),
            config,
        )
        result = _run_comparison(
            partitions,
            estimator_names,
            max_training_rows,
            config.random_seed,
            max_false_alarm_rate,
        )
        metrics = result.metrics.copy()
        metrics["scenario"] = f"label_normal_{normal_start}_{normal_end}"
        metrics["horizon_days"] = 1
        metric_frames.append(metrics)
        run_records.append(
            {
                "scenario": f"label_normal_{normal_start}_{normal_end}",
                "normal_years": [normal_start, normal_end],
                "selected_model": selected_model,
                "best_validation_model": result.summary["best_validation_model"],
                "train_rows": result.summary["n_training_rows"],
                "validation_rows": result.summary["n_validation_rows"],
                "test_rows": result.summary["n_test_rows"],
            }
        )
        print(f"Completed label-normal sensitivity {normal_start}-{normal_end}", flush=True)

    for block_degrees in (3.0, 5.0, 7.0):
        blocked = add_spatial_block_ids(usable_primary, block_degrees=block_degrees)
        folds_count = min(args.spatial_folds, blocked["spatial_block"].nunique())
        folds = make_spatial_block_folds(blocked, n_splits=folds_count, seed=config.random_seed)
        for buffer_degrees in (0.0, 1.0, 2.0):
            scenario = f"spatial_block_{block_degrees:g}_buffer_{buffer_degrees:g}"
            for fold_index, held_out in enumerate(folds, start=1):
                record: dict[str, object] = {
                    "scenario": scenario,
                    "spatial_block_degrees": block_degrees,
                    "spatial_buffer_degrees": buffer_degrees,
                    "fold": fold_index,
                    "held_out_blocks": list(held_out),
                    "selected_model": selected_model,
                }
                try:
                    partitions = buffered_spatiotemporal_split(
                        blocked,
                        config,
                        held_out_blocks=held_out,
                        buffer_degrees=buffer_degrees,
                    )
                    result = _run_comparison(
                        partitions,
                        estimator_names,
                        max_training_rows,
                        config.random_seed + fold_index,
                        max_false_alarm_rate,
                    )
                except ValueError as exc:
                    record["status"] = "failed"
                    record["error"] = str(exc)
                    record["test_rows"] = 0
                    run_records.append(record)
                    print(f"Skipped {scenario} fold {fold_index}: {exc}", flush=True)
                    continue

                metrics = result.metrics.copy()
                metrics["scenario"] = scenario
                metrics["horizon_days"] = 1
                metrics["spatial_block_degrees"] = block_degrees
                metrics["spatial_buffer_degrees"] = buffer_degrees
                metrics["spatial_fold"] = fold_index
                metric_frames.append(metrics)
                record.update(
                    {
                        "status": "complete",
                        "best_validation_model": result.summary["best_validation_model"],
                        "train_rows": result.summary["n_training_rows"],
                        "validation_rows": result.summary["n_validation_rows"],
                        "test_rows": result.summary["n_test_rows"],
                    }
                )
                run_records.append(record)
                print(
                    f"Completed {scenario} fold {fold_index}/{folds_count}: "
                    f"{result.summary['n_test_rows']:,} test rows",
                    flush=True,
                )

    if metric_frames:
        pd.concat(metric_frames, ignore_index=True).to_csv(
            reports / "sensitivity_metrics.csv", index=False
        )
    (reports / "sensitivity_summary.json").write_text(
        json.dumps(
            {
                "selected_model_from_primary_temporal_validation": selected_model,
                "predictor_source": summary["data"]["predictor_source"],
                "sensitivity_estimator_names": list(estimator_names),
                "max_training_rows": max_training_rows,
                "max_validation_false_alarm_rate": max_false_alarm_rate,
                "sensitivity_runs": run_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved sensitivity results under {reports}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
