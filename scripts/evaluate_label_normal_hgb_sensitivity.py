"""Refit HGB and reference baselines under alternative IMD normal periods.

This is a descriptive sensitivity on the previously inspected 2014--2025
test period. It does not provide confirmatory evidence.
"""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from heatwave_benchmark.config import ExperimentConfig  # noqa: E402
from heatwave_benchmark.data import load_imd_terrain_mask  # noqa: E402
from heatwave_benchmark.experiments import run_model_comparison  # noqa: E402
from heatwave_benchmark.labels import compute_daily_normals  # noqa: E402
from heatwave_benchmark.pipeline import build_benchmark_frame  # noqa: E402
from heatwave_benchmark.splits import temporal_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    reports = PROJECT_ROOT / "reports" / "generated_era5"
    summary_path = reports / "benchmark_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["data"]["predictor_source"] != "era5":
        raise ValueError("This sensitivity must use the primary ERA5 predictor source")

    tmax_path = PROJECT_ROOT / "data" / "interim" / "imd_tmax_1981_2025.nc"
    predictor_path = PROJECT_ROOT / summary["data"]["predictor_netcdf"]
    tmax = xr.load_dataset(tmax_path, engine="scipy")["tmax"]
    meteorology = xr.load_dataset(predictor_path, engine="scipy")
    terrain = load_imd_terrain_mask()

    config_values = summary["config"]
    config = ExperimentConfig(
        label_normal_years=(1981, 2010),
        train_years=tuple(config_values["train_years"]),
        validation_years=tuple(config_values["validation_years"]),
        test_years=tuple(config_values["test_years"]),
        forecast_horizons=(1,),
        max_lag_days=int(config_values["max_lag_days"]),
        random_seed=int(config_values["random_seed"]),
    )
    max_training_rows = int(config_values["max_training_rows"])
    max_false_alarm_rate = float(
        config_values["threshold_max_validation_false_alarm_rate"]
    )

    metrics: list[pd.DataFrame] = []
    run_records: list[dict[str, object]] = []
    scenarios = ((1981, 2000), (1981, 2010), (1991, 2010))
    for normal_start, normal_end in scenarios:
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
            max_lag_days=config.max_lag_days,
        )
        partitions = temporal_split(frame.loc[frame["target_h1"].notna()].copy(), config)
        renamed = {
            name: partition.rename(columns={"target_h1": "target"})
            for name, partition in partitions.items()
        }
        result = run_model_comparison(
            renamed["train"],
            renamed["validation"],
            renamed["test"],
            target_col="target",
            max_training_rows=max_training_rows,
            max_false_alarm_rate=max_false_alarm_rate,
            random_state=config.random_seed,
            estimator_names=("hist_gradient_boosting",),
        )
        metrics_frame = result.metrics.copy()
        metrics_frame["label_normal_period"] = f"{normal_start}-{normal_end}"
        metrics.append(metrics_frame)
        run_records.append(
            {
                "label_normal_period": [normal_start, normal_end],
                "train_rows": result.summary["n_training_rows"],
                "validation_rows": result.summary["n_validation_rows"],
                "test_rows": result.summary["n_test_rows"],
                "positive_test_events": int(renamed["test"]["target"].sum()),
                "test_prevalence": float(renamed["test"]["target"].mean()),
                "best_validation_model": result.summary["best_validation_model"],
                "hgb_threshold_calibrated": result.summary[
                    "thresholds_calibrated"
                ].get("hist_gradient_boosting"),
            }
        )
        print(
            f"Completed {normal_start}--{normal_end}: "
            f"{result.summary['n_test_rows']:,} rows; "
            f"{int(renamed['test']['target'].sum()):,} events",
            flush=True,
        )

    output = args.output_directory or reports / "label_normal_hgb_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(metrics, ignore_index=True).to_csv(output / "metrics.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "status": "exploratory; the 2014--2025 test period had been inspected",
                "predictor_source": "ERA5",
                "estimators": ["hist_gradient_boosting", "climatology", "persistence"],
                "validation_only_calibration_and_threshold_selection": True,
                "runs": run_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved label-normal HGB sensitivity results to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
