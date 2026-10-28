"""Build a separate percentile-duration heatwave target for sensitivity checks.

This target follows Dar and Apurv (2024)'s relative-threshold form: Tmax above
the local 90th percentile based on a centered 31-day window, with events
restricted to runs of at least three consecutive days. The reference period is
fixed here to 1981--2010 for reproducibility; this is not an official IMD
declaration target.
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from heatwave_benchmark.data import load_imd_terrain_mask  # noqa: E402
from heatwave_benchmark.config import ExperimentConfig  # noqa: E402
from heatwave_benchmark.experiments import run_model_comparison  # noqa: E402
from heatwave_benchmark.pipeline import select_analysis_points  # noqa: E402
from heatwave_benchmark.splits import temporal_split  # noqa: E402


def percentile90_thresholds(
    field: xr.DataArray,
    reference_start: int = 1981,
    reference_end: int = 2010,
    window_days: int = 31,
) -> xr.DataArray:
    """Compute per-cell circular day-of-year 90th-percentile thresholds."""
    if window_days % 2 != 1:
        raise ValueError("window_days must be odd")
    dates = pd.DatetimeIndex(pd.to_datetime(field.time.values))
    reference = field.sel(time=slice(f"{reference_start}-01-01", f"{reference_end}-12-31"))
    if reference.sizes.get("time", 0) == 0:
        raise ValueError("reference period is absent from Tmax")
    ref_dates = pd.DatetimeIndex(pd.to_datetime(reference.time.values))
    spatial_dims = tuple(dim for dim in reference.dims if dim != "time")
    ref_values = reference.transpose("time", *spatial_dims).values.astype(float)
    ref_doy = ref_dates.day_of_year.to_numpy()
    thresholds = np.full((366, *ref_values.shape[1:]), np.nan)
    half = window_days // 2
    for doy in range(1, 367):
        window = ((ref_doy - doy + 183) % 366) - 183
        values = ref_values[np.abs(window) <= half]
        with np.errstate(all="ignore"):
            thresholds[doy - 1] = np.nanpercentile(values, 90, axis=0)
    return xr.DataArray(
        thresholds,
        dims=("dayofyear", *spatial_dims),
        coords={
            "dayofyear": np.arange(1, 367),
            **{dim: reference.coords[dim] for dim in spatial_dims if dim in reference.coords},
        },
        name="tmax_q90_31day",
        attrs={"reference_years": f"{reference_start}-{reference_end}", "window_days": window_days},
    )


def three_day_events(field: xr.DataArray, thresholds: xr.DataArray) -> xr.DataArray:
    """Mark days in above-threshold runs of at least three consecutive days."""
    spatial_dims = tuple(dim for dim in field.dims if dim != "time")
    ordered = field.transpose("time", *spatial_dims)
    dates = pd.DatetimeIndex(pd.to_datetime(ordered.time.values))
    q = thresholds.values[dates.day_of_year.to_numpy() - 1]
    exceed = np.isfinite(ordered.values) & np.isfinite(q) & (ordered.values > q)
    result = np.full(exceed.shape, np.nan, dtype=float)
    known = np.isfinite(ordered.values) & np.isfinite(q)
    result[known] = 0.0
    # Mark all days belonging to a run of length >= 3; missing days break runs.
    # A day belongs to a qualifying run iff some 3-day window containing it
    # consists entirely of exceedances. The array is small (275 selected cells).
    for start in range(exceed.shape[0] - 2):
        run = exceed[start] & exceed[start + 1] & exceed[start + 2]
        result[start][run] = 1.0
        result[start + 1][run] = 1.0
        result[start + 2][run] = 1.0
    return xr.DataArray(
        result,
        dims=ordered.dims,
        coords=ordered.coords,
        name="percentile_heatwave_event",
        attrs={"definition": "Tmax > local centered 31-day q90 on >=3 consecutive days"},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    tmax_path = PROJECT_ROOT / "data/interim/imd_tmax_1981_2025.nc"
    out_dir = args.output_directory or PROJECT_ROOT / "reports/percentile_label_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    field = xr.load_dataset(tmax_path, engine="scipy")["tmax"]
    terrain = load_imd_terrain_mask()
    points = select_analysis_points(
        field, terrain, observation_period=("1981-01-01", "2025-12-31")
    )
    lat = xr.DataArray(points.latitude.to_numpy(), dims="cell")
    lon = xr.DataArray(points.longitude.to_numpy(), dims="cell")
    selected = field.sel(latitude=lat, longitude=lon).assign_coords(cell=points.grid_id.to_numpy())
    thresholds = percentile90_thresholds(selected)
    event = three_day_events(selected, thresholds)
    stacked = event.to_dataframe(name="percentile_event").reset_index()
    stacked = stacked.rename(columns={"cell": "grid_id"})
    stacked["date"] = pd.to_datetime(stacked.time) - pd.Timedelta(days=1)
    stacked = stacked.drop(columns="time")
    stacked = stacked.rename(columns={"percentile_event": "alternative_target_h1"})
    primary = pd.read_csv(
        PROJECT_ROOT / "data/interim/benchmark_features_era5.csv.gz",
        usecols=["date", "grid_id", "target_h1"],
        parse_dates=["date"],
    )
    primary = primary.merge(stacked, on=["date", "grid_id"], how="left", validate="one_to_one")
    test = primary.loc[
        primary.date.dt.year.between(2014, 2025)
        & primary.target_h1.notna()
        & primary.alternative_target_h1.notna()
    ]
    report = {
        "definition": "Local Tmax > 90th percentile from centered 31-day window, reference 1981-2010, and >=3 consecutive days; a sensitivity target, not official IMD declaration.",
        "reference_period": [1981, 2010],
        "window_days": 31,
        "minimum_duration_days": 3,
        "temporal_test_rows": int(len(test)),
        "temporal_test_cells": int(test.grid_id.nunique()),
        "temporal_test_percentile_event_days": int(test.alternative_target_h1.sum()),
        "temporal_test_percentile_prevalence": float(test.alternative_target_h1.mean()),
        "temporal_test_primary_event_days": int(test.target_h1.sum()),
        "temporal_test_primary_prevalence": float(test.target_h1.mean()),
        "jaccard_overlap_on_test_rows": float(
            ((test.alternative_target_h1 == 1) & (test.target_h1 == 1)).sum()
            / ((test.alternative_target_h1 == 1) | (test.target_h1 == 1)).sum()
        ),
        "limitation": "This diagnostic quantifies target-definition sensitivity only; it does not refit and validate models against the alternative target.",
    }
    full = pd.read_csv(
        PROJECT_ROOT / "data/interim/benchmark_features_era5.csv.gz",
        parse_dates=["date"],
    ).drop(columns=["target_h1"])
    full = full.merge(
        primary[["date", "grid_id", "alternative_target_h1"]],
        on=["date", "grid_id"],
        how="inner",
        validate="one_to_one",
    ).rename(columns={"alternative_target_h1": "target"})
    full = full.loc[full.target.notna()].copy()
    partitions = temporal_split(
        full,
        ExperimentConfig(
            label_normal_years=(1981, 2010),
            train_years=(1981, 2007),
            validation_years=(2008, 2013),
            test_years=(2014, 2025),
        ),
    )
    # Refit the complete prespecified model suite to the alternative target.
    # This is a target-sensitivity analysis on the already inspected 2014--2025
    # period, not a confirmatory evaluation or an independent test.
    result = run_model_comparison(
        partitions["train"],
        partitions["validation"],
        partitions["test"],
        target_col="target",
        estimator_names=None,
        max_training_rows=200_000,
        max_false_alarm_rate=0.10,
        random_state=42,
    )
    report["alternative_target_validation_test_metrics"] = result.metrics.loc[
        result.metrics.partition.eq("test")
    ].to_dict(orient="records")
    report["alternative_target_model_evaluation_note"] = (
        "All supervised estimators in the benchmark suite were refit to this alternative target using the same chronological train/validation/test years. This is a post hoc target-sensitivity analysis on the already inspected 2014-2025 period, not a confirmatory test."
    )
    report["limitation"] = (
        "This sensitivity changes the target definition and refits the benchmark model suite, but uses the already inspected 2014-2025 period. It documents target dependence only and is not independent confirmation."
    )
    (out_dir / "label_overlap_metrics.json").write_text(
        __import__("json").dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    primary.to_csv(out_dir / "alternative_target_by_grid_day.csv.gz", index=False, compression="gzip")
    print(__import__("json").dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
