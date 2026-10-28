"""Reconstruct the already-selected onset model and save a frozen artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from heatwave_benchmark.experiments import select_model_features  # noqa: E402
from heatwave_benchmark.features import (  # noqa: E402
    add_temperature_anomaly_features,
    fit_calendar_temperature_normals,
)
from heatwave_benchmark.transitions import fit_onset_model_search  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=PROJECT_ROOT / "data" / "interim" / "benchmark_features_era5.csv.gz")
    parser.add_argument("--era5-daily", type=Path, default=PROJECT_ROOT / "data" / "interim" / "era5_1981_2025.nc")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "reports" / "onset_model_era5" / "onset_summary.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "onset_model_era5" / "onset_model_frozen.joblib")
    args = parser.parse_args()
    for path in (args.features, args.era5_daily, args.summary):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    params = summary["selected_parameters"]
    expected_threshold = float(summary["final_test_metrics"]["validation_threshold"])
    frame = pd.read_csv(args.features, parse_dates=["date"])
    target = pd.to_numeric(frame["target_h1"], errors="coerce")
    state = pd.to_numeric(frame["current_heatwave"], errors="coerce")
    usable = frame.loc[target.notna() & state.notna()].copy()
    usable["target"] = target.loc[usable.index].astype(np.int8)
    usable["current_heatwave"] = state.loc[usable.index].astype(np.int8)

    with xr.open_dataset(args.era5_daily) as dataset:
        if "t2m_max" not in dataset or not {"time", "grid_id"}.issubset(dataset["t2m_max"].dims):
            raise ValueError("ERA5 daily dataset must contain t2m_max on time and grid_id dimensions")
        weather = dataset["t2m_max"].sel(time=slice("1981-01-01", "2007-12-31"))
        weather_frame = weather.to_dataframe(name="t2m_max").reset_index()
    weather_frame = weather_frame.rename(columns={"time": "date"}).dropna(subset=["t2m_max"])
    normals = fit_calendar_temperature_normals(weather_frame)

    training = usable.loc[usable["date"].dt.year.between(1981, 2007)].copy()
    validation = usable.loc[usable["date"].dt.year.between(2008, 2013)].copy()
    training = add_temperature_anomaly_features(training, normals)
    validation = add_temperature_anomaly_features(validation, normals)
    anomaly_columns = [
        "t2m_max_anom_lag_0", "t2m_max_anom_lag_3", "t2m_max_anom_lag_7",
        "t2m_max_anom_mean_3d", "t2m_max_anom_mean_7d", "t2m_max_anom_change_0_3",
    ]
    features = list(dict.fromkeys([*select_model_features(training), *anomaly_columns]))
    result = fit_onset_model_search(
        training,
        validation,
        validation,
        feature_columns=features,
        parameter_grid=[params],
        random_state=42,
    )
    actual_threshold = float(result.metrics["validation_threshold"])
    if not np.isclose(actual_threshold, expected_threshold, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"reconstructed threshold {actual_threshold:.16g} differs from frozen report {expected_threshold:.16g}"
        )

    bundle = {
        "format_version": 1,
        "estimator": result.model,
        "calibrator": result.calibrator,
        "temperature_normals": normals,
        "feature_columns": features,
        "selected_parameters": params,
        "probability_threshold": actual_threshold,
        "target_definition": "next-day heatwave label equals 1, conditional on current_heatwave=0",
        "training_years": [1981, 2007],
        "calibration_and_threshold_years": [2008, 2013],
        "normal_fit_years": [1981, 2007],
        "predictor_snapshot_id": "TZ2KX6A24CJ5DG17T7MG",
        "selection_folds": summary["selection_folds"],
        "protocol_status": "frozen after previous 2014-2025 temporal and spatial holdouts had been inspected; subsequent evaluation is exploratory",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)
    print(json.dumps({"output": str(args.output), "threshold": actual_threshold, "features": len(features)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
