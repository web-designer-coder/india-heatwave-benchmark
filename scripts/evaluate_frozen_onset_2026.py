"""Score a frozen onset estimator and development-fitted baselines on 2026.

The HGB estimator, features, calibration, and threshold are loaded unchanged.
The added climatology comparator is fitted on 1981--2007 and calibrated and
thresholded on 2008--2013 only. Neither comparator uses 2026 outcomes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heatwave_benchmark.data import (  # noqa: E402
    IMD_GRID_SHAPE,
    load_imd_terrain_mask,
)
from heatwave_benchmark.features import add_temperature_anomaly_features  # noqa: E402
from heatwave_benchmark.labels import compute_daily_normals  # noqa: E402
from heatwave_benchmark.models import apply_platt_calibrator  # noqa: E402
from heatwave_benchmark.models import ClimatologyModel, fit_platt_calibrator  # noqa: E402
from heatwave_benchmark.metrics import threshold_at_max_false_alarm_rate  # noqa: E402
from heatwave_benchmark.pipeline import (  # noqa: E402
    assemble_era5_dataset,
    build_benchmark_frame,
)

ARTIFACT = ROOT / "reports" / "onset_model_era5" / "onset_model_frozen.joblib"
POINTS_PATH = ROOT / "data" / "interim" / "analysis_grid_points.csv"
HISTORICAL_TMAX_PATH = ROOT / "data" / "interim" / "imd_tmax_1981_2025.nc"
IMD_DIR = ROOT / "data" / "raw" / "imd_realtime_2026"
IMD_MANIFEST = IMD_DIR / "download_manifest.json"
ERA5_DIR = ROOT / "data" / "raw" / "era5_onset_independent_2026"
ERA5_SEASONAL_DIR = ROOT / "tmp" / "era5_onset_independent_2026_seasonal"
OUTPUT = ROOT / "reports" / "independent_onset_2026"
START = pd.Timestamp("2026-02-22")
END = pd.Timestamp("2026-06-30")
ISSUE_START = pd.Timestamp("2026-03-01")
FROZEN_THRESHOLD = 0.014854158899725712


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_realtime_tmax(points: pd.DataFrame) -> xr.DataArray:
    manifest = json.loads(IMD_MANIFEST.read_text(encoding="utf-8"))
    records = {row["date"]: row for row in manifest["files"]}
    expected_dates = pd.date_range(START, END, freq="D")
    grids: list[np.ndarray] = []
    for day in expected_dates:
        record = records.get(day.date().isoformat())
        if record is None:
            if day >= ISSUE_START:
                raise ValueError(f"missing IMD real-time target for {day.date()}")
            grids.append(np.full(IMD_GRID_SHAPE, np.nan, dtype=np.float32))
            continue
        path = IMD_DIR / record["file"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"missing or changed IMD real-time file for {day.date()}")
        values = np.fromfile(path, dtype="<f4")
        if values.size != int(np.prod(IMD_GRID_SHAPE)):
            raise ValueError(f"bad IMD grid shape for {day.date()}")
        values = values.reshape(IMD_GRID_SHAPE)
        values[~np.isfinite(values) | np.isclose(values, 99.9)] = np.nan
        grids.append(values)
    data = np.stack(grids, axis=0)
    if data.shape[1:] != IMD_GRID_SHAPE:
        raise ValueError("daily IMD fields do not share the expected 31x31 grid")
    return xr.DataArray(
        data,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": expected_dates,
            "latitude": np.arange(7.5, 38.0, 1.0),
            "longitude": np.arange(67.5, 98.0, 1.0),
        },
        name="tmax",
        attrs={"units": "degree_Celsius", "source": "IMD real-time 1-degree Tmax"},
    )


def create_seasonal_era5_cache(points: pd.DataFrame) -> Path:
    """Trim the padded annual JSON caches to the complete 2026 study window."""
    expected = pd.date_range(START, END, freq="D")
    ERA5_SEASONAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = ERA5_DIR / "download_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str)
    if len(manifest) != 275 or manifest["grid_id"].nunique() != 275:
        raise ValueError("ERA5 per-cell manifest does not cover the frozen 275-cell grid")
    manifest_rows = manifest.set_index("grid_id").to_dict(orient="index")
    for record in points.sort_values("grid_id", kind="stable").itertuples(index=False):
        source = ERA5_DIR / f"{record.grid_id}.json"
        acquisition = manifest_rows.get(str(record.grid_id))
        if acquisition is None or acquisition.get("source_snapshot_id") != "TZ2KX6A24CJ5DG17T7MG":
            raise ValueError(f"missing or mismatched ERA5 manifest entry for {record.grid_id}")
        if not source.is_file() or sha256(source) != acquisition.get("sha256"):
            raise ValueError(f"ERA5 cache checksum mismatch for {record.grid_id}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("source_metadata", {}).get("snapshot_id") != "TZ2KX6A24CJ5DG17T7MG":
            raise ValueError(f"snapshot mismatch in ERA5 cache for {record.grid_id}")
        daily = payload["daily"]
        dates = pd.DatetimeIndex(pd.to_datetime(daily["time"], errors="raise"))
        keep = np.flatnonzero((dates >= START) & (dates <= END))
        if not dates[keep].equals(expected):
            raise ValueError(f"incomplete ERA5 window for {record.grid_id}")
        trimmed = dict(payload)
        trimmed_daily = {key: [values[int(i)] for i in keep] for key, values in daily.items()}
        trimmed["daily"] = trimmed_daily
        (ERA5_SEASONAL_DIR / source.name).write_text(
            json.dumps(trimmed, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
        )
    manifest = ERA5_DIR / "source_manifest.json"
    (ERA5_SEASONAL_DIR / "source_manifest.json").write_bytes(manifest.read_bytes())
    return ERA5_SEASONAL_DIR


def onset_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = probability >= threshold
    positive = y == 1
    negative = ~positive
    return {
        "rows": int(y.size),
        "events": int(positive.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "precision_at_frozen_threshold": float((predicted & positive).sum() / predicted.sum()) if predicted.any() else 0.0,
        "recall_at_frozen_threshold": float((predicted & positive).sum() / positive.sum()) if positive.any() else float("nan"),
        "false_alarm_rate_at_frozen_threshold": float((predicted & negative).sum() / negative.sum()) if negative.any() else float("nan"),
        "predicted_positive_rows": int(predicted.sum()),
    }


def moving_block_intervals(
    frame: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
    block_days: int = 7,
    replicates: int = 2000,
    seed: int = 20261027,
) -> pd.DataFrame:
    dates = np.array(sorted(frame["date"].drop_duplicates().to_numpy(dtype="datetime64[ns]")))
    per_date_positions = {day: np.flatnonzero(frame["date"].to_numpy(dtype="datetime64[ns]") == day) for day in dates}
    rng = np.random.default_rng(seed)
    values: list[dict[str, float]] = []
    n_dates = len(dates)
    if n_dates < block_days:
        raise ValueError("too few test dates for the prespecified 7-day block bootstrap")
    for _ in range(replicates):
        sampled: list[np.datetime64] = []
        while len(sampled) < n_dates:
            start = int(rng.integers(0, n_dates - block_days + 1))
            sampled.extend(dates[start : start + block_days])
        positions = np.concatenate([per_date_positions[day] for day in sampled[:n_dates]])
        y = frame.iloc[positions]["target_h1"].to_numpy(dtype=np.int8)
        p = probability[positions]
        result = onset_metrics(y, p, threshold)
        values.append({key: float(result[key]) for key in (
            "average_precision", "brier_score", "precision_at_frozen_threshold",
            "recall_at_frozen_threshold", "false_alarm_rate_at_frozen_threshold",
        )})
    boot = pd.DataFrame(values)
    return pd.DataFrame([
        {"metric": column, "lower_95": float(boot[column].quantile(0.025)),
         "upper_95": float(boot[column].quantile(0.975)), "replicates": replicates,
         "resampling": f"{block_days}-day moving blocks of calendar dates"}
        for column in boot.columns
    ])


def paired_moving_block_differences(
    frame: pd.DataFrame,
    candidate_probability: np.ndarray,
    candidate_threshold: float,
    reference_probability: np.ndarray,
    reference_threshold: float,
    reference_name: str,
    block_days: int = 7,
    replicates: int = 2000,
    seed: int = 20261028,
) -> pd.DataFrame:
    """Paired date-block intervals for candidate-minus-reference metrics."""
    dates = np.array(sorted(frame["date"].drop_duplicates().to_numpy(dtype="datetime64[ns]")))
    per_date_positions = {
        day: np.flatnonzero(frame["date"].to_numpy(dtype="datetime64[ns]") == day)
        for day in dates
    }
    if len(dates) < block_days:
        raise ValueError("too few test dates for the prespecified 7-day block bootstrap")
    rng = np.random.default_rng(seed)
    metrics = (
        "average_precision",
        "brier_score",
        "precision_at_frozen_threshold",
        "recall_at_frozen_threshold",
        "false_alarm_rate_at_frozen_threshold",
    )
    differences: list[dict[str, float]] = []
    for _ in range(replicates):
        sampled: list[np.datetime64] = []
        while len(sampled) < len(dates):
            start = int(rng.integers(0, len(dates) - block_days + 1))
            sampled.extend(dates[start : start + block_days])
        positions = np.concatenate([per_date_positions[day] for day in sampled[: len(dates)]])
        sampled_frame = frame.iloc[positions]
        y = sampled_frame["target_h1"].to_numpy(dtype=np.int8)
        candidate = onset_metrics(y, candidate_probability[positions], candidate_threshold)
        reference = onset_metrics(y, reference_probability[positions], reference_threshold)
        differences.append({key: float(candidate[key]) - float(reference[key]) for key in metrics})
    values = pd.DataFrame(differences)
    return pd.DataFrame([
        {
            "reference_model": reference_name,
            "metric": metric,
            "difference_candidate_minus_reference": float(values[metric].mean()),
            "lower_95": float(values[metric].quantile(0.025)),
            "upper_95": float(values[metric].quantile(0.975)),
            "replicates": replicates,
            "resampling": f"paired {block_days}-day moving blocks of issue dates",
        }
        for metric in metrics
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT)
    args = parser.parse_args()
    artifact_path = args.artifact
    output_dir = args.output_directory
    output_dir.mkdir(parents=True, exist_ok=True)
    if not artifact_path.is_file() or not IMD_MANIFEST.is_file():
        raise FileNotFoundError("frozen model or IMD real-time manifest is missing")
    points = pd.read_csv(POINTS_PATH)
    if len(points) != 275 or points["grid_id"].astype(str).nunique() != 275:
        raise ValueError("expected the frozen 275 unique India grid cells")
    bundle = joblib.load(artifact_path)
    if bundle.get("protocol_status", "").find("frozen") < 0:
        raise ValueError("model artifact does not declare the frozen protocol")
    if not np.isclose(float(bundle["probability_threshold"]), FROZEN_THRESHOLD, rtol=0, atol=1e-15):
        raise ValueError("artifact threshold differs from the prespecified threshold")
    era5_manifest = json.loads((ERA5_DIR / "source_manifest.json").read_text(encoding="utf-8"))
    expected_snapshot = str(bundle["predictor_snapshot_id"])
    if str(era5_manifest.get("snapshot_id")) != expected_snapshot:
        raise ValueError("independent ERA5 input does not match the frozen predictor snapshot")

    historical = xr.load_dataset(HISTORICAL_TMAX_PATH, engine="scipy")["tmax"]
    reference = historical.sel(time=slice("1981-01-01", "2010-12-31"))
    normals = compute_daily_normals(reference, 1981, 2010, window_days=15, min_years=10)
    tmax_2026 = read_realtime_tmax(points)
    terrain = load_imd_terrain_mask()
    seasonal_era5_dir = create_seasonal_era5_cache(points)
    meteorology = assemble_era5_dataset(
        seasonal_era5_dir, points, START.date().isoformat(), END.date().isoformat()
    )
    frame = build_benchmark_frame(
        meteorology,
        tmax_2026,
        terrain,
        normals,
        forecast_horizons=(1,),
        max_lag_days=7,
        issue_months=(3, 4, 5, 6),
    )
    frame = add_temperature_anomaly_features(frame, bundle["temperature_normals"])
    frame = frame.loc[frame["date"].ge(ISSUE_START)].copy()
    features = list(bundle["feature_columns"])
    known = frame["target_h1"].notna() & frame["current_heatwave"].notna()
    finite = np.isfinite(frame[features].to_numpy(dtype=float)).all(axis=1)
    # Preserve frozen scores for every eligible grid/date, including rows whose
    # IMD grid label is unavailable or whose grid-day is active. This enables
    # a separate station-observation audit to use station-defined onset state.
    score_frame = frame.loc[finite].copy()
    score_raw = bundle["estimator"].predict_proba(
        score_frame[features].to_numpy(dtype=float)
    )[:, 1]
    score_hgb = apply_platt_calibrator(bundle["calibrator"], score_raw)
    known_onset = known & frame["current_heatwave"].eq(0)
    onset = frame.loc[known_onset & finite].copy()
    if onset.empty or onset["target_h1"].nunique() != 2:
        raise ValueError("independent 2026 onset test must contain both outcome classes")
    y = onset["target_h1"].to_numpy(dtype=np.int8)
    raw = bundle["estimator"].predict_proba(onset[features].to_numpy(dtype=float))[:, 1]
    probability = apply_platt_calibrator(bundle["calibrator"], raw)
    threshold = float(bundle["probability_threshold"])
    metrics = onset_metrics(y, probability, threshold)

    # A fixed seasonal/location climatology is fitted and calibrated only on
    # historical development data, then scored on the same untouched rows.
    # This adds an independent reference without changing the frozen HGB.
    historical_features = pd.read_csv(
        ROOT / "data" / "interim" / "benchmark_features_era5.csv.gz",
        parse_dates=["date"],
        usecols=["date", "grid_id", "current_heatwave", "target_h1"],
    )
    historical_known = historical_features.loc[
        historical_features.target_h1.notna()
        & historical_features.current_heatwave.notna()
        & historical_features.current_heatwave.eq(0)
    ].copy()
    training = historical_known.loc[historical_known.date.dt.year.between(1981, 2007)]
    validation = historical_known.loc[historical_known.date.dt.year.between(2008, 2013)]
    climatology = ClimatologyModel().fit(training, target_col="target_h1")
    validation_raw = climatology.predict_proba(validation)
    validation_y = validation.target_h1.to_numpy(dtype=np.int8)
    climatology_calibrator = fit_platt_calibrator(validation_y, validation_raw)
    validation_calibrated = apply_platt_calibrator(climatology_calibrator, validation_raw)
    climatology_threshold = threshold_at_max_false_alarm_rate(
        validation_y, validation_calibrated, max_false_alarm_rate=0.10
    )
    # Export scores for all eligible grid/date rows so external station labels
    # can be evaluated without conditioning on the IMD grid label/state.
    score_climatology_raw = climatology.predict_proba(score_frame)
    score_climatology = apply_platt_calibrator(
        climatology_calibrator, score_climatology_raw
    )
    all_grid_scores = score_frame[["date", "grid_id"]].copy()
    all_grid_scores["hgb_calibrated_probability"] = score_hgb
    all_grid_scores["hgb_frozen_prediction"] = (
        score_hgb >= float(bundle["probability_threshold"])
    ).astype(np.int8)
    all_grid_scores["climatology_calibrated_probability"] = score_climatology
    all_grid_scores["climatology_threshold"] = float(climatology_threshold)
    all_grid_scores.to_csv(
        output_dir / "frozen_model_all_grid_scores_2026.csv.gz",
        index=False,
        compression="gzip",
    )
    climatology_raw_2026 = climatology.predict_proba(onset)
    climatology_probability = apply_platt_calibrator(
        climatology_calibrator, climatology_raw_2026
    )
    climatology_metrics = onset_metrics(
        y, climatology_probability, climatology_threshold
    )
    persistence_probability = np.zeros(y.shape, dtype=float)
    persistence_metrics = onset_metrics(y, persistence_probability, threshold)
    metrics["same_rows_baselines"] = {
        "climatology": {
            **climatology_metrics,
            "threshold": float(climatology_threshold),
            "calibration": "Platt fitted on 2008-2013 issue-state-zero rows",
            "threshold_selection": "validation rows, maximum 10% false-alarm rate",
        },
        "persistence": {
            **persistence_metrics,
            "threshold": threshold,
            "definition": "zero probability for all issue-state-zero onset rows",
        },
    }
    metrics.update({
        "candidate_issue_rows": int(len(frame)),
        "known_target_and_issue_state_rows": int(known.sum()),
        "known_issue_state_zero_rows": int(known_onset.sum()),
        "onset_rows_excluded_for_nonfinite_model_features": int((known_onset & ~finite).sum()),
        "active_issue_state_rows_excluded_by_onset_target": int((known & frame["current_heatwave"].eq(1)).sum()),
        "start_date": str(onset["date"].min().date()),
        "end_date": str(onset["date"].max().date()),
        "unique_issue_dates": int(onset["date"].nunique()),
        "unique_grid_cells": int(onset["grid_id"].nunique()),
        "frozen_threshold": threshold,
        "frozen_artifact_sha256": sha256(artifact_path),
        "predictor_snapshot_id": expected_snapshot,
        "evaluation_scope": "independent 2026 retrospective reanalysis temporal evaluation; not operational forecast verification",
        "protocol_note": "The frozen HGB estimator, features, calibrator, and threshold were unchanged. The added climatology reference was fitted on 1981-2007 and calibrated/thresholded on 2008-2013; no 2026 outcome informed it.",
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = onset[["date", "grid_id", "spatial_block", "current_heatwave", "target_h1"]].copy()
    prediction["raw_probability"] = raw
    prediction["calibrated_probability"] = probability
    prediction["frozen_prediction"] = (probability >= threshold).astype(np.int8)
    prediction["climatology_calibrated_probability"] = climatology_probability
    prediction["climatology_prediction"] = (
        climatology_probability >= climatology_threshold
    ).astype(np.int8)
    prediction["persistence_probability"] = persistence_probability
    prediction.to_csv(output_dir / "frozen_onset_predictions_2026.csv", index=False)
    intervals = moving_block_intervals(onset.reset_index(drop=True), probability, threshold)
    intervals.to_csv(output_dir / "frozen_onset_2026_moving_block_intervals.csv", index=False)
    baseline_frame = onset.reset_index(drop=True)
    baseline_intervals = moving_block_intervals(
        baseline_frame, climatology_probability, climatology_threshold, seed=20261029
    )
    baseline_intervals.insert(0, "model", "climatology")
    baseline_intervals.to_csv(
        output_dir / "climatology_onset_2026_moving_block_intervals.csv", index=False
    )
    paired = pd.concat(
        [
            paired_moving_block_differences(
                baseline_frame,
                probability,
                threshold,
                climatology_probability,
                climatology_threshold,
                "climatology",
                seed=20261030,
            ),
            paired_moving_block_differences(
                baseline_frame,
                probability,
                threshold,
                persistence_probability,
                threshold,
                "persistence",
                seed=20261031,
            ),
        ],
        ignore_index=True,
    )
    paired.to_csv(output_dir / "frozen_onset_2026_paired_baseline_intervals.csv", index=False)
    (output_dir / "frozen_onset_2026_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Metrics: {output_dir / 'frozen_onset_2026_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
