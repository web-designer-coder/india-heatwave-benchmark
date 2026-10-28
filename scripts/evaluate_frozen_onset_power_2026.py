"""Exploratory frozen-onset evaluation against NASA POWER product labels.

The estimator and all model settings remain frozen. POWER labels use POWER's
own 1981-2010 Tmax normals; this is a separate gridded analysis-product check,
not independent station-observation or operational forecast verification.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heatwave_benchmark.data import (  # noqa: E402
    NASA_POWER_DAILY_API,
    download_nasa_power_grid,
    parse_nasa_power_daily_response,
)
from heatwave_benchmark.labels import compute_daily_normals, make_daily_gridcell_labels  # noqa: E402

POINTS_PATH = ROOT / "data/interim/analysis_grid_points.csv"
HISTORICAL_POWER_PATH = ROOT / "data/interim/nasa_power_1981_2025.nc"
SCORES_PATH = ROOT / "reports/independent_onset_2026/frozen_model_all_grid_scores_2026.csv.gz"
FROZEN_ARTIFACT = ROOT / "reports/onset_model_era5/onset_model_frozen.joblib"
DATA_DIR = ROOT / "data/raw/nasa_power_2026_product_check"
OUTPUT_DIR = ROOT / "reports/independent_power_2026"
START = pd.Timestamp("2026-03-01")
END = pd.Timestamp("2026-07-01")  # verifying day for the 30 June issue row
ISSUE_END = pd.Timestamp("2026-06-30")
PARAMETERS = ("T2M_MAX",)
HGB_THRESHOLD = 0.014854158899725712
CLIMATOLOGY_THRESHOLD = 0.024442883933580933
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261028


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_values(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = probability >= threshold
    positive = y == 1
    negative = ~positive
    return {
        "rows": int(len(y)),
        "events": int(positive.sum()),
        "prevalence": float(np.mean(y)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "precision_at_frozen_threshold": float((predicted & positive).sum() / predicted.sum()) if predicted.any() else 0.0,
        "recall_at_frozen_threshold": float((predicted & positive).sum() / positive.sum()) if positive.any() else float("nan"),
        "false_alarm_rate_at_frozen_threshold": float((predicted & negative).sum() / negative.sum()) if negative.any() else float("nan"),
        "predicted_positive_rows": int(predicted.sum()),
    }


def paired_block_intervals(frame: pd.DataFrame) -> list[dict[str, object]]:
    dates = np.array(sorted(frame["date"].drop_duplicates().to_numpy(dtype="datetime64[ns]")))
    if len(dates) < 7:
        raise ValueError("fewer than seven distinct issue dates for the fixed block bootstrap")
    positions = {
        day: np.flatnonzero(frame["date"].to_numpy(dtype="datetime64[ns]") == day)
        for day in dates
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: list[dict[str, float]] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected: list[np.datetime64] = []
        while len(selected) < len(dates):
            start = int(rng.integers(0, len(dates) - 6))
            selected.extend(dates[start : start + 7])
        ix = np.concatenate([positions[day] for day in selected[: len(dates)]])
        rows = frame.iloc[ix]
        y = rows["target_h1"].to_numpy(dtype=np.int8)
        hgb = metric_values(y, rows["hgb_probability"].to_numpy(), HGB_THRESHOLD)
        climatology = metric_values(y, rows["climatology_probability"].to_numpy(), CLIMATOLOGY_THRESHOLD)
        samples.append({name: float(hgb[name]) - float(climatology[name]) for name in (
            "average_precision", "brier_score", "precision_at_frozen_threshold",
            "recall_at_frozen_threshold", "false_alarm_rate_at_frozen_threshold",
        )})
    values = pd.DataFrame(samples)
    return [
        {
            "metric": name,
            "lower_95": float(values[name].quantile(0.025)),
            "upper_95": float(values[name].quantile(0.975)),
            "replicates": BOOTSTRAP_REPLICATES,
            "resampling": "paired 7-day moving blocks of issue dates",
            "difference": "HGB minus fixed historical climatology",
        }
        for name in values.columns
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(POINTS_PATH, dtype={"grid_id": str}).sort_values("grid_id", kind="stable")
    if len(points) != 275 or points["grid_id"].duplicated().any():
        raise ValueError("expected the frozen set of 275 unique study cells")
    historical = xr.open_dataset(HISTORICAL_POWER_PATH, engine="scipy")
    historical_ids = historical.grid_id.values.astype(str)
    if set(historical_ids) != set(points.grid_id):
        raise ValueError("cached NASA POWER normals do not match the frozen study grid")
    historical = historical.sel(grid_id=points.grid_id.to_numpy())
    if not np.array_equal(pd.DatetimeIndex(historical.time.values), pd.date_range("1981-01-01", "2025-12-31", freq="D")):
        raise ValueError("cached NASA POWER data do not provide the complete expected daily period")

    manifest = download_nasa_power_grid(
        points[["grid_id", "latitude", "longitude"]],
        DATA_DIR,
        START.date().isoformat(),
        END.date().isoformat(),
        parameters=PARAMETERS,
        max_workers=5,
    )
    if len(manifest) != 275 or not set(manifest["status"]).issubset({"downloaded", "cached"}):
        raise ValueError("NASA POWER acquisition did not validate all 275 points")

    expected_dates = pd.date_range(START, END, freq="D")
    daily_frames: list[pd.DataFrame] = []
    source_headers: dict[str, object] = {}
    for row in points.itertuples(index=False):
        path = DATA_DIR / f"{row.grid_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        header = payload.get("header", {})
        source_headers[json.dumps(header.get("sources", []), sort_keys=True)] = {
            "sources": header.get("sources"),
            "api_version": header.get("api", {}).get("version"),
            "time_standard": header.get("time_standard"),
            "start": header.get("start"),
            "end": header.get("end"),
        }
        parsed = parse_nasa_power_daily_response(
            payload,
            grid_id=str(row.grid_id),
            latitude=float(row.latitude),
            longitude=float(row.longitude),
            parameters=PARAMETERS,
            expected_time_standard="LST",
        )
        if not pd.DatetimeIndex(parsed["date"]).equals(expected_dates):
            raise ValueError(f"incomplete NASA POWER dates for {row.grid_id}")
        if not np.allclose(parsed["api_response_latitude"], historical["api_response_latitude"].sel(grid_id=row.grid_id).item(), atol=1e-6):
            raise ValueError(f"NASA POWER response latitude changed for {row.grid_id}")
        if not np.allclose(parsed["api_response_longitude"], historical["api_response_longitude"].sel(grid_id=row.grid_id).item(), atol=1e-6):
            raise ValueError(f"NASA POWER response longitude changed for {row.grid_id}")
        daily_frames.append(parsed[["grid_id", "date", "t2m_max", "api_response_latitude", "api_response_longitude"]])
    fresh = pd.concat(daily_frames, ignore_index=True).sort_values(["grid_id", "date"], kind="stable")
    if fresh.duplicated(["grid_id", "date"]).any() or len(fresh) != len(points) * len(expected_dates):
        raise ValueError("NASA POWER 2026 cache has duplicate or missing cell-days")

    reference = historical["t2m_max"].sel(time=slice("1981-01-01", "2010-12-31"))
    normals = compute_daily_normals(reference, 1981, 2010, window_days=15, min_years=10)
    tmax = xr.DataArray(
        fresh.pivot(index="date", columns="grid_id", values="t2m_max").reindex(
            index=expected_dates, columns=points.grid_id
        ).to_numpy(dtype=float),
        dims=("time", "grid_id"),
        coords={"time": expected_dates, "grid_id": points.grid_id.to_numpy()},
        name="tmax",
        attrs={"units": "degree_Celsius", "source": "NASA POWER daily T2M_MAX; LST"},
    )
    terrain_map = {0: "plains", 1: "hilly", 2: "coastal"}
    terrain = points.terrain_code.map(terrain_map)
    if terrain.isna().any():
        raise ValueError("study grid contains an unknown terrain code")
    labels = make_daily_gridcell_labels(tmax, normals, terrain.to_numpy())

    score = pd.read_csv(SCORES_PATH, compression="gzip", parse_dates=["date"])
    score["grid_id"] = score.grid_id.astype(str)
    if len(score) != 122 * 275 or score.duplicated(["date", "grid_id"]).any():
        raise ValueError("frozen all-grid score file has unexpected coverage or duplicate rows")
    label_frame = pd.DataFrame(
        labels.values,
        index=pd.DatetimeIndex(labels.time.values),
        columns=labels.grid_id.values.astype(str),
    )
    label_frame.index.name = "date"
    label_frame.columns.name = "grid_id"
    label_long = label_frame.stack(future_stack=True).rename("power_label").reset_index()
    if "level_1" in label_long.columns:
        label_long = label_long.rename(columns={"level_1": "grid_id"})
    joined = score.merge(label_long, on=["date", "grid_id"], how="left", validate="one_to_one")
    future = label_long.rename(columns={"date": "future_date", "power_label": "future_label"})
    future["date"] = future["future_date"] - pd.Timedelta(days=1)
    joined = joined.merge(future[["date", "grid_id", "future_label"]], on=["date", "grid_id"], how="left", validate="one_to_one")
    eligible = joined.loc[
        joined["date"].between(START, ISSUE_END)
        & joined["power_label"].eq(0)
        & joined["future_label"].notna()
    ].copy()
    eligible["target_h1"] = (eligible["future_label"] >= 1).astype(np.int8)
    if len(eligible) == 0 or eligible["target_h1"].sum() == 0:
        raise ValueError("NASA POWER label join produced no eligible rows or no onset events")
    eligible = eligible.rename(columns={
        "hgb_calibrated_probability": "hgb_probability",
        "climatology_calibrated_probability": "climatology_probability",
    })
    probability_columns = ["hgb_probability", "climatology_probability"]
    if not np.isfinite(eligible[probability_columns].to_numpy(dtype=float)).all():
        raise ValueError("frozen probabilities contain non-finite values on eligible rows")

    hgb_metrics = metric_values(eligible.target_h1.to_numpy(dtype=np.int8), eligible.hgb_probability.to_numpy(), HGB_THRESHOLD)
    climate_metrics = metric_values(eligible.target_h1.to_numpy(dtype=np.int8), eligible.climatology_probability.to_numpy(), CLIMATOLOGY_THRESHOLD)
    paired = paired_block_intervals(eligible)
    eligible.to_csv(OUTPUT_DIR / "frozen_onset_power_predictions_2026.csv.gz", index=False, compression="gzip")
    npower_manifest = DATA_DIR / "download_manifest.csv"
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory product-level evaluation; post hoc product selection",
        "dataset": "NASA POWER daily T2M_MAX",
        "api": NASA_POWER_DAILY_API,
        "api_version_and_sources": list(source_headers.values()),
        "accessed_on": datetime.now(timezone.utc).date().isoformat(),
        "time_standard": "LST",
        "period": [START.date().isoformat(), END.date().isoformat()],
        "issue_period": [START.date().isoformat(), ISSUE_END.date().isoformat()],
        "study_cells": len(points),
        "historical_normal_source": str(HISTORICAL_POWER_PATH.relative_to(ROOT)),
        "historical_normal_period": ["1981-01-01", "2010-12-31"],
        "label_rule": "Existing IMD-aligned terrain and departure thresholds, applied to POWER T2M_MAX and POWER-derived normals; not an official IMD declaration.",
        "frozen_model_sha256": sha256(FROZEN_ARTIFACT),
        "frozen_all_grid_scores_sha256": sha256(SCORES_PATH),
        "frozen_hgb_threshold": HGB_THRESHOLD,
        "fixed_climatology_threshold": CLIMATOLOGY_THRESHOLD,
        "eligible_row_rule": "POWER issue-day label equals zero and next-day POWER label is known; target is next-day POWER label greater than or equal to one.",
        "hgb_metrics": hgb_metrics,
        "historical_climatology_metrics_same_rows": climate_metrics,
        "paired_hgb_minus_climatology_intervals": paired,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "acquisition_manifest": str(npower_manifest.relative_to(ROOT)),
        "acquisition_manifest_sha256": sha256(npower_manifest),
        "cache_directory": str(DATA_DIR.relative_to(ROOT)),
        "prediction_rows": str((OUTPUT_DIR / "frozen_onset_power_predictions_2026.csv.gz").relative_to(ROOT)),
        "limitations": [
            "NASA POWER is a gridded analysis product based on reanalysis/near-real-time inputs, not independent station observations.",
            "Potential overlap in assimilated observations with ERA5 or IMD cannot be excluded.",
            "POWER uses local solar time, which differs from the IMD observation window.",
            "The product check was selected after inspecting the IMD and CPC 2026 results and is exploratory, not confirmatory.",
            "ERA5 issue-time predictors are retrospective reanalysis and do not establish operational forecast skill.",
        ],
    }
    report_path = OUTPUT_DIR / "frozen_onset_power_metrics_2026.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"hgb": hgb_metrics, "climatology": climate_metrics, "paired": paired, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
