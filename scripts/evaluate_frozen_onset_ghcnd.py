"""Exploratory GHCN station-label evaluator (disabled for hourly proxies).

This script maps each station to
its nearest study cell (<=50 km), constructs station-specific normals from
1981--2010 quality-accepted Tmax records, and applies the same IMD-aligned
grid-cell temperature/departure rule. It does not refit or recalibrate models.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from acquire_ghcnd_india_station_data import OUT, RAW, parse_dly  # noqa: E402
from heatwave_benchmark.data import load_imd_terrain_mask  # noqa: E402
from heatwave_benchmark.labels import classify_gridcell_heatwave, compute_daily_normals  # noqa: E402

PREDICTIONS = ROOT / "reports/independent_onset_2026/frozen_model_all_grid_scores_2026.csv.gz"
POINTS = ROOT / "data/interim/analysis_grid_points.csv"
OUTPUT = ROOT / "reports/independent_ghcnd_2026"
TIME_START = pd.Timestamp("1981-01-01")
TIME_END = pd.Timestamp("2026-06-30")
REF_START = pd.Timestamp("1981-01-01")
REF_END = pd.Timestamp("2010-12-31")
TEST_START = pd.Timestamp("2026-03-01")
TEST_END = pd.Timestamp("2026-06-29")
MAX_DISTANCE_KM = 50.0
MIN_REFERENCE_YEARS = 10
MIN_TEST_DAYS = 90
HGB_THRESHOLD = 0.014854158899725712


def distance_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def station_daily_records(path: Path, station: dict[str, object]) -> pd.DataFrame:
    rows = parse_dly(path.read_bytes(), station)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame.date)
    frame = frame.loc[frame.quality_flag.astype(str).str.strip().eq("")]
    return frame.drop_duplicates("date", keep="last").sort_values("date")


def station_labels(
    station_id: str,
    observed: pd.DataFrame,
    terrain_code: int,
) -> pd.DataFrame:
    dates = pd.date_range(TIME_START, TIME_END, freq="D")
    values = pd.Series(observed.tmax_c.to_numpy(float), index=observed.date).reindex(dates).to_numpy()
    full = xr.DataArray(
        values[:, None],
        dims=("time", "cell"),
        coords={"time": dates, "cell": [station_id]},
        name="tmax",
        attrs={"units": "degree_Celsius"},
    )
    reference = full.sel(time=slice(REF_START, REF_END))
    normals = compute_daily_normals(
        reference,
        1981,
        2010,
        window_days=15,
        min_years=MIN_REFERENCE_YEARS,
    )
    doy_index = dates.day_of_year.to_numpy() - 1
    normal_values = normals.values[doy_index, 0]
    label = classify_gridcell_heatwave(
        values,
        values - normal_values,
        str({0: "plains", 1: "hilly", 2: "coastal"}.get(terrain_code, "none")),
    )
    return pd.DataFrame({"date": dates, "station_id": station_id, "station_tmax_c": values,
                         "station_normal_c": normal_values, "station_label": label})


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = score >= threshold
    positive = y == 1
    negative = y == 0
    return {
        "rows": int(len(y)),
        "events": int(positive.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, score)),
        "brier_score": float(brier_score_loss(y, score)),
        "threshold": float(threshold),
        "precision": float((predicted & positive).sum() / predicted.sum()) if predicted.any() else 0.0,
        "recall": float((predicted & positive).sum() / positive.sum()) if positive.any() else 0.0,
        "false_alarm_rate": float((predicted & negative).sum() / negative.sum()) if negative.any() else 0.0,
        "predicted_positive_rows": int(predicted.sum()),
    }


def station_date_block_bootstrap(
    frame: pd.DataFrame,
    replicates: int = 2000,
    block_days: int = 7,
    seed: int = 20261029,
) -> pd.DataFrame:
    """Resample stations and shared 7-day date blocks to retain both dependencies."""
    stations = frame.station_id.astype(str).unique()
    dates = np.array(sorted(frame.date.drop_duplicates().to_numpy(dtype="datetime64[ns]")))
    if len(dates) < block_days:
        raise ValueError("too few distinct dates for the station/date-block bootstrap")
    rng = np.random.default_rng(seed)
    records: list[dict[str, float]] = []
    station_values = frame.station_id.astype(str).to_numpy()
    date_values = frame.date.to_numpy(dtype="datetime64[ns]")
    y_all = frame.target.to_numpy(dtype=np.int8)
    hgb_all = frame.hgb_calibrated_probability.to_numpy(float)
    clim_all = frame.climatology_calibrated_probability.to_numpy(float)
    hgb_prediction = hgb_all >= HGB_THRESHOLD
    climatology_threshold = float(frame.climatology_threshold.iloc[0])
    clim_prediction = clim_all >= climatology_threshold

    def summarize(y: np.ndarray, score: np.ndarray, predicted: np.ndarray,
                  weights: np.ndarray) -> dict[str, float]:
        positive_weight = float(weights[y == 1].sum())
        negative_weight = float(weights[y == 0].sum())
        predicted_weight = float(weights[predicted].sum())
        true_positive_weight = float(weights[predicted & (y == 1)].sum())
        return {
            "average_precision": float(average_precision_score(y, score, sample_weight=weights)),
            "brier_score": float(brier_score_loss(y, score, sample_weight=weights)),
            "precision": true_positive_weight / predicted_weight if predicted_weight else 0.0,
            "recall": true_positive_weight / positive_weight if positive_weight else 0.0,
            "false_alarm_rate": float(weights[predicted & (y == 0)].sum() / negative_weight) if negative_weight else 0.0,
        }

    for _ in range(replicates):
        selected_stations = rng.choice(stations, size=len(stations), replace=True)
        station_multiplicity = pd.Series(selected_stations).value_counts().to_dict()
        sampled_dates: list[np.datetime64] = []
        while len(sampled_dates) < len(dates):
            start = int(rng.integers(0, len(dates) - block_days + 1))
            sampled_dates.extend(dates[start : start + block_days])
        date_multiplicity = pd.Series(sampled_dates[:len(dates)]).value_counts().to_dict()
        weights = np.fromiter(
            (station_multiplicity.get(station, 0) * date_multiplicity.get(day, 0)
             for station, day in zip(station_values, date_values, strict=True)),
            dtype=float,
            count=len(frame),
        )
        selected_rows = weights > 0
        y = y_all[selected_rows]
        weights = weights[selected_rows]
        if np.unique(y).size < 2:
            continue
        hgb = summarize(y, hgb_all[selected_rows], hgb_prediction[selected_rows], weights)
        clim = summarize(y, clim_all[selected_rows], clim_prediction[selected_rows], weights)
        records.append({
            "hgb_ap": hgb["average_precision"],
            "climatology_ap": clim["average_precision"],
            "ap_difference": hgb["average_precision"] - clim["average_precision"],
            "hgb_brier": hgb["brier_score"],
            "climatology_brier": clim["brier_score"],
            "brier_difference": hgb["brier_score"] - clim["brier_score"],
            "hgb_precision": hgb["precision"],
            "climatology_precision": clim["precision"],
            "hgb_recall": hgb["recall"],
            "climatology_recall": clim["recall"],
            "hgb_false_alarm_rate": hgb["false_alarm_rate"],
            "climatology_false_alarm_rate": clim["false_alarm_rate"],
        })
    values = pd.DataFrame(records)
    return pd.DataFrame([
        {"metric": column, "lower_95": float(values[column].quantile(.025)),
         "upper_95": float(values[column].quantile(.975)), "valid_replicates": len(values),
          "resampling": f"two-way bootstrap: station clusters and shared {block_days}-day calendar-date blocks"}
        for column in values.columns
    ])


def main(max_distance_km: float = MAX_DISTANCE_KM) -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit_path = OUTPUT / "coverage_summary.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        flags = audit.get("measurement_flags", {})
        if int(flags.get("H", 0)) > 0:
            raise RuntimeError(
                "GHCN TMAX records with measurement flag H are highest-hourly-temperature "
                "proxies, not conventional daily maxima. Exclude them before any daily-Tmax "
                "model scoring; these data currently provide source-screening evidence only."
            )
    inventory = json.loads((OUT / "station_inventory.json").read_text(encoding="utf-8"))
    points = pd.read_csv(POINTS)
    scores = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    if scores.duplicated(["date", "grid_id"]).any():
        raise ValueError("frozen all-grid prediction file has duplicate date/grid rows")
    score_lookup = scores.set_index(["date", "grid_id"])
    terrain = load_imd_terrain_mask()
    point_lats = points.latitude.to_numpy(float)
    point_lons = points.longitude.to_numpy(float)
    candidates: list[pd.DataFrame] = []
    station_status: list[dict[str, object]] = []

    for station in inventory:
        station_id = str(station["station_id"])
        raw_path = RAW / f"{station_id}.dly"
        if not raw_path.is_file():
            continue
        observed = station_daily_records(raw_path, station)
        if observed.empty:
            continue
        ref = observed.loc[observed.date.between(REF_START, REF_END)]
        recent = observed.loc[observed.date.between(TEST_START, TEST_END + pd.Timedelta(days=1))]
        distinct_reference_years = int(ref.date.dt.year.nunique())
        if len(recent) < MIN_TEST_DAYS or distinct_reference_years < MIN_REFERENCE_YEARS:
            continue
        distances = distance_km(float(station["latitude"]), float(station["longitude"]), point_lats, point_lons)
        nearest = int(np.argmin(distances))
        nearest_km = float(distances[nearest])
        if nearest_km > max_distance_km:
            continue
        grid = points.iloc[nearest]
        terrain_code = int(grid.terrain_code)
        labels = station_labels(station_id, observed, terrain_code)
        issue = labels.loc[labels.date.between(TEST_START, TEST_END)].copy()
        next_day = labels[["date", "station_label"]].copy()
        next_day["date"] = next_day.date - pd.Timedelta(days=1)
        issue = issue.merge(next_day, on="date", how="left", suffixes=("", "_next"), validate="one_to_one")
        issue["current_state"] = np.where(issue.station_label.notna(), (issue.station_label >= 1).astype(float), np.nan)
        issue["target"] = np.where(issue.station_label_next.notna(), (issue.station_label_next >= 1).astype(float), np.nan)
        issue = issue.loc[issue.current_state.eq(0) & issue.target.notna()].copy()
        issue["grid_id"] = str(grid.grid_id)
        issue["distance_to_grid_km"] = nearest_km
        issue = issue.join(score_lookup, on=["date", "grid_id"], how="left", validate="many_to_one")
        issue = issue.dropna(subset=["hgb_calibrated_probability", "climatology_calibrated_probability"])
        if not issue.empty:
            candidates.append(issue)
        station_status.append({
            "station_id": station_id,
            "station_name": station.get("station_name", ""),
            "grid_id": str(grid.grid_id),
            "distance_km": nearest_km,
            "terrain_code": terrain_code,
            "reference_years": distinct_reference_years,
            "quality_accepted_2026_records": int(len(recent)),
            "onset_test_rows": int(len(issue)),
            "onset_events": int(issue.target.sum()) if not issue.empty else 0,
        })

    if not candidates:
        raise RuntimeError("No station onset rows could be matched to frozen grid scores")
    joined = pd.concat(candidates, ignore_index=True)
    if joined.target.nunique() != 2:
        raise RuntimeError("Station onset test requires both event and non-event rows")
    y = joined.target.to_numpy(dtype=np.int8)
    hgb_score = joined.hgb_calibrated_probability.to_numpy(float)
    clim_score = joined.climatology_calibrated_probability.to_numpy(float)
    threshold_clim = float(joined.climatology_threshold.iloc[0])
    results = {
        "scope": "station-level exploratory check; not a like-for-like gridded target validation",
        "target_definition": "Station-specific centered 15-day circular 1981-2010 Tmax normal with at least 10 reference years; same IMD-aligned absolute/departure thresholds as the primary label; onset is station state zero followed by positive station label.",
        "source": "NOAA NCEI GHCN-Daily; only blank quality flags retained; source flags are recorded in the acquisition audit.",
        "model": "Frozen HGB estimator, features, calibrator, and probability threshold; no refitting or recalibration against GHCN data.",
        "station_filter": {"maximum_nearest_grid_distance_km": max_distance_km,
                           "minimum_reference_years": MIN_REFERENCE_YEARS,
                           "minimum_quality_accepted_2026_days": MIN_TEST_DAYS},
        "stations": len(joined.station_id.unique()),
        "grid_cells": len(joined.grid_id.unique()),
        "dates": len(joined.date.unique()),
        "station_day_rows": len(joined),
        "event_rows": int(y.sum()),
        "event_prevalence": float(y.mean()),
        "hgb": metrics(y, hgb_score, HGB_THRESHOLD),
        "climatology": metrics(y, clim_score, threshold_clim),
        "paired_station_date_block_bootstrap": station_date_block_bootstrap(joined).to_dict(orient="records"),
        "limits": [
            "Only nearby stations with sufficient data qualify; results are not nationally representative.",
            "GHCN source flag 2 denotes Synoptic Summary of the Day data, which may share underlying reports with other products; source-record independence from IMD station inputs is unverified.",
            "The point station target and 1-degree model prediction have different spatial support; nearest-cell mapping introduces representativeness error.",
            "The station-specific target normal and terrain assignment are a sensitivity adaptation, not an official IMD station/subdivision declaration.",
            "The two-way station/date-block bootstrap has few station clusters and events; intervals are descriptive and may remain unstable.",
            "ERA5 predictors remain retrospective reanalysis, not archived operational forecasts.",
        ],
    }
    suffix = "" if max_distance_km == MAX_DISTANCE_KM else f"_{int(max_distance_km)}km"
    (OUTPUT / f"frozen_onset_ghcnd_metrics{suffix}.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    joined.to_csv(
        OUTPUT / f"frozen_onset_ghcnd_station_days{suffix}.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(station_status).to_csv(
        OUTPUT / f"eligible_stations{suffix}.csv", index=False
    )
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-distance-km", type=float, default=MAX_DISTANCE_KM)
    args = parser.parse_args()
    if args.max_distance_km < 0:
        parser.error("--max-distance-km must be non-negative")
    raise SystemExit(main(args.max_distance_km))
