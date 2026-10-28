"""Evaluate frozen 2026 onset probabilities against the NOAA CPC Tmax product.

The CPC daily Tmax product is an external gridded analysis, not an independent
raw-observation source: CPC says its daily Tmax inputs come from GTS reports,
which may overlap with IMD source observations. CPC 0.5-degree values are
averaged within each study 1-degree cell. The fixed IMD 1981--2010 cell normals
and the already frozen model, calibrator, and threshold are retained. No CPC
outcome is used to fit or tune any model component.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from heatwave_benchmark.labels import classify_gridcell_heatwave, compute_daily_normals  # noqa: E402
from evaluate_frozen_onset_2026 import (  # noqa: E402
    onset_metrics,
    moving_block_intervals,
    paired_moving_block_differences,
)


CPC_DIR = ROOT / "data/raw/cpc_tmax_2026"
MANIFEST = CPC_DIR / "acquisition_manifest.json"
FEATURES = ROOT / "data/interim/benchmark_features_era5.csv.gz"
HISTORICAL_TMAX = ROOT / "data/interim/imd_tmax_1981_2025.nc"
SCORE_FILE = ROOT / "reports/independent_onset_2026/frozen_model_all_grid_scores_2026.csv.gz"
FROZEN_METRICS = ROOT / "reports/independent_onset_2026/frozen_onset_2026_metrics.json"
OUT = ROOT / "reports/independent_cpc_2026"
START = pd.Timestamp("2026-02-22")
END = pd.Timestamp("2026-06-30")
NX, NY, FIELD_BYTES = 720, 360, 720 * 360 * 4
UNDEF = -999.0
MIN_VALID_SUBCELLS = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_acquisition() -> dict[str, object]:
    metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_dates = pd.date_range(START, END, freq="D").strftime("%Y-%m-%d").tolist()
    records = metadata["daily_records"]
    if [record["date"] for record in records] != expected_dates:
        raise ValueError("CPC acquisition manifest does not cover the frozen evaluation window")
    for record in records:
        path = ROOT / record["file"]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"CPC file failed manifest verification: {path}")
    return metadata


def _study_cells() -> pd.DataFrame:
    columns = ["grid_id", "latitude", "longitude", "terrain_code"]
    frame = pd.read_csv(FEATURES, usecols=columns)
    cells = frame[columns].drop_duplicates("grid_id").sort_values("grid_id", kind="stable")
    if len(cells) != 275:
        raise ValueError(f"expected 275 study cells, found {len(cells)}")
    return cells.reset_index(drop=True)


def _cpc_neighbors(cells: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    latitudes = -89.75 + 0.5 * np.arange(NY)
    longitudes = 0.25 + 0.5 * np.arange(NX)
    y_pairs, x_pairs = [], []
    for row in cells.itertuples(index=False):
        y_upper = int(np.searchsorted(latitudes, float(row.latitude)))
        x_upper = int(np.searchsorted(longitudes, float(row.longitude)))
        ys, xs = [y_upper - 1, y_upper], [x_upper - 1, x_upper]
        if min(ys) < 0 or max(ys) >= NY or min(xs) < 0 or max(xs) >= NX:
            raise ValueError(f"study cell outside CPC grid: {row.grid_id}")
        y_pairs.append(ys)
        x_pairs.append(xs)
    return np.asarray(y_pairs, dtype=int), np.asarray(x_pairs, dtype=int)


def _read_cpc_series(cells: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    y_pairs, x_pairs = _cpc_neighbors(cells)
    dates = pd.date_range(START, END, freq="D")
    tmax_1deg = np.full((len(dates), len(cells)), np.nan, dtype=np.float32)
    report_count_1deg = np.zeros((len(dates), len(cells)), dtype=np.float32)
    n_supported = np.zeros((len(dates), len(cells)), dtype=np.int8)
    for day_index, day in enumerate(dates):
        path = CPC_DIR / f"cpc_tmax_nmax_{day:%Y%m%d}.bin"
        data = np.fromfile(path, dtype="<f4")
        if data.size != 2 * NX * NY:
            raise ValueError(f"unexpected TMAX/NMAX payload size: {path}")
        tmax = data[: NX * NY].reshape(NY, NX)
        nmax = data[NX * NY :].reshape(NY, NX)
        four_tmax = np.stack([
            tmax[y_pairs[:, yi], x_pairs[:, xi]]
            for yi in range(2) for xi in range(2)
        ], axis=1)
        four_nmax = np.stack([
            nmax[y_pairs[:, yi], x_pairs[:, xi]]
            for yi in range(2) for xi in range(2)
        ], axis=1)
        valid = np.isfinite(four_tmax) & (four_tmax != UNDEF)
        valid_count = valid.sum(axis=1)
        safe = np.where(valid, four_tmax, 0.0)
        means = np.full(len(cells), np.nan, dtype=np.float32)
        enough = valid_count >= MIN_VALID_SUBCELLS
        means[enough] = safe[enough].sum(axis=1) / valid_count[enough]
        tmax_1deg[day_index] = means
        n_supported[day_index] = valid_count
        report_count_1deg[day_index] = np.where(valid, np.maximum(four_nmax, 0), 0).sum(axis=1)
    return dates, tmax_1deg, np.stack([n_supported, report_count_1deg], axis=-1)


def _build_labels(cells: pd.DataFrame, dates: pd.DatetimeIndex, cpc_tmax: np.ndarray) -> pd.DataFrame:
    history = xr.load_dataset(HISTORICAL_TMAX, engine="scipy")["tmax"]
    lat = xr.DataArray(cells.latitude.to_numpy(), dims="cell", coords={"cell": cells.grid_id.astype(str)})
    lon = xr.DataArray(cells.longitude.to_numpy(), dims="cell", coords={"cell": cells.grid_id.astype(str)})
    ref = history.sel(time=slice("1981-01-01", "2010-12-31")).sel(latitude=lat, longitude=lon)
    normals = compute_daily_normals(ref, 1981, 2010, window_days=15, min_years=10)
    normal_values = normals.values[dates.dayofyear.to_numpy() - 1]
    terrain_names = cells.terrain_code.map({0: "plains", 1: "hilly", 2: "coastal"}).to_numpy()
    labels = classify_gridcell_heatwave(cpc_tmax, cpc_tmax - normal_values, terrain_names)
    frame = pd.DataFrame({"date": np.repeat(dates.to_numpy(), len(cells)), "grid_id": np.tile(cells.grid_id.astype(str), len(dates)), "cpc_label": labels.reshape(-1)})
    frame["target_h1"] = frame.groupby("grid_id", sort=False)["cpc_label"].shift(-1)
    frame["cpc_next_date"] = frame.groupby("grid_id", sort=False)["date"].shift(-1)
    frame.loc[frame["cpc_next_date"] != frame["date"] + pd.Timedelta(days=1), "target_h1"] = np.nan
    frame["target_h1"] = np.where(frame.target_h1 >= 1, 1.0, np.where(frame.target_h1 == 0, 0.0, np.nan))
    return frame.drop(columns="cpc_next_date")


def main() -> None:
    metadata = _verify_acquisition()
    cells = _study_cells()
    dates, cpc_tmax, support = _read_cpc_series(cells)
    labels = _build_labels(cells, dates, cpc_tmax)
    scores = pd.read_csv(SCORE_FILE, parse_dates=["date"])
    frozen = json.loads(FROZEN_METRICS.read_text(encoding="utf-8"))
    threshold = float(frozen["frozen_threshold"])
    labels["current_cpc_label"] = labels["cpc_label"]
    eval_rows = labels.loc[
        labels.date.between(pd.Timestamp("2026-03-01"), pd.Timestamp("2026-06-29"))
        & labels.current_cpc_label.eq(0)
        & labels.target_h1.notna()
    ].merge(scores, on=["date", "grid_id"], how="inner", validate="one_to_one")
    eval_rows = eval_rows.rename(columns={"hgb_calibrated_probability": "hgb_probability"})
    if eval_rows.empty or eval_rows.target_h1.nunique() < 2:
        raise ValueError("CPC external-product test lacks both onset classes")
    y = eval_rows.target_h1.to_numpy(dtype=np.int8)
    hgb = eval_rows.hgb_probability.to_numpy(dtype=float)
    climate = eval_rows.climatology_calibrated_probability.to_numpy(dtype=float)
    climate_threshold = float(eval_rows.climatology_threshold.iloc[0])
    persistence = np.zeros(len(eval_rows), dtype=float)
    point_metrics = {
        "frozen_hgb": onset_metrics(y, hgb, threshold),
        "climatology": onset_metrics(y, climate, climate_threshold),
        "persistence_state_zero": onset_metrics(y, persistence, threshold),
    }
    hgb_intervals = moving_block_intervals(eval_rows, hgb, threshold, seed=20261101)
    climate_intervals = moving_block_intervals(eval_rows, climate, climate_threshold, seed=20261102)
    paired = pd.concat([
        paired_moving_block_differences(eval_rows, hgb, threshold, climate, climate_threshold, "climatology", seed=20261103),
        paired_moving_block_differences(eval_rows, hgb, threshold, persistence, threshold, "persistence", seed=20261104),
    ], ignore_index=True)
    report = {
        "evaluation": "frozen onset model against the CPC external gridded Tmax analysis",
        "evaluation_scope": "product-level external test; CPC source reports may overlap IMD observations; not raw-source-independent validation or operational forecast verification",
        "cpc_dataset": metadata["dataset"],
        "source_url": metadata["source_url"],
        "acquisition_dates": metadata["date_range_inclusive"],
        "study_cells": int(len(cells)),
        "dates_with_cpc_values": int(len(dates)),
        "spatial_aggregation": f"Mean of the four CPC 0.5-degree grid centers in each IMD 1-degree cell; require at least {MIN_VALID_SUBCELLS}/4 finite CPC values",
        "target_definition": "Apply the primary IMD-aligned thresholds and the fixed IMD 1981-2010 cell normal to CPC daily Tmax aggregates; do not interpret as official IMD declarations.",
        "mean_valid_cpc_subcells_per_cell_day": float(support[..., 0].mean()),
        "mean_nmax_reports_across_valid_subcells": float(support[..., 1].mean()),
        "frozen_threshold": threshold,
        "climatology_threshold": climate_threshold,
        "metrics": point_metrics,
        "uncertainty": "2,000 paired 7-day moving-block resamples of issue dates; one season only",
        "frozen_hgb_intervals": hgb_intervals.to_dict(orient="records"),
        "climatology_intervals": climate_intervals.to_dict(orient="records"),
        "paired_difference_intervals": paired.to_dict(orient="records"),
        "dependence_and_limitations": [
            "CPC daily Tmax is a separate gridded analysis based on CPC GTS daily reports; source reports may overlap with IMD station inputs.",
            "CPC 0.5-degree analysis is aggregated to the study's 1-degree cell support; this is not a station-level validation.",
            "The fixed IMD 1981-2010 normal is shared with the primary target; the external product supplies daily 2026 Tmax values.",
            "The onset model, features, calibrator, and threshold were not changed after CPC outcome evaluation.",
            "ERA5 predictors are retrospective and do not represent archived operational forecasts.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUT / "cpc_cell_day_labels_2026.csv.gz", index=False, compression="gzip")
    eval_rows.to_csv(OUT / "cpc_frozen_onset_predictions_2026.csv", index=False)
    (OUT / "cpc_frozen_onset_metrics_2026.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(eval_rows), "events": int(y.sum()), "metrics": point_metrics, "report": str(OUT / "cpc_frozen_onset_metrics_2026.json")}, indent=2))


if __name__ == "__main__":
    main()
