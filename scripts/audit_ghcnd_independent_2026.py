"""Audit India GHCN-Daily station coverage against the frozen study grid."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from acquire_ghcnd_india_station_data import OUT, RAW, parse_dly  # noqa: E402

POINTS = ROOT / "data/interim/analysis_grid_points.csv"
AUDIT = ROOT / "reports/independent_ghcnd_2026"


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((OUT / "station_inventory.json").read_text(encoding="utf-8"))
    points = pd.read_csv(POINTS)
    latitudes = points.latitude.to_numpy(float)
    longitudes = points.longitude.to_numpy(float)
    records: list[dict[str, object]] = []
    all_days: list[dict[str, object]] = []
    for station in inventory:
        station_id = str(station["station_id"])
        path = RAW / f"{station_id}.dly"
        if not path.exists():
            continue
        rows = parse_dly(path.read_bytes(), station)
        accepted = [row for row in rows if not str(row["quality_flag"]).strip()]
        recent = [
            row for row in accepted
            if "2026-03-01" <= str(row["date"]) <= "2026-06-30"
        ]
        if not recent:
            continue
        distance = haversine_km(
            float(station["latitude"]), float(station["longitude"]), latitudes, longitudes
        )
        nearest = int(np.argmin(distance))
        station_record = {
            "station_id": station_id,
            "station_name": station.get("station_name", ""),
            "station_latitude": station["latitude"],
            "station_longitude": station["longitude"],
            "inventory_first_year": station["first_year"],
            "inventory_last_year": station["last_year"],
            "nearest_grid_id": points.iloc[nearest].grid_id,
            "nearest_grid_latitude": latitudes[nearest],
            "nearest_grid_longitude": longitudes[nearest],
            "distance_to_nearest_study_cell_km": float(distance[nearest]),
            "good_2026_march_june_records": len(recent),
            "quality_accepted_reference_years_1981_2010": len({
                str(row["date"])[:4] for row in accepted
                if "1981-01-01" <= str(row["date"]) <= "2010-12-31"
            }),
            "first_2026_date": min(str(row["date"]) for row in recent),
            "last_2026_date": max(str(row["date"]) for row in recent),
            "source_flags": ";".join(sorted({str(row["source_flag"]).strip() or "blank" for row in recent})),
            "measurement_flags": ";".join(sorted({str(row["measurement_flag"]).strip() or "blank" for row in recent})),
        }
        records.append(station_record)
        for row in recent:
            all_days.append({**station_record, **row})

    station_df = pd.DataFrame(records)
    day_df = pd.DataFrame(all_days)
    if station_df.empty:
        raise RuntimeError("No quality-accepted GHCN-Daily records in March-June 2026")
    station_df.to_csv(AUDIT / "station_coverage_2026.csv", index=False)
    day_df.to_csv(AUDIT / "station_days_2026.csv.gz", index=False, compression="gzip")
    near25 = station_df.distance_to_nearest_study_cell_km.le(25)
    near50 = station_df.distance_to_nearest_study_cell_km.le(50)
    eligible = station_df.loc[
        station_df.quality_accepted_reference_years_1981_2010.ge(10)
        & station_df.good_2026_march_june_records.ge(90)
    ]
    radius_eligibility = {
        str(radius): {
            "stations": int(eligible.distance_to_nearest_study_cell_km.le(radius).sum()),
            "unique_nearest_grid_cells": int(eligible.loc[
                eligible.distance_to_nearest_study_cell_km.le(radius), "nearest_grid_id"
            ].nunique()),
        }
        for radius in (25, 50, 75, 100, 150, 200)
    }
    summary = {
        "station_inventory_total": len(inventory),
        "stations_with_quality_accepted_2026_march_june_data": int(len(station_df)),
        "quality_accepted_station_day_records": int(len(day_df)),
        "stations_within_25km_of_a_study_cell": int(near25.sum()),
        "stations_within_50km_of_a_study_cell": int(near50.sum()),
        "quality_accepted_records_within_25km": int(day_df.distance_to_nearest_study_cell_km.le(25).sum()),
        "quality_accepted_records_within_50km": int(day_df.distance_to_nearest_study_cell_km.le(50).sum()),
        "unique_grid_cells_with_station_within_25km": int(station_df.loc[near25, "nearest_grid_id"].nunique()),
        "unique_grid_cells_with_station_within_50km": int(station_df.loc[near50, "nearest_grid_id"].nunique()),
        "eligible_station_coverage_by_maximum_distance_km": radius_eligibility,
        "source_flags": day_df.source_flag.astype(str).value_counts().to_dict(),
        "measurement_flags": day_df.measurement_flag.astype(str).str.strip().replace("", "blank").value_counts().to_dict(),
        "distance_quantiles_km": station_df.distance_to_nearest_study_cell_km.quantile([0, .25, .5, .75, 1]).to_dict(),
        "limitations": [
            "GHCN-Daily station observations are an independent data product, but their underlying national/GTS reports may overlap with observations contributing to IMD grids.",
            "A station label is point support, while the trained model predicts an IMD-aligned grid-cell label; agreement is not a like-for-like validation unless target construction is carefully harmonized.",
            "Daily station coverage and source flags vary; only blank quality flags are counted in this audit.",
        ],
    }
    (AUDIT / "coverage_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
