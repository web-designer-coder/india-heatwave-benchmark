"""Acquire the frozen ERA5 snapshot window for the 2026 onset evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heatwave_benchmark.data import download_era5_icechunk_seasonal_grid  # noqa: E402

SNAPSHOT = "TZ2KX6A24CJ5DG17T7MG"
POINTS_PATH = ROOT / "data" / "interim" / "analysis_grid_points.csv"
DESTINATION = ROOT / "data" / "raw" / "era5_onset_independent_2026"


def main() -> None:
    points = pd.read_csv(POINTS_PATH)
    if len(points) != 275 or points["grid_id"].astype(str).nunique() != 275:
        raise ValueError("expected the frozen 275 unique India grid cells")
    manifest = download_era5_icechunk_seasonal_grid(
        points,
        DESTINATION,
        start_year=2026,
        end_year=2026,
        snapshot_id=SNAPSHOT,
        max_workers=8,
    )
    if len(manifest) != 275:
        raise ValueError(f"expected 275 ERA5 cell records, received {len(manifest)}")
    if set(manifest["source_snapshot_id"].astype(str)) != {SNAPSHOT}:
        raise ValueError("downloaded ERA5 cells do not all match the frozen snapshot")
    print(f"Complete: {len(manifest)} frozen-grid ERA5 cell windows from {SNAPSHOT}")
    print(f"Source manifest: {DESTINATION / 'source_manifest.json'}")


if __name__ == "__main__":
    main()
