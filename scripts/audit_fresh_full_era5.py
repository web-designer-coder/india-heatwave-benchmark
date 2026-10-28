"""Reacquire the full pinned ERA5 window and compare it with the analysis input.

The fresh download goes to a separate directory and does not overwrite the
primary cache or assembled dataset. The resulting audit checks all annual
cell-window checksums and compares the reassembled predictors to the field
used by the primary benchmark.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
FRESH_DIR = ROOT / "data/raw/era5_fresh_full"
FRESH_NETCDF = ROOT / "data/interim/era5_fresh_full_1981_2025.nc"
REFERENCE_NETCDF = ROOT / "data/interim/era5_1981_2025.nc"
SOURCE_MANIFEST = ROOT / "data/raw/era5_icechunk/source_manifest.json"
EXPECTED_SNAPSHOT = "TZ2KX6A24CJ5DG17T7MG"
START_YEAR, END_YEAR = 1981, 2025
EXPECTED_GRID_COUNT = 275
ABSOLUTE_TOLERANCE = 1e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifests(points: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, object]:
    if len(manifest) != EXPECTED_GRID_COUNT:
        raise ValueError(f"expected {EXPECTED_GRID_COUNT} consolidated rows, found {len(manifest)}")
    if set(manifest.grid_id.astype(str)) != set(points.grid_id.astype(str)):
        raise ValueError("fresh ERA5 manifest does not match the fixed study-cell set")
    if set(manifest.status) != {"downloaded"}:
        raise ValueError(f"fresh consolidated manifest has non-downloaded rows: {manifest.status.value_counts().to_dict()}")
    if set(manifest.source_snapshot_id.astype(str)) != {EXPECTED_SNAPSHOT}:
        raise ValueError("fresh consolidated caches do not all use the frozen snapshot")

    bad_consolidated = []
    for row in manifest.itertuples(index=False):
        path = FRESH_DIR / row.cache_file
        if not path.is_file() or sha256(path) != row.sha256:
            bad_consolidated.append(str(row.grid_id))
    if bad_consolidated:
        raise ValueError(f"consolidated cache checksum mismatch for {len(bad_consolidated)} cells")

    checked_windows = 0
    per_year = {}
    for year in range(START_YEAR, END_YEAR + 1):
        year_dir = FRESH_DIR / "windows" / str(year)
        year_manifest_path = year_dir / "download_manifest.csv"
        if not year_manifest_path.is_file():
            raise FileNotFoundError(f"missing annual manifest: {year_manifest_path}")
        yearly = pd.read_csv(year_manifest_path)
        if len(yearly) != EXPECTED_GRID_COUNT or set(yearly.grid_id.astype(str)) != set(points.grid_id.astype(str)):
            raise ValueError(f"annual manifest does not cover all fixed cells: {year}")
        if not set(yearly.status).issubset({"cached", "downloaded"}):
            raise ValueError(f"annual manifest has failed cells for {year}")
        bad = []
        for row in yearly.itertuples(index=False):
            path = year_dir / row.cache_file
            if not path.is_file() or sha256(path) != row.sha256:
                bad.append(str(row.grid_id))
        if bad:
            raise ValueError(f"annual cache checksum mismatch for {year}: {len(bad)} cells")
        checked_windows += len(yearly)
        per_year[str(year)] = {"status_counts": yearly.status.value_counts().to_dict(), "cells": len(yearly)}

    return {
        "consolidated_cells": len(manifest),
        "annual_windows": checked_windows,
        "years": per_year,
    }


def compare_fields(fresh: xr.Dataset, reference: xr.Dataset) -> dict[str, object]:
    if dict(fresh.sizes) != dict(reference.sizes):
        raise ValueError(f"field dimensions differ: fresh={dict(fresh.sizes)} reference={dict(reference.sizes)}")
    if set(fresh.data_vars) != set(reference.data_vars):
        raise ValueError("fresh and reference ERA5 fields have different variables")
    for coordinate in fresh.coords:
        if coordinate not in reference.coords or not np.array_equal(fresh[coordinate].values, reference[coordinate].values):
            raise ValueError(f"coordinate mismatch: {coordinate}")

    results = {}
    for variable in sorted(fresh.data_vars):
        left = np.asarray(fresh[variable].values, dtype=np.float64)
        right = np.asarray(reference[variable].values, dtype=np.float64)
        if not np.array_equal(np.isfinite(left), np.isfinite(right)):
            raise ValueError(f"finite-value mask mismatch in {variable}")
        valid = np.isfinite(left)
        difference = np.abs(left[valid] - right[valid])
        max_abs = float(difference.max(initial=0.0))
        results[variable] = {
            "finite_values_compared": int(valid.sum()),
            "max_absolute_difference": max_abs,
            "within_tolerance": max_abs <= ABSOLUTE_TOLERANCE,
        }
        if max_abs > ABSOLUTE_TOLERANCE:
            raise ValueError(f"{variable} differs from the benchmark input by {max_abs}")
    return results


def main() -> None:
    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(SOURCE_MANIFEST)
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source.get("snapshot_id") != EXPECTED_SNAPSHOT:
        raise ValueError("the local primary source manifest does not match the frozen snapshot")
    if FRESH_DIR.exists():
        saved = FRESH_DIR / "source_manifest.json"
        if not saved.is_file() or json.loads(saved.read_text(encoding="utf-8")).get("snapshot_id") != EXPECTED_SNAPSHOT:
            raise FileExistsError(f"refusing to reuse unrelated fresh-data directory: {FRESH_DIR}")

    points = pd.read_csv(ROOT / "data/interim/analysis_grid_points.csv")
    if len(points) != EXPECTED_GRID_COUNT or points.grid_id.duplicated().any():
        raise ValueError("analysis grid does not contain the fixed 275 unique cells")

    from heatwave_benchmark.data import download_era5_icechunk_seasonal_grid
    from heatwave_benchmark.pipeline import assemble_era5_dataset

    manifest = download_era5_icechunk_seasonal_grid(
        points,
        FRESH_DIR,
        start_year=START_YEAR,
        end_year=END_YEAR,
        snapshot_id=EXPECTED_SNAPSHOT,
        max_workers=4,
        progress_callback=lambda year_index, year_total, year, current, total, complete, error: print(
            f"fresh ERA5 {year} ({year_index}/{year_total}): {complete}/{EXPECTED_GRID_COUNT} cells validated",
            flush=True,
        ),
    )
    manifest_checks = verify_manifests(points, manifest)
    fresh = assemble_era5_dataset(FRESH_DIR, points, "1981-01-01", "2025-12-31")
    fresh.to_netcdf(FRESH_NETCDF, engine="scipy")
    reference = xr.load_dataset(REFERENCE_NETCDF, engine="scipy")
    field_comparison = compare_fields(fresh, reference)
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source["dataset"],
        "registry_url": source["registry_url"],
        "snapshot_id": EXPECTED_SNAPSHOT,
        "period": [START_YEAR, END_YEAR],
        "study_cells": EXPECTED_GRID_COUNT,
        "fresh_cache_directory": str(FRESH_DIR.relative_to(ROOT)),
        "fresh_dataset": str(FRESH_NETCDF.relative_to(ROOT)),
        "fresh_dataset_sha256": sha256(FRESH_NETCDF),
        "reference_dataset_sha256": sha256(REFERENCE_NETCDF),
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "manifest_checks": manifest_checks,
        "field_comparison": field_comparison,
        "all_passed": True,
    }
    output = ROOT / "reports/fresh_era5_full_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": True, "annual_windows": manifest_checks["annual_windows"], "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
