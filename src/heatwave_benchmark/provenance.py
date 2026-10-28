"""Integrity checks for locally cached direct ERA5 Icechunk extracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import ERA5_DAILY_VARIABLES, parse_era5_daily_response


def _read_manifest(path: Path, expected_grid_ids: set[str], kind: str) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"missing {kind} manifest: {path}")
    manifest = pd.read_csv(path, dtype={"grid_id": str})
    required = {"grid_id", "status", "cache_file", "sha256", "source_snapshot_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{kind} manifest is missing columns: {sorted(missing)}")
    if manifest["grid_id"].duplicated().any():
        raise ValueError(f"{kind} manifest contains duplicate grid IDs")
    actual_grid_ids = set(manifest["grid_id"].astype(str))
    if actual_grid_ids != expected_grid_ids:
        raise ValueError(
            f"{kind} manifest grid IDs differ from the requested analysis grid "
            f"(missing={len(expected_grid_ids - actual_grid_ids)}, "
            f"extra={len(actual_grid_ids - expected_grid_ids)})"
        )
    if not manifest["status"].isin({"cached", "downloaded"}).all():
        raise ValueError(f"{kind} manifest contains incomplete cache statuses")
    return manifest.set_index("grid_id", drop=False)


def _load_and_check_payload(
    path: Path,
    expected_sha256: str,
    snapshot_id: str,
    grid_id: str,
    latitude: float,
    longitude: float,
    expected_dates: pd.DatetimeIndex,
    *,
    check_padding: bool = False,
) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"missing ERA5 cache file: {path}")
    content = path.read_bytes()
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"checksum mismatch for ERA5 cache file {path}")
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid JSON in ERA5 cache file {path}") from exc
    if payload.get("source_metadata", {}).get("snapshot_id") != snapshot_id:
        raise ValueError(f"snapshot mismatch in ERA5 cache file {path}")
    parsed = parse_era5_daily_response(payload, grid_id, latitude, longitude)
    if not pd.DatetimeIndex(parsed["date"]).equals(expected_dates):
        raise ValueError(f"invalid date coverage in ERA5 cache file {path}")

    feature_names = [feature for _, feature in ERA5_DAILY_VARIABLES]
    if check_padding:
        seasonal_dates: set[str] = set()
        for year in range(expected_dates[0].year, expected_dates[-1].year + 1):
            seasonal_dates.update(
                pd.date_range(f"{year}-02-22", f"{year}-06-30", freq="D")
                .strftime("%Y-%m-%d")
                .tolist()
            )
        raw_dates = payload["daily"]["time"]
        for variable, feature in ERA5_DAILY_VARIABLES:
            values = payload["daily"][variable]
            numeric = parsed[feature].to_numpy(dtype=float)
            for index, date_value in enumerate(raw_dates):
                if date_value in seasonal_dates:
                    if values[index] is None or not np.isfinite(numeric[index]):
                        raise ValueError(f"missing seasonal ERA5 value in {path} for {date_value}")
                elif values[index] is not None:
                    raise ValueError(f"unexpected off-season ERA5 value in {path} for {date_value}")
    elif not np.isfinite(parsed[feature_names].to_numpy(dtype=float)).all():
        raise ValueError(f"missing seasonal ERA5 predictor values in {path}")
    return payload


def validate_era5_icechunk_cache(
    destination_dir: str | Path,
    points: pd.DataFrame,
    start_year: int,
    end_year: int,
    snapshot_id: str | None = None,
    require_final: bool = True,
) -> dict[str, Any]:
    """Validate annual and final ERA5 caches against their hashes and coverage.

    Annual manifests must cover every requested grid cell for every year. If
    ``require_final`` is true, the combined full-calendar JSON caches and their
    top-level manifest are also checked, including null padding outside the
    declared seasonal windows.
    """
    if not isinstance(start_year, int) or not isinstance(end_year, int) or start_year > end_year:
        raise ValueError("start_year and end_year must be ordered integer years")
    required_points = {"grid_id", "latitude", "longitude"}
    missing_points = required_points - set(points.columns)
    if missing_points:
        raise ValueError(f"point table is missing columns: {sorted(missing_points)}")
    if points.empty or points[list(required_points)].isna().any().any():
        raise ValueError("point table must contain valid, non-empty locations")
    if points["grid_id"].duplicated().any():
        raise ValueError("grid_id values must be unique")

    destination = Path(destination_dir)
    source_manifest_path = destination / "source_manifest.json"
    if not source_manifest_path.exists():
        raise ValueError(f"missing ERA5 source manifest: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    recorded_snapshot = str(source_manifest.get("snapshot_id", ""))
    if not recorded_snapshot:
        raise ValueError("ERA5 source manifest has no immutable snapshot ID")
    if snapshot_id is not None and str(snapshot_id) != recorded_snapshot:
        raise ValueError("requested ERA5 snapshot differs from the source manifest")
    snapshot_id = recorded_snapshot

    ordered = points.sort_values("grid_id", kind="stable").reset_index(drop=True)
    expected_grid_ids = set(ordered["grid_id"].astype(str))
    years = list(range(start_year, end_year + 1))
    annual_files = 0
    seasonal_rows = 0
    for year in years:
        year_dir = destination / "windows" / str(year)
        manifest = _read_manifest(year_dir / "download_manifest.csv", expected_grid_ids, f"{year} annual")
        expected_dates = pd.date_range(f"{year}-02-22", f"{year}-06-30", freq="D")
        if "daily_rows" not in manifest or not pd.to_numeric(
            manifest["daily_rows"], errors="coerce"
        ).eq(len(expected_dates)).all():
            raise ValueError(f"{year} annual manifest has incorrect daily row counts")
        seasonal_rows += len(expected_dates) * len(expected_grid_ids)
        for point in ordered.itertuples(index=False):
            grid_id = str(point.grid_id)
            row = manifest.loc[grid_id]
            if str(row["source_snapshot_id"]) != snapshot_id:
                raise ValueError(f"snapshot mismatch in {year} annual manifest for {grid_id}")
            _load_and_check_payload(
                year_dir / str(row["cache_file"]),
                str(row["sha256"]),
                snapshot_id,
                grid_id,
                float(point.latitude),
                float(point.longitude),
                expected_dates,
            )
            annual_files += 1

    final_files = 0
    if require_final:
        manifest = _read_manifest(
            destination / "download_manifest.csv", expected_grid_ids, "final"
        )
        expected_dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
        expected_seasonal_days = sum(
            len(pd.date_range(f"{year}-02-22", f"{year}-06-30", freq="D"))
            for year in years
        )
        if (
            "daily_rows" not in manifest
            or "cache_calendar_rows" not in manifest
            or not pd.to_numeric(manifest["daily_rows"], errors="coerce")
            .eq(expected_seasonal_days)
            .all()
            or not pd.to_numeric(manifest["cache_calendar_rows"], errors="coerce")
            .eq(len(expected_dates))
            .all()
        ):
            raise ValueError("final ERA5 manifest has incorrect daily/calendar row counts")
        for point in ordered.itertuples(index=False):
            grid_id = str(point.grid_id)
            row = manifest.loc[grid_id]
            if str(row["source_snapshot_id"]) != snapshot_id:
                raise ValueError(f"snapshot mismatch in final manifest for {grid_id}")
            _load_and_check_payload(
                destination / str(row["cache_file"]),
                str(row["sha256"]),
                snapshot_id,
                grid_id,
                float(point.latitude),
                float(point.longitude),
                expected_dates,
                check_padding=True,
            )
            final_files += 1

    return {
        "snapshot_id": snapshot_id,
        "start_year": start_year,
        "end_year": end_year,
        "points": len(expected_grid_ids),
        "annual_cache_files": annual_files,
        "final_cache_files": final_files,
        "seasonal_predictor_rows": seasonal_rows,
        "final_cache_required": require_final,
    }
