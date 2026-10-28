"""Chronological and buffered geographic evaluation partitions."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from heatwave_benchmark.config import ExperimentConfig


def add_spatial_block_ids(
    frame: pd.DataFrame,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    block_degrees: float = 5.0,
) -> pd.DataFrame:
    """Return a copy with deterministic fixed-size latitude/longitude blocks."""
    if block_degrees <= 0:
        raise ValueError("block_degrees must be positive")
    missing = {latitude_col, longitude_col} - set(frame.columns)
    if missing:
        raise ValueError(f"coordinate columns are missing: {sorted(missing)}")

    result = frame.copy()
    latitude = pd.to_numeric(result[latitude_col], errors="coerce")
    longitude = pd.to_numeric(result[longitude_col], errors="coerce")
    if latitude.isna().any() or longitude.isna().any():
        raise ValueError("latitude and longitude must be finite numbers")
    if not np.isfinite(latitude.to_numpy()).all() or not np.isfinite(longitude.to_numpy()).all():
        raise ValueError("latitude and longitude must be finite numbers")

    lat_index = np.floor(latitude.to_numpy() / block_degrees).astype(int)
    lon_index = np.floor(longitude.to_numpy() / block_degrees).astype(int)
    result["spatial_block"] = [
        f"lat{lat}_lon{lon}" for lat, lon in zip(lat_index, lon_index, strict=True)
    ]
    return result


def temporal_split(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    date_col: str = "date",
) -> dict[str, pd.DataFrame]:
    """Split records into fixed, disjoint chronological year ranges."""
    if date_col not in frame.columns:
        raise ValueError(f"date column {date_col!r} is missing")
    dates = pd.to_datetime(frame[date_col], errors="raise")
    if dates.isna().any():
        raise ValueError("date values cannot be missing")
    years = dates.dt.year

    ranges = {
        "train": config.train_years,
        "validation": config.validation_years,
        "test": config.test_years,
    }
    partitions: dict[str, pd.DataFrame] = {}
    for name, (start_year, end_year) in ranges.items():
        mask = years.between(start_year, end_year, inclusive="both")
        partitions[name] = frame.loc[mask].copy().reset_index(drop=True)
        if partitions[name].empty:
            raise ValueError(f"{name} partition has no rows for configured years {start_year}-{end_year}")
    return partitions


def make_spatial_block_folds(
    frame: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    block_col: str = "spatial_block",
) -> tuple[tuple[str, ...], ...]:
    """Create deterministic, exhaustive folds over unique spatial blocks."""
    if block_col not in frame.columns or frame[block_col].isna().any():
        raise ValueError(f"{block_col} must exist and contain no missing values")
    blocks = sorted(frame[block_col].astype(str).unique().tolist())
    if not isinstance(n_splits, int) or n_splits < 2 or n_splits > len(blocks):
        raise ValueError("n_splits must be an integer between 2 and the number of spatial blocks")

    shuffled = np.asarray(blocks, dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    return tuple(tuple(str(block) for block in fold) for fold in np.array_split(shuffled, n_splits))


def buffered_spatiotemporal_split(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    held_out_blocks: Sequence[str],
    block_col: str = "spatial_block",
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    date_col: str = "date",
    buffer_degrees: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Build temporal partitions while holding out buffered spatial blocks.

    Training and validation use only non-held-out grid cells outside the
    expanded bounding boxes of held-out cells. Test data include only held-out
    blocks in the configured final test years.
    """
    if buffer_degrees < 0:
        raise ValueError("buffer_degrees cannot be negative")
    required = {block_col, latitude_col, longitude_col, date_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"required split columns are missing: {sorted(missing)}")
    if not held_out_blocks:
        raise ValueError("held_out_blocks cannot be empty")

    working = frame.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="raise")
    if working[[date_col, block_col, latitude_col, longitude_col]].isna().any().any():
        raise ValueError("date, block, latitude, and longitude values cannot be missing")
    working[latitude_col] = pd.to_numeric(working[latitude_col], errors="raise")
    working[longitude_col] = pd.to_numeric(working[longitude_col], errors="raise")
    if not np.isfinite(working[[latitude_col, longitude_col]].to_numpy(dtype=float)).all():
        raise ValueError("latitude and longitude must be finite numbers")

    held_out = set(str(block) for block in held_out_blocks)
    known_blocks = set(working[block_col].astype(str).unique())
    unknown = held_out - known_blocks
    if unknown:
        raise ValueError(f"held_out_blocks contains unknown values: {sorted(unknown)}")
    is_held_out = working[block_col].astype(str).isin(held_out)
    held_points = working.loc[is_held_out, [latitude_col, longitude_col]]

    in_buffer = pd.Series(False, index=working.index)
    for block in held_out:
        block_points = working.loc[
            working[block_col].astype(str) == block, [latitude_col, longitude_col]
        ]
        lat_min, lat_max = block_points[latitude_col].min(), block_points[latitude_col].max()
        lon_min, lon_max = block_points[longitude_col].min(), block_points[longitude_col].max()
        in_buffer |= (
            working[latitude_col].between(lat_min - buffer_degrees, lat_max + buffer_degrees)
            & working[longitude_col].between(lon_min - buffer_degrees, lon_max + buffer_degrees)
        )

    years = working[date_col].dt.year
    train_start, train_end = config.train_years
    val_start, val_end = config.validation_years
    test_start, test_end = config.test_years
    training_domain = ~is_held_out & ~in_buffer

    partitions = {
        "train": working.loc[
            years.between(train_start, train_end, inclusive="both") & training_domain
        ].copy(),
        "validation": working.loc[
            years.between(val_start, val_end, inclusive="both") & training_domain
        ].copy(),
        "test": working.loc[
            years.between(test_start, test_end, inclusive="both") & is_held_out
        ].copy(),
    }
    for name, partition in partitions.items():
        if partition.empty:
            raise ValueError(f"{name} partition is empty after applying time and spatial filters")
        partitions[name] = partition.reset_index(drop=True)
    return partitions
