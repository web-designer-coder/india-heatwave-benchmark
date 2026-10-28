"""Public data acquisition and validation for the IMD temperature archive."""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


IMD_TMAX_PAGE = "https://www.imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html"
IMD_TMAX_DOWNLOAD = "https://www.imdpune.gov.in/cmpg/Griddata/maxtemp.php"
IMD_GRID_SHAPE = (31, 31)
IMD_UNDEFINED_VALUE = 99.9
NASA_POWER_DAILY_API = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_PARAMETERS = (
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "WS2M",
    "PS",
    "PRECTOTCORR",
)
ERA5_ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
ERA5_DAILY_VARIABLES = (
    ("temperature_2m_mean", "t2m"),
    ("temperature_2m_max", "t2m_max"),
    ("temperature_2m_min", "t2m_min"),
    ("relative_humidity_2m_mean", "rh2m"),
    ("wind_speed_10m_mean", "ws10m"),
    ("surface_pressure_mean", "sp"),
    ("precipitation_sum", "tp"),
)
ERA5_ICECHUNK_BUCKET = "earthmover-icechunk-era5"
ERA5_ICECHUNK_PREFIX = "icechunkV2"
ERA5_ICECHUNK_GROUP = "single/temporal"
ERA5_ICECHUNK_REGISTRY = "https://registry.opendata.aws/earthmover-era5/"
ERA5_HOURLY_VARIABLES = ("t2m", "d2m", "u10", "v10", "sp", "tp")


def expected_imd_file_size(year: int) -> int:
    """Return the expected number of bytes in an IMD annual Tmax grid file."""
    _validate_imd_year(year)
    days = 366 if pd.Timestamp(year=year, month=12, day=31).day_of_year == 366 else 365
    return days * IMD_GRID_SHAPE[0] * IMD_GRID_SHAPE[1] * np.dtype("<f4").itemsize


def _validate_imd_year(year: int) -> None:
    if not isinstance(year, int) or year < 1951 or year > 2025:
        raise ValueError("IMD Tmax archive year must be between 1951 and 2025")


def read_imd_tmax_year(path: str | Path, year: int) -> xr.DataArray:
    """Read an IMD daily Tmax annual binary file into a georeferenced array."""
    _validate_imd_year(year)
    source = Path(path)
    expected_size = expected_imd_file_size(year)
    actual_size = source.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"unexpected IMD file size for {year}: got {actual_size} bytes, "
            f"expected {expected_size} bytes"
        )

    # Each raw record is stored as (latitude, longitude) in C order. Expose the
    # same conventional xarray order used by the IMD reference implementation.
    values = np.fromfile(source, dtype="<f4").reshape(
        (-1, IMD_GRID_SHAPE[1], IMD_GRID_SHAPE[0])
    )
    values[~np.isfinite(values) | np.isclose(values, IMD_UNDEFINED_VALUE)] = np.nan
    times = pd.date_range(f"{year}-01-01", periods=values.shape[0], freq="D")
    latitudes = np.arange(7.5, 38.0, 1.0)
    longitudes = np.arange(67.5, 98.0, 1.0)
    return xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
        name="tmax",
        attrs={
            "long_name": "daily gridded maximum temperature",
            "units": "degree_Celsius",
            "source": "India Meteorological Department 1 degree gridded Tmax archive",
            "year": year,
        },
    )


def load_imd_terrain_mask() -> xr.DataArray:
    """Load the documented 1-degree IMD terrain mask in lat/lon order.

    The bundled CSV is the transposed, text-form copy of imdlib 0.4.0's
    ``region_mask_100.npy`` (MIT licensed), included so label generation does
    not depend on private imdlib APIs at runtime. Codes: -1 ocean/outside,
    0 plains, 1 hilly, 2 coastal.
    """
    path = Path(__file__).with_name("imd_region_mask_1deg.csv")
    values = np.loadtxt(path, delimiter=",", dtype=np.int8)
    if values.shape != IMD_GRID_SHAPE or not set(np.unique(values)).issubset({-1, 0, 1, 2}):
        raise ValueError(f"invalid bundled IMD terrain mask: {path}")
    return xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.arange(7.5, 38.0, 1.0),
            "longitude": np.arange(67.5, 98.0, 1.0),
        },
        name="terrain_code",
        attrs={
            "flag_values": [-1, 0, 1, 2],
            "flag_meanings": "ocean_or_outside plains hilly coastal",
            "source": "imdlib 0.4.0 bundled region_mask_100.npy, transposed to latitude/longitude order",
            "license": "MIT",
        },
    )


def download_imd_tmax_year(
    year: int,
    destination_dir: str | Path,
    session: Any | None = None,
    overwrite: bool = False,
) -> Path:
    """Download and atomically save one official IMD annual Tmax file."""
    _validate_imd_year(year)
    import requests

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"MaxT_{year}.GRD"
    if target.exists() and not overwrite:
        read_imd_tmax_year(target, year)
        return target

    client = session if session is not None else requests.Session()
    response = client.post(
        IMD_TMAX_DOWNLOAD,
        data={"maxtemp": str(year)},
        timeout=(10, 120),
    )
    response.raise_for_status()
    disposition = response.headers.get("Content-Disposition", "")
    if f"MaxT_{year}.GRD".lower() not in disposition.lower():
        raise ValueError(f"IMD response did not identify the requested {year} grid file")
    expected_size = expected_imd_file_size(year)
    if len(response.content) != expected_size:
        raise ValueError(
            f"IMD response has {len(response.content)} bytes for {year}; "
            f"expected {expected_size}"
        )

    temporary = target.with_suffix(target.suffix + ".part")
    try:
        temporary.write_bytes(response.content)
        read_imd_tmax_year(temporary, year)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def build_file_manifest(
    path: str | Path,
    source_url: str,
    dataset: str,
    year: int,
    resolution_degrees: float,
) -> dict[str, Any]:
    """Create a JSON-serializable provenance record for one downloaded file."""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "file": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": digest.hexdigest(),
        "source_url": source_url,
        "dataset": dataset,
        "year": year,
        "resolution_degrees": resolution_degrees,
        "accessed_on": date.today().isoformat(),
    }


def parse_nasa_power_daily_response(
    payload: dict[str, Any],
    grid_id: str,
    latitude: float,
    longitude: float,
    parameters: tuple[str, ...] = NASA_POWER_PARAMETERS,
    expected_time_standard: str = "LST",
) -> pd.DataFrame:
    """Parse one NASA POWER daily point response into a date-indexed table."""
    try:
        response_parameters = payload["properties"]["parameter"]
        header = payload["header"]
        coordinates = payload["geometry"]["coordinates"]
    except (KeyError, TypeError) as exc:
        raise ValueError("NASA POWER response is missing required metadata or parameters") from exc

    if header.get("time_standard") != expected_time_standard:
        raise ValueError(
            f"NASA POWER response time standard must be {expected_time_standard}, "
            f"got {header.get('time_standard')!r}"
        )
    missing = set(parameters) - set(response_parameters)
    if missing:
        raise ValueError(f"NASA POWER response is missing parameter(s): {sorted(missing)}")

    date_keys = sorted(
        set().union(*(set(response_parameters[name]) for name in parameters))
    )
    if not date_keys:
        raise ValueError("NASA POWER response contains no daily values")
    fill_value = float(header.get("fill_value", -999.0))
    dates = pd.to_datetime(date_keys, format="%Y%m%d", errors="raise")
    row_count = len(date_keys)
    columns: dict[str, Any] = {
        "grid_id": np.repeat(grid_id, row_count),
        "date": dates,
        "latitude": np.full(row_count, float(latitude)),
        "longitude": np.full(row_count, float(longitude)),
        "api_response_latitude": np.full(row_count, float(coordinates[1])),
        "api_response_longitude": np.full(row_count, float(coordinates[0])),
    }
    for parameter in parameters:
        values = (
            pd.Series(response_parameters[parameter], dtype="float64")
            .reindex(date_keys)
                .to_numpy(dtype=float, copy=True)
        )
        values[np.isclose(values, fill_value)] = np.nan
        columns[parameter.lower()] = values
    return pd.DataFrame(columns)


def download_nasa_power_daily_point(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    grid_id: str,
    parameters: tuple[str, ...] = NASA_POWER_PARAMETERS,
    time_standard: str = "LST",
    session: Any | None = None,
    timeout: float = 60.0,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch one unique NASA POWER point series for a fixed variable list."""
    import requests

    if not np.isfinite([latitude, longitude]).all():
        raise ValueError("latitude and longitude must be finite numbers")
    if not parameters or len(parameters) > 20 or len(set(parameters)) != len(parameters):
        raise ValueError("parameters must be a unique non-empty list of at most 20 variables")
    if time_standard not in {"LST", "UTC"}:
        raise ValueError("time_standard must be 'LST' or 'UTC'")

    start = pd.to_datetime(start_date, errors="raise")
    end = pd.to_datetime(end_date, errors="raise")
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    target = Path(cache_path) if cache_path is not None else None
    if target is not None and target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
    else:
        client = session if session is not None else requests.Session()
        response = client.get(
            NASA_POWER_DAILY_API,
            params={
                "parameters": ",".join(parameters),
                "community": "RE",
                "longitude": float(longitude),
                "latitude": float(latitude),
                "start": start.strftime("%Y%m%d"),
                "end": end.strftime("%Y%m%d"),
                "format": "JSON",
                "time-standard": time_standard,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    table = parse_nasa_power_daily_response(
        payload,
        grid_id=grid_id,
        latitude=latitude,
        longitude=longitude,
        parameters=parameters,
        expected_time_standard=time_standard,
    )
    table = table.loc[table["date"].between(start.normalize(), end.normalize())].reset_index(drop=True)
    expected_dates = pd.date_range(start.normalize(), end.normalize(), freq="D")
    if not pd.DatetimeIndex(table["date"]).equals(expected_dates):
        raise ValueError("NASA POWER response does not provide complete daily coverage")
    if target is not None and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            if "response" not in locals():
                raise RuntimeError("internal error: missing NASA POWER response for cache write")
            temporary.write_bytes(response.content)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return table


def download_nasa_power_grid(
    points: pd.DataFrame,
    destination_dir: str | Path,
    start_date: str,
    end_date: str,
    parameters: tuple[str, ...] = NASA_POWER_PARAMETERS,
    max_workers: int = 5,
    retries: int = 4,
    retry_delay_seconds: float = 1.0,
) -> pd.DataFrame:
    """Download/cache one NASA POWER daily JSON response per unique grid point.

    NASA's public API guidance recommends no more than five concurrent
    requests. Successful point files are retained if another point fails, so
    rerunning resumes from the cache.
    """
    required = {"grid_id", "latitude", "longitude"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"point table is missing columns: {sorted(missing)}")
    if points.empty or points[list(required)].isna().any().any():
        raise ValueError("point table must contain valid, non-empty locations")
    if points["grid_id"].duplicated().any():
        raise ValueError("grid_id values must be unique")
    if not isinstance(max_workers, int) or max_workers < 1 or max_workers > 5:
        raise ValueError("max_workers must be between 1 and 5")
    if not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a non-negative integer")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds cannot be negative")

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records = points[["grid_id", "latitude", "longitude"]].copy()
    records["grid_id"] = records["grid_id"].astype(str)
    if not records["grid_id"].map(lambda value: re.fullmatch(r"[A-Za-z0-9_.-]+", value) is not None).all():
        raise ValueError("grid_id values may contain only letters, numbers, dots, hyphens, and underscores")

    def fetch(record: Any) -> dict[str, Any]:
        grid_id = str(record.grid_id)
        cache = destination / f"{grid_id}.json"
        was_cached = cache.exists()
        for attempt in range(retries + 1):
            try:
                table = download_nasa_power_daily_point(
                    latitude=float(record.latitude),
                    longitude=float(record.longitude),
                    start_date=start_date,
                    end_date=end_date,
                    grid_id=grid_id,
                    parameters=parameters,
                    cache_path=cache,
                )
                return {
                    "grid_id": grid_id,
                    "latitude": float(record.latitude),
                    "longitude": float(record.longitude),
                    "api_response_latitude": float(table["api_response_latitude"].iloc[0]),
                    "api_response_longitude": float(table["api_response_longitude"].iloc[0]),
                    "status": "cached" if was_cached else "downloaded",
                    "daily_rows": len(table),
                    "start_date": table["date"].min().date().isoformat(),
                    "end_date": table["date"].max().date().isoformat(),
                    "parameters": ",".join(parameters),
                    "time_standard": "LST",
                    "cache_file": cache.name,
                    "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                    "error": "",
                }
            except Exception as exc:  # Report failed grid IDs and allow safe resume.
                if attempt >= retries:
                    return {
                        "grid_id": grid_id,
                        "latitude": float(record.latitude),
                        "longitude": float(record.longitude),
                        "api_response_latitude": np.nan,
                        "api_response_longitude": np.nan,
                        "status": "failed",
                        "daily_rows": 0,
                        "start_date": pd.to_datetime(start_date).date().isoformat(),
                        "end_date": pd.to_datetime(end_date).date().isoformat(),
                        "parameters": ",".join(parameters),
                        "time_standard": "LST",
                        "cache_file": cache.name,
                        "sha256": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                time.sleep(retry_delay_seconds * (2**attempt))

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch, record) for record in records.itertuples(index=False)]
        for future in as_completed(futures):
            results.append(future.result())
    manifest = pd.DataFrame(results).sort_values("grid_id", kind="stable").reset_index(drop=True)
    manifest.to_csv(destination / "download_manifest.csv", index=False)
    return manifest


def parse_era5_daily_response(
    payload: dict[str, Any],
    grid_id: str,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """Parse Open-Meteo's explicit ERA5 daily response into benchmark inputs."""
    if payload.get("error"):
        raise ValueError(f"ERA5 API error: {payload.get('reason', 'unknown error')}")
    try:
        daily = payload["daily"]
        units = payload["daily_units"]
        source_latitude = float(payload["latitude"])
        source_longitude = float(payload["longitude"])
        timezone = payload["timezone"]
        raw_dates = daily["time"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ERA5 response is missing required metadata or daily fields") from exc
    if timezone != "Asia/Kolkata":
        raise ValueError(f"ERA5 response timezone must be Asia/Kolkata, got {timezone!r}")
    dates = pd.DatetimeIndex(pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="raise"))
    if dates.empty or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("ERA5 response dates must be non-empty, unique, and increasing")
    row_count = len(dates)
    columns: dict[str, Any] = {
        "grid_id": np.repeat(str(grid_id), row_count),
        "date": dates,
        "latitude": np.full(row_count, float(latitude)),
        "longitude": np.full(row_count, float(longitude)),
        "api_response_latitude": np.full(row_count, source_latitude),
        "api_response_longitude": np.full(row_count, source_longitude),
        "api_response_elevation": np.full(row_count, float(payload.get("elevation", np.nan))),
    }
    for variable, feature_name in ERA5_DAILY_VARIABLES:
        values = daily.get(variable)
        if values is None or len(values) != row_count:
            raise ValueError(f"ERA5 response is missing daily variable {variable!r}")
        columns[feature_name] = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    result = pd.DataFrame(columns)
    result.attrs["units"] = {feature: str(units.get(variable, "")) for variable, feature in ERA5_DAILY_VARIABLES}
    return result


def aggregate_era5_hourly_to_daily(
    hourly: pd.DataFrame,
    start_date: str,
    end_date: str,
    timezone: str = "Asia/Kolkata",
) -> pd.DataFrame:
    """Aggregate direct ERA5 hourly fields into complete local-day predictors.

    ERA5 timestamps are UTC. Each record is assigned to its civil date after
    conversion to ``timezone``; an India local date therefore contains the 24
    UTC-hourly samples from 19:00 UTC on the previous date through 18:00 UTC.
    Relative humidity is calculated hourly from 2 m temperature and dewpoint
    with the Magnus saturation-vapour-pressure approximation.
    """
    required = {
        "valid_time",
        "grid_id",
        "latitude",
        "longitude",
        "t2m",
        "d2m",
        "u10",
        "v10",
        "sp",
        "tp",
    }
    missing = required - set(hourly.columns)
    if missing:
        raise ValueError(f"direct ERA5 hourly frame is missing columns: {sorted(missing)}")
    if hourly.empty:
        raise ValueError("direct ERA5 hourly frame must be non-empty")
    start = pd.to_datetime(start_date, errors="raise").normalize()
    end = pd.to_datetime(end_date, errors="raise").normalize()
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    frame = hourly.copy()
    timestamps = pd.to_datetime(frame["valid_time"], utc=True, errors="raise")
    frame["valid_time"] = timestamps
    if frame.duplicated(["grid_id", "valid_time"]).any():
        raise ValueError("direct ERA5 hourly timestamps must be unique within each grid cell")
    frame["date"] = (
        timestamps.dt.tz_convert(timezone).dt.normalize().dt.tz_localize(None)
    )
    frame = frame.loc[frame["date"].between(start, end)].copy()
    if frame.empty:
        raise ValueError("direct ERA5 hourly frame contains no records in the requested dates")

    variable_columns = ["t2m", "d2m", "u10", "v10", "sp", "tp"]
    for column in variable_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[variable_columns].to_numpy(dtype=float)).all():
        raise ValueError("direct ERA5 hourly predictors must be finite")

    grouped = frame.groupby(["grid_id", "date"], sort=True, observed=True)
    row_counts = grouped.size()
    if not row_counts.eq(24).all():
        raise ValueError("each requested local date must contain exactly 24 hourly ERA5 records")
    if grouped["valid_time"].nunique().ne(24).any():
        raise ValueError("each grid cell and local date must contain 24 unique hourly ERA5 timestamps")

    frame["t2m_c"] = frame["t2m"] - 273.15
    frame["d2m_c"] = frame["d2m"] - 273.15
    es_t = 6.1121 * np.exp(17.502 * frame["t2m_c"] / (240.97 + frame["t2m_c"]))
    es_td = 6.1121 * np.exp(17.502 * frame["d2m_c"] / (240.97 + frame["d2m_c"]))
    frame["rh2m_hourly"] = np.clip(100.0 * es_td / es_t, 0.0, 100.0)
    frame["ws10m_hourly"] = np.hypot(frame["u10"], frame["v10"])
    frame["sp_hpa"] = frame["sp"] / 100.0
    frame["tp_mm"] = frame["tp"] * 1000.0

    aggregations: dict[str, tuple[str, str]] = {
        "t2m": ("t2m_c", "mean"),
        "t2m_max": ("t2m_c", "max"),
        "t2m_min": ("t2m_c", "min"),
        "rh2m": ("rh2m_hourly", "mean"),
        "ws10m": ("ws10m_hourly", "mean"),
        "sp": ("sp_hpa", "mean"),
        "tp": ("tp_mm", "sum"),
        "latitude": ("latitude", "first"),
        "longitude": ("longitude", "first"),
    }
    result = grouped.agg(**aggregations).reset_index()
    result["api_response_latitude"] = result["latitude"]
    result["api_response_longitude"] = result["longitude"]
    if "elevation_m" in frame:
        elevation = frame.groupby(["grid_id", "date"], sort=True, observed=True)["elevation_m"].first()
        result["api_response_elevation"] = elevation.to_numpy(dtype=float)
    else:
        result["api_response_elevation"] = np.nan

    expected_dates = pd.date_range(start, end, freq="D")
    for grid_id, rows in result.groupby("grid_id", sort=False, observed=True):
        if not pd.DatetimeIndex(rows["date"]).equals(expected_dates):
            raise ValueError(f"direct ERA5 hourly records do not fully cover {grid_id} from {start.date()} to {end.date()}")
    result.attrs["units"] = {
        "t2m": "°C",
        "t2m_max": "°C",
        "t2m_min": "°C",
        "rh2m": "%",
        "ws10m": "m/s",
        "sp": "hPa",
        "tp": "mm",
    }
    result.attrs["timezone"] = timezone
    return result.sort_values(["grid_id", "date"], kind="stable").reset_index(drop=True)


def group_era5_points_by_spatial_chunk(
    points: pd.DataFrame,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    chunk_shape: tuple[int, int] = (12, 12),
) -> list[pd.DataFrame]:
    """Group requested locations by their nearest native ERA5 Zarr tile."""
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    if latitudes.ndim != 1 or longitudes.ndim != 1 or not len(latitudes) or not len(longitudes):
        raise ValueError("ERA5 coordinate arrays must be non-empty one-dimensional values")
    if len(chunk_shape) != 2 or min(chunk_shape) < 1:
        raise ValueError("ERA5 spatial chunk dimensions must be positive")
    if not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all():
        raise ValueError("ERA5 coordinate arrays must contain finite values")
    if not np.isfinite(points[["latitude", "longitude"]].to_numpy(dtype=float)).all():
        raise ValueError("requested ERA5 points must have finite coordinates")

    latitude_positions = np.abs(
        points["latitude"].astype(float).to_numpy()[:, None] - latitudes[None, :]
    ).argmin(axis=1)
    requested_longitudes = points["longitude"].astype(float).to_numpy() % 360.0
    source_longitudes = longitudes % 360.0
    longitude_distances = np.abs(
        (source_longitudes[None, :] - requested_longitudes[:, None] + 180.0) % 360.0 - 180.0
    )
    longitude_positions = longitude_distances.argmin(axis=1)

    tagged = points.copy()
    tagged["_era5_lat_chunk"] = latitude_positions // chunk_shape[0]
    tagged["_era5_lon_chunk"] = longitude_positions // chunk_shape[1]
    return [
        group.drop(columns=["_era5_lat_chunk", "_era5_lon_chunk"])
        .sort_values("grid_id", kind="stable")
        .reset_index(drop=True)
        for _, group in tagged.groupby(["_era5_lat_chunk", "_era5_lon_chunk"], sort=True)
    ]


def select_era5_icechunk_hourly_window(
    dataset: xr.Dataset,
    year: int,
    points: pd.DataFrame,
) -> xr.Dataset:
    """Select a season before spatial points to keep vectorized indexers bounded."""
    time_window = dataset[list(ERA5_HOURLY_VARIABLES)].sel(
        valid_time=slice(f"{year}-02-21T19:00:00", f"{year}-06-30T18:00:00")
    )
    grid_ids = points["grid_id"].astype(str).tolist()
    latitudes = xr.DataArray(
        points["latitude"].astype(float).to_numpy(),
        dims=("grid_id",),
        coords={"grid_id": grid_ids},
    )
    longitudes = xr.DataArray(
        points["longitude"].astype(float).to_numpy(),
        dims=("grid_id",),
        coords={"grid_id": grid_ids},
    )
    return time_window.sel(
        latitude=latitudes,
        longitude=longitudes,
        method="nearest",
    )


def open_era5_icechunk_dataset(snapshot_id: str | None = None) -> tuple[xr.Dataset, str]:
    """Open the public Earthmover ERA5 temporal layout at a fixed snapshot."""
    try:
        import icechunk
    except ImportError as exc:
        raise RuntimeError(
            "Install optional dependencies with `pip install -e '.[era5-cloud]'`."
        ) from exc

    storage = icechunk.s3_storage(
        bucket=ERA5_ICECHUNK_BUCKET,
        prefix=ERA5_ICECHUNK_PREFIX,
        region="us-east-1",
        anonymous=True,
    )
    repository = icechunk.Repository.open(storage)
    selected_snapshot = str(snapshot_id or repository.lookup_branch("main"))
    session = repository.readonly_session(snapshot_id=selected_snapshot)
    dataset = xr.open_zarr(
        session.store,
        group=ERA5_ICECHUNK_GROUP,
        consolidated=False,
        chunks=None,
    )
    # The Icechunk store retains the read-only session for the lifetime of the
    # open dataset; Icechunk Session currently has no public close() method.
    return dataset, selected_snapshot


def _era5_icechunk_daily_payload(daily: pd.DataFrame, snapshot_id: str) -> dict[str, Any]:
    """Serialize one point's local-day summaries in the existing ERA5 cache format."""
    ordered = daily.sort_values("date", kind="stable")
    values: dict[str, Any] = {
        "time": ordered["date"].dt.strftime("%Y-%m-%d").tolist(),
    }
    for variable, feature in ERA5_DAILY_VARIABLES:
        values[variable] = ordered[feature].astype(float).tolist()
    return {
        "latitude": float(ordered["latitude"].iloc[0]),
        "longitude": float(ordered["longitude"].iloc[0]),
        "timezone": "Asia/Kolkata",
        "daily_units": {
            "time": "iso8601",
            "temperature_2m_mean": "°C",
            "temperature_2m_max": "°C",
            "temperature_2m_min": "°C",
            "relative_humidity_2m_mean": "%",
            "wind_speed_10m_mean": "m/s",
            "surface_pressure_mean": "hPa",
            "precipitation_sum": "mm",
        },
        "daily": values,
        "source_metadata": {
            "dataset": "Earthmover ERA5 Icechunk public mirror",
            "registry_url": ERA5_ICECHUNK_REGISTRY,
            "bucket": ERA5_ICECHUNK_BUCKET,
            "prefix": ERA5_ICECHUNK_PREFIX,
            "group": ERA5_ICECHUNK_GROUP,
            "snapshot_id": snapshot_id,
            "license": "CC-BY-4.0",
            "raw_hourly_variables": list(ERA5_HOURLY_VARIABLES),
        },
    }


def download_era5_icechunk_seasonal_grid(
    points: pd.DataFrame,
    destination_dir: str | Path,
    start_year: int,
    end_year: int,
    snapshot_id: str | None = None,
    progress_callback: Any | None = None,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Extract daily India-grid predictors from public hourly ERA5 Zarr data.

    Only Feb 22-Jun 30 is retained in each year, covering March-June issue dates
    and their seven preceding daily lags. All windows are read from one immutable
    Icechunk snapshot and padded to full-year JSON caches for the existing loader.
    """
    if not isinstance(start_year, int) or not isinstance(end_year, int) or start_year > end_year:
        raise ValueError("start_year and end_year must be ordered integer years")
    required = {"grid_id", "latitude", "longitude"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"point table is missing columns: {sorted(missing)}")
    if points.empty or points[list(required)].isna().any().any():
        raise ValueError("point table must contain valid, non-empty locations")
    if points["grid_id"].duplicated().any():
        raise ValueError("grid_id values must be unique")
    if not isinstance(max_workers, int) or not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be an integer between 1 and 8")

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source_manifest_path = destination / "source_manifest.json"
    pinned_snapshot = snapshot_id
    if source_manifest_path.exists():
        saved = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        saved_snapshot = str(saved["snapshot_id"])
        if pinned_snapshot is not None and str(pinned_snapshot) != saved_snapshot:
            raise ValueError("requested ERA5 snapshot differs from the snapshot already cached here")
        pinned_snapshot = saved_snapshot

    dataset, resolved_snapshot = open_era5_icechunk_dataset(pinned_snapshot)
    resolved_snapshot = str(resolved_snapshot)
    if pinned_snapshot is not None and resolved_snapshot != str(pinned_snapshot):
        dataset.close()
        raise ValueError("ERA5 dataset opener did not return the requested immutable snapshot")
    missing_vars = set(ERA5_HOURLY_VARIABLES) - set(dataset.data_vars)
    if missing_vars:
        dataset.close()
        raise ValueError(f"ERA5 cloud dataset is missing fields: {sorted(missing_vars)}")
    missing_coords = {"valid_time", "latitude", "longitude"} - set(dataset.coords)
    if missing_coords:
        dataset.close()
        raise ValueError(f"ERA5 cloud dataset is missing coordinates: {sorted(missing_coords)}")

    source_manifest = {
        "accessed_on": date.today().isoformat(),
        "dataset": "Earthmover ERA5 Icechunk public mirror",
        "registry_url": ERA5_ICECHUNK_REGISTRY,
        "bucket": ERA5_ICECHUNK_BUCKET,
        "prefix": ERA5_ICECHUNK_PREFIX,
        "group": ERA5_ICECHUNK_GROUP,
        "snapshot_id": resolved_snapshot,
        "license": "CC-BY-4.0",
        "variables": list(ERA5_HOURLY_VARIABLES),
        "native_resolution_degrees": 0.25,
        "time_standard": "UTC hourly",
        "daily_timezone": "Asia/Kolkata",
        "daily_window": "February 22 through June 30, inclusive, annually",
        "daily_aggregation": (
            "24 hourly records per local date; mean/min/max temperature; hourly Magnus "
            "relative humidity mean; scalar wind-speed mean; mean surface pressure; "
            "sum of hourly total-precipitation accumulations"
        ),
    }
    manifest_chunks = dataset["t2m"].encoding.get("chunks")
    source_manifest["spatial_chunk_shape"] = (
        list(manifest_chunks[-2:]) if manifest_chunks else [12, 12]
    )
    source_manifest["max_spatial_workers"] = max_workers
    manifest_temp = source_manifest_path.with_suffix(".json.part")
    manifest_temp.write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    manifest_temp.replace(source_manifest_path)

    ordered = points.sort_values("grid_id", kind="stable").reset_index(drop=True)
    windows_root = destination / "windows"
    years = list(range(start_year, end_year + 1))
    date_windows: dict[int, pd.DatetimeIndex] = {}
    try:
        for year_index, year in enumerate(years, start=1):
            year_dir = windows_root / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)
            expected_dates = pd.date_range(f"{year}-02-22", f"{year}-06-30", freq="D")
            date_windows[year] = expected_dates
            paths = {str(r.grid_id): year_dir / f"{r.grid_id}.json" for r in ordered.itertuples(index=False)}
            cache_status: dict[str, str] = {}
            for record in ordered.itertuples(index=False):
                grid_id = str(record.grid_id)
                try:
                    payload = json.loads(paths[grid_id].read_text(encoding="utf-8"))
                    if payload.get("source_metadata", {}).get("snapshot_id") != resolved_snapshot:
                        raise ValueError("cache snapshot differs")
                    parsed = parse_era5_daily_response(
                        payload, grid_id, float(record.latitude), float(record.longitude)
                    )
                    if not pd.DatetimeIndex(parsed["date"]).equals(expected_dates):
                        raise ValueError("cache has incomplete seasonal dates")
                    features = [feature for _, feature in ERA5_DAILY_VARIABLES]
                    if not np.isfinite(parsed[features].to_numpy(dtype=float)).all():
                        raise ValueError("cache has missing predictor values")
                    cache_status[grid_id] = "cached"
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    cache_status[grid_id] = "pending"

            missing_points = ordered.loc[
                ordered["grid_id"].astype(str).isin(
                    [grid_id for grid_id, status in cache_status.items() if status == "pending"]
                )
            ].copy()
            encoding_chunks = dataset["t2m"].encoding.get("chunks")
            if not encoding_chunks and getattr(dataset["t2m"].data, "chunks", None):
                encoding_chunks = tuple(size[0] for size in dataset["t2m"].data.chunks)
            spatial_chunk_shape = tuple(encoding_chunks[-2:]) if encoding_chunks else (12, 12)
            spatial_groups = group_era5_points_by_spatial_chunk(
                missing_points,
                dataset["latitude"].to_numpy(),
                dataset["longitude"].to_numpy(),
                chunk_shape=spatial_chunk_shape,
            ) if not missing_points.empty else []
            def extract_spatial_group(group_points: pd.DataFrame) -> dict[str, bytes]:
                selected = select_era5_icechunk_hourly_window(dataset, year, group_points)
                hourly = selected.to_dataframe().reset_index()
                hourly["grid_id"] = hourly["grid_id"].astype(str)
                daily = aggregate_era5_hourly_to_daily(
                    hourly, start_date=f"{year}-02-22", end_date=f"{year}-06-30"
                )
                expected_grid_ids = set(group_points["grid_id"].astype(str))
                if set(daily["grid_id"].astype(str)) != expected_grid_ids:
                    raise ValueError(f"ERA5 cloud extraction for {year} omitted cells in one spatial chunk")
                payload_bytes: dict[str, bytes] = {}
                for record in group_points.itertuples(index=False):
                    grid_id = str(record.grid_id)
                    cell_daily = daily.loc[daily["grid_id"].astype(str).eq(grid_id)]
                    payload = _era5_icechunk_daily_payload(cell_daily, resolved_snapshot)
                    payload_bytes[grid_id] = json.dumps(
                        payload, separators=(",", ":"), ensure_ascii=False
                    ).encode("utf-8")
                return payload_bytes

            if spatial_groups:
                with ThreadPoolExecutor(max_workers=min(max_workers, len(spatial_groups))) as executor:
                    futures = {
                        executor.submit(extract_spatial_group, group_points): group_index
                        for group_index, group_points in enumerate(spatial_groups, start=1)
                    }
                    completed_groups = 0
                    for future in as_completed(futures):
                        group_payloads = future.result()
                        for grid_id, encoded in group_payloads.items():
                            path = paths[grid_id]
                            temporary = path.with_suffix(".json.part")
                            temporary.write_bytes(encoded)
                            temporary.replace(path)
                            cache_status[grid_id] = "downloaded"
                        completed_groups += 1
                        if progress_callback is not None:
                            progress_callback(
                                year_index,
                                len(years),
                                year,
                                completed_groups,
                                len(spatial_groups),
                                sum(status != "pending" for status in cache_status.values()),
                                "",
                            )

            rows = []
            for record in ordered.itertuples(index=False):
                grid_id = str(record.grid_id)
                path = paths[grid_id]
                encoded = path.read_bytes()
                rows.append({
                    "grid_id": grid_id, "status": cache_status[grid_id], "daily_rows": len(expected_dates),
                    "cache_file": path.name, "sha256": hashlib.sha256(encoded).hexdigest(),
                    "source_snapshot_id": resolved_snapshot,
                })
            pd.DataFrame(rows).to_csv(year_dir / "download_manifest.csv", index=False)
            if progress_callback is not None and not spatial_groups:
                progress_callback(year_index, len(years), year, 0, 0, len(ordered), "")

        dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
        positions = {value.strftime("%Y-%m-%d"): index for index, value in enumerate(dates)}
        variables = [variable for variable, _ in ERA5_DAILY_VARIABLES]
        manifest_rows: list[dict[str, Any]] = []
        for record in ordered.itertuples(index=False):
            grid_id = str(record.grid_id)
            payloads = [
                json.loads((windows_root / str(year) / f"{grid_id}.json").read_text(encoding="utf-8"))
                for year in years
            ]
            daily_values: dict[str, Any] = {"time": dates.strftime("%Y-%m-%d").tolist()}
            for variable in variables:
                daily_values[variable] = [None] * len(dates)
            for payload in payloads:
                for index, date_value in enumerate(payload["daily"]["time"]):
                    target_index = positions[date_value]
                    for variable in variables:
                        daily_values[variable][target_index] = payload["daily"][variable][index]
            combined = dict(payloads[0])
            combined["daily"] = daily_values
            combined["benchmark_predictor_window"] = source_manifest["daily_window"]
            cache = destination / f"{grid_id}.json"
            encoded = json.dumps(combined, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            temporary = cache.with_suffix(".json.part")
            temporary.write_bytes(encoded)
            temporary.replace(cache)
            parsed = parse_era5_daily_response(
                combined, grid_id, float(record.latitude), float(record.longitude)
            )
            if not pd.DatetimeIndex(parsed["date"]).equals(dates):
                raise ValueError(f"combined ERA5 cache for {grid_id} has invalid calendar coverage")
            manifest_rows.append({
                "grid_id": grid_id,
                "latitude": float(record.latitude),
                "longitude": float(record.longitude),
                "api_response_latitude": float(parsed["api_response_latitude"].iloc[0]),
                "api_response_longitude": float(parsed["api_response_longitude"].iloc[0]),
                "status": "downloaded",
                "daily_rows": sum(len(window) for window in date_windows.values()),
                "cache_calendar_rows": len(dates),
                "start_date": dates[0].date().isoformat(),
                "end_date": dates[-1].date().isoformat(),
                "predictor_window": source_manifest["daily_window"],
                "model": "era5",
                "timezone": "Asia/Kolkata",
                "variables": ",".join(variables),
                "cache_file": cache.name,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "source_snapshot_id": resolved_snapshot,
                "source_manifest": source_manifest_path.name,
                "error": "",
            })
        manifest = pd.DataFrame(manifest_rows).sort_values("grid_id", kind="stable").reset_index(drop=True)
        manifest.to_csv(destination / "download_manifest.csv", index=False)
        return manifest
    finally:
        dataset.close()


def fetch_era5_daily_batch(
    points: pd.DataFrame,
    start_date: str,
    end_date: str,
    session: Any | None = None,
    timeout: float = 240.0,
) -> tuple[dict[str, dict[str, Any]], bytes]:
    """Fetch a multi-coordinate ERA5 daily response and map elements by ID."""
    import requests

    required = {"grid_id", "latitude", "longitude"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"point table is missing columns: {sorted(missing)}")
    if points.empty or points[list(required)].isna().any().any():
        raise ValueError("points must contain valid, non-empty locations")
    if points["grid_id"].duplicated().any():
        raise ValueError("grid_id values must be unique")
    latitudes = points["latitude"].astype(float).tolist()
    longitudes = points["longitude"].astype(float).tolist()
    if not np.isfinite(np.asarray([*latitudes, *longitudes], dtype=float)).all():
        raise ValueError("coordinates must be finite")
    start = pd.to_datetime(start_date, errors="raise").normalize()
    end = pd.to_datetime(end_date, errors="raise").normalize()
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    client = session if session is not None else requests.Session()
    response = client.get(
        ERA5_ARCHIVE_API,
        params={
            "latitude": ",".join(f"{value:g}" for value in latitudes),
            "longitude": ",".join(f"{value:g}" for value in longitudes),
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "daily": ",".join(variable for variable, _ in ERA5_DAILY_VARIABLES),
            "timezone": "Asia/Kolkata",
            "models": "era5",
            "cell_selection": "nearest",
            "elevation": ",".join("nan" for _ in latitudes),
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    raw_body = response.content
    payload = response.json()
    if len(points) == 1 and isinstance(payload, dict):
        payloads = [payload]
    elif isinstance(payload, list):
        payloads = payload
    else:
        raise ValueError("ERA5 multi-location response must be a list of location objects")
    if len(payloads) != len(points) or not all(isinstance(item, dict) for item in payloads):
        raise ValueError("ERA5 response location count does not match requested coordinates")
    expected_dates = pd.date_range(start, end, freq="D")
    mapped: dict[str, dict[str, Any]] = {}
    for record, item in zip(points.itertuples(index=False), payloads, strict=True):
        table = parse_era5_daily_response(
            item,
            grid_id=str(record.grid_id),
            latitude=float(record.latitude),
            longitude=float(record.longitude),
        )
        if not pd.DatetimeIndex(table["date"]).equals(expected_dates):
            raise ValueError(f"ERA5 response for {record.grid_id} lacks complete daily coverage")
        mapped[str(record.grid_id)] = item
    return mapped, raw_body


def download_era5_daily_point(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    grid_id: str,
    session: Any | None = None,
    timeout: float = 120.0,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch and validate one consistent ERA5 daily series through Open-Meteo."""
    import requests

    if not np.isfinite([latitude, longitude]).all():
        raise ValueError("latitude and longitude must be finite numbers")
    start = pd.to_datetime(start_date, errors="raise").normalize()
    end = pd.to_datetime(end_date, errors="raise").normalize()
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    target = Path(cache_path) if cache_path is not None else None
    response = None
    if target is not None and target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
    else:
        client = session if session is not None else requests.Session()
        response = client.get(
            ERA5_ARCHIVE_API,
            params={
                "latitude": float(latitude),
                "longitude": float(longitude),
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "daily": ",".join(variable for variable, _ in ERA5_DAILY_VARIABLES),
                "timezone": "Asia/Kolkata",
                "models": "era5",
                "cell_selection": "nearest",
                "elevation": "nan",
                "temperature_unit": "celsius",
                "wind_speed_unit": "ms",
                "precipitation_unit": "mm",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    table = parse_era5_daily_response(payload, grid_id, latitude, longitude)
    expected_dates = pd.date_range(start, end, freq="D")
    if not pd.DatetimeIndex(table["date"]).equals(expected_dates):
        raise ValueError("ERA5 response does not provide complete daily coverage")
    if target is not None and not target.exists():
        if response is None:
            raise RuntimeError("internal error: missing ERA5 HTTP response for cache write")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            temporary.write_bytes(response.content)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return table


def download_era5_grid(
    points: pd.DataFrame,
    destination_dir: str | Path,
    start_date: str,
    end_date: str,
    max_workers: int = 1,
    retries: int = 4,
    retry_delay_seconds: float = 1.0,
    batch_size: int = 5,
    progress_callback: Any | None = None,
    request_interval_seconds: float = 1.0,
) -> pd.DataFrame:
    """Download or resume auditable ERA5 daily point responses and manifest."""
    required = {"grid_id", "latitude", "longitude"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"point table is missing columns: {sorted(missing)}")
    if points.empty or points[list(required)].isna().any().any():
        raise ValueError("point table must contain valid, non-empty locations")
    if points["grid_id"].duplicated().any():
        raise ValueError("grid_id values must be unique")
    if max_workers != 1:
        raise ValueError("ERA5 requests must be sequential (max_workers=1) to avoid API throttling")
    if not isinstance(retries, int) or retries < 0 or retry_delay_seconds < 0 or request_interval_seconds < 0:
        raise ValueError("retries and request delays must be non-negative")
    if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 5:
        raise ValueError("batch_size must be between 1 and 5 locations")

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    previous_manifest_path = destination / "download_manifest.csv"
    if previous_manifest_path.is_file():
        try:
            previous_manifest = pd.read_csv(previous_manifest_path).set_index("grid_id")
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            previous_manifest = pd.DataFrame()
    else:
        previous_manifest = pd.DataFrame()
    records = points[["grid_id", "latitude", "longitude"]].copy()
    records["grid_id"] = records["grid_id"].astype(str)
    if not records["grid_id"].map(lambda value: re.fullmatch(r"[A-Za-z0-9_.-]+", value) is not None).all():
        raise ValueError("grid_id values may contain only letters, numbers, dots, hyphens, and underscores")

    def cached_record(record: Any) -> dict[str, Any] | None:
        grid_id = str(record.grid_id)
        cache = destination / f"{grid_id}.json"
        if not cache.exists():
            return None
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            table = parse_era5_daily_response(
                payload, grid_id, float(record.latitude), float(record.longitude)
            )
            expected_dates = pd.date_range(start_date, end_date, freq="D")
            if not pd.DatetimeIndex(table["date"]).equals(expected_dates):
                return None
            return {
                "grid_id": grid_id,
                "latitude": float(record.latitude),
                "longitude": float(record.longitude),
                "api_response_latitude": float(table["api_response_latitude"].iloc[0]),
                "api_response_longitude": float(table["api_response_longitude"].iloc[0]),
                "api_response_elevation": float(table["api_response_elevation"].iloc[0]),
                "status": "cached",
                "daily_rows": len(table),
                "start_date": table["date"].min().date().isoformat(),
                "end_date": table["date"].max().date().isoformat(),
                "model": "era5",
                "timezone": "Asia/Kolkata",
                "variables": ",".join(variable for variable, _ in ERA5_DAILY_VARIABLES),
                "cache_file": cache.name,
                "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "source_batch_file": (
                    str(previous_manifest.loc[grid_id, "source_batch_file"])
                    if grid_id in previous_manifest.index and pd.notna(previous_manifest.loc[grid_id, "source_batch_file"])
                    else ""
                ),
                "source_batch_sha256": (
                    str(previous_manifest.loc[grid_id, "source_batch_sha256"])
                    if grid_id in previous_manifest.index and pd.notna(previous_manifest.loc[grid_id, "source_batch_sha256"])
                    else ""
                ),
                "error": "",
            }
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    results = []
    missing_records = []
    for record in records.itertuples(index=False):
        existing = cached_record(record)
        if existing is None:
            missing_records.append(record)
        else:
            results.append(existing)

    batch_directory = destination / "batches"
    batch_directory.mkdir(parents=True, exist_ok=True)
    total_batches = (len(missing_records) + batch_size - 1) // batch_size
    for batch_index, offset in enumerate(range(0, len(missing_records), batch_size), start=1):
        batch_records = missing_records[offset : offset + batch_size]
        batch_points = pd.DataFrame.from_records(
            [(row.grid_id, row.latitude, row.longitude) for row in batch_records],
            columns=["grid_id", "latitude", "longitude"],
        )
        batch_file: Path | None = None
        last_error: Exception | None = None
        batch_succeeded = False
        daily_quota_exhausted = False
        for attempt in range(retries + 1):
            try:
                if batch_index > 1 and attempt == 0:
                    time.sleep(request_interval_seconds)
                payloads, raw_body = fetch_era5_daily_batch(batch_points, start_date, end_date)
                batch_sha256 = hashlib.sha256(raw_body).hexdigest()
                batch_file = batch_directory / f"era5_batch_{batch_sha256}.json"
                if not batch_file.exists():
                    temporary_batch = batch_file.with_suffix(batch_file.suffix + ".part")
                    temporary_batch.write_bytes(raw_body)
                    temporary_batch.replace(batch_file)
                batch_rows = []
                for record in batch_records:
                    grid_id = str(record.grid_id)
                    payload = payloads[grid_id]
                    cache = destination / f"{grid_id}.json"
                    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                    temporary = cache.with_suffix(cache.suffix + ".part")
                    temporary.write_bytes(encoded)
                    temporary.replace(cache)
                    table = parse_era5_daily_response(
                        payload, grid_id, float(record.latitude), float(record.longitude)
                    )
                    batch_rows.append(
                        {
                            "grid_id": grid_id,
                            "latitude": float(record.latitude),
                            "longitude": float(record.longitude),
                            "api_response_latitude": float(table["api_response_latitude"].iloc[0]),
                            "api_response_longitude": float(table["api_response_longitude"].iloc[0]),
                            "api_response_elevation": float(table["api_response_elevation"].iloc[0]),
                            "status": "downloaded",
                            "daily_rows": len(table),
                            "start_date": table["date"].min().date().isoformat(),
                            "end_date": table["date"].max().date().isoformat(),
                            "model": "era5",
                            "timezone": "Asia/Kolkata",
                            "variables": ",".join(variable for variable, _ in ERA5_DAILY_VARIABLES),
                            "cache_file": cache.name,
                            "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                            "source_batch_file": str(batch_file.relative_to(destination)),
                            "source_batch_sha256": batch_sha256,
                            "error": "",
                        }
                    )
                results.extend(batch_rows)
                batch_succeeded = True
                last_error = None
                if progress_callback is not None:
                    progress_callback(batch_index, total_batches, sum(row["status"] != "failed" for row in results), "")
                break
            except Exception as exc:
                last_error = exc
                response = getattr(exc, "response", None)
                reason = str(getattr(response, "text", "")).lower()
                if (
                    getattr(response, "status_code", None) == 429
                    and "daily api request limit" in reason
                ):
                    daily_quota_exhausted = True
                    break
                if attempt >= retries:
                    break
                if getattr(response, "status_code", None) == 429:
                    retry_after = getattr(response, "headers", {}).get("Retry-After")
                    try:
                        delay = max(1.0, float(retry_after)) if retry_after is not None else 60.0
                    except (TypeError, ValueError):
                        delay = 60.0
                    reason = str(getattr(exc.response, "text", "")).lower()
                    if retry_after is None and "next hour" in reason:
                        now = datetime.now(timezone.utc)
                        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                        delay = max(delay, (next_hour - now).total_seconds() + 2.0)
                    elif retry_after is None and ("one minute" in reason or "60 seconds" in reason):
                        delay = max(delay, 60.0)
                else:
                    delay = retry_delay_seconds * (2**attempt)
                time.sleep(delay)
        else:
            last_error = last_error or RuntimeError("ERA5 batch exhausted retries")

        if daily_quota_exhausted:
            provider_message = str(getattr(getattr(last_error, "response", None), "text", ""))
            error = f"{type(last_error).__name__}: {last_error}; provider response: {provider_message}"
            for record in missing_records[offset:]:
                results.append(
                    {
                        "grid_id": str(record.grid_id),
                        "latitude": float(record.latitude),
                        "longitude": float(record.longitude),
                        "api_response_latitude": np.nan,
                        "api_response_longitude": np.nan,
                        "api_response_elevation": np.nan,
                        "status": "deferred",
                        "daily_rows": 0,
                        "start_date": pd.to_datetime(start_date).date().isoformat(),
                        "end_date": pd.to_datetime(end_date).date().isoformat(),
                        "model": "era5",
                        "timezone": "Asia/Kolkata",
                        "variables": ",".join(variable for variable, _ in ERA5_DAILY_VARIABLES),
                        "cache_file": f"{record.grid_id}.json",
                        "sha256": "",
                        "source_batch_file": "",
                        "source_batch_sha256": "",
                        "error": error,
                    }
                )
            if progress_callback is not None:
                progress_callback(
                    batch_index,
                    total_batches,
                    sum(row["status"] in {"cached", "downloaded"} for row in results),
                    error,
                )
            break

        if not batch_succeeded:
            for record in batch_records:
                results.append(
                    {
                        "grid_id": str(record.grid_id),
                        "latitude": float(record.latitude),
                        "longitude": float(record.longitude),
                        "api_response_latitude": np.nan,
                        "api_response_longitude": np.nan,
                        "api_response_elevation": np.nan,
                        "status": "failed",
                        "daily_rows": 0,
                        "start_date": pd.to_datetime(start_date).date().isoformat(),
                        "end_date": pd.to_datetime(end_date).date().isoformat(),
                        "model": "era5",
                        "timezone": "Asia/Kolkata",
                        "variables": ",".join(variable for variable, _ in ERA5_DAILY_VARIABLES),
                        "cache_file": f"{record.grid_id}.json",
                        "sha256": "",
                        "source_batch_file": "",
                        "source_batch_sha256": "",
                        "error": f"{type(last_error).__name__}: {last_error}",
                    }
                )
            if progress_callback is not None:
                progress_callback(
                    batch_index,
                    total_batches,
                    sum(row["status"] != "failed" for row in results),
                    f"{type(last_error).__name__}: {last_error}",
                )
    manifest = pd.DataFrame(results).sort_values("grid_id", kind="stable").reset_index(drop=True)
    manifest.to_csv(destination / "download_manifest.csv", index=False)
    return manifest


def download_era5_seasonal_grid(
    points: pd.DataFrame,
    destination_dir: str | Path,
    start_year: int,
    end_year: int,
    max_workers: int = 1,
    retries: int = 4,
    retry_delay_seconds: float = 1.0,
    batch_size: int = 5,
    progress_callback: Any | None = None,
    request_interval_seconds: float = 1.0,
    wait_for_daily_quota: bool = True,
) -> pd.DataFrame:
    """Fetch ERA5 only for issue dates and their seven-day feature history.

    Separate annual windows remain independently resumable. The final per-cell
    caches are expanded to the complete requested calendar period with nulls
    outside the predictor windows, preserving the existing dataset assembler.
    """
    if not isinstance(start_year, int) or not isinstance(end_year, int) or start_year > end_year:
        raise ValueError("start_year and end_year must be ordered integer years")
    required = {"grid_id", "latitude", "longitude"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"point table is missing columns: {sorted(missing)}")
    if points.empty or points[list(required)].isna().any().any():
        raise ValueError("point table must contain valid, non-empty locations")

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    window_root = destination / "windows"
    completed_years: list[int] = []
    for year in range(start_year, end_year + 1):
        while True:
            window_manifest = download_era5_grid(
                points,
                window_root / str(year),
                start_date=f"{year}-02-22",
                end_date=f"{year}-06-30",
                max_workers=max_workers,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
                batch_size=batch_size,
                progress_callback=(
                    lambda current, total, complete, error, year=year: progress_callback(
                        year - start_year + 1,
                        end_year - start_year + 1,
                        year,
                        current,
                        total,
                        complete,
                        error,
                    )
                    if progress_callback is not None
                    else None
                ),
                request_interval_seconds=request_interval_seconds,
            )
            unsuccessful = window_manifest.loc[
                ~window_manifest["status"].isin(("cached", "downloaded"))
            ]
            if unsuccessful.empty:
                break
            error = "; ".join(unsuccessful["error"].dropna().astype(str).unique())
            is_daily_quota = (
                unsuccessful["status"].eq("deferred").any()
                and "daily api request limit" in error.lower()
            )
            if is_daily_quota and wait_for_daily_quota:
                now = datetime.now(timezone.utc)
                next_reset = (now + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) + timedelta(minutes=5)
                time.sleep(max(1.0, (next_reset - now).total_seconds()))
                continue

            status = "deferred" if unsuccessful["status"].eq("deferred").any() else "failed"
            summary = points[["grid_id", "latitude", "longitude"]].copy()
            summary["status"] = status
            summary["completed_years"] = ",".join(map(str, completed_years))
            summary["incomplete_year"] = year
            summary["error"] = error
            summary.to_csv(destination / "download_manifest.csv", index=False)
            return summary
        completed_years.append(year)

    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    positions = {date_value.strftime("%Y-%m-%d"): index for index, date_value in enumerate(dates)}
    manifest_rows: list[dict[str, Any]] = []
    aliases = [feature for _, feature in ERA5_DAILY_VARIABLES]
    variables = [variable for variable, _ in ERA5_DAILY_VARIABLES]
    ordered = points.sort_values("grid_id", kind="stable").reset_index(drop=True)

    for record in ordered.itertuples(index=False):
        grid_id = str(record.grid_id)
        yearly_payloads = []
        for year in completed_years:
            path = window_root / str(year) / f"{grid_id}.json"
            yearly_payloads.append(json.loads(path.read_text(encoding="utf-8")))
        merged_daily: dict[str, Any] = {"time": dates.strftime("%Y-%m-%d").tolist()}
        for variable in variables:
            values: list[float | None] = [None] * len(dates)
            for payload in yearly_payloads:
                for date_value, value in zip(
                    payload["daily"]["time"], payload["daily"][variable], strict=True
                ):
                    values[positions[date_value]] = value
            merged_daily[variable] = values

        combined = dict(yearly_payloads[0])
        combined["daily"] = merged_daily
        combined["benchmark_predictor_window"] = (
            "ERA5 daily summaries for February 22 through June 30 in each year; "
            "all other calendar days are null. This covers March-June issue dates "
            "and seven preceding calendar-day lags."
        )
        cache = destination / f"{grid_id}.json"
        encoded = json.dumps(combined, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        temporary = cache.with_suffix(cache.suffix + ".part")
        temporary.write_bytes(encoded)
        temporary.replace(cache)
        table = parse_era5_daily_response(
            combined, grid_id, float(record.latitude), float(record.longitude)
        )
        if not pd.DatetimeIndex(table["date"]).equals(dates):
            raise ValueError(f"combined ERA5 cache for {grid_id} has invalid calendar coverage")
        manifest_rows.append(
            {
                "grid_id": grid_id,
                "latitude": float(record.latitude),
                "longitude": float(record.longitude),
                "api_response_latitude": float(table["api_response_latitude"].iloc[0]),
                "api_response_longitude": float(table["api_response_longitude"].iloc[0]),
                "api_response_elevation": float(table["api_response_elevation"].iloc[0]),
                "status": "downloaded",
                "daily_rows": sum(len(payload["daily"]["time"]) for payload in yearly_payloads),
                "cache_calendar_rows": len(dates),
                "start_date": dates[0].date().isoformat(),
                "end_date": dates[-1].date().isoformat(),
                "predictor_window": "February 22-June 30 annually",
                "model": "era5",
                "timezone": "Asia/Kolkata",
                "variables": ",".join(variables),
                "cache_file": cache.name,
                "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "window_manifests": "windows/",
                "error": "",
            }
        )

    manifest = pd.DataFrame(manifest_rows).sort_values("grid_id", kind="stable").reset_index(drop=True)
    manifest.to_csv(destination / "download_manifest.csv", index=False)
    return manifest
