"""Assembly helpers for the IMD target grid and cached weather predictors."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import xarray as xr

from heatwave_benchmark.data import (
    ERA5_DAILY_VARIABLES,
    IMD_GRID_SHAPE,
    NASA_POWER_DAILY_API,
    NASA_POWER_PARAMETERS,
    parse_era5_daily_response,
    parse_nasa_power_daily_response,
    read_imd_tmax_year,
)
from heatwave_benchmark.features import build_lagged_features
from heatwave_benchmark.geography import india_grid_mask
from heatwave_benchmark.labels import make_daily_gridcell_labels, make_horizon_target
from heatwave_benchmark.splits import add_spatial_block_ids


def read_imd_archive(
    directory: str | Path,
    start_year: int,
    end_year: int,
) -> xr.DataArray:
    """Read consecutive validated IMD annual files into one daily field."""
    if not isinstance(start_year, int) or not isinstance(end_year, int) or start_year > end_year:
        raise ValueError("start_year and end_year must be an ordered year range")
    folder = Path(directory)
    annual = [read_imd_tmax_year(folder / f"MaxT_{year}.GRD", year) for year in range(start_year, end_year + 1)]
    result = xr.concat(annual, dim="time")
    expected_dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    if not pd.DatetimeIndex(result.time.values).equals(expected_dates):
        raise ValueError("IMD archive is not a complete, consecutive daily series")
    if result.sizes["latitude"] != IMD_GRID_SHAPE[0] or result.sizes["longitude"] != IMD_GRID_SHAPE[1]:
        raise ValueError("IMD annual files do not share the expected 1-degree grid")
    return result


def select_analysis_points(
    tmax: xr.DataArray,
    terrain_mask: xr.DataArray,
    country_mask: xr.DataArray | None = None,
    observation_period: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Select observed IMD cells inside India and on the land/terrain mask.

    The country boundary is applied to grid centers, not fractional cell areas.
    When supplied, ``observation_period`` fixes domain eligibility to those
    dates so later target availability cannot determine the study domain.
    """
    if not {"time", "latitude", "longitude"}.issubset(tmax.dims):
        raise ValueError("tmax must have time, latitude, and longitude dimensions")
    if set(terrain_mask.dims) != {"latitude", "longitude"}:
        raise ValueError("terrain_mask must have latitude and longitude dimensions")
    field = tmax.transpose("time", "latitude", "longitude")
    mask = terrain_mask.transpose("latitude", "longitude")
    if country_mask is None:
        country_mask = india_grid_mask(field.latitude.values, field.longitude.values)
    country_mask = country_mask.transpose("latitude", "longitude")
    for dim in ("latitude", "longitude"):
        if not field.coords[dim].equals(mask.coords[dim]):
            raise ValueError(f"terrain mask {dim} coordinates do not match the IMD grid")
        if not field.coords[dim].equals(country_mask.coords[dim]):
            raise ValueError(f"country mask {dim} coordinates do not match the IMD grid")

    observation_field = field
    if observation_period is not None:
        if len(observation_period) != 2:
            raise ValueError("observation_period must contain a start and end date")
        start, end = pd.to_datetime(observation_period, errors="raise")
        if start > end:
            raise ValueError("observation_period start must be on or before its end")
        observation_field = field.sel(time=slice(start, end))
        if observation_field.sizes["time"] == 0:
            raise ValueError("observation_period does not overlap the Tmax time axis")
    observed = np.isfinite(observation_field.values).any(axis=0)
    terrain = mask.values.astype(np.int8, copy=False)
    selected = observed & (terrain >= 0) & country_mask.values.astype(bool, copy=False)
    rows = []
    for lat_index, lon_index in np.argwhere(selected):
        latitude = float(field.latitude.values[lat_index])
        longitude = float(field.longitude.values[lon_index])
        rows.append(
            {
                "grid_id": f"lat{latitude:.1f}_lon{longitude:.1f}",
                "latitude": latitude,
                "longitude": longitude,
                "terrain_code": int(terrain[lat_index, lon_index]),
            }
        )
    if not rows:
        raise ValueError("no IMD cells have both Tmax observations and a land terrain code")
    return pd.DataFrame(rows).sort_values("grid_id", kind="stable").reset_index(drop=True)


def assemble_era5_dataset(
    cache_directory: str | Path,
    points: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> xr.Dataset:
    """Assemble per-cell ERA5 daily JSON caches into a time-by-cell dataset."""
    required = {"grid_id", "latitude", "longitude"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"point table is missing columns: {sorted(missing)}")
    if points.empty or points["grid_id"].duplicated().any():
        raise ValueError("point table must be non-empty and grid IDs unique")
    dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="D")
    if dates.empty:
        raise ValueError("end_date must be on or after start_date")

    ordered = points.sort_values("grid_id", kind="stable").reset_index(drop=True)
    grid_ids = ordered["grid_id"].astype(str).to_numpy()
    aliases = [feature for _, feature in ERA5_DAILY_VARIABLES]
    arrays = {name: np.full((len(dates), len(ordered)), np.nan, dtype=np.float32) for name in aliases}
    source_latitude = np.full(len(ordered), np.nan, dtype=float)
    source_longitude = np.full(len(ordered), np.nan, dtype=float)
    source_elevation = np.full(len(ordered), np.nan, dtype=float)
    units: dict[str, str] = {}
    cache_root = Path(cache_directory)
    for column, record in enumerate(ordered.itertuples(index=False)):
        grid_id = str(record.grid_id)
        path = cache_root / f"{grid_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing ERA5 raw cache for {grid_id}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        table = parse_era5_daily_response(
            payload,
            grid_id,
            float(record.latitude),
            float(record.longitude),
        )
        if not pd.DatetimeIndex(table.date).equals(dates):
            raise ValueError(f"ERA5 cache for {grid_id} lacks complete requested daily coverage")
        for variable, feature in ERA5_DAILY_VARIABLES:
            arrays[feature][:, column] = table[feature].to_numpy(dtype=np.float32)
            units[feature] = str(payload.get("daily_units", {}).get(variable, ""))
        source_latitude[column] = table["api_response_latitude"].iloc[0]
        source_longitude[column] = table["api_response_longitude"].iloc[0]
        source_elevation[column] = table["api_response_elevation"].iloc[0]

    source_manifest_path = cache_root / "source_manifest.json"
    if source_manifest_path.is_file():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        resolution = float(source_manifest.get("native_resolution_degrees", 0.25))
        source_attrs = {
            "source": source_manifest.get("dataset", "Copernicus ERA5 via public Icechunk archive"),
            "api": source_manifest.get("registry_url", ""),
            "model": "ERA5",
            "timezone": source_manifest.get("daily_timezone", "Asia/Kolkata"),
            "cell_selection": (
                f"nearest {resolution:g}-degree ERA5 grid cell to each IMD 1-degree cell center"
            ),
            "period": f"{dates[0].date()} through {dates[-1].date()}",
            "grid_cell_resolution": (
                f"ERA5 reanalysis: {resolution:g} degree; returned source coordinates retained per point"
            ),
            "snapshot_id": source_manifest.get("snapshot_id", ""),
            "data_license": source_manifest.get("license", ""),
            "daily_aggregation": source_manifest.get("daily_aggregation", ""),
        }
    else:
        source_attrs = {
            "source": "Open-Meteo Historical Weather API, explicit ERA5 model",
            "api": "https://archive-api.open-meteo.com/v1/archive",
            "model": "ERA5",
            "timezone": "Asia/Kolkata",
            "cell_selection": "nearest; elevation downscaling disabled",
            "period": f"{dates[0].date()} through {dates[-1].date()}",
            "grid_cell_resolution": "IMD 1-degree label-cell centers mapped to nearest ERA5 0.25-degree cell; API response grid coordinates retained",
        }

    return xr.Dataset(
        {
            name: (('time', 'grid_id'), values, {"units": units.get(name, "")})
            for name, values in arrays.items()
        },
        coords={
            "time": dates,
            "grid_id": grid_ids,
            "latitude": ("grid_id", ordered["latitude"].to_numpy(dtype=float)),
            "longitude": ("grid_id", ordered["longitude"].to_numpy(dtype=float)),
            "api_response_latitude": ("grid_id", source_latitude),
            "api_response_longitude": ("grid_id", source_longitude),
            "api_response_elevation": ("grid_id", source_elevation),
        },
        attrs=source_attrs,
    )


def assemble_power_dataset(
    cache_directory: str | Path,
    points: pd.DataFrame,
    start_date: str,
    end_date: str,
    parameters: tuple[str, ...] = NASA_POWER_PARAMETERS,
) -> xr.Dataset:
    """Assemble validated raw POWER point caches into a compact xarray dataset."""
    required = {"grid_id", "latitude", "longitude"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"point table is missing columns: {sorted(missing)}")
    if points.empty or points["grid_id"].duplicated().any():
        raise ValueError("point table must be non-empty and grid IDs unique")
    dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="D")
    if dates.empty:
        raise ValueError("end_date must be on or after start_date")

    ordered = points.sort_values("grid_id", kind="stable").reset_index(drop=True)
    grid_ids = ordered["grid_id"].astype(str).to_numpy()
    arrays = {
        parameter.lower(): np.full((len(dates), len(ordered)), np.nan, dtype=np.float32)
        for parameter in parameters
    }
    api_response_latitude = np.full(len(ordered), np.nan, dtype=float)
    api_response_longitude = np.full(len(ordered), np.nan, dtype=float)
    units: dict[str, str] = {}
    api_versions: set[str] = set()
    source_tags: set[str] = set()
    cache_root = Path(cache_directory)
    for column, record in enumerate(ordered.itertuples(index=False)):
        grid_id = str(record.grid_id)
        path = cache_root / f"{grid_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing NASA POWER raw cache for {grid_id}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        api_versions.add(str(payload.get("header", {}).get("api", {}).get("version", "unknown")))
        source_tags.update(str(value) for value in payload.get("header", {}).get("sources", []))
        table = parse_nasa_power_daily_response(
            payload,
            grid_id=grid_id,
            latitude=float(record.latitude),
            longitude=float(record.longitude),
            parameters=parameters,
            expected_time_standard="LST",
        )
        if not pd.DatetimeIndex(table.date).equals(dates):
            raise ValueError(f"NASA POWER cache for {grid_id} lacks complete requested daily coverage")
        for parameter in parameters:
            arrays[parameter.lower()][:, column] = table[parameter.lower()].to_numpy(dtype=np.float32)
            units[parameter.lower()] = str(payload.get("parameters", {}).get(parameter, {}).get("units", ""))
        api_response_latitude[column] = table["api_response_latitude"].iloc[0]
        api_response_longitude[column] = table["api_response_longitude"].iloc[0]

    coords = {
        "time": dates,
        "grid_id": grid_ids,
        "latitude": ("grid_id", ordered["latitude"].to_numpy(dtype=float)),
        "longitude": ("grid_id", ordered["longitude"].to_numpy(dtype=float)),
        "api_response_latitude": ("grid_id", api_response_latitude),
        "api_response_longitude": ("grid_id", api_response_longitude),
    }
    result = xr.Dataset(
        {
            name: (("time", "grid_id"), values, {"units": units.get(name, "")})
            for name, values in arrays.items()
        },
        coords=coords,
        attrs={
            "source": "NASA POWER daily point API",
            "api": NASA_POWER_DAILY_API,
            "api_version": ", ".join(sorted(api_versions)),
            "source_products": ", ".join(sorted(source_tags)),
            "time_standard": "LST",
            "period": f"{dates[0].date()} through {dates[-1].date()}",
            "native_resolution": "MERRA-2 meteorology: 0.5 degree latitude by 0.625 degree longitude",
            "grid_cell_resolution": "IMD 1-degree cell centers queried through NASA POWER point API; the returned native source coordinates are retained per point",
            "parameters": ", ".join(parameters),
        },
    )
    return result


def build_benchmark_frame(
    meteorology: xr.Dataset,
    tmax: xr.DataArray,
    terrain_mask: xr.DataArray,
    normals: xr.DataArray,
    forecast_horizons: tuple[int, ...] = (1, 2, 3),
    max_lag_days: int = 7,
    issue_months: tuple[int, ...] = (3, 4, 5, 6),
) -> pd.DataFrame:
    """Build issue-day rows from contemporaneous and lagged meteorological inputs."""
    if not meteorology.data_vars:
        raise ValueError("meteorology dataset must contain at least one variable")
    if "grid_id" not in meteorology.dims or "time" not in meteorology.dims:
        raise ValueError("meteorology dataset must have time and grid_id dimensions")
    if not pd.DatetimeIndex(meteorology.time.values).equals(pd.DatetimeIndex(tmax.time.values)):
        raise ValueError("meteorology and IMD time coordinates must match exactly")
    if not issue_months or any(month not in range(1, 13) for month in issue_months):
        raise ValueError("issue_months must contain valid calendar months")
    if len(set(issue_months)) != len(issue_months):
        raise ValueError("issue_months must be unique")

    labels = make_daily_gridcell_labels(tmax, normals, terrain_mask)
    dates = pd.DatetimeIndex(meteorology.time.values)
    dayofyear = dates.day_of_year.to_numpy()
    sin_doy = np.sin(2 * np.pi * (dayofyear - 1) / 366.0).astype(np.float32)
    cos_doy = np.cos(2 * np.pi * (dayofyear - 1) / 366.0).astype(np.float32)
    predictor_columns = tuple(meteorology.data_vars)
    context_columns = ("latitude", "longitude", "terrain_code", "doy_sin", "doy_cos")
    frames = []

    for grid_id in meteorology.grid_id.values.astype(str):
        latitude = float(meteorology.latitude.sel(grid_id=grid_id).item())
        longitude = float(meteorology.longitude.sel(grid_id=grid_id).item())
        terrain_value = int(terrain_mask.sel(latitude=latitude, longitude=longitude).item())
        weather = pd.DataFrame({"grid_id": grid_id, "date": dates})
        for variable in predictor_columns:
            weather[variable] = meteorology[variable].sel(grid_id=grid_id).values.astype(np.float32)
        weather["latitude"] = np.float32(latitude)
        weather["longitude"] = np.float32(longitude)
        weather["terrain_code"] = np.int8(terrain_value)
        weather["doy_sin"] = sin_doy
        weather["doy_cos"] = cos_doy

        lagged = build_lagged_features(
            weather,
            predictor_columns=predictor_columns,
            context_columns=context_columns,
            max_lag_days=max_lag_days,
            group_col="grid_id",
            date_col="date",
        )
        daily_labels = labels.sel(latitude=latitude, longitude=longitude).values.astype(float)
        targets = {f"target_h{horizon}": make_horizon_target(daily_labels, horizon) for horizon in forecast_horizons}
        known_label = np.isfinite(daily_labels)
        metadata = pd.DataFrame(
            {
                "grid_id": grid_id,
                "date": dates,
                "current_heatwave": np.where(known_label, (daily_labels >= 1).astype(float), np.nan),
                **targets,
            }
        )
        part = lagged.merge(metadata, on=["grid_id", "date"], how="inner", validate="one_to_one")
        part = part.loc[part["date"].dt.month.isin(issue_months)].copy()
        frames.append(part)

    result = pd.concat(frames, ignore_index=True)
    result = add_spatial_block_ids(result, block_degrees=5.0)
    numeric_features = [column for column in result if "_lag_" in column]
    numeric_features.extend(context_columns)
    result[numeric_features] = result[numeric_features].astype(np.float32)
    return result.sort_values(["date", "grid_id"], kind="stable").reset_index(drop=True)
    parse_era5_daily_response,
