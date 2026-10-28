"""IMD-aligned grid-cell labels and short-lead event targets."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr


_ENTRY_TEMPERATURE = {"plains": 40.0, "hilly": 30.0, "coastal": 37.0}


def classify_gridcell_heatwave(
    tmax: np.ndarray | Sequence[float],
    departure: np.ndarray | Sequence[float],
    terrain: np.ndarray | Sequence[str] | str,
) -> np.ndarray:
    """Classify grid-cell days using an IMD-aligned temperature rule.

    Returns float labels: 0 for no event, 1 for heatwave, 2 for severe
    heatwave, and NaN when the available inputs cannot determine a label.
    This is a grid-cell adaptation and does not implement IMD station-count
    or subdivision-duration declaration rules.
    """
    tmax_array, departure_array, terrain_array = np.broadcast_arrays(
        np.asarray(tmax, dtype=float),
        np.asarray(departure, dtype=float),
        np.asarray(terrain),
    )
    if np.issubdtype(terrain_array.dtype, np.number):
        terrain_text = np.full(terrain_array.shape, "none", dtype="<U8")
        for code, name in ((0, "plains"), (1, "hilly"), (2, "coastal"), (-1, "none")):
            terrain_text[terrain_array == code] = name
        terrain_text[np.isnan(terrain_array)] = "none"
    else:
        terrain_text = np.char.lower(terrain_array.astype(str))
    known_terrain = np.isin(terrain_text, tuple(_ENTRY_TEMPERATURE))
    unknown_terrain = ~known_terrain & ~np.isin(terrain_text, ("", "none", "nan"))
    if np.any(unknown_terrain):
        invalid = sorted(set(terrain_text[unknown_terrain].tolist()))
        raise ValueError(f"terrain contains unsupported values: {invalid}")

    entry_threshold = np.full(tmax_array.shape, np.nan, dtype=float)
    for name, threshold in _ENTRY_TEMPERATURE.items():
        entry_threshold[terrain_text == name] = threshold

    valid_temperature = np.isfinite(tmax_array)
    eligible_temperature = valid_temperature & (tmax_array >= entry_threshold)
    absolute_eligible = np.isin(terrain_text, ("plains", "hilly"))
    absolute_event = absolute_eligible & (tmax_array >= 45.0)
    absolute_severe = absolute_eligible & (tmax_array >= 47.0)
    known_departure = np.isfinite(departure_array)

    can_classify = valid_temperature & known_terrain & (known_departure | absolute_event)
    standard_event = eligible_temperature & (
        absolute_event | (known_departure & (departure_array >= 4.5))
    )
    severe_event = eligible_temperature & (
        absolute_severe | (known_departure & (departure_array > 6.4))
    )

    labels = np.full(tmax_array.shape, np.nan, dtype=float)
    labels[can_classify] = 0.0
    labels[standard_event & can_classify] = 1.0
    labels[severe_event & can_classify] = 2.0
    return labels


def compute_daily_normals(
    tmax: xr.DataArray,
    start_year: int,
    end_year: int,
    window_days: int = 15,
    min_years: int = 10,
) -> xr.DataArray:
    """Compute a circular, centered daily Tmax climatology for fixed years.

    The day-of-year means are smoothed across the year boundary. A cell/day
    normal is retained only when the smoothing window has observations from
    at least ``min_years`` distinct reference years. Feb 29 follows the
    conventional pandas day-of-year index (366 positions), matching IMD's
    documented IMDlib implementation.
    """
    if not isinstance(tmax, xr.DataArray) or "time" not in tmax.dims:
        raise ValueError("tmax must be an xarray DataArray with a time dimension")
    if not isinstance(start_year, int) or not isinstance(end_year, int) or start_year > end_year:
        raise ValueError("start_year and end_year must be an ordered year range")
    if not isinstance(window_days, int) or window_days < 1 or window_days % 2 == 0:
        raise ValueError("window_days must be a positive odd integer")
    if not isinstance(min_years, int) or min_years < 1 or min_years > end_year - start_year + 1:
        raise ValueError("min_years must fit within the reference period")

    dates = pd.DatetimeIndex(pd.to_datetime(tmax.time.values, errors="raise"))
    expected = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    if not dates.is_unique or not dates.equals(expected):
        raise ValueError(
            "field time must cover every year from start_year through end_year without gaps"
        )

    spatial_dims = tuple(dim for dim in tmax.dims if dim != "time")
    if not spatial_dims:
        raise ValueError("tmax must have at least one non-time dimension")
    field = tmax.transpose("time", *spatial_dims).values.astype(float, copy=False)
    years = dates.year.to_numpy()
    doys = dates.day_of_year.to_numpy()
    year_values = np.arange(start_year, end_year + 1)
    shape = field.shape[1:]

    day_means = np.full((366, *shape), np.nan, dtype=float)
    yearly_support = np.zeros((len(year_values), 366, *shape), dtype=bool)
    for doy in range(1, 367):
        day_mask = doys == doy
        if not day_mask.any():
            continue
        day_values = field[day_mask]
        valid = np.isfinite(day_values)
        counts = valid.sum(axis=0)
        totals = np.where(valid, day_values, 0.0).sum(axis=0)
        np.divide(totals, counts, out=day_means[doy - 1], where=counts > 0)
        for year_index, year in enumerate(year_values):
            year_day_mask = day_mask & (years == year)
            if year_day_mask.any():
                yearly_support[year_index, doy - 1] = np.isfinite(field[year_day_mask]).any(axis=0)

    half = window_days // 2
    normals = np.full_like(day_means, np.nan)
    for doy_index in range(366):
        window_indices = (np.arange(doy_index - half, doy_index + half + 1) % 366)
        window_means = day_means[window_indices]
        valid = np.isfinite(window_means)
        counts = valid.sum(axis=0)
        totals = np.where(valid, window_means, 0.0).sum(axis=0)
        smoothed = np.full(shape, np.nan, dtype=float)
        np.divide(totals, counts, out=smoothed, where=counts > 0)
        support_years = yearly_support[:, window_indices].any(axis=1).sum(axis=0)
        smoothed[support_years < min_years] = np.nan
        normals[doy_index] = smoothed

    coords = {dim: tmax.coords[dim] for dim in spatial_dims if dim in tmax.coords}
    coords["dayofyear"] = np.arange(1, 367)
    return xr.DataArray(
        normals,
        dims=("dayofyear", *spatial_dims),
        coords=coords,
        name="tmax_daily_normal",
        attrs={
            "units": tmax.attrs.get("units", "degree_Celsius"),
            "reference_years": f"{start_year}-{end_year}",
            "smoothing_window_days": window_days,
            "minimum_support_years": min_years,
        },
    )


def make_daily_gridcell_labels(
    tmax: xr.DataArray,
    normals: xr.DataArray,
    terrain: np.ndarray | Sequence[str] | str,
) -> xr.DataArray:
    """Apply the IMD-aligned grid-cell rule to daily Tmax and normals."""
    if "time" not in tmax.dims or "dayofyear" not in normals.dims:
        raise ValueError("tmax requires time and normals require dayofyear dimensions")
    spatial_dims = tuple(dim for dim in tmax.dims if dim != "time")
    if set(normals.dims) != {"dayofyear", *spatial_dims}:
        raise ValueError("normal spatial dimensions must match Tmax spatial dimensions")
    ordered_tmax = tmax.transpose("time", *spatial_dims)
    ordered_normals = normals.transpose("dayofyear", *spatial_dims)
    if isinstance(terrain, xr.DataArray):
        if set(terrain.dims) != set(spatial_dims):
            raise ValueError("terrain dimensions must match Tmax spatial dimensions")
        terrain = terrain.transpose(*spatial_dims)
        for dim in spatial_dims:
            if dim in ordered_tmax.coords and dim in terrain.coords and not ordered_tmax.coords[dim].equals(terrain.coords[dim]):
                raise ValueError(f"terrain {dim} coordinates do not match Tmax grid coordinates")
    if any(
        not ordered_tmax.coords[dim].equals(ordered_normals.coords[dim])
        for dim in spatial_dims
        if dim in ordered_tmax.coords and dim in ordered_normals.coords
    ):
        raise ValueError("normal grid coordinates must match Tmax grid coordinates")

    dates = pd.DatetimeIndex(pd.to_datetime(ordered_tmax.time.values, errors="raise"))
    normal_values = ordered_normals.values.astype(float, copy=False)[dates.day_of_year.to_numpy() - 1]
    tmax_values = ordered_tmax.values.astype(float, copy=False)
    terrain_values = terrain.values if isinstance(terrain, xr.DataArray) else terrain
    labels = classify_gridcell_heatwave(
        tmax_values,
        tmax_values - normal_values,
        terrain_values,
    )
    return xr.DataArray(
        labels,
        dims=("time", *spatial_dims),
        coords={dim: ordered_tmax.coords[dim] for dim in ("time", *spatial_dims) if dim in ordered_tmax.coords},
        name="heatwave_label",
        attrs={
            "flag_values": [0, 1, 2],
            "flag_meanings": "no_heatwave heatwave severe_heatwave",
            "definition": "IMD-aligned grid-cell day; not an official subdivision declaration",
        },
    )


def make_horizon_target(
    daily_label: np.ndarray | Sequence[float], horizon: int
) -> np.ndarray:
    """Return whether a heatwave occurs exactly ``horizon`` days after issue.

    A target is NaN when the future label is missing or outside the supplied
    record. Input labels use 0 for no heatwave, 1 for heatwave, and 2 for
    severe heatwave.
    """
    if not isinstance(horizon, int) or horizon not in (1, 2, 3):
        raise ValueError("horizon must be one of the supported leads: 1, 2, or 3")

    labels = np.asarray(daily_label, dtype=float)
    if labels.ndim != 1:
        raise ValueError("daily_label must be a one-dimensional sequence")

    target = np.full(labels.shape, np.nan, dtype=float)
    if horizon < labels.size:
        future = labels[horizon:]
        observed = np.isfinite(future)
        aligned_target = target[:-horizon]
        aligned_target[observed] = (future[observed] >= 1.0).astype(float)
    return target
