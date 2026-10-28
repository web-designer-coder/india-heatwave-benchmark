"""Country-center mask based on pinned Natural Earth 10m India geometry."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import xarray as xr


@lru_cache(maxsize=1)
def _india_geometry() -> dict:
    path = Path(__file__).with_name("india_boundary_10m.geojson")
    feature = json.loads(path.read_text(encoding="utf-8"))
    if feature.get("type") != "Feature" or feature.get("properties", {}).get("adm0_a3") != "IND":
        raise ValueError(f"invalid Natural Earth India boundary file: {path}")
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"unsupported India boundary geometry in {path}")
    return geometry


def _point_in_ring(longitude: float, latitude: float, ring: Sequence[Sequence[float]]) -> bool:
    inside = False
    if len(ring) < 4:
        return False
    previous_x, previous_y = ring[-1][:2]
    for point in ring:
        current_x, current_y = point[:2]
        cross = (longitude - previous_x) * (current_y - previous_y) - (
            latitude - previous_y
        ) * (current_x - previous_x)
        if (
            abs(cross) <= 1e-10
            and min(previous_x, current_x) - 1e-10 <= longitude <= max(previous_x, current_x) + 1e-10
            and min(previous_y, current_y) - 1e-10 <= latitude <= max(previous_y, current_y) + 1e-10
        ):
            return True
        if (previous_y > latitude) != (current_y > latitude):
            crossing_x = previous_x + (latitude - previous_y) * (current_x - previous_x) / (
                current_y - previous_y
            )
            if longitude < crossing_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _point_in_polygon(longitude: float, latitude: float, polygon: Sequence[Sequence[Sequence[float]]]) -> bool:
    if not polygon or not _point_in_ring(longitude, latitude, polygon[0]):
        return False
    return not any(_point_in_ring(longitude, latitude, hole) for hole in polygon[1:])


def is_point_in_india(latitude: float, longitude: float) -> bool:
    """Return whether a coordinate is inside Natural Earth's India polygon."""
    if not np.isfinite([latitude, longitude]).all():
        raise ValueError("latitude and longitude must be finite")
    geometry = _india_geometry()
    polygons = (
        geometry["coordinates"]
        if geometry["type"] == "MultiPolygon"
        else [geometry["coordinates"]]
    )
    return any(_point_in_polygon(float(longitude), float(latitude), polygon) for polygon in polygons)


def india_grid_mask(
    latitudes: Sequence[float],
    longitudes: Sequence[float],
) -> xr.DataArray:
    """Build a boolean mask for grid centers inside India (not cell overlap)."""
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1 or not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("latitudes and longitudes must be finite one-dimensional sequences")
    values = np.zeros((lat.size, lon.size), dtype=bool)
    for lat_index, latitude in enumerate(lat):
        for lon_index, longitude in enumerate(lon):
            values[lat_index, lon_index] = is_point_in_india(latitude, longitude)
    return xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        name="india_center_mask",
        attrs={
            "source": "Natural Earth Admin 0 - Countries, 1:10m, version 5.1.1",
            "license": "Public domain",
            "selection": "grid center within India polygon; not fractional cell intersection",
        },
    )
