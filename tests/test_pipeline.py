import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from heatwave_benchmark.pipeline import (
    assemble_era5_dataset,
    assemble_power_dataset,
    build_benchmark_frame,
    read_imd_archive,
    select_analysis_points,
)


def test_reads_and_concatenates_complete_imd_annual_archive(tmp_path):
    for year, days in ((2023, 365), (2024, 366)):
        values = np.full((days, 31, 31), 34.0, dtype="<f4")
        values.tofile(tmp_path / f"MaxT_{year}.GRD")

    result = read_imd_archive(tmp_path, 2023, 2024)

    assert result.sizes["time"] == 731
    assert pd.Timestamp(result.time.values[0]) == pd.Timestamp("2023-01-01")
    assert pd.Timestamp(result.time.values[-1]) == pd.Timestamp("2024-12-31")
    assert result.dims == ("time", "latitude", "longitude")


def test_selects_observed_land_grid_cells_with_deterministic_ids():
    times = pd.date_range("2020-01-01", periods=2)
    tmax = xr.DataArray(
        np.array([[[30.0, np.nan], [np.nan, 35.0]], [[31.0, np.nan], [np.nan, np.nan]]]),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": [10.5, 11.5], "longitude": [70.5, 71.5]},
    )
    terrain = xr.DataArray(
        np.array([[0, 1], [-1, 2]], dtype=np.int8),
        dims=("latitude", "longitude"),
        coords={"latitude": [10.5, 11.5], "longitude": [70.5, 71.5]},
    )
    india = xr.DataArray(
        np.array([[True, False], [False, True]]),
        dims=("latitude", "longitude"),
        coords={"latitude": [10.5, 11.5], "longitude": [70.5, 71.5]},
    )

    points = select_analysis_points(tmax, terrain, country_mask=india)

    assert points["grid_id"].tolist() == ["lat10.5_lon70.5", "lat11.5_lon71.5"]
    assert points["terrain_code"].tolist() == [0, 2]


def test_domain_selection_uses_only_the_reference_period_when_requested():
    times = pd.to_datetime(["1981-01-01", "2015-01-01"])
    tmax = xr.DataArray(
        np.array([[[30.0, np.nan]], [[31.0, 38.0]]]),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": [18.5], "longitude": [73.5, 74.5]},
    )
    terrain = xr.DataArray(
        [[0, 0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [18.5], "longitude": [73.5, 74.5]},
    )
    india = xr.DataArray(
        [[True, True]],
        dims=("latitude", "longitude"),
        coords={"latitude": [18.5], "longitude": [73.5, 74.5]},
    )

    points = select_analysis_points(
        tmax,
        terrain,
        country_mask=india,
        observation_period=("1981-01-01", "2010-12-31"),
    )

    assert points["grid_id"].tolist() == ["lat18.5_lon73.5"]


def test_assembles_cached_nasa_power_json_into_time_by_grid_dataset(tmp_path):
    dates = pd.date_range("2024-01-01", periods=2)
    points = pd.DataFrame(
        [{"grid_id": "lat18.5_lon73.5", "latitude": 18.5, "longitude": 73.5, "terrain_code": 0}]
    )
    parameters = ("T2M", "RH2M")
    payload = {
        "geometry": {"coordinates": [73.75, 18.5, 600.0]},
        "properties": {
            "parameter": {
                "T2M": {"20240101": 24.0, "20240102": -999.0},
                "RH2M": {"20240101": 70.0, "20240102": 65.0},
            }
        },
        "header": {"fill_value": -999.0, "time_standard": "LST", "sources": ["MERRA2"]},
        "parameters": {"T2M": {"units": "C"}, "RH2M": {"units": "%"}},
    }
    (tmp_path / f"{points.loc[0, 'grid_id']}.json").write_text(json.dumps(payload), encoding="utf-8")

    result = assemble_power_dataset(
        tmp_path,
        points,
        start_date=dates[0].strftime("%Y-%m-%d"),
        end_date=dates[-1].strftime("%Y-%m-%d"),
        parameters=parameters,
    )

    assert result.sizes == {"time": 2, "grid_id": 1}
    assert result["t2m"].values[0, 0] == pytest.approx(24.0)
    assert np.isnan(result["t2m"].values[1, 0])
    assert result["rh2m"].values[1, 0] == pytest.approx(65.0)
    assert result["t2m"].attrs["units"] == "C"


def test_assembles_cached_era5_daily_json_and_records_source_grid(tmp_path):
    from heatwave_benchmark.data import ERA5_DAILY_VARIABLES

    dates = ["2024-01-01", "2024-01-02"]
    values = {
        "temperature_2m_mean": [24.0, 25.0],
        "temperature_2m_max": [31.0, 32.0],
        "temperature_2m_min": [19.0, 20.0],
        "relative_humidity_2m_mean": [70.0, 65.0],
        "wind_speed_10m_mean": [2.1, 2.2],
        "surface_pressure_mean": [950.0, 949.0],
        "precipitation_sum": [0.0, 1.5],
    }
    point = {"grid_id": "lat18.5_lon73.5", "latitude": 18.5, "longitude": 73.5}
    payload = {
        "latitude": 18.5,
        "longitude": 73.5,
        "elevation": 600.0,
        "timezone": "Asia/Kolkata",
        "daily_units": {"time": "iso8601", **{name: "unit" for name, _ in ERA5_DAILY_VARIABLES}},
        "daily": {"time": dates, **{name: values[name] for name, _ in ERA5_DAILY_VARIABLES}},
    }
    (tmp_path / f"{point['grid_id']}.json").write_text(json.dumps(payload), encoding="utf-8")

    result = assemble_era5_dataset(tmp_path, pd.DataFrame([point]), dates[0], dates[-1])

    assert result.sizes == {"time": 2, "grid_id": 1}
    assert result["t2m"].values[:, 0].tolist() == pytest.approx([24.0, 25.0])
    assert result["rh2m"].values[1, 0] == pytest.approx(65.0)
    assert result["ws10m"].attrs["units"] == "unit"
    assert result.attrs["model"] == "ERA5"
    assert result.api_response_latitude.values[0] == pytest.approx(18.5)


def test_assembles_seasonally_padded_era5_cache_with_missing_offseason_days(tmp_path):
    from heatwave_benchmark.data import ERA5_DAILY_VARIABLES

    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    in_window = (dates >= "2024-02-22") & (dates <= "2024-06-30")
    values = [1.0 if in_window[index] else None for index in range(len(dates))]
    variables = [name for name, _ in ERA5_DAILY_VARIABLES]
    point = {"grid_id": "lat18.5_lon73.5", "latitude": 18.5, "longitude": 73.5}
    payload = {
        "latitude": 18.5,
        "longitude": 73.5,
        "elevation": 600.0,
        "timezone": "Asia/Kolkata",
        "daily_units": {"time": "iso8601", **{name: "unit" for name in variables}},
        "daily": {
            "time": dates.strftime("%Y-%m-%d").tolist(),
            **{name: values for name in variables},
        },
    }
    (tmp_path / f"{point['grid_id']}.json").write_text(json.dumps(payload), encoding="utf-8")

    result = assemble_era5_dataset(tmp_path, pd.DataFrame([point]), dates[0], dates[-1])

    assert result.sizes == {"time": 366, "grid_id": 1}
    assert np.isnan(result.t2m.sel(time="2024-01-01").item())
    assert result.t2m.sel(time="2024-02-22").item() == pytest.approx(1.0)
    assert result.t2m.sel(time="2024-06-30").item() == pytest.approx(1.0)
    assert np.isnan(result.t2m.sel(time="2024-07-01").item())


def test_assembles_era5_icechunk_dataset_metadata_from_source_manifest(tmp_path):
    from heatwave_benchmark.data import ERA5_DAILY_VARIABLES

    dates = pd.date_range("2024-03-01", periods=2, freq="D")
    variables = [name for name, _ in ERA5_DAILY_VARIABLES]
    point = {"grid_id": "lat18.5_lon73.5", "latitude": 18.5, "longitude": 73.5}
    payload = {
        "latitude": 18.5,
        "longitude": 73.5,
        "elevation": 600.0,
        "timezone": "Asia/Kolkata",
        "daily_units": {"time": "iso8601", **{name: "unit" for name in variables}},
        "daily": {
            "time": dates.strftime("%Y-%m-%d").tolist(),
            **{name: [1.0, 1.0] for name in variables},
        },
    }
    (tmp_path / f"{point['grid_id']}.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "source_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "Earthmover ERA5 Icechunk public mirror",
                "registry_url": "https://registry.opendata.aws/earthmover-era5/",
                "snapshot_id": "TZ2KX6A24CJ5DG17T7MG",
                "native_resolution_degrees": 0.25,
                "daily_timezone": "Asia/Kolkata",
            }
        ),
        encoding="utf-8",
    )

    result = assemble_era5_dataset(tmp_path, pd.DataFrame([point]), dates[0], dates[-1])

    assert result.attrs["source"] == "Earthmover ERA5 Icechunk public mirror"
    assert result.attrs["api"] == "https://registry.opendata.aws/earthmover-era5/"
    assert result.attrs["snapshot_id"] == "TZ2KX6A24CJ5DG17T7MG"
    assert result.attrs["grid_cell_resolution"].startswith("ERA5 reanalysis: 0.25 degree")


def test_benchmark_frame_uses_only_issue_day_and_earlier_weather_values():
    dates = pd.date_range("2020-02-27", periods=8)
    tmax = xr.DataArray(
        np.array([35.0, 36.0, 40.0, 41.0, 39.0, 35.0, 42.0, 45.0]).reshape(8, 1, 1),
        dims=("time", "latitude", "longitude"),
        coords={"time": dates, "latitude": [18.5], "longitude": [73.5]},
    )
    power = xr.Dataset(
        {"t2m": (("time", "grid_id"), np.arange(8, dtype=np.float32).reshape(8, 1))},
        coords={
            "time": dates,
            "grid_id": ["lat18.5_lon73.5"],
            "latitude": ("grid_id", [18.5]),
            "longitude": ("grid_id", [73.5]),
        },
    )
    terrain = xr.DataArray(
        [[0]], dims=("latitude", "longitude"), coords={"latitude": [18.5], "longitude": [73.5]}
    )
    normals = xr.DataArray(
        np.full((366, 1, 1), 35.0),
        dims=("dayofyear", "latitude", "longitude"),
        coords={"dayofyear": np.arange(1, 367), "latitude": [18.5], "longitude": [73.5]},
    )

    result = build_benchmark_frame(power, tmax, terrain, normals, max_lag_days=2)

    march_2 = result.loc[result.date == pd.Timestamp("2020-03-02")].iloc[0]
    assert march_2["t2m_lag_0"] == pytest.approx(4.0)
    assert march_2["t2m_lag_1"] == pytest.approx(3.0)
    assert march_2["t2m_lag_2"] == pytest.approx(2.0)
    assert march_2["target_h1"] == 0.0
    assert march_2["spatial_block"] == "lat3_lon14"

\n