from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from heatwave_benchmark.data import (
    ERA5_ARCHIVE_API,
    ERA5_DAILY_VARIABLES,
    NASA_POWER_PARAMETERS,
    build_file_manifest,
    download_imd_tmax_year,
    download_nasa_power_daily_point,
    download_nasa_power_grid,
    download_era5_daily_point,
    download_era5_grid,
    download_era5_seasonal_grid,
    load_imd_terrain_mask,
    parse_nasa_power_daily_response,
    parse_era5_daily_response,
    fetch_era5_daily_batch,
    read_imd_tmax_year,
)


def _era5_payload(dates=("2024-01-01", "2024-01-02")):
    variables = [name for name, _ in ERA5_DAILY_VARIABLES]
    values = {
        "temperature_2m_mean": [24.0, 25.0],
        "temperature_2m_max": [31.0, 32.0],
        "temperature_2m_min": [19.0, 20.0],
        "relative_humidity_2m_mean": [70.0, 65.0],
        "wind_speed_10m_mean": [2.1, 2.2],
        "surface_pressure_mean": [950.0, 949.0],
        "precipitation_sum": [0.0, 1.5],
    }
    units = {
        "temperature_2m_mean": "°C",
        "temperature_2m_max": "°C",
        "temperature_2m_min": "°C",
        "relative_humidity_2m_mean": "%",
        "wind_speed_10m_mean": "m/s",
        "surface_pressure_mean": "hPa",
        "precipitation_sum": "mm",
    }
    return {
        "latitude": 18.5,
        "longitude": 73.5,
        "elevation": 600.0,
        "timezone": "Asia/Kolkata",
        "daily_units": {"time": "iso8601", **{v: units[v] for v in variables}},
        "daily": {"time": list(dates), **{v: values[v][: len(dates)] for v in variables}},
    }


def test_parses_era5_daily_values_to_stable_feature_names():
    result = parse_era5_daily_response(
        _era5_payload(),
        grid_id="lat18.5_lon73.5",
        latitude=18.5,
        longitude=73.5,
    )

    assert result.date.tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert result.loc[0, "t2m"] == pytest.approx(24.0)
    assert result.loc[1, "t2m_max"] == pytest.approx(32.0)
    assert result.loc[0, "rh2m"] == pytest.approx(70.0)
    assert result.loc[1, "ws10m"] == pytest.approx(2.2)
    assert result.loc[1, "sp"] == pytest.approx(949.0)
    assert result.loc[1, "tp"] == pytest.approx(1.5)
    assert result.loc[0, "api_response_latitude"] == pytest.approx(18.5)
    assert result.loc[0, "api_response_longitude"] == pytest.approx(73.5)


def test_aggregates_direct_era5_hourly_fields_into_ist_daily_predictors():
    import heatwave_benchmark.data as data_module

    times = pd.date_range(
        "1981-02-21 19:00", periods=24, freq="h", tz="UTC"
    )
    temperatures = np.linspace(300.0, 323.0, len(times))
    hourly = pd.DataFrame(
        {
            "valid_time": times,
            "grid_id": "cell_1",
            "latitude": 20.0,
            "longitude": 80.0,
            "t2m": temperatures,
            "d2m": 290.0,
            "u10": 3.0,
            "v10": 4.0,
            "sp": 100000.0,
            "tp": 0.001,
        }
    )

    result = data_module.aggregate_era5_hourly_to_daily(
        hourly, start_date="1981-02-22", end_date="1981-02-22"
    )

    temperature_c = temperatures - 273.15
    dewpoint_c = 290.0 - 273.15
    saturation_vapour = lambda value: 6.1121 * np.exp(  # noqa: E731
        17.502 * value / (240.97 + value)
    )
    expected_hourly_rh = (
        100.0 * saturation_vapour(dewpoint_c) / saturation_vapour(temperature_c)
    )
    assert len(result) == 1
    assert result.loc[0, "date"] == pd.Timestamp("1981-02-22")
    assert result.loc[0, "t2m"] == pytest.approx(float(temperature_c.mean()))
    assert result.loc[0, "t2m_max"] == pytest.approx(float(temperature_c.max()))
    assert result.loc[0, "t2m_min"] == pytest.approx(float(temperature_c.min()))
    assert result.loc[0, "rh2m"] == pytest.approx(float(expected_hourly_rh.mean()))
    assert result.loc[0, "ws10m"] == pytest.approx(5.0)
    assert result.loc[0, "sp"] == pytest.approx(1000.0)
    assert result.loc[0, "tp"] == pytest.approx(24.0)


def test_direct_era5_hourly_aggregation_accepts_repeated_times_across_cells():
    import heatwave_benchmark.data as data_module

    times = pd.date_range("1981-02-21 19:00", periods=24, freq="h", tz="UTC")
    one_cell = pd.DataFrame(
        {
            "valid_time": times,
            "grid_id": "cell_1",
            "latitude": 20.0,
            "longitude": 80.0,
            "t2m": 300.0,
            "d2m": 290.0,
            "u10": 3.0,
            "v10": 4.0,
            "sp": 100000.0,
            "tp": 0.001,
        }
    )
    second_cell = one_cell.assign(
        grid_id="cell_2",
        latitude=20.25,
        longitude=80.25,
        t2m=301.0,
    )

    result = data_module.aggregate_era5_hourly_to_daily(
        pd.concat([one_cell, second_cell], ignore_index=True),
        start_date="1981-02-22",
        end_date="1981-02-22",
    )

    assert result.grid_id.tolist() == ["cell_1", "cell_2"]
    assert result.t2m.tolist() == pytest.approx([300.0 - 273.15, 301.0 - 273.15])


def test_direct_era5_hourly_aggregation_rejects_duplicate_cell_timestamp_pairs():
    import heatwave_benchmark.data as data_module

    times = pd.date_range("1981-02-21 19:00", periods=24, freq="h", tz="UTC")
    hourly = pd.DataFrame(
        {
            "valid_time": list(times) + [times[0]],
            "grid_id": "cell_1",
            "latitude": 20.0,
            "longitude": 80.0,
            "t2m": 300.0,
            "d2m": 290.0,
            "u10": 3.0,
            "v10": 4.0,
            "sp": 100000.0,
            "tp": 0.001,
        }
    )

    with pytest.raises(ValueError, match="timestamps must be unique within each grid cell"):
        data_module.aggregate_era5_hourly_to_daily(
            hourly, start_date="1981-02-22", end_date="1981-02-22"
        )


def test_direct_era5_hourly_aggregation_rejects_incomplete_local_days():
    import heatwave_benchmark.data as data_module

    times = pd.date_range(
        "1981-02-21 19:00", periods=23, freq="h", tz="UTC"
    )
    hourly = pd.DataFrame(
        {
            "valid_time": times,
            "grid_id": "cell_1",
            "latitude": 20.0,
            "longitude": 80.0,
            "t2m": 300.0,
            "d2m": 290.0,
            "u10": 3.0,
            "v10": 4.0,
            "sp": 100000.0,
            "tp": 0.0,
        }
    )

    with pytest.raises(ValueError, match="24 hourly ERA5 records"):
        data_module.aggregate_era5_hourly_to_daily(
            hourly, start_date="1981-02-22", end_date="1981-02-22"
        )


def test_era5_icechunk_download_builds_snapshot_pinned_daily_cache(tmp_path, monkeypatch):
    import xarray as xr
    import heatwave_benchmark.data as data_module

    times = pd.date_range("1981-02-21 19:00", "1981-06-30 18:00", freq="h")
    shape = (len(times), 1, 2)
    temperatures = np.broadcast_to(np.array([300.0, 301.0])[None, None, :], shape)
    dataset = xr.Dataset(
        {
            "t2m": (("valid_time", "latitude", "longitude"), temperatures),
            "d2m": (("valid_time", "latitude", "longitude"), np.full(shape, 290.0)),
            "u10": (("valid_time", "latitude", "longitude"), np.full(shape, 3.0)),
            "v10": (("valid_time", "latitude", "longitude"), np.full(shape, 4.0)),
            "sp": (("valid_time", "latitude", "longitude"), np.full(shape, 100000.0)),
            "tp": (("valid_time", "latitude", "longitude"), np.full(shape, 0.001)),
        },
        coords={"valid_time": times, "latitude": [20.0], "longitude": [80.0, 81.0]},
    )
    dataset["t2m"].encoding["chunks"] = (8736, 1, 1)
    monkeypatch.setattr(
        data_module,
        "open_era5_icechunk_dataset",
        lambda snapshot_id=None: (dataset, snapshot_id or "snapshot-123"),
    )
    points = pd.DataFrame(
        {
            "grid_id": ["cell_a", "cell_b"],
            "latitude": [20.0, 20.0],
            "longitude": [80.0, 81.0],
        }
    )

    manifest = data_module.download_era5_icechunk_seasonal_grid(
        points, tmp_path, start_year=1981, end_year=1981
    )

    assert manifest.status.tolist() == ["downloaded", "downloaded"]
    assert set(manifest.source_snapshot_id) == {"snapshot-123"}
    payload = json.loads((tmp_path / "cell_b.json").read_text(encoding="utf-8"))
    assert payload["source_metadata"]["snapshot_id"] == "snapshot-123"
    assert payload["daily"]["time"][0] == "1981-01-01"
    assert payload["daily"]["time"][-1] == "1981-12-31"
    assert payload["daily"]["temperature_2m_mean"][52] == pytest.approx(301.0 - 273.15)
    assert payload["daily"]["temperature_2m_mean"][0] is None
    parsed = data_module.parse_era5_daily_response(
        payload, grid_id="cell_b", latitude=20.0, longitude=81.0
    )
    assert parsed.loc[52, "t2m"] == pytest.approx(301.0 - 273.15)
    assert parsed.loc[52, "tp"] == pytest.approx(24.0)
    resumed = data_module.download_era5_icechunk_seasonal_grid(
        points, tmp_path, start_year=1981, end_year=1981
    )
    assert resumed.status.tolist() == ["downloaded", "downloaded"]
    window_manifest = pd.read_csv(tmp_path / "windows" / "1981" / "download_manifest.csv")
    assert window_manifest.status.tolist() == ["cached", "cached"]
    from heatwave_benchmark.provenance import validate_era5_icechunk_cache

    audit = validate_era5_icechunk_cache(
        tmp_path, points, start_year=1981, end_year=1981, snapshot_id="snapshot-123"
    )
    assert audit["points"] == 2
    assert audit["annual_cache_files"] == 2
    assert audit["final_cache_files"] == 2

    damaged = tmp_path / "windows" / "1981" / "cell_a.json"
    damaged.write_text(damaged.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_era5_icechunk_cache(
            tmp_path, points, start_year=1981, end_year=1981, snapshot_id="snapshot-123"
        )


def test_era5_year_selector_trims_time_before_vectorized_spatial_selection():
    import heatwave_benchmark.data as data_module

    class RecordingDataset:
        def __init__(self):
            self.data_vars = set(data_module.ERA5_HOURLY_VARIABLES)
            self.coords = {"valid_time", "latitude", "longitude"}
            self.calls = []

        def __getitem__(self, variables):
            assert set(variables) == set(data_module.ERA5_HOURLY_VARIABLES)
            return self

        def sel(self, **kwargs):
            self.calls.append(kwargs)
            return self

    dataset = RecordingDataset()
    points = pd.DataFrame(
        {"grid_id": ["cell_1"], "latitude": [20.0], "longitude": [80.0]}
    )

    data_module.select_era5_icechunk_hourly_window(dataset, 1981, points)

    assert list(dataset.calls[0]) == ["valid_time"]
    assert dataset.calls[0]["valid_time"] == slice(
        "1981-02-21T19:00:00", "1981-06-30T18:00:00"
    )
    assert set(dataset.calls[1]) == {"latitude", "longitude", "method"}


def test_era5_points_are_batched_by_native_spatial_chunks():
    import heatwave_benchmark.data as data_module

    points = pd.DataFrame(
        {
            "grid_id": ["cell_c", "cell_a", "cell_b"],
            "latitude": [20.0, 20.0, 19.75],
            "longitude": [83.0, 80.0, 80.25],
        }
    )
    groups = data_module.group_era5_points_by_spatial_chunk(
        points,
        latitudes=np.array([20.0, 19.75, 19.5]),
        longitudes=np.array([80.0, 80.25, 81.0, 83.0]),
        chunk_shape=(3, 3),
    )

    assert [group.grid_id.tolist() for group in groups] == [["cell_a", "cell_b"], ["cell_c"]]


def test_era5_download_uses_explicit_era5_ist_and_nearest_cell_options(tmp_path):
    payload = _era5_payload(dates=("2024-01-01",))

    class Response:
        content = json.dumps(payload).encode()

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return payload

    class Session:
        def get(self, url, params, timeout):
            assert url == ERA5_ARCHIVE_API
            assert params["models"] == "era5"
            assert params["timezone"] == "Asia/Kolkata"
            assert params["cell_selection"] == "nearest"
            assert params["elevation"] == "nan"
            assert params["daily"] == ",".join(name for name, _ in ERA5_DAILY_VARIABLES)
            assert timeout
            return Response()

    cache = tmp_path / "point.json"
    result = download_era5_daily_point(
        18.5,
        73.5,
        "2024-01-01",
        "2024-01-01",
        "lat18.5_lon73.5",
        session=Session(),
        cache_path=cache,
    )

    assert result.loc[0, "t2m"] == pytest.approx(24.0)
    assert json.loads(cache.read_text(encoding="utf-8"))["timezone"] == "Asia/Kolkata"


def test_era5_grid_downloader_caches_manifest_and_enforces_request_limit(tmp_path, monkeypatch):
    import heatwave_benchmark.data as data_module

    def fake_download(points, start_date, end_date):
        payload = _era5_payload(dates=("2024-01-01",))
        return {str(points.grid_id.iloc[0]): payload}, b"[]"

    monkeypatch.setattr(data_module, "fetch_era5_daily_batch", fake_download)
    points = pd.DataFrame(
        [{"grid_id": "lat18.5_lon73.5", "latitude": 18.5, "longitude": 73.5}]
    )
    manifest = download_era5_grid(
        points,
        tmp_path,
        "2024-01-01",
        "2024-01-01",
        max_workers=1,
        retries=0,
    )

    assert manifest.loc[0, "status"] == "downloaded"
    assert manifest.loc[0, "api_response_longitude"] == pytest.approx(73.5)
    assert manifest.loc[0, "timezone"] == "Asia/Kolkata"
    assert len(manifest.loc[0, "sha256"]) == 64
    with pytest.raises(ValueError, match="sequential"):
        download_era5_grid(points, tmp_path, "2024-01-01", "2024-01-01", max_workers=6)


def test_era5_batch_request_maps_multiple_locations_in_requested_order():
    points = pd.DataFrame(
        [
            {"grid_id": "lat18.5_lon73.5", "latitude": 18.5, "longitude": 73.5},
            {"grid_id": "lat19.5_lon74.5", "latitude": 19.5, "longitude": 74.5},
        ]
    )
    first = _era5_payload(dates=("2024-01-01",))
    second = _era5_payload(dates=("2024-01-01",))
    second["latitude"] = 19.5
    second["longitude"] = 74.5
    second["daily"]["temperature_2m_mean"] = [29.0]

    class Response:
        content = json.dumps([first, second]).encode()

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return [first, second]

    class Session:
        def get(self, url, params, timeout):
            assert params["latitude"] == "18.5,19.5"
            assert params["longitude"] == "73.5,74.5"
            assert params["models"] == "era5"
            return Response()

    payloads, raw_body = fetch_era5_daily_batch(
        points,
        "2024-01-01",
        "2024-01-01",
        session=Session(),
    )

    assert len(raw_body) > 0
    assert list(payloads) == points.grid_id.tolist()
    assert payloads["lat18.5_lon73.5"]["daily"]["temperature_2m_mean"] == [24.0]
    assert payloads["lat19.5_lon74.5"]["daily"]["temperature_2m_mean"] == [29.0]


def test_era5_grid_waits_for_the_provider_hourly_reset_and_resumes(tmp_path, monkeypatch):
    import requests
    import heatwave_benchmark.data as data_module

    payload = _era5_payload(dates=("2024-01-01",))
    calls = 0
    delays = []

    def fetch(points, start_date, end_date):
        nonlocal calls
        calls += 1
        if calls == 1:
            response = type(
                "Response",
                (),
                {
                    "status_code": 429,
                    "headers": {},
                    "text": '{"reason":"Hourly API request limit exceeded. Please try again in the next hour."}',
                },
            )()
            raise requests.HTTPError("rate limited", response=response)
        return {str(points.grid_id.iloc[0]): payload}, b"[]"

    monkeypatch.setattr(data_module, "fetch_era5_daily_batch", fetch)
    monkeypatch.setattr(data_module.time, "sleep", delays.append)
    points = pd.DataFrame(
        [{"grid_id": "lat18.5_lon73.5", "latitude": 18.5, "longitude": 73.5}]
    )

    result = download_era5_grid(
        points,
        tmp_path,
        "2024-01-01",
        "2024-01-01",
        max_workers=1,
        retries=1,
        batch_size=1,
    )

    assert calls == 2
    assert delays and 60 <= delays[0] <= 3602
    assert result.loc[0, "status"] == "downloaded"


def test_era5_grid_stops_on_daily_quota_and_records_remaining_cells_as_deferred(
    tmp_path, monkeypatch
):
    import requests
    import heatwave_benchmark.data as data_module

    calls = 0

    def fetch(points, start_date, end_date):
        nonlocal calls
        calls += 1
        response = type(
            "Response",
            (),
            {
                "status_code": 429,
                "headers": {},
                "text": '{"reason":"Daily API request limit exceeded. Please try again tomorrow."}',
            },
        )()
        raise requests.HTTPError("rate limited", response=response)

    monkeypatch.setattr(data_module, "fetch_era5_daily_batch", fetch)
    monkeypatch.setattr(data_module.time, "sleep", lambda _: None)
    points = pd.DataFrame(
        [
            {"grid_id": "lat18.5_lon73.5", "latitude": 18.5, "longitude": 73.5},
            {"grid_id": "lat19.5_lon74.5", "latitude": 19.5, "longitude": 74.5},
        ]
    )

    result = download_era5_grid(
        points,
        tmp_path,
        "2024-01-01",
        "2024-01-01",
        max_workers=1,
        retries=4,
        batch_size=1,
    )

    assert calls == 1
    assert result.grid_id.tolist() == points.grid_id.tolist()
    assert result.status.tolist() == ["deferred", "deferred"]
    assert result.error.str.contains("Daily API request limit exceeded").all()
    saved = pd.read_csv(tmp_path / "download_manifest.csv")
    assert saved.status.tolist() == ["deferred", "deferred"]


def test_era5_seasonal_grid_downloads_only_predictor_issue_windows_and_pads_other_days(
    tmp_path, monkeypatch
):
    import heatwave_benchmark.data as data_module

    requested = []
    points = pd.DataFrame(
        [{"grid_id": "lat18.5_lon73.5", "latitude": 18.5, "longitude": 73.5}]
    )

    def fake_download(window_points, destination_dir, start_date, end_date, **kwargs):
        requested.append((start_date, end_date))
        destination = Path(destination_dir)
        destination.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range(start_date, end_date, freq="D")
        variables = [name for name, _ in ERA5_DAILY_VARIABLES]
        payload = {
            "latitude": 18.5,
            "longitude": 73.5,
            "elevation": 600,
            "timezone": "Asia/Kolkata",
            "daily_units": {"time": "iso8601", **{name: "unit" for name in variables}},
            "daily": {
                "time": dates.strftime("%Y-%m-%d").tolist(),
                **{name: [1.0] * len(dates) for name in variables},
            },
        }
        (destination / f"{points.grid_id.iloc[0]}.json").write_text(json.dumps(payload))
        return pd.DataFrame(
            [{"grid_id": points.grid_id.iloc[0], "status": "downloaded", "daily_rows": len(dates)}]
        )

    monkeypatch.setattr(data_module, "download_era5_grid", fake_download)

    manifest = download_era5_seasonal_grid(points, tmp_path, 2024, 2024)

    assert requested == [("2024-02-22", "2024-06-30")]
    assert manifest.loc[0, "status"] == "downloaded"
    payload = json.loads((tmp_path / f"{points.grid_id.iloc[0]}.json").read_text())
    assert len(payload["daily"]["time"]) == 366
    assert payload["daily"]["temperature_2m_mean"][0] is None
    start = payload["daily"]["time"].index("2024-02-22")
    end = payload["daily"]["time"].index("2024-06-30")
    assert payload["daily"]["temperature_2m_mean"][start] == 1.0
    assert payload["daily"]["temperature_2m_mean"][end] == 1.0
    assert payload["daily"]["temperature_2m_mean"][end + 1] is None


def test_era5_seasonal_grid_waits_until_daily_quota_reset_and_resumes(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    import heatwave_benchmark.data as data_module

    points = pd.DataFrame(
        [{"grid_id": "lat18.5_lon73.5", "latitude": 18.5, "longitude": 73.5}]
    )
    calls = 0
    delays = []

    def fake_download(window_points, destination_dir, start_date, end_date, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return pd.DataFrame(
                [{
                    "grid_id": points.grid_id.iloc[0],
                    "status": "deferred",
                    "error": "Daily API request limit exceeded. Please try again tomorrow.",
                }]
            )
        destination = Path(destination_dir)
        destination.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range(start_date, end_date, freq="D")
        variables = [name for name, _ in ERA5_DAILY_VARIABLES]
        payload = {
            "latitude": 18.5,
            "longitude": 73.5,
            "elevation": 600,
            "timezone": "Asia/Kolkata",
            "daily_units": {"time": "iso8601", **{name: "unit" for name in variables}},
            "daily": {
                "time": dates.strftime("%Y-%m-%d").tolist(),
                **{name: [1.0] * len(dates) for name in variables},
            },
        }
        (destination / f"{points.grid_id.iloc[0]}.json").write_text(json.dumps(payload))
        return pd.DataFrame(
            [{"grid_id": points.grid_id.iloc[0], "status": "downloaded", "daily_rows": len(dates)}]
        )

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2024, 2, 1, 23, 59, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(data_module, "download_era5_grid", fake_download)
    monkeypatch.setattr(data_module, "datetime", FixedDatetime)
    monkeypatch.setattr(data_module.time, "sleep", delays.append)

    result = download_era5_seasonal_grid(points, tmp_path, 2024, 2024)

    assert calls == 2
    assert delays == [pytest.approx(6 * 60, abs=1)]
    assert result.loc[0, "status"] == "downloaded"


def test_reads_imd_little_endian_grid_binary_with_coordinates_and_missing_values(tmp_path):
    values = np.zeros((366, 31, 31), dtype="<f4")
    values[0, 0, 0] = 99.9
    values[1, 30, 30] = 44.5
    path = tmp_path / "MaxT_2024.GRD"
    values.tofile(path)

    result = read_imd_tmax_year(path, year=2024)

    assert result.dims == ("time", "latitude", "longitude")
    assert result.sizes == {"time": 366, "latitude": 31, "longitude": 31}
    assert pd.Timestamp(result.time.values[0]) == pd.Timestamp("2024-01-01")
    assert float(result.latitude.values[0]) == 7.5
    assert float(result.latitude.values[-1]) == 37.5
    assert float(result.longitude.values[0]) == 67.5
    assert float(result.longitude.values[-1]) == 97.5
    assert np.isnan(result.values[0, 0, 0])
    assert float(result.values[1, 30, 30]) == 44.5


def test_reads_imd_binary_axis_order_as_latitude_then_longitude(tmp_path):
    # Raw records use latitude, longitude. This asymmetric test catches an
    # accidental axis transpose that a square grid could otherwise hide.
    values = np.zeros((365, 31, 31), dtype="<f4")
    values[0, 11, 2] = 42.0  # latitude index 11, longitude index 2
    path = tmp_path / "MaxT_2023.GRD"
    values.tofile(path)

    result = read_imd_tmax_year(path, year=2023)

    assert result.isel(time=0).sel(latitude=18.5, longitude=69.5).item() == 42.0
    assert result.isel(time=0).sel(latitude=18.5, longitude=69.5).item() != result.isel(
        time=0
    ).sel(
        latitude=9.5, longitude=78.5
    ).item()


def test_reads_non_leap_year_with_365_daily_records(tmp_path):
    values = np.full((365, 31, 31), 35.0, dtype="<f4")
    path = tmp_path / "MaxT_2023.GRD"
    values.tofile(path)

    result = read_imd_tmax_year(path, year=2023)

    assert result.sizes["time"] == 365
    assert pd.Timestamp(result.time.values[-1]) == pd.Timestamp("2023-12-31")


def test_loads_provenance_tracked_imd_terrain_mask_in_xarray_axis_order():
    mask = load_imd_terrain_mask()

    assert mask.shape == (31, 31)
    assert mask.dims == ("latitude", "longitude")
    assert set(np.unique(mask.values)) == {-1, 0, 1, 2}
    assert float(mask.latitude.values[0]) == 7.5
    assert float(mask.longitude.values[0]) == 67.5


def test_rejects_binary_file_with_wrong_record_count(tmp_path):
    path = tmp_path / "bad.GRD"
    np.zeros(10, dtype="<f4").tofile(path)

    with pytest.raises(ValueError, match="expected"):
        read_imd_tmax_year(path, year=2024)


def test_download_saves_only_a_valid_official_year_file(tmp_path):
    expected = np.zeros((365, 31, 31), dtype="<f4").tobytes()

    class Response:
        content = expected
        headers = {"Content-Disposition": 'attachment; filename="MaxT_2023.GRD"'}

        @staticmethod
        def raise_for_status():
            return None

    class Session:
        called = False

        def post(self, url, data, timeout):
            self.called = True
            assert url.endswith("/maxtemp.php")
            assert data == {"maxtemp": "2023"}
            assert timeout
            return Response()

    session = Session()
    path = download_imd_tmax_year(2023, tmp_path, session=session)

    assert session.called
    assert path.name == "MaxT_2023.GRD"
    assert path.read_bytes() == expected
    assert not Path(f"{path}.part").exists()


def test_manifest_records_checksum_size_and_source(tmp_path):
    path = tmp_path / "sample.GRD"
    path.write_bytes(b"grid")

    manifest = build_file_manifest(
        path,
        source_url="https://example.org/data",
        dataset="IMD daily maximum temperature",
        year=2023,
        resolution_degrees=1.0,
    )

    assert manifest["file"] == "sample.GRD"
    assert manifest["size_bytes"] == 4
    assert manifest["sha256"] == "0f2a693e93e2ed7008feb5b7714bbab236e3907e64188199e3be1ae367f49c40"
    assert manifest["source_url"] == "https://example.org/data"
    assert manifest["year"] == 2023


def test_parses_nasa_power_daily_json_into_longitudinal_wide_rows():
    payload = {
        "geometry": {"type": "Point", "coordinates": [73.75, 18.5, 600.0]},
        "properties": {
            "parameter": {
                "T2M": {"19810101": 24.0, "19810102": -999.0},
                "T2M_MAX": {"19810101": 32.0},
            }
        },
        "header": {"fill_value": -999.0, "time_standard": "LST", "sources": ["MERRA2"]},
        "parameters": {"T2M": {"units": "C"}, "T2M_MAX": {"units": "C"}},
    }

    result = parse_nasa_power_daily_response(
        payload,
        grid_id="lat18_lon73",
        latitude=18.5,
        longitude=73.5,
        parameters=("T2M", "T2M_MAX"),
    )

    assert result["date"].tolist() == [pd.Timestamp("1981-01-01"), pd.Timestamp("1981-01-02")]
    assert result.loc[0, "t2m"] == 24.0
    assert np.isnan(result.loc[1, "t2m"])
    assert result.loc[0, "t2m_max"] == 32.0
    assert np.isnan(result.loc[1, "t2m_max"])
    assert result["grid_id"].tolist() == ["lat18_lon73", "lat18_lon73"]
    assert result.loc[0, "api_response_latitude"] == 18.5
    assert result.loc[0, "api_response_longitude"] == 73.75


def test_nasa_power_download_requests_only_supported_parameters_and_lst():
    payload = {
        "geometry": {"coordinates": [73.5, 18.5, 600.0]},
        "properties": {"parameter": {"T2M": {"20240101": 24.0}}},
        "header": {"fill_value": -999.0, "time_standard": "LST", "sources": ["MERRA2"]},
        "parameters": {"T2M": {"units": "C"}},
    }

    class Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return payload

    class Session:
        def get(self, url, params, timeout):
            assert url.endswith("/daily/point")
            assert params["parameters"] == "T2M"
            assert params["time-standard"] == "LST"
            assert params["community"] == "RE"
            assert timeout
            return Response()

    result = download_nasa_power_daily_point(
        latitude=18.5,
        longitude=73.5,
        start_date="20240101",
        end_date="20240101",
        grid_id="sample",
        parameters=("T2M",),
        session=Session(),
    )

    assert result.loc[0, "t2m"] == 24.0


def test_nasa_power_parser_rejects_incomplete_parameter_response():
    payload = {
        "geometry": {"coordinates": [73.5, 18.5, 600.0]},
        "properties": {"parameter": {"T2M": {"20240101": 24.0}}},
        "header": {"fill_value": -999.0, "time_standard": "LST", "sources": ["MERRA2"]},
        "parameters": {"T2M": {"units": "C"}},
    }

    with pytest.raises(ValueError, match="parameter"):
        parse_nasa_power_daily_response(
            payload,
            grid_id="sample",
            latitude=18.5,
            longitude=73.5,
            parameters=("T2M", "T2M_MAX"),
        )


def test_nasa_power_download_writes_and_reuses_valid_raw_json_cache(tmp_path):
    payload = {
        "geometry": {"coordinates": [73.5, 18.5, 600.0]},
        "properties": {"parameter": {"T2M": {"20240101": 24.0, "20240102": 25.0}}},
        "header": {"fill_value": -999.0, "time_standard": "LST", "sources": ["MERRA2"]},
        "parameters": {"T2M": {"units": "C"}},
    }

    class Response:
        content = json.dumps(payload).encode()

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return payload

    class Session:
        calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            return Response()

    session = Session()
    cache = tmp_path / "point.json"
    first = download_nasa_power_daily_point(
        18.5, 73.5, "20240101", "20240102", "sample", ("T2M",), session=session, cache_path=cache
    )
    second = download_nasa_power_daily_point(
        18.5, 73.5, "20240101", "20240102", "sample", ("T2M",), session=session, cache_path=cache
    )

    assert session.calls == 1
    assert cache.read_bytes() == Response.content
    pd.testing.assert_frame_equal(first, second)


def test_nasa_power_download_rejects_partial_daily_response():
    payload = {
        "geometry": {"coordinates": [73.5, 18.5, 600.0]},
        "properties": {"parameter": {"T2M": {"20240101": 24.0}}},
        "header": {"fill_value": -999.0, "time_standard": "LST", "sources": ["MERRA2"]},
        "parameters": {"T2M": {"units": "C"}},
    }

    class Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return payload

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    with pytest.raises(ValueError, match="daily coverage"):
        download_nasa_power_daily_point(
            18.5, 73.5, "20240101", "20240102", "sample", ("T2M",), session=Session()
        )


def test_nasa_grid_downloader_records_point_provenance_and_enforces_api_limit(tmp_path, monkeypatch):
    import heatwave_benchmark.data as data_module

    def fake_download(**kwargs):
        Path(kwargs["cache_path"]).write_text("{}", encoding="utf-8")
        return pd.DataFrame(
            {
                "grid_id": [kwargs["grid_id"]],
                "date": [pd.Timestamp("2024-01-01")],
                "api_response_latitude": [18.5],
                "api_response_longitude": [73.75],
            }
        )

    monkeypatch.setattr(data_module, "download_nasa_power_daily_point", fake_download)
    points = pd.DataFrame(
        [{"grid_id": "lat18.5_lon73.5", "latitude": 18.5, "longitude": 73.5}]
    )

    manifest = download_nasa_power_grid(
        points,
        tmp_path,
        start_date="2024-01-01",
        end_date="2024-01-01",
        parameters=("T2M",),
        max_workers=1,
        retries=0,
    )

    assert manifest.loc[0, "status"] == "downloaded"
    assert manifest.loc[0, "api_response_longitude"] == 73.75
    assert manifest.loc[0, "parameters"] == "T2M"
    assert manifest.loc[0, "start_date"] == "2024-01-01"
    assert len(manifest.loc[0, "sha256"]) == 64
    with pytest.raises(ValueError, match="1 and 5"):
        download_nasa_power_grid(points, tmp_path, "2024-01-01", "2024-01-01", max_workers=6)
