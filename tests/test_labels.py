import numpy as np
import pytest

from heatwave_benchmark.labels import (
    classify_gridcell_heatwave,
    compute_daily_normals,
    make_daily_gridcell_labels,
    make_horizon_target,
)


def test_classifies_imd_aligned_gridcell_events_at_threshold_boundaries():
    labels = classify_gridcell_heatwave(
        tmax=np.array([40.0, 45.0, 30.0, 37.0, 39.9, 39.0, 47.0]),
        departure=np.array([4.5, 0.0, 6.5, 4.5, 8.0, 6.4, 0.0]),
        terrain=np.array([0, 0, 1, 2, 0, 0, 0]),
    )

    np.testing.assert_array_equal(labels, np.array([1, 1, 2, 1, 0, 0, 2], dtype=float))


def test_departure_boundary_6_point_4_is_standard_and_above_is_severe():
    labels = classify_gridcell_heatwave(
        tmax=np.array([40.0, 40.0]),
        departure=np.array([6.4, 6.4001]),
        terrain=np.array(["plains", "plains"]),
    )

    np.testing.assert_array_equal(labels, np.array([1, 2], dtype=float))


def test_missing_inputs_stay_unknown_unless_absolute_threshold_is_sufficient():
    labels = classify_gridcell_heatwave(
        tmax=np.array([np.nan, 45.0, 41.0]),
        departure=np.array([5.0, np.nan, np.nan]),
        terrain=np.array(["plains", "plains", "plains"]),
    )

    assert np.isnan(labels[0])
    assert labels[1] == 1
    assert np.isnan(labels[2])


def test_absolute_temperature_path_does_not_apply_to_coastal_grid_cells():
    labels = classify_gridcell_heatwave(
        tmax=np.array([45.0, 45.0, 47.0]),
        departure=np.array([np.nan, 4.5, np.nan]),
        terrain=np.array(["coastal", "coastal", "coastal"]),
    )

    assert np.isnan(labels[0])
    assert labels[1] == 1
    assert np.isnan(labels[2])


def test_invalid_terrain_is_rejected():
    with pytest.raises(ValueError, match="terrain"):
        classify_gridcell_heatwave(
            tmax=np.array([40.0]),
            departure=np.array([5.0]),
            terrain=np.array(["desert"]),
        )


def test_horizon_target_uses_the_exact_future_day_and_masks_unknowns():
    labels = np.array([0.0, 1.0, 2.0, 0.0, np.nan])

    np.testing.assert_array_equal(
        make_horizon_target(labels, horizon=1),
        np.array([1.0, 1.0, 0.0, np.nan, np.nan]),
    )
    np.testing.assert_array_equal(
        make_horizon_target(labels, horizon=2),
        np.array([1.0, 0.0, np.nan, np.nan, np.nan]),
    )


def test_horizon_target_rejects_unsupported_lead():
    with pytest.raises(ValueError, match="horizon"):
        make_horizon_target(np.array([0.0, 1.0]), horizon=4)


def test_daily_normals_use_reference_years_and_circular_15_day_smoothing():
    import pandas as pd
    import xarray as xr

    dates = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    values = np.full((len(dates), 1), 10.0)
    values[dates == pd.Timestamp("2001-01-01")] = 30.0
    values[dates == pd.Timestamp("2000-12-31")] = 50.0
    field = xr.DataArray(
        values,
        dims=("time", "cell"),
        coords={"time": dates, "cell": ["grid-1"]},
        name="tmax",
    )

    normals = compute_daily_normals(
        field, start_year=2000, end_year=2002, window_days=3, min_years=2
    )

    # DOY 1 averages to 50/3, DOY 2 averages to 10, and leap-year
    # DOY 366 is 50; the centered window wraps across New Year.
    assert normals.sizes["dayofyear"] == 366
    assert normals.sel(dayofyear=1, cell="grid-1").item() == pytest.approx(230 / 9)


def test_daily_normals_reject_missing_reference_year_coverage():
    import pandas as pd
    import xarray as xr

    field = xr.DataArray(
        np.ones((365, 1)),
        dims=("time", "cell"),
        coords={"time": pd.date_range("2001-01-01", periods=365), "cell": [0]},
    )

    with pytest.raises(ValueError, match="cover every year"):
        compute_daily_normals(field, start_year=2000, end_year=2001, min_years=1)


def test_builds_gridcell_labels_on_the_matching_imd_terrain_grid():
    import pandas as pd
    import xarray as xr

    coordinates = {
        "latitude": [10.5],
        "longitude": [70.5, 71.5, 72.5],
    }
    tmax = xr.DataArray(
        np.array([[[40.0, 40.0, 45.0]]]),
        dims=("time", "latitude", "longitude"),
        coords={"time": pd.date_range("2023-01-01", periods=1), **coordinates},
    )
    normals = xr.DataArray(
        np.full((366, 1, 3), 35.0),
        dims=("dayofyear", "latitude", "longitude"),
        coords={"dayofyear": np.arange(1, 367), **coordinates},
    )
    terrain = xr.DataArray(
        np.array([[0, 2, -1]], dtype=np.int8),
        dims=("latitude", "longitude"),
        coords=coordinates,
    )

    labels = make_daily_gridcell_labels(tmax, normals, terrain)

    np.testing.assert_array_equal(labels.values, np.array([[[1.0, 1.0, np.nan]]]))
