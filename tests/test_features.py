import pandas as pd
import pytest

from heatwave_benchmark import features
from heatwave_benchmark.features import build_lagged_features


def test_lagged_features_use_same_day_and_prior_calendar_days_only():
    frame = pd.DataFrame(
        {
            "grid_id": ["g1"] * 4,
            "date": pd.date_range("2020-04-01", periods=4, freq="D"),
            "tmax": [30.0, 31.0, 32.0, 33.0],
            "humidity": [40.0, 42.0, 45.0, 50.0],
            "terrain": ["plains"] * 4,
            "target": [0, 0, 1, 1],
        }
    )

    result = build_lagged_features(
        frame,
        predictor_columns=("tmax", "humidity"),
        context_columns=("terrain",),
        max_lag_days=2,
    )

    assert result.loc[2, "tmax_lag_0"] == 32.0
    assert result.loc[2, "tmax_lag_1"] == 31.0
    assert result.loc[2, "tmax_lag_2"] == 30.0
    assert result.loc[2, "humidity_lag_1"] == 42.0
    assert result.loc[2, "terrain"] == "plains"
    assert "target" not in result.columns


def test_lagged_features_match_calendar_dates_across_missing_days():
    frame = pd.DataFrame(
        {
            "grid_id": ["g1", "g1", "g1"],
            "date": pd.to_datetime(["2020-04-01", "2020-04-03", "2020-04-04"]),
            "tmax": [30.0, 32.0, 33.0],
        }
    )

    result = build_lagged_features(frame, predictor_columns=("tmax",), max_lag_days=2)

    assert result.loc[1, "tmax_lag_1"] != result.loc[1, "tmax_lag_1"]
    assert result.loc[1, "tmax_lag_2"] == 30.0


def test_lagged_features_keep_grids_separate():
    frame = pd.DataFrame(
        {
            "grid_id": ["g1", "g1", "g2", "g2"],
            "date": list(pd.date_range("2020-04-01", periods=2, freq="D")) * 2,
            "tmax": [30.0, 31.0, 40.0, 41.0],
        }
    )

    result = build_lagged_features(frame, predictor_columns=("tmax",), max_lag_days=1)

    assert result.loc[2, "tmax_lag_1"] != result.loc[2, "tmax_lag_1"]
    assert result.loc[3, "tmax_lag_1"] == 40.0


def test_lagged_features_reject_duplicate_grid_dates():
    frame = pd.DataFrame(
        {
            "grid_id": ["g1", "g1"],
            "date": pd.to_datetime(["2020-04-01", "2020-04-01"]),
            "tmax": [30.0, 31.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_lagged_features(frame, predictor_columns=("tmax",), max_lag_days=1)


def test_lagged_features_reject_missing_predictor_column():
    frame = pd.DataFrame(
        {"grid_id": ["g1"], "date": pd.to_datetime(["2020-04-01"])}
    )

    with pytest.raises(ValueError, match="predictor"):
        build_lagged_features(frame, predictor_columns=("tmax",))


def test_calendar_temperature_normals_are_grid_specific_and_wrap_year_boundary():
    dates = pd.date_range("2001-01-01", "2002-12-31", freq="D")
    weather = pd.concat(
        [
            pd.DataFrame(
                {
                    "grid_id": grid_id,
                    "date": dates,
                    "t2m_max": 20.0 + offset + 5.0 * (pd.Series(dates).dt.day_of_year % 10),
                }
            )
            for grid_id, offset in (("a", 0.0), ("b", 100.0))
        ],
        ignore_index=True,
    )

    normals = features.fit_calendar_temperature_normals(weather)

    assert set(normals.columns) == {"grid_id", "dayofyear", "t2m_max_normal"}
    assert len(normals) == 2 * 366
    jan_first_a = normals.loc[
        (normals.grid_id == "a") & (normals.dayofyear == 1), "t2m_max_normal"
    ].item()
    jan_first_b = normals.loc[
        (normals.grid_id == "b") & (normals.dayofyear == 1), "t2m_max_normal"
    ].item()
    assert jan_first_b - jan_first_a == pytest.approx(100.0)
    assert pd.notna(jan_first_a)


def test_temperature_anomaly_features_use_exact_lag_dates_and_expected_averages():
    normals = pd.DataFrame(
        {
            "grid_id": "a",
            "dayofyear": range(1, 367),
            "t2m_max_normal": range(1, 367),
        }
    )
    issue_date = pd.Timestamp("2000-03-05")
    frame = pd.DataFrame(
        {
            "grid_id": ["a"],
            "date": [issue_date],
            **{
                f"t2m_max_lag_{lag}": [
                    float((issue_date - pd.Timedelta(days=lag)).day_of_year + lag)
                ]
                for lag in range(8)
            },
        }
    )

    result = features.add_temperature_anomaly_features(frame, normals)

    assert result.loc[0, "t2m_max_anom_lag_0"] == pytest.approx(0.0)
    assert result.loc[0, "t2m_max_anom_lag_3"] == pytest.approx(3.0)
    assert result.loc[0, "t2m_max_anom_lag_7"] == pytest.approx(7.0)
    assert result.loc[0, "t2m_max_anom_mean_3d"] == pytest.approx(1.0)
    assert result.loc[0, "t2m_max_anom_mean_7d"] == pytest.approx(3.0)
    assert result.loc[0, "t2m_max_anom_change_0_3"] == pytest.approx(-3.0)


def test_temperature_anomaly_feature_api_is_available():
    assert callable(getattr(features, "fit_calendar_temperature_normals", None))
    assert callable(getattr(features, "add_temperature_anomaly_features", None))
