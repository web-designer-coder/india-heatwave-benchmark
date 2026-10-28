import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.run_transition_benchmark import _run_onset_model, default_output_directory, main


def make_annual_feature_frame(seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(1981, 2026):
        for day_index in range(10):
            for cell_index in range(8):
                state = (day_index + cell_index) % 2
                feature = float(rng.normal())
                transition_signal = feature if state == 0 else -feature
                probability = 1 / (1 + np.exp(-1.5 * transition_signal))
                rows.append(
                    {
                        "date": pd.Timestamp(year, 3, 1) + pd.Timedelta(days=day_index),
                        "grid_id": f"cell_{cell_index}",
                        "spatial_block": f"block_{cell_index % 4}",
                        "latitude": 8.5 + cell_index,
                        "longitude": 70.5 + cell_index * 5,
                        "terrain_code": cell_index % 3,
                        "doy_sin": 0.1,
                        "doy_cos": 0.9,
                        "t2m_lag_0": feature,
                        "current_heatwave": state,
                        "target_h1": int(rng.binomial(1, probability)),
                    }
                )
    return pd.DataFrame(rows)


def test_default_output_directory_separates_predictor_sources():
    assert default_output_directory("nasa_power").name == "generated_state_transition_nasa_power"
    assert default_output_directory("era5").name == "generated_state_transition_era5"


def test_transition_runner_writes_source_scoped_temporal_outputs(tmp_path):
    features_path = tmp_path / "features.csv.gz"
    output_directory = tmp_path / "transition_results"
    make_annual_feature_frame().to_csv(features_path, index=False, compression="gzip")

    result = main(
        [
            "--predictor-source",
            "nasa_power",
            "--features",
            str(features_path),
            "--output-directory",
            str(output_directory),
            "--max-training-rows",
            "40",
            "--skip-spatial",
        ]
    )

    assert result == 0
    metrics = pd.read_csv(output_directory / "transition_metrics.csv")
    predictions = pd.read_csv(output_directory / "transition_test_predictions.csv.gz")
    summary = pd.read_json(output_directory / "transition_summary.json", typ="series")
    assert set(metrics.scenario) == {"temporal"}
    assert "state_transition_hist_gradient_boosting" in set(metrics.model)
    assert predictions.current_heatwave.isin((0, 1)).all()
    assert summary["predictor_source"] == "nasa_power"
    grouped = pd.read_csv(output_directory / "transition_group_metrics.csv")
    intervals = pd.read_csv(output_directory / "transition_macro_bootstrap_intervals.csv")
    paired = pd.read_csv(output_directory / "transition_paired_intervals.csv")
    assert set(grouped.probability_type) == {"raw", "platt_calibrated"}
    assert set(intervals.probability_type) == {"raw", "platt_calibrated"}
    assert set(paired.baseline_model) == {
        "pooled_hist_gradient_boosting",
        "pooled_hist_gradient_boosting_with_issue_state",
    }
    assert (
        paired.model.eq("state_transition_hist_gradient_boosting")
        & paired.baseline_model.eq("pooled_hist_gradient_boosting_with_issue_state")
    ).any()


def test_transition_runner_rejects_a_missing_feature_frame(tmp_path):
    missing = tmp_path / "missing.csv.gz"
    with pytest.raises(FileNotFoundError, match="feature frame"):
        main(["--features", str(missing), "--skip-spatial"])


def test_onset_runner_selects_on_expanding_folds_and_scores_state_zero(tmp_path):
    rng = np.random.default_rng(73)
    dates = pd.date_range("1981-01-01", "2025-12-31", freq="D")
    grid_ids = [f"cell_{index}" for index in range(4)]
    weather_values = rng.normal(30, 5, size=(len(dates), len(grid_ids))).astype("float32")
    weather_path = tmp_path / "era5.nc"
    xr.Dataset(
        {"t2m_max": (("time", "grid_id"), weather_values)},
        coords={"time": dates, "grid_id": grid_ids},
    ).to_netcdf(weather_path)

    rows = []
    weather_lookup = {
        (grid_id, date): weather_values[index, cell_index]
        for cell_index, grid_id in enumerate(grid_ids)
        for index, date in enumerate(dates)
    }
    for date in dates[(dates.month == 3) | (dates.month == 4)]:
        for cell_index, grid_id in enumerate(grid_ids):
            state = cell_index % 2
            feature = float(weather_lookup[(grid_id, date)])
            row = {
                "date": date,
                "grid_id": grid_id,
                "spatial_block": f"block_{cell_index // 2}",
                "latitude": 10.0 + cell_index,
                "longitude": 70.0 + cell_index,
                "terrain_code": 0,
                "doy_sin": 0.1,
                "doy_cos": 0.9,
                "current_heatwave": state,
                "target_h1": int(rng.random() < 0.3),
            }
            for lag in range(8):
                row[f"t2m_max_lag_{lag}"] = float(weather_lookup[(grid_id, date - pd.Timedelta(days=lag))])
            row["t2m_lag_0"] = feature
            rows.append(row)
    frame = pd.DataFrame(rows)

    candidates, predictions, summary, frozen_bundle = _run_onset_model(frame, weather_path)

    assert set(candidates["fold"]) == {1, 2, 3}
    assert candidates.groupby("fold").size().eq(8).all()
    assert summary["selected_parameters"]
    assert len(summary["feature_columns"]) == len(set(summary["feature_columns"]))
    assert summary["selection_folds"][0]["validation_years"] == [1996, 2001]
    assert predictions["current_heatwave"].eq(0).all()
    assert summary["test_status"].startswith("exploratory")
    assert frozen_bundle["selected_parameters"] == summary["selected_parameters"]
    assert frozen_bundle["probability_threshold"] > 0
    assert frozen_bundle["estimator"] is not None
    assert frozen_bundle["calibrator"] is not None
    assert not frozen_bundle["temperature_normals"].empty
