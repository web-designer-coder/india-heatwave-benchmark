import numpy as np
import pandas as pd
import pytest
from importlib.util import find_spec

from heatwave_benchmark.experiments import (
    run_model_comparison,
    select_model_features,
    stratified_training_sample,
)


def test_select_model_features_excludes_targets_and_identifiers():
    frame = pd.DataFrame(
        {
            "grid_id": ["a"],
            "spatial_block": ["b"],
            "date": pd.to_datetime(["2020-04-01"]),
            "t2m_lag_0": [20.0],
            "latitude": [18.5],
            "doy_sin": [0.9],
            "target_h1": [1],
            "current_heatwave": [0],
        }
    )

    features = select_model_features(frame)

    assert features == ["t2m_lag_0", "latitude", "doy_sin"]


def test_training_sample_is_deterministic_and_keeps_both_classes():
    frame = pd.DataFrame({"target": [0] * 90 + [1] * 10, "x": np.arange(100)})

    first = stratified_training_sample(frame, "target", maximum_rows=20, random_state=7)
    second = stratified_training_sample(frame, "target", maximum_rows=20, random_state=7)

    assert first.index.tolist() == second.index.tolist()
    assert len(first) == 20
    assert set(first.target) == {0, 1}


def test_model_comparison_selects_calibration_and_threshold_without_test_labels():
    rng = np.random.default_rng(22)
    training = pd.DataFrame({"x": rng.normal(size=60), "target": np.tile([0, 1], 30)})
    training["x"] = training["x"] + training["target"] * 1.2
    validation = training.sample(frac=1, random_state=1).reset_index(drop=True)
    test = training.sample(frac=1, random_state=2).reset_index(drop=True)
    for frame in (training, validation, test):
        frame["grid_id"] = "cell"
        frame["date"] = pd.date_range("2020-03-01", periods=len(frame))
        frame["current_heatwave"] = frame["target"]
        frame["spatial_block"] = "block"

    result = run_model_comparison(
        training,
        validation,
        test,
        target_col="target",
        feature_columns=["x"],
        max_training_rows=60,
        random_state=3,
    )

    expected_models = {
        "climatology",
        "persistence",
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    if find_spec("xgboost") is not None:
        expected_models.add("xgboost")
    assert set(result.metrics["model"]) == expected_models
    assert set(result.metrics["partition"]) == {"validation", "test"}
    assert "best_validation_model" in result.summary
    assert set(result.test_predictions.columns).issuperset({"target", "spatial_block"})
    assert np.isfinite(result.test_predictions.select_dtypes(include=["number"]).to_numpy()).all()


def test_model_comparison_can_run_one_preselected_estimator_for_sensitivity_analysis():
    rng = np.random.default_rng(44)
    training = pd.DataFrame({"x": rng.normal(size=60), "target": np.tile([0, 1], 30)})
    training["x"] += training["target"] * 0.8
    validation = training.copy()
    test = training.copy()
    for frame in (training, validation, test):
        frame["grid_id"] = "cell"
        frame["date"] = pd.date_range("2020-03-01", periods=len(frame))
        frame["current_heatwave"] = frame["target"]
        frame["spatial_block"] = "block"

    result = run_model_comparison(
        training,
        validation,
        test,
        target_col="target",
        feature_columns=["x"],
        max_training_rows=60,
        random_state=9,
        estimator_names=("hist_gradient_boosting",),
    )

    assert set(result.metrics.model) == {
        "climatology",
        "persistence",
        "hist_gradient_boosting",
    }

\n