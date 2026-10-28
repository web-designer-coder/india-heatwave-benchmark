import numpy as np
import pandas as pd
import pytest
from importlib.util import find_spec

from heatwave_benchmark.models import (
    ClimatologyModel,
    PersistenceModel,
    build_estimators,
    make_training_weights,
    fit_platt_calibrator,
    apply_platt_calibrator,
)


def _sample_frame():
    return pd.DataFrame(
        {
            "grid_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "date": pd.to_datetime(
                ["2020-04-01", "2020-04-16", "2020-04-01", "2020-04-16",
                 "2020-04-01", "2020-04-16", "2020-04-01", "2020-04-16"]
            ),
            "current_heatwave": [0, 1, 0, 1, 0, 0, 0, 1],
            "target": [0, 1, 0, 1, 0, 0, 0, 1],
            "x": [0.0, 1.0, 0.1, 0.9, 0.2, 0.3, 0.3, 0.8],
            "spatial_block": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )


def test_persistence_returns_issue_day_heatwave_probability_with_prevalence_fallback():
    frame = _sample_frame()
    model = PersistenceModel().fit(frame, "target")
    query = pd.DataFrame({"current_heatwave": [1.0, 0.0, np.nan]})

    probabilities = model.predict_proba(query)

    np.testing.assert_array_equal(probabilities, np.array([1.0, 0.0, 3 / 8]))


def test_climatology_model_uses_training_records_and_falls_back_for_unseen_grid():
    train = _sample_frame()
    model = ClimatologyModel().fit(train, "target")
    query = pd.DataFrame(
        {
            "grid_id": ["a", "new"],
            "date": pd.to_datetime(["2020-04-16", "2020-04-16"]),
        }
    )

    probabilities = model.predict_proba(query)

    assert probabilities.shape == (2,)
    assert probabilities[0] > probabilities[1]
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_spatial_block_balancing_equalizes_block_mass_and_class_weights_are_finite():
    labels = np.array([0, 0, 1, 0, 0, 0, 1], dtype=float)
    blocks = np.array(["A", "A", "A", "B", "B", "B", "B"])

    weights = make_training_weights(
        labels, blocks, balance_classes=True, balance_spatial_blocks=True
    )

    assert np.isfinite(weights).all()
    assert np.all(weights > 0)
    assert weights[blocks == "A"].sum() == pytest.approx(weights[blocks == "B"].sum())


def test_all_estimators_fit_and_return_finite_probabilities():
    frame = _sample_frame()
    features = frame[["x"]].to_numpy(dtype=float)
    labels = frame["target"].to_numpy(dtype=int)
    weights = make_training_weights(labels)

    for name, model in build_estimators(random_state=9).items():
        if name in {"hist_gradient_boosting", "xgboost"}:
            model.fit(features, labels, sample_weight=weights)
        else:
            model.fit(features, labels, classifier__sample_weight=weights)
        probabilities = model.predict_proba(features)[:, 1]
        assert np.isfinite(probabilities).all()
        assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_validation_only_platt_calibrator_outputs_bounded_probabilities():
    y = np.array([0, 0, 0, 1, 1, 1])
    raw = np.array([0.05, 0.15, 0.4, 0.55, 0.8, 0.95])
    calibrator = fit_platt_calibrator(y, raw)

    calibrated = apply_platt_calibrator(calibrator, np.array([0.1, 0.9]))

    assert calibrated[0] < calibrated[1]
    assert np.all((calibrated >= 0) & (calibrated <= 1))

\n