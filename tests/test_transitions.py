import numpy as np
import pandas as pd
import pytest

from heatwave_benchmark.transitions import (
    fit_onset_model_search,
    run_state_transition_comparison,
)


def make_transition_frames(seed: int, rows: int = 240):
    partitions = []
    for partition_index in range(3):
        rng = np.random.default_rng(seed + partition_index)
        state = np.arange(rows, dtype=np.int8) % 2
        x = rng.normal(size=rows)
        transition_signal = np.where(state == 0, x, -x)
        probability = 1 / (1 + np.exp(-1.5 * transition_signal))
        target = rng.binomial(1, probability).astype(np.int8)
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-03-01", periods=rows, freq="D"),
                "grid_id": [f"cell_{i % 8}" for i in range(rows)],
                "spatial_block": [f"block_{i % 3}" for i in range(rows)],
                "x": x,
                "current_heatwave": state,
                "target": target,
            }
        )
        partitions.append(frame)
    return tuple(partitions)


def test_state_transition_comparison_routes_issue_states_to_separate_branches():
    training, validation, test = make_transition_frames(seed=11)

    result = run_state_transition_comparison(
        training,
        validation,
        test,
        feature_columns=["x"],
        max_training_rows=80,
        random_state=5,
    )

    predictions = result.test_predictions
    assert predictions["transition_raw_probability"].notna().all()
    assert predictions["transition_calibrated_probability"].between(0, 1).all()
    assert np.any(
        ~np.isclose(
            predictions["transition_raw_probability"],
            predictions["pooled_hist_gradient_boosting_raw_probability"],
        )
    )
    assert set(result.metrics["issue_state"]) == {
        "all",
        "no_active_heatwave",
        "active_heatwave",
    }
    assert set(result.metrics["model"]) == {
        "pooled_hist_gradient_boosting",
        "pooled_hist_gradient_boosting_with_issue_state",
        "state_transition_hist_gradient_boosting",
        "climatology",
        "persistence",
    }
    assert "pooled_hist_gradient_boosting_with_issue_state_calibrated_probability" in predictions
    assert "state-specific Platt" in result.summary["calibration_policy_by_model"][
        "pooled_hist_gradient_boosting_with_issue_state"
    ]


def test_state_transition_thresholds_are_fitted_on_validation_by_state():
    training, validation, test = make_transition_frames(seed=12)

    result = run_state_transition_comparison(
        training,
        validation,
        test,
        feature_columns=["x"],
        max_training_rows=80,
        random_state=5,
    )

    thresholds = result.summary["transition_thresholds_calibrated_by_state"]
    for state in (0, 1):
        observed = result.test_predictions.loc[
            result.test_predictions["current_heatwave"].eq(state),
            "transition_threshold_calibrated",
        ].unique()
        assert len(observed) == 1
        assert observed[0] == pytest.approx(thresholds[state])

    altered_test = test.copy()
    altered_test["target"] = 1 - altered_test["target"]
    repeated = run_state_transition_comparison(
        training,
        validation,
        altered_test,
        feature_columns=["x"],
        max_training_rows=80,
        random_state=5,
    )
    assert repeated.summary["transition_thresholds_calibrated_by_state"] == thresholds
    np.testing.assert_allclose(
        repeated.test_predictions["transition_calibrated_probability"],
        result.test_predictions["transition_calibrated_probability"],
    )


def test_state_transition_rejects_missing_issue_state_and_insufficient_train_class():
    training, validation, test = make_transition_frames(seed=13)
    test.loc[test.index[0], "current_heatwave"] = np.nan

    with pytest.raises(ValueError, match="valid target and binary issue-day state"):
        run_state_transition_comparison(
            training,
            validation,
            test,
            feature_columns=["x"],
            max_training_rows=80,
        )

    training, validation, test = make_transition_frames(seed=14)
    training.loc[
        training["current_heatwave"].eq(0), "target"
    ] = 0
    with pytest.raises(ValueError, match="both target classes.*state 0"):
        run_state_transition_comparison(
            training,
            validation,
            test,
            feature_columns=["x"],
            max_training_rows=80,
        )


def test_state_transition_requires_two_training_rows_per_branch():
    training, validation, test = make_transition_frames(seed=15)

    with pytest.raises(ValueError, match="at least four"):
        run_state_transition_comparison(
            training,
            validation,
            test,
            feature_columns=["x"],
            max_training_rows=2,
        )


def test_onset_search_uses_only_state_zero_and_returns_onset_predictions():
    training, validation, test = make_transition_frames(seed=51, rows=400)
    result = fit_onset_model_search(
        training,
        validation,
        test,
        feature_columns=["x"],
        parameter_grid=[{"max_leaf_nodes": 7, "min_samples_leaf": 10, "l2_regularization": 1.0}],
    )
    assert result.summary["training_rows_used"] == int(training["current_heatwave"].eq(0).sum())
    assert result.summary["validation_rows_used"] == int(validation["current_heatwave"].eq(0).sum())
    assert result.test_predictions["current_heatwave"].eq(0).all()
    assert result.test_predictions["onset_probability"].between(0, 1).all()
    assert {"average_precision", "brier_score", "event_recall", "false_alarm_rate"}.issubset(result.metrics)


def test_onset_search_uses_all_state_zero_rows_and_rejects_missing_onset_class():
    training, validation, test = make_transition_frames(seed=52, rows=240)
    # Replication checks the onset fit count against the complete eligible set.
    training = pd.concat([training] * 3, ignore_index=True)
    result = fit_onset_model_search(
        training,
        validation,
        test,
        feature_columns=["x"],
        parameter_grid=[{"max_leaf_nodes": 7, "min_samples_leaf": 10, "l2_regularization": 1.0}],
    )
    assert result.summary["training_rows_used"] == int(training["current_heatwave"].eq(0).sum())

    training.loc[training["current_heatwave"].eq(0), "target"] = 0
    with pytest.raises(ValueError, match="onset.*both target classes"):
        fit_onset_model_search(
            training,
            validation,
            test,
            feature_columns=["x"],
            parameter_grid=[{"max_leaf_nodes": 7, "min_samples_leaf": 10, "l2_regularization": 1.0}],
        )
