import numpy as np
import pandas as pd
import pytest

from heatwave_benchmark.metrics import (
    binary_metrics,
    bootstrap_group_metric_means,
    bootstrap_paired_group_differences,
    threshold_at_max_false_alarm_rate,
)


def test_binary_metrics_report_rare_event_pr_auc_and_brier_score():
    y = np.array([0, 0, 0, 1])
    probabilities = np.array([0.1, 0.2, 0.1, 0.8])

    result = binary_metrics(y, probabilities, threshold=0.5)

    assert result["prevalence"] == pytest.approx(0.25)
    assert result["average_precision"] == pytest.approx(1.0)
    assert result["brier_score"] == pytest.approx((0.01 + 0.04 + 0.01 + 0.04) / 4)
    assert result["event_recall"] == 1.0
    assert result["false_alarm_rate"] == 0.0


def test_threshold_is_selected_from_validation_at_a_fixed_false_alarm_rate():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.9])

    threshold = threshold_at_max_false_alarm_rate(y, p, max_false_alarm_rate=0.25)

    assert threshold == pytest.approx(0.7)


def test_binary_metrics_reject_invalid_probabilities():
    with pytest.raises(ValueError, match="probabilities"):
        binary_metrics(np.array([0, 1]), np.array([0.2, 1.2]))


def test_group_bootstrap_reports_reproducible_macro_metric_intervals():
    groups = pd.DataFrame(
        {
            "average_precision": [0.2, 0.5, 0.8],
            "brier_score": [0.4, 0.3, 0.2],
        }
    )

    first = bootstrap_group_metric_means(groups, ["average_precision", "brier_score"], n_bootstrap=200, seed=4)
    second = bootstrap_group_metric_means(groups, ["average_precision", "brier_score"], n_bootstrap=200, seed=4)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[first.metric == "average_precision", "estimate"].item() == pytest.approx(0.5)
    assert np.all(first.lower_95 <= first.estimate)
    assert np.all(first.upper_95 >= first.estimate)


def test_paired_group_bootstrap_compares_each_model_to_the_same_group_baseline():
    groups = pd.DataFrame(
        {
            "spatial_block": ["a", "b", "c", "a", "b", "c"],
            "model": ["climatology"] * 3 + ["hist_gradient_boosting"] * 3,
            "average_precision": [0.1, 0.2, 0.3, 0.4, 0.4, np.nan],
            "brier_score": [0.5, 0.4, 0.3, 0.4, 0.3, 0.2],
        }
    )

    result = bootstrap_paired_group_differences(
        groups,
        group_col="spatial_block",
        metric_columns=["average_precision", "brier_score"],
        baseline_model="climatology",
        n_bootstrap=200,
        seed=5,
    )

    ap = result.loc[result.metric == "average_precision"].iloc[0]
    brier = result.loc[result.metric == "brier_score"].iloc[0]
    assert ap.model == "hist_gradient_boosting"
    assert ap.estimate == pytest.approx(0.25)
    assert ap.n_groups == 2
    assert brier.estimate == pytest.approx(-0.1)
    assert brier.n_groups == 3
    assert ap.lower_95 <= ap.estimate <= ap.upper_95

\n