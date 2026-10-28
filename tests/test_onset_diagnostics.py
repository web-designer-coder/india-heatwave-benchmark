import pandas as pd

from heatwave_benchmark.onset import (
    state_conditioned_group_metrics,
    state_conditioned_metrics,
)


def test_state_conditioned_metrics_carries_row_thresholds_into_no_event_slice():
    frame = pd.DataFrame(
        {
            "target": [1, 0, 1, 0, 0],
            "current_heatwave": [0, 0, 1, 1, 0],
            "probability": [0.8, 0.2, 0.9, 0.1, 0.3],
            "threshold": [0.5, 0.5, 0.95, 0.5, 0.5],
        }
    )

    result = state_conditioned_metrics(frame, "probability", "threshold")
    no_active = result.set_index("issue_state").loc["no_active_heatwave"]
    active = result.set_index("issue_state").loc["active_heatwave"]

    assert no_active["n"] == 3
    assert no_active["events"] == 1
    assert no_active["average_precision"] == 1.0
    assert no_active["event_recall"] == 1.0
    assert no_active["false_alarm_rate"] == 0.0
    assert active["n"] == 2
    assert active["event_recall"] == 0.0


def test_state_conditioned_group_metrics_keeps_year_and_state_strata():
    frame = pd.DataFrame(
        {
            "year": [2020, 2020, 2021, 2021],
            "target": [1, 0, 0, 1],
            "current_heatwave": [0, 0, 0, 1],
            "probability": [0.9, 0.2, 0.1, 0.9],
            "threshold": [0.5, 0.5, 0.5, 0.5],
        }
    )

    result = state_conditioned_group_metrics(
        frame,
        group_col="year",
        probability_col="probability",
        threshold_col="threshold",
    )

    onset_2020 = result.set_index(["year", "issue_state"]).loc[(2020, "no_active_heatwave")]
    onset_2021 = result.set_index(["year", "issue_state"]).loc[(2021, "no_active_heatwave")]
    active_2021 = result.set_index(["year", "issue_state"]).loc[(2021, "active_heatwave")]
    assert onset_2020["average_precision"] == 1.0
    assert onset_2020["events"] == 1
    assert pd.isna(onset_2021["average_precision"])
    assert onset_2021["events"] == 0
    assert active_2021["event_recall"] == 1.0

\n