"""State-conditioned diagnostics for new versus ongoing heatwave events."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def state_conditioned_metrics(
    frame: pd.DataFrame,
    probability_col: str,
    threshold_col: str,
    label_col: str = "target",
    current_event_col: str = "current_heatwave",
) -> pd.DataFrame:
    """Score all cases, event onsets, and ongoing events with carried thresholds.

    ``current_heatwave`` is used only to partition the evaluation records. Each
    row's threshold must have been selected on validation data before this
    function is called; thresholds are not retuned within state subsets.
    """
    required = {label_col, current_event_col, probability_col, threshold_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"state-conditioned metric columns are missing: {sorted(missing)}")
    if frame.empty:
        raise ValueError("state-conditioned metrics require at least one row")

    values = frame[[label_col, current_event_col, probability_col, threshold_col]].apply(
        pd.to_numeric, errors="raise"
    )
    if values.isna().any().any():
        raise ValueError("labels, issue-day states, probabilities, and thresholds cannot be missing")
    y = values[label_col].to_numpy(dtype=float)
    current = values[current_event_col].to_numpy(dtype=float)
    probabilities = values[probability_col].to_numpy(dtype=float)
    thresholds = values[threshold_col].to_numpy(dtype=float)
    if not np.isin(y, (0, 1)).all() or not np.isin(current, (0, 1)).all():
        raise ValueError("target and current_heatwave values must be binary")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite values within [0, 1]")
    if not np.isfinite(thresholds).all() or np.any(thresholds < 0):
        raise ValueError("validation thresholds must be finite and non-negative")

    conditions = (
        ("all", np.ones(len(frame), dtype=bool)),
        ("no_active_heatwave", current == 0),
        ("active_heatwave", current == 1),
    )
    rows: list[dict[str, float | int | str]] = []
    for issue_state, mask in conditions:
        if not mask.any():
            continue
        state_y = y[mask].astype(np.int8)
        state_probabilities = probabilities[mask]
        state_thresholds = thresholds[mask]
        predicted = state_probabilities >= state_thresholds
        true_positive = int(np.sum(predicted & (state_y == 1)))
        false_positive = int(np.sum(predicted & (state_y == 0)))
        false_negative = int(np.sum(~predicted & (state_y == 1)))
        true_negative = int(np.sum(~predicted & (state_y == 0)))
        positives = int(state_y.sum())
        negatives = int(len(state_y) - positives)
        rows.append(
            {
                "issue_state": issue_state,
                "n": int(len(state_y)),
                "events": positives,
                "prevalence": float(state_y.mean()),
                "average_precision": (
                    float(average_precision_score(state_y, state_probabilities))
                    if positives
                    else float("nan")
                ),
                "roc_auc": (
                    float(roc_auc_score(state_y, state_probabilities))
                    if positives and negatives
                    else float("nan")
                ),
                "brier_score": float(brier_score_loss(state_y, state_probabilities)),
                "mean_validation_threshold": float(state_thresholds.mean()),
                "event_recall": float(true_positive / positives) if positives else float("nan"),
                "false_alarm_rate": float(false_positive / negatives) if negatives else float("nan"),
                "precision": (
                    float(true_positive / (true_positive + false_positive))
                    if true_positive + false_positive
                    else 0.0
                ),
                "true_positives": true_positive,
                "false_positives": false_positive,
                "false_negatives": false_negative,
                "true_negatives": true_negative,
            }
        )
    return pd.DataFrame(rows)


def state_conditioned_group_metrics(
    frame: pd.DataFrame,
    group_col: str,
    probability_col: str,
    threshold_col: str,
    label_col: str = "target",
    current_event_col: str = "current_heatwave",
) -> pd.DataFrame:
    """Return state-conditioned metrics separately for each resampling group.

    The resulting rows can be passed to ``bootstrap_group_metric_means`` so
    uncertainty is resampled over years or spatial blocks rather than over
    dependent grid-day records.
    """
    if group_col not in frame.columns:
        raise ValueError(f"group column {group_col!r} is missing")
    if frame.empty:
        raise ValueError("grouped state-conditioned metrics require at least one row")
    if frame[group_col].isna().any():
        raise ValueError("resampling group identifiers cannot be missing")

    rows: list[dict[str, object]] = []
    for group, values in frame.groupby(group_col, sort=True):
        metrics = state_conditioned_metrics(
            values,
            probability_col=probability_col,
            threshold_col=threshold_col,
            label_col=label_col,
            current_event_col=current_event_col,
        )
        for record in metrics.to_dict(orient="records"):
            rows.append({group_col: group, **record})
    return pd.DataFrame(rows)
