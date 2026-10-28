"""Metrics for imbalanced heatwave events and validation-derived thresholds."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def _validated_binary_inputs(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError("labels and probabilities must be non-empty, paired one-dimensional arrays")
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be finite binary values")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite values within [0, 1]")
    return y.astype(np.int8), p


def binary_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Calculate event-sensitive discrimination, probability, and error metrics."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    y, p = _validated_binary_inputs(labels, probabilities)
    predicted = (p >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    positives = int(y.sum())
    negatives = int(y.size - positives)
    recall = float(tp / positives) if positives else float("nan")
    false_alarm_rate = float(fp / negatives) if negatives else float("nan")
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    return {
        "n": int(y.size),
        "events": positives,
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)) if positives else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if positives and negatives else float("nan"),
        "brier_score": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "event_recall": recall,
        "false_alarm_rate": false_alarm_rate,
        "precision": precision,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def threshold_at_max_false_alarm_rate(
    validation_labels: Sequence[int] | np.ndarray,
    validation_probabilities: Sequence[float] | np.ndarray,
    max_false_alarm_rate: float = 0.1,
) -> float:
    """Choose a decision threshold using validation data only."""
    y, p = _validated_binary_inputs(validation_labels, validation_probabilities)
    if not np.isfinite(max_false_alarm_rate) or not 0 <= max_false_alarm_rate <= 1:
        raise ValueError("max_false_alarm_rate must be within [0, 1]")
    if np.unique(y).size != 2:
        raise ValueError("threshold selection requires both positive and negative validation examples")
    false_alarm, recall, thresholds = roc_curve(y, p, drop_intermediate=False)
    allowed = np.flatnonzero(false_alarm <= max_false_alarm_rate)
    best_recall = recall[allowed].max()
    best = allowed[recall[allowed] == best_recall]
    threshold = float(np.max(thresholds[best]))
    if np.isinf(threshold):
        # The ROC origin represents an always-negative rule; a threshold just
        # above the largest score preserves it even when a score equals 1.
        threshold = float(np.nextafter(np.max(p), np.inf))
    return threshold


def calibration_bins(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return equal-count calibration bins with observed event rates."""
    y, p = _validated_binary_inputs(labels, probabilities)
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least two")
    rank = pd.Series(p).rank(method="first")
    bins = pd.qcut(rank, q=min(n_bins, len(p)), labels=False, duplicates="drop")
    table = pd.DataFrame({"bin": bins, "probability": p, "event": y}).groupby("bin", observed=True)
    result = table.agg(
        n=("event", "size"),
        mean_probability=("probability", "mean"),
        observed_frequency=("event", "mean"),
    ).reset_index()
    result["absolute_gap"] = (result["mean_probability"] - result["observed_frequency"]).abs()
    return result


def expected_calibration_error(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute weighted absolute calibration error over equal-count bins."""
    bins = calibration_bins(labels, probabilities, n_bins=n_bins)
    return float(np.average(bins["absolute_gap"], weights=bins["n"]))


def grouped_metric_summary(
    frame: pd.DataFrame,
    group_col: str,
    label_col: str,
    probability_col: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Calculate per-group scores so geographic and year failures remain visible."""
    required = {group_col, label_col, probability_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"required metric columns are missing: {sorted(missing)}")
    rows = []
    for group, values in frame.groupby(group_col, dropna=False, sort=True):
        result = binary_metrics(values[label_col], values[probability_col], threshold=threshold)
        rows.append({group_col: group, **result})
    return pd.DataFrame(rows)


def bootstrap_group_metric_means(
    group_metrics: pd.DataFrame,
    metric_columns: Sequence[str],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap equal-weight macro means over already-computed groups.

    The resampling unit is a year or spatial block, not a grid-day. These
    intervals describe uncertainty in the macro-group mean and are not
    confidence intervals for the pooled per-row metric.
    """
    if not isinstance(n_bootstrap, int) or n_bootstrap < 100:
        raise ValueError("n_bootstrap must be an integer of at least 100")
    missing = set(metric_columns) - set(group_metrics.columns)
    if missing:
        raise ValueError(f"group metrics are missing columns: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    rows = []
    for metric in metric_columns:
        values = pd.to_numeric(group_metrics[metric], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            rows.append(
                {"metric": metric, "estimate": np.nan, "lower_95": np.nan, "upper_95": np.nan, "n_groups": 0, "n_bootstrap": n_bootstrap}
            )
            continue
        estimates = np.empty(n_bootstrap, dtype=float)
        for replicate in range(n_bootstrap):
            sample = rng.choice(values, size=values.size, replace=True)
            estimates[replicate] = sample.mean()
        rows.append(
            {
                "metric": metric,
                "estimate": float(values.mean()),
                "lower_95": float(np.quantile(estimates, 0.025)),
                "upper_95": float(np.quantile(estimates, 0.975)),
                "n_groups": int(values.size),
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_paired_group_differences(
    group_metrics: pd.DataFrame,
    group_col: str,
    metric_columns: Sequence[str],
    baseline_model: str = "climatology",
    model_col: str = "model",
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap paired per-group differences against a shared baseline.

    The same spatial block or year is compared for each model and the
    baseline. Positive differences favor a model for higher-is-better metrics
    such as average precision; negative differences favor it for lower-is-
    better metrics such as Brier score.
    """
    if not isinstance(n_bootstrap, int) or n_bootstrap < 100:
        raise ValueError("n_bootstrap must be an integer of at least 100")
    required = {group_col, model_col, *metric_columns}
    missing = required - set(group_metrics.columns)
    if missing:
        raise ValueError(f"group metrics are missing columns: {sorted(missing)}")
    if not metric_columns:
        raise ValueError("metric_columns cannot be empty")
    if group_col == model_col:
        raise ValueError("group_col and model_col must be different")
    if group_metrics[[group_col, model_col]].isna().any().any():
        raise ValueError("group and model identifiers cannot be missing")
    if group_metrics.duplicated([group_col, model_col]).any():
        raise ValueError("each model must have at most one row per group")

    model_names = set(group_metrics[model_col].astype(str))
    if baseline_model not in model_names:
        raise ValueError(f"baseline model {baseline_model!r} is missing")
    baseline = group_metrics.loc[group_metrics[model_col].astype(str) == baseline_model].set_index(group_col)
    rng = np.random.default_rng(seed)
    rows = []
    for model_name in sorted(model_names - {baseline_model}):
        model = group_metrics.loc[group_metrics[model_col].astype(str) == model_name].set_index(group_col)
        paired = model[list(metric_columns)].join(
            baseline[list(metric_columns)],
            how="inner",
            suffixes=("_model", "_baseline"),
        )
        for metric in metric_columns:
            difference = (
                pd.to_numeric(paired[f"{metric}_model"], errors="coerce")
                - pd.to_numeric(paired[f"{metric}_baseline"], errors="coerce")
            ).to_numpy(dtype=float)
            difference = difference[np.isfinite(difference)]
            if difference.size == 0:
                rows.append(
                    {
                        "model": model_name,
                        "baseline_model": baseline_model,
                        "metric": metric,
                        "estimate": np.nan,
                        "lower_95": np.nan,
                        "upper_95": np.nan,
                        "n_groups": 0,
                        "n_bootstrap": n_bootstrap,
                    }
                )
                continue
            estimates = np.empty(n_bootstrap, dtype=float)
            for replicate in range(n_bootstrap):
                sample = rng.choice(difference, size=difference.size, replace=True)
                estimates[replicate] = sample.mean()
            rows.append(
                {
                    "model": model_name,
                    "baseline_model": baseline_model,
                    "metric": metric,
                    "estimate": float(difference.mean()),
                    "lower_95": float(np.quantile(estimates, 0.025)),
                    "upper_95": float(np.quantile(estimates, 0.975)),
                    "n_groups": int(difference.size),
                    "n_bootstrap": n_bootstrap,
                }
            )
    return pd.DataFrame(rows)
