"""State-conditioned Day-1 heatwave transition model and fair comparators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss

from heatwave_benchmark.experiments import (
    select_model_features,
    stratified_training_sample,
)
from heatwave_benchmark.metrics import threshold_at_max_false_alarm_rate
from heatwave_benchmark.models import (
    ClimatologyModel,
    PersistenceModel,
    apply_platt_calibrator,
    fit_platt_calibrator,
    fit_probability_estimators,
)
from heatwave_benchmark.onset import state_conditioned_metrics


@dataclass(slots=True)
class TransitionComparisonResult:
    """Held-out transition scores, predictions, and fit metadata."""

    metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    summary: dict[str, Any]


@dataclass(slots=True)
class OnsetModelResult:
    """Validation-selected onset model results and held-out predictions."""

    metrics: dict[str, float]
    test_predictions: pd.DataFrame
    candidate_scores: pd.DataFrame
    summary: dict[str, Any]
    model: object
    calibrator: Any


class InsufficientTransitionDataError(ValueError):
    """Raised when a state lacks both target classes in train or validation."""


def fit_onset_model_search(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "target",
    state_col: str = "current_heatwave",
    feature_columns: list[str] | None = None,
    parameter_grid: list[dict[str, Any]] | None = None,
    max_false_alarm_rate: float = 0.1,
    random_state: int = 42,
) -> OnsetModelResult:
    """Fit an onset-only HGB search and score state-zero rows.

    Model selection uses validation onset average precision, with validation
    Brier score as a tie breaker. Calibration and the operating threshold are
    also fit on validation onset rows only; test labels are used only for
    final reporting.
    """
    if not np.isfinite(max_false_alarm_rate) or not 0 <= max_false_alarm_rate <= 1:
        raise ValueError("max_false_alarm_rate must be within [0, 1]")
    train = _validated_transition_partition(training, "training", target_col, state_col, False)
    valid = _validated_transition_partition(validation, "validation", target_col, state_col, False)
    held_out = _validated_transition_partition(test, "test", target_col, state_col, False)
    train = train.loc[train[state_col].eq(0)].copy()
    valid = valid.loc[valid[state_col].eq(0)].copy()
    held_out = held_out.loc[held_out[state_col].eq(0)].copy()
    for name, frame in (("training", train), ("validation", valid)):
        if frame.empty or frame[target_col].nunique() != 2:
            raise ValueError(f"{name} onset subset must contain both target classes")
    if held_out.empty:
        raise ValueError("test onset subset must contain at least one row")

    features = feature_columns if feature_columns is not None else select_model_features(train)
    if not features or state_col in features:
        raise ValueError("onset features must be non-empty and exclude the issue-day state")
    missing = set(features) - set(train.columns) | set(features) - set(valid.columns) | set(features) - set(held_out.columns)
    if missing:
        raise ValueError(f"partitions are missing onset features: {sorted(missing)}")
    for name, frame in (("training", train), ("validation", valid), ("test", held_out)):
        bad = [column for column in features if not pd.api.types.is_numeric_dtype(frame[column])]
        if bad:
            raise ValueError(f"{name} onset features must be numeric: {bad}")

    grid = parameter_grid or [
        {"max_leaf_nodes": leaves, "min_samples_leaf": leaf, "l2_regularization": l2}
        for leaves in (15, 31)
        for leaf in (50, 200)
        for l2 in (1.0, 10.0)
    ]
    if not grid:
        raise ValueError("parameter_grid must contain at least one candidate")
    required_params = {"max_leaf_nodes", "min_samples_leaf", "l2_regularization"}
    if any(set(params) != required_params for params in grid):
        raise ValueError(f"each parameter-grid entry must contain exactly {sorted(required_params)}")

    X_train = train[features].to_numpy(dtype=float)
    y_train = train[target_col].to_numpy(dtype=np.int8)
    X_valid = valid[features].to_numpy(dtype=float)
    y_valid = valid[target_col].to_numpy(dtype=np.int8)
    scores: list[dict[str, Any]] = []
    models: list[HistGradientBoostingClassifier] = []
    for index, params in enumerate(grid):
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=150,
            early_stopping=False,
            random_state=random_state + index,
            **params,
        )
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_valid)[:, 1]
        scores.append(
            {
                "candidate": index,
                **params,
                "validation_average_precision": float(average_precision_score(y_valid, probabilities)),
                "validation_brier_score": float(brier_score_loss(y_valid, probabilities)),
            }
        )
        models.append(model)
    selected_index = max(
        range(len(scores)),
        key=lambda index: (scores[index]["validation_average_precision"], -scores[index]["validation_brier_score"]),
    )
    selected = models[selected_index]
    valid_raw = selected.predict_proba(X_valid)[:, 1]
    test_raw = selected.predict_proba(held_out[features].to_numpy(dtype=float))[:, 1]
    calibrator = fit_platt_calibrator(y_valid, valid_raw)
    valid_calibrated = apply_platt_calibrator(calibrator, valid_raw)
    test_calibrated = apply_platt_calibrator(calibrator, test_raw)
    threshold = float(threshold_at_max_false_alarm_rate(y_valid, valid_calibrated, max_false_alarm_rate))
    y_test = held_out[target_col].to_numpy(dtype=np.int8)
    predicted = test_calibrated >= threshold
    positives = y_test == 1
    negatives = ~positives
    metrics = {
        "n": int(len(y_test)),
        "events": int(positives.sum()),
        "prevalence": float(y_test.mean()),
        "average_precision": float(average_precision_score(y_test, test_calibrated)) if positives.any() else float("nan"),
        "brier_score": float(brier_score_loss(y_test, test_calibrated)),
        "event_recall": float((predicted & positives).sum() / positives.sum()) if positives.any() else float("nan"),
        "false_alarm_rate": float((predicted & negatives).sum() / negatives.sum()) if negatives.any() else float("nan"),
        "precision": float((predicted & positives).sum() / predicted.sum()) if predicted.any() else 0.0,
        "validation_threshold": threshold,
        "validation_average_precision": float(scores[selected_index]["validation_average_precision"]),
        "validation_brier_score_raw": float(scores[selected_index]["validation_brier_score"]),
    }
    metadata = [column for column in ("date", "grid_id", "spatial_block", target_col, state_col) if column in held_out]
    predictions = held_out[metadata].reset_index(drop=True).copy()
    predictions["onset_raw_probability"] = test_raw
    predictions["onset_probability"] = test_calibrated
    predictions["onset_threshold"] = threshold
    summary = {
        "primary_selection_metric": "validation average precision among issue-day state-zero rows",
        "tie_breaker": "lower raw validation Brier score",
        "selected_parameters": grid[selected_index],
        "selected_candidate": selected_index,
        "training_rows_used": int(len(train)),
        "validation_rows_used": int(len(valid)),
        "test_rows_scored": int(len(held_out)),
        "training_onset_prevalence": float(y_train.mean()),
        "validation_onset_prevalence": float(y_valid.mean()),
        "max_validation_false_alarm_rate": float(max_false_alarm_rate),
        "threshold_fit_rows": int(len(valid)),
        "features": list(features),
    }
    return OnsetModelResult(metrics, predictions, pd.DataFrame(scores), summary, selected, calibrator)


def _validated_transition_partition(
    frame: pd.DataFrame,
    partition_name: str,
    target_col: str,
    state_col: str,
    require_both_target_classes_by_state: bool,
) -> pd.DataFrame:
    required = {target_col, state_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{partition_name} partition is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{partition_name} partition must be non-empty")

    result = frame.copy()
    values = result[[target_col, state_col]].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(
            f"{partition_name} must contain valid target and binary issue-day state values"
        )
    target = values[target_col].to_numpy(dtype=float)
    state = values[state_col].to_numpy(dtype=float)
    if not np.isfinite(target).all() or not np.isin(target, (0, 1)).all():
        raise ValueError(f"{partition_name} targets must be finite binary values")
    if not np.isfinite(state).all() or not np.isin(state, (0, 1)).all():
        raise ValueError(
            f"{partition_name} must contain valid target and binary issue-day state values"
        )
    result[target_col] = target.astype(np.int8)
    result[state_col] = state.astype(np.int8)

    if require_both_target_classes_by_state:
        for issue_state in (0, 1):
            state_targets = result.loc[result[state_col].eq(issue_state), target_col]
            if state_targets.empty or state_targets.nunique() != 2:
                raise InsufficientTransitionDataError(
                    f"{partition_name} must contain both target classes for issue-day state {issue_state}"
                )
    return result


def _state_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    states: np.ndarray,
    max_false_alarm_rate: float,
) -> dict[int, float]:
    return {
        issue_state: threshold_at_max_false_alarm_rate(
            labels[states == issue_state],
            probabilities[states == issue_state],
            max_false_alarm_rate=max_false_alarm_rate,
        )
        for issue_state in (0, 1)
    }


def _threshold_vector(states: np.ndarray, thresholds: dict[int, float]) -> np.ndarray:
    return np.asarray([thresholds[int(state)] for state in states], dtype=float)


def run_state_transition_comparison(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "target",
    state_col: str = "current_heatwave",
    feature_columns: list[str] | None = None,
    max_training_rows: int = 200_000,
    max_false_alarm_rate: float = 0.1,
    random_state: int = 42,
) -> TransitionComparisonResult:
    """Fit separate onset/continuation models and score paired baselines.

    The caller must pass only rows with known binary targets and issue-day
    states. The state is used to route each record to its branch and is never
    included among the numeric weather model features. Calibration and the
    per-state operating thresholds use validation data only.
    """
    if not isinstance(max_training_rows, int) or max_training_rows < 4:
        raise ValueError("max_training_rows must be an integer of at least four")
    if not np.isfinite(max_false_alarm_rate) or not 0 <= max_false_alarm_rate <= 1:
        raise ValueError("max_false_alarm_rate must be within [0, 1]")

    train = _validated_transition_partition(
        training, "training", target_col, state_col, require_both_target_classes_by_state=True
    )
    valid = _validated_transition_partition(
        validation, "validation", target_col, state_col, require_both_target_classes_by_state=True
    )
    held_out = _validated_transition_partition(
        test, "test", target_col, state_col, require_both_target_classes_by_state=False
    )

    features = feature_columns if feature_columns is not None else select_model_features(train)
    if state_col in features:
        raise ValueError(f"{state_col!r} is a routing state and cannot be a transition-model feature")
    missing_features = (
        set(features) - set(train.columns)
        | set(features) - set(valid.columns)
        | set(features) - set(held_out.columns)
    )
    if missing_features:
        raise ValueError(f"partitions are missing model features: {sorted(missing_features)}")
    for partition_name, frame in (("training", train), ("validation", valid), ("test", held_out)):
        nonnumeric = [name for name in features if not pd.api.types.is_numeric_dtype(frame[name])]
        if nonnumeric:
            raise ValueError(f"{partition_name} model features must be numeric: {nonnumeric}")

    # The pooled reference and both conditional branches share one total row cap.
    pooled_sample = stratified_training_sample(
        train,
        target_col,
        maximum_rows=max_training_rows,
        random_state=random_state,
    )
    pooled_model = fit_probability_estimators(
        pooled_sample,
        feature_columns=features,
        target_col=target_col,
        random_state=random_state,
        estimator_names=("hist_gradient_boosting",),
    )["hist_gradient_boosting"]
    pooled_state_features = [*features, state_col]
    pooled_state_model = fit_probability_estimators(
        pooled_sample,
        feature_columns=pooled_state_features,
        target_col=target_col,
        random_state=random_state,
        estimator_names=("hist_gradient_boosting",),
    )["hist_gradient_boosting"]

    branch_budget = max_training_rows // 2
    state_models: dict[int, object] = {}
    state_fit_rows: dict[int, int] = {}
    for issue_state in (0, 1):
        state_training = train.loc[train[state_col].eq(issue_state)].copy()
        state_sample = stratified_training_sample(
            state_training,
            target_col,
            maximum_rows=branch_budget,
            random_state=random_state + issue_state + 1,
        )
        state_models[issue_state] = fit_probability_estimators(
            state_sample,
            feature_columns=features,
            target_col=target_col,
            random_state=random_state + issue_state + 1,
            estimator_names=("hist_gradient_boosting",),
        )["hist_gradient_boosting"]
        state_fit_rows[issue_state] = int(len(state_sample))

    climatology = ClimatologyModel().fit(train, target_col)
    persistence = PersistenceModel().fit(train, target_col)

    validation_labels = valid[target_col].to_numpy(dtype=np.int8)
    validation_states = valid[state_col].to_numpy(dtype=np.int8)
    test_labels = held_out[target_col].to_numpy(dtype=np.int8)
    test_states = held_out[state_col].to_numpy(dtype=np.int8)

    validation_raw: dict[str, np.ndarray] = {
        "pooled_hist_gradient_boosting": pooled_model.predict_proba(
            valid[features].to_numpy(dtype=float)
        )[:, 1],
        "pooled_hist_gradient_boosting_with_issue_state": pooled_state_model.predict_proba(
            valid[pooled_state_features].to_numpy(dtype=float)
        )[:, 1],
        "climatology": climatology.predict_proba(valid),
        "persistence": persistence.predict_proba(valid),
    }
    test_raw: dict[str, np.ndarray] = {
        "pooled_hist_gradient_boosting": pooled_model.predict_proba(
            held_out[features].to_numpy(dtype=float)
        )[:, 1],
        "pooled_hist_gradient_boosting_with_issue_state": pooled_state_model.predict_proba(
            held_out[pooled_state_features].to_numpy(dtype=float)
        )[:, 1],
        "climatology": climatology.predict_proba(held_out),
        "persistence": persistence.predict_proba(held_out),
    }

    transition_validation_raw = np.full(len(valid), np.nan, dtype=float)
    transition_test_raw = np.full(len(held_out), np.nan, dtype=float)
    for issue_state in (0, 1):
        validation_positions = np.flatnonzero(validation_states == issue_state)
        test_positions = np.flatnonzero(test_states == issue_state)
        transition_validation_raw[validation_positions] = state_models[issue_state].predict_proba(
            valid.iloc[validation_positions][features].to_numpy(dtype=float)
        )[:, 1]
        if len(test_positions):
            transition_test_raw[test_positions] = state_models[issue_state].predict_proba(
                held_out.iloc[test_positions][features].to_numpy(dtype=float)
            )[:, 1]
    if not np.isfinite(transition_validation_raw).all() or not np.isfinite(transition_test_raw).all():
        raise RuntimeError("state-routed predictions are incomplete or non-finite")
    validation_raw["state_transition_hist_gradient_boosting"] = transition_validation_raw
    test_raw["state_transition_hist_gradient_boosting"] = transition_test_raw

    calibrated_validation: dict[str, np.ndarray] = {}
    calibrated_test: dict[str, np.ndarray] = {}
    calibrators: dict[str, Any] = {}
    for model_name in validation_raw:
        if model_name in {
            "state_transition_hist_gradient_boosting",
            "pooled_hist_gradient_boosting_with_issue_state",
        }:
            state_calibrators = {}
            validation_calibrated = np.full(len(valid), np.nan, dtype=float)
            test_calibrated = np.full(len(held_out), np.nan, dtype=float)
            for issue_state in (0, 1):
                validation_mask = validation_states == issue_state
                test_mask = test_states == issue_state
                calibrator = fit_platt_calibrator(
                    validation_labels[validation_mask],
                    validation_raw[model_name][validation_mask],
                )
                state_calibrators[issue_state] = calibrator
                validation_calibrated[validation_mask] = apply_platt_calibrator(
                    calibrator, validation_raw[model_name][validation_mask]
                )
                test_calibrated[test_mask] = apply_platt_calibrator(
                    calibrator, test_raw[model_name][test_mask]
                )
            calibrators[model_name] = state_calibrators
        else:
            calibrator = fit_platt_calibrator(validation_labels, validation_raw[model_name])
            calibrators[model_name] = calibrator
            validation_calibrated = apply_platt_calibrator(calibrator, validation_raw[model_name])
            test_calibrated = apply_platt_calibrator(calibrator, test_raw[model_name])
        calibrated_validation[model_name] = validation_calibrated
        calibrated_test[model_name] = test_calibrated

    raw_thresholds = {
        model_name: _state_thresholds(
            validation_labels,
            probabilities,
            validation_states,
            max_false_alarm_rate,
        )
        for model_name, probabilities in validation_raw.items()
    }
    calibrated_thresholds = {
        model_name: _state_thresholds(
            validation_labels,
            probabilities,
            validation_states,
            max_false_alarm_rate,
        )
        for model_name, probabilities in calibrated_validation.items()
    }

    metadata_columns = [
        column
        for column in ("date", "grid_id", "spatial_block", target_col, state_col)
        if column in held_out
    ]
    predictions = held_out.loc[:, metadata_columns].reset_index(drop=True).copy()
    if target_col != "target":
        predictions = predictions.rename(columns={target_col: "target"})
    if state_col != "current_heatwave":
        predictions = predictions.rename(columns={state_col: "current_heatwave"})

    metrics_rows: list[dict[str, Any]] = []
    for model_name, raw_test_probability in test_raw.items():
        raw_threshold = _threshold_vector(test_states, raw_thresholds[model_name])
        calibrated_threshold = _threshold_vector(test_states, calibrated_thresholds[model_name])
        calibrated_test_probability = calibrated_test[model_name]
        predictions[f"{model_name}_raw_probability"] = raw_test_probability
        predictions[f"{model_name}_calibrated_probability"] = calibrated_test_probability
        predictions[f"{model_name}_threshold_raw"] = raw_threshold
        predictions[f"{model_name}_threshold_calibrated"] = calibrated_threshold
        if model_name == "state_transition_hist_gradient_boosting":
            predictions["transition_raw_probability"] = raw_test_probability
            predictions["transition_calibrated_probability"] = calibrated_test_probability
            predictions["transition_threshold_calibrated"] = calibrated_threshold

        for probability_type, values, thresholds in (
            ("raw", raw_test_probability, raw_threshold),
            ("platt_calibrated", calibrated_test_probability, calibrated_threshold),
        ):
            state_metrics = state_conditioned_metrics(
                pd.DataFrame(
                    {
                        "target": test_labels,
                        "current_heatwave": test_states,
                        "probability": values,
                        "threshold": thresholds,
                    }
                ),
                probability_col="probability",
                threshold_col="threshold",
            )
            for row in state_metrics.to_dict(orient="records"):
                metrics_rows.append(
                    {
                        "model": model_name,
                        "probability_type": probability_type,
                        "max_validation_false_alarm_rate": float(max_false_alarm_rate),
                        **row,
                    }
                )

    metrics = pd.DataFrame(metrics_rows)
    summary = {
        "target": target_col,
        "state": state_col,
        "positive_transition_definition": "current_heatwave=0 and next-day target=1",
        "continuation_definition": "current_heatwave=1 and next-day target=1",
        "model": "two separate histogram gradient boosting classifiers routed by issue-day state",
        "calibration_policy": "pooled Platt calibration for pooled HGB without issue state, climatology, and persistence; separate validation-fitted Platt calibrators by state for the pooled HGB state-feature control and transition branches",
        "calibration_policy_by_model": {
            "pooled_hist_gradient_boosting": "global validation-fitted Platt calibration",
            "pooled_hist_gradient_boosting_with_issue_state": "state-specific Platt calibration on validation rows",
            "state_transition_hist_gradient_boosting": "state-specific Platt calibration on validation rows",
            "climatology": "global validation-fitted Platt calibration",
            "persistence": "global validation-fitted Platt calibration",
        },
        "threshold_policy": "state-specific validation thresholds selected to the configured maximum false-alarm rate for every model",
        "max_validation_false_alarm_rate": float(max_false_alarm_rate),
        "max_training_rows_total": int(max_training_rows),
        "pooled_training_rows_fit": int(len(pooled_sample)),
        "transition_training_rows_fit_by_state": state_fit_rows,
        "training_rows_by_state": {
            issue_state: int(train[state_col].eq(issue_state).sum()) for issue_state in (0, 1)
        },
        "validation_rows_by_state": {
            issue_state: int(valid[state_col].eq(issue_state).sum()) for issue_state in (0, 1)
        },
        "test_rows_by_state": {
            issue_state: int(held_out[state_col].eq(issue_state).sum()) for issue_state in (0, 1)
        },
        "thresholds_raw_by_state": raw_thresholds,
        "thresholds_calibrated_by_state": calibrated_thresholds[
            "state_transition_hist_gradient_boosting"
        ],
        "transition_thresholds_calibrated_by_state": calibrated_thresholds[
            "state_transition_hist_gradient_boosting"
        ],
        "all_model_thresholds_calibrated_by_state": calibrated_thresholds,
        "feature_columns": list(features),
    }
    return TransitionComparisonResult(metrics, predictions, summary)
