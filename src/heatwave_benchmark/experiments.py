"""Leakage-aware model fitting, validation calibration, and final evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from heatwave_benchmark.metrics import binary_metrics, threshold_at_max_false_alarm_rate
from heatwave_benchmark.models import (
    ClimatologyModel,
    PersistenceModel,
    apply_platt_calibrator,
    fit_platt_calibrator,
    fit_probability_estimators,
    predict_probability_estimators,
)


_STATIC_FEATURES = {"latitude", "longitude", "terrain_code", "doy_sin", "doy_cos"}


@dataclass(slots=True)
class ModelComparisonResult:
    metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    summary: dict[str, Any]


def select_model_features(frame: pd.DataFrame) -> list[str]:
    """Select only numeric weather lags and fixed geographic/calendar context."""
    columns = [
        column
        for column in frame.columns
        if "_lag_" in column or column in _STATIC_FEATURES
    ]
    if not columns:
        raise ValueError("no model features found in frame")
    nonnumeric = [column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])]
    if nonnumeric:
        raise ValueError(f"model features must be numeric: {nonnumeric}")
    return columns


def stratified_training_sample(
    frame: pd.DataFrame,
    target_col: str,
    maximum_rows: int = 200_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a deterministic, class-stratified training subset if needed."""
    if maximum_rows < 1:
        raise ValueError("maximum_rows must be positive")
    if target_col not in frame:
        raise ValueError(f"target column {target_col!r} is missing")
    labels = frame[target_col].to_numpy()
    if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("training targets must be finite binary labels")
    if len(frame) <= maximum_rows:
        return frame.copy()
    if np.unique(labels).size != 2:
        raise ValueError("stratified sampling requires both target classes")
    sample, _ = train_test_split(
        frame,
        train_size=maximum_rows,
        stratify=labels,
        random_state=random_state,
    )
    return sample.sort_index(kind="stable").copy()


def _validate_partitions(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    for name, frame in (("training", training), ("validation", validation), ("test", test)):
        if target_col not in frame:
            raise ValueError(f"{name} partition is missing target {target_col!r}")
        y = frame[target_col].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isin(y, (0, 1))
        if not valid.all():
            raise ValueError(f"{name} partition must contain only known binary targets")
        if name != "test" and np.unique(y).size != 2:
            raise ValueError(f"{name} partition must contain both target classes")
        labels.append(y.astype(np.int8))
    if training.empty or validation.empty or test.empty:
        raise ValueError("training, validation, and test partitions must be non-empty")
    return labels[0], labels[1], labels[2]


def run_model_comparison(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "target",
    feature_columns: list[str] | None = None,
    max_training_rows: int = 200_000,
    max_false_alarm_rate: float = 0.1,
    random_state: int = 42,
    include_block_balanced_boosting: bool = False,
    estimator_names: tuple[str, ...] | None = None,
) -> ModelComparisonResult:
    """Fit models on train, calibrate/select thresholds on validation, test once."""
    y_train, y_validation, y_test = _validate_partitions(training, validation, test, target_col)
    features = feature_columns if feature_columns is not None else select_model_features(training)
    missing_features = set(features) - set(training.columns) | set(features) - set(validation.columns) | set(features) - set(test.columns)
    if missing_features:
        raise ValueError(f"partitions are missing model features: {sorted(missing_features)}")

    model_frame = stratified_training_sample(training, target_col, max_training_rows, random_state)
    feature_models = fit_probability_estimators(
        model_frame,
        feature_columns=features,
        target_col=target_col,
        random_state=random_state,
        estimator_names=estimator_names,
    )
    validation_probability = predict_probability_estimators(feature_models, validation, features)
    test_probability = predict_probability_estimators(feature_models, test, features)

    if include_block_balanced_boosting:
        if "hist_gradient_boosting" not in feature_models:
            raise ValueError("block-balanced boosting requires hist_gradient_boosting in estimator_names")
        block_models = fit_probability_estimators(
            model_frame,
            feature_columns=features,
            target_col=target_col,
            balance_spatial_blocks=True,
            random_state=random_state,
            estimator_names=("hist_gradient_boosting",),
        )
        validation_probability["hist_gradient_boosting_block_balanced"] = block_models[
            "hist_gradient_boosting"
        ].predict_proba(validation[features].to_numpy(dtype=float))[:, 1]
        test_probability["hist_gradient_boosting_block_balanced"] = block_models[
            "hist_gradient_boosting"
        ].predict_proba(test[features].to_numpy(dtype=float))[:, 1]

    climatology = ClimatologyModel().fit(training, target_col)
    persistence = PersistenceModel().fit(training, target_col)
    for name, model in (("climatology", climatology), ("persistence", persistence)):
        validation_probability[name] = model.predict_proba(validation)
        test_probability[name] = model.predict_proba(test)

    calibrators = {}
    validation_calibrated = {}
    test_calibrated = {}
    thresholds_raw = {}
    thresholds_calibrated = {}
    for name in validation_probability:
        raw_val = validation_probability[name]
        raw_test = test_probability[name]
        calibrator = fit_platt_calibrator(y_validation, raw_val)
        calibrated_val = apply_platt_calibrator(calibrator, raw_val)
        calibrated_test = apply_platt_calibrator(calibrator, raw_test)
        validation_calibrated[name] = calibrated_val
        test_calibrated[name] = calibrated_test
        calibrators[name] = calibrator
        thresholds_raw[name] = threshold_at_max_false_alarm_rate(
            y_validation, raw_val, max_false_alarm_rate=max_false_alarm_rate
        )
        thresholds_calibrated[name] = threshold_at_max_false_alarm_rate(
            y_validation, calibrated_val, max_false_alarm_rate=max_false_alarm_rate
        )

    metric_rows: list[dict[str, Any]] = []
    prediction_columns: dict[str, np.ndarray] = {}
    for name in validation_probability:
        for partition, labels, probabilities, calibrated, threshold_raw, threshold_cal in (
            (
                "validation",
                y_validation,
                validation_probability[name],
                validation_calibrated[name],
                thresholds_raw[name],
                thresholds_calibrated[name],
            ),
            (
                "test",
                y_test,
                test_probability[name],
                test_calibrated[name],
                thresholds_raw[name],
                thresholds_calibrated[name],
            ),
        ):
            for probability_type, values, threshold in (
                ("raw", probabilities, threshold_raw),
                ("platt_calibrated", calibrated, threshold_cal),
            ):
                metric_rows.append(
                    {
                        "model": name,
                        "partition": partition,
                        "probability_type": probability_type,
                        "max_validation_false_alarm_rate": max_false_alarm_rate,
                        **binary_metrics(labels, values, threshold=threshold),
                    }
                )
        prediction_columns[f"{name}_raw_probability"] = test_probability[name]
        prediction_columns[f"{name}_calibrated_probability"] = test_calibrated[name]
        prediction_columns[f"{name}_threshold_raw"] = np.full(len(test), thresholds_raw[name])
        prediction_columns[f"{name}_threshold_calibrated"] = np.full(len(test), thresholds_calibrated[name])

    prediction_keys = [column for column in ("date", "grid_id", "spatial_block", target_col) if column in test]
    predictions = test.loc[:, prediction_keys].reset_index(drop=True).copy()
    for name, values in prediction_columns.items():
        predictions[name] = values
    metrics = pd.DataFrame(metric_rows)
    raw_val_ap = metrics.loc[
        (metrics.partition == "validation") & (metrics.probability_type == "raw")
    ].set_index("model")["average_precision"]
    best_model = str(raw_val_ap.idxmax())
    summary = {
        "best_validation_model": best_model,
        "best_validation_average_precision": float(raw_val_ap.loc[best_model]),
        "n_training_rows": int(len(training)),
        "n_training_rows_fit": int(len(model_frame)),
        "n_validation_rows": int(len(validation)),
        "n_test_rows": int(len(test)),
        "feature_columns": list(features),
        "max_validation_false_alarm_rate": float(max_false_alarm_rate),
        "thresholds_raw": thresholds_raw,
        "thresholds_calibrated": thresholds_calibrated,
    }
    return ModelComparisonResult(metrics=metrics, test_predictions=predictions, summary=summary)
