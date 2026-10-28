"""Transparent baselines and supervised estimators for the benchmark."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from importlib.util import find_spec
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PersistenceModel:
    """Use an observed issue-day heatwave as the next-day risk estimate."""

    def fit(self, frame: pd.DataFrame, target_col: str = "target") -> "PersistenceModel":
        if target_col not in frame or "current_heatwave" not in frame:
            raise ValueError("persistence fitting requires target and current_heatwave columns")
        target = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(target)
        if not valid.any():
            raise ValueError("persistence model has no observed training targets")
        self.prevalence_ = float(target[valid].mean())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "prevalence_"):
            raise RuntimeError("persistence model must be fit first")
        if "current_heatwave" not in frame:
            raise ValueError("persistence predictions require current_heatwave")
        current = pd.to_numeric(frame["current_heatwave"], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(current), np.clip(current, 0.0, 1.0), self.prevalence_)


class ClimatologyModel:
    """Estimate day-of-year event risk from training years only.

    Grid-specific 15-day climatology is shrunk toward a pooled 15-day rate;
    unseen grid cells use the pooled rate for the same day-of-year bin.
    """

    def __init__(self, shrinkage: float = 20.0, bin_days: int = 15):
        if shrinkage < 0 or not np.isfinite(shrinkage):
            raise ValueError("shrinkage must be finite and non-negative")
        if not isinstance(bin_days, int) or bin_days < 1:
            raise ValueError("bin_days must be a positive integer")
        self.shrinkage = float(shrinkage)
        self.bin_days = bin_days

    def fit(self, frame: pd.DataFrame, target_col: str = "target") -> "ClimatologyModel":
        required = {target_col, "grid_id", "date"}
        if required - set(frame.columns):
            raise ValueError(f"climatology fitting requires {sorted(required)}")
        y = pd.to_numeric(frame[target_col], errors="coerce")
        dates = pd.to_datetime(frame["date"], errors="raise")
        valid = y.notna()
        if not valid.any():
            raise ValueError("climatology model has no observed training targets")
        training = pd.DataFrame(
            {
                "grid_id": frame.loc[valid, "grid_id"].astype(str).to_numpy(),
                "doy_bin": ((dates.loc[valid].dt.day_of_year.to_numpy() - 1) // self.bin_days),
                "target": y.loc[valid].to_numpy(dtype=float),
            }
        )
        self.prevalence_ = float(training["target"].mean())
        pooled = training.groupby("doy_bin")["target"].agg(["sum", "count"])
        self.pooled_ = {
            int(key): float((row["sum"] + self.shrinkage * self.prevalence_) / (row["count"] + self.shrinkage))
            for key, row in pooled.iterrows()
        }
        grouped = training.groupby(["grid_id", "doy_bin"])["target"].agg(["sum", "count"])
        self.cell_rates_ = {
            (str(grid_id), int(doy_bin)): float(
                (row["sum"] + self.shrinkage * self.pooled_.get(int(doy_bin), self.prevalence_))
                / (row["count"] + self.shrinkage)
            )
            for (grid_id, doy_bin), row in grouped.iterrows()
        }
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "cell_rates_"):
            raise RuntimeError("climatology model must be fit first")
        if not {"grid_id", "date"}.issubset(frame.columns):
            raise ValueError("climatology predictions require grid_id and date")
        dates = pd.to_datetime(frame["date"], errors="raise")
        bins = (dates.dt.day_of_year.to_numpy() - 1) // self.bin_days
        probabilities = [
            self.cell_rates_.get(
                (str(grid_id), int(doy_bin)),
                self.pooled_.get(int(doy_bin), self.prevalence_),
            )
            for grid_id, doy_bin in zip(frame["grid_id"], bins, strict=True)
        ]
        return np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)


def make_training_weights(
    labels: Sequence[int] | np.ndarray,
    spatial_blocks: Sequence[str] | np.ndarray | None = None,
    balance_classes: bool = True,
    balance_spatial_blocks: bool = False,
) -> np.ndarray:
    """Return normalized sample weights calculated from training labels only."""
    y = np.asarray(labels, dtype=float)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be a non-empty, finite one-dimensional binary array")
    weights = np.ones(y.size, dtype=float)
    classes, counts = np.unique(y, return_counts=True)
    if balance_classes and classes.size > 1:
        class_weights = {value: y.size / (classes.size * count) for value, count in zip(classes, counts, strict=True)}
        weights *= np.asarray([class_weights[value] for value in y])
    if balance_spatial_blocks:
        if spatial_blocks is None:
            raise ValueError("spatial_blocks are required when block balancing is enabled")
        blocks = np.asarray(spatial_blocks, dtype=str)
        if blocks.shape != y.shape or np.any(blocks == ""):
            raise ValueError("spatial_blocks must be a non-empty ID for every training label")
        block_mass = {block: weights[blocks == block].sum() for block in np.unique(blocks)}
        target_mass = float(np.mean(list(block_mass.values())))
        for block, mass in block_mass.items():
            weights[blocks == block] *= target_mass / mass
    weights /= weights.mean()
    return weights


def build_estimators(random_state: int = 42, include_xgboost: bool = True) -> dict[str, object]:
    """Construct fixed, interpretable sklearn estimators for the benchmark."""
    estimators: dict[str, object] = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=120,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=150,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=random_state,
        ),
    }
    if include_xgboost and find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        estimators["xgboost"] = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=4,
            random_state=random_state,
        )
    return estimators


def fit_probability_estimators(
    training: pd.DataFrame,
    feature_columns: Sequence[str],
    target_col: str = "target",
    balance_spatial_blocks: bool = False,
    random_state: int = 42,
    estimator_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Fit every supervised estimator on training rows and weights only."""
    X = training.loc[:, feature_columns].to_numpy(dtype=float)
    y = training[target_col].to_numpy(dtype=int)
    weights = make_training_weights(
        y,
        training["spatial_block"].to_numpy() if balance_spatial_blocks else None,
        balance_classes=True,
        balance_spatial_blocks=balance_spatial_blocks,
    )
    fitted = build_estimators(random_state=random_state)
    if estimator_names is not None:
        unknown = set(estimator_names) - set(fitted)
        if unknown:
            raise ValueError(f"unknown estimators requested: {sorted(unknown)}")
        fitted = {name: fitted[name] for name in estimator_names}
    for name, model in fitted.items():
        if name in {"hist_gradient_boosting", "xgboost"}:
            model.fit(X, y, sample_weight=weights)
        else:
            model.fit(X, y, classifier__sample_weight=weights)
    return fitted


def predict_probability_estimators(
    estimators: dict[str, object],
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return positive-class probabilities from fitted estimators."""
    X = frame.loc[:, feature_columns].to_numpy(dtype=float)
    return {name: model.predict_proba(X)[:, 1] for name, model in estimators.items()}


def fit_platt_calibrator(validation_labels: Sequence[int], validation_probabilities: Sequence[float]):
    """Fit a Platt scaling map on validation predictions only."""
    y = np.asarray(validation_labels, dtype=int)
    probabilities = np.asarray(validation_probabilities, dtype=float)
    if y.ndim != 1 or probabilities.shape != y.shape or np.unique(y).size != 2:
        raise ValueError("calibration needs paired validation predictions with both classes")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("validation probabilities must be finite and within [0, 1]")
    logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1))
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(logits.reshape(-1, 1), y)
    return calibrator


def apply_platt_calibrator(calibrator: LogisticRegression, probabilities: Sequence[float]) -> np.ndarray:
    """Apply a validation-fitted Platt map to held-out probabilities."""
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("probabilities must be finite and within [0, 1]")
    logits = np.log(np.clip(values, 1e-6, 1 - 1e-6) / np.clip(1 - values, 1e-6, 1))
    return calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
