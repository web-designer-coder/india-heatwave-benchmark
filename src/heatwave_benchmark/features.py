"""Calendar-aware feature construction with explicit leakage boundaries."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def build_lagged_features(
    frame: pd.DataFrame,
    predictor_columns: Sequence[str],
    context_columns: Sequence[str] = (),
    max_lag_days: int = 7,
    group_col: str = "grid_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """Build features from issue-day and previous calendar-day observations.

    ``lag_0`` is the observation on the issue date. ``lag_n`` is from exactly
    n calendar days before that date; missing calendar dates remain missing.
    Target columns are omitted unless explicitly and incorrectly supplied as
    predictors, so callers should pass only meteorological inputs.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not isinstance(max_lag_days, int) or max_lag_days < 0:
        raise ValueError("max_lag_days must be a non-negative integer")
    predictors = tuple(predictor_columns)
    context = tuple(context_columns)
    if not predictors:
        raise ValueError("at least one predictor column is required")

    required = {group_col, date_col, *predictors, *context}
    missing = required - set(frame.columns)
    if missing:
        if set(predictors) & missing:
            raise ValueError(f"predictor columns are missing: {sorted(set(predictors) & missing)}")
        raise ValueError(f"required columns are missing: {sorted(missing)}")
    if len(set(predictors)) != len(predictors) or len(set(context)) != len(context):
        raise ValueError("predictor and context columns must be unique")
    if set(predictors) & set(context):
        raise ValueError("predictor columns and context columns must be disjoint")
    if group_col in predictors or date_col in predictors or group_col in context or date_col in context:
        raise ValueError("group and date columns cannot be predictors or context columns")

    source = frame.loc[:, [group_col, date_col, *predictors, *context]].copy()
    source[date_col] = pd.to_datetime(source[date_col], errors="raise")
    if source[[group_col, date_col]].isna().any().any():
        raise ValueError("group and date keys cannot be missing")
    if source.duplicated([group_col, date_col]).any():
        raise ValueError("duplicate grid/date rows are not allowed")

    source = source.sort_values([group_col, date_col], kind="stable").reset_index(drop=True)
    output = source.loc[:, [group_col, date_col, *context]].copy()
    keys = [group_col, date_col]

    for lag in range(max_lag_days + 1):
        lagged = source.loc[:, [group_col, date_col, *predictors]].copy()
        lagged[date_col] = lagged[date_col] + pd.to_timedelta(lag, unit="D")
        lagged = lagged.rename(
            columns={column: f"{column}_lag_{lag}" for column in predictors}
        )
        output = output.merge(lagged, on=keys, how="left", validate="one_to_one", sort=False)

    return output


def _reference_dayofyear(values: pd.Series) -> np.ndarray:
    """Map month/day values to a leap-year day-of-year index (1..366)."""
    dates = pd.to_datetime(values, errors="raise")
    month_day = dates.dt.strftime("%m-%d")
    return pd.to_datetime("2000-" + month_day, format="%Y-%m-%d").dt.day_of_year.to_numpy()


def fit_calendar_temperature_normals(
    weather: pd.DataFrame,
    grid_col: str = "grid_id",
    date_col: str = "date",
    value_col: str = "t2m_max",
    window_days: int = 31,
    min_observations: int = 10,
) -> pd.DataFrame:
    """Fit grid-cell day-of-year temperature normals on a training-period table.

    The caller is responsible for passing predictor observations from the
    training period only. A circular moving window handles the year boundary
    and aligns leap/non-leap dates by month and day.
    """
    if not isinstance(weather, pd.DataFrame):
        raise TypeError("weather must be a pandas DataFrame")
    required = {grid_col, date_col, value_col}
    missing = required - set(weather.columns)
    if missing:
        raise ValueError(f"weather columns are missing: {sorted(missing)}")
    if not isinstance(window_days, int) or window_days < 1 or window_days > 365 or window_days % 2 == 0:
        raise ValueError("window_days must be an odd integer between 1 and 365")
    if not isinstance(min_observations, int) or min_observations < 1:
        raise ValueError("min_observations must be a positive integer")

    source = weather.loc[:, [grid_col, date_col, value_col]].copy()
    source[date_col] = pd.to_datetime(source[date_col], errors="raise")
    source[value_col] = pd.to_numeric(source[value_col], errors="raise")
    if source[[grid_col, date_col]].isna().any().any():
        raise ValueError("grid and date keys cannot be missing")
    if source.duplicated([grid_col, date_col]).any():
        raise ValueError("weather grid/date rows must be unique")
    if np.isinf(source[value_col].to_numpy(dtype=float)).any():
        raise ValueError("temperature observations cannot be infinite")
    source = source.loc[source[value_col].notna()].copy()
    if source.empty:
        raise ValueError("weather contains no finite temperature observations")
    source["dayofyear"] = _reference_dayofyear(source[date_col])

    half_window = window_days // 2
    offsets = range(-half_window, half_window + 1)
    rows: list[dict[str, object]] = []
    for grid_id, group in source.groupby(grid_col, sort=True, observed=True):
        daily = group.groupby("dayofyear", sort=True)[value_col].agg(["sum", "count"])
        weighted_sum = np.zeros(366, dtype=float)
        counts = np.zeros(366, dtype=float)
        for dayofyear, record in daily.iterrows():
            index = int(dayofyear) - 1
            weighted_sum[index] = float(record["sum"])
            counts[index] = float(record["count"])

        window_sum = np.zeros(366, dtype=float)
        window_count = np.zeros(366, dtype=float)
        for offset in offsets:
            window_sum += np.roll(weighted_sum, offset)
            window_count += np.roll(counts, offset)
        estimable = window_count >= min_observations
        if not estimable.any():
            raise ValueError(f"insufficient training observations for calendar normals at grid {grid_id!r}")
        for index in np.flatnonzero(estimable):
            normal = window_sum[index] / window_count[index]
            rows.append(
                {grid_col: grid_id, "dayofyear": int(index + 1), "t2m_max_normal": float(normal)}
            )

    return pd.DataFrame(rows, columns=[grid_col, "dayofyear", "t2m_max_normal"])


def add_temperature_anomaly_features(
    frame: pd.DataFrame,
    normals: pd.DataFrame,
    grid_col: str = "grid_id",
    date_col: str = "date",
    value_prefix: str = "t2m_max_lag_",
    max_lag: int = 7,
) -> pd.DataFrame:
    """Add fold-fitted temperature anomaly and recent-trend features to issue rows."""
    if not isinstance(frame, pd.DataFrame) or not isinstance(normals, pd.DataFrame):
        raise TypeError("frame and normals must be pandas DataFrames")
    if not isinstance(max_lag, int) or max_lag < 7:
        raise ValueError("max_lag must be an integer of at least 7")
    required = {grid_col, date_col, *(f"{value_prefix}{lag}" for lag in range(max_lag + 1))}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"issue-frame columns are missing: {sorted(missing)}")
    normal_required = {grid_col, "dayofyear", "t2m_max_normal"}
    missing_normals = normal_required - set(normals.columns)
    if missing_normals:
        raise ValueError(f"normal columns are missing: {sorted(missing_normals)}")
    if normals.duplicated([grid_col, "dayofyear"]).any():
        raise ValueError("normal grid/day-of-year keys must be unique")

    result = frame.copy()
    dates = pd.to_datetime(result[date_col], errors="raise")
    normal_index = normals.set_index([grid_col, "dayofyear"])["t2m_max_normal"]
    anomalies: dict[int, np.ndarray] = {}
    for lag in range(max_lag + 1):
        lag_dates = dates - pd.to_timedelta(lag, unit="D")
        doy = _reference_dayofyear(lag_dates)
        keys = pd.MultiIndex.from_arrays([result[grid_col].to_numpy(), doy])
        normal_values = normal_index.reindex(keys).to_numpy(dtype=float)
        values = pd.to_numeric(result[f"{value_prefix}{lag}"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(normal_values).all():
            raise ValueError(f"temperature normals are missing for lag {lag}")
        if not np.isfinite(values).all():
            raise ValueError(f"temperature lag values are missing or non-finite for lag {lag}")
        anomalies[lag] = values - normal_values

    result["t2m_max_anom_lag_0"] = anomalies[0]
    result["t2m_max_anom_lag_3"] = anomalies[3]
    result["t2m_max_anom_lag_7"] = anomalies[7]
    result["t2m_max_anom_mean_3d"] = np.mean(
        np.column_stack([anomalies[lag] for lag in range(3)]), axis=1
    )
    result["t2m_max_anom_mean_7d"] = np.mean(
        np.column_stack([anomalies[lag] for lag in range(7)]), axis=1
    )
    result["t2m_max_anom_change_0_3"] = anomalies[0] - anomalies[3]
    return result
