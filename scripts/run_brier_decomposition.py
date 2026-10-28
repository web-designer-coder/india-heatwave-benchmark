"""Estimate a grouped Murphy Brier decomposition for temporal-test predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "generated_era5" / "test_predictions.csv.gz"
OUTPUT = ROOT / "reports" / "generated_era5" / "brier_decomposition_100bins.csv"


def decompose(y: np.ndarray, probability: np.ndarray, bins: int = 100) -> dict[str, float | int]:
    """Return the binned reliability/resolution/uncertainty decomposition.

    Reliability and resolution use equal-frequency probability bins. Their
    decomposition is approximate because forecasts are grouped before scoring.
    """
    valid = np.isfinite(y) & np.isfinite(probability)
    y = np.asarray(y[valid], dtype=float)
    probability = np.asarray(probability[valid], dtype=float)
    if y.size == 0 or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError(
            "decomposition requires non-empty, finite binary targets; "
            f"n={y.size}, sample={np.unique(y[:20]).tolist()}"
        )
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")

    prevalence = float(y.mean())
    unique_probability = np.unique(probability)
    if unique_probability.size <= bins:
        bin_id = pd.Series(pd.Categorical(probability, categories=unique_probability).codes)
    else:
        bin_id = pd.qcut(pd.Series(probability), q=min(bins, y.size), labels=False, duplicates="drop")
    grouped = pd.DataFrame({"event": y, "probability": probability, "bin": bin_id}).groupby(
        "bin", observed=True
    )
    table = grouped.agg(
        n=("event", "size"),
        observed=("event", "mean"),
        forecast=("probability", "mean"),
    )
    weights = table["n"].to_numpy(dtype=float) / y.size
    reliability = float(np.sum(weights * np.square(table["forecast"] - table["observed"])))
    resolution = float(np.sum(weights * np.square(table["observed"] - prevalence)))
    uncertainty = prevalence * (1.0 - prevalence)
    brier = float(np.mean(np.square(probability - y)))
    approximate = reliability - resolution + uncertainty
    return {
        "n": int(y.size),
        "events": int(y.sum()),
        "prevalence": prevalence,
        "equal_frequency_bins": int(len(table)),
        "brier_score": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "decomposition_approximation": float(approximate),
        "binning_residual": float(brier - approximate),
    }


def main() -> int:
    if not INPUT.exists():
        raise FileNotFoundError(f"prediction file is missing: {INPUT}")
    frame = pd.read_csv(INPUT)
    if "scenario" in frame:
        frame = frame.loc[frame["scenario"].eq("temporal")].copy()
    target = "target" if "target" in frame else "target_h1"
    required = {target, "horizon_days"}
    if not required.issubset(frame.columns):
        raise ValueError(f"prediction file lacks required columns: {sorted(required - set(frame.columns))}")

    records: list[dict[str, object]] = []
    for horizon, rows in frame.groupby("horizon_days", sort=True):
        horizon_target = f"target_h{int(horizon)}" if f"target_h{int(horizon)}" in rows else target
        for column in sorted(name for name in rows if name.endswith("_calibrated_probability")):
            model = column.removesuffix("_calibrated_probability")
            valid_rows = rows.loc[rows[horizon_target].notna() & rows[column].notna()]
            if valid_rows.empty:
                continue
            records.append(
                {
                    "horizon_days": int(horizon),
                    "model": model,
                    "probability_type": "platt_calibrated",
                    **decompose(
                        valid_rows[horizon_target].to_numpy(dtype=float),
                        valid_rows[column].to_numpy(dtype=float),
                    ),
                }
            )
    if not records:
        raise ValueError("no calibrated-probability columns found in temporal test predictions")
    result = pd.DataFrame.from_records(records)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(json.dumps({"output": str(OUTPUT), "rows": len(result), "bins": 100}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
