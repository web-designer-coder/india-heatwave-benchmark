"""Summarize the 2014--2025 temporal test by calendar year.

This script reads saved predictions only; it does not fit or select a model.
The table is descriptive because only 12 test years are available.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reports/generated_era5/test_predictions.csv.gz"
OUTPUT = ROOT / "reports/generated_era5/yearwise_temporal_test_metrics.csv"
PROBABILITY_COLUMNS = {
    "persistence": "persistence_calibrated_probability",
    "climatology": "climatology_calibrated_probability",
    "histogram_gradient_boosting": "hist_gradient_boosting_calibrated_probability",
}
ONSET_PREDICTIONS = ROOT / "reports/onset_model_era5/onset_test_predictions.csv.gz"


def main() -> None:
    frame = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    frame = frame.loc[
        frame.scenario.eq("temporal") & frame.horizon_days.eq(1)
    ].copy()
    records: list[dict[str, object]] = []
    for year, group in frame.groupby(frame.date.dt.year, sort=True):
        for model, column in PROBABILITY_COLUMNS.items():
            known = group.target.notna() & group[column].notna()
            rows = group.loc[known]
            y = rows.target.astype(int)
            if y.nunique() < 2:
                ap = float("nan")
            else:
                ap = float(average_precision_score(y, rows[column]))
            records.append({
                "target_scope": "full_event",
                "year": int(year),
                "model": model,
                "rows": int(len(rows)),
                "events": int(y.sum()),
                "prevalence": float(y.mean()) if len(y) else float("nan"),
                "average_precision": ap,
                "brier_score": float(brier_score_loss(y, rows[column])) if len(y) else float("nan"),
            })
    onset = pd.read_csv(ONSET_PREDICTIONS, parse_dates=["date"])
    onset = onset.loc[
        onset.scenario.eq("temporal_onset_exploratory")
        & onset.horizon_days.eq(1)
        & onset.current_heatwave.eq(0)
    ]
    for year, group in onset.groupby(onset.date.dt.year, sort=True):
        known = group.target.notna() & group.onset_probability.notna()
        rows = group.loc[known]
        y = rows.target.astype(int)
        records.append({
            "target_scope": "onset_redesign_exploratory",
            "year": int(year),
            "model": "onset_HGB",
            "rows": int(len(rows)),
            "events": int(y.sum()),
            "prevalence": float(y.mean()) if len(y) else float("nan"),
            "average_precision": float(average_precision_score(y, rows.onset_probability)) if y.nunique() == 2 else float("nan"),
            "brier_score": float(brier_score_loss(y, rows.onset_probability)) if len(y) else float("nan"),
        })
    result = pd.DataFrame(records)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
