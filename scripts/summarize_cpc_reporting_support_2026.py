"""Post hoc CPC target sensitivity restricted to cell-days with report support.

This does not alter the primary CPC product evaluation or any frozen model
setting. It retains onset rows only when CPC NMAX is positive on both the
issue and verifying day, and is reported as an exploratory support sensitivity.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_frozen_onset_cpc_2026 import _read_cpc_series, _study_cells  # noqa: E402
from evaluate_frozen_onset_2026 import onset_metrics, moving_block_intervals, paired_moving_block_differences  # noqa: E402


OUT = ROOT / "reports/independent_cpc_2026"
SCORES = OUT / "cpc_frozen_onset_predictions_2026.csv"
METRICS = OUT / "cpc_frozen_onset_metrics_2026.json"


def main() -> None:
    cells = _study_cells()
    dates, _, support = _read_cpc_series(cells)
    support_frame = pd.DataFrame({
        "date": np.repeat(dates.to_numpy(), len(cells)),
        "grid_id": np.tile(cells.grid_id.astype(str), len(dates)),
        "nmax_reports": support[:, :, 1].reshape(-1),
    })
    support_frame["nmax_reports_next"] = support_frame.groupby("grid_id", sort=False)["nmax_reports"].shift(-1)
    support_frame["next_date"] = support_frame.groupby("grid_id", sort=False)["date"].shift(-1)
    support_frame.loc[
        support_frame.next_date != support_frame.date + pd.Timedelta(days=1),
        "nmax_reports_next",
    ] = np.nan
    frame = pd.read_csv(SCORES, parse_dates=["date"])
    frame = frame.merge(support_frame, on=["date", "grid_id"], how="left", validate="one_to_one")
    selected = frame.loc[
        frame.nmax_reports.gt(0)
        & frame.nmax_reports_next.gt(0)
        & frame.target_h1.notna()
    ].copy()
    selected["target_h1"] = selected.target_h1.astype(np.int8)
    if selected.empty or selected.target_h1.nunique() < 2:
        raise ValueError("CPC report-supported subset lacks both onset classes")
    frozen = json.loads(METRICS.read_text(encoding="utf-8"))
    hgb = selected.hgb_probability.to_numpy(float)
    climatology = selected.climatology_calibrated_probability.to_numpy(float)
    y = selected.target_h1.to_numpy(np.int8)
    threshold = float(frozen["frozen_threshold"])
    climate_threshold = float(selected.climatology_threshold.iloc[0])
    persistence = np.zeros(len(selected), dtype=float)
    report = {
        "scope": "post hoc exploratory support sensitivity; only CPC NMAX-positive issue and verifying cell-days",
        "support_rule": "CPC NMAX sum across the four 0.5-degree subcells is greater than zero on both issue day and target day",
        "fraction_of_all_study_cell_days_with_any_nmax_report": float((support[:, :, 1] > 0).mean()),
        "rows": int(len(selected)),
        "events": int(y.sum()),
        "prevalence": float(y.mean()),
        "unique_cells": int(selected.grid_id.nunique()),
        "unique_issue_dates": int(selected.date.nunique()),
        "fraction_of_primary_cpc_onset_rows_retained": float(len(selected) / len(frame.loc[frame.target_h1.notna()])),
        "frozen_threshold": threshold,
        "climatology_threshold": climate_threshold,
        "metrics": {
            "frozen_hgb": onset_metrics(y, hgb, threshold),
            "climatology": onset_metrics(y, climatology, climate_threshold),
            "persistence": onset_metrics(y, persistence, threshold),
        },
        "frozen_hgb_intervals": moving_block_intervals(selected, hgb, threshold, seed=20261105).to_dict(orient="records"),
        "paired_difference_intervals": paired_moving_block_differences(
            selected, hgb, threshold, climatology, climate_threshold, "climatology", seed=20261106
        ).to_dict(orient="records"),
        "limitation": "This subset is selected by CPC report-support metadata and remains a single-season external-product analysis; it does not remove possible GTS/IMD source overlap or retrospective ERA5 limitations.",
    }
    (OUT / "cpc_report_supported_sensitivity_2026.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(selected), "events": int(y.sum()), "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
