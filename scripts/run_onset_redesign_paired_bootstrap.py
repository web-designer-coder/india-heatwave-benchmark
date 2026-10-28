"""Paired year/fold bootstrap for the exploratory onset redesign comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


ROOT = Path(__file__).resolve().parents[1]
REDESIGN_FILE = ROOT / "reports/onset_model_era5/onset_test_predictions.csv.gz"
REFERENCE_FILE = ROOT / "reports/generated_state_transition_era5/transition_test_predictions.csv.gz"
OUTPUT_FILE = ROOT / "reports/onset_model_era5/onset_paired_bootstrap_intervals.csv"
MODEL_COLUMNS = {
    "pooled_state_feature": "pooled_hist_gradient_boosting_with_issue_state_calibrated_probability",
    "two_branch": "state_transition_hist_gradient_boosting_calibrated_probability",
}


def grouped_scores(frame: pd.DataFrame, probability: str, group_column: str) -> pd.DataFrame:
    rows = []
    for group, subset in frame.groupby(group_column, sort=True):
        valid = subset[["target", probability]].dropna()
        if valid.empty:
            continue
        y = valid["target"].to_numpy(dtype=int)
        p = valid[probability].to_numpy(dtype=float)
        if np.unique(y).size != 2:
            ap = np.nan
        else:
            ap = float(average_precision_score(y, p))
        rows.append(
            {
                "group": group,
                "average_precision": ap,
                "brier_score": float(brier_score_loss(y, p)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    redesign = pd.read_csv(REDESIGN_FILE)
    references = pd.read_csv(REFERENCE_FILE)
    keys = ["date", "grid_id", "target", "current_heatwave", "spatial_fold"]
    missing = set(keys + ["onset_probability", "scenario"]) - set(redesign.columns)
    if missing:
        raise ValueError(f"redesign predictions lack columns: {sorted(missing)}")
    missing = set(keys + ["scenario", *MODEL_COLUMNS.values()]) - set(references.columns)
    if missing:
        raise ValueError(f"reference predictions lack columns: {sorted(missing)}")

    output_rows = []
    scenarios = (
        ("temporal", "temporal_onset_exploratory", "year"),
        ("spatial_transfer", "spatial_transfer_exploratory", "spatial_fold"),
    )
    rng = np.random.default_rng(20261027)
    for reference_scenario, redesign_scenario, group_column in scenarios:
        left = redesign.loc[redesign["scenario"].eq(redesign_scenario) & redesign["current_heatwave"].eq(0)].copy()
        right = references.loc[references["scenario"].eq(reference_scenario) & references["current_heatwave"].eq(0)].copy()
        if group_column == "year":
            left[group_column] = pd.to_datetime(left["date"], errors="raise").dt.year
            right[group_column] = pd.to_datetime(right["date"], errors="raise").dt.year
        left_columns = list(dict.fromkeys(keys + [group_column, "onset_probability"]))
        right_columns = list(dict.fromkeys(keys + [group_column, *MODEL_COLUMNS.values()]))
        left = left[left_columns].drop_duplicates(keys)
        right = right[right_columns].drop_duplicates(keys)
        paired = left.merge(right, on=keys + [group_column], how="inner", validate="one_to_one")
        if paired.empty:
            raise ValueError(f"no paired predictions for {reference_scenario}")

        model_scores = {"redesign": grouped_scores(paired, "onset_probability", group_column)}
        for model, column in MODEL_COLUMNS.items():
            model_scores[model] = grouped_scores(paired, column, group_column)
        for reference_model in MODEL_COLUMNS:
            comparison = model_scores["redesign"].merge(
                model_scores[reference_model],
                on="group",
                suffixes=("_redesign", "_reference"),
            )
            for metric in ("average_precision", "brier_score"):
                delta = (
                    comparison[f"{metric}_redesign"] - comparison[f"{metric}_reference"]
                ).to_numpy(dtype=float)
                delta = delta[np.isfinite(delta)]
                if delta.size == 0:
                    continue
                boot = np.array(
                    [rng.choice(delta, size=delta.size, replace=True).mean() for _ in range(2000)]
                )
                output_rows.append(
                    {
                        "scenario": reference_scenario,
                        "group_unit": group_column,
                        "candidate": "onset_redesign",
                        "reference": reference_model,
                        "metric": metric,
                        "difference_candidate_minus_reference": float(delta.mean()),
                        "lower_95": float(np.quantile(boot, 0.025)),
                        "upper_95": float(np.quantile(boot, 0.975)),
                        "n_groups": int(delta.size),
                        "n_bootstrap": 2000,
                        "paired_rows": int(len(paired)),
                        "interpretation": "exploratory; test periods were inspected before redesign",
                    }
                )

    result = pd.DataFrame(output_rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(result)} paired exploratory intervals to {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
