"""Render paired PR curves for the frozen HGB and climatology baseline."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reports" / "independent_onset_2026" / "frozen_onset_predictions_2026.csv"
METRICS = ROOT / "reports" / "independent_onset_2026" / "frozen_onset_2026_metrics.json"
OUTPUT = ROOT / "paper" / "overleaf_ems_draft" / "figs" / "independent-onset-2026-pr.pdf"


def main() -> None:
    data = pd.read_csv(PREDICTIONS)
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    y = data["target_h1"].to_numpy(dtype=np.int8)
    score = data["calibrated_probability"].to_numpy(dtype=float)
    climatology_score = data["climatology_calibrated_probability"].to_numpy(dtype=float)
    predicted = data["frozen_prediction"].to_numpy(dtype=np.int8)
    threshold = float(metrics["frozen_threshold"])
    prevalence = float(y.mean())
    precision, recall, _ = precision_recall_curve(y, score)
    ap = float(average_precision_score(y, score))
    climatology_precision, climatology_recall, _ = precision_recall_curve(y, climatology_score)
    climatology_ap = float(average_precision_score(y, climatology_score))
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    point_precision = tp / (tp + fp) if tp + fp else 0.0
    point_recall = tp / int((y == 1).sum()) if (y == 1).any() else 0.0

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(6.8, 3.5), constrained_layout=True)
    ax.plot(recall, precision, color="#0072B2", linewidth=1.8, label=f"Frozen HGB (AP={ap:.3f})")
    ax.plot(climatology_recall, climatology_precision, color="#CC79A7", linewidth=1.5,
            label=f"Climatology (AP={climatology_ap:.3f})")
    ax.axhline(prevalence, color="#D55E00", linewidth=1.2, linestyle="--",
               label=f"Event prevalence ({prevalence:.1%})")
    ax.scatter([point_recall], [point_precision], s=38, color="#009E73", edgecolor="white",
               linewidth=0.7, zorder=3, label="Frozen operating threshold")
    ax.annotate(f"precision={point_precision:.1%}\nrecall={point_recall:.1%}",
                xy=(point_recall, point_precision), xytext=(point_recall - 0.28, point_precision + 0.045),
                arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.8},
                ha="left", va="bottom", fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 0.32), xlabel="Recall", ylabel="Precision",
           title="Independent 2026 onset test (IMD real-time Tmax; ERA5 reanalysis)")
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT} and {OUTPUT.with_suffix('.png')}; HGB AP={ap:.6f}; climatology AP={climatology_ap:.6f}; frozen threshold={threshold:.15f}")


if __name__ == "__main__":
    main()
