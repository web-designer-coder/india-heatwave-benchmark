from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from heatwave_benchmark.artifacts import benchmark_output_directory  # noqa: E402
from scripts.run_benchmark import _plot_temporal_test  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a benchmark figure from saved test predictions.")
    parser.add_argument(
        "--predictor-source", choices=("nasa_power", "era5"), default="nasa_power"
    )
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report_directory = benchmark_output_directory(args.predictor_source)
    predictions_path = args.predictions or report_directory / "test_predictions.csv.gz"
    output_path = args.output or report_directory / "temporal_test_precision_recall_calibration.png"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"test prediction file not found: {predictions_path}")

    header = pd.read_csv(predictions_path, nrows=0).columns.tolist()
    probability_columns = [
        column
        for column in header
        if column.endswith("_raw_probability") or column.endswith("_calibrated_probability")
    ]
    required = {"target", "scenario", "horizon_days", *probability_columns}
    missing = required - set(header)
    if missing:
        raise ValueError(f"test prediction file is missing columns: {sorted(missing)}")

    predictions = pd.read_csv(predictions_path, usecols=sorted(required))
    temporal_day1 = predictions.loc[
        predictions["scenario"].eq("temporal") & predictions["horizon_days"].eq(1)
    ]
    if temporal_day1.empty:
        raise ValueError("test prediction file has no temporal Day-1 rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_temporal_test(temporal_day1, output_path)
    print(f"Saved figure: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
