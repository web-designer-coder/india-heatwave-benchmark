"""Verify ERA5 annual/final cache hashes, snapshots, dates, and seasonal padding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from heatwave_benchmark.provenance import validate_era5_icechunk_cache  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1981)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--annual-only",
        action="store_true",
        help="Validate complete annual windows without requiring combined full-period caches",
    )
    args = parser.parse_args()
    points_path = PROJECT_ROOT / "data" / "interim" / "analysis_grid_points.csv"
    points = pd.read_csv(points_path)
    summary = validate_era5_icechunk_cache(
        PROJECT_ROOT / "data" / "raw" / "era5_icechunk",
        points,
        args.start_year,
        args.end_year,
        require_final=not args.annual_only,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
