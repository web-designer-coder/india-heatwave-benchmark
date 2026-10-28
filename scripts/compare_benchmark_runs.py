"""Compare saved benchmark runs, including large prediction files in chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_csv(reference: Path, candidate: Path, chunksize: int) -> dict[str, object]:
    left = pd.read_csv(reference, chunksize=chunksize)
    right = pd.read_csv(candidate, chunksize=chunksize)
    compared_rows = 0
    max_abs_diff: dict[str, float] = {}
    categorical_mismatches: dict[str, int] = {}
    while True:
        try:
            a = next(left)
        except StopIteration:
            a = None
        try:
            b = next(right)
        except StopIteration:
            b = None
        if a is None or b is None:
            if a is not None or b is not None:
                raise ValueError(f"row count differs between {reference} and {candidate}")
            break
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            raise ValueError(f"schema or chunk row count differs for {reference.name}")
        compared_rows += len(a)
        for column in a.columns:
            if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(b[column]):
                delta = (a[column].astype(float) - b[column].astype(float)).abs()
                value = float(delta.max()) if len(delta) else 0.0
                max_abs_diff[column] = max(max_abs_diff.get(column, 0.0), value)
            else:
                equal = a[column].fillna("<NULL>").astype(str).to_numpy() == b[column].fillna("<NULL>").astype(str).to_numpy()
                categorical_mismatches[column] = categorical_mismatches.get(column, 0) + int((~equal).sum())
    return {
        "rows": compared_rows,
        "max_absolute_numeric_difference": max_abs_diff,
        "categorical_mismatches": categorical_mismatches,
        "reference_sha256": sha256(reference),
        "candidate_sha256": sha256(candidate),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "reference_directory": str(args.reference),
        "candidate_directory": str(args.candidate),
        "absolute_tolerance": args.atol,
        "files": {},
    }
    for filename in ("metrics.csv", "test_predictions.csv.gz"):
        result = compare_csv(args.reference / filename, args.candidate / filename, args.chunksize)
        result["within_tolerance"] = (
            max(result["max_absolute_numeric_difference"].values(), default=0.0) <= args.atol
            and not any(result["categorical_mismatches"].values())
        )
        report["files"][filename] = result
    for directory in (args.reference, args.candidate):
        summary_path = directory / "benchmark_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
    reference_summary = json.loads((args.reference / "benchmark_summary.json").read_text(encoding="utf-8"))
    candidate_summary = json.loads((args.candidate / "benchmark_summary.json").read_text(encoding="utf-8"))
    selected = {}
    for key in ("temporal_h1", "temporal_h2", "temporal_h3"):
        selected[key] = {
            "reference": reference_summary[key]["best_validation_model"],
            "candidate": candidate_summary[key]["best_validation_model"],
        }
    selected["spatial_folds"] = [
        {
            "fold": index + 1,
            "reference": reference_summary["spatial_transfer_h1"][index]["best_validation_model"],
            "candidate": candidate_summary["spatial_transfer_h1"][index]["best_validation_model"],
        }
        for index in range(len(reference_summary["spatial_transfer_h1"]))
    ]
    report["selected_models"] = selected
    report["selected_models_match"] = all(
        row["reference"] == row["candidate"]
        for row in [*selected.values()]
        if isinstance(row, dict)
    ) and all(row["reference"] == row["candidate"] for row in selected["spatial_folds"])
    report["all_passed"] = all(item["within_tolerance"] for item in report["files"].values()) and report["selected_models_match"]
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
