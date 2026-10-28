"""Independently reacquire one pinned ERA5 cell/year and compare its cached values."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heatwave_benchmark.data import download_era5_icechunk_seasonal_grid  # noqa: E402

POINT = pd.DataFrame([{"grid_id": "lat10.5_lon76.5", "latitude": 10.5, "longitude": 76.5}])
SNAPSHOT = "TZ2KX6A24CJ5DG17T7MG"
OUT = ROOT / "data/raw/era5_fresh_check"
OUT.mkdir(parents=True, exist_ok=True)
print("Starting independent one-cell ERA5 fetch from pinned snapshot", flush=True)
manifest = download_era5_icechunk_seasonal_grid(POINT, OUT, 2025, 2025, snapshot_id=SNAPSHOT, max_workers=1)
fresh_path = OUT / "lat10.5_lon76.5.json"
reference_path = ROOT / "data/raw/era5_icechunk/lat10.5_lon76.5.json"
fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
reference = json.loads(reference_path.read_text(encoding="utf-8"))
window = [d for d in fresh["daily"]["time"] if "2025-02-22" <= d <= "2025-06-30"]
idx_f = [fresh["daily"]["time"].index(d) for d in window]
idx_r = [reference["daily"]["time"].index(d) for d in window]
variables = sorted(set(fresh["daily"]) & set(reference["daily"]) - {"time"})
diffs = {}
for variable in variables:
    a = np.asarray([fresh["daily"][variable][i] for i in idx_f], dtype=float)
    b = np.asarray([reference["daily"][variable][i] for i in idx_r], dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    diffs[variable] = {"max_abs_diff": float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else None,
                       "compared_values": int(valid.sum()), "missing_pattern_equal": bool(np.array_equal(np.isfinite(a), np.isfinite(b)))}
result = {
    "verified_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    "source": "Earthmover public ERA5 Icechunk mirror",
    "snapshot_id": SNAPSHOT,
    "grid_id": "lat10.5_lon76.5",
    "year": 2025,
    "daily_window": [window[0], window[-1]],
    "days_compared": len(window),
    "fresh_file_sha256": hashlib.sha256(fresh_path.read_bytes()).hexdigest(),
    "reference_file_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    "variable_differences": diffs,
    "source_manifest_status": manifest.to_dict(orient="records"),
    "scope": "single grid cell and one year; not a fresh full 275-cell, 45-year reacquisition",
}
result["all_values_within_1e-4"] = all(
    v["missing_pattern_equal"] and v["max_abs_diff"] is not None and v["max_abs_diff"] <= 1e-4
    for v in diffs.values()
)
(ROOT / "reports/fresh_era5_sample_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: result[k] for k in ("days_compared", "variable_differences", "all_values_within_1e-4")}, indent=2), flush=True)
