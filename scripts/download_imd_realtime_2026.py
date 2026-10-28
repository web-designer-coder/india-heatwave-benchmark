"""Acquire IMD daily 1-degree Tmax real-time grids for March-June 2026.

The files are retained locally for the frozen, independent onset evaluation.
This script does not redistribute IMD data or make a claim about publication
reuse rights.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heatwave_benchmark.data import IMD_GRID_SHAPE  # noqa: E402

SOURCE_PAGE = "https://www.imdpune.gov.in/cmpg/Realtimedata/maxone/Max1_Download.html"
DOWNLOAD_URL = "https://www.imdpune.gov.in/cmpg/Realtimedata/maxone/maxone.php"
DESTINATION = ROOT / "data" / "raw" / "imd_realtime_2026"
MANIFEST_PATH = DESTINATION / "download_manifest.json"
START = pd.Timestamp("2026-03-01")
END = pd.Timestamp("2026-06-30")
EXPECTED_BYTES = int(np.prod(IMD_GRID_SHAPE) * np.dtype("<f4").itemsize)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        saved = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if saved.get("start_date") != START.date().isoformat() or saved.get("end_date") != END.date().isoformat():
            raise ValueError("existing IMD real-time manifest covers a different date range")
        return saved
    return {
        "dataset": "IMD daily maximum temperature, 1.0 degree real-time binary grids",
        "source_page": SOURCE_PAGE,
        "download_endpoint": DOWNLOAD_URL,
        "download_method": "POST with form field maxone=DDMMYYYY",
        "start_date": START.date().isoformat(),
        "end_date": END.date().isoformat(),
        "grid_shape_latitude_longitude": list(IMD_GRID_SHAPE),
        "expected_file_size_bytes": EXPECTED_BYTES,
        "accessed_on": date.today().isoformat(),
        "reuse_note": "Files retained locally for analysis; redistribution and publication reuse rights are not asserted.",
        "files": [],
    }


def validate_file(path: Path) -> None:
    if path.stat().st_size != EXPECTED_BYTES:
        raise ValueError(f"unexpected size for {path.name}: {path.stat().st_size} bytes")
    values = np.fromfile(path, dtype="<f4")
    valid = values[np.isfinite(values) & ~np.isclose(values, 99.9)]
    if valid.size == 0 or np.any((valid < -25.0) | (valid > 60.0)):
        raise ValueError(f"invalid Tmax values in {path.name}")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    records = {str(row["date"]): row for row in manifest["files"]}
    dates = pd.date_range(START, END, freq="D")
    with requests.Session() as session:
        session.headers.update({"User-Agent": "IndiaHeatwaveBenchmark/0.1 (research data verification)"})
        for index, day in enumerate(dates, start=1):
            day_key = day.date().isoformat()
            form_date = day.strftime("%d%m%Y")
            filename = f"max1_{form_date}.grd"
            target = DESTINATION / filename
            if target.exists():
                validate_file(target)
                digest = sha256_file(target)
                old = records.get(day_key)
                if old and old.get("sha256") != digest:
                    raise ValueError(f"existing manifest checksum differs for {filename}")
                if old:
                    print(f"[{index}/{len(dates)}] verified cached {day_key}", flush=True)
                    continue
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    response = session.post(
                        DOWNLOAD_URL,
                        data={"maxone": form_date},
                        timeout=(15, 90),
                    )
                    response.raise_for_status()
                    disposition = response.headers.get("Content-Disposition", "")
                    expected_name = filename.lower()
                    if expected_name not in disposition.lower():
                        raise ValueError(f"unexpected response filename for {day_key}: {disposition!r}")
                    temporary = target.with_suffix(target.suffix + ".part")
                    temporary.write_bytes(response.content)
                    try:
                        validate_file(temporary)
                        temporary.replace(target)
                    finally:
                        temporary.unlink(missing_ok=True)
                    break
                except Exception as exc:  # retry transient server/network failures
                    last_error = exc
                    if attempt == 3:
                        raise
                    time.sleep(attempt * 1.5)
            if last_error is not None and not target.exists():
                raise last_error
            records[day_key] = {
                "date": day_key,
                "file": filename,
                "size_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "content_disposition": f"attachment; filename={filename}",
            }
            manifest["files"] = [records[key] for key in sorted(records)]
            MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"[{index}/{len(dates)}] acquired {day_key}", flush=True)
            time.sleep(0.15)

    manifest["files"] = [records[key] for key in sorted(records)]
    if [row["date"] for row in manifest["files"]] != [day.date().isoformat() for day in dates]:
        raise ValueError("IMD real-time acquisition did not produce a complete consecutive date series")
    manifest["completed_on"] = date.today().isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Complete: {len(dates)} daily grids; manifest {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
