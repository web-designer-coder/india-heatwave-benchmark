"""Acquire the CPC 2026 Tmax and report-count grids by HTTP byte range.

Only the TMAX and NMAX planes are retained for 22 February--30 June 2026.
The official GrADS control file documents the 720 x 360 grid, little-endian
float32 encoding, four-variable daily record order, and missing-value code.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
import time

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/cpc_tmax_2026"
BASE_URL = "https://ftp.cpc.ncep.noaa.gov/precip/PEOPLE/wd52ws/global_temp"
DATA_URL = f"{BASE_URL}/CPC_GLOBAL_T_V0.x_0.5deg.lnx.2026"
CONTROL_URL = f"{BASE_URL}/CPC_GLOBAL_T_V0.x_0.5deg.lnx.2024.ctl"
START = date(2026, 2, 22)
END = date(2026, 6, 30)
NX, NY, NVAR = 720, 360, 4
FLOAT_BYTES = 4
FIELD_BYTES = NX * NY * FLOAT_BYTES
DAY_BYTES = FIELD_BYTES * NVAR
TMAX_NMAX_BYTES = FIELD_BYTES * 2


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_control_file() -> dict[str, str]:
    response = requests.get(CONTROL_URL, timeout=60)
    response.raise_for_status()
    text = response.text
    if "xdef 720" not in text.lower() or "ydef 360" not in text.lower():
        raise ValueError("Unexpected CPC grid dimensions in control file")
    if "tmax" not in text.lower() or "nmax" not in text.lower():
        raise ValueError("CPC control file does not describe expected variables")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "CPC_GLOBAL_T_V0.x_0.5deg.lnx.2024.ctl"
    path.write_bytes(response.content)
    return {"url": CONTROL_URL, "path": str(path.relative_to(ROOT)), "sha256": sha256(response.content)}


def fetch_day(day: date) -> dict[str, object]:
    day_index = day.timetuple().tm_yday - 1
    start = day_index * DAY_BYTES
    end = start + TMAX_NMAX_BYTES - 1
    target = OUT / f"cpc_tmax_nmax_{day:%Y%m%d}.bin"
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = requests.get(DATA_URL, headers=headers, timeout=(30, 120))
            if response.status_code != 206:
                raise ValueError(f"Expected HTTP 206 byte range, got {response.status_code}")
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if not match or (int(match.group(1)), int(match.group(2))) != (start, end):
                raise ValueError(f"Unexpected Content-Range: {content_range!r}")
            if len(response.content) != TMAX_NMAX_BYTES:
                raise ValueError(f"Expected {TMAX_NMAX_BYTES} bytes, received {len(response.content)}")
            target.write_bytes(response.content)
            return {
                "date": day.isoformat(),
                "file": str(target.relative_to(ROOT)),
                "byte_range": [start, end],
                "bytes": len(response.content),
                "sha256": sha256(response.content),
            }
        except Exception as exc:  # retry transient network and server errors
            last_error = exc
            if attempt == 4:
                break
            time.sleep(attempt * 1.5)
    assert last_error is not None
    raise RuntimeError(f"CPC range acquisition failed for {day}") from last_error


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    control = fetch_control_file()
    days: list[date] = []
    current = START
    while current <= END:
        days.append(current)
        current += timedelta(days=1)
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_day, day): day for day in days}
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda record: str(record["date"]))
    metadata = {
        "dataset": "CPC Global Unified Temperature, V0.x",
        "provider": "NOAA/NWS/NCEP Climate Prediction Center",
        "source_url": DATA_URL,
        "control_file": control,
        "accessed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "date_range_inclusive": [START.isoformat(), END.isoformat()],
        "grid": {"nx": NX, "ny": NY, "longitude_first_center": 0.25, "latitude_first_center": -89.75, "spacing_degrees": 0.5},
        "encoding": "little-endian float32; daily GrADS variable order tmax,nmax,tmin,nmin; missing value -999",
        "retained_variables": ["tmax", "nmax"],
        "bytes_per_date": TMAX_NMAX_BYTES,
        "daily_records": records,
        "attribution": "NOAA/NWS/NCEP Climate Prediction Center",
        "known_independence_limit": "CPC states Tmax/Tmin inputs originate from its GTS daily-report archive. Some underlying station reports may overlap with the IMD network; this is an independent gridded analysis product, not guaranteed source-independent station evidence.",
    }
    manifest = OUT / "acquisition_manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"days": len(records), "total_bytes": sum(int(row["bytes"]) for row in records), "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
