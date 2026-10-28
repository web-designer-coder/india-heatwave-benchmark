"""Redownload each IMD Tmax year to a separate cache and verify SHA-256 parity."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from heatwave_benchmark.data import IMD_TMAX_DOWNLOAD, download_imd_tmax_year

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "raw" / "imd" / "download_manifest.json"
DESTINATION = ROOT / "data" / "raw" / "imd_fresh_check"
OUTPUT = ROOT / "reports" / "fresh_imd_acquisition_audit.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    reference = {int(row["year"]): row for row in json.loads(REFERENCE.read_text(encoding="utf-8"))}
    years = sorted(reference)
    checked = []
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        for year in years:
            path = download_imd_tmax_year(year, DESTINATION, session=session, overwrite=True)
            actual = digest(path)
            expected = reference[year]["sha256"]
            checked.append({
                "year": year,
                "bytes": path.stat().st_size,
                "expected_sha256": expected,
                "fresh_download_sha256": actual,
                "matches_reference": actual == expected,
                "source_url": IMD_TMAX_DOWNLOAD,
            })
            print(f"IMD {year}: {'match' if actual == expected else 'MISMATCH'}", flush=True)
            time.sleep(0.2)
    result = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "India Meteorological Department 1-degree daily gridded Tmax",
        "source_url": IMD_TMAX_DOWNLOAD,
        "period": [years[0], years[-1]],
        "years_checked": len(checked),
        "all_checksums_match": all(row["matches_reference"] for row in checked),
        "raw_data_redownloaded_to": DESTINATION.relative_to(ROOT).as_posix(),
        "redistribution": "No raw data included in this audit report or public release.",
        "files": checked,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Audit: {OUTPUT.relative_to(ROOT)}; all_match={result['all_checksums_match']}", flush=True)
    return 0 if result["all_checksums_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
