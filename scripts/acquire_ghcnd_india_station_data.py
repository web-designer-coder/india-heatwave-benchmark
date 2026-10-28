"""Acquire NOAA GHCN-Daily India TMAX station records for the robustness audit.

Downloaded observations remain local and are excluded from the source archive.
The script records source flags and checksums rather than assuming that public
download availability establishes reuse or redistribution permission.
"""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
DAILY_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station}.dly"
OUT = ROOT / "data/raw/ghcnd_india"
RAW = OUT / "stations"


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def get(url: str, timeout: int = 90) -> bytes:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}") from last_error


def parse_inventory(text: str) -> list[dict[str, object]]:
    rows = []
    for line in text.splitlines():
        if not line.startswith("IN") or line[31:35] != "TMAX":
            continue
        rows.append({
            "station_id": line[:11].strip(),
            "latitude": float(line[12:20]),
            "longitude": float(line[21:30]),
            "element": line[31:35],
            "first_year": int(line[36:40]),
            "last_year": int(line[41:45]),
        })
    return rows


def parse_station_metadata(text: str) -> dict[str, dict[str, object]]:
    result = {}
    for line in text.splitlines():
        if line.startswith("IN"):
            result[line[:11].strip()] = {
                "station_name": line[41:71].strip(),
                "state": line[72:74].strip(),
                "gsn_flag": line[75:76].strip(),
                "hcn_crn_flag": line[76:77].strip(),
                "wmo_id": line[77:81].strip(),
            }
    return result


def parse_dly(content: bytes, station: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in content.decode("ascii", errors="replace").splitlines():
        if len(raw) < 21 or raw[17:21] != "TMAX":
            continue
        # GHCN .dly fixed-width layout: ID(11), year(4), month(2), element(4).
        year = int(raw[11:15])
        month = int(raw[15:17])
        if not 1 <= month <= 12:
            continue
        prefix = f"{station['station_id']}{year:04d}{month:02d}TMAX"
        if not raw.startswith(prefix):
            continue
        n_days = (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).day
        for day in range(1, n_days + 1):
            offset = 21 + (day - 1) * 8
            value = int(raw[offset : offset + 5])
            mflag = raw[offset + 5 : offset + 6]
            qflag = raw[offset + 6 : offset + 7]
            sflag = raw[offset + 7 : offset + 8]
            if value == -9999:
                continue
            rows.append({
                "station_id": station["station_id"],
                "date": f"{year:04d}-{month:02d}-{day:02d}",
                "tmax_c": value / 10.0,
                "measurement_flag": mflag,
                "quality_flag": qflag,
                "source_flag": sflag,
            })
    return rows


def acquire_one(station: dict[str, object]) -> dict[str, object]:
    station_id = str(station["station_id"])
    url = DAILY_URL.format(station=station_id)
    content = get(url)
    if not content.startswith(station_id.encode("ascii")):
        raise ValueError(f"unexpected station payload for {station_id}")
    path = RAW / f"{station_id}.dly"
    path.write_bytes(content)
    rows = parse_dly(content, station)
    years = [int(str(row["date"])[:4]) for row in rows]
    recent = [row for row in rows if str(row["date"]) >= "2026-03-01" and str(row["date"]) <= "2026-06-30"]
    good_recent = [row for row in recent if not str(row["quality_flag"]).strip()]
    return {
        **station,
        "url": url,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(content),
        "bytes": len(content),
        "parsed_records": len(rows),
        "parsed_first_year": min(years) if years else None,
        "parsed_last_date": max((str(row["date"]) for row in rows), default=None),
        "2026_march_june_records": len(recent),
        "2026_march_june_unflagged_quality_records": len(good_recent),
        "source_flag_counts_2026": pd.Series([str(r["source_flag"]).strip() or "blank" for r in recent]).value_counts().to_dict(),
    }


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    inventory_bytes = get(INVENTORY_URL, timeout=180)
    station_bytes = get(STATIONS_URL, timeout=180)
    inventory = parse_inventory(inventory_bytes.decode("ascii"))
    metadata = parse_station_metadata(station_bytes.decode("ascii", errors="replace"))
    for station in inventory:
        station.update(metadata.get(str(station["station_id"]), {}))
    (OUT / "inventory.txt").write_bytes(inventory_bytes)
    (OUT / "stations.txt").write_bytes(station_bytes)
    (OUT / "station_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(f"India TMAX station inventory: {len(inventory)}", flush=True)
    results = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(acquire_one, station): station for station in inventory}
        for future in concurrent.futures.as_completed(futures):
            station = futures[future]
            try:
                record = future.result()
                results.append(record)
                print(f"{record['station_id']}: parsed {record['parsed_records']} TMAX rows; 2026 Mar-Jun {record['2026_march_june_records']}", flush=True)
            except Exception as exc:  # noqa: BLE001
                errors.append({"station_id": station["station_id"], "error": repr(exc)})
                print(f"{station['station_id']}: ERROR {exc}", flush=True)
    result = {
        "source": "NOAA NCEI GHCN-Daily",
        "inventory_url": INVENTORY_URL,
        "stations_url": STATIONS_URL,
        "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "inventory_sha256": sha256(inventory_bytes),
        "stations_metadata_sha256": sha256(station_bytes),
        "inventory_tmax_station_count": len(inventory),
        "downloaded_station_count": len(results),
        "errors": errors,
        "stations": sorted(results, key=lambda r: str(r["station_id"])),
        "data_rights_note": "Station values remain local and are excluded from the source archive. Consult current provider and source-specific conditions; download access alone is not treated as permission to redistribute.",
    }
    (OUT / "download_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "inventory": len(inventory), "downloaded": len(results), "errors": len(errors),
        "2026_march_june_stations_with_any_data": sum(int(r["2026_march_june_records"]) > 0 for r in results),
        "2026_march_june_unflagged_stations": sum(int(r["2026_march_june_unflagged_quality_records"]) > 0 for r in results),
    }, indent=2), flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
