"""Recalculate GHCN-Daily station checksums and quality counts offline.

Use this after correcting manifest logic without reacquiring mutable station
files. Every station file must still match its recorded SHA-256 checksum.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from acquire_ghcnd_india_station_data import OUT, RAW, parse_dly, sha256  # noqa: E402


def main() -> None:
    manifest_path = OUT / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stations = json.loads((OUT / "station_inventory.json").read_text(encoding="utf-8"))
    station_by_id = {str(station["station_id"]): station for station in stations}
    corrected = 0
    for record in manifest["stations"]:
        station_id = str(record["station_id"])
        station = station_by_id[station_id]
        path = ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        content = path.read_bytes()
        digest = sha256(content)
        if digest != record["sha256"]:
            raise ValueError(f"station file changed since acquisition: {station_id}")
        rows = parse_dly(content, station)
        recent = [
            row for row in rows
            if "2026-03-01" <= str(row["date"]) <= "2026-06-30"
        ]
        good = [row for row in recent if not str(row["quality_flag"]).strip()]
        record["parsed_records"] = len(rows)
        record["parsed_first_year"] = min(
            (int(str(row["date"])[:4]) for row in rows), default=None
        )
        record["parsed_last_date"] = max(
            (str(row["date"]) for row in rows), default=None
        )
        record["2026_march_june_records"] = len(recent)
        record["2026_march_june_unflagged_quality_records"] = len(good)
        record["source_flag_counts_2026"] = pd.Series(
            [str(row["source_flag"]).strip() or "blank" for row in recent], dtype="string"
        ).value_counts().to_dict()
        corrected += 1
    manifest["manifest_quality_count_correction_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    provider_metadata_path = OUT / "provider_metadata.json"
    if provider_metadata_path.is_file():
        manifest["provider_metadata"] = json.loads(
            provider_metadata_path.read_text(encoding="utf-8")
        )
    manifest["data_rights_note"] = (
        "Station values remain local and are excluded from the source archive. "
        "Consult current provider and source-specific conditions; download access "
        "alone is not treated as permission to redistribute."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "checksum_verified_station_files": corrected,
        "quality_accepted_2026_station_days": sum(
            int(row["2026_march_june_unflagged_quality_records"])
            for row in manifest["stations"]
        ),
        "stations_with_quality_accepted_2026_data": sum(
            int(row["2026_march_june_unflagged_quality_records"]) > 0
            for row in manifest["stations"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
