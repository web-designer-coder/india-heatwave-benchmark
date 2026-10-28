"""Capture the NOAA GHCN-Daily version and readme without refreshing data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/ghcnd_india"
BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
FILES = {
    "version": f"{BASE}/ghcnd-version.txt",
    "readme": f"{BASE}/readme.txt",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acquired: dict[str, dict[str, str]] = {}
    for name, url in FILES.items():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        payload = response.content
        path = OUT / f"ghcnd-{name}.txt"
        path.write_bytes(payload)
        acquired[name] = {
            "url": url,
            "file": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    version_text = (OUT / "ghcnd-version.txt").read_text(encoding="ascii").splitlines()[0]
    manifest = {
        "provider": "NOAA National Centers for Environmental Information",
        "dataset": "Global Historical Climatology Network-Daily (GHCN-Daily)",
        "provider_version_string": version_text,
        "provider_metadata_retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "files": acquired,
        "note": "Provider metadata snapshot only; it does not change the station files or their acquisition checksums.",
    }
    path = OUT / "provider_metadata.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
