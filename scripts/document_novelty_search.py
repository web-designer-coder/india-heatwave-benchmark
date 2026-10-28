"""Record reproducible Crossref metadata searches for the related-work audit.

Crossref is a bibliographic registry. These searches support a focused novelty
check but do not replace a Scopus/Web of Science systematic review.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / f"docs/novelty_search_crossref_{date.today().isoformat()}.json"
QUERIES = [
    "India heatwave prediction machine learning grid",
    "Indian heatwave onset machine learning",
    "India heatwave forecast spatial validation grid cell",
    "IMD gridded heatwave event probability prediction India",
]


def main() -> None:
    records = []
    for query in QUERIES:
        response = requests.get(
            "https://api.crossref.org/works",
            params={"query.bibliographic": query, "rows": 30, "select": "DOI,title,author,published,container-title,type,URL"},
            headers={"User-Agent": "IndiaHeatwaveBenchmark/0.2 (bibliographic audit)"},
            timeout=60,
        )
        response.raise_for_status()
        items = response.json()["message"]["items"]
        records.append({"query": query, "returned": len(items), "results": items})
    OUTPUT.write_text(
        json.dumps({
            "database": "Crossref REST API",
            "search_date": date.today().isoformat(),
            "queries": records,
            "screening_note": "Bibliographic metadata search only. Results require manual relevance screening and full-text review; absence from these results is not evidence of novelty.",
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {OUTPUT}")
    for item in records:
        print(f"{item['query']}: {item['returned']} records")


if __name__ == "__main__":
    main()
