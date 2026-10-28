"""Run a documented OpenAlex search for India heatwave prediction literature."""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "openalex_heatwave_search_2026-10-25.json"
QUERIES = [
    "India heatwave machine learning prediction forecast IMD",
    "Indian heatwave forecasting machine learning reanalysis grid",
    "India heatwave onset prediction detection machine learning",
    "India heatwave geographic transfer spatial validation benchmark",
    "India heatwave operational forecast evaluation machine learning",
    "India short-range heatwave forecast temperature zones SVR XGBoost",
    "Central India regional heatwave prediction statistical recurrent models",
    "object-based forecasting heat waves India MODE",
    "India heatwave IMD ERA5 heatwave predictor benchmark",
]
FIELDS = "id,title,publication_year,publication_date,doi,type,authorships,primary_location,abstract_inverted_index"


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    if not inverted:
        return None
    words: dict[int, str] = {}
    for token, positions in inverted.items():
        for position in positions:
            words[position] = token
    return " ".join(words[position] for position in sorted(words))


def main() -> int:
    session = requests.Session()
    session.headers.update({"User-Agent": "IndiaHeatwaveBenchmark-literature-search/0.1 (research use)"})
    result = {
        "database": "OpenAlex Works API",
        "endpoint": "https://api.openalex.org/works",
        "search_date": date.today().isoformat(),
        "searched_at_utc": datetime.now(timezone.utc).isoformat(),
        "search_semantics": "OpenAlex full-text search (default relevance ordering); first 50 results per query; overlapping records deduplicated by OpenAlex ID.",
        "queries": [],
        "records": {},
    }
    for query in QUERIES:
        for attempt in range(6):
            response = session.get(
                "https://api.openalex.org/works",
                params={"search": query, "per-page": 50, "select": FIELDS},
                timeout=60,
            )
            if response.status_code != 429:
                break
            time.sleep(min(30, 2 ** attempt))
        response.raise_for_status()
        payload = response.json()
        query_record = {"query": query, "total_matching_works_reported": payload["meta"]["count"], "returned": len(payload["results"])}
        result["queries"].append(query_record)
        for work in payload["results"]:
            work_id = work["id"]
            location = work.get("primary_location") or {}
            source = location.get("source") or {}
            result["records"].setdefault(work_id, {
                "openalex_id": work_id,
                "title": work.get("title"),
                "publication_year": work.get("publication_year"),
                "publication_date": work.get("publication_date"),
                "doi": work.get("doi"),
                "type": work.get("type"),
                "venue": source.get("display_name"),
                "landing_page_url": location.get("landing_page_url"),
                "authors": [auth.get("author", {}).get("display_name") for auth in work.get("authorships", [])],
                "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
                "matched_queries": [],
            })["matched_queries"].append(query)
        print(f"{query}: {query_record['returned']} returned; {query_record['total_matching_works_reported']} matching", flush=True)
        time.sleep(1.2)
    result["deduplicated_record_count"] = len(result["records"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(result['records'])} unique records", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
