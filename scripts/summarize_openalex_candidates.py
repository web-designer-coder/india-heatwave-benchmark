"""Print high-relevance India heatwave forecasting records for manual screening."""
import json
import sys
from pathlib import Path

records = json.loads(Path("reports/openalex_heatwave_search_2026-10-25.json").read_text(encoding="utf-8"))["records"].values()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for work in records:
    title = (work.get("title") or "").lower()
    abstract = (work.get("abstract") or "").lower()
    text = title + " " + abstract
    if "heat" not in title or not any(x in text for x in ("india", "indian")):
        continue
    if not any(x in text for x in ("forecast", "predict", "prediction", "machine learning", "model")):
        continue
    print("{} | {} | {} | {}".format(work.get("publication_year"), work.get("title"), work.get("doi"), work.get("venue")))
