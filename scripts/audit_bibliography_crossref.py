"""Check bibliography DOI resolution and metadata against Crossref."""

from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper/overleaf_ems_draft/cas-refs.bib"
OUTPUT = ROOT / "reports" / f"bibliography_doi_audit_{date.today().isoformat()}.json"


def field_value(entry: str, field: str) -> str | None:
    match = re.search(rf"(?im)^\s*{re.escape(field)}\s*=\s*", entry)
    if not match:
        return None
    start = match.end()
    if start >= len(entry):
        return None
    if entry[start] == "{":
        depth = 1
        index = start + 1
        while index < len(entry) and depth:
            if entry[index] == "\\":
                index += 2
                continue
            if entry[index] == "{":
                depth += 1
            elif entry[index] == "}":
                depth -= 1
            index += 1
        if depth:
            return None
        value = entry[start + 1:index - 1]
    elif entry[start] == '"':
        end = entry.find('"', start + 1)
        if end < 0:
            return None
        value = entry[start + 1:end]
    else:
        value = re.split(r"[,\n]", entry[start:], maxsplit=1)[0]
    return re.sub(r"\s+", " ", value).strip()


def entries() -> list[dict[str, str | None]]:
    text = BIB.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^@\w+\s*\{([^,]+),", text))
    result = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end]
        doi = field_value(body, "doi")
        if doi:
            result.append({
                "key": match.group(1).strip(),
                "doi": doi.strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/"),
                "title": field_value(body, "title"),
                "year": field_value(body, "year"),
            })
    return result


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def main() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "IndiaHeatwaveBenchmark-bibliography-audit/1.0 (research)"})
    checks = []
    for record in entries():
        response = session.get(f"https://api.crossref.org/works/{record['doi']}", timeout=45)
        item = {**record, "crossref_status": response.status_code, "registration_agency": "Crossref"}
        if response.ok:
            message = response.json()["message"]
            crossref_title = (message.get("title") or [""])[0]
            crossref_year = None
            for key in ("published-print", "published-online", "published", "created"):
                date_parts = (message.get(key) or {}).get("date-parts")
                if date_parts and date_parts[0]:
                    crossref_year = str(date_parts[0][0])
                    break
            expected_title = record.get("title") or ""
            title_similarity = SequenceMatcher(None, normalize(expected_title), normalize(crossref_title)).ratio()
            item.update({
                "http_status": response.status_code,
                "crossref_title": crossref_title,
                "crossref_year": crossref_year,
                "crossref_container_title": (message.get("container-title") or [None])[0],
                "title_similarity": round(title_similarity, 3),
                "title_match": title_similarity >= 0.8,
                "year_match": not record.get("year") or not crossref_year or record["year"] == crossref_year,
            })
        else:
            quoted_doi = requests.utils.quote(str(record["doi"]), safe="")
            datacite_response = session.get(f"https://api.datacite.org/dois/{quoted_doi}", timeout=45)
            item["datacite_status"] = datacite_response.status_code
            item["registration_agency"] = "DataCite" if datacite_response.ok else None
            item["http_status"] = datacite_response.status_code
            if datacite_response.ok:
                attributes = datacite_response.json()["data"]["attributes"]
                datacite_title = (attributes.get("titles") or [{}])[0].get("title", "")
                datacite_year = str(attributes.get("publicationYear")) if attributes.get("publicationYear") else None
                expected_title = record.get("title") or ""
                title_similarity = SequenceMatcher(None, normalize(expected_title), normalize(datacite_title)).ratio()
                item.update({
                    "datacite_title": datacite_title,
                    "datacite_year": datacite_year,
                    "title_similarity": round(title_similarity, 3),
                    "title_match": title_similarity >= 0.8,
                    "year_match": not record.get("year") or not datacite_year or record["year"] == datacite_year,
                })
            else:
                item["error"] = f"Crossref: {response.text[:250]}; DataCite: {datacite_response.text[:250]}"
        checks.append(item)
        time.sleep(0.15)

    report = {
        "audit_date": date.today().isoformat(),
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Crossref REST API DOI lookup",
        "bib_file": BIB.relative_to(ROOT).as_posix(),
        "entries_with_doi": len(checks),
        "checks": checks,
        "summary": {
            "resolved": sum(item["http_status"] == 200 for item in checks),
            "unresolved": sum(item["http_status"] != 200 for item in checks),
            "title_mismatches": sum(item.get("title_match") is False for item in checks),
            "year_mismatches": sum(item.get("year_match") is False for item in checks),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
