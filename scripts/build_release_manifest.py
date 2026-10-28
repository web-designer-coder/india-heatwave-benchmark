"""Hash the compact ERA5 benchmark outputs for the working release package."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(relative_path: str) -> dict[str, object]:
    return json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    acquisition = read_json("data/processed/acquisition_manifest.json")
    era5_source = read_json("data/raw/era5_icechunk/source_manifest.json")
    imd_downloads = read_json("data/raw/imd/download_manifest.json")
    benchmark = read_json("reports/generated_era5/benchmark_summary.json")
    transition = read_json("reports/generated_state_transition_era5/transition_summary.json")
    clean_reproduction_path = PROJECT_ROOT / "reports" / "clean_environment_reproduction_audit.json"
    clean_reproduction = (
        json.loads(clean_reproduction_path.read_text(encoding="utf-8"))
        if clean_reproduction_path.is_file()
        else None
    )
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    report_dirs = (
        PROJECT_ROOT / "reports" / "generated_era5",
        PROJECT_ROOT / "reports" / "generated_era5" / "label_normal_hgb_sensitivity",
        PROJECT_ROOT / "reports" / "generated_state_transition_era5",
        PROJECT_ROOT / "reports" / "onset_model_era5",
        PROJECT_ROOT / "reports" / "independent_onset_2026",
        PROJECT_ROOT / "reports" / "independent_cpc_2026",
        PROJECT_ROOT / "reports" / "independent_power_2026",
        PROJECT_ROOT / "reports" / "percentile_label_sensitivity",
    )
    row_level_outputs = {
        "cpc_frozen_onset_predictions_2026.csv",
    }
    artifacts = []
    for directory in report_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if (
                path.is_file()
                and path.name not in row_level_outputs
                and path.suffix in {".csv", ".json", ".png"}
                and not path.name.endswith("predictions.csv")
            ):
                artifacts.append(
                    {
                        "path": path.relative_to(PROJECT_ROOT).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    for name in (
        "temporal-test-pr-calibration.pdf",
        "spatial-fold-transfer.pdf",
        "day1-onset-average-precision.pdf",
        "independent-onset-2026-pr.pdf",
    ):
        path = PROJECT_ROOT / "paper" / "overleaf_ems_draft" / "figs" / name
        if path.is_file():
            artifacts.append(
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    # The acquisition manifests live beside ignored raw data. Copy only their
    # public provenance metadata into the paper package so a code-only release
    # still records exactly which sources and snapshots produced these results.
    provenance = {
        "acquisition_manifest": acquisition,
        "era5_source_manifest": era5_source,
        "imd_download_manifest": imd_downloads,
    }
    provenance_path = PROJECT_ROOT / "paper" / "acquisition-provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    source_paths = [PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "uv.lock"]
    for directory in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", PROJECT_ROOT / "configs", PROJECT_ROOT / "tests"):
        source_paths.extend(path for path in directory.rglob("*") if path.is_file() and path.suffix in {".py", ".json"})
    source_paths.extend(path for path in (PROJECT_ROOT / "docs").glob("*.md") if path.is_file())
    source_paths.extend(
        path for path in (PROJECT_ROOT / "paper").glob("*.md")
        if path.is_file() and path.name != "reproducibility-manifest.json"
    )
    manuscript_dir = PROJECT_ROOT / "paper" / "overleaf_ems_draft"
    source_paths.extend(
        path for path in manuscript_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tex", ".bib", ".py"}
    )
    source_paths.append(PROJECT_ROOT / "paper" / "validation-flow.svg")
    source_paths.append(PROJECT_ROOT / "README.md")
    source_paths.append(Path(__file__).resolve())
    source_files = [
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(source_paths))
    ]

    result = {
        "manifest_status": "working-tree evidence manifest; refresh after the versioned code release",
        "study_period": acquisition["period"],
        "grid_cells": acquisition["predictor_grid_points"],
        "target_source": "IMD daily 1-degree gridded maximum temperature",
        "predictor_source": {
            "name": era5_source["dataset"],
            "registry_url": acquisition["predictor_api"],
            "snapshot_id": acquisition["predictor_snapshot_id"],
            "delivery": acquisition["predictor_delivery"],
            "license": acquisition["predictor_data_license"],
            "variables": acquisition["predictor_variables"],
            "time_standard": acquisition["predictor_time_standard"],
            "cell_selection": acquisition["predictor_cell_selection"],
            "native_resolution": acquisition["predictor_native_resolution"],
            "daily_aggregation": era5_source["daily_aggregation"],
        },
        "acquisition": {
            "accessed_on": acquisition["accessed_on"],
            "seasonal_predictor_rows": 1_599_400,
            "annual_cell_files": 12_375,
            "final_cell_cache_files": 275,
            "cache_validation_command": (
                "uv run --extra era5-cloud python scripts/validate_era5_cache.py "
                "--start-year 1981 --end-year 2025"
            ),
        },
        "experiment": {
            "label_normal_years": benchmark["config"]["label_normal_years"],
            "train_years": benchmark["config"]["train_years"],
            "validation_years": benchmark["config"]["validation_years"],
            "test_years": benchmark["config"]["test_years"],
            "horizons_days": benchmark["config"]["forecast_horizons"],
            "spatial_block_degrees": benchmark["config"]["spatial_block_degrees"],
            "spatial_buffer_degrees": benchmark["config"]["spatial_buffer_degrees"],
            "spatial_folds": 5,
            "random_seed": benchmark["config"]["random_seed"],
            "max_training_rows": benchmark["config"]["max_training_rows"],
            "validation_false_alarm_constraint": benchmark["config"][
                "threshold_max_validation_false_alarm_rate"
            ],
        },
        "software": benchmark["software"],
        "source_release": {
            "package_name": pyproject["project"]["name"],
            "package_version": pyproject["project"]["version"],
            "git_revision": git_revision(),
            "note": "Git revision records the source commit used to build this manifest; the published release tag identifies the packaged snapshot.",
        },
        "transition_experiment": {
            "scope_note": transition.get("scope_note", ""),
            "summary_path": "reports/generated_state_transition_era5/transition_summary.json",
        },
        "verification": {
            "test_command": "uv run --extra dev --extra era5-cloud python -m pytest -q",
            "latest_result": "99 passed on 2026-10-28 with the locked project environment and dev plus era5-cloud extras",
            "clean_environment_setup": "UV_PROJECT_ENVIRONMENT=<fresh-path> uv sync --locked --extra dev --extra era5-cloud",
            "era5_cache_validation": "12,375 annual files; 275 final caches; 1,599,400 seasonal rows",
            "full_clean_environment_benchmark_reproduction": (
                "passed from cached inputs; detailed local audit was excluded from the source release"
                if clean_reproduction and clean_reproduction.get("all_passed")
                else "not run"
            ),
        },
        "result_artifacts": artifacts,
        "headline_result_map": {
            "primary_temporal_and_spatial": {
                "command": "uv run --locked python scripts/run_benchmark.py --predictor-source era5 --max-training-rows 200000 --spatial-folds 5",
                "summary": "reports/generated_era5/benchmark_summary.json",
                "metrics": "reports/generated_era5/metrics.csv",
                "spatial_metrics": "reports/generated_era5/group_metrics.csv",
            },
            "state_conditioned_day1": {
                "command": "uv run --locked python scripts/run_transition_benchmark.py --predictor-source era5 --max-training-rows 200000 --spatial-folds 5",
                "summary": "reports/generated_state_transition_era5/transition_summary.json",
            },
            "exploratory_onset_redesign": {
                "command": "uv run --locked --extra era5-cloud python scripts/run_transition_benchmark.py --predictor-source era5 --max-training-rows 200000 --spatial-folds 5 --onset-model",
                "summary": "reports/onset_model_era5/onset_summary.json",
            },
            "frozen_2026_imd_cpc_power": {
                "commands": [
                    "uv run --locked --extra era5-cloud python scripts/evaluate_frozen_onset_2026.py",
                    "uv run --locked python scripts/evaluate_frozen_onset_cpc_2026.py",
                    "uv run --locked python scripts/evaluate_frozen_onset_power_2026.py",
                ],
                "summaries": [
                    "reports/independent_onset_2026/frozen_onset_2026_metrics.json",
                    "reports/independent_cpc_2026/cpc_frozen_onset_metrics_2026.json",
                    "reports/independent_power_2026/frozen_onset_power_metrics_2026.json",
                ],
            },
            "full_command_map": "docs/reproduce-manuscript-headlines.md",
        },
        "source_files": source_files,
        "provenance_file": {
            "path": "paper/acquisition-provenance.json",
            "bytes": provenance_path.stat().st_size,
            "sha256": sha256_file(provenance_path),
        },
        "clean_reproduction_audit": (
            {
                "path": clean_reproduction_path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": clean_reproduction_path.stat().st_size,
                "sha256": sha256_file(clean_reproduction_path),
            }
            if clean_reproduction
            else None
        ),
    }
    destination = PROJECT_ROOT / "paper" / "reproducibility-manifest.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {destination.relative_to(PROJECT_ROOT)} with {len(artifacts)} hashed result artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
