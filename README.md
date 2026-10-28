# India Heatwave Benchmark

## Software release and citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23721963.svg)](https://doi.org/10.5281/zenodo.23721963)

The **India Heatwave Benchmark** software is available as release [v0.2.0 on GitHub](https://github.com/web-designer-coder/india-heatwave-benchmark/releases/tag/v0.2.0). Its archived, version-specific Zenodo record is [doi:10.5281/zenodo.23721964](https://doi.org/10.5281/zenodo.23721964); the [concept DOI](https://doi.org/10.5281/zenodo.23721963) identifies the software across versions. The source archive contains the benchmark code, dependency lockfile, reproducibility manifest, and headline-results reproduction guide. It excludes raw source datasets. The reported verification used cached local inputs; a fresh full-period data acquisition and rerun were not performed. Researchers must obtain any needed inputs from the relevant providers and follow their applicable access and reuse terms.

Suggested software citation: Wankhade, A. (2026). *India Heatwave Benchmark* (Version 0.2.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23721964

This project evaluates short-lead heatwave event probabilities on the Indian 1-degree IMD grid. The primary experiment uses IMD daily maximum-temperature labels and ERA5 reanalysis predictors for 1981–2025. A separate, frozen onset-model evaluation uses IMD real-time Tmax and matching ERA5 data for March–June 2026. The exploratory NASA POWER/MERRA-2 pilot is retained separately and is not used for the primary manuscript results.

The benchmark compares climatology, persistence, logistic regression, random forest, and histogram gradient boosting using a fixed chronological split and five buffered geographic holdouts. It reports rare-event probability metrics, validation-selected operating thresholds, onset-conditioned diagnostics, grouped uncertainty intervals, and predeclared temporal-normal and spatial-geometry sensitivities. A separate Day-1 comparison evaluates a two-branch state-conditioned model against pooled HGB controls. The contribution is evaluation and geographic-transfer evidence; the models are established methods. ERA5 is retrospective reanalysis, so these results do not establish operational warning skill.

## Study design

- Grid: 275 selected IMD 1-degree cell centers within the study-domain mask.
- Period: 1981–2025; issue dates March–June.
- Heatwave labels: IMD-aligned daily grid-cell criteria with a fixed 1981–2010 normal; see [`docs/data-sources.md`](docs/data-sources.md).
- Temporal split: train 1981–2007, validation 2008–2013, test 2014–2025.
- Geographic split: five held-out 5-degree blocks with a 1-degree buffer; sensitivity runs cover 3-, 5-, and 7-degree blocks and 0-, 1-, and 2-degree buffers.
- ERA5: hourly fields from a pinned public Icechunk snapshot, aggregated to Asia/Kolkata local daily summaries. The snapshot ID and checksums are recorded in the acquisition manifests.
- Independent temporal test: the onset estimator, features, calibration, and threshold were frozen before acquiring March–June 2026 outcomes. See [`docs/independent_evaluation_status_2026-10-27.md`](docs/independent_evaluation_status_2026-10-27.md) for its result and limits.

## Environment

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required. Install the exact locked dependencies from the project root:

```powershell
uv sync --locked --extra dev --extra era5-cloud
```

The lockfile records the environment used for the saved run. The primary pipelines use scikit-learn; xgboost is optional and was not installed for the saved ERA5 benchmark.

## Reproduce the primary ERA5 results

For a full manuscript-claim-to-command map, including the 2026 IMD, CPC, and NASA POWER evaluations, see [`docs/reproduce-manuscript-headlines.md`](docs/reproduce-manuscript-headlines.md). The commands below provide the data acquisition and primary benchmark detail.

### 1. Acquire public inputs

The ERA5 Icechunk downloader is resumable. The first command acquires IMD targets and domain inputs; the second fetches ERA5 hourly windows for all selected cells and years. ERA5 retrieval is a large cloud-data operation and requires a working network connection.

```powershell
uv run python scripts/download_public_data.py
uv run --extra era5-cloud python scripts/download_public_data.py --predictor-source era5 --era5-provider icechunk
```

Then validate the pinned annual and full-period caches:

```powershell
uv run python scripts/validate_era5_cache.py --start-year 1981 --end-year 2025
```

The acquisition manifest, annual cache hashes, snapshot ID, source metadata, and final dataset are saved under `data/raw/era5_icechunk/`, `data/processed/`, and `data/interim/`. The release-manifest builder copies source acquisition metadata to [`paper/acquisition-provenance.json`](paper/acquisition-provenance.json), since local data directories are excluded from this code release. See [`docs/data-sources.md`](docs/data-sources.md) for provider access routes and attribution notes. Source datasets are not bundled or redistributed; researchers must obtain them from providers and follow the applicable terms.

### 2. Run the benchmark and diagnostics

```powershell
uv run python scripts/run_benchmark.py --predictor-source era5 --max-training-rows 200000 --spatial-folds 5
uv run python scripts/run_transition_benchmark.py --predictor-source era5 --max-training-rows 200000 --spatial-folds 5
uv run python scripts/run_onset_diagnostics.py --predictor-source era5
uv run python scripts/run_onset_bootstrap.py --predictor-source era5
uv run python scripts/run_sensitivity.py --predictor-source era5 --spatial-folds 5
uv run python scripts/render_benchmark_figure.py --predictor-source era5
```

Each command is resumable only where explicitly implemented; do not assume that a partially written report directory is a complete run. Inspect command output and confirm the corresponding summary/manifest reports completion before using results. Main outputs are in `reports/generated_era5/`; the state-conditioned experiment is in `reports/generated_state_transition_era5/`. Feature frames are generated in `data/interim/`.

### 3. Reproduce tests and inspect saved results

```powershell
uv run --extra dev --extra era5-cloud python -m pytest -q
```

The suite covers label construction, time aggregation, split logic, feature generation, metrics, model calibration, bootstrap comparisons, acquisition/cache validation, and artifact metadata. On 28 October 2026, all 99 tests passed in the locked project environment with both `dev` and `era5-cloud` extras. This checks the test suite and package installation; it does not download or recompute the full ERA5 benchmark. The source release does not contain result reports. After running the commands, generated summaries and metrics appear at the following paths:

The full temporal and five-fold spatial benchmark was also rerun from that fresh environment using cached IMD and ERA5 NetCDF inputs. The 1,509,750-row feature frame was rebuilt, and the regenerated predictions were compared chunkwise with the saved reference. All 1,610,304 rows matched in categorical fields; numeric differences were at most `1e-7`; all temporal and spatial validation-selected models matched. The detailed audit is retained with the local research workspace and is not included in this source release. This validates recomputation from cached inputs; a fresh full-period ERA5 acquisition followed by an end-to-end benchmark rerun was not performed.

- `reports/generated_era5/benchmark_summary.json`
- `reports/generated_era5/metrics.csv`
- `reports/generated_era5/group_metrics.csv`
- `reports/generated_era5/group_macro_bootstrap_intervals.csv`
- `reports/generated_era5/paired_model_vs_climatology_bootstrap_intervals.csv`
- `reports/generated_era5/onset_diagnostic_metrics.csv`
- `reports/generated_era5/onset_bootstrap_intervals.csv`
- `reports/generated_era5/sensitivity_summary.json` and `sensitivity_metrics.csv`
- `reports/generated_state_transition_era5/transition_summary.json` and its CSV outputs

The current full run uses the immutable ERA5 snapshot `TZ2KX6A24CJ5DG17T7MG`. Reproduction should retain that snapshot rather than silently following a newer archive state. Some cached intermediate and raw files are large; the code archive excludes source data and cell-level derived outputs.

### Reproduce the frozen 2026 onset evaluation

The 2026 evaluation uses a new calendar period but the same IMD and ERA5 source families; it is not independent-source validation or operational forecast verification. The frozen artifact is evaluated without refitting or threshold changes. IMD source data and row-level evaluation outputs are not included in the code archive.

```powershell
uv run --locked python scripts/download_imd_realtime_2026.py
uv run --locked --extra era5-cloud python scripts/acquire_independent_era5_2026.py
uv run --locked --extra era5-cloud python scripts/evaluate_frozen_onset_2026.py
uv run --locked python scripts/render_independent_onset_2026_figure.py
```

### Reproduce the station-data and year-wise audits

The GHCN-Daily station check is limited to nine nearby stations and does not establish independent-source validation. Download the station subset and inspect source flags, then score the frozen probabilities without refitting:

```powershell
uv run --locked python scripts/acquire_ghcnd_india_station_data.py
uv run --locked python scripts/audit_ghcnd_independent_2026.py
uv run --locked --extra era5-cloud python scripts/evaluate_frozen_onset_2026.py
uv run --locked python scripts/evaluate_frozen_onset_ghcnd.py
uv run --locked python scripts/evaluate_frozen_onset_ghcnd.py --max-distance-km 100
uv run --locked python scripts/summarize_yearwise_temporal_test.py
uv run --locked python scripts/document_novelty_search.py
```

The raw NOAA station files, IMD grids, and row-level evaluation records are not redistributed. The focused Crossref/ScienceDirect search log and its scope limits are in [`docs/novelty_search_log_2026-10-27.md`](docs/novelty_search_log_2026-10-27.md). The public software release is MIT-licensed; see [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

After regenerating the results, refresh the compact checksum manifest for the tables, summaries, and figure intended for a source release:

```powershell
uv run python scripts/build_release_manifest.py
```

It writes [`paper/reproducibility-manifest.json`](paper/reproducibility-manifest.json), including hashes for compact result artifacts, source/configuration files, and the copied acquisition provenance. Large compressed per-example prediction files and source caches are excluded from the manifest's release-artifact list. The release manifest records the Git revision associated with the published source snapshot.

## Other project materials

- [`docs/data-sources.md`](docs/data-sources.md): source descriptions, access routes, and attribution cautions.
- [`docs/reproducibility.md`](docs/reproducibility.md): acquisition, environment, and reproducibility scope.
- [`docs/independent_evaluation_status_2026-10-27.md`](docs/independent_evaluation_status_2026-10-27.md): scope and limitations of the separate 2026 evaluations.
- [`paper/reproducibility-manifest.json`](paper/reproducibility-manifest.json): pinned source metadata, experiment settings, software versions, and SHA-256 hashes for compact result artifacts.
- [`paper/acquisition-provenance.json`](paper/acquisition-provenance.json): release-safe copies of IMD and ERA5 acquisition metadata and input checksums; it contains no source data.

Journal fit or quartile does not predict acceptance. Verify current indexing, category-specific metrics, article type, costs, data policy, and instructions before selecting a journal. Publication acceptance cannot be guaranteed.
