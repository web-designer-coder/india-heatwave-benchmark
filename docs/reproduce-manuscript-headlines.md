# Reproducing the manuscript headline results

This page maps the numerical claims in `paper/overleaf_ems_draft/main.tex` to the scripts and output files that generate them. Run commands from the project root in PowerShell. They use the locked environment and overwrite the corresponding generated reports. The full 1981–2025 model reruns are computationally expensive.

## Inputs and limits

Install the environment once:

```powershell
uv sync --locked --extra dev --extra era5-cloud
```

The primary runs require local copies of `data/interim/imd_tmax_1981_2025.nc` and `data/interim/era5_1981_2025.nc`. They also require the study-grid metadata and terrain mask included in the source package. Verify the acquired files and pinned ERA5 snapshot with the commands in `docs/reproducibility.md` before running. If the source files are absent, follow the provider access instructions there; the scripts do not bypass IMD data-use terms.

The 2026 IMD, CPC, and NASA POWER results require their matching local 2026 source inputs and the frozen model artifacts. These files are not included in the source archive. Evaluation scripts score the existing inputs; separate acquisition scripts are documented in `docs/reproducibility.md`. Raw provider data and row-level predictions must not be added to the public source archive.

## Regenerate result tables and numerical claims

Run these commands in order. All include the fixed ERA5 predictor, fit cap, and five spatial folds used for the reported primary analysis.

```powershell
uv run --locked python scripts/run_benchmark.py --predictor-source era5 --max-training-rows 200000 --spatial-folds 5
uv run --locked python scripts/run_transition_benchmark.py --predictor-source era5 --max-training-rows 200000 --spatial-folds 5
uv run --locked --extra era5-cloud python scripts/run_transition_benchmark.py --predictor-source era5 --max-training-rows 200000 --spatial-folds 5 --onset-model
uv run --locked python scripts/run_onset_diagnostics.py --predictor-source era5
uv run --locked python scripts/run_onset_bootstrap.py --predictor-source era5
uv run --locked python scripts/run_onset_redesign_paired_bootstrap.py
uv run --locked python scripts/run_sensitivity.py --predictor-source era5 --max-training-rows 200000 --spatial-folds 5
uv run --locked python scripts/evaluate_label_normal_hgb_sensitivity.py
uv run --locked python scripts/percentile_label_sensitivity.py
uv run --locked python scripts/run_brier_decomposition.py
uv run --locked python scripts/summarize_yearwise_temporal_test.py
uv run --locked --extra era5-cloud python scripts/evaluate_frozen_onset_2026.py
uv run --locked python scripts/evaluate_frozen_onset_cpc_2026.py
uv run --locked python scripts/summarize_cpc_reporting_support_2026.py
uv run --locked python scripts/evaluate_frozen_onset_power_2026.py
```

The source and output map is:

| Manuscript result | Regeneration script | Main output |
|---|---|---|
| Temporal Day-1/2/3 model scores, selected models and thresholds | `scripts/run_benchmark.py` | `reports/generated_era5/benchmark_summary.json`, `metrics.csv`, `test_predictions.csv.gz` |
| Five buffered geographic folds and paired/group intervals | `scripts/run_benchmark.py` | `reports/generated_era5/group_metrics.csv`, `group_macro_bootstrap_intervals.csv`, `paired_model_vs_climatology_bootstrap_intervals.csv` |
| Day-1 active-event versus onset diagnostics and intervals | `scripts/run_transition_benchmark.py`; `scripts/run_onset_diagnostics.py`; `scripts/run_onset_bootstrap.py` | `reports/generated_state_transition_era5/transition_summary.json` and CSVs; `reports/generated_era5/onset_diagnostic_summary.json`, `onset_diagnostic_metrics.csv`, `onset_bootstrap_intervals.csv` |
| Exploratory onset redesign on previously inspected 2014–2025 holdouts | `scripts/run_transition_benchmark.py --onset-model`; `scripts/run_onset_redesign_paired_bootstrap.py` | `reports/onset_model_era5/onset_summary.json`, `onset_test_predictions.csv.gz`, `onset_paired_bootstrap_intervals.csv` |
| 3/5/7-degree block and 0/1/2-degree buffer sensitivities | `scripts/run_sensitivity.py` | `reports/generated_era5/sensitivity_summary.json`, `sensitivity_metrics.csv` |
| Alternative climatological normals; percentile-duration target | `scripts/evaluate_label_normal_hgb_sensitivity.py`; `scripts/percentile_label_sensitivity.py` | `reports/generated_era5/label_normal_hgb_sensitivity/`; `reports/percentile_label_sensitivity/label_overlap_metrics.json` |
| Brier decomposition; year-wise headline checks | `scripts/run_brier_decomposition.py`; `scripts/summarize_yearwise_temporal_test.py` | `reports/generated_era5/brier_decomposition_100bins.csv`; `yearwise_temporal_test_metrics.csv` |
| Frozen 2026 IMD test and baseline/paired intervals | `scripts/evaluate_frozen_onset_2026.py` | `reports/independent_onset_2026/frozen_onset_2026_metrics.json` and prediction/interval files |
| CPC product check and report-support sensitivity | `scripts/evaluate_frozen_onset_cpc_2026.py`; `scripts/summarize_cpc_reporting_support_2026.py` | `reports/independent_cpc_2026/cpc_frozen_onset_metrics_2026.json`, `cpc_report_supported_sensitivity_2026.json` |
| NASA POWER/MERRA-2 product check | `scripts/evaluate_frozen_onset_power_2026.py` | `reports/independent_power_2026/frozen_onset_power_metrics_2026.json` |

The IMD, CPC, and POWER results are separate product evaluations. CPC and POWER do not establish independent station validation, and the one-season ERA5-based test is not operational forecast verification. The onset redesign's 2014–2025 comparison remains exploratory because those test periods were inspected before redesign.

## Regenerate figures, manifests, and source archive

After the numerical runs, rebuild the generated figures and local package metadata:

```powershell
uv run --locked python scripts/render_benchmark_figure.py --predictor-source era5 --output paper/overleaf_ems_draft/figures/temporal-test-pr-calibration.png
uv run --locked python paper/overleaf_ems_draft/plot_spatial.py
uv run --locked python scripts/render_independent_onset_2026_figure.py
uv run --locked python paper/overleaf_ems_draft/export_figure_pdfs.py
uv run --locked python scripts/build_release_manifest.py
uv run --locked python scripts/build_source_archive.py
```

`paper/reproducibility-manifest.json` records SHA-256 digests for the compact generated reports and source files. Refresh it only after all intended outputs are regenerated. The source archive is a local artifact; this procedure does not publish it or issue a DOI.

## Reproduction status

Regeneration from locally cached inputs establishes computational reproducibility for those exact inputs. It does not verify data-use permissions or substitute for a clean fresh full-period acquisition. The fresh ERA5 acquisition check is separate and is described in `docs/reproducibility.md`. Never label a result as regenerated unless the corresponding command completed and the output files were checked.

## Follow-up reproduction audit (2026-10-28)

The previously unrerun sensitivity analyses were rerun to separate folders under `reports/reproduction_check_2026-10-28/`, preserving the canonical reports:

- Alternative-normal sensitivity: the summary and all 36 metric rows match the retained outputs exactly.
- Percentile-duration target sensitivity: the 1,509,750-row grid/day table matches exactly. The ten model metric records match within floating-point noise (maximum absolute difference 1.8e-9).
- Spatial sensitivity: all 45 folds completed. Its 376 metric rows and run summary match the retained outputs exactly.

The original frozen onset artifact was not replaced. Its SHA-256 is `c93db74fef286a23bddc392442b611c97ce7c7d004c46deb9232f4ccea4970a1`; a separately reconstructed artifact has a different SHA-256. The bundles match on estimator class/configuration, the ordered 67-feature list, temperature normals, selected hyperparameters, train/calibration periods, Platt coefficients, and threshold. The reconstructed bundle omits `selection_folds` and `protocol_status`; two private fitted-state fields also differ (`_feature_subsample_rng` and `_loss.closs`). Despite these differences, the saved 2014–2025 predictions are exactly equal (789,256 rows), as are the 2026 all-grid score and onset prediction tables (33,550 and 32,289 rows) and the reported 2026 metrics. See the local `reports/reproduction_check_2026-10-28/reproduction_audit.md` and `model_artifact_comparison.json` for detailed evidence.

These checks use the cached local input files and are not a clean reproduction from newly acquired source data. The onset redesign remains exploratory; the independent test covers one 2026 season only.
