# Frozen onset-model temporal evaluation status (27 October 2026)

## Prespecified model

The model, features, training-period ERA5 anomaly normals, Platt calibrator, and operating threshold were frozen before acquisition and scoring of the new 2026 period. The estimator was fitted on 1981--2007 state-zero rows; calibration and threshold selection used 2008--2013 state-zero rows. The threshold is `0.014854158899725712`. The serialized artifact is `reports/onset_model_era5/onset_model_frozen.joblib`, SHA-256 `c93db74fef286a23bddc392442b611c97ce7c7d004c46deb9232f4ccea4970a1`.

The original 2014--2025 temporal test and five geographic test folds were inspected before model redesign and remain exploratory. They are not used as confirmation here. No model, feature, calibration, or threshold choice was changed after evaluation of the new period.

## Data acquired for the new period

- **IMD target:** Daily 1-degree real-time Tmax binary grids for 1 March--30 June 2026, acquired 27 October 2026 from IMD's [real-time download page](https://www.imdpune.gov.in/cmpg/Realtimedata/maxone/Max1_Download.html). All 122 daily files have the expected 31 x 31 x 4-byte size and passed basic finite/range checks. Each SHA-256, date, source endpoint, and file size is recorded in `data/raw/imd_realtime_2026/download_manifest.json`.
- **ERA5 predictors:** Hourly fields for the frozen 275 study-cell centers, February 22--June 30, 2026, from the already pinned Icechunk snapshot `TZ2KX6A24CJ5DG17T7MG`. All 275 per-cell records use that snapshot and are recorded with SHA-256 values in `data/raw/era5_onset_independent_2026/download_manifest.csv`. The source and aggregation metadata are in `data/raw/era5_onset_independent_2026/source_manifest.json`.
- The February predictor window supplies seven-day lags for the March issue dates. Label normals remain the fixed 1981--2010 Tmax normals. ERA5 anomaly normals remain the frozen 1981--2007 values embedded in the model artifact.
- The 2026 evaluation is independent **in time** from prior holdouts. It uses the same source families as the primary analysis and retrospective ERA5 predictors; it is not an independent-source evaluation or operational forecast verification. The IMD source files and row-level predictions remain local and are not redistributed.

## Frozen model result

The evaluator is `scripts/evaluate_frozen_onset_2026.py`. It loads the frozen HGB without changes. For a same-row comparator, it fits seasonal/location climatology on 1981--2007 onset rows and calibrates/selects its threshold on 2008--2013 only; it also reports the zero-risk persistence baseline for issue-state-zero rows. Neither baseline uses 2026 outcomes. Results are in `reports/independent_onset_2026/frozen_onset_2026_metrics.json`, row-level output in `frozen_onset_predictions_2026.csv`, and paired intervals in `frozen_onset_2026_paired_baseline_intervals.csv`.

| Metric | Estimate | 95% interval |
|---|---:|---:|
| Known onset rows | 32,289 | -- |
| Positive next-day onsets | 355 (1.099%) | -- |
| Unique issue dates / cells | 121 / 273 | -- |
| Average precision | 0.06071 | 0.04085--0.08656 |
| Brier score | 0.010717 | 0.006101--0.016269 |
| Precision at frozen threshold | 0.05701 | 0.04123--0.07162 |
| Recall at frozen threshold | 0.56620 | 0.47933--0.65069 |
| False-alarm rate at frozen threshold | 0.10412 | 0.07030--0.13764 |

The seasonal/location climatology comparator scored AP 0.02514, Brier 0.01080, precision 0.0260, recall 0.2423, and false-alarm rate 0.1009. The persistence baseline assigns zero risk to issue-state-zero rows, so its AP and Brier equal prevalence (0.01099), with zero recall at the frozen HGB threshold. In paired 2,000-replicate 7-day date-block resampling, HGB minus climatology AP was +0.03640 (95% interval +0.02115 to +0.05238), while the Brier difference was -0.000102 (-0.000294 to +0.000052), including zero. Intervals retain all grid-cell rows per sampled issue date and describe within-season variation only. Precision remains low, and the HGB false-alarm interval spans the nominal 10% target. HGB AP is below its 0.1097 score on the previously inspected 2014--2025 onset subset; the periods are different and the 2014--2025 comparison remains exploratory.

This new evaluation supplies one out-of-time temporal check, but it does not resolve the broader scientific review or data-rights questions. The IMD product's applicable reuse terms and publication permissions remain to be clarified before submission or redistribution.

## GHCN-Daily source screen (excluded from model validation)

The project acquired the NOAA NCEI India TMAX inventory and station `.dly` files to assess whether they could support an independent observational test. An audit of 5,229 quality-accepted March--June 2026 station-day records found measurement flag `H` on every record. NCEI defines this flag for TMAX as the highest hourly temperature, not a conventional daily maximum. All records also carry source flag `2`, identifying the Synoptic Summary of the Day version 2 product. These observations are not compatible with the study's daily-Tmax target and were excluded from label construction and scoring. No GHCN-derived model metrics are valid evidence for this paper; any previously generated 50-km or 100-km exploratory scores must not be cited as validation results. Underlying source-report overlap with IMD cannot be ruled out.

The version, station inventory, source files, checksums, and flag audit remain in `data/raw/ghcnd_india/` and `reports/independent_ghcnd_2026/` for provenance. This GHCN-Daily screen did not identify a suitable conventional daily-Tmax validation dataset. The 2026 frozen evaluation above is independent in time only, uses the same IMD and ERA5 source families, and is not operational forecast verification. An official IMD station-data availability query conducted on 28 October 2026 identified a separate candidate route; its status and limitations are recorded below.

## Official IMD station-data availability check (28 October 2026)

The IMD Data Service Portal's public [Data Availability & Cost Estimation](https://dsp.imdpune.gov.in/home_availability.php) form allows a read-only availability and price estimate before a data requisition. For `Surface` / `Table-II (Day Summary)` / `Daily`, the 2026 selector returned 243--273 records for each of 14 station entries sampled across India (13 distinct cities; Chennai was queried at two stations). The portal itself cautions that one record may not contain every parameter and that gaps may exist.

| Station | IMD code | Portal records for 2026 | Displayed estimate including 18% GST |
|---|---:|---:|---:|
| Ahmedabad | 42647 | 273 | Rs. 4,029 |
| Bhubaneswar | 42971 | 243 | Rs. 3,715 |
| Chennai (Meenambakkam) | 43279 | 273 | Rs. 4,029 |
| Chennai (Nungambakkam) | 43278 | 273 | Rs. 4,029 |
| Guwahati | 42410 | 273 | Rs. 4,029 |
| Hyderabad | 43128 | 243 | Rs. 3,715 |
| Jaipur | 42348 | 273 | Rs. 4,029 |
| Kolkata (Alipore) | 42807 | 273 | Rs. 4,029 |
| Mumbai (Colaba) | 43057 | 243 | Rs. 3,715 |
| Nagpur | 42867 | 273 | Rs. 4,029 |
| New Delhi (Safdarjung) | 42182 | 273 | Rs. 4,029 |
| Pune | 43063 | 243 | Rs. 3,715 |
| Srinagar | 42027 | 243 | Rs. 3,715 |
| Thiruvananthapuram | 43371 | 243 | Rs. 3,715 |

These are portal estimates from year-level queries, not a formal quotation or evidence that each record contains valid Tmax. They amount to Rs. 54,522 across all 14 queried entries; removing one of the two Chennai entries gives Rs. 50,493 across 13 cities, assuming one separate request per station. The estimates cover all available 2026 records rather than a selectable March--June subset. No requisition or payment was made, and no station data were acquired. The portal says that data availability can be checked before request and that charges apply to procurement; the IMD [daily surface request form](https://dsp.imdpune.gov.in/data_request_form_dsp.php) describes departmental observations at three-hourly intervals and non-departmental observations at 03 and 12 UTC. The applicable research-publication/reuse terms still require confirmation.

This establishes a potentially usable station-observation access route, but does **not** yet establish a genuinely independent test. The IMD gridded Tmax documentation states that its post-2008 product uses around 180 stations, while station identifiers used in the gridding are not mapped here to the candidate stations. A valid external station test would need the actual records, verified daily-Tmax fields and date coverage, permission for analysis/publication, and a defensible exclusion or provenance check for stations contributing to the gridded target. Until those gates are met, this route remains a paid, unacquired candidate and no station-based model evaluation is claimed. The primary grid documentation is at [IMD's 1-degree Tmax page](https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html); the portal warns that reproduction of website material for commercial purposes requires permission.

## CPC 2026 external-product evaluation

NOAA CPC daily Tmax and NMAX fields were acquired from its 2026 global analysis file for 22 February--30 June using HTTP byte ranges. All 129 daily payloads and the official control file are covered by `data/raw/cpc_tmax_2026/acquisition_manifest.json` with SHA-256 checksums. `scripts/evaluate_frozen_onset_cpc_2026.py` verifies the manifest, averages at least three of four 0.5-degree CPC values per 1-degree study cell, applies the fixed primary IMD 1981--2010 normal and heatwave thresholds, and scores the already frozen 2026 probabilities without fitting or retuning.

The CPC-derived onset test contains 32,321 rows, 375 events (1.160\%), 275 study cells, and 121 issue dates. HGB AP is 0.04723 (95\% 7-day block interval 0.02908--0.06694), versus 0.02189 for historical climatology and 0.01160 for zero-risk persistence. HGB Brier score is 0.011459 (0.006860--0.016891), while climatology is 0.011423. At the unchanged HGB threshold, precision is 4.79\%, recall 46.13\%, and false-alarm rate 10.76\% (0.0731--0.1444). The paired HGB-minus-climatology AP difference is +0.02587 (95\% interval +0.01358 to +0.03979); the Brier difference is +0.000016 (95\% interval -0.000148 to +0.000164). The paired AP interval favors HGB; the Brier difference includes zero.

CPC describes this product as a 0.5-degree daily surface-air-temperature analysis built from GTS daily reports. Some reports may overlap with observations contributing to the IMD grid; CPC is therefore a separate gridded product, not guaranteed source-independent ground truth. The target also reuses the fixed IMD climatological normal, and ERA5 predictors remain retrospective. This provides external-product robustness evidence, not independent raw-source validation or operational forecast verification. Full metrics, paired intervals, labels, and model predictions are in `reports/independent_cpc_2026/`; the frozen evaluation protocol is `docs/frozen_cpc_2026_protocol.md`.

The CPC NMAX field indicates at least one report in the four 0.5-degree subcells for only 43.8\% of study cell-days. In a post hoc sensitivity requiring positive NMAX support on both the issue and verifying day, 13,966 rows from 125 cells remain (162 onsets; 1.160\% prevalence; 43.2\% of primary CPC onset rows). HGB AP is 0.04742 versus 0.02251 for climatology; the paired AP difference is +0.02618 (95\% interval +0.01148 to +0.04212). The paired Brier difference is +0.000090 (95\% interval -0.000061 to +0.000256), including zero. At the frozen threshold, HGB precision is 5.87\%, recall 39.51\%, and false-alarm rate 7.43\%. This is a post hoc, single-season report-support sensitivity; it does not establish source independence or operational forecast skill.

## NASA POWER/MERRA-2 2026 product check (exploratory)

After inspecting the IMD and CPC 2026 results, we used the saved frozen probabilities for one additional post hoc product comparison. `scripts/evaluate_frozen_onset_power_2026.py` acquired daily `T2M_MAX` for the 275 study cells over 1 March--1 July 2026 through the NASA POWER Daily Point API. The raw API responses, source tags, returned coordinates, and per-cell checksums are recorded under `data/raw/nasa_power_2026_product_check/`. The script derived 1981--2010 POWER-specific normals from the local NASA POWER history, applied the existing grid-cell threshold rule, and scored the original saved probabilities without refitting or retuning. The dated metrics and paired 2,000-replicate 7-day block intervals are in `reports/independent_power_2026/frozen_onset_power_metrics_2026.json`; eligible row-level scores and labels are in `frozen_onset_power_predictions_2026.csv.gz`.

The 32,294 eligible onset rows contain 484 positives (1.499\%). HGB AP is 0.05333 versus 0.03155 for the fixed historical climatology (paired difference +0.02178; 95\% interval +0.00288 to +0.03937). Brier scores are 0.014774 and 0.014698, respectively; their paired difference interval is -0.000101 to +0.000240 and includes zero. At the frozen HGB threshold, precision is 5.92\%, recall 40.08\%, and false-alarm rate 9.69\%. The evaluator verified all 275 response date ranges and that POWER returned the same source coordinates as the archived historical POWER series.

This additional product test was selected post hoc. POWER is a gridded analysis product, not independent station truth; the product check uses POWER-derived normals and local-solar-time dates, and possible overlap in assimilated observations with IMD or ERA5 cannot be excluded. Treat it as exploratory product-level robustness only, not independent source validation or operational forecast verification.

## Reproduction commands

```powershell
uv run --locked python scripts/download_imd_realtime_2026.py
uv run --locked --extra era5-cloud python scripts/acquire_independent_era5_2026.py
uv run --locked --extra era5-cloud python scripts/evaluate_frozen_onset_2026.py
uv run --locked python scripts/render_independent_onset_2026_figure.py
uv run --locked python scripts/audit_ghcnd_independent_2026.py
uv run --locked python scripts/evaluate_frozen_onset_power_2026.py
```

Do not change the frozen model, features, calibration, threshold, or target definition and still describe the result as this evaluation. Any changes require a newly reserved untouched test period.
