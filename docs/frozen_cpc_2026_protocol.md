# Frozen CPC 2026 external-product evaluation protocol

**Protocol date:** 27 October 2026. The scoring process began before this note was written, but after the target construction, support rule, comparison, and interval procedure had been encoded in the evaluator. This note records that code-level protocol; it is not a time-stamped preregistration.

## Frozen model

- Estimator artifact: `reports/onset_model_era5/onset_model_frozen.joblib`
- Artifact SHA-256: `c93db74fef286a23bddc392442b611c97ce7c7d004c46deb9232f4ccea4970a1`
- Features and ERA5 anomaly normals: as recorded in `docs/frozen_onset_protocol_2026-10-27.md`
- Platt calibrator and operating threshold: unchanged; threshold `0.014854158899725712`
- The external-product outcomes will not be used to refit, recalibrate, change the threshold, or select features.

## External product and target

- Product: NOAA CPC Global Unified Temperature, V0.x, daily Tmax and NMAX fields on a 0.5-degree grid. CPC documentation says daily Tmax/Tmin originate from its GTS daily-report archive. Some underlying reports may overlap with IMD observations, so this is a separate-product test, not guaranteed source-independent station validation.
- Acquisition window: 22 February--30 June 2026, using the CPC 2026 file and verified byte ranges. Individual file hashes and control-file metadata are recorded in `data/raw/cpc_tmax_2026/acquisition_manifest.json`.
- Spatial alignment: arithmetic mean of the four CPC grid-center values within each study 1-degree cell; require at least three of four finite CPC values for that cell-day to receive a temperature value.
- Label rule: use the primary IMD-aligned terrain-specific thresholds and fixed IMD 1981--2010 cell normal. This deliberately holds the event definition constant while substituting the external product's daily 2026 Tmax values; it does not create an official IMD declaration.
- Onset: CPC-derived issue-day label is zero and the following calendar day's CPC-derived label is positive. Evaluate issue dates 1 March--29 June, retaining only finite frozen model scores and known issue/target labels.

## Scoring and uncertainty

- Same-row comparators: the historical seasonal/location climatology probabilities and validation-selected threshold already frozen in `reports/independent_onset_2026/frozen_model_all_grid_scores_2026.csv.gz`; zero-risk persistence for onset rows.
- Metrics: AP, Brier score, precision, recall, and false-alarm rate. Do not tune the threshold on CPC outcomes.
- Intervals: 2,000 paired moving-block bootstrap replicates, sampling consecutive 7-day issue-date blocks and retaining all available cells for each date. They describe within-season variation only.
- Acquisition script SHA-256: `eca9fb725d5e0b5cd67825e17c32bb0e5fe283ded580ae4e517e6a5c2dbef407` (`scripts/acquire_cpc_tmax_2026.py`).
- Scoring script SHA-256: `c1e7aa7116945c6e0880a06eba80d73e93f29baa9a2281bc140262a57c36c7f7` (`scripts/evaluate_frozen_onset_cpc_2026.py`).

## Interpretation limits

The CPC analysis is based on GTS reports that may overlap the IMD inputs. Its 0.5-degree analysis is aggregated to the 1-degree study grid, and it reuses the primary IMD normal. The same retrospective ERA5 predictors are retained. The result can support external-product robustness only; it cannot establish independent raw-source validation, year-to-year generalization, or operational forecast skill.
