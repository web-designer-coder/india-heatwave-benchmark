# Temporal and spatial leakage audit (27 October 2026)

## Checks supported by the code and saved run artifacts

- Primary temporal partitions are 1981--2007 training, 2008--2013 validation, and 2014--2025 test. Test metrics do not select model settings, calibrators, or thresholds in the primary benchmark.
- Predictors use issue-date ERA5 daily summaries and lags from the preceding seven calendar days. The retrospective nature of same-day ERA5 summaries is disclosed; this does not simulate an archived operational forecast.
- The primary benchmark does not perform data-driven feature selection. Estimators (including any scaler inside an estimator pipeline) are fitted by the model-fitting procedure on training data; calibration and decision thresholds are determined from validation predictions.
- In each spatial fold, the held-out 5-degree blocks and the specified 1-degree neighboring buffer are excluded from training and validation. The geographic result is conditional transfer where the issue-day label is observed; it is not zero-shot geographic transfer.
- The onset redesign's ERA5 anomaly normals are computed from predictor years in the corresponding training fold only. The finalized model uses 1981--2007 for fitting and 2008--2013 for calibration and threshold selection. Its 2014--2025 and spatial-fold comparisons are explicitly exploratory because the earlier benchmark results were already inspected.
- Before acquiring or scoring 2026 labels, the onset estimator, features, 1981--2007 ERA5 anomaly normals, Platt calibrator, and 2008--2013 threshold were frozen and hashed. The 2026 evaluator checks the frozen artifact digest, IMD daily-file SHA-256 values, ERA5 per-cell SHA-256 values, and pinned snapshot ID. It runs inference only; the new outcomes do not select or alter the model.
- The independent 2026 issue dates (1 March--29 June) do not overlap training, calibration, or prior test dates. The fixed 1981--2010 normal still defines labels, and the predictors remain retrospective ERA5; this is out-of-time evaluation, not an independent-source or operational forecast test.
- Missing/undefined targets remain unlabelled and are excluded from binary metrics. The annual count of known issue rows, observed cells, events, and prevalence is retained in `reports/generated_era5/annual_target_coverage.csv`.

## Target-normal qualification

The primary daily labels use an 1981--2010 Tmax normal, so the reference climatology includes 2008--2010 observations that lie in the chronological validation period. This is part of the retrospective target definition; it is not a predictor, model input, or fitted model parameter. It does mean that the training labels are not a simulation of labels that could have been finalized by the end of 2007. Alternative normal periods (1981--2000 and 1991--2010) were evaluated and alter the target prevalence and event count. This qualification is now stated in Methods and should remain prominent in any interpretation.

## Scope and remaining checks

The 2026 frozen IMD-grid holdout includes development-fitted climatology and persistence comparators scored on the same onset rows; it supplies one season of out-of-time evidence but does not establish year-to-year generalization, prospective label availability, or operational forecast skill. A separate CPC 2026 Tmax product evaluation is documented in `reports/independent_cpc_2026/`. CPC's GTS reports may overlap with IMD inputs, so it is not guaranteed raw-source-independent validation; the CPC target also reuses the primary IMD normal. The 2026 ERA5 window covers all 275 cells; a fresh full ERA5 reacquisition for the original 1981--2025 primary benchmark was not performed, and the earlier sample audit covers one cell and 129 dates. A qualified meteorology/climate-science reviewer has not reviewed the labels or interpretation.

\n