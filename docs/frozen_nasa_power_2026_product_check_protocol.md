# Frozen-model NASA POWER product-check protocol (28 October 2026)

## Purpose and status

This is an exploratory, product-level robustness check of the onset model frozen in `docs/frozen_onset_protocol_2026-10-27.md`. It uses NASA POWER daily maximum 2-m temperature for March--June 2026 and the same 275 study grid-cell centers. NASA POWER's meteorological series is based on MERRA-2 historically and a near-real-time GEOS-IT extension; it is a distinct analysis product, not independent station observations. The product check was selected after inspection of the 2026 IMD and CPC results, so it is not a confirmatory, preregistered test.

## Frozen model and eligible rows

- Use the existing all-cell 2026 probability file `reports/independent_onset_2026/frozen_model_all_grid_scores_2026.csv.gz` and the already frozen estimator artifact recorded in `docs/frozen_onset_protocol_2026-10-27.md`.
- Do not refit the estimator, calibration, or threshold; do not change predictors, features, issue dates, grid, or model settings.
- Use the HGB probability and frozen threshold `0.014854158899725712`. Keep the saved historical climatology probability as a same-row reference only; do not refit or recalibrate it to NASA POWER outcomes.
- An eligible row has a known NASA POWER issue-day grid-cell label of 0 and a known next-day label. A positive target is a 0-to-1/2 transition in the POWER-derived labels.

## Product-derived target construction

1. Download NASA POWER `T2M_MAX` from the Daily Point API for each study location and dates 1 March--1 July 2026, requesting local-solar-time (`LST`) values and retaining the complete raw JSON response, endpoint, API metadata, returned coordinates, and SHA-256 in an acquisition manifest.
2. From the already acquired NASA POWER archive `data/interim/nasa_power_1981_2025.nc`, calculate each requested location's 1981--2010 daily Tmax normal with the same centered circular 15-day method and 10-year minimum support used by the primary target.
3. Apply the existing terrain-entry and heatwave/departure thresholds without changing them, replacing only daily Tmax and its normal with the NASA POWER product values. Treat the resulting labels as an alternative product target, not as official IMD declarations or observed ground truth.
4. Apply the fixed model probabilities to eligible rows and report event prevalence, average precision, Brier score, precision, recall, and false-alarm rate at the frozen threshold. Estimate paired score-difference uncertainty using the existing seven-day date-block bootstrap where feasible.

## Interpretation rules

This evaluation can assess consistency against a distinct gridded analysis product. It cannot establish independent raw-observation validation because reanalyses may assimilate overlapping observations, and POWER daily values use local solar time rather than the IMD observing window. It also tests only one season and uses retrospective ERA5 predictors. Any results remain exploratory, must be kept separate from the primary IMD evaluation, and must not be used to tune the model.
