# Frozen onset-model protocol (27 October 2026)

## Status and scope

This record freezes the onset-focused model for any subsequent evaluation. The original 2014-2025 temporal test and five geographic test folds had already been inspected before the redesign was selected. The reported comparisons on those periods remain exploratory. This freeze prevents subsequent adaptation to a newly obtained evaluation dataset; it does not retroactively make the existing holdouts independent.

## Target and input data

- Positive case: `current_heatwave == 0` on the issue day and `target_h1 == 1` on the next calendar day.
- Predictor source: the saved ERA5 daily feature frame and Icechunk snapshot `TZ2KX6A24CJ5DG17T7MG`.
- Grid and label: existing IMD-aligned 1-degree cell labels and study mask. No label rule changes are permitted for the frozen evaluation.
- Development/model-fit period: 1981-2007.
- Calibration and operating-threshold period: 2008-2013.
- Previously inspected temporal and geographic test periods are excluded as independent confirmation.

## Frozen features and model

Use the existing meteorological lag and context features plus the six grid-cell T2m-maximum anomaly/trend features produced by `add_temperature_anomaly_features`. Day-of-year normals are grid-specific circular 31-day means fitted on ERA5 predictor observations from 1981-2007 only. Use the selected HGB configuration: `max_leaf_nodes=15`, `min_samples_leaf=200`, `l2_regularization=10`, `learning_rate=0.08`, `max_iter=150`, `early_stopping=False`, and `random_state=42`.

The selected hyperparameters came from the eight-point grid in the onset benchmark, ranked by mean onset AP over these expanding development folds: 1981-1995/1996-2001, 1981-2001/2002-2007, and 1981-2007/2008-2013. The chosen configuration had mean validation AP 0.1708727892. For the final temporal model, fit the estimator on all state-zero training rows from 1981-2007.

## Frozen calibration and decision rule

Fit the Platt calibrator on state-zero validation rows from 2008-2013. Keep the recorded threshold **0.014854158899725712**, selected on those same validation rows to target a maximum false-alarm rate of 0.10. Do not recalibrate, retune the threshold, change features, or select a different model after inspecting any new evaluation labels. Report the attained false-alarm rate even if it exceeds 0.10.

## Saved artifact and reproducibility

- Serialized fitted estimator, calibrator, normals, feature list, selected parameters, and threshold: `reports/onset_model_era5/onset_model_frozen.joblib`.
- SHA-256: `c93db74fef286a23bddc392442b611c97ce7c7d004c46deb9232f4ccea4970a1`.
- Reconstruction command: `uv run --extra era5-cloud python scripts/freeze_onset_model.py`.
- The reconstruction script verifies that the refitted threshold matches the recorded threshold within 1e-12 before writing the artifact.

Code SHA-256 values at freeze:

| File | SHA-256 |
|---|---|
| `src/heatwave_benchmark/transitions.py` | `6de779314a6ffc0b30a3eed55b8447375f51ada6c62a3bc4b1088657ad063810` |
| `src/heatwave_benchmark/features.py` | `deaa3c1584080dcee3d8bb97b5c802472b404802fac5fc08bd500ec85adf5b49` |
| `scripts/run_transition_benchmark.py` | `744753cef14e692dc82749745daf7d666d04a0faa414e219720fec1484680d98` |
| `scripts/freeze_onset_model.py` | `ed685406f163edbedcd90af2bf05df873c1212134ab8c912c8d3508d059bdcdc` |

## Interpretation rule

Any result on a newly acquired period or independent dataset must be reported as a frozen-model evaluation and must not be used to revise this model before reporting. If a scientific or data-compatibility issue requires a change, declare that evaluation exploratory and establish a new untouched test set before making confirmatory claims.
