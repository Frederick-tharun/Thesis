# Post-hoc cross-regime numerical correction

This correction reads the completed benchmark's stored predictions and targets.
It does not train models or generate forecasts. The original NPZ archives,
raw-results JSON, aggregates, status and frozen manifest remain unchanged.

From the repository root on `chapter2-cross-regime-parameter-aware`, with the
original HPC artifacts and execution status present, run:

```sh
python -m chapter2.correct_cross_regime_numerics
```

The command requires all 345 unique record identities and archives, verifies
each archive's stored SHA-256 before loading it, and checks target, time and
current arrays against the original source trajectories and evaluation windows.
Model, dataset and raw-result manifest locks are verified as well. It refuses
an existing output directory. No benchmark runner or model prediction method
is called.

Outputs are written separately under
`chapter2/cross_regime_results/post_hoc_numerical_correction/`:

- `corrected_results.json`: recomputed metadata referencing the original archives.
- `corrected_aggregate_results.json` and CSV tables: the existing aggregation
  formulas applied to corrected records.
- `correction_changes.json`: exact field changes, old/new classification and
  penalties, reasons, and aggregate changes. New provenance fields are counted
  separately from classification/penalty changes.
- `correction_manifest.json`: original input hashes, output hashes, source
  provenance, timestamp, audit results and explicit no-retraining/no-rerun claims.

The historical pointwise-error arrays remain in their original archives. A
separate `derived_pointwise_sha256` binds each corrected record to recomputed
pointwise errors (contiguous little-endian float64 bytes). The derived audit
recomputes this digest; it does not demand that corrected errors equal the old
archive's derived error array.

## Numerical policy

`cross_regime_numerics.evaluate_predictions` is the shared entry point for the
evaluator, correction and auditor. The policy is numerical, not tuned to results:
float64 representability and its rounding precision determine safety boundaries.

During forecasting, each scaled output is checked and converted to physical
units before storage. Scaled nonfiniteness or physical conversion overflow stops
feedback at that step; the physical valid prefix is preserved and the failed
step and suffix are NaN. Forecast generation never checks future targets.

After forecasting, the canonical evaluation failure is the earliest physical
nonfinite row or row whose residual subtraction, normalization, square, or
sum-of-squares accumulation cannot safely fit in float64. The failure reason
identifies the unsafe operation. `physical_failure_step` separately records
physical nonfiniteness; `failure_step` includes post-forecast metric unsafety.
Generation-specific reasons remain separate from this reproducible evaluation
classification.

Finite metric-unsafe predictions remain intact as evidence. A temporary metric
view has a NaN suffix from the canonical failure step. Whole-forecast error
metrics are undefined on failure, the existing `NONFINITE_FAILURE_SCORE`
(`1_000_000.0`) applies, and prefix VPT/divergence and event invalidation follow
the existing thresholds. Ordinary finite metrics retain their original formulas.
The shared general-purpose `esn_metrics.py` is unchanged.

The previously reported 37 records with finite residual squaring overflow fall
under this explicit numerical-failure policy, including the 18 previously
unclassified records. These counts are not hardcoded. The actual correction log
must establish the resulting counts from the preserved HPC artifacts.

## Two-stage provenance and audit

The original manifest's dirty freeze-time preflight HEAD predates implementation.
The original execution status's `implementation_commit` identifies the execution
source. Every frozen source SHA-256, including the complete original source-path
inventory, must match exact Git blob bytes at that recorded commit. No line-ending
normalization or comparison to repaired working files is substituted. Missing or
conflicting provenance fails closed.

The correction records its own HEAD, branch, dirty status and actual source-file
hashes, independently of the original execution lock. The normal pre-benchmark
source equality gate is unchanged. Reaudit the separate correction with either:

```sh
python -m chapter2.correct_cross_regime_numerics --audit-only
python -m chapter2.audit_cross_regime --post-hoc-correction chapter2/cross_regime_results/post_hoc_numerical_correction
```

The correction audit verifies both provenance stages, the original artifacts,
derived records, error digests, aggregates, change log and output hashes. Its
verdict states: "Original forecasts were produced under the frozen source lock;
numerical classification and derived metrics were corrected afterward without
rerunning the forecasts."
