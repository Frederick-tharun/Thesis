# Parameter-aware cross-regime experiment

This extension compares the effect of training-current coverage using the
previously selected Step 7 parameter-aware ESN architecture. No further
hyperparameter search or benchmark-driven tuning is performed. Original
Chapter 1 and Chapter 2 scientific evidence is retained unchanged.

## Frozen design

Five paired reservoir seeds (42, 123, 456, 789, 2026) are used in each scenario:

| Scenario | Training currents | Primary transfer targets |
|---|---|---|
| regular_to_chaotic | 1.67, 3.29, 3.50 | 3.20, 3.34 |
| chaotic_to_regular | 3.20, 3.34 | 1.67, 3.29, 3.50 |
| mixed_shuffled | all five | temporal holdouts on all five |

The regular/non-chaotic classification of I=3.29 is provisional: the existing
qualitative diagnostic labels it uncertain and its converged Lyapunov estimate
is near zero. Full numerical evidence is recorded in the protocol manifest.

Each scenario uses exactly 130,000 effective readout transitions. Raw prefixes
contain 45,334/45,333/45,333 transitions for regular training, 67,000 per current
for chaotic training, and 28,000 per current for mixed training. Each block has
its own reservoir reset and 2,000-transition washout. State and current scalers
fit only that scenario's training inputs. No artificial transition joins blocks.
Mixed block order is a complete-block permutation fixed by seed, never an
individual-sample shuffle. Because ridge statistics are summed across reset
blocks, block order can affect floating-point accumulation but does not model
a continuous switch or provide an independent shuffling treatment.

The frozen model has 100 reservoir units and four inputs (x, y, z, I), predicting
three next-state values. Exact hyperparameters and block orders are defined in
`cross_regime_config.py` and copied into the machine-readable manifest.

## Evaluation

All models evaluate all five fixed currents. Short windows begin at transition
70,000, 80,000 and 89,999, with 2,000 warm-up transitions and 8,000 recursive
predictions each. The long window warms on [70,000,72,000) and forecasts
[72,000,99,999). Windows describe transitions k -> k+1. The last short and long
windows end at state 99,999; target state 100,000 does not exist.

Three supporting continuous schedules each contain 100,000 samples per current:

- regular_then_chaotic: 1.67 -> 3.29 -> 3.50 -> 3.20 -> 3.34
- chaotic_then_regular: 3.20 -> 3.34 -> 1.67 -> 3.29 -> 3.50
- alternating_mixed: 3.50 -> 3.34 -> 1.67 -> 3.20 -> 3.29

The simulator discards one 100,000-step initial transient, then preserves state
across all current switches. Each ESN warms once on [0,2,000) and forecasts
[2,000,499,999), feeding back predictions and receiving the true supplied current.
There is no reset or true-state rewarming at switches.

Totals: 15 models, 225 short records, 75 long records, 45 continuous records;
345 evaluations and 363 NPZ artifacts including three schedule datasets.

## Metrics and interpretation

Stored metrics include physical RMSE, NRMSE, valid prediction time (threshold
0.4), divergence (threshold 5), collapse (standard deviation ratio below 0.05),
and spike/burst measurements. Undefined values remain explicit JSON nulls;
nonfinite rollouts remain in the record matrix with the existing Chapter 2
failure penalty. Divergent event measurements are invalidated. Continuous
boundary diagnostics cover 2,000 transitions on either side of each switch.

NRMSE and threshold-based metrics use each scenario's own training-only scale.
Their cross-scenario differences include normalization differences; physical
RMSE is available in each record/table and as a finite-only summary. A finite
summary must always be read alongside failure/divergence counts. Pooled means
mix horizons and training-current versus transfer tasks. The primary scientific
comparison is the fixed-current transfer subset, with mixed-trained models on
the same target currents as context. Paired pooled NRMSE differences and
20,000 bootstrap resamples over five seeds are descriptive only.

Architecture selection previously used mixed-regime validation data. This
experiment isolates final training-data coverage under that architecture; it
cannot establish that all design decisions used a single regime exclusively.
Previously evaluated Chapter 2 benchmark trajectories are reused for this
prespecified extension; they are not a newly collected independent test set.

## Validation and execution

Before benchmark execution, run the focused tests and existing Chapter 2 suite,
then the small synthetic pilot, then freeze the source/data/protocol manifest.
The freeze gate requires a passing pilot with matching source hashes. The full
run requires the designated branch, a clean committed tree, a Slurm allocation,
and unchanged original evidence (226 binary artifacts).

```bash
python -m pytest -q chapter2/tests
python -m chapter2.run_cross_regime --pilot
python -m chapter2.run_cross_regime --freeze
# Commit the implementation, protocol and manifest before submission.
sbatch run_chapter2_cross_regime.slurm
```

The launcher runs the benchmark and then the audit. It reserves a GPU to match
the cluster partition allocation policy; this NumPy implementation uses CPU
computation. One BLAS thread avoids threading overhead for small reservoirs.
Progress is written to `cross_regime_results/cross_regime_status.json` and the
Slurm log. Interrupted runs resume only with matching source, protocol, model,
dataset and raw-array hashes. A filesystem lock rejects concurrent execution.
Completed experiments refuse accidental reruns.

The audit checks the full matrix, source targets/time/current, model/scaler
provenance, metrics, failure penalties, transition diagnostics, aggregates,
paired summaries and original evidence before creating figures and marking
COMPLETE. It recomputes metrics using shared metric functions; it is not a
second independent implementation of the metrics. The artifact hash inventory
excludes mutable status and verification records to avoid circular/stale hashes.
Runtime arrays, reports and figures are ignored by Git; the protocol manifest
is committed. Existing binary evidence remains in its original locations.
