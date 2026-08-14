# Chapter 2 frozen experimental protocol

## Scientific question and prediction task

This chapter asks:

> Can a parameter-aware Echo State Network, trained using Hindmarsh--Rose time
> series at selected current values, predict the neuron's next state and
> recursively generate its future time series for a supplied current `I`,
> including currents not used during training?

At transition index `t`, the parameter-aware model will map

```text
[x_t, y_t, z_t, I_t] -> [x_hat_(t+1), y_hat_(t+1), z_hat_(t+1)].
```

The three neuronal states `(x, y, z)` are predicted. The applied current `I_t`
is known, supplied to the model, and never predicted. An ordinary-ESN baseline
will use `[x_t, y_t, z_t]` as input while retaining the same three-state target.
This is parameter-aware reservoir computing, not NGRC/NVAR.

## Frozen current allocation

- Training currents: `I = (1.67, 3.20, 3.50)`.
- Fixed unseen currents: `I = (3.29, 3.34)`.
- Continuous sequence: `I(t) = (1.67, 3.29, 3.50, 3.34, 3.20)`, with
  switches at state indices `(100000, 200000, 300000, 400000)` and no state
  reset.
- Final reservoir seeds: `(42, 123, 456, 789, 2026)`.

The two unseen fixed currents are the primary generalisation test. The
continuous-current dataset is a predefined supporting transition benchmark. It
is not the primary unseen-current generalisation test. It tests whether a
locked model follows changes in supplied `I(t)` without resetting its state.
Neither unseen fixed data nor continuous data may be exposed through the
optimisation-data or scaler-fitting APIs.

## Frozen chronological transition split

A transition at index `k` is the pair

```text
[x_k, y_k, z_k, I_k] -> [x_(k+1), y_(k+1), z_(k+1)].
```

Ranges below are half-open ranges of transition indices, not raw state-array
ranges:

| View | Transition indices | Count for 100,000 states |
|---|---:|---:|
| Fitting | `[0, 40000)` | 40,000 |
| Validation | `[40000, 70000)` | 30,000 |
| Held out | `[70000, N-1)` | 29,999 |

A trajectory of `N` states contains only `N-1` next-state pairs. Consequently,
the held-out view of a 100,000-state trajectory has 29,999 transitions, not
30,000. No transition is present in more than one split.

## Frozen validation windows

Each window has 10,000 transitions and therefore consumes 10,001 state
samples. Adjacent windows may share their single boundary state, but never a
transition or target pair.

| Window | Full transition range | True-state warm-up | Autonomous scored rollout |
|---|---:|---:|---:|
| 1 | `[40000, 50000)` | `[40000, 42000)` | `[42000, 50000)` |
| 2 | `[50000, 60000)` | `[50000, 52000)` | `[52000, 60000)` |
| 3 | `[60000, 70000)` | `[60000, 62000)` | `[62000, 70000)` |

Every window contains 2,000 warm-up transitions and 8,000 scored transitions.
The three training currents will therefore produce nine validation rollouts.
These windows are intended for state error, valid prediction time, and
divergence. They are too short for reliable burst-count optimisation;
long-horizon spike and burst behaviour is assessed only after model locking.

## Frozen scaling policy

- Fit one shared state standardiser for `(x, y, z)` using state inputs from the
  fitting transitions `[0, 40000)` of all three training currents only.
- Fit one shared current standardiser for `I` using those same permitted
  fitting-transition inputs.
- Apply the identical state transformation to state inputs and next-state
  targets. This makes a predicted scaled state suitable for recursive feedback
  under the same convention.
- Apply the current transformation only to the supplied current column.
- Do not fit on validation, held-out, unseen-current, or continuous data.
- Do not fit scalers at import time and do not persist scalers at this
  checkpoint.

## Leakage prevention and later locking order

1. Construct one-step pairs inside each fixed-current trajectory before any
   concatenation. Never join the final state of one current to the first state
   of another.
2. Fit preprocessing statistics only from fitting transitions of the three
   training currents.
3. A later optimisation stage may use fitting transitions and the nine frozen
   validation rollouts from training currents only. It may not inspect unseen,
   held-out, or continuous benchmark results.
4. Select and then lock preprocessing, ESN hyperparameters, seed protocol,
   training procedure, and evaluation rules before opening any benchmark.
5. After locking, evaluate without further selection or tuning. Fixed unseen
   currents are the primary generalisation test; training-current held-out
   views and the continuous transition benchmark provide separate supporting
   tests. Benchmark outcomes must not cause the model or protocol to be
   revised.
6. The continuous rollout supplies the predefined `I(t)` sequence at every
   step and keeps the model state continuous across current switches.

## Rollout evaluation metrics

Metrics compare each autonomous prediction `s_hat_(t+1)` with the aligned
target `s_(t+1)`. Let `sigma_j` be the positive state scale for state
`j in {x,y,z}`, supplied from fitting-data preprocessing rather than fitted
on the evaluated rollout.

- Per-state RMSE is
  `sqrt(mean_t((s_hat_(t,j) - s_(t,j))^2))`.
- All-state RMSE is the square root of the mean squared error over all scored
  times and all three states.
- Per-state NRMSE divides each state error by its supplied `sigma_j`.
- All-state NRMSE is the square root of the mean squared normalized error over
  all scored times and states.
- Pointwise normalized error is
  `e_t = sqrt(mean_j(((s_hat_(t,j) - s_(t,j)) / sigma_j)^2))`.
- Valid prediction time is the length of the consecutive scored prefix for
  which `e_t` remains strictly below an explicitly supplied threshold,
  multiplied by `dt`. Failure on the first prediction gives zero valid
  prediction time.
- Divergence is the first non-finite prediction row or the first row at which
  `e_t` reaches an explicitly supplied divergence threshold. Its zero-based
  index, elapsed time `(index + 1) * dt`, and reason are recorded.

- Per-state R² uses the target-state sum of squares; macro R² averages only
  defined states and records how many states contributed.
- Per-state Pearson correlation and its macro use the same defined-state rule.
- Prediction and target standard deviations, their per-state ratios, and
  prediction-collapse flags are reporting diagnostics only. Collapse means a
  defined ratio strictly below `0.05`.
- Constant targets make R², correlation, and the standard-deviation ratio
  undefined; strict JSON stores `None` with explicit defined flags.

Metric normalization scales and the valid-prediction, divergence, and collapse
thresholds must be supplied explicitly and stored with results. They must be
locked before later optimisation or benchmark evaluation.

## Step 5 small real-data pilot

The single mechanics pilot uses only permitted training-current data:

- parameter-aware inputs and three-state targets;
- the first 5,000 fitting transitions from each of `I=(1.67, 3.20, 3.50)`;
- 100 fitting washout transitions per independent current trajectory;
- one 30-unit reservoir with seed 42 and explicitly recorded provisional
  hyperparameters;
- the full `[40000, 42000)` true-state warm-up at seen current `I=3.20`;
- the first 1,000 scored transitions `[42000, 43000)`;
- fitting-data state standard deviations for metric normalization;
- provisional pilot thresholds 0.4 for valid prediction and 5.0 for divergence.

The result is recorded in
`chapter2/pilot_results/step5_real_data_pilot.json`. This pilot is not
hyperparameter selection and does not access held-out, unseen-current, or
continuous-benchmark arrays. Its outcome must not be presented as biological
accuracy or unseen-current generalisation evidence.

## Step 7 leakage-safe Bayesian optimisation

Step 7 uses only fitting `[0, 40000)` and validation `[40000, 70000)`
transitions from `I=(1.67, 3.20, 3.50)`. Its data API constructs no held-out
view and never calls the unseen-current or continuous-benchmark loaders.

The parameter-aware and ordinary models receive independent, equal searches:

| Hyperparameter | Search space | Prior |
|---|---:|---|
| Reservoir size | `{100, 200, 300}` | categorical |
| Reservoir connectivity | `[0.01, 1.00]` | uniform |
| Input scaling | `[0.01, 3.00]` | uniform |
| Spectral radius | `[0.01, 3.00]` | uniform |
| Ridge regularisation | `[1e-10, 1e-2]` | log-uniform |
| Leak rate | `[0.01, 1.00]` | uniform |

Both use fixed bias scaling `0.1`, an unregularised bias, three outputs, and
2,000 fitting washout transitions per independently reset current trajectory.
Each Gaussian-process expected-improvement search has 40 calls, 10 initial
random calls, and candidate reservoir seed 42. Optimiser seeds are 2026 for the
four-input parameter-aware model and 2027 for the three-input ordinary model.

Every candidate is scored on the arithmetic mean of all-state NRMSE across the
nine equally weighted validation rollouts. Each independently reset rollout
uses exactly 2,000 teacher-forced transitions followed by 8,000 recursive
predictions. The locked valid-prediction and divergence thresholds are `0.4`
and `5.0`; numerical rollout failure scores `1_000_000.0`. R², correlation,
valid prediction time, divergence, collapse, spikes, and bursts are excluded
from the objective.

After each 40-call search, the five lowest seed-42 candidates are confirmed on
all five final seeds. Selection minimizes mean NRMSE over 45 rollouts, then
breaks ties by lower worst-current mean NRMSE, higher mean valid-prediction
steps, and lexicographically serialized hyperparameters.

Histories are atomically checkpointed as strict JSON after every completed
candidate and confirmation seed. The required artifacts are:

- `optimisation_results/step7_parameter_aware_history.json`;
- `optimisation_results/step7_ordinary_baseline_history.json`;
- `optimisation_results/step7_selection.json`.

The selection is validation-only and labelled
`VALIDATION-SELECTED — BENCHMARKS NOT OPENED`. Step 7 saves no candidate or
final weights and creates no `selected_model.json`.

## Frozen dataset inventory

All paths are relative to the repository root. The inspected archive key order
is `(t, x, y, z, I)`. Every listed array is one-dimensional, numeric, finite,
and has dtype `float64`; `t` is strictly increasing with `dt = 0.01`.

| Dataset path | Current or sequence | Keys, shapes, and dtypes | States | SHA-256 |
|---|---|---|---:|---|
| `chapter2/outputs/data/fixed_I_1p67.npz` | `I = 1.67` | `t,x,y,z,I`: each `(100000,)`, `float64` | 100,000 | `cc89af5e9a27d05a9501ea995a2ef361c146d6513ec1b19765238603af9ffb2b` |
| `chapter2/outputs/data/fixed_I_3p20.npz` | `I = 3.20` | `t,x,y,z,I`: each `(100000,)`, `float64` | 100,000 | `c4292a1e0fa5575d08419e7d302f980e2ed085878187f67659aa59decc486ded` |
| `chapter2/outputs/data/fixed_I_3p29.npz` | `I = 3.29` | `t,x,y,z,I`: each `(100000,)`, `float64` | 100,000 | `2ed394f457e5de5f0d14a5dd450611fd9774c02cb13fc68f7b21a2621273b7b4` |
| `chapter2/outputs/data/fixed_I_3p34.npz` | `I = 3.34` | `t,x,y,z,I`: each `(100000,)`, `float64` | 100,000 | `8cad49948c553df9b65a24deacf067dc999325dadb580a4787ded12b6315585d` |
| `chapter2/outputs/data/fixed_I_3p50.npz` | `I = 3.50` | `t,x,y,z,I`: each `(100000,)`, `float64` | 100,000 | `4d042da6adbde026468207c4e9f74443f8587c635dff2d28b69e2b1784d5cbbf` |
| `chapter2/outputs/data/continuous_switched_currents.npz` | `(1.67, 3.29, 3.50, 3.34, 3.20)`; switches `(100000, 200000, 300000, 400000)` | `t,x,y,z,I`: each `(500000,)`, `float64` | 500,000 | `0921cbad321da1830433dc84e58f25ff2b0b6a6d571a31cae769bdfd6dc00a7b` |

These six existing datasets are frozen inputs. They must be verified by hash
when loaded and must not be regenerated, rewritten, renamed, or modified by
data preparation.
## Step 8 final training and untouched evaluation

Step 8 permanently locks the corrected Step 7 selections before any benchmark
numerical value is read. The parameter-aware model uses source trial 4 and the
ordinary baseline uses source trial 12. Both use final reservoir seeds
`(42, 123, 456, 789, 2026)`; every seed is an equal scientific repetition and
benchmark performance cannot change selection, settings, preprocessing, or
evaluation.

Final readouts are fitted independently for each model and seed using only
transitions `[0, 70000)` from `I=(1.67, 3.20, 3.50)`. Each independently
reset trajectory has a 2,000-transition washout. Shared state/input/output
standardization is fitted from the three training prefixes only; the aware
current standardizer is also fitted only from their current inputs. The ten
pickle-free NPZ model bundles are saved and round-trip checked before benchmark
access.

### Final fixed-current short windows

The initially proposed starts `(70000, 100000, 130000)` were corrected before
benchmark access using only raw-byte hashes and NPZ headers, shapes, dtypes, and
transition counts. No held-out, unseen-current, or continuous numerical value
was inspected. Each frozen fixed dataset contains 100,000 states, hence 99,999
transitions, and its held-out range `[70000, 99999)` contains only 29,999
transitions. The final approved starts are `(70000, 80000, 89999)`:

| Window | Teacher-forced warm-up | Recursive scored forecast |
|---|---:|---:|
| 1 | `[70000, 72000)` | `[72000, 80000)` |
| 2 | `[80000, 82000)` | `[82000, 90000)` |
| 3 | `[89999, 91999)` | `[91999, 99999)` |

Transition 89,999 is scored in window 2 and appears only in the unscored
teacher-forced warm-up of window 3. The three 8,000-transition scored forecast
intervals therefore do not overlap. These windows apply identically to known
currents `(1.67, 3.20, 3.50)` and unseen currents `(3.29, 3.34)`.

Long-horizon fixed-current evaluation resets at transition 70,000, teacher
forces `[70000, 72000)`, and recursively forecasts `[72000, 99999)` without
another reset. The continuous benchmark resets once, teacher forces only
`[0, 2000)`, and recursively forecasts `[2000, 499999)`; it never resets or
warms at current changes. The aware transition input is exactly
`[predicted_state_t, I_t]`, while the baseline never receives current.

The five benchmark families remain separate: known short windows, unseen short
windows, known long horizon, unseen long horizon, and continuous changing
current. Short-window records receive equal seed/current/window weight. Failed,
divergent, or collapsed repetitions remain in all counts and aggregates;
undefined nonfinite NRMSE contributes the frozen failure score
`1_000_000.0`.

State metrics retain the Step 7 definitions and thresholds: valid prediction
threshold `0.4`, divergence threshold `5.0`, and collapse standard-deviation
ratio threshold `0.05`. Long and continuous trajectories additionally use the
already documented Chapter 2 physical-state spike and adaptive log-ISI burst
definitions. Chapter 1's 20-step greedy one-to-one spike matching convention is
used for spike-time error; Chapter 1 is not modified.

Run the resumable Slurm workflow from the repository root with:

```bash
sbatch run_chapter2_step8.slurm
```

Immutable locks and results are written under `chapter2/final_results/`; ten
safe model bundles and their manifest are written under
`chapter2/final_models/`.
