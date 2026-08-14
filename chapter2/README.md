# Chapter 2 Hindmarsh–Rose dynamics diagnostics

This directory is isolated from the completed Chapter 1 implementation. It
contains deterministic Hindmarsh–Rose simulation, waveform measurements,
dynamical diagnostics, isolated Echo State Network mechanics and evaluation
metrics, the small mechanics pilot, leakage-safe validation-only Bayesian
optimisation, and the locked Step 8 final-training and untouched-benchmark
workflow. Chapter 1 files are never modified.

## Numerical definition

The equations and classical RK4 step reproduce `data_loader.py` without
modifying it. The fixed parameters are `a=1`, `b=3`, `c=1`, `d=5`,
`r=0.006`, `s=4`, and `x_r=-1.6`, with initial state `[-1, -3, 3]` and
`dt=0.01`.

Each fixed-current calculation discards **100,000 steps (1,000 model-time
units)** and then retains 100,000 samples. The continuous calculation applies
that transient once under `I=1.67`, then follows
`1.67 -> 3.29 -> 3.50 -> 3.34 -> 3.20` for 100,000 samples per segment. Only
`I` changes at a boundary; `(x,y,z)` is never reset. The continuous-current
dataset is a predefined supporting transition benchmark. It is not the primary
unseen-current generalisation test. The fixed unseen currents `I=3.29` and
`I=3.34` remain the primary generalisation test.

## Half-window consistency check

For each fixed current, the first and second halves of the retained trajectory
are compared using:

- state-mean changes normalized by the full retained state standard deviation;
- state-standard-deviation changes relative to the full retained standard
  deviation;
- the relative change in mean interspike interval.

A current is reported as `consistent` only when every available measurement is
at or below the documented 10% tolerance. `consistent` means only that the two
retained halves have similar measurements. `inconsistent` does not by itself
show that the discarded transient was insufficient: chaotic fluctuations or
incomplete burst cycles can also produce differences. Insufficient spike
evidence produces `uncertain`.

## Spike, burst, and Lyapunov methods

Spikes are detected as local maxima of `x` with height at least `0.0`,
prominence at least `0.5`, and a minimum distance of 20 integration steps
(`0.2` model-time units).

Burst separation is adaptive. The largest adjacent gap in sorted log-ISI values
is considered as the within-burst/between-burst split. The gap must be at least
`0.15` log units, at least four times the median other positive log gap, and leave at
least two intervals on each side. Every accepted burst contains at least two
spikes. A regular tonic train is not split; ambiguous separation is recorded as
uncertain. Periodic bursting is assessed from within-burst ISI regularity,
inter-burst regularity, and consistency of spikes per burst, not overall ISI
coefficient of variation.

Largest Lyapunov exponents reuse the validated Benettin tangent-linear RK4
implementation in `scripts/analysis/estimate_hr_lyapunov.py` without modifying
that Chapter 1 file. Each current uses a 100,000-step transient, 500,000
evaluation steps, and tangent renormalization every 10 steps. Running estimates
are retained at 100,000-step intervals. The estimate is converged when both
consecutive changes among the last three checkpoints are within the larger of
`0.0005` and 20% of the final estimate. Very small or unconverged estimates are
treated cautiously.

## ESN mechanics milestone

`esn_model.py` implements deterministic parameter-aware four-input and ordinary
three-input ESNs with a three-state readout. It provides streaming ridge fitting,
teacher-forced warm-up, one-step prediction, autonomous recursive rollout,
explicit reservoir reset, state-continuous supplied-current switching, and
pickle-free NPZ save/load. The readout feature is `[1, input, reservoir_state]`,
matching the compatible Chapter 1 convention while allowing the input and output
dimensions to differ. The synthetic unit tests verify mechanics only; they do
not establish biological prediction accuracy.

## Evaluation metrics and small pilot

`esn_metrics.py` computes aligned per-state and all-state RMSE/NRMSE,
pointwise normalised error, valid prediction time, and explicit divergence
status. Reporting-only diagnostics include per-state and macro R² and Pearson
correlation, prediction and target standard deviations, their ratios, and
collapse flags at the locked ratio threshold `0.05`. Undefined constant-state
diagnostics use `None` plus explicit defined flags and contributor counts.
Normalisation scales are caller-supplied so validation data cannot silently fit
its own metric scale. Thresholds are explicit and recorded with every result.

`run_esn_pilot.py` runs the one authorized small real-data pilot. It uses the
first 5,000 fitting transitions from each training current, a 30-unit
parameter-aware reservoir, the complete 2,000-transition warm-up from validation
window 1 at `I=3.20`, and the first 1,000 scored transitions. The pilot result
is stored in `pilot_results/step5_real_data_pilot.json`. Its configuration and
thresholds are provisional mechanics settings, not selected thesis values, and
its seen-current result makes no claim about unseen-current generalisation.

## Step 7 validation-only Bayesian optimisation

`esn_optimisation.py` and `run_bayesian_optimisation.py` implement independent
Gaussian-process expected-improvement searches for the parameter-aware and
ordinary models. Each receives 40 calls (10 initial random calls), the same six
hyperparameter dimensions, all 40,000 fitting transitions per training current,
and the same nine validation rollouts. Fitting washout is 2,000 transitions per
independent trajectory; every validation window resets, warms for 2,000 true
transitions, then scores 8,000 recursive predictions.

The candidate seed is 42 and the optimiser seeds are 2026 (parameter-aware) and
2027 (ordinary). The objective is the equal arithmetic mean of the nine
all-state NRMSE values. After each search, its five best seed-42 candidates are
confirmed over seeds `(42, 123, 456, 789, 2026)` and selected by the frozen
robust tie-break. Strict atomically replaced JSON checkpoints support resume.

The validation-only histories and combined selection are stored under
`optimisation_results/` and retain their original
`VALIDATION-SELECTED — BENCHMARKS NOT OPENED` labels. Step 8 consumes those
artifacts without reopening optimisation or changing the selected trials.

## Step 8 locked final evaluation

The historical execution used `esn_step8.py` and `run_step8.py` for
fail-closed preflight, selection/evaluation locks, prefix-only final training,
ten pickle-free model bundles, and fixed/continuous recursive benchmarks. Final
training used transitions `[0, 70000)` at `I=(1.67, 3.20, 3.50)`, a
2,000-transition washout for each reset trajectory, and seeds
`(42, 123, 456, 789, 2026)`. Scaling was fitted only from those training
prefixes. All ten models were saved and round-trip verified before held-out,
unseen-current, or continuous numerical arrays were opened.

The approved fixed-current windows start at `(70000, 80000, 89999)`. Exact
warm-up/forecast intervals and the pre-benchmark dataset-length correction are
documented in `EXPERIMENT_PROTOCOL.md`. The frozen result set contains exactly
210 records. Known-current, unseen-current, long-horizon, and continuous
families remain separate, and every divergent record remains in the summaries.

Step 7 and Step 8 are complete. Do not rerun either for repository cleanup,
release verification, documentation changes, or Git publication. Their Python
and SLURM entry points are retained only as historical scientific-reproduction
workflows. The current `esn_step8.py` includes post-evaluation resume
hardening and is not the exact execution source; see `PROVENANCE.md`.

## Result interpretation

The five final tables report mean, population standard deviation, median,
worst case, valid-prediction summaries, divergence/collapse counts, and event
contributor counts as applicable. They do not contain IQR columns.

The parameter-aware ESN achieved substantially lower typical prediction errors
for stable reservoir initializations and responded to changing external
currents. However, seed-dependent divergence increased its mean error above the
ordinary baseline, demonstrating improved typical accuracy but insufficient
robustness across reservoir initializations.

In the frozen aggregate table, the parameter-aware mean NRMSE is worse than the
ordinary baseline in all five families. Its divergence rate is 20% in the four
fixed-current families and 40% in the continuous family; the ordinary baseline
has no recorded divergence. Seed 456 and every associated record remain
included. These results do not support unconditional parameter-aware
superiority.

## Post-benchmark correction and audit

A post-run audit found six divergent continuous-current interval event
subrecords that incorrectly remained defined. The deterministic correction
changed only event-validity metadata. Forecasts, state metrics, models, scalers,
locks, optimisation, aggregates, seeds, and conclusions did not change.
Correction evidence is preserved under
`final_results/post_benchmark_event_correction/`.

`audit_step8.py` is non-mutating with respect to primary predictions, model
bundles, and scientific aggregates. It can, however, rewrite derived audit JSON
and the five generated final CSV tables, so it is not filesystem read-only.
Future writes are atomic, but the live audit must not be run during cleanup.
The saved audit is historical and remains unchanged. Any future audit output
must use a new versioned filename and record the producing audit-source hash.

The saved audit verdict is
`AUDIT PASSED — SCIENTIFIC SEED INSTABILITY PRESERVED`. It reports no generic
implementation defect and attributes the preserved failures to genuine
closed-loop reservoir-seed instability. The saved audit and current audit
source differ in their legacy figure reporting; this limitation is documented
in `PROVENANCE.md`.

## Official thesis figures

The only official Chapter 2 figure directory is
`final_results/figures_thesis/`, containing exactly four PDFs and four
matching PNGs:

1. `01_fixed_current_predictions_known_and_unseen`
2. `02_continuous_current_prediction`
3. `03_current_transition_tracking`
4. `04_overall_predictive_performance`

Figures 1--3 use the predetermined seed 42; Figure 1 uses locked window 1.
Figure 4 includes all five seeds and every divergent record. Historical JSON
references to removed `final_results/figures/` and
`final_results/figures_final/` directories are intentionally preserved as
provenance. Those directories are superseded and must not be restored or
treated as release figures.

Do not regenerate the official figures merely for cleanup or appearance. Their
complete hashes and the existing figure-manifest hash are frozen in
`release/release_manifest.json`.

## Release, provenance, and artifact policy

- `PROVENANCE.md` distinguishes the missing exact execution source, the
  byte-preserved post-correction/pre-hardening source, and current hardened
  release source.
- `REPRODUCIBILITY.md` records the scientific environment, seeds, clean test
  environment, and the difference between reproduction and verification.
- `ARTIFACT_DISTRIBUTION.md` classifies all original 288 status paths and
  documents ordinary Git versus LFS/external archive policy.
- `release/release_manifest.json` is the versioned strict-JSON release
  inventory.

The 210 raw rollout arrays consume approximately 682 MiB on this HPC
filesystem (355,396,721 apparent bytes). They remain visible in `git status`
until an explicitly approved Git LFS or external archive decision is made.
Never use a broad `*.npz` ignore rule. Do not commit a repository ZIP/TAR.

## Safe verification

Run all commands from the repository root. The release verifier is
read-only by default and does not load a model for prediction, train, evaluate,
audit, or generate figures:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD" \
python3 -m chapter2.verify_release
```

Run the test suites with:

```bash
python3 -m compileall -q chapter2

PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD" \
python3 -m pytest -q chapter2/tests

PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD" \
python3 -m pytest -q
```

Chapter 2 deliberately has no `data_loader.py`. Its standalone simulation
implementation is `chapter2/hr_data_ch2.py`. One compatibility test imports
the repository-root Chapter 1 `data_loader.py`; this is why the root must be
on `PYTHONPATH`. It is a compatibility dependency, not a missing Chapter 2
file.
