# Chapter 2 Hindmarsh–Rose dynamics diagnostics

This directory is isolated from the completed Chapter 1 implementation. It
contains deterministic Hindmarsh–Rose simulation, waveform measurements,
half-window checks, and preliminary dynamical classifications. No Echo State
Network is created, trained, optimized, or evaluated at this stage.

## Numerical definition

The equations and classical RK4 step reproduce `data_loader.py` without
modifying it. The fixed parameters are `a=1`, `b=3`, `c=1`, `d=5`,
`r=0.006`, `s=4`, and `x_r=-1.6`, with initial state `[-1, -3, 3]` and
`dt=0.01`.

Each fixed-current calculation discards **100,000 steps (1,000 model-time
units)** and then retains 100,000 samples. The continuous calculation applies
that transient once under `I=1.67`, then follows
`1.67 -> 3.29 -> 3.50 -> 3.34 -> 3.20` for 100,000 samples per segment. Only
`I` changes at a boundary; `(x,y,z)` is never reset. The continuous signal is
diagnostic and is not a locked final test dataset.

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

## Generate diagnostics

From the repository root:

```bash
PYTHONPATH="$PWD" python3 chapter2/generate_diagnostics.py
```

Regenerate the analysis tables and manifest from the existing NPZ trajectories
without rewriting datasets or Figures 1--3 with:

```bash
PYTHONPATH="$PWD" python3 chapter2/generate_diagnostics.py --analysis-only
```

The script writes deterministic NPZ datasets with named `t`, `x`, `y`, `z`,
and `I` arrays; descriptive statistics; CSV and Markdown dynamics summaries;
exactly four diagnostic PNG figures; and a SHA-256 manifest under
`chapter2/outputs/`.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD" \
python3 -m pytest -q chapter2/tests
```

Run the unchanged Chapter 1 suite with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD" \
python3 -m pytest -q tests
```
