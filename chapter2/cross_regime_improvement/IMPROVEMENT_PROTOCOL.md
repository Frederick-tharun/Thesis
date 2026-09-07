# Chapter 2 cross-regime improvement protocol

## Scope and model lock

This extension leaves Chapter 1, the original Chapter 2 experiment, and the
completed 345-record cross-regime baseline immutable. It retrains and
scenario-optimises the existing parameter-aware `EchoStateNetwork` only:

```text
[x_t, y_t, z_t, I_t] -> EchoStateNetwork -> [x_(t+1), y_(t+1), z_(t+1)]
```

Input and output dimensions remain four and three. Current is supplied and is
never predicted. The SHA-256 of `chapter2/esn_model.py` is locked to
`130c0f0f1753c7429a37bf14dbfd49de5bc8a0e04741893ef0b360126d8edcd3`.

## Fixed regimes and scenarios

- Regular: `I=(1.67, 3.29, 3.50)`.
- Chaotic: `I=(3.20, 3.34)`.
- `regular_to_chaotic` optimisation may open only regular currents.
- `chaotic_to_regular` optimisation may open only chaotic currents.
- `mixed_shuffled` optimisation may open all five currents.

For `mixed_shuffled`, only the order of complete current blocks is permuted,
using the already frozen seed-specific orders. Samples remain chronological
inside every block. One-step pairs are constructed inside a block, never
across block boundaries, and `EchoStateNetwork.fit` resets its reservoir for
each independent block.

## Chronological optimisation views

Each scenario uses transition inputs `[0,40000)` for fitting. Scalers are fit
only on state/current inputs from these fitting views. Validation is confined
to the same scenario's currents and disjoint range `[40000,70000)`:

| Window | Teacher-forced warm-up | Recursive scored rollout |
|---|---:|---:|
| 1 | `[40000,42000)` | `[42000,50000)` |
| 2 | `[50000,52000)` | `[52000,60000)` |
| 3 | `[60000,62000)` | `[62000,70000)` |

No held-out suffix, opposite-regime target, continuous schedule, or historical
baseline result is available to model selection.

## Search and stability-aware selection

Each scenario receives a deterministic 40-call Gaussian-process
expected-improvement search with 10 initial random calls. The unchanged search
family is reservoir size `{100,200,300}`; connectivity `[0.01,1]`; input
scaling `[0.01,3]`; spectral radius `[0.01,3]`; ridge `[1e-10,1e-2]`
log-uniform; and leak rate `[0.01,1]`. Bias scaling remains `0.1`, the bias is
unregularised, and there are no new architecture parameters. Histories are
strict JSON, written atomically after every trial, and deterministically
resumable.

The Bayesian optimiser receives a documented bounded scalar proxy. Exact
selection is lexicographic: lowest numerical-failure count; lowest divergence
count; lowest worst, median, then mean validation NRMSE; highest median then
mean VPT; and serialized-hyperparameter tie-break. Every validation rollout
also stores RMSE, R², Pearson correlation, collapse, and failure information.
Thresholds remain VPT `0.4`, divergence `5.0`, collapse `0.05`, and numerical
failure `1,000,000`.

The top five distinct seed-42 candidates are confirmed with seeds
`(42,123,456,789,2026)`. Per-seed results and aggregate worst seed, medians,
and means are retained. One configuration per scenario is locked before any
final benchmark access.

## Final training and evaluation

After selection locking, five final models per scenario use the existing
130,000-effective-sample allocation and independent-block reset contract.
Regular models see only regular currents; chaotic models see only chaotic
currents; mixed models see all five in their seed-specific block orders.

Only after all 15 models are locked may held-out targets be opened. The fixed
matrix is `RR`, `RC`, `CC`, `CR`, `MR`, and `MC`. The predefined
`regular_then_chaotic`, `chaotic_then_regular`, and `alternating_mixed`
schedules are supporting continuous rollouts without switch reset or re-warm.

The historical 345 results are read only after improved-model locking and only
for the final comparison figure. They never enter optimisation. All new output
is isolated under `chapter2/cross_regime_improvement/results`; writers refuse
accidental overwrite, and production work requires Slurm.
