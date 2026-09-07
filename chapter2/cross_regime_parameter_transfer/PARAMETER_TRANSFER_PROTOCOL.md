# Parameter-transfer ESN follow-up

Starting scientific commit: d5cc55f6d2ea4210f6c06be6d79398ec90749907.
This is a new exploratory follow-up motivated by already inspected historical
results. Those results are immutable and never enter selection. Fresh trajectories
provide a new evaluation, not a retrospective claim that the study design was blind.

## Frozen model and preprocessing

Reuse chapter2.esn_model.EchoStateNetwork unchanged: [x,y,z,I] -> [x_next,y_next,z_next].
Four inputs, three outputs, tanh reservoir, linear ridge readout, bias scaling 0.1,
unregularised bias. Current is supplied and never predicted.
Use I_scaled = 2*(I-1.67)/(3.50-1.67)-1, equivalently fixed centre 2.585 and
scale 0.915; these constants are not estimated from trajectory data.
State scalers fit only the fitting inputs of each fold's training currents.

## Parameter validation

Regular currents: (1.67,3.29,3.50); chaotic currents: (3.20,3.34).
regular_parameter_transfer: R1 fit (3.29,3.50), validate 1.67; R2 fit
(1.67,3.50), validate 3.29; R3 fit (1.67,3.29), validate 3.50.
chaotic_parameter_transfer: C1 fit (3.20), validate 3.34; C2 fit (3.34),
validate 3.20. Opposite-regime trajectory data are forbidden during selection.
Readout and state-scaler fitting use transitions [0,40000) only.
Each independent fitting block resets its reservoir and discards 2000 washout
transitions; pairs never cross block boundaries. Validation held-out currents
are absent from both readout and scaler fitting. Three validation windows use
warm-up/scored ranges [40000,42000)/[42000,50000),
[50000,52000)/[52000,60000), [60000,62000)/[62000,70000).
Warm-up is teacher-forced; scoring feeds back only predictions and supplied I.
Validation uses 9 rollouts per regular candidate and 6 per chaotic candidate.

## Search and seed confirmation

Separate deterministic GP expected-improvement searches: 40 calls, 10 random
initial calls; optimiser seeds 41101 and 41102; candidate model seed 42.
Reservoir sizes exactly {100,200,300,500,800}; connectivity [0.01,1],
input scaling [0.01,3], spectral radius [0.01,3], ridge [1e-10,1e-2]
log-uniform, leak [0.01,1]. No new architecture parameters.
Rank all held-out-current rollouts lexicographically by numerical failures,
divergences, worst/median/mean NRMSE, negative median/mean VPT, serialized
hyperparameters. Bayesian feedback uses the existing bounded stability proxy;
the exact lexicographic ranking is authoritative, not the proxy minimum.
Thresholds: VPT 0.4, divergence 5.0, collapse 0.05; failure penalty 1000000.
Confirm the top five distinct configurations over seeds (42,123,456,789,2026),
repeating all folds; seed 42 may reuse the exact saved search evaluation.
Reject any configuration with a numerical failure or divergence in any seed.
Collapse remains reported rather than a new ranking term. If no confirmed
configuration survives, stop without a selection lock. This is a stability
eligibility rule, not a declaration of accurate transfer.
Store every rollout and per-fold/current/seed summary. JSON is strict and atomic;
resume verifies source/design/runtime metadata and deterministic ask/tell replay.
Completed histories are never reopened for more search after selection locking.

## Final training and fresh benchmark

After both selections lock, train all permitted currents with the existing
130000-effective-sample allocation and independent-block washout contract.
Five seeds per scenario produce ten final models. Validate and lock all model
hashes before generating or opening fresh targets. No fitting after target access.
The benchmark protocol is frozen as FRESH_BENCHMARK_PROTOCOL.json before any
prediction; production generation remains deferred in this implementation stage.
Reuse the alternative initial state (0.1,0,0) already specified for Hindmarsh-Rose
in config.py HR_PARAMETER_SETS['periodic_spiking']. Its a,b,c,d,r,s,xr equal
Chapter 2's coefficients; its old current 2.5 is NOT used. Use all five prescribed
currents, unchanged Chapter 2 RK4, dt=0.01 and parameters, discard 100000
transient steps, retain 100000 states. This is a different deterministic initial
condition, not a claim of statistical independence or a guarantee of a different
attractor. Do not use target performance to change this initial state.
Three fresh short windows use warm-up starts 70000,80000,89999, 2000 warm-up
and 8000 forecast transitions; long window warm-up [70000,72000), forecast
[72000,99999). Final records: 150 short plus 50 long, RR/RC/CC/CR only.
No new continuous schedules in this compact follow-up. No mixed model is fitted.
Fresh datasets, model inventory and selection hashes must match on resumption.

## Descriptive diagnostics and reporting

Primary metrics remain NRMSE/VPT/divergence/numerical failure/collapse/R2/Pearson.
Use existing Chapter 2 spike/burst detection for supplementary spike frequency,
ISI distributions, burst frequency/statistics and finite state mean/std/range.
No new Lyapunov estimator. Figures include current scaling, fold schematic,
search convergence, seed robustness, fresh RR/RC/CC/CR, NRMSE/VPT/divergence,
phase portraits and historical comparison. Historical results may be parsed
only after all new final records are complete and provenance verified; comparisons
across different trajectories are descriptive, not paired performance estimates.
All outputs are confined to this package's results directory. Previous scientific
files and generated artifacts are protected by a pre-work SHA-256 inventory.
Production requires Slurm. This task runs tests and a synthetic tiny pilot only;
no production submission, fresh benchmark generation, commit, or merge.
