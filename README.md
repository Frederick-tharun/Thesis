# Chapter 1: ESN digital twin and control of Hindmarsh–Rose dynamics

This repository contains the final Chapter 1 pipeline for the
periodic_spiking, periodic_bursting and chaotic_bursting synthetic
Hindmarsh–Rose regimes.

The code compares GP, random (dummy), random-forest and GBRT optimization
backends, selects one ESN per regime using validation data only, locks and
saves that model, and evaluates it once on the untouched held-out trajectory.
Linear feedback, finite-time feedback and Pyragas delayed feedback all reuse
the same saved validation-selected chaotic-bursting ESN.

## Definitive run

The source tree must be committed and clean. Submit exactly one job:

~~~bash
sbatch run_final_thesis_pipeline.slurm
~~~

The Slurm preflight refuses tracked, staged, or relevant untracked source
changes. It invokes final_pipeline.py once; it does not rerun main.py for each
controller. Completed evidence is written outside the repository:

~~~text
~/Thesis_evidence_20260624/FINAL_THESIS_RUN_<JOBID>/
~~~

A run is definitive only if
00_manifest/final_package_validation.json contains valid=true. Submitting a
job alone does not make its results final.

## Prediction selection protocol

The final prediction split is 70% training and 30% untouched held-out test.
Model selection receives the 70% training array only. By default its final
24,000 samples form three identical, non-overlapping 8,000-step recursive
validation windows for every optimizer and candidate. The preceding training
samples fit each candidate.

Every window records recursive x NRMSE, multistate NRMSE, spike count, spike
frequency, relative frequency error, mean inter-spike interval,
inter-spike-interval error and divergence status. The configurable score
combines state error, spike-frequency/timing error and stability penalties.
Window scores use the configured mean-plus-maximum aggregation.

selected_model.json is written before the held-out array is normalized or
predicted. The selected model is never replaced using held-out performance.
For periodic spiking, the locked model must pass both predefined gates:
held-out x NRMSE at most 0.20 and relative spike-frequency error at most 0.10.
A failure stops the pipeline as a scientific failure.

## Model reuse and controllers

One final trained ESN bundle per regime stores the reservoir seed, input
weights, recurrent weights, readout weights, scaling statistics, configuration
hash and deterministic model-identity hash. All controller candidates and all
three final controllers load and reuse one chaotic-bursting bundle. Controller
code cannot invoke optimization.

Controller parameters are selected only on controller_validation. Divergent
candidates are recorded as stable=false and rejected=true, with their reason
and evaluated steps; they do not access controller_test. The selected
controller is evaluated once on controller_test, and divergence there is fatal.

The finite-time law is global and piecewise: linear feedback when the
normalized error norm is at least one and fractional-power feedback inside the
unit error sphere. Pyragas uses only sign -1, the convention
next_input = raw_readout - control_signal, and raw_readout as its delayed
observable.

The Chapter 1 objective is regulation toward an empirical quiet-state
reference. It is the median of quiet training samples. It is data-derived, and
its Hindmarsh–Rose right-hand-side residual norm is reported as a diagnostic.
The final pipeline does not run a separate equilibrium-target experiment.

## Curated output

~~~text
FINAL_THESIS_RUN_<JOBID>/
├── 00_manifest/
├── 01_prediction_all_regimes/
├── 02_bo_optimization/
├── 03_linear_feedback/
├── 04_finite_time/
├── 05_pyragas/
├── 06_comparison_tables/
├── 07_report_figures/
└── 08_logs/
~~~

Canonical uncontrolled prediction figures occur only under
01_prediction_all_regimes/<regime>/. Candidate folders contain compact
JSON/CSV evidence; only final controllers receive full controller plots.
07_report_figures uses relative symlinks to canonical figures, avoiding
physical duplicate PNGs. Internal package references are relative. The
validator checks structure, JSON, CSV, readable PNG files, absolute paths,
duplicate hashes, controller summaries, shared model identity, commit
identity, clean-start evidence and final quality gates.

The diagnostic run FINAL_THESIS_RUN_1751888 is preserved unchanged. It is
near-final evidence but is not the definitive package.

## Verification

Use the thesis Conda environment:

~~~bash
python -m py_compile \
  config.py data_loader.py main.py model.py optimize_model.py \
  control_experiment.py neuron_controllers.py plotting.py \
  experiment_report.py final_pipeline.py final_package.py \
  finalization_smoke.py

python -m unittest discover -s tests -p "test_*.py"
bash -n run_final_thesis_pipeline.slurm
python finalization_smoke.py --output-dir /tmp/chapter1_finalization_smoke
~~~

## Documented limitations

The deterministic Chapter 1 case study uses reservoir seed 42. This supports
reproducibility but not broad claims across random initializations. The ridge
solve can emit a numerical conditioning warning for some candidates;
regularization, finite-result checks and a pseudoinverse fallback are present,
but the warning remains a numerical limitation. The empirical quiet-state
reference has a nonzero HR right-hand-side residual because it is data-derived.
These limitations are documented rather than used to reopen the frozen
Chapter 1 methodology.
