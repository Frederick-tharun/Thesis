# Chapter 1 final methodology fixes

This records the final, frozen Chapter 1 methodology implemented by the
definitive one-pass pipeline.

## Final changes

1. Prediction selection receives only the 70% training portion. Three shared,
   non-overlapping 8,000-step recursive validation windows contribute state
   error, spike-frequency/timing error and divergence evidence.
2. selected_model.json is locked before the untouched held-out array is
   evaluated. Periodic spiking has predefined held-out quality gates; failure
   is scientific failure and never triggers test-based reselection.
3. Bayesian optimization runs as one overall stage per regime. The selected
   ESN is trained and serialized once per regime. Controller searches do not
   call optimization or train a base model.
4. Linear, finite-time and Pyragas experiments load one validation-selected
   chaotic-bursting bundle and share its deterministic identity hash.
5. Finite-time control is global and piecewise: linear outside the normalized
   unit error sphere and fractional-power feedback inside it.
6. Pyragas uses sign -1, next_input = raw_readout - control_signal, and the
   recorded delayed observable raw_readout.
7. Divergent validation candidates are compact, nonfatal rejections.
   Divergence of a selected controller on controller_test is fatal.
8. The objective is regulation toward an empirical quiet-state reference: the
   median of quiet training samples. Its HR right-hand-side residual norm is a
   diagnostic of the data-derived reference.
9. Prediction figures are canonical under 01_prediction_all_regimes.
   Candidate output is compact, report figures are relative symlinks, and
   temporary output is excluded from the curated package.
10. Slurm requires a clean committed tree and records the running commit,
    branch, source/configuration hashes, environment, commands, seeds and stage
    timings with portable internal paths.

## Required evidence

Each regime provides validation_windows.json,
optimizer_validation_summary.csv, best_params.json, selected_model.json,
model_bundle.npz and heldout_test_metrics.json in its canonical BO/prediction
sections.

Every final control_summary.json includes source regime, selected optimizer,
parameter file, Git commit, seed, model identity, configuration hash, cache
status and relative canonical prediction path. The manifest records one
overall BO-stage invocation for each regime.

00_manifest/final_package_validation.json is the authority for package
completeness. It validates the eight required sections, readability, portable
paths, duplicate hashes, shared controller identity, Git revision, clean-start
status and prediction gates.

## Verification

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

The reduced smoke pass covers training-only periodic selection with three
recursive windows, spike-frequency metrics, cached chaotic-model reuse,
nonfatal validation rejection, fatal selected-controller divergence and a
portable tiny package.

## Frozen limitations

- BO uses the single deterministic reservoir seed 42. This is a reproducible
  case study, not evidence of performance across random seeds.
- Some candidates can produce a linear-algebra conditioning warning. The
  regularized solver has finite checks and a pseudoinverse fallback; the
  warning remains documented.
- The empirical quiet-state reference is data-derived and generally has a
  nonzero HR right-hand-side residual.

`FINAL_THESIS_RUN_1752614`, produced from commit
`2e953b065b3208c1556cf3d56ce690a1f7cd48c5`, is the definitive Chapter 1
evidence package. Its final-package validator reports success.

## Final command

~~~bash
sbatch run_final_thesis_pipeline.slurm
~~~
