# ESN prediction and control of Hindmarsh–Rose neuron dynamics

This repository contains the code used for Chapter 1 of my master's thesis. It
examines whether an Echo State Network (ESN) can learn three regimes of the
Hindmarsh–Rose neuron model—periodic spiking, periodic bursting and chaotic
bursting—and whether the trained ESN can then be controlled.

The project compares Gaussian-process, random, random-forest and GBRT
hyperparameter searches. The selected models are evaluated on held-out data
that are not used during model selection. The chaotic-bursting ESN is then
reused for linear feedback, global finite-time feedback and Pyragas delayed
feedback experiments.

The controllers act on the trained ESN digital twin. They do not directly
control a biological neuron or the original Hindmarsh–Rose differential
equations.

## Main files

| File | Purpose |
|---|---|
| `main.py` | General command-line entry point |
| `final_pipeline.py` | Runs the complete Chapter 1 experiment |
| `final_package.py` | Validates a completed final result package |
| `model.py` | Echo State Network implementation |
| `optimize_model.py` | Hyperparameter search and validation scoring |
| `neuron_controllers.py` | Linear, finite-time and Pyragas control laws |
| `control_experiment.py` | Controller selection and evaluation |
| `data_loader.py` | Hindmarsh–Rose simulation and preprocessing |
| `plotting.py` | Prediction, optimization and control figures |
| `finalization_smoke.py` | Reduced end-to-end workflow check |
| `run_final_thesis_pipeline.slurm` | Slurm job for the definitive run |


## Installation

The definitive experiment used Python 3.12. To create a local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On the HPC system, the final job used the Conda environment `thesis_final`.
The exact software versions are stored in the final run manifest.

## Running the code

Use the standard command-line interface for individual experiments:

```bash
python main.py --help
```

Submit the complete Chapter 1 workflow with:

```bash
sbatch run_final_thesis_pipeline.slurm
```

The Slurm workflow requires a clean, committed Git tree. This connects every
result to an exact source revision. It writes the complete evidence package
outside the repository:

```text
~/Thesis_evidence_20260624/FINAL_THESIS_RUN_<JOB_ID>/
```

A run is complete only when
`00_manifest/final_package_validation.json` contains `"valid": true`.

## Evaluation protocol

For each dynamical regime, 70% of the trajectory is used for training and
model selection. The remaining 30% is an untouched held-out test trajectory.
Bayesian optimization receives only the training portion.

Candidates are evaluated on three non-overlapping recursive validation
windows. The validation score considers state error, spike frequency and
timing, and prediction stability. The selected ESN is trained once and then
evaluated on the held-out trajectory.

Controller parameters are selected on a controller-validation segment. The
locked controller is evaluated once on a separate controller-test segment.
All controllers reuse the same saved chaotic-bursting ESN.

## Definitive results

The definitive experiment is Slurm run **1752614**, produced from Git commit
`2e953b065b3208c1556cf3d56ce690a1f7cd48c5`. Its package validator passed with
no missing or corrupted files.

### ESN prediction

| Regime | Optimizer | Held-out x NRMSE | All-state NRMSE |
|---|---:|---:|---:|
| Periodic spiking | Random forest | 0.0000726 | 0.0000445 |
| Periodic bursting | Random forest | 0.04037 | 0.02754 |
| Chaotic bursting | GBRT | 0.002275 | 0.001528 |

### Controller evaluation

| Controller | Selected parameters | Held-out controller-test result |
|---|---|---|
| Linear feedback | `K = 1.005` | state RMSE `2.36e-06`; 100% spike reduction |
| Global finite-time feedback | `s = 0.9`, `K = 0.914456` | state RMSE `2.50e-04`; 100% spike reduction |
| Pyragas delayed feedback | delay `1600`, sign `-1`, `K = 0.768091` | period `1600`; recurrence error `0.01343`; correlation `0.99991` |

Linear and finite-time feedback regulate the ESN toward an empirical
quiet-state reference calculated from training data. This reference is not an
exact equilibrium of the Hindmarsh–Rose equations.

Pyragas control has a different objective: it changes irregular bursting into
a regular periodic-spiking trajectory. An increased spike count is therefore
not a failure for this controller.

## Thesis figures

Thesis-ready validation-sweep figures are in
`THESIS_SWEEP_FIGURES_1752614/`. PDFs are intended for the thesis, while PNGs
are included for convenient previewing. The combined source data are in
`controller_sweeps_combined.csv` in the same folder.

These figures document controller selection on validation data. They should
not be described as held-out controller-test figures.

## Verification

```bash
python -m py_compile \
  config.py data_loader.py main.py model.py optimize_model.py \
  control_experiment.py neuron_controllers.py plotting.py \
  experiment_report.py final_pipeline.py final_package.py \
  finalization_smoke.py

python finalization_smoke.py --output-dir /tmp/chapter1_finalization_smoke
bash -n run_final_thesis_pipeline.slurm
```

The final package records source and configuration hashes, the Git revision,
software versions, random seeds, stage timings and integrity checks under
`00_manifest/`.

## Limitations

This is a deterministic computational case study using reservoir seed 42. It
is reproducible for that realization, but it is not a statistical study over
many reservoir initializations.

Some ESN candidates can emit a numerical-conditioning warning during the
regularized readout solve. The implementation checks for finite results and
has a pseudoinverse fallback, but the warning remains a numerical limitation.

The Pyragas result is empirical and finite-horizon. It demonstrates a
sustained regular trajectory in the tested ESN rollout, not a mathematical
proof of asymptotic stability or perfectly non-invasive control.

## Reproducibility note

The complete validated evidence package is included in
`FINAL_THESIS_RUN_1752614/`. It contains the saved models, full controller
rollouts, figures, comparison tables, logs and reproducibility manifests used
for the reported results. The smaller `THESIS_SWEEP_FIGURES_1752614/` folder
contains additional thesis-ready validation-sweep figures and their source
CSV data.
