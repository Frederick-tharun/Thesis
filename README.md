# Hindmarsh-Rose ESN Prediction and Control

This project trains a full-state echo state network (ESN) for the
Hindmarsh-Rose system:

```text
[x, y, z] -> [x_next, y_next, z_next]
```

The primary thesis workflow uses the chaotic-bursting regime and evaluates
three controllers on the ESN digital twin. See [FINAL_RESULTS.md](FINAL_RESULTS.md)
for the validated parameters, evidence hashes, and claim boundaries.

## Environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The final HPC validation used Python 3.12, scikit-optimize 0.10.2, and
scikit-learn 1.9.0.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Prediction Baseline

Run the chaotic-bursting ESN with the configured default parameters:

```bash
.venv/bin/python main.py \
  --dataset hr \
  --hr-mode chaotic_bursting \
  --no-opt
```

Run one optimizer explicitly:

```bash
.venv/bin/python main.py \
  --dataset hr \
  --hr-mode chaotic_bursting \
  --optimizer dummy
```

## Validated Chaotic Controllers

These commands reproduce the fixed controller candidates validated at source
commit `ae50ce2a434114902c72d0f76895fba9d73da0c1`.

Linear feedback:

```bash
.venv/bin/python main.py \
  --dataset hr \
  --hr-mode chaotic_bursting \
  --optimizer dummy \
  --control \
  --controller linear_feedback \
  --control-k 1.0
```

Finite-time feedback:

```bash
.venv/bin/python main.py \
  --dataset hr \
  --hr-mode chaotic_bursting \
  --optimizer dummy \
  --control \
  --controller finite_time \
  --control-k 0.4582142857142857 \
  --finite-s 0.8
```

Pyragas delayed feedback:

```bash
.venv/bin/python main.py \
  --dataset hr \
  --hr-mode chaotic_bursting \
  --optimizer dummy \
  --control \
  --controller pyragas \
  --control-k 0.8 \
  --pyragas-delay 2400 \
  --pyragas-sign -1
```

The corresponding HPC jobs are:

```bash
sbatch run_final_linear_finite_validation.slurm
sbatch run_pyragas_final_validation.slurm
```

## Exploratory Gain Search

Automatic K selection already exists. Omit `--control-k` and pass
`--auto-control-k` to run the configured coarse and refined gain search. This
is exploratory and is not required to reproduce the locked final candidates.

## Outputs

Generated files are written below `outputs/`. Final validation scripts copy
selected evidence to `selected_controller_results/` or
`selected_pyragas_results/`. These generated directories are ignored by Git;
archive them separately with checksums.

Existing outputs are preserved unless `--clean-output` is explicitly supplied.
