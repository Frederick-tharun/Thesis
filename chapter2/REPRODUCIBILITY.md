# Chapter 2 reproducibility and integrity verification

All commands below are run from the repository root.
The login shell exposes `python3` (not an unversioned `python`).

## Recorded scientific environment

The frozen Step 7 selection and final-model metadata record this
training/evaluation stack:

- Python 3.12.13
- NumPy 1.26.4
- pandas 2.2.3
- SciPy 1.13.1
- Matplotlib 3.9.2
- scikit-learn 1.5.2
- scikit-optimize 0.10.2

The final seeds are `42, 123, 456, 789, 2026`. Step 7 optimizer seeds are
2026 for the parameter-aware search and 2027 for the ordinary search; the
candidate reservoir seed is 42. Thread counts recorded for the definitive
Step 8 job were 8 for OMP, OpenBLAS, and MKL.

`chapter2/environment.yml` is the canonical Chapter 2 environment
specification. The root `requirements.txt` also supports repository-wide
work and currently contains newer pandas/plotting pins; it is not the
historical Chapter 2 scientific lock.

The official figure manifest does not record a separate plotting package
environment. Therefore no exact plotting environment beyond the source and
artifact hashes can honestly be claimed. The current shell has Python 3.12.3,
NumPy 2.5.0, pandas 3.0.3, SciPy 1.18.0, and Matplotlib 3.10.9; those are not
the recorded training/evaluation versions.

## Clean isolated environment

Create a new environment outside tracked source directories:

```bash
conda env create \
  --prefix /scratch/"$USER"/chapter2-release \
  --file chapter2/environment.yml

conda run --prefix /scratch/"$USER"/chapter2-release \
  python -m pip check
```

Then run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD" \
conda run --prefix /scratch/"$USER"/chapter2-release \
  python3 -m pytest -q chapter2/tests
```

The existing `thesis_final` environment currently passes `pip check`, but
it now contains NumPy 2.5.0 rather than the recorded NumPy 1.26.4 and is not a
clean historical reproduction environment. The general interactive shell also
reports an unrelated missing `cffi` dependency for PyNaCl. Mixed NumPy
2.5/system components can emit a Matplotlib warning about modules compiled
against NumPy 1.x. Use the pinned clean environment rather than suppressing or
normalizing that warning.

## Safe release-integrity verification

This verifies frozen files only. It does not load models for prediction or run
Step 7, Step 8, the audit, or figure generation:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD" \
python3 -m chapter2.verify_release
```

The verifier writes nothing by default. To create a new report outside the
protected historical directories, pass an explicit path:

```bash
python3 -m chapter2.verify_release \
  --output /tmp/chapter2_release_verification.json
```

Safe test commands are:

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

The Chapter 2 compatibility test imports the repository-root Chapter 1
`data_loader.py`. Chapter 2 deliberately has no second `data_loader.py`;
the import works when `PYTHONPATH="$PWD"` is set. All other Chapter 2
simulation code is isolated in `chapter2/hr_data_ch2.py`.

## Reproduction versus verification

Release-integrity verification checks strict JSON, file sets, counts, sizes,
and SHA-256 hashes. It is the correct operation for submission cleanup.

Scientific reproduction would create new datasets, redo Step 7 selection,
retrain models, rerun Step 8, and regenerate outputs. That is a separate,
expensive experiment and is prohibited during final repository cleanup. The
SLURM scripts remain historical/reproduction launchers only; do not submit them
to verify this release. `CHAPTER2_CONDA_ENV` may override their preserved
default environment path when a separately authorized reproduction is planned.
