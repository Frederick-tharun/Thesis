"""Validate the dynamical regime at I = 3.29 with the Chapter 1 regime checks.

The cross-regime experiments use I = 3.29 as a non-chaotic current inside the
chaotic training range. Its preliminary label in outputs/dynamics_summary.md
is "uncertain". This script applies the same validator and thresholds that
the Chapter 1 regimes pass (hr_regime_validation.validate_hr_trajectory) under
the hypothesis "periodic bursting", and writes the result to
regime_validation/current_3p29_validation.json.

Steps:
1. ISI cycle search on the frozen dataset outputs/data/fixed_I_3p29.npz
   (1,000 time units of discarded transient).
2. Re-simulation with the Chapter 1 transient of 2,000 time units, checked
   to reproduce the frozen dataset, followed by the full regime validation.
3. Largest Lyapunov exponent over 5,000 time units at I = 3.29 and at the
   chaotic neighbours I = 3.20 and I = 3.34 for comparison.

The trajectory at I = 3.29 is not chaotic, so the result does not depend on
the CPU type. Run from the repository root:

    python chapter2/regime_validation/validate_current_3p29.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader import _rk4_hr
from hr_regime_validation import extract_inter_spike_intervals, validate_hr_trajectory
from scripts.analysis.estimate_hr_lyapunov import estimate_largest_lyapunov

CHAPTER2 = REPO_ROOT / "chapter2"
DATASET = CHAPTER2 / "outputs" / "data" / "fixed_I_3p29.npz"
OUTPUT = CHAPTER2 / "regime_validation" / "current_3p29_validation.json"

# Chapter 2 HR settings (config_ch2.py) with the current under test.
PARAMETERS = dict(a=1.0, b=3.0, c=1.0, d=5.0, r=0.006, s=4.0, xr=-1.6, I=3.29)
INITIAL_STATE = (-1.0, -3.0, 3.0)
DT = 0.01
DATASET_TRANSIENT_STEPS = 100_000   # used for the frozen Chapter 2 dataset
CHAPTER1_TRANSIENT_STEPS = 200_000  # used for the Chapter 1 periodic regimes
RETAINED_STEPS = 100_000
LONG_LYAPUNOV_STEPS = 500_000
CYCLE_LENGTH = 8                    # two alternating four-spike bursts
NEIGHBOUR_CURRENTS = (3.20, 3.34)


def lyapunov(current: float, transient_steps: int, estimation_steps: int) -> dict:
    params = dict(PARAMETERS, I=current)
    result = estimate_largest_lyapunov(
        params, INITIAL_STATE, dt=DT,
        transient_steps=transient_steps, estimation_steps=estimation_steps,
        renormalization_steps=10,
    )
    exponents = np.asarray(result.convergence_exponents)
    times = np.asarray(result.convergence_times)
    tail_std = float(np.std(exponents[-int(np.ceil(0.2 * len(exponents))):]))
    checkpoints = {
        f"{times[int(f * len(exponents)) - 1]:.0f}": float(exponents[int(f * len(exponents)) - 1])
        for f in (0.2, 0.4, 0.6, 0.8, 1.0)
    }
    return {
        "current": current,
        "estimation_time": estimation_steps * DT,
        "largest_lyapunov_exponent": float(result.exponent),
        "tail_std": tail_std,
        "ratio_to_tail_std": abs(float(result.exponent)) / tail_std,
        "running_estimate_by_time": checkpoints,
    }


def cycle_errors(isi: np.ndarray, max_lag: int = 12) -> dict:
    return {
        str(lag): float(np.max(np.abs(isi[lag:] - isi[:-lag])) / isi.mean())
        for lag in range(1, max_lag + 1) if len(isi) > lag
    }


def main() -> None:
    frozen = np.load(DATASET)
    frozen_traj = np.c_[frozen["x"], frozen["y"], frozen["z"]]
    _, frozen_isi = extract_inter_spike_intervals(frozen_traj[:, 0], DT)
    lag_errors = np.abs(frozen_isi[CYCLE_LENGTH:] - frozen_isi[:-CYCLE_LENGTH]) / frozen_isi.mean()
    half = len(lag_errors) // 2

    full = _rk4_hr(INITIAL_STATE, CHAPTER1_TRANSIENT_STEPS + RETAINED_STEPS, DT, PARAMETERS)
    reproduces = float(np.abs(
        full[DATASET_TRANSIENT_STEPS:DATASET_TRANSIENT_STEPS + RETAINED_STEPS, 0] - frozen["x"]
    ).max())
    retained = full[CHAPTER1_TRANSIENT_STEPS:]

    short = lyapunov(3.29, CHAPTER1_TRANSIENT_STEPS, RETAINED_STEPS)
    validation = validate_hr_trajectory(
        retained, dt=DT, expected_regime="periodic_bursting",
        expected_cycle_length=CYCLE_LENGTH, transient_steps=CHAPTER1_TRANSIENT_STEPS,
        largest_lyapunov_exponent=short["largest_lyapunov_exponent"],
        lyapunov_tail_std=short["tail_std"],
    )
    long_runs = [lyapunov(I, CHAPTER1_TRANSIENT_STEPS, LONG_LYAPUNOV_STEPS)
                 for I in (3.29, *NEIGHBOUR_CURRENTS)]

    report = {
        "current": 3.29,
        "hypothesis": "periodic_bursting",
        "frozen_dataset": {
            "path": str(DATASET.relative_to(REPO_ROOT)),
            "transient_time": DATASET_TRANSIENT_STEPS * DT,
            "spike_count": int(len(frozen_isi) + 1),
            "cycle_error_by_lag": cycle_errors(frozen_isi),
            "lag8_max_error_first_half": float(lag_errors[:half].max()),
            "lag8_max_error_second_half": float(lag_errors[half:].max()),
        },
        "rerun_reproduces_frozen_dataset_max_abs_diff": reproduces,
        "chapter1_validation": {
            "transient_time": CHAPTER1_TRANSIENT_STEPS * DT,
            "passed": bool(validation["passed"]),
            "conditions": validation["conditions"],
            "cycle_normalized_max_error": validation["isi_diagnostics"]["cycle_normalized_max_error"],
            "isi_max_min_ratio": validation["isi_diagnostics"]["isi_max_min_ratio"],
            "stationarity": validation["stationarity_diagnostics"],
            "lyapunov_1000_time_units": short,
        },
        "lyapunov_5000_time_units": long_runs,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"passed={report['chapter1_validation']['passed']}  saved {OUTPUT}")


if __name__ == "__main__":
    main()
