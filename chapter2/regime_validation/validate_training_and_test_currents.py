"""Validate the regime labels of the Chapter 2 currents with the Chapter 1 checks.

The regime of each current (1.67, 3.20, 3.34, 3.50) is taken from the
supervisor's specification. This script provides the supporting evidence with
the validator and thresholds used for the Chapter 1 regimes
(hr_regime_validation.validate_hr_trajectory). I = 3.29 is handled by
validate_current_3p29.py.

For each current:
1. The frozen dataset outputs/data/fixed_I_<I>.npz (1,000 time units of
   discarded transient, 1,000 retained) is summarised: spike count and the
   ISI cycle error for lags 1-12.
2. The trajectory is re-simulated with the Chapter 1 protocol (2,000 time
   units of transient for periodic regimes, 4,000 for chaotic ones,
   1,500 retained time units) and validated against the stated regime.
3. The largest Lyapunov exponent is also estimated over 5,000 time units.

Chaotic trajectories depend on the CPU type, so run this on a node of the
`work` partition. Run from the repository root:

    python chapter2/regime_validation/validate_training_and_test_currents.py
"""

from __future__ import annotations

import json
import platform
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
OUTPUT = CHAPTER2 / "regime_validation" / "current_regimes_validation.json"

BASE = dict(a=1.0, b=3.0, c=1.0, d=5.0, r=0.006, s=4.0, xr=-1.6)
INITIAL_STATE = (-1.0, -3.0, 3.0)
DT = 0.01
RETAINED_STEPS = 150_000         # 1,500 time units, as in Chapter 1
LONG_LYAPUNOV_STEPS = 500_000    # 5,000 time units
MAX_LAG = 12

# Regime stated by the supervisor, with the Chapter 1 transient for that type.
CURRENTS = {
    1.67: ("periodic_bursting", 200_000),
    3.50: ("periodic_spiking", 200_000),
    3.20: ("chaotic_bursting", 400_000),
    3.34: ("chaotic_bursting", 400_000),
}


def cycle_errors(isi: np.ndarray) -> dict[int, float]:
    return {
        lag: float(np.max(np.abs(isi[lag:] - isi[:-lag])) / isi.mean())
        for lag in range(1, MAX_LAG + 1) if len(isi) > lag
    }


def lyapunov(params: dict, transient_steps: int, estimation_steps: int) -> dict:
    result = estimate_largest_lyapunov(
        params, INITIAL_STATE, dt=DT, transient_steps=transient_steps,
        estimation_steps=estimation_steps, renormalization_steps=10,
    )
    exponents = np.asarray(result.convergence_exponents)
    tail_std = float(np.std(exponents[-int(np.ceil(0.2 * len(exponents))):]))
    return {
        "estimation_time": estimation_steps * DT,
        "largest_lyapunov_exponent": float(result.exponent),
        "tail_std": tail_std,
        "ratio_to_tail_std": abs(float(result.exponent)) / tail_std,
    }


def validate(current: float, regime: str, transient_steps: int) -> dict:
    params = dict(BASE, I=current)
    tag = f"{current:.2f}".replace(".", "p")
    frozen = np.load(CHAPTER2 / "outputs" / "data" / f"fixed_I_{tag}.npz")
    _, frozen_isi = extract_inter_spike_intervals(frozen["x"], DT)

    trajectory = _rk4_hr(INITIAL_STATE, transient_steps + RETAINED_STEPS, DT, params)
    retained = trajectory[transient_steps:]
    _, isi = extract_inter_spike_intervals(retained[:, 0], DT)
    errors = cycle_errors(isi)

    if regime == "periodic_spiking":
        cycle_length = 1
    elif regime == "periodic_bursting":
        passing = [lag for lag in range(2, MAX_LAG + 1) if errors.get(lag, 1.0) <= 0.005]
        cycle_length = passing[0] if passing else min(
            (lag for lag in errors if lag >= 2), key=errors.get)
    else:
        cycle_length = None

    short = lyapunov(params, transient_steps, RETAINED_STEPS)
    result = validate_hr_trajectory(
        retained, dt=DT, expected_regime=regime, expected_cycle_length=cycle_length,
        transient_steps=transient_steps,
        largest_lyapunov_exponent=short["largest_lyapunov_exponent"],
        lyapunov_tail_std=short["tail_std"],
    )
    isi_diag = result["isi_diagnostics"]
    return {
        "current": current,
        "stated_regime": regime,
        "frozen_dataset": {
            "spike_count": int(len(frozen_isi) + 1),
            "cycle_error_by_lag": {str(k): v for k, v in cycle_errors(frozen_isi).items()},
        },
        "chapter1_protocol": {
            "transient_time": transient_steps * DT,
            "retained_time": RETAINED_STEPS * DT,
            "spike_count": result["detected_spike_count"],
            "cycle_length_tested": cycle_length,
            "cycle_error_by_lag": {str(k): v for k, v in errors.items()},
            "passed": bool(result["passed"]),
            "conditions": result["conditions"],
            "isi_cv": isi_diag.get("isi_cv"),
            "isi_max_min_ratio": isi_diag.get("isi_max_min_ratio"),
            "cycle_normalized_max_error": isi_diag.get("cycle_normalized_max_error"),
            "best_short_lag_normalized_mae": isi_diag.get("best_short_lag_normalized_mae"),
            "stationarity": result["stationarity_diagnostics"],
            "lyapunov_1500_time_units": short,
        },
        "lyapunov_5000_time_units": lyapunov(params, transient_steps, LONG_LYAPUNOV_STEPS),
    }


def main() -> None:
    report = {
        "host": platform.node(),
        "currents": [validate(I, regime, steps) for I, (regime, steps) in CURRENTS.items()],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, default=float) + "\n")
    for entry in report["currents"]:
        print(f"I={entry['current']:.2f} {entry['stated_regime']}: "
              f"passed={entry['chapter1_protocol']['passed']}")
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
