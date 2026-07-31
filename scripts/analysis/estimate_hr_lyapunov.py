#!/usr/bin/env python3
"""Estimate the largest Lyapunov exponent of a Hindmarsh--Rose regime.

The estimator integrates the Hindmarsh--Rose state and its tangent-linear
system with the same fourth-order Runge--Kutta step.  The tangent vector is
periodically normalised and its accumulated logarithmic growth is converted
to a largest Lyapunov exponent using the Benettin algorithm.

Example
-------
Run the estimator with the repository's chaotic-bursting parameters:

    python scripts/analysis/estimate_hr_lyapunov.py

Use a shorter calculation for a quick smoke test:

    python scripts/analysis/estimate_hr_lyapunov.py \
        --transient-steps 1000 --steps 5000
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config


REQUIRED_PARAMETERS = ("a", "b", "c", "d", "r", "s", "xr", "I")


@dataclass(frozen=True)
class LyapunovResult:
    """Numerical result and convergence history for one HR calculation."""

    exponent: float
    elapsed_time: float
    final_state: np.ndarray
    convergence_times: np.ndarray
    convergence_exponents: np.ndarray


def hr_rhs(
    state: Sequence[float] | np.ndarray,
    parameters: Mapping[str, float],
) -> np.ndarray:
    """Return the Hindmarsh--Rose vector field at ``state``."""
    x, y, z = np.asarray(state, dtype=float)
    a = float(parameters["a"])
    b = float(parameters["b"])
    c = float(parameters["c"])
    d = float(parameters["d"])
    r = float(parameters["r"])
    s = float(parameters["s"])
    x_r = float(parameters["xr"])
    current = float(parameters["I"])

    return np.array(
        [
            y - a * x**3 + b * x**2 - z + current,
            c - d * x**2 - y,
            r * (s * (x - x_r) - z),
        ],
        dtype=float,
    )


def hr_jacobian(
    state: Sequence[float] | np.ndarray,
    parameters: Mapping[str, float],
) -> np.ndarray:
    """Return the Jacobian of the Hindmarsh--Rose vector field."""
    x = float(np.asarray(state, dtype=float)[0])
    a = float(parameters["a"])
    b = float(parameters["b"])
    d = float(parameters["d"])
    r = float(parameters["r"])
    s = float(parameters["s"])

    return np.array(
        [
            [-3.0 * a * x**2 + 2.0 * b * x, 1.0, -1.0],
            [-2.0 * d * x, -1.0, 0.0],
            [r * s, 0.0, -r],
        ],
        dtype=float,
    )


def rk4_state_step(
    state: np.ndarray,
    dt: float,
    parameters: Mapping[str, float],
) -> np.ndarray:
    """Advance only the HR state by one RK4 step."""
    k1 = hr_rhs(state, parameters)
    k2 = hr_rhs(state + 0.5 * dt * k1, parameters)
    k3 = hr_rhs(state + 0.5 * dt * k2, parameters)
    k4 = hr_rhs(state + dt * k3, parameters)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_variational_step(
    state: np.ndarray,
    tangent: np.ndarray,
    dt: float,
    parameters: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the HR state and tangent-linear system by one RK4 step."""

    def derivative(
        stage_state: np.ndarray,
        stage_tangent: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            hr_rhs(stage_state, parameters),
            hr_jacobian(stage_state, parameters) @ stage_tangent,
        )

    state_k1, tangent_k1 = derivative(state, tangent)
    state_k2, tangent_k2 = derivative(
        state + 0.5 * dt * state_k1,
        tangent + 0.5 * dt * tangent_k1,
    )
    state_k3, tangent_k3 = derivative(
        state + 0.5 * dt * state_k2,
        tangent + 0.5 * dt * tangent_k2,
    )
    state_k4, tangent_k4 = derivative(
        state + dt * state_k3,
        tangent + dt * tangent_k3,
    )

    next_state = state + (dt / 6.0) * (
        state_k1 + 2.0 * state_k2 + 2.0 * state_k3 + state_k4
    )
    next_tangent = tangent + (dt / 6.0) * (
        tangent_k1
        + 2.0 * tangent_k2
        + 2.0 * tangent_k3
        + tangent_k4
    )
    return next_state, next_tangent


def _validate_inputs(
    parameters: Mapping[str, float],
    initial_state: Sequence[float] | np.ndarray,
    dt: float,
    transient_steps: int,
    estimation_steps: int,
    renormalization_steps: int,
) -> np.ndarray:
    missing = [name for name in REQUIRED_PARAMETERS if name not in parameters]
    if missing:
        raise KeyError(f"Missing Hindmarsh--Rose parameters: {missing}")

    state = np.asarray(initial_state, dtype=float)
    if state.shape != (3,):
        raise ValueError(
            f"initial_state must contain exactly three values; got {state.shape}"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain only finite values")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite number")
    if transient_steps < 0:
        raise ValueError("transient_steps must be non-negative")
    if estimation_steps <= 0:
        raise ValueError("estimation_steps must be positive")
    if renormalization_steps <= 0:
        raise ValueError("renormalization_steps must be positive")

    for name in REQUIRED_PARAMETERS:
        if not np.isfinite(float(parameters[name])):
            raise ValueError(f"Parameter {name!r} must be finite")

    return state.copy()


def estimate_largest_lyapunov(
    parameters: Mapping[str, float],
    initial_state: Sequence[float] | np.ndarray,
    *,
    dt: float,
    transient_steps: int,
    estimation_steps: int,
    renormalization_steps: int = 10,
    initial_tangent: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0),
) -> LyapunovResult:
    """Estimate the largest Lyapunov exponent with tangent renormalisation.

    Parameters
    ----------
    parameters:
        Hindmarsh--Rose parameters ``a, b, c, d, r, s, xr, I``.
    initial_state:
        Initial ``(x, y, z)`` state.
    dt:
        Integration time step.
    transient_steps:
        State-only RK4 steps discarded before the estimate begins.
    estimation_steps:
        Number of RK4 steps used to accumulate tangent growth.
    renormalization_steps:
        Maximum steps between tangent-vector normalisations.
    initial_tangent:
        Non-zero direction used to initialise the tangent dynamics.

    Returns
    -------
    LyapunovResult
        Exponent in inverse model-time units and its running convergence
        history.  A positive converged exponent is evidence of chaos.
    """
    state = _validate_inputs(
        parameters,
        initial_state,
        dt,
        transient_steps,
        estimation_steps,
        renormalization_steps,
    )

    tangent = np.asarray(initial_tangent, dtype=float)
    if tangent.shape != (3,):
        raise ValueError(
            f"initial_tangent must contain exactly three values; got "
            f"{tangent.shape}"
        )
    tangent_norm = float(np.linalg.norm(tangent))
    if not np.isfinite(tangent_norm) or tangent_norm == 0.0:
        raise ValueError("initial_tangent must be finite and non-zero")
    tangent = tangent / tangent_norm

    for _ in range(transient_steps):
        state = rk4_state_step(state, dt, parameters)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(
                "HR state became non-finite during transient integration"
            )

    accumulated_log_growth = 0.0
    completed_steps = 0
    convergence_times: list[float] = []
    convergence_exponents: list[float] = []

    while completed_steps < estimation_steps:
        block_steps = min(
            renormalization_steps,
            estimation_steps - completed_steps,
        )

        for _ in range(block_steps):
            state, tangent = rk4_variational_step(
                state,
                tangent,
                dt,
                parameters,
            )

        if not np.all(np.isfinite(state)):
            raise FloatingPointError(
                "HR state became non-finite during exponent estimation"
            )

        tangent_norm = float(np.linalg.norm(tangent))
        if not np.isfinite(tangent_norm) or tangent_norm == 0.0:
            raise FloatingPointError(
                "Tangent norm became zero or non-finite; reduce dt or the "
                "renormalization interval"
            )

        accumulated_log_growth += float(np.log(tangent_norm))
        tangent = tangent / tangent_norm
        completed_steps += block_steps

        elapsed_time = completed_steps * dt
        convergence_times.append(elapsed_time)
        convergence_exponents.append(
            accumulated_log_growth / elapsed_time
        )

    return LyapunovResult(
        exponent=convergence_exponents[-1],
        elapsed_time=estimation_steps * dt,
        final_state=state.copy(),
        convergence_times=np.asarray(convergence_times, dtype=float),
        convergence_exponents=np.asarray(
            convergence_exponents,
            dtype=float,
        ),
    )


def estimate_hr_lyapunov(
    parameters: Mapping[str, float],
    initial_state: Sequence[float] | np.ndarray,
    **kwargs: object,
) -> LyapunovResult:
    """Compatibility alias for :func:`estimate_largest_lyapunov`."""
    return estimate_largest_lyapunov(
        parameters,
        initial_state,
        **kwargs,
    )


def result_summary(
    result: LyapunovResult,
    *,
    regime: str,
    dt: float,
    transient_steps: int,
    estimation_steps: int,
    renormalization_steps: int,
    parameters: Mapping[str, float],
    initial_state: Sequence[float],
) -> dict[str, object]:
    """Build the machine-readable summary written by the CLI."""
    return {
        "method": "Benettin tangent-linear method with RK4",
        "regime": regime,
        "largest_lyapunov_exponent": result.exponent,
        "units": "inverse model-time",
        "positive_exponent_indicates_chaos": bool(result.exponent > 0.0),
        "dt": dt,
        "transient_steps": transient_steps,
        "estimation_steps": estimation_steps,
        "renormalization_steps": renormalization_steps,
        "estimation_time": result.elapsed_time,
        "initial_state": [float(value) for value in initial_state],
        "final_state": result.final_state.tolist(),
        "parameters": {
            name: float(parameters[name]) for name in REQUIRED_PARAMETERS
        },
        "convergence_samples": len(result.convergence_times),
    }


def write_json(path: Path, summary: Mapping[str, object]) -> None:
    """Write an indented JSON result, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")


def write_convergence_csv(path: Path, result: LyapunovResult) -> None:
    """Write running exponent estimates for convergence diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["estimation_time", "largest_lyapunov_exponent"]
        )
        writer.writerows(
            zip(
                result.convergence_times,
                result.convergence_exponents,
            )
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the largest Lyapunov exponent of a configured "
            "Hindmarsh--Rose regime."
        )
    )
    parser.add_argument(
        "--regime",
        choices=tuple(config.HR_PARAMETER_SETS),
        default="chaotic_bursting",
        help="HR parameter set to analyse (default: chaotic_bursting).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=float(config.HR_DT),
        help=f"RK4 time step (default: {config.HR_DT}).",
    )
    parser.add_argument(
        "--transient-steps",
        type=int,
        default=int(config.HR_TRANSIENT),
        help=(
            "Discarded state-only integration steps "
            f"(default: {config.HR_TRANSIENT})."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=int(config.HR_TOTAL_STEPS),
        help=(
            "Steps used for exponent estimation "
            f"(default: {config.HR_TOTAL_STEPS})."
        ),
    )
    parser.add_argument(
        "--renormalize-every",
        type=int,
        default=10,
        help="Tangent renormalisation interval in steps (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON summary.",
    )
    parser.add_argument(
        "--convergence-csv",
        type=Path,
        help="Optional path for the running convergence estimates.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line estimator."""
    args = parse_args(argv)
    parameter_set = config.HR_PARAMETER_SETS[args.regime]
    parameters = {
        name: float(parameter_set[name]) for name in REQUIRED_PARAMETERS
    }
    initial_state = np.asarray(parameter_set["x0"], dtype=float)

    result = estimate_largest_lyapunov(
        parameters,
        initial_state,
        dt=args.dt,
        transient_steps=args.transient_steps,
        estimation_steps=args.steps,
        renormalization_steps=args.renormalize_every,
    )
    summary = result_summary(
        result,
        regime=args.regime,
        dt=args.dt,
        transient_steps=args.transient_steps,
        estimation_steps=args.steps,
        renormalization_steps=args.renormalize_every,
        parameters=parameters,
        initial_state=initial_state,
    )

    if args.output is not None:
        write_json(args.output, summary)
    if args.convergence_csv is not None:
        write_convergence_csv(args.convergence_csv, result)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
