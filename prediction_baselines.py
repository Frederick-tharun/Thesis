"""Simple leakage-free autonomous baselines for Chapter 1 prediction."""

from __future__ import annotations

import numpy as np


def persistence_forecast(last_training_state: np.ndarray, steps: int) -> np.ndarray:
    """Repeat the final training state throughout the forecast horizon."""
    state = np.asarray(last_training_state, dtype=float).reshape(1, -1)
    if steps <= 0:
        raise ValueError("steps must be positive")
    return np.repeat(state, int(steps), axis=0)


def fit_linear_ar1(
    training: np.ndarray,
    *,
    regularization: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a multivariate AR(1) map using training data only."""
    values = np.asarray(training, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("training must have shape (n>=3, n_states)")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    design = np.column_stack([values[:-1], np.ones(values.shape[0] - 1)])
    targets = values[1:]
    penalty = np.sqrt(float(regularization)) * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    augmented_design = np.vstack([design, penalty])
    augmented_targets = np.vstack(
        [targets, np.zeros((design.shape[1], targets.shape[1]))]
    )
    coefficients, *_ = np.linalg.lstsq(
        augmented_design,
        augmented_targets,
        rcond=None,
    )
    return coefficients[:-1], coefficients[-1]


def recursive_linear_ar1_forecast(
    last_training_state: np.ndarray,
    transition: np.ndarray,
    intercept: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Roll a fitted AR(1) model autonomously without test observations."""
    state = np.asarray(last_training_state, dtype=float).reshape(-1)
    transition = np.asarray(transition, dtype=float)
    intercept = np.asarray(intercept, dtype=float).reshape(-1)
    if transition.shape != (state.size, state.size):
        raise ValueError("transition shape does not match the state dimension")
    if intercept.shape != state.shape:
        raise ValueError("intercept shape does not match the state dimension")
    if steps <= 0:
        raise ValueError("steps must be positive")

    forecast = np.empty((int(steps), state.size), dtype=float)
    for index in range(int(steps)):
        state = state @ transition + intercept
        forecast[index] = state
    return forecast


def autonomous_baselines(
    training: np.ndarray,
    steps: int,
    *,
    ar_regularization: float = 1e-6,
) -> dict[str, dict]:
    """Generate predeclared persistence and linear-AR baselines."""
    values = np.asarray(training, dtype=float)
    transition, intercept = fit_linear_ar1(
        values,
        regularization=ar_regularization,
    )
    return {
        "persistence": {
            "prediction": persistence_forecast(values[-1], steps),
            "description": "last training state repeated autonomously",
            "uses_heldout_observations": False,
        },
        "linear_ar1": {
            "prediction": recursive_linear_ar1_forecast(
                values[-1], transition, intercept, steps
            ),
            "description": (
                "training-only multivariate ridge AR(1), recursively rolled out"
            ),
            "uses_heldout_observations": False,
            "regularization": float(ar_regularization),
            "transition": transition,
            "intercept": intercept,
        },
    }
