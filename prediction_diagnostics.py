"""Diagnostics that separate short chaotic predictability from climatology."""

from __future__ import annotations

import numpy as np


def chaotic_valid_prediction_horizon(
    predicted_normalized: np.ndarray,
    reference_normalized: np.ndarray,
    *,
    dt: float,
    largest_lyapunov_exponent: float,
    error_threshold: float = 0.4,
    smoothing_steps: int = 100,
    persistence_steps: int = 100,
) -> tuple[dict, list[dict]]:
    """Return a persistent threshold-crossing horizon in time and Lyapunov units."""
    predicted = np.asarray(predicted_normalized, dtype=float)
    reference = np.asarray(reference_normalized, dtype=float)
    if predicted.shape != reference.shape or predicted.ndim != 2:
        raise ValueError("predicted and reference must share a 2D shape")
    if len(predicted) == 0:
        raise ValueError("prediction arrays must not be empty")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(reference)):
        raise ValueError("prediction arrays must be finite")
    if dt <= 0 or largest_lyapunov_exponent <= 0:
        raise ValueError("dt and largest_lyapunov_exponent must be positive")
    if error_threshold <= 0 or smoothing_steps <= 0 or persistence_steps <= 0:
        raise ValueError("threshold and window lengths must be positive")

    squared_state_error = np.mean((predicted - reference) ** 2, axis=1)
    cumulative = np.r_[0.0, np.cumsum(squared_state_error)]
    rolling = np.empty(len(predicted), dtype=float)
    for index in range(len(predicted)):
        start = max(0, index + 1 - int(smoothing_steps))
        rolling[index] = np.sqrt(
            (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)
        )
    instantaneous = np.sqrt(squared_state_error)

    above = rolling > float(error_threshold)
    crossing_index = None
    run = 0
    for index, is_above in enumerate(above):
        run = run + 1 if is_above else 0
        if run >= int(persistence_steps):
            crossing_index = index - int(persistence_steps) + 1
            break
    if crossing_index is None:
        horizon_steps = len(predicted)
        censored = True
    else:
        horizon_steps = int(crossing_index)
        censored = False

    horizon_time = float(horizon_steps * dt)
    summary = {
        "definition": (
            "first persistent crossing of the rolling normalized multistate "
            "RMSE threshold"
        ),
        "error_threshold": float(error_threshold),
        "smoothing_steps": int(smoothing_steps),
        "persistence_steps": int(persistence_steps),
        "horizon_steps": int(horizon_steps),
        "horizon_time": horizon_time,
        "horizon_lyapunov_units": float(
            horizon_time * largest_lyapunov_exponent
        ),
        "largest_lyapunov_exponent": float(largest_lyapunov_exponent),
        "right_censored_at_rollout_end": censored,
        "rollout_steps": int(len(predicted)),
        "rollout_time": float(len(predicted) * dt),
    }
    curve = [
        {
            "step": int(index),
            "time": float(index * dt),
            "time_in_lyapunov_units": float(
                index * dt * largest_lyapunov_exponent
            ),
            "instantaneous_normalized_multistate_error": float(
                instantaneous[index]
            ),
            "rolling_normalized_multistate_rmse": float(rolling[index]),
        }
        for index in range(len(predicted))
    ]
    return summary, curve
