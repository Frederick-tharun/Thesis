from __future__ import annotations

import numpy as np


def estimate_target_state(
    train_states: np.ndarray,
    spike_threshold_x: float,
    mode: str = "rest_state",
) -> np.ndarray:
    """
    Estimate the control target state from training data.

    Parameters
    ----------
    train_states : array, shape (T, D)
        Training states in NORMALIZED coordinates.
    spike_threshold_x : float
        Spike threshold for x in NORMALIZED coordinates.
    mode : {"rest_state", "zero", "mean"}
        - "rest_state": median of non-spike training states
        - "zero": exact zero vector
        - "mean": mean of all training states

    Returns
    -------
    target : array, shape (D,)
        Target state in NORMALIZED coordinates.
    """
    states = np.asarray(train_states, dtype=float)

    if states.ndim == 1:
        states = states.reshape(-1, 1)

    mode = str(mode).strip().lower()

    if mode == "zero":
        return np.zeros(states.shape[1], dtype=float)

    if mode == "mean":
        return np.mean(states, axis=0)

    # Default: rest-like state estimated from low-x, non-spike samples.
    mask = states[:, 0] < float(spike_threshold_x)

    # Fallback if very few non-spike samples are available.
    min_count = max(25, int(0.02 * len(states)))
    if np.sum(mask) < min_count:
        return np.median(states, axis=0)

    return np.median(states[mask], axis=0)


def _apply_clip(x: np.ndarray, clip):
    if clip is None:
        return np.asarray(x, dtype=float)

    x = np.asarray(x, dtype=float)

    if np.isscalar(clip):
        c = float(clip)
        return np.clip(x, -c, c)

    lo, hi = clip
    return np.clip(x, float(lo), float(hi))


def linear_feedback(
    pred_state: np.ndarray,
    target_state: np.ndarray,
    K: float,
    clip=None,
):
    """
    Linear feedback control:
        error = pred - target
        control_signal = -K * error
        corrected_input = pred + control_signal
                        = pred - K * (pred - target)

    Returns
    -------
    corrected_input : array, shape (D,)
        Controlled signal fed back into the ESN input at next step.
    control_signal : array, shape (D,)
        Additive control term.
    error : array, shape (D,)
        Tracking error pred - target.
    """
    pred = np.asarray(pred_state, dtype=float).reshape(-1)
    target = np.asarray(target_state, dtype=float).reshape(-1)

    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch in linear_feedback: pred={pred.shape}, target={target.shape}"
        )

    error = pred - target
    control_signal = -float(K) * error
    corrected_input = _apply_clip(pred + control_signal, clip)

    return corrected_input, control_signal, error


def format_k_for_path(K: float) -> str:
    """
    Safe folder name fragment for K values.
    Example: 0.5 -> '0p500'
    """
    return f"{float(K):.3f}".replace(".", "p")
