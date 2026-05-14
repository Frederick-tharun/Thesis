from __future__ import annotations

import numpy as np


def detect_spikes_x(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Simple local-maximum spike detector on x.
    """
    x = np.asarray(x, dtype=float).reshape(-1)

    if len(x) < 3:
        return np.array([], dtype=int)

    peaks = []

    for i in range(1, len(x) - 1):
        if x[i] >= threshold and x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            peaks.append(i)

    return np.asarray(peaks, dtype=int)


def count_spikes_x(x: np.ndarray, threshold: float) -> int:
    return int(len(detect_spikes_x(x, threshold)))


def compute_settling_time(
    error_norm: np.ndarray,
    dt: float,
    tol: float,
    hold_steps: int,
):
    """
    Settling time is the first time the error stays below tol
    for 'hold_steps' consecutive samples.

    Returns
    -------
    settling_time_s : float
        Time after control start, in seconds. NaN if not settled.
    settling_step : int
        Step after control start. -1 if not settled.
    """
    err = np.asarray(error_norm, dtype=float).reshape(-1)
    hold_steps = max(1, int(hold_steps))

    if len(err) < hold_steps:
        return float("nan"), -1

    for i in range(0, len(err) - hold_steps + 1):
        if np.all(err[i : i + hold_steps] <= tol):
            return float(i * dt), int(i)

    return float("nan"), -1


def compute_linear_feedback_metrics(
    *,
    uncontrolled: np.ndarray,
    controlled: np.ndarray,
    target_state: np.ndarray,
    control_signal: np.ndarray,
    control_start_idx: int,
    dt: float,
    spike_threshold_x: float,
    settling_tol: float,
    settling_hold_steps: int,
    abs_limit: float,
) -> dict:
    """
    Control metrics used after linear-feedback control begins.

    Notes
    -----
    These are control metrics, not standard prediction metrics:
    - target_rmse_state
    - target_rmse_x
    - settling_time
    - spike reduction
    - control energy
    - divergence flag
    """
    uncontrolled = np.asarray(uncontrolled, dtype=float)
    controlled = np.asarray(controlled, dtype=float)
    target = np.asarray(target_state, dtype=float).reshape(-1)
    control_signal = np.asarray(control_signal, dtype=float)

    if uncontrolled.ndim == 1:
        uncontrolled = uncontrolled.reshape(-1, 1)

    if controlled.ndim == 1:
        controlled = controlled.reshape(-1, 1)

    if control_signal.ndim == 1:
        control_signal = control_signal.reshape(-1, 1)

    horizon = len(controlled)
    control_start_idx = int(max(0, min(control_start_idx, max(horizon - 1, 0))))

    post_u = uncontrolled[control_start_idx:]
    post_c = controlled[control_start_idx:]
    post_ctrl = control_signal[control_start_idx:]

    finite_post_c = np.all(np.isfinite(post_c), axis=1)
    finite_post_ctrl = np.all(np.isfinite(post_ctrl), axis=1)

    safe_post_c = post_c[finite_post_c]
    safe_post_ctrl = post_ctrl[finite_post_ctrl]

    if len(safe_post_c) > 0:
        err = safe_post_c - target.reshape(1, -1)
        err_norm = np.linalg.norm(err, axis=1)
        target_rmse_state = float(np.sqrt(np.mean(err_norm**2)))
        target_rmse_x = float(np.sqrt(np.mean(err[:, 0] ** 2)))
        final_error_norm = float(err_norm[-1])
        stable_fraction_under_tol = float(np.mean(err_norm <= float(settling_tol)))
        settling_time_s, settling_step = compute_settling_time(
            err_norm,
            dt=float(dt),
            tol=float(settling_tol),
            hold_steps=int(settling_hold_steps),
        )
    else:
        err_norm = np.array([], dtype=float)
        target_rmse_state = float("inf")
        target_rmse_x = float("inf")
        final_error_norm = float("inf")
        stable_fraction_under_tol = 0.0
        settling_time_s, settling_step = float("nan"), -1

    if len(safe_post_ctrl) > 0:
        ctrl_energy_total = float(np.sum(np.sum(safe_post_ctrl**2, axis=1)))
        ctrl_energy_mean = float(np.mean(np.sum(safe_post_ctrl**2, axis=1)))
    else:
        ctrl_energy_total = float("inf")
        ctrl_energy_mean = float("inf")

    spikes_pre_control = count_spikes_x(
        uncontrolled[:control_start_idx, 0],
        threshold=float(spike_threshold_x),
    )
    spikes_uncontrolled_post = count_spikes_x(
        post_u[:, 0],
        threshold=float(spike_threshold_x),
    )
    spikes_controlled_post = count_spikes_x(
        post_c[:, 0],
        threshold=float(spike_threshold_x),
    )

    if spikes_uncontrolled_post > 0:
        spike_reduction_percent = float(
            100.0 * (spikes_uncontrolled_post - spikes_controlled_post) / spikes_uncontrolled_post
        )
    else:
        spike_reduction_percent = 0.0

    finite_all = np.isfinite(controlled)
    if np.any(finite_all):
        max_abs_x_post = float(np.nanmax(np.abs(post_c[:, 0])))
        max_abs_state = float(np.nanmax(np.abs(controlled[finite_all])))
        diverged = (not np.all(finite_all)) or (max_abs_state > float(abs_limit))
    else:
        max_abs_x_post = float("inf")
        diverged = True

    return {
        "horizon_steps": int(horizon),
        "control_start_idx": int(control_start_idx),
        "post_control_steps": int(len(post_c)),
        "spikes_pre_control": int(spikes_pre_control),
        "spikes_uncontrolled_post": int(spikes_uncontrolled_post),
        "spikes_controlled_post": int(spikes_controlled_post),
        "spike_reduction_percent": float(spike_reduction_percent),
        "target_rmse_state": float(target_rmse_state),
        "target_rmse_x": float(target_rmse_x),
        "final_error_norm": float(final_error_norm),
        "settling_time_s": float(settling_time_s),
        "settling_step": int(settling_step),
        "stable_fraction_under_tol": float(stable_fraction_under_tol),
        "control_energy_total": float(ctrl_energy_total),
        "control_energy_mean": float(ctrl_energy_mean),
        "max_abs_x_post": float(max_abs_x_post),
        "diverged": bool(diverged),
    }
