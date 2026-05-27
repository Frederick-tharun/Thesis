from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np

from plotting import (
    plot_controlled_vs_uncontrolled_x,
    plot_controlled_all_states,
    plot_control_signal,
    plot_control_error,
    plot_k_sweep_summary,
)


def _as_1d(x):
    return np.asarray(x, dtype=float).reshape(-1)


def _as_2d(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _json_safe(x: Any):
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    if isinstance(x, tuple):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, float):
        if not np.isfinite(x):
            return None
        return x
    return x


def _save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(_json_safe(obj), f, indent=2)

    print(f"[Save] -> {path}")


def _save_csv(rows, path):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Save] -> {path}")


def _save_rollout_csv(
    path,
    times,
    truth,
    uncontrolled,
    controlled,
    control_signal,
    target_state,
):
    times = _as_1d(times)
    truth = _as_2d(truth)
    uncontrolled = _as_2d(uncontrolled)
    controlled = _as_2d(controlled)
    control_signal = _as_2d(control_signal)
    target_state = _as_1d(target_state)

    n = min(
        len(times),
        len(truth),
        len(uncontrolled),
        len(controlled),
        len(control_signal),
    )

    rows = []

    for i in range(n):
        rows.append(
            {
                "time": float(times[i]),
                "true_x": float(truth[i, 0]),
                "true_y": float(truth[i, 1]),
                "true_z": float(truth[i, 2]),
                "uncontrolled_x": float(uncontrolled[i, 0]),
                "uncontrolled_y": float(uncontrolled[i, 1]),
                "uncontrolled_z": float(uncontrolled[i, 2]),
                "controlled_x": float(controlled[i, 0]),
                "controlled_y": float(controlled[i, 1]),
                "controlled_z": float(controlled[i, 2]),
                "u_x": float(control_signal[i, 0]),
                "u_y": float(control_signal[i, 1]),
                "u_z": float(control_signal[i, 2]),
                "target_x": float(target_state[0]),
                "target_y": float(target_state[1]),
                "target_z": float(target_state[2]),
            }
        )

    _save_csv(rows, path)


def _slug_k(k):
    return f"{float(k):.6f}".replace("-", "m").replace(".", "p")


def _denormalize(arr, mean, std):
    arr = _as_2d(arr)
    return arr * np.asarray(std, dtype=float) + np.asarray(mean, dtype=float)


def _compute_error_norms(states, target_state):
    states = _as_2d(states)
    target_state = _as_1d(target_state)
    return np.linalg.norm(states - target_state.reshape(1, -1), axis=1)


def _rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _count_spikes(x, threshold):
    x = _as_1d(x)

    above = x > threshold
    if len(above) < 3:
        return int(np.sum(above))

    peaks = 0

    for i in range(1, len(x) - 1):
        if above[i] and x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            peaks += 1

    return int(peaks)


def _settling_time(times, error_norm, control_start_idx, tolerance, consecutive):
    times = _as_1d(times)
    error_norm = _as_1d(error_norm)

    n = min(len(times), len(error_norm))
    times = times[:n]
    error_norm = error_norm[:n]

    start = int(max(0, min(control_start_idx, n - 1)))
    consecutive = int(max(1, consecutive))

    for i in range(start, n - consecutive):
        window = error_norm[i : i + consecutive]
        if np.all(window <= tolerance):
            return float(times[i] - times[start])

    return float("nan")


def _choose_target_state(train_raw, mean, std, target_mode, config):
    """
    Target is chosen in original/raw scale first, then normalized.
    """
    train_raw = _as_2d(train_raw)

    spike_threshold = float(getattr(config, "SPIKE_THRESHOLD", 2.0))
    x = train_raw[:, 0]

    if target_mode == "rest_state":
        mask = x < spike_threshold

        if np.count_nonzero(mask) < max(25, len(x) // 100):
            cutoff = np.percentile(x, 70.0)
            mask = x <= cutoff

        if np.count_nonzero(mask) == 0:
            target_raw = np.median(train_raw, axis=0)
        else:
            target_raw = np.median(train_raw[mask], axis=0)

    elif target_mode == "zero":
        target_raw = np.zeros(train_raw.shape[1], dtype=float)

    elif target_mode == "mean":
        target_raw = np.mean(train_raw, axis=0)

    else:
        raise ValueError(f"Unknown target mode: {target_mode}")

    target_norm = ((target_raw.reshape(1, -1) - mean) / std).reshape(-1)

    return target_raw.reshape(-1), target_norm.reshape(-1)


def _get_k_values(control_k, config):
    if control_k is not None:
        return [float(control_k)]

    values = getattr(config, "CONTROL_LINEAR_K_SWEEP", [0.05, 0.10, 0.20, 0.50, 1.00])

    clean = []

    for v in values:
        v = float(v)
        if v >= 0.0 and v not in clean:
            clean.append(v)

    return sorted(clean)


def _summarize_control_metrics(
    times,
    uncontrolled,
    controlled,
    target_state,
    control_signal,
    control_start_idx,
    spike_threshold,
    settling_tolerance,
    settling_consecutive,
):
    times = _as_1d(times)
    uncontrolled = _as_2d(uncontrolled)
    controlled = _as_2d(controlled)
    target_state = _as_1d(target_state)
    control_signal = _as_2d(control_signal)

    n = min(len(times), len(uncontrolled), len(controlled), len(control_signal))
    times = times[:n]
    uncontrolled = uncontrolled[:n]
    controlled = controlled[:n]
    control_signal = control_signal[:n]

    start = int(max(0, min(control_start_idx, n - 1)))

    uncontrolled_post = uncontrolled[start:]
    controlled_post = controlled[start:]
    control_post = control_signal[start:]

    target_post = np.tile(target_state.reshape(1, -1), (len(controlled_post), 1))

    uncontrolled_error_norm = _compute_error_norms(uncontrolled, target_state)
    controlled_error_norm = _compute_error_norms(controlled, target_state)

    target_rmse_state = _rmse(controlled_post, target_post)
    uncontrolled_target_rmse_state = _rmse(uncontrolled_post, target_post)

    target_rmse_x = _rmse(controlled_post[:, 0], target_post[:, 0])
    uncontrolled_target_rmse_x = _rmse(uncontrolled_post[:, 0], target_post[:, 0])

    spike_count_before = _count_spikes(controlled[:start, 0], spike_threshold)
    spike_count_after = _count_spikes(controlled_post[:, 0], spike_threshold)
    uncontrolled_spike_count_after = _count_spikes(uncontrolled_post[:, 0], spike_threshold)

    if uncontrolled_spike_count_after > 0:
        spike_reduction_percent = (
            100.0
            * (uncontrolled_spike_count_after - spike_count_after)
            / uncontrolled_spike_count_after
        )
    else:
        spike_reduction_percent = 0.0

    control_norm = np.linalg.norm(control_post, axis=1)

    control_energy = float(np.mean(control_norm**2))
    mean_control_norm = float(np.mean(control_norm))
    max_control_norm = float(np.max(control_norm))

    settling = _settling_time(
        times=times,
        error_norm=controlled_error_norm,
        control_start_idx=start,
        tolerance=settling_tolerance,
        consecutive=settling_consecutive,
    )

    return {
        "target_rmse_state": float(target_rmse_state),
        "target_rmse_x": float(target_rmse_x),
        "uncontrolled_target_rmse_state": float(uncontrolled_target_rmse_state),
        "uncontrolled_target_rmse_x": float(uncontrolled_target_rmse_x),
        "spike_count_before": int(spike_count_before),
        "spike_count_after": int(spike_count_after),
        "uncontrolled_spike_count_after": int(uncontrolled_spike_count_after),
        "spike_reduction_percent": float(spike_reduction_percent),
        "control_energy": float(control_energy),
        "mean_control_norm": float(mean_control_norm),
        "max_control_norm": float(max_control_norm),
        "settling_time": float(settling),
        "mean_error_norm_post": float(np.mean(controlled_error_norm[start:])),
        "max_error_norm_post": float(np.max(controlled_error_norm[start:])),
    }



def _append_global_control_comparison(output_root, row):
    os.makedirs(output_root, exist_ok=True)

    path = os.path.join(output_root, "control_comparison.csv")

    existing_rows = []

    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    new_rows = []
    replaced = False

    for old in existing_rows:
        if old.get("regime") == row.get("regime"):
            new_rows.append(row)
            replaced = True
        else:
            new_rows.append(old)

    if not replaced:
        new_rows.append(row)

    fieldnames = []
    for r in new_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"[Control Report] Saved global comparison -> {path}")


def _score_row(row):
    rmse = float(row.get("target_rmse_state", np.inf))
    energy = float(row.get("control_energy", np.inf))
    return rmse, energy


def _fmt_num(x):
    try:
        x = float(x)
        if np.isfinite(x):
            return f"{x:.6f}"
    except Exception:
        pass
    return "nan"

def _safe_float(value, default=np.inf):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return float(default)


def _selection_score(row, config):
    """
    Lower score = better K.

    We choose K based on:
    - low target tracking error
    - low control energy
    - fast settling time

    Spike reduction is saved as a metric, but by default it is not used
    in the score because fixed spike thresholds can be misleading for HR.
    """
    if not bool(row.get("stable", False)):
        return float("inf")

    rmse = _safe_float(row.get("target_rmse_state"), np.inf)
    energy = _safe_float(row.get("control_energy"), np.inf)
    settling = _safe_float(row.get("settling_time"), np.inf)
    spike_reduction = _safe_float(row.get("spike_reduction_percent"), 0.0)

    energy_weight = float(getattr(config, "CONTROL_SCORE_ENERGY_WEIGHT", 0.01))
    settling_weight = float(getattr(config, "CONTROL_SCORE_SETTLING_WEIGHT", 0.001))
    spike_weight = float(getattr(config, "CONTROL_SCORE_SPIKE_WEIGHT", 0.0))

    score = (
        rmse
        + energy_weight * energy
        + settling_weight * settling
        - spike_weight * (spike_reduction / 100.0)
    )

    return float(score)


def _best_row(rows, config):
    if not rows:
        raise ValueError("No control rows available for K selection.")

    for row in rows:
        row["selection_score"] = _selection_score(row, config)

    stable_rows = [
        r
        for r in rows
        if bool(r.get("stable", False))
        and np.isfinite(_safe_float(r.get("selection_score"), np.inf))
    ]

    return min(
        stable_rows or rows,
        key=lambda r: _safe_float(r.get("selection_score"), np.inf),
    )


def _unique_sorted(values, ndigits=10):
    clean = []
    seen = set()

    for v in values:
        v = float(v)

        if not np.isfinite(v) or v < 0.0:
            continue

        key = round(v, ndigits)

        if key not in seen:
            clean.append(v)
            seen.add(key)

    return sorted(clean)


def _auto_k_coarse_values(config, k_min=None, k_max=None, k_num=None):
    k_min = float(
        k_min if k_min is not None else getattr(config, "CONTROL_AUTO_K_MIN", 0.05)
    )
    k_max = float(
        k_max if k_max is not None else getattr(config, "CONTROL_AUTO_K_MAX", 2.0)
    )
    k_num = int(
        k_num if k_num is not None else getattr(config, "CONTROL_AUTO_K_NUM", 25)
    )

    k_min = max(0.0, k_min)
    k_max = max(k_min, k_max)
    k_num = max(2, k_num)

    return _unique_sorted(np.linspace(k_min, k_max, k_num))


def _auto_k_refined_values(best_k, config, k_min=None, k_max=None, k_refine_num=None):
    k_min = float(
        k_min if k_min is not None else getattr(config, "CONTROL_AUTO_K_MIN", 0.05)
    )
    k_max = float(
        k_max if k_max is not None else getattr(config, "CONTROL_AUTO_K_MAX", 2.0)
    )
    k_refine_num = int(
        k_refine_num
        if k_refine_num is not None
        else getattr(config, "CONTROL_AUTO_K_REFINE_NUM", 15)
    )

    k_min = max(0.0, k_min)
    k_max = max(k_min, k_max)
    k_refine_num = max(3, k_refine_num)

    width_frac = float(getattr(config, "CONTROL_AUTO_K_REFINE_WIDTH_FRAC", 0.15))
    half_width = max((k_max - k_min) * width_frac, 1e-6)

    lo = max(k_min, float(best_k) - half_width)
    hi = min(k_max, float(best_k) + half_width)

    return _unique_sorted(np.linspace(lo, hi, k_refine_num))


def _evaluate_linear_feedback_k(
    *,
    esn,
    train_norm,
    n_base,
    K,
    target_norm,
    target_raw,
    mean,
    std,
    uncontrolled,
    test_aligned,
    test_times_aligned,
    control_start_idx,
    control_start_time,
    control_start_frac,
    control_target_mode,
    hr_mode,
    optimizer_name,
    spike_threshold,
    settling_to