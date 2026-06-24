from __future__ import annotations

import csv
import inspect
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

SUPPORTED_CONTROLLERS = ("linear_feedback", "finite_time", "pyragas")


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
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, float):
        return x if np.isfinite(x) else None
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


def _save_rollout_csv(path, times, truth, uncontrolled, controlled, control_signal, target_state):
    times = _as_1d(times)
    truth = _as_2d(truth)
    uncontrolled = _as_2d(uncontrolled)
    controlled = _as_2d(controlled)
    control_signal = _as_2d(control_signal)
    target_state = _as_1d(target_state)

    n = min(len(times), len(truth), len(uncontrolled), len(controlled), len(control_signal))
    rows = []
    for i in range(n):
        row = {
            "time": float(times[i]),
            "target_x": float(target_state[0]),
            "target_y": float(target_state[1]),
            "target_z": float(target_state[2]),
        }
        for j, name in enumerate(["x", "y", "z"]):
            row[f"true_{name}"] = float(truth[i, j])
            row[f"uncontrolled_{name}"] = float(uncontrolled[i, j])
            row[f"controlled_{name}"] = float(controlled[i, j])
            row[f"u_{name}"] = float(control_signal[i, j])
        rows.append(row)

    _save_csv(rows, path)


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


def _peak_indices(x, threshold):
    """
    Return local-maximum spike indices for one signal.

    This is intentionally separate from _count_spikes so that Pyragas can also
    measure spike-interval regularity. Linear feedback and finite-time control
    still use the original _count_spikes behaviour.
    """
    x = _as_1d(x)
    if len(x) < 3:
        return np.asarray([], dtype=int)

    above = np.isfinite(x) & (x > float(threshold))
    if not np.any(above):
        return np.asarray([], dtype=int)

    transitions = np.diff(np.concatenate(([False], above, [False])).astype(int))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)

    peaks = []
    for left, right in zip(starts, stops):
        if right > left:
            peaks.append(int(left + np.argmax(x[left:right])))

    return np.asarray(peaks, dtype=int)


def _coefficient_of_variation(values):
    values = _as_1d(values)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")

    mean = float(np.mean(values))
    if abs(mean) < 1e-12:
        return float("nan")

    return float(np.std(values) / abs(mean))


def _pyragas_cycle_markers(peaks):
    """Return spike or burst-cycle markers for an HR trajectory."""
    peaks = np.asarray(peaks, dtype=int).reshape(-1)
    if len(peaks) < 2:
        return "undetermined", peaks

    intervals = np.diff(peaks).astype(float)
    median_interval = float(np.median(intervals))
    mad = float(np.median(np.abs(intervals - median_interval)))
    gap_threshold = max(2.0 * median_interval, median_interval + 3.0 * max(mad, 1.0))
    long_gap_indices = np.flatnonzero(intervals > gap_threshold)

    # At least two large inter-burst gaps are needed before calling the rhythm
    # bursting. Otherwise, each detected spike is treated as one cycle marker.
    if len(long_gap_indices) >= 2:
        burst_starts = np.concatenate(([peaks[0]], peaks[long_gap_indices + 1]))
        return "bursting", burst_starts.astype(int)

    return "spiking", peaks


def _pyragas_recurrence_metrics(states, lag):
    """Measure full-state recurrence at an empirically detected cycle lag."""
    states = _as_2d(states)
    lag = int(lag)
    if lag <= 0 or len(states) <= 2 * lag:
        return float("nan"), float("nan"), float("nan")

    centered = states - np.mean(states, axis=0, keepdims=True)
    scale = max(float(np.sqrt(np.mean(centered**2))), 1e-12)
    current = states[lag:]
    delayed = states[:-lag]
    recurrence_error = float(np.sqrt(np.mean((current - delayed) ** 2))) / scale

    current_centered = centered[lag:]
    delayed_centered = centered[:-lag]
    denominator = float(
        np.sqrt(np.sum(current_centered**2) * np.sum(delayed_centered**2))
    )
    recurrence_correlation = (
        float(np.sum(current_centered * delayed_centered) / denominator)
        if denominator > 1e-12
        else float("nan")
    )

    previous_cycle = states[-2 * lag:-lag]
    final_cycle = states[-lag:]
    tail_closure = float(np.sqrt(np.mean((final_cycle - previous_cycle) ** 2))) / scale
    return recurrence_error, recurrence_correlation, tail_closure


def _pyragas_dynamics_metrics(
    controlled,
    uncontrolled,
    control_signal,
    times,
    control_start_idx,
    pyragas_delay,
    spike_threshold,
):
    """Measure sustained post-transient Pyragas behaviour.

    Metrics are intentionally evaluated after a settling allowance. Otherwise,
    a few regularly spaced transient peaks can produce a very small interval CV
    even when the trajectory later drifts to a fixed state.
    """
    controlled = _as_2d(controlled)
    uncontrolled = _as_2d(uncontrolled)
    control_signal = _as_2d(control_signal)
    times = _as_1d(times)

    n = min(len(controlled), len(uncontrolled), len(control_signal), len(times))
    controlled = controlled[:n]
    uncontrolled = uncontrolled[:n]
    control_signal = control_signal[:n]
    times = times[:n]

    empty = {
            "pyragas_periodicity_rmse_state": float("nan"),
            "pyragas_periodicity_rmse_x": float("nan"),
            "pyragas_periodicity_rmse_state_norm": float("nan"),
            "pyragas_periodicity_rmse_x_norm": float("nan"),
            "pyragas_x_amplitude_post": float("nan"),
            "pyragas_uncontrolled_x_amplitude_post": float("nan"),
            "pyragas_x_amplitude_ratio": float("nan"),
            "pyragas_x_std_post": float("nan"),
            "pyragas_uncontrolled_x_std_post": float("nan"),
            "pyragas_x_std_ratio": float("nan"),
            "pyragas_spike_interval_cv": float("nan"),
            "pyragas_spike_interval_mean": float("nan"),
            "pyragas_spike_interval_std": float("nan"),
            "pyragas_spike_interval_mean_time": float("nan"),
            "pyragas_delay_time": float("nan"),
            "pyragas_delay_period_mismatch": float("nan"),
            "pyragas_detected_peak_count": 0,
            "pyragas_uncontrolled_detected_peak_count": 0,
            "pyragas_rhythm_type": "undetermined",
            "pyragas_detected_cycle_count": 0,
            "pyragas_cycle_counts_by_window": [0, 0, 0, 0],
            "pyragas_cycle_window_coverage": 0.0,
            "pyragas_rhythm_interval_mean": float("nan"),
            "pyragas_rhythm_interval_std": float("nan"),
            "pyragas_rhythm_interval_cv": float("nan"),
            "pyragas_empirical_period_steps": float("nan"),
            "pyragas_empirical_period_time": float("nan"),
            "pyragas_empirical_recurrence_error_norm": float("nan"),
            "pyragas_empirical_recurrence_correlation": float("nan"),
            "pyragas_empirical_tail_closure_error_norm": float("nan"),
            "pyragas_control_tail_rms": float("nan"),
            "pyragas_control_decay_ratio": float("nan"),
            "pyragas_noninvasiveness_ratio": float("nan"),
            "pyragas_evaluation_start_idx": 0,
            "pyragas_evaluation_start_time": float("nan"),
            "pyragas_transient_samples_ignored": 0,
            "pyragas_peak_window_coverage": 0.0,
            "pyragas_peak_counts_by_window": [0, 0, 0, 0],
            "pyragas_peak_amplitude_cv": float("nan"),
            "pyragas_window_amplitude_cv": float("nan"),
            "pyragas_drift_ratio": float("nan"),
            "pyragas_tail_activity_ratio": float("nan"),
            "pyragas_tail_closure_error_norm": float("nan"),
        }

    if n == 0:
        return empty

    start = int(max(0, min(control_start_idx, n - 1)))
    delay = int(max(1, pyragas_delay))

    post_len = n - start
    desired_transient = max(2 * delay, int(round(0.25 * post_len)))
    minimum_eval_len = max(delay + 3, 16)
    max_transient = max(0, post_len - minimum_eval_len)
    transient_samples = min(desired_transient, max_transient)
    eval_start = start + transient_samples

    controlled_post = controlled[eval_start:]
    uncontrolled_post = uncontrolled[eval_start:]
    control_post = control_signal[eval_start:]

    if len(controlled_post) < 3:
        result = dict(empty)
        result.update(
            {
                "pyragas_evaluation_start_idx": int(eval_start),
                "pyragas_evaluation_start_time": (
                    float(times[eval_start]) if eval_start < len(times) else float("nan")
                ),
                "pyragas_transient_samples_ignored": int(transient_samples),
            }
        )
        return result

    controlled_x = controlled_post[:, 0]
    uncontrolled_x = uncontrolled_post[:, 0]

    x_amp = float(np.max(controlled_x) - np.min(controlled_x)) if len(controlled_x) else float("nan")
    unctrl_x_amp = float(np.max(uncontrolled_x) - np.min(uncontrolled_x)) if len(uncontrolled_x) else float("nan")
    x_std = float(np.std(controlled_x)) if len(controlled_x) else float("nan")
    unctrl_x_std = float(np.std(uncontrolled_x)) if len(uncontrolled_x) else float("nan")
    state_scale = max(
        float(np.sqrt(np.mean((controlled_post - np.mean(controlled_post, axis=0)) ** 2))),
        1e-12,
    )

    amp_ratio = x_amp / max(unctrl_x_amp, 1e-12) if np.isfinite(x_amp) and np.isfinite(unctrl_x_amp) else float("nan")
    std_ratio = x_std / max(unctrl_x_std, 1e-12) if np.isfinite(x_std) and np.isfinite(unctrl_x_std) else float("nan")

    if len(controlled_post) > delay:
        current = controlled_post[delay:]
        delayed = controlled_post[:-delay]
        diff = current - delayed

        periodicity_state = float(np.sqrt(np.mean(diff**2)))
        periodicity_x = float(np.sqrt(np.mean(diff[:, 0] ** 2)))

        # Normalize by the post-control signal scale. This prevents a flat/rest
        # signal from looking artificially good just because all values are small.
        x_scale = max(float(np.std(controlled_x)), 1e-12)
        periodicity_state_norm = periodicity_state / state_scale
        periodicity_x_norm = periodicity_x / x_scale
    else:
        periodicity_state = float("nan")
        periodicity_x = float("nan")
        periodicity_state_norm = float("nan")
        periodicity_x_norm = float("nan")

    peaks = _peak_indices(controlled_x, spike_threshold)
    uncontrolled_peaks = _peak_indices(uncontrolled_x, spike_threshold)
    if len(peaks) >= 3:
        intervals = np.diff(peaks).astype(float)
        interval_mean = float(np.mean(intervals))
        interval_std = float(np.std(intervals))
        interval_cv = _coefficient_of_variation(intervals)
        delay_period_mismatch = abs(interval_mean - delay) / max(interval_mean, float(delay), 1e-12)
    else:
        interval_mean = float("nan")
        interval_std = float("nan")
        interval_cv = float("nan")
        delay_period_mismatch = float("nan")

    if len(times) >= 2:
        positive_steps = np.diff(times)
        positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0.0)]
        sample_time = float(np.median(positive_steps)) if len(positive_steps) else float("nan")
    else:
        sample_time = float("nan")

    delay_time = delay * sample_time if np.isfinite(sample_time) else float("nan")
    interval_mean_time = (
        interval_mean * sample_time
        if np.isfinite(interval_mean) and np.isfinite(sample_time)
        else float("nan")
    )

    rhythm_type, cycle_markers = _pyragas_cycle_markers(peaks)
    if len(cycle_markers) >= 3:
        rhythm_intervals = np.diff(cycle_markers).astype(float)
        rhythm_interval_mean = float(np.mean(rhythm_intervals))
        rhythm_interval_std = float(np.std(rhythm_intervals))
        rhythm_interval_cv = _coefficient_of_variation(rhythm_intervals)
        empirical_period_steps = int(max(1, round(float(np.median(rhythm_intervals)))))
    else:
        rhythm_interval_mean = float("nan")
        rhythm_interval_std = float("nan")
        rhythm_interval_cv = float("nan")
        empirical_period_steps = 0

    empirical_period_time = (
        empirical_period_steps * sample_time
        if empirical_period_steps > 0 and np.isfinite(sample_time)
        else float("nan")
    )
    (
        empirical_recurrence_error,
        empirical_recurrence_correlation,
        empirical_tail_closure,
    ) = _pyragas_recurrence_metrics(controlled_post, empirical_period_steps)

    # Sustained activity must be present across the evaluation interval, not
    # concentrated in the initial transient.
    window_edges = np.linspace(0, len(controlled_x), 5, dtype=int)
    peak_counts_by_window = []
    cycle_counts_by_window = []
    window_amplitudes = []
    window_means = []
    for window_idx in range(4):
        left = int(window_edges[window_idx])
        right = int(window_edges[window_idx + 1])
        segment = controlled_x[left:right]
        peak_counts_by_window.append(int(np.sum((peaks >= left) & (peaks < right))))
        cycle_counts_by_window.append(
            int(np.sum((cycle_markers >= left) & (cycle_markers < right)))
        )
        if len(segment):
            window_amplitudes.append(float(np.max(segment) - np.min(segment)))
            window_means.append(float(np.mean(segment)))

    peak_window_coverage = float(np.mean(np.asarray(peak_counts_by_window) > 0))
    cycle_window_coverage = float(np.mean(np.asarray(cycle_counts_by_window) > 0))
    peak_amplitude_cv = _coefficient_of_variation(controlled_x[peaks]) if len(peaks) >= 3 else float("nan")
    window_amplitude_cv = (
        _coefficient_of_variation(window_amplitudes)
        if len(window_amplitudes) >= 2
        else float("nan")
    )
    drift_ratio = (
        abs(float(np.polyfit(np.linspace(0.0, 1.0, len(window_means)), window_means, 1)[0]))
        / max(x_amp, 1e-12)
        if len(window_means) >= 2
        else float("nan")
    )

    tail_len = max(1, len(controlled_x) // 4)
    tail_activity_ratio = float(np.std(controlled_x[-tail_len:])) / max(x_std, 1e-12)

    if len(controlled_post) >= 2 * delay:
        previous_cycle = controlled_post[-2 * delay:-delay]
        final_cycle = controlled_post[-delay:]
        tail_closure_error = float(np.sqrt(np.mean((final_cycle - previous_cycle) ** 2)))
        tail_closure_error_norm = tail_closure_error / state_scale
    else:
        tail_closure_error_norm = float("nan")

    # Classical Pyragas feedback should become small after the target orbit is
    # stabilized. Measure this on the post-transient evaluation interval.
    if len(control_post):
        control_tail_len = max(1, len(control_post) // 4)
        early_len = min(control_tail_len, len(control_post))
        early_control = control_post[:early_len]
        tail_control = control_post[-control_tail_len:]
        early_control_rms = float(np.sqrt(np.mean(np.sum(early_control**2, axis=1))))
        tail_control_rms = float(np.sqrt(np.mean(np.sum(tail_control**2, axis=1))))
        control_decay_ratio = tail_control_rms / max(early_control_rms, 1e-12)
        noninvasiveness_ratio = tail_control_rms / state_scale
    else:
        tail_control_rms = float("nan")
        control_decay_ratio = float("nan")
        noninvasiveness_ratio = float("nan")

    return {
        "pyragas_periodicity_rmse_state": periodicity_state,
        "pyragas_periodicity_rmse_x": periodicity_x,
        "pyragas_periodicity_rmse_state_norm": periodicity_state_norm,
        "pyragas_periodicity_rmse_x_norm": periodicity_x_norm,
        "pyragas_x_amplitude_post": x_amp,
        "pyragas_uncontrolled_x_amplitude_post": unctrl_x_amp,
        "pyragas_x_amplitude_ratio": float(amp_ratio),
        "pyragas_x_std_post": x_std,
        "pyragas_uncontrolled_x_std_post": unctrl_x_std,
        "pyragas_x_std_ratio": float(std_ratio),
        "pyragas_spike_interval_cv": interval_cv,
        "pyragas_spike_interval_mean": interval_mean,
        "pyragas_spike_interval_std": interval_std,
        "pyragas_spike_interval_mean_time": interval_mean_time,
        "pyragas_delay_time": delay_time,
        "pyragas_delay_period_mismatch": float(delay_period_mismatch),
        "pyragas_detected_peak_count": int(len(peaks)),
        "pyragas_uncontrolled_detected_peak_count": int(len(uncontrolled_peaks)),
        "pyragas_rhythm_type": rhythm_type,
        "pyragas_detected_cycle_count": int(len(cycle_markers)),
        "pyragas_cycle_counts_by_window": cycle_counts_by_window,
        "pyragas_cycle_window_coverage": cycle_window_coverage,
        "pyragas_rhythm_interval_mean": rhythm_interval_mean,
        "pyragas_rhythm_interval_std": rhythm_interval_std,
        "pyragas_rhythm_interval_cv": rhythm_interval_cv,
        "pyragas_empirical_period_steps": int(empirical_period_steps),
        "pyragas_empirical_period_time": empirical_period_time,
        "pyragas_empirical_recurrence_error_norm": empirical_recurrence_error,
        "pyragas_empirical_recurrence_correlation": empirical_recurrence_correlation,
        "pyragas_empirical_tail_closure_error_norm": empirical_tail_closure,
        "pyragas_control_tail_rms": tail_control_rms,
        "pyragas_control_decay_ratio": float(control_decay_ratio),
        "pyragas_noninvasiveness_ratio": float(noninvasiveness_ratio),
        "pyragas_evaluation_start_idx": int(eval_start),
        "pyragas_evaluation_start_time": float(times[eval_start]),
        "pyragas_transient_samples_ignored": int(transient_samples),
        "pyragas_peak_window_coverage": peak_window_coverage,
        "pyragas_peak_counts_by_window": peak_counts_by_window,
        "pyragas_peak_amplitude_cv": peak_amplitude_cv,
        "pyragas_window_amplitude_cv": window_amplitude_cv,
        "pyragas_drift_ratio": drift_ratio,
        "pyragas_tail_activity_ratio": tail_activity_ratio,
        "pyragas_tail_closure_error_norm": tail_closure_error_norm,
    }
def _safe_float(value, default=np.inf):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return float(default)


def _count_spikes(x, threshold):
    x = _as_1d(x)
    if len(x) < 3:
        return 0

    peaks = 0
    for i in range(1, len(x) - 1):
        if x[i] > threshold and x[i] >= x[i - 1] and x[i] >= x[i + 1]:
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

    last_start = n - consecutive
    if start > last_start:
        return float("nan")

    for i in range(start, last_start + 1):
        if np.all(error_norm[i : i + consecutive] <= tolerance):
            return float(times[i] - times[start])

    return float("nan")


def _choose_target_state(train_raw, mean, std, target_mode, config):
    train_raw = _as_2d(train_raw)

    spike_threshold = float(getattr(config, "SPIKE_THRESHOLD", 2.0))
    x = train_raw[:, 0]

    if target_mode == "rest_state":
        mask = x < spike_threshold

        if np.count_nonzero(mask) < max(25, len(x) // 100):
            cutoff = np.percentile(x, 70.0)
            mask = x <= cutoff

        if np.count_nonzero(mask):
            target_raw = np.median(train_raw[mask], axis=0)
        else:
            target_raw = np.median(train_raw, axis=0)

    elif target_mode == "zero":
        target_raw = np.zeros(train_raw.shape[1], dtype=float)

    elif target_mode == "mean":
        target_raw = np.mean(train_raw, axis=0)

    else:
        raise ValueError(f"Unknown target mode: {target_mode}")

    target_norm = ((target_raw.reshape(1, -1) - mean) / std).reshape(-1)
    return target_raw.reshape(-1), target_norm.reshape(-1)


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


def _k_values(control_k, config, k_min=None, k_max=None, k_num=None):
    if control_k is not None:
        return [float(control_k)]

    if k_min is None and k_max is None and k_num is None:
        vals = getattr(config, "CONTROL_LINEAR_K_SWEEP", None)
        if vals is not None:
            return _unique_sorted(vals)

    k_min = float(k_min if k_min is not None else getattr(config, "CONTROL_AUTO_K_MIN", 0.05))
    k_max = float(k_max if k_max is not None else getattr(config, "CONTROL_AUTO_K_MAX", 2.0))
    k_num = int(k_num if k_num is not None else getattr(config, "CONTROL_AUTO_K_NUM", 25))

    return _unique_sorted(np.linspace(max(0.0, k_min), max(k_min, k_max), max(2, k_num)))


def _refined_k_values(best_k, config, k_min=None, k_max=None, k_refine_num=None):
    k_min = float(k_min if k_min is not None else getattr(config, "CONTROL_AUTO_K_MIN", 0.05))
    k_max = float(k_max if k_max is not None else getattr(config, "CONTROL_AUTO_K_MAX", 2.0))
    k_refine_num = int(
        k_refine_num if k_refine_num is not None else getattr(config, "CONTROL_AUTO_K_REFINE_NUM", 15)
    )

    width_frac = float(getattr(config, "CONTROL_AUTO_K_REFINE_WIDTH_FRAC", 0.15))
    half_width = max((k_max - k_min) * width_frac, 1e-6)

    lo = max(k_min, float(best_k) - half_width)
    hi = min(k_max, float(best_k) + half_width)

    return _unique_sorted(np.linspace(lo, hi, max(3, k_refine_num)))


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

    control_norm = np.linalg.norm(control_post, axis=1)

    spike_after = _count_spikes(controlled_post[:, 0], spike_threshold)
    unctrl_spike_after = _count_spikes(uncontrolled_post[:, 0], spike_threshold)

    if unctrl_spike_after > 0:
        spike_reduction = 100.0 * (unctrl_spike_after - spike_after) / unctrl_spike_after
    else:
        spike_reduction = 0.0

    return {
        "target_rmse_state": _rmse(controlled_post, target_post),
        "target_rmse_x": _rmse(controlled_post[:, 0], target_post[:, 0]),
        "uncontrolled_target_rmse_state": _rmse(uncontrolled_post, target_post),
        "uncontrolled_target_rmse_x": _rmse(uncontrolled_post[:, 0], target_post[:, 0]),
        "spike_count_before": _count_spikes(controlled[:start, 0], spike_threshold),
        "spike_count_after": spike_after,
        "uncontrolled_spike_count_after": unctrl_spike_after,
        "spike_reduction_percent": float(spike_reduction),
        "control_energy": float(np.mean(control_norm**2)) if len(control_norm) else float("nan"),
        "mean_control_norm": float(np.mean(control_norm)) if len(control_norm) else float("nan"),
        "max_control_norm": float(np.max(control_norm)) if len(control_norm) else float("nan"),
        "settling_time": _settling_time(
            times,
            controlled_error_norm,
            start,
            settling_tolerance,
            settling_consecutive,
        ),
        "mean_error_norm_post": float(np.mean(controlled_error_norm[start:])),
        "max_error_norm_post": float(np.max(controlled_error_norm[start:])),
    }


def _selection_score(row, config):
    if not bool(row.get("stable", False)):
        return float("inf")

    controller = str(row.get("controller", "")).strip().lower()

    if controller == "pyragas":
        peak_count = int(_safe_float(row.get("pyragas_detected_peak_count"), 0.0))
        cycle_count = int(_safe_float(row.get("pyragas_detected_cycle_count"), 0.0))
        uncontrolled_peak_count = int(
            _safe_float(row.get("pyragas_uncontrolled_detected_peak_count"), 0.0)
        )
        rhythm_type = str(row.get("pyragas_rhythm_type", "undetermined"))

        amp_ratio = _safe_float(row.get("pyragas_x_amplitude_ratio"), 0.0)
        std_ratio = _safe_float(row.get("pyragas_x_std_ratio"), 0.0)
        rhythm_cv = _safe_float(row.get("pyragas_rhythm_interval_cv"), np.inf)
        peak_coverage = _safe_float(row.get("pyragas_peak_window_coverage"), 0.0)
        cycle_coverage = _safe_float(row.get("pyragas_cycle_window_coverage"), 0.0)
        peak_amplitude_cv = _safe_float(row.get("pyragas_peak_amplitude_cv"), np.inf)
        window_amplitude_cv = _safe_float(row.get("pyragas_window_amplitude_cv"), np.inf)
        drift_ratio = _safe_float(row.get("pyragas_drift_ratio"), np.inf)
        tail_activity = _safe_float(row.get("pyragas_tail_activity_ratio"), 0.0)
        empirical_recurrence = _safe_float(
            row.get("pyragas_empirical_recurrence_error_norm"), np.inf
        )
        empirical_correlation = _safe_float(
            row.get("pyragas_empirical_recurrence_correlation"), -1.0
        )
        empirical_closure = _safe_float(
            row.get("pyragas_empirical_tail_closure_error_norm"), np.inf
        )
        energy = _safe_float(row.get("control_energy"), 0.0)
        max_control = _safe_float(row.get("max_control_norm"), 0.0)
        K = _safe_float(row.get("K"), 0.0)

        min_peaks = int(getattr(config, "PYRAGAS_MIN_EVALUATION_PEAKS", 6))
        min_cycles = int(getattr(config, "PYRAGAS_MIN_EVALUATION_CYCLES", 3))
        min_amp_ratio = float(getattr(config, "PYRAGAS_MIN_AMPLITUDE_RATIO", 0.20))
        min_std_ratio = float(getattr(config, "PYRAGAS_MIN_STD_RATIO", 0.15))
        preferred_amp_min = float(getattr(config, "PYRAGAS_PREFERRED_AMPLITUDE_MIN", 0.80))
        preferred_amp_max = float(getattr(config, "PYRAGAS_PREFERRED_AMPLITUDE_MAX", 1.30))
        preferred_std_min = float(getattr(config, "PYRAGAS_PREFERRED_STD_MIN", 0.50))
        preferred_std_max = float(getattr(config, "PYRAGAS_PREFERRED_STD_MAX", 1.50))
        target_rhythm_cv = float(getattr(config, "PYRAGAS_TARGET_RHYTHM_CV", 0.10))
        target_peak_coverage = float(getattr(config, "PYRAGAS_TARGET_PEAK_WINDOW_COVERAGE", 0.75))
        target_cycle_coverage = float(getattr(config, "PYRAGAS_TARGET_CYCLE_WINDOW_COVERAGE", 0.50))
        max_peak_amplitude_cv = float(getattr(config, "PYRAGAS_MAX_PEAK_AMPLITUDE_CV", 0.25))
        max_window_amplitude_cv = float(getattr(config, "PYRAGAS_MAX_WINDOW_AMPLITUDE_CV", 0.50))
        max_drift_ratio = float(getattr(config, "PYRAGAS_MAX_DRIFT_RATIO", 0.20))
        min_tail_activity = float(getattr(config, "PYRAGAS_MIN_TAIL_ACTIVITY_RATIO", 0.50))
        max_empirical_recurrence = float(
            getattr(config, "PYRAGAS_MAX_EMPIRICAL_RECURRENCE_ERROR_NORM", 0.35)
        )
        min_empirical_correlation = float(
            getattr(config, "PYRAGAS_MIN_EMPIRICAL_RECURRENCE_CORRELATION", 0.65)
        )
        max_empirical_closure = float(
            getattr(config, "PYRAGAS_MAX_EMPIRICAL_TAIL_CLOSURE_ERROR_NORM", 0.35)
        )

        def deficit(value, target):
            return max(0.0, target - value) / max(target, 1e-12)

        def excess(value, target):
            return max(0.0, value - target) / max(target, 1e-12)

        if not np.isfinite(rhythm_cv):
            rhythm_cv = 2.0
        if not np.isfinite(peak_amplitude_cv):
            peak_amplitude_cv = 10.0
        if not np.isfinite(window_amplitude_cv):
            window_amplitude_cv = 10.0
        if not np.isfinite(drift_ratio):
            drift_ratio = 10.0
        if not np.isfinite(empirical_recurrence):
            empirical_recurrence = 10.0
        if not np.isfinite(empirical_closure):
            empirical_closure = 10.0
        if not np.isfinite(empirical_correlation):
            empirical_correlation = -1.0
        if not np.isfinite(tail_activity):
            tail_activity = 0.0

        few_peaks_penalty = deficit(float(peak_count), float(min_peaks))
        few_cycles_penalty = deficit(float(cycle_count), float(min_cycles))
        amplitude_penalty = deficit(amp_ratio, min_amp_ratio)
        std_penalty = deficit(std_ratio, min_std_ratio)
        preferred_amplitude_penalty = deficit(amp_ratio, preferred_amp_min) + excess(
            amp_ratio, preferred_amp_max
        )
        preferred_std_penalty = deficit(std_ratio, preferred_std_min) + excess(
            std_ratio, preferred_std_max
        )
        rhythm_penalty = excess(rhythm_cv, target_rhythm_cv)
        peak_coverage_penalty = deficit(peak_coverage, target_peak_coverage)
        cycle_coverage_penalty = deficit(cycle_coverage, target_cycle_coverage)
        peak_amplitude_penalty = excess(peak_amplitude_cv, max_peak_amplitude_cv)
        window_amplitude_penalty = excess(window_amplitude_cv, max_window_amplitude_cv)
        drift_penalty = excess(drift_ratio, max_drift_ratio)
        tail_activity_penalty = deficit(tail_activity, min_tail_activity)
        empirical_recurrence_penalty = excess(
            empirical_recurrence, max_empirical_recurrence
        )
        empirical_correlation_penalty = deficit(
            empirical_correlation, min_empirical_correlation
        )
        empirical_closure_penalty = excess(empirical_closure, max_empirical_closure)

        quality_issues = []
        if peak_count < min_peaks:
            quality_issues.append("too_few_post_transient_peaks")
        if cycle_count < min_cycles:
            quality_issues.append("too_few_repeated_cycles")
        if rhythm_cv > target_rhythm_cv:
            quality_issues.append("irregular_empirical_rhythm")
        if not preferred_amp_min <= amp_ratio <= preferred_amp_max:
            quality_issues.append("amplitude_outside_preferred_range")
        if peak_coverage < target_peak_coverage:
            quality_issues.append("peaks_not_sustained_across_evaluation")
        if cycle_coverage < target_cycle_coverage:
            quality_issues.append("cycles_not_sustained_across_evaluation")
        if rhythm_type == "spiking" and peak_amplitude_cv > max_peak_amplitude_cv:
            quality_issues.append("inconsistent_spike_amplitudes")
        if drift_ratio > max_drift_ratio:
            quality_issues.append("trajectory_drifted")
        if tail_activity < min_tail_activity:
            quality_issues.append("tail_activity_collapsed")
        if empirical_recurrence > max_empirical_recurrence:
            quality_issues.append("high_empirical_recurrence_error")
        if empirical_correlation < min_empirical_correlation:
            quality_issues.append("weak_empirical_cycle_correlation")
        if empirical_closure > max_empirical_closure:
            quality_issues.append("empirical_orbit_not_closed")

        row["pyragas_quality_pass"] = not quality_issues
        row["pyragas_quality_issues"] = ";".join(quality_issues)
        row["pyragas_paper_style_diagnostics"] = (
            "delay_mismatch, fixed-delay recurrence, feedback decay, and "
            "noninvasiveness are reported but are not hard pass criteria"
        )

        too_many_spikes_penalty = (
            max(0.0, float(peak_count - 2 * uncontrolled_peak_count))
            / max(float(uncontrolled_peak_count), 1.0)
            if uncontrolled_peak_count > 0
            else 0.0
        )

        return float(
            30.0 * few_peaks_penalty
            + 30.0 * few_cycles_penalty
            + 25.0 * amplitude_penalty
            + 10.0 * std_penalty
            + 8.0 * preferred_amplitude_penalty
            + 3.0 * preferred_std_penalty
            + 8.0 * rhythm_cv
            + 15.0 * rhythm_penalty
            + 15.0 * peak_coverage_penalty
            + 15.0 * cycle_coverage_penalty
            + 6.0 * peak_amplitude_penalty
            + 4.0 * window_amplitude_penalty
            + 20.0 * drift_penalty
            + 20.0 * tail_activity_penalty
            + 10.0 * empirical_recurrence
            + 20.0 * empirical_recurrence_penalty
            + 10.0 * empirical_correlation_penalty
            + 15.0 * empirical_closure_penalty
            + 25.0 * len(quality_issues)
            + 2.0 * too_many_spikes_penalty
            + float(getattr(config, "PYRAGAS_SCORE_ENERGY_WEIGHT", 0.05)) * energy
            + float(getattr(config, "PYRAGAS_SCORE_MAX_CONTROL_WEIGHT", 0.01)) * max_control
            + float(getattr(config, "PYRAGAS_SCORE_K_WEIGHT", 0.02)) * K
        )

    rmse = _safe_float(row.get("target_rmse_state"), np.inf)
    energy = _safe_float(row.get("control_energy"), np.inf)
    settling = _safe_float(row.get("settling_time"), np.inf)
    spike_reduction = _safe_float(row.get("spike_reduction_percent"), 0.0)

    return float(
        rmse
        + float(getattr(config, "CONTROL_SCORE_ENERGY_WEIGHT", 0.01)) * energy
        + float(getattr(config, "CONTROL_SCORE_SETTLING_WEIGHT", 0.001)) * settling
        - float(getattr(config, "CONTROL_SCORE_SPIKE_WEIGHT", 0.0)) * (spike_reduction / 100.0)
    )

def _best_row(rows, config):
    if not rows:
        raise ValueError("No K sweep rows available.")

    for row in rows:
        row["selection_score"] = _selection_score(row, config)

    stable_rows = [
        r
        for r in rows
        if bool(r.get("stable", False))
        and np.isfinite(_safe_float(r.get("selection_score"), np.inf))
    ]

    pyragas_rows = [
        r for r in stable_rows if str(r.get("controller", "")).strip().lower() == "pyragas"
    ]
    if pyragas_rows:
        quality_rows = [r for r in pyragas_rows if bool(r.get("pyragas_quality_pass", False))]
        if quality_rows:
            best = min(
                quality_rows,
                key=lambda r: _safe_float(r.get("selection_score"), np.inf),
            )
            best["pyragas_selection_status"] = "quality_pass"
            return best

        best = min(
            pyragas_rows,
            key=lambda r: _safe_float(r.get("selection_score"), np.inf),
        )
        best["pyragas_selection_status"] = "best_available_only_no_quality_pass"
        return best

    return min(stable_rows or rows, key=lambda r: _safe_float(r.get("selection_score"), np.inf))


def _call_predict_controlled(
    esn,
    train_norm,
    n_base,
    target_norm,
    K,
    control_start_idx,
    controller,
    finite_s,
    pyragas_delay,
    pyragas_sign,
):
    sig = inspect.signature(esn.predict_controlled)

    kwargs = {
        "train_sequence": train_norm,
        "horizon_steps": n_base,
        "target": target_norm,
        "K": float(K),
        "control_start_idx": int(control_start_idx),
    }

    if "controller" in sig.parameters:
        kwargs["controller"] = controller
    elif controller != "linear_feedback":
        raise TypeError(
            "Your model.py predict_controlled() currently supports only linear_feedback. "
            "Update model.py before running finite_time or pyragas."
        )

    if "finite_s" in sig.parameters:
        kwargs["finite_s"] = finite_s

    if "pyragas_delay" in sig.parameters:
        kwargs["pyragas_delay"] = pyragas_delay

    if "pyragas_sign" in sig.parameters:
        kwargs["pyragas_sign"] = pyragas_sign

    return esn.predict_controlled(**kwargs)


def _evaluate_k(
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
    settling_tolerance,
    settling_consecutive,
    controller,
    finite_s,
    pyragas_delay,
    pyragas_sign=-1,
    save_dir=None,
):
    result = _call_predict_controlled(
        esn=esn,
        train_norm=train_norm,
        n_base=n_base,
        target_norm=target_norm,
        K=K,
        control_start_idx=control_start_idx,
        controller=controller,
        finite_s=finite_s,
        pyragas_delay=pyragas_delay,
        pyragas_sign=pyragas_sign,
    )

    controlled_norm = _as_2d(result["controlled_output_norm"])
    control_signal_norm = _as_2d(result["control_signal_norm"])

    stable = bool(result.get("stable", True)) and np.all(np.isfinite(controlled_norm))

    controlled = _denormalize(controlled_norm, mean, std)
    control_signal = control_signal_norm * np.asarray(std, dtype=float)

    metrics = _summarize_control_metrics(
        times=test_times_aligned,
        uncontrolled=uncontrolled,
        controlled=controlled,
        target_state=target_raw,
        control_signal=control_signal,
        control_start_idx=control_start_idx,
        spike_threshold=spike_threshold,
        settling_tolerance=settling_tolerance,
        settling_consecutive=settling_consecutive,
    )

    if controller == "pyragas":
        metrics.update(
            _pyragas_dynamics_metrics(
                controlled=controlled,
                uncontrolled=uncontrolled,
                control_signal=control_signal,
                times=test_times_aligned,
                control_start_idx=control_start_idx,
                pyragas_delay=pyragas_delay,
                spike_threshold=spike_threshold,
            )
        )

    row = {
        "controller": controller,
        "K": float(K),
        "finite_s": float(finite_s),
        "pyragas_delay": int(pyragas_delay),
        "pyragas_sign": int(pyragas_sign),
        "stable": stable,
        "hr_mode": hr_mode,
        "optimizer": optimizer_name,
        "control_start_idx": int(control_start_idx),
        "control_start_time": float(control_start_time),
        "control_start_frac": float(control_start_frac),
        "target_mode": control_target_mode,
        **metrics,
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _save_json(row, os.path.join(save_dir, "metrics.json"))
        _save_rollout_csv(
            os.path.join(save_dir, "rollout.csv"),
            test_times_aligned,
            test_aligned,
            uncontrolled,
            controlled,
            control_signal,
            target_raw,
        )

    return row, controlled, control_signal


def _append_global_control_comparison(output_root, row):
    os.makedirs(output_root, exist_ok=True)
    path = os.path.join(output_root, "control_comparison.csv")

    rows = []
    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            rows = list(csv.DictReader(f))

    key = (str(row.get("regime")), str(row.get("controller")), str(row.get("optimizer")))

    out = []
    replaced = False

    for old in rows:
        old_key = (str(old.get("regime")), str(old.get("controller")), str(old.get("optimizer")))
        if old_key == key:
            out.append(row)
            replaced = True
        else:
            out.append(old)

    if not replaced:
        out.append(row)

    _save_csv(out, path)


def run_control_experiment(
    *,
    esn,
    loader,
    config,
    train,
    test,
    train_norm,
    test_norm,
    mean,
    std,
    times,
    base_output_dir,
    hr_mode,
    best_params=None,
    optimizer_name="unknown",
    control_k=None,
    control_start_frac=0.30,
    control_target_mode="rest_state",
    auto_control_k=False,
    k_min=None,
    k_max=None,
    k_num=None,
    k_refine_num=None,
    controller="linear_feedback",
    finite_s=0.8,
    pyragas_delay=20,
    pyragas_sign=-1,
):
    if controller not in SUPPORTED_CONTROLLERS:
        raise ValueError(f"Unknown controller '{controller}'. Choose from {SUPPORTED_CONTROLLERS}")

    train = _as_2d(train)
    test = _as_2d(test)
    train_norm = _as_2d(train_norm)
    mean = _as_2d(mean)
    std = _as_2d(std)

    n_base = len(test)
    test_aligned = test[:n_base]

    test_times_aligned = _as_1d(times)[len(train) : len(train) + n_base]
    if len(test_times_aligned) != n_base:
        test_times_aligned = np.arange(n_base, dtype=float)

    control_start_idx = int(max(0, min(n_base - 1, round(float(control_start_frac) * n_base))))
    control_start_time = float(test_times_aligned[control_start_idx])

    target_raw, target_norm = _choose_target_state(train, mean, std, control_target_mode, config)

    eval_norm = np.vstack([train_norm, test_norm])
    warmup_steps = len(train_norm) - 1

    pred_result = esn.predict(eval_norm, n_warmup=warmup_steps)

    if isinstance(pred_result, tuple):
        pred_norm = pred_result[0]
    else:
        pred_norm = pred_result

    pred_norm = _as_2d(pred_norm)[:n_base]
    if isinstance(pred_norm, dict):
        pred_norm = pred_norm.get("prediction_norm", pred_norm.get("prediction", pred_norm))

    uncontrolled = _denormalize(pred_norm, mean, std)

    output_dir = os.path.join(base_output_dir, "control", controller)
    os.makedirs(output_dir, exist_ok=True)

    spike_threshold = float(getattr(config, "SPIKE_THRESHOLD", 2.0))
    settling_tolerance = float(getattr(config, "CONTROL_SETTLING_TOLERANCE", 0.25))
    settling_consecutive = int(getattr(config, "CONTROL_SETTLING_CONSECUTIVE", 25))

    print(f"[Control] Controller: {controller}")
    print(f"[Control] Output dir : {output_dir}")
    print(f"[Control] Target raw : {target_raw}")
    if controller == "pyragas":
        print(f"[Control] Pyragas delay: {int(pyragas_delay)}")
        print(f"[Control] Pyragas sign : {int(pyragas_sign)}")

    rows = []

    coarse_values = _k_values(control_k, config, k_min, k_max, k_num)

    for K in coarse_values:
        print(f"[Control] Coarse K={K:.6f}")

        row, _, _ = _evaluate_k(
            esn=esn,
            train_norm=train_norm,
            n_base=n_base,
            K=K,
            target_norm=target_norm,
            target_raw=target_raw,
            mean=mean,
            std=std,
            uncontrolled=uncontrolled,
            test_aligned=test_aligned,
            test_times_aligned=test_times_aligned,
            control_start_idx=control_start_idx,
            control_start_time=control_start_time,
            control_start_frac=control_start_frac,
            control_target_mode=control_target_mode,
            hr_mode=hr_mode,
            optimizer_name=optimizer_name,
            spike_threshold=spike_threshold,
            settling_tolerance=settling_tolerance,
            settling_consecutive=settling_consecutive,
            controller=controller,
            finite_s=finite_s,
            pyragas_delay=pyragas_delay,
            pyragas_sign=pyragas_sign,
        )
        rows.append(row)

    if auto_control_k and control_k is None:
        coarse_best = _best_row(rows, config)
        refine_values = _refined_k_values(coarse_best["K"], config, k_min, k_max, k_refine_num)
        existing = {round(float(r["K"]), 10) for r in rows}

        for K in refine_values:
            if round(float(K), 10) in existing:
                continue

            print(f"[Control] Refined K={K:.6f}")

            row, _, _ = _evaluate_k(
                esn=esn,
                train_norm=train_norm,
                n_base=n_base,
                K=K,
                target_norm=target_norm,
                target_raw=target_raw,
                mean=mean,
                std=std,
                uncontrolled=uncontrolled,
                test_aligned=test_aligned,
                test_times_aligned=test_times_aligned,
                control_start_idx=control_start_idx,
                control_start_time=control_start_time,
                control_start_frac=control_start_frac,
                control_target_mode=control_target_mode,
                hr_mode=hr_mode,
                optimizer_name=optimizer_name,
                spike_threshold=spike_threshold,
                settling_tolerance=settling_tolerance,
                settling_consecutive=settling_consecutive,
                controller=controller,
                finite_s=finite_s,
                pyragas_delay=pyragas_delay,
                pyragas_sign=pyragas_sign,
            )
            rows.append(row)

    best = _best_row(rows, config)
    best_K = float(best["K"])

    print(f"[Control] Best K={best_K:.6f}")

    best_row, best_controlled, best_control_signal = _evaluate_k(
        esn=esn,
        train_norm=train_norm,
        n_base=n_base,
        K=best_K,
        target_norm=target_norm,
        target_raw=target_raw,
        mean=mean,
        std=std,
        uncontrolled=uncontrolled,
        test_aligned=test_aligned,
        test_times_aligned=test_times_aligned,
        control_start_idx=control_start_idx,
        control_start_time=control_start_time,
        control_start_frac=control_start_frac,
        control_target_mode=control_target_mode,
        hr_mode=hr_mode,
        optimizer_name=optimizer_name,
        spike_threshold=spike_threshold,
        settling_tolerance=settling_tolerance,
        settling_consecutive=settling_consecutive,
        controller=controller,
        finite_s=finite_s,
        pyragas_delay=pyragas_delay,
        pyragas_sign=pyragas_sign,
        save_dir=os.path.join(output_dir, "best_rollout"),
    )

    # The final rollout is evaluated again for saving, so attach the same
    # Pyragas-only quality assessment that was used during K selection.
    if controller == "pyragas":
        best_row["selection_score"] = _selection_score(best_row, config)
        best_row["pyragas_selection_status"] = (
            "quality_pass"
            if bool(best_row.get("pyragas_quality_pass", False))
            else "best_available_only_no_quality_pass"
        )
        _save_json(best_row, os.path.join(output_dir, "best_rollout", "metrics.json"))

    _save_csv(rows, os.path.join(output_dir, "k_sweep.csv"))
    _save_json(
        {
            "best": best_row,
            "all_rows": rows,
            "target_raw": target_raw,
            "target_norm": target_norm,
        },
        os.path.join(output_dir, "control_summary.json"),
    )

    plot_controlled_vs_uncontrolled_x(
        test_times_aligned,
        test_aligned,
        uncontrolled,
        best_controlled,
        target_raw,
        control_start_idx,
        {**best_row, "K": best_K},
        output_dir,
        controller_name=controller,
    )

    plot_controlled_all_states(
        test_times_aligned,
        test_aligned,
        uncontrolled,
        best_controlled,
        target_raw,
        control_start_idx,
        output_dir,
        controller_name=controller,
        metrics=best_row,
    )

    plot_control_signal(
        test_times_aligned,
        best_control_signal,
        control_start_idx,
        output_dir,
        controller_name=controller,
    )

    uncontrolled_error = _compute_error_norms(uncontrolled, target_raw)
    controlled_error = _compute_error_norms(best_controlled, target_raw)

    plot_control_error(
        test_times_aligned,
        uncontrolled_error,
        controlled_error,
        control_start_idx,
        settling_tolerance,
        output_dir,
        controller_name=controller,
    )

    plot_k_sweep_summary(rows, output_dir, controller_name=controller)

    global_row = {
        "regime": hr_mode,
        "controller": controller,
        "optimizer": optimizer_name,
        "best_K": best_K,
        "finite_s": float(finite_s),
        "pyragas_delay": int(pyragas_delay),
        "pyragas_sign": int(pyragas_sign),
        "best_target_rmse_state": best_row.get("target_rmse_state"),
        "best_target_rmse_x": best_row.get("target_rmse_x"),
        "best_spike_reduction_percent": best_row.get("spike_reduction_percent"),
        "best_control_energy": best_row.get("control_energy"),
        "best_settling_time": best_row.get("settling_time"),
        "best_stable": best_row.get("stable"),
        "output_dir": output_dir,
    }

    if controller == "pyragas":
        global_row.update(
            {
                "best_pyragas_periodicity_rmse_state": best_row.get("pyragas_periodicity_rmse_state"),
                "best_pyragas_periodicity_rmse_state_norm": best_row.get("pyragas_periodicity_rmse_state_norm"),
                "best_pyragas_x_amplitude_ratio": best_row.get("pyragas_x_amplitude_ratio"),
                "best_pyragas_x_std_ratio": best_row.get("pyragas_x_std_ratio"),
                "best_pyragas_spike_interval_cv": best_row.get("pyragas_spike_interval_cv"),
                "best_pyragas_delay_period_mismatch": best_row.get("pyragas_delay_period_mismatch"),
                "best_pyragas_noninvasiveness_ratio": best_row.get("pyragas_noninvasiveness_ratio"),
                "best_pyragas_control_decay_ratio": best_row.get("pyragas_control_decay_ratio"),
                "best_pyragas_rhythm_type": best_row.get("pyragas_rhythm_type"),
                "best_pyragas_detected_cycle_count": best_row.get("pyragas_detected_cycle_count"),
                "best_pyragas_rhythm_interval_cv": best_row.get("pyragas_rhythm_interval_cv"),
                "best_pyragas_empirical_period_steps": best_row.get("pyragas_empirical_period_steps"),
                "best_pyragas_empirical_period_time": best_row.get("pyragas_empirical_period_time"),
                "best_pyragas_empirical_recurrence_error_norm": best_row.get("pyragas_empirical_recurrence_error_norm"),
                "best_pyragas_empirical_recurrence_correlation": best_row.get("pyragas_empirical_recurrence_correlation"),
                "best_pyragas_empirical_tail_closure_error_norm": best_row.get("pyragas_empirical_tail_closure_error_norm"),
                "best_pyragas_quality_pass": best_row.get("pyragas_quality_pass"),
                "best_pyragas_quality_issues": best_row.get("pyragas_quality_issues"),
                "best_pyragas_selection_status": best_row.get("pyragas_selection_status"),
                "best_spike_count_after": best_row.get("spike_count_after"),
            }
        )

    _append_global_control_comparison(os.path.dirname(base_output_dir), global_row)

    return global_row


def run_linear_feedback_control_experiment(**kwargs):
    """
    Backward-compatible wrapper for your current main.py.
    """
    kwargs["controller"] = "linear_feedback"
    return run_control_experiment(**kwargs)
