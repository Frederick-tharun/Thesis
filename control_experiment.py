from __future__ import annotations

import csv
import inspect
import json
import os
import time
import warnings
from typing import Any

import numpy as np

from plotting import (
    plot_controlled_vs_uncontrolled_x,
    plot_controlled_all_states,
    plot_control_signal,
    plot_control_error,
    plot_k_sweep_summary,
    plot_raw_readout_vs_corrected_feedback_input_x,
)

SUPPORTED_CONTROLLERS = ("linear_feedback", "finite_time", "pyragas")


class NoStableControllerCandidateError(RuntimeError):
    """Raised when a controller search has no stable validation candidate."""


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


def _save_rollout_csv(
    path,
    times,
    truth,
    uncontrolled,
    raw_readout,
    corrected_feedback_input,
    control_signal,
    target_state,
):
    """Save canonical control signals plus documented legacy aliases."""
    times = _as_1d(times)
    truth = _as_2d(truth)
    uncontrolled = _as_2d(uncontrolled)
    raw_readout = _as_2d(raw_readout)
    corrected_feedback_input = _as_2d(corrected_feedback_input)
    control_signal = _as_2d(control_signal)
    target_state = _as_1d(target_state)

    n = min(
        len(times),
        len(truth),
        len(uncontrolled),
        len(raw_readout),
        len(corrected_feedback_input),
        len(control_signal),
    )
    rows = []
    for i in range(n):
        row = {
            "time_index": int(i),
            "time": float(times[i]),
            "target_x": float(target_state[0]),
            "target_y": float(target_state[1]),
            "target_z": float(target_state[2]),
        }
        for j, name in enumerate(["x", "y", "z"]):
            row[f"true_{name}"] = float(truth[i, j])
            row[f"uncontrolled_{name}"] = float(uncontrolled[i, j])
            row[f"raw_readout_{name}"] = float(raw_readout[i, j])
            row[f"corrected_feedback_input_{name}"] = float(
                corrected_feedback_input[i, j]
            )
            row[f"control_signal_{name}"] = float(control_signal[i, j])
            # Legacy aliases: "controlled" means the value fed back.
            row[f"controlled_{name}"] = row[f"corrected_feedback_input_{name}"]
            row[f"u_{name}"] = row[f"control_signal_{name}"]
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

    All controller metrics share this detector so one contiguous
    above-threshold episode counts as one spike. Pyragas also uses the returned
    indices to measure spike-interval regularity.
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
    window_start_idx=None,
    window_end_idx=None,
    discard_initial_transient=True,
):
    """Measure Pyragas behaviour on an explicit evaluation window.

    Validation may discard an initial settling interval so that transient peaks
    cannot mimic a sustained orbit. The final held-out controller-test window
    is evaluated from its first sample, with no additional transient removed.
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
            "pyragas_evaluation_end_idx": 0,
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
    window_start = start if window_start_idx is None else int(max(0, min(window_start_idx, n)))
    window_end = n if window_end_idx is None else int(max(window_start, min(window_end_idx, n)))
    empty["pyragas_evaluation_end_idx"] = int(window_end)
    if window_end <= window_start:
        return empty

    delay = int(max(1, pyragas_delay))

    post_len = window_end - window_start
    if discard_initial_transient:
        desired_transient = max(2 * delay, int(round(0.25 * post_len)))
        minimum_eval_len = max(delay + 3, 16)
        max_transient = max(0, post_len - minimum_eval_len)
        transient_samples = min(desired_transient, max_transient)
    else:
        transient_samples = 0
    eval_start = window_start + transient_samples

    controlled_post = controlled[eval_start:window_end]
    uncontrolled_post = uncontrolled[eval_start:window_end]
    control_post = control_signal[eval_start:window_end]

    if len(controlled_post) < 3:
        result = dict(empty)
        result.update(
            {
                "pyragas_evaluation_start_idx": int(eval_start),
                "pyragas_evaluation_end_idx": int(window_end),
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
    # stabilized. Measure this only on the explicit evaluation interval.
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
        "pyragas_evaluation_end_idx": int(window_end),
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
    return int(len(_peak_indices(x, threshold)))


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


def _choose_target_state(
    train_raw,
    mean,
    std,
    target_mode,
    config,
    hr_mode=None,
    return_metadata=False,
):
    """Return a target in raw/normalized coordinates and optional provenance."""
    train_raw = _as_2d(train_raw)
    requested_mode = str(target_mode)
    if requested_mode == "rest_state":
        warnings.warn(
            "'rest_state' is deprecated; use "
            "'rest_state_from_quiet_training_data'.",
            FutureWarning,
            stacklevel=2,
        )
        canonical_mode = "rest_state_from_quiet_training_data"
    else:
        canonical_mode = requested_mode

    metadata = {
        "requested_target_mode": requested_mode,
        "target_mode": canonical_mode,
    }

    spike_threshold = float(getattr(config, "SPIKE_THRESHOLD", 2.0))
    x = train_raw[:, 0]

    if canonical_mode == "rest_state_from_quiet_training_data":
        mask = x < spike_threshold

        if np.count_nonzero(mask) < max(25, len(x) // 100):
            cutoff = np.percentile(x, 70.0)
            mask = x <= cutoff

        if np.count_nonzero(mask):
            target_raw = np.median(train_raw[mask], axis=0)
        else:
            target_raw = np.median(train_raw, axis=0)
        metadata.update(
            {
                "target_source": "median_of_quiet_training_samples",
                "reference_type": "empirical_quiet_state_reference",
                "target_interpretation": (
                    "empirical quiet-state reference; data-derived, not an "
                    "exact Hindmarsh-Rose equilibrium"
                ),
                "regulation_objective": (
                    "regulation toward an empirical quiet-state reference"
                ),
                "is_exact_equilibrium": False,
                "quiet_training_sample_count": int(np.count_nonzero(mask)),
                "quiet_x_threshold": spike_threshold,
            }
        )
        active_mode = str(hr_mode or getattr(config, "HR_MODE", ""))
        parameter_sets = getattr(config, "HR_PARAMETER_SETS", {})
        if train_raw.shape[1] == 3 and active_mode in parameter_sets:
            params = dict(parameter_sets[active_mode])
            tx, ty, tz = np.asarray(target_raw, dtype=float)
            residual_vector = np.asarray(
                [
                    ty
                    - float(params["a"]) * tx**3
                    + float(params["b"]) * tx**2
                    - tz
                    + float(params["I"]),
                    float(params["c"]) - float(params["d"]) * tx**2 - ty,
                    float(params["r"])
                    * (
                        float(params["s"]) * (tx - float(params["xr"]))
                        - tz
                    ),
                ],
                dtype=float,
            )
            metadata.update(
                {
                    "hr_mode": active_mode,
                    "hr_rhs_residual_vector": residual_vector,
                    "hr_rhs_residual_norm": float(
                        np.linalg.norm(residual_vector)
                    ),
                    "hr_rhs_residual_diagnostic": (
                        "nonzero values are expected because this is a "
                        "data-derived empirical quiet-state reference"
                    ),
                }
            )

    elif canonical_mode == "zero":
        target_raw = np.zeros(train_raw.shape[1], dtype=float)
        metadata["target_source"] = "exact_zero_vector"

    elif canonical_mode == "mean":
        target_raw = np.mean(train_raw, axis=0)
        metadata["target_source"] = "mean_of_training_data"

    else:
        raise ValueError(f"Unknown target mode: {requested_mode}")

    target_norm = ((target_raw.reshape(1, -1) - mean) / std).reshape(-1)
    target_raw = target_raw.reshape(-1)
    target_norm = target_norm.reshape(-1)
    if return_metadata:
        return target_raw, target_norm, metadata
    return target_raw, target_norm


def _controller_split_indices(
    n_samples,
    control_start_idx,
    validation_fraction,
    test_fraction,
):
    """Partition the post-control horizon into half-open validation/test windows."""
    n_samples = int(n_samples)
    start = int(max(0, min(control_start_idx, n_samples)))
    post_control_count = n_samples - start
    validation_fraction = float(validation_fraction)
    test_fraction = float(test_fraction)
    if (
        not np.isfinite(validation_fraction)
        or not np.isfinite(test_fraction)
        or validation_fraction <= 0.0
        or test_fraction <= 0.0
    ):
        raise ValueError("Controller validation/test fractions must both be positive.")
    if post_control_count < 2:
        raise ValueError(
            "At least two post-control samples are required for separate "
            "controller validation and test windows."
        )

    fraction_sum = validation_fraction + test_fraction
    validation_count = int(
        round(post_control_count * validation_fraction / fraction_sum)
    )
    validation_count = int(max(1, min(validation_count, post_control_count - 1)))
    validation_end = start + validation_count
    return {
        "control_start_idx": start,
        "controller_validation_start_idx": start,
        "controller_validation_end_idx": validation_end,
        "controller_test_start_idx": validation_end,
        "controller_test_end_idx": n_samples,
        "controller_validation_fraction_requested": validation_fraction,
        "controller_test_fraction_requested": test_fraction,
        "controller_validation_fraction_normalized": (
            validation_fraction / fraction_sum
        ),
        "controller_test_fraction_normalized": test_fraction / fraction_sum,
        "controller_validation_sample_count": validation_count,
        "controller_test_sample_count": n_samples - validation_end,
        "index_semantics": "zero_based_half_open_[start,end)",
    }


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
    raw_readout=None,
    eval_start_idx=None,
    eval_end_idx=None,
    x_normalization_scale=None,
):
    """Summarize one explicitly bounded controller evaluation segment."""
    times = _as_1d(times)
    uncontrolled = _as_2d(uncontrolled)
    corrected = _as_2d(controlled)
    raw = corrected if raw_readout is None else _as_2d(raw_readout)
    target_state = _as_1d(target_state)
    control_signal = _as_2d(control_signal)

    n = min(
        len(times),
        len(uncontrolled),
        len(raw),
        len(corrected),
        len(control_signal),
    )
    if n == 0:
        raise ValueError("Cannot summarize an empty controlled rollout.")
    times = times[:n]
    uncontrolled = uncontrolled[:n]
    raw = raw[:n]
    corrected = corrected[:n]
    control_signal = control_signal[:n]

    start = control_start_idx if eval_start_idx is None else eval_start_idx
    end = n if eval_end_idx is None else eval_end_idx
    start = int(max(0, min(start, n)))
    end = int(max(start, min(end, n)))
    if end <= start:
        raise ValueError(
            f"Controller metric window [{start}, {end}) contains no samples."
        )

    uncontrolled_window = uncontrolled[start:end]
    raw_window = raw[start:end]
    corrected_window = corrected[start:end]
    control_window = control_signal[start:end]
    target_window = np.tile(target_state.reshape(1, -1), (end - start, 1))

    if x_normalization_scale is None:
        x_scale = float(np.std(uncontrolled_window[:, 0]))
    else:
        x_scale = abs(float(np.asarray(x_normalization_scale).reshape(-1)[0]))
    x_scale = x_scale if np.isfinite(x_scale) and x_scale > 1e-12 else float("nan")

    raw_rmse_state = _rmse(raw_window, target_window)
    raw_rmse_x = _rmse(raw_window[:, 0], target_window[:, 0])
    corrected_rmse_state = _rmse(corrected_window, target_window)
    corrected_rmse_x = _rmse(corrected_window[:, 0], target_window[:, 0])
    raw_nrmse_x = raw_rmse_x / x_scale if np.isfinite(x_scale) else float("nan")
    corrected_nrmse_x = (
        corrected_rmse_x / x_scale if np.isfinite(x_scale) else float("nan")
    )

    corrected_error_norm = _compute_error_norms(corrected, target_state)
    control_norm_sq = np.sum(control_window**2, axis=1)
    control_norm = np.sqrt(control_norm_sq)

    spike_after = _count_spikes(corrected_window[:, 0], spike_threshold)
    unctrl_spike_after = _count_spikes(uncontrolled_window[:, 0], spike_threshold)
    spike_reduction = (
        100.0 * (unctrl_spike_after - spike_after) / unctrl_spike_after
        if unctrl_spike_after > 0
        else 0.0
    )

    positive_dt = np.diff(times[start:end])
    positive_dt = positive_dt[
        np.isfinite(positive_dt) & (positive_dt > 0.0)
    ]
    sample_dt = float(np.median(positive_dt)) if len(positive_dt) else float("nan")
    effort = float(np.mean(control_norm_sq))
    energy_dt_sum = (
        float(sample_dt * np.sum(control_norm_sq))
        if np.isfinite(sample_dt)
        else float("nan")
    )
    evaluation_time_to_tolerance = _settling_time(
        times[:end],
        corrected_error_norm[:end],
        start,
        settling_tolerance,
        settling_consecutive,
    )

    return {
        "evaluation_start_idx": start,
        "evaluation_end_idx": end,
        "evaluation_index_semantics": "zero_based_half_open_[start,end)",
        "evaluation_sample_count": end - start,
        "raw_readout_target_rmse_state": raw_rmse_state,
        "raw_readout_target_rmse_x": raw_rmse_x,
        "raw_readout_target_nrmse_x": raw_nrmse_x,
        "corrected_feedback_input_target_rmse_state": corrected_rmse_state,
        "corrected_feedback_input_target_rmse_x": corrected_rmse_x,
        "corrected_feedback_input_target_nrmse_x": corrected_nrmse_x,
        "uncontrolled_target_rmse_state": _rmse(
            uncontrolled_window, target_window
        ),
        "uncontrolled_target_rmse_x": _rmse(
            uncontrolled_window[:, 0], target_window[:, 0]
        ),
        "spike_count_before": _count_spikes(
            corrected[: int(max(0, min(control_start_idx, n))), 0],
            spike_threshold,
        ),
        "spike_count_after": spike_after,
        "uncontrolled_spike_count_after": unctrl_spike_after,
        "spike_reduction_percent": float(spike_reduction),
        "control_effort_mean_sq": effort,
        "control_energy_dt_sum": energy_dt_sum,
        "control_sample_dt": sample_dt,
        "mean_control_norm": float(np.mean(control_norm)),
        "max_control_norm": float(np.max(control_norm)),
        "evaluation_time_to_tolerance": evaluation_time_to_tolerance,
        "settling_time": evaluation_time_to_tolerance,
        "mean_error_norm_post": float(np.mean(corrected_error_norm[start:end])),
        "max_error_norm_post": float(np.max(corrected_error_norm[start:end])),
        # Deprecated aliases retained for old analysis scripts.
        "target_rmse_state": corrected_rmse_state,
        "target_rmse_x": corrected_rmse_x,
        "target_nrmse_x": corrected_nrmse_x,
        "control_energy": effort,
        "control_energy_alias_of": "control_effort_mean_sq",
        "target_rmse_alias_of": "corrected_feedback_input_target_rmse",
        "settling_time_alias_of": "evaluation_time_to_tolerance",
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
        energy = _safe_float(
            row.get("control_effort_mean_sq", row.get("control_energy")), 0.0
        )
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
            quality_issues.append("too_few_evaluation_peaks")
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

        def score_weight(name, default):
            return float(getattr(config, name, default))

        return float(
            score_weight("PYRAGAS_SCORE_FEW_SPIKES_WEIGHT", 30.0)
            * few_peaks_penalty
            + score_weight("PYRAGAS_SCORE_FEW_CYCLES_WEIGHT", 30.0)
            * few_cycles_penalty
            + score_weight("PYRAGAS_SCORE_FLAT_AMPLITUDE_WEIGHT", 25.0)
            * amplitude_penalty
            + score_weight("PYRAGAS_SCORE_FLAT_STD_WEIGHT", 10.0) * std_penalty
            + score_weight("PYRAGAS_SCORE_AMPLITUDE_RANGE_WEIGHT", 8.0)
            * preferred_amplitude_penalty
            + score_weight("PYRAGAS_SCORE_STD_RANGE_WEIGHT", 3.0)
            * preferred_std_penalty
            + score_weight("PYRAGAS_SCORE_INTERVAL_CV_WEIGHT", 8.0) * rhythm_cv
            + score_weight("PYRAGAS_SCORE_INTERVAL_CV_EXCESS_WEIGHT", 15.0)
            * rhythm_penalty
            + score_weight("PYRAGAS_SCORE_PEAK_COVERAGE_WEIGHT", 15.0)
            * peak_coverage_penalty
            + score_weight("PYRAGAS_SCORE_CYCLE_COVERAGE_WEIGHT", 15.0)
            * cycle_coverage_penalty
            + score_weight("PYRAGAS_SCORE_PEAK_AMPLITUDE_WEIGHT", 6.0)
            * peak_amplitude_penalty
            + score_weight("PYRAGAS_SCORE_WINDOW_AMPLITUDE_WEIGHT", 4.0)
            * window_amplitude_penalty
            + score_weight("PYRAGAS_SCORE_DRIFT_WEIGHT", 20.0) * drift_penalty
            + score_weight("PYRAGAS_SCORE_TAIL_ACTIVITY_WEIGHT", 20.0)
            * tail_activity_penalty
            + score_weight("PYRAGAS_SCORE_PERIODICITY_WEIGHT", 10.0)
            * empirical_recurrence
            + score_weight("PYRAGAS_SCORE_PERIODICITY_EXCESS_WEIGHT", 20.0)
            * empirical_recurrence_penalty
            + score_weight("PYRAGAS_SCORE_EMPIRICAL_CORRELATION_WEIGHT", 10.0)
            * empirical_correlation_penalty
            + score_weight("PYRAGAS_SCORE_TAIL_CLOSURE_WEIGHT", 15.0)
            * empirical_closure_penalty
            + score_weight("PYRAGAS_SCORE_QUALITY_ISSUE_WEIGHT", 25.0)
            * len(quality_issues)
            + score_weight("PYRAGAS_SCORE_TOO_MANY_SPIKES_WEIGHT", 2.0)
            * too_many_spikes_penalty
            + score_weight("PYRAGAS_SCORE_ENERGY_WEIGHT", 0.05) * energy
            + score_weight("PYRAGAS_SCORE_MAX_CONTROL_WEIGHT", 0.01) * max_control
            + score_weight("PYRAGAS_SCORE_K_WEIGHT", 0.02) * K
        )

    rmse = _safe_float(
        row.get(
            "corrected_feedback_input_target_rmse_state",
            row.get("target_rmse_state"),
        ),
        np.inf,
    )
    energy = _safe_float(
        row.get("control_effort_mean_sq", row.get("control_energy")), np.inf
    )
    settling = _safe_float(row.get("settling_time"), np.nan)
    if not np.isfinite(settling):
        sample_dt = _safe_float(row.get("control_sample_dt"), 1.0)
        sample_count = _safe_float(row.get("evaluation_sample_count"), 1.0)
        settling = max(sample_dt, 1e-12) * max(sample_count, 1.0)
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
        and not bool(r.get("divergence_detected", False))
        and np.isfinite(_safe_float(r.get("selection_score"), np.inf))
    ]
    if not stable_rows:
        reasons = sorted(
            {
                str(row.get("divergence_reason") or "nonfinite_selection_score")
                for row in rows
            }
        )
        raise NoStableControllerCandidateError(
            "No stable controller candidate passed validation: "
            + ", ".join(reasons)
        )

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

    return min(
        stable_rows,
        key=lambda r: _safe_float(r.get("selection_score"), np.inf),
    )


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
    pyragas_history_signal,
    control_input_clip,
    divergence_abs_limit,
):
    """Call current and legacy ESN control APIs without hiding capabilities."""
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

    optional_values = {
        "finite_s": finite_s,
        "pyragas_delay": pyragas_delay,
        "pyragas_sign": pyragas_sign,
        "pyragas_history_signal": pyragas_history_signal,
        "control_input_clip": control_input_clip,
        "divergence_abs_limit": divergence_abs_limit,
    }
    for name, value in optional_values.items():
        if name in sig.parameters:
            kwargs[name] = value

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
    pyragas_history_signal="raw_readout",
    control_input_clip=None,
    divergence_abs_limit=None,
    metric_start_idx=None,
    metric_end_idx=None,
    metric_segment="controller_validation",
    discard_pyragas_transient=True,
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
        pyragas_history_signal=pyragas_history_signal,
        control_input_clip=control_input_clip,
        divergence_abs_limit=divergence_abs_limit,
    )
    if not isinstance(result, dict):
        raise TypeError("predict_controlled() must return a signal dictionary.")

    corrected_norm = result.get(
        "corrected_feedback_input_norm",
        result.get("controlled_output_norm", result.get("feedback_input_norm")),
    )
    if corrected_norm is None:
        raise KeyError("Controlled rollout did not return corrected feedback input.")
    corrected_norm = _as_2d(corrected_norm)
    control_signal_norm = _as_2d(result["control_signal_norm"])
    raw_norm = result.get("raw_readout_norm", result.get("raw_prediction_norm"))
    if raw_norm is None:
        raw_norm = corrected_norm + control_signal_norm
    raw_norm = _as_2d(raw_norm)

    raw_readout = _denormalize(raw_norm, mean, std)
    corrected_feedback_input = _denormalize(corrected_norm, mean, std)
    control_signal = control_signal_norm * np.asarray(std, dtype=float)

    stable = (
        bool(result.get("stable", True))
        and not bool(result.get("divergence_detected", False))
        and np.all(np.isfinite(raw_norm))
        and np.all(np.isfinite(corrected_norm))
        and np.all(np.isfinite(control_signal_norm))
    )
    reported_divergence = bool(result.get("divergence_detected", False))
    divergence_detected = reported_divergence or not stable
    divergence_reason = result.get("divergence_reason")
    if divergence_detected and not divergence_reason:
        divergence_reason = (
            "rollout_reported_unstable"
            if not bool(result.get("stable", True))
            else "nonfinite_rollout_signal"
        )

    start = control_start_idx if metric_start_idx is None else int(metric_start_idx)
    end = n_base if metric_end_idx is None else int(metric_end_idx)
    metrics = _summarize_control_metrics(
        times=test_times_aligned,
        uncontrolled=uncontrolled,
        raw_readout=raw_readout,
        controlled=corrected_feedback_input,
        target_state=target_raw,
        control_signal=control_signal,
        control_start_idx=control_start_idx,
        eval_start_idx=start,
        eval_end_idx=end,
        x_normalization_scale=np.asarray(std).reshape(-1)[0],
        spike_threshold=spike_threshold,
        settling_tolerance=settling_tolerance,
        settling_consecutive=settling_consecutive,
    )

    if controller == "pyragas":
        metrics.update(
            _pyragas_dynamics_metrics(
                controlled=corrected_feedback_input,
                uncontrolled=uncontrolled,
                control_signal=control_signal,
                times=test_times_aligned,
                control_start_idx=control_start_idx,
                pyragas_delay=pyragas_delay,
                spike_threshold=spike_threshold,
                window_start_idx=start,
                window_end_idx=end,
                discard_initial_transient=discard_pyragas_transient,
            )
        )

    row = {
        "controller": controller,
        "K": float(K),
        "finite_s": float(finite_s),
        "pyragas_delay": int(pyragas_delay),
        "pyragas_sign": int(pyragas_sign),
        "pyragas_history_signal": str(
            result.get("pyragas_history_signal", pyragas_history_signal)
        ),
        "stable": stable,
        "divergence_detected": divergence_detected,
        "divergence_reason": divergence_reason,
        "divergence_index": result.get("divergence_index"),
        "steps_completed": int(result.get("steps_completed", len(raw_norm))),
        "controller_law_coordinate_system": "normalized_esn_coordinates",
        "control_input_clip_normalized": result.get(
            "control_input_clip", control_input_clip
        ),
        "divergence_abs_limit_normalized": result.get(
            "divergence_abs_limit", divergence_abs_limit
        ),
        "metric_segment": metric_segment,
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
            raw_readout,
            corrected_feedback_input,
            control_signal,
            target_raw,
        )

    return row, raw_readout, corrected_feedback_input, control_signal


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
    control_target_mode="rest_state_from_quiet_training_data",
    auto_control_k=False,
    k_min=None,
    k_max=None,
    k_num=None,
    k_refine_num=None,
    controller="linear_feedback",
    finite_s=0.8,
    pyragas_delay=20,
    pyragas_sign=-1,
    pyragas_history_signal="raw_readout",
    validation_only=False,
    generate_plots=True,
    uncontrolled_prediction_norm=None,
    model_provenance=None,
    controller_output_dir=None,
    artifact_relative_path=None,
    append_global_comparison=True,
    locked_validation_selection=None,
):
    """Select on controller validation and optionally evaluate held-out test once."""
    if controller not in SUPPORTED_CONTROLLERS:
        raise ValueError(
            f"Unknown controller '{controller}'. Choose from {SUPPORTED_CONTROLLERS}"
        )

    train = _as_2d(train)
    test = _as_2d(test)
    train_norm = _as_2d(train_norm)
    test_norm = _as_2d(test_norm)
    mean = _as_2d(mean)
    std = _as_2d(std)

    n_base = len(test)
    if n_base < 3:
        raise ValueError("The held-out controller horizon is too short.")
    test_aligned = test[:n_base]
    test_times_aligned = _as_1d(times)[len(train) : len(train) + n_base]
    if len(test_times_aligned) != n_base:
        test_times_aligned = np.arange(n_base, dtype=float)

    control_start_idx = int(
        max(0, min(n_base - 1, round(float(control_start_frac) * n_base)))
    )
    control_start_time = float(test_times_aligned[control_start_idx])

    split = _controller_split_indices(
        n_samples=n_base,
        control_start_idx=control_start_idx,
        validation_fraction=getattr(config, "CONTROL_VALIDATION_FRAC", 0.5),
        test_fraction=getattr(config, "CONTROL_TEST_FRAC", 0.5),
    )
    validation_start = split["controller_validation_start_idx"]
    validation_end = split["controller_validation_end_idx"]
    controller_test_start = split["controller_test_start_idx"]
    controller_test_end = split["controller_test_end_idx"]

    target_raw, target_norm, target_metadata = _choose_target_state(
        train,
        mean,
        std,
        control_target_mode,
        config,
        hr_mode=hr_mode,
        return_metadata=True,
    )
    canonical_target_mode = target_metadata["target_mode"]

    if uncontrolled_prediction_norm is None:
        eval_norm = np.vstack([train_norm, test_norm])
        warmup_steps = len(train_norm) - 1
        pred_result = esn.predict(eval_norm, n_warmup=warmup_steps)
        if isinstance(pred_result, tuple):
            pred_result = pred_result[0]
        if isinstance(pred_result, dict):
            pred_result = pred_result.get(
                "prediction_norm", pred_result.get("prediction")
            )
            if pred_result is None:
                raise KeyError(
                    "Uncontrolled prediction dictionary has no prediction array."
                )
        pred_norm = _as_2d(pred_result)[:n_base]
    else:
        pred_norm = _as_2d(uncontrolled_prediction_norm)[:n_base]
        if len(pred_norm) != n_base:
            raise ValueError(
                "Cached uncontrolled prediction does not cover the full "
                "controller horizon."
            )
    uncontrolled = _denormalize(pred_norm, mean, std)

    output_dir = (
        os.fspath(controller_output_dir)
        if controller_output_dir is not None
        else os.path.join(base_output_dir, "control", controller)
    )
    os.makedirs(output_dir, exist_ok=True)
    summary_output_dir = (
        str(artifact_relative_path)
        if artifact_relative_path is not None
        else output_dir
    )
    provenance_fields = dict(model_provenance or {})
    provenance_fields.setdefault(
        "control_model_source",
        str(getattr(config, "CONTROL_MODEL_SOURCE", "validation_selected")),
    )
    provenance_fields.setdefault(
        "reference_type", "empirical_quiet_state_reference"
    )
    provenance_fields.setdefault(
        "regulation_objective",
        "regulation toward an empirical quiet-state reference",
    )

    spike_threshold = float(getattr(config, "SPIKE_THRESHOLD", 2.0))
    settling_tolerance = float(
        getattr(config, "CONTROL_SETTLING_TOLERANCE", 0.25)
    )
    settling_consecutive = int(
        getattr(config, "CONTROL_SETTLING_CONSECUTIVE", 25)
    )
    control_input_clip = getattr(config, "CONTROL_INPUT_CLIP", None)
    divergence_abs_limit = float(
        getattr(config, "CONTROL_DIVERGENCE_ABS_LIMIT", 1e6)
    )
    pyragas_history_signal = str(pyragas_history_signal).strip().lower()
    if pyragas_history_signal not in {
        "raw_readout",
        "corrected_feedback_input",
    }:
        raise ValueError(
            "pyragas_history_signal must be 'raw_readout' or "
            "'corrected_feedback_input'."
        )

    print(f"[Control] Controller: {controller}")
    print(f"[Control] Output dir : {output_dir}")
    print(f"[Control] Target raw : {target_raw}")
    print(
        "[Control] Controller validation/test windows: "
        f"[{validation_start}, {validation_end}) / "
        f"[{controller_test_start}, {controller_test_end})"
    )
    if controller == "pyragas":
        print(f"[Control] Pyragas delay: {int(pyragas_delay)}")
        print(f"[Control] Pyragas sign : {int(pyragas_sign)}")
        print(f"[Control] Pyragas history signal: {pyragas_history_signal}")

    def evaluate_validation_gain(gain):
        row, _, _, _ = _evaluate_k(
            esn=esn,
            train_norm=train_norm,
            n_base=validation_end,
            K=gain,
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
            control_target_mode=canonical_target_mode,
            hr_mode=hr_mode,
            optimizer_name=optimizer_name,
            spike_threshold=spike_threshold,
            settling_tolerance=settling_tolerance,
            settling_consecutive=settling_consecutive,
            controller=controller,
            finite_s=finite_s,
            pyragas_delay=pyragas_delay,
            pyragas_sign=pyragas_sign,
            pyragas_history_signal=pyragas_history_signal,
            control_input_clip=control_input_clip,
            divergence_abs_limit=divergence_abs_limit,
            metric_start_idx=validation_start,
            metric_end_idx=validation_end,
            metric_segment="controller_validation",
            discard_pyragas_transient=True,
        )
        return row

    selection_started = time.perf_counter()
    validation_rows = []
    if locked_validation_selection is not None:
        selected_validation = dict(locked_validation_selection)
        if (
            not bool(selected_validation.get("stable", False))
            or bool(selected_validation.get("divergence_detected", False))
            or not np.isfinite(
                _safe_float(selected_validation.get("selection_score"))
            )
        ):
            raise ValueError(
                "Locked controller validation selection must be stable and "
                "have a finite validation score."
            )
        validation_rows = [selected_validation]
        selection_runtime_seconds = 0.0
    else:
        coarse_values = _k_values(control_k, config, k_min, k_max, k_num)
        for gain in coarse_values:
            print(f"[Control] Validation coarse K={gain:.6f}")
            validation_rows.append(evaluate_validation_gain(gain))

        try:
            if auto_control_k and control_k is None:
                coarse_best = _best_row(validation_rows, config)
                refine_values = _refined_k_values(
                    coarse_best["K"], config, k_min, k_max, k_refine_num
                )
                existing = {round(float(row["K"]), 10) for row in validation_rows}
                for gain in refine_values:
                    if round(float(gain), 10) in existing:
                        continue
                    print(f"[Control] Validation refined K={gain:.6f}")
                    validation_rows.append(evaluate_validation_gain(gain))

            selected_validation = _best_row(validation_rows, config)
            selection_runtime_seconds = time.perf_counter() - selection_started
        except NoStableControllerCandidateError as exc:
            selection_runtime_seconds = time.perf_counter() - selection_started
            if not validation_only:
                raise

            rejection_reasons = sorted(
                {
                    str(row.get("divergence_reason") or "nonfinite_selection_score")
                    for row in validation_rows
                }
            )
            rejection_summary = {
                "schema_version": "chapter1_control_v2",
                **provenance_fields,
                "controller": controller,
                "optimizer": optimizer_name,
                "validation_only": True,
                "controller_test_evaluated": False,
                "candidate_status": "rejected_no_stable_validation_gain",
                "candidate_rejected": True,
                "rejected": True,
                "stable": False,
                "evaluated_steps": int(
                    max(
                        [
                            int(row.get("steps_completed", 0))
                            for row in validation_rows
                        ]
                        or [0]
                    )
                ),
                "selection_runtime_seconds": float(selection_runtime_seconds),
                "final_test_runtime_seconds": None,
                "selection_metric_name": "controller_validation_selection_score",
                "selection_metric_value": None,
                "selection_metric_segment": "controller_validation",
                "final_test_metric_name": None,
                "final_test_metric_value": None,
                "final_test_metric_segment": None,
                "target_mode": canonical_target_mode,
                "target_state": target_raw,
                "target_state_normalized": target_norm,
                "target_metadata": target_metadata,
                "control_start_index": int(control_start_idx),
                "control_start_idx": int(control_start_idx),
                "control_start_time": control_start_time,
                "control_horizon": int(n_base),
                "evaluated_horizon": int(validation_end),
                **split,
                "divergence_detected": True,
                "divergence_reason": "; ".join(rejection_reasons),
                "validation_rejection_reasons": rejection_reasons,
                "finite_s": float(finite_s) if controller == "finite_time" else None,
                "finite_time_feedback_law": (
                    "global_piecewise_linear_and_fractional_power"
                    if controller == "finite_time"
                    else None
                ),
                "pyragas_delay": int(pyragas_delay) if controller == "pyragas" else None,
                "pyragas_sign": int(pyragas_sign) if controller == "pyragas" else None,
                "pyragas_history_signal": (
                    pyragas_history_signal if controller == "pyragas" else None
                ),
                "validation_metrics": {},
                "test_metrics": None,
                "best": None,
                "all_rows": validation_rows,
                "output_dir": summary_output_dir,
            }
            _save_csv(validation_rows, os.path.join(output_dir, "k_sweep.csv"))
            _save_json(
                rejection_summary,
                os.path.join(output_dir, "control_summary.json"),
            )
            if generate_plots:
                plot_k_sweep_summary(
                    validation_rows, output_dir, controller_name=controller
                )
            print(f"[Control] Validation candidate rejected: {exc}")
            print("[Control] Controller test was not evaluated; outer search may continue.")
            return rejection_summary

    best_k = float(selected_validation["K"])
    selection_metric_value = _safe_float(
        selected_validation.get("selection_score"), np.inf
    )
    print(
        f"[Control] Selected K={best_k:.6f} on controller validation "
        f"(score={selection_metric_value:.6g})"
    )

    if validation_only:
        validation_summary = {
            "schema_version": "chapter1_control_v2",
            **provenance_fields,
            "controller": controller,
            "optimizer": optimizer_name,
            "best_k": best_k,
            "best_K": best_k,
            "stable": bool(selected_validation.get("stable", False)),
            "rejected": False,
            "evaluated_steps": int(
                selected_validation.get("steps_completed", validation_end)
            ),
            "selection_runtime_seconds": float(selection_runtime_seconds),
            "final_test_runtime_seconds": None,
            "target_mode": canonical_target_mode,
            "target_state": target_raw,
            "target_state_normalized": target_norm,
            "target_metadata": target_metadata,
            "control_start_index": int(control_start_idx),
            "control_start_idx": int(control_start_idx),
            "control_start_time": control_start_time,
            "control_horizon": int(n_base),
            "evaluated_horizon": int(validation_end),
            "controller_validation_start": int(validation_start),
            "controller_validation_end": int(validation_end),
            "controller_test_start": int(controller_test_start),
            "controller_test_end": int(controller_test_end),
            "controller_validation_frac": split[
                "controller_validation_fraction_normalized"
            ],
            "controller_test_frac": split[
                "controller_test_fraction_normalized"
            ],
            **split,
            "validation_only": True,
            "controller_test_evaluated": False,
            "selection_metric_name": "controller_validation_selection_score",
            "selection_metric_value": selection_metric_value,
            "selection_metric_segment": "controller_validation",
            "final_test_metric_name": None,
            "final_test_metric_value": None,
            "final_test_metric_segment": None,
            "raw_readout_metrics": {},
            "corrected_feedback_input_metrics": {},
            "control_effort_mean_sq": None,
            "control_energy_dt_sum": None,
            "control_input_clip": control_input_clip,
            "control_divergence_abs_limit": divergence_abs_limit,
            "control_safety_coordinate_system": "normalized_esn_coordinates",
            "controller_law_coordinate_system": "normalized_esn_coordinates",
            "divergence_detected": bool(
                selected_validation.get("divergence_detected", False)
            ),
            "divergence_reason": selected_validation.get("divergence_reason"),
            "divergence_index": selected_validation.get("divergence_index"),
            "finite_s": float(finite_s) if controller == "finite_time" else None,
            "finite_time_feedback_law": (
                "global_piecewise_linear_and_fractional_power"
                if controller == "finite_time"
                else None
            ),
            "pyragas_delay": (
                int(pyragas_delay) if controller == "pyragas" else None
            ),
            "pyragas_sign": (
                int(pyragas_sign) if controller == "pyragas" else None
            ),
            "pyragas_history_signal": (
                pyragas_history_signal if controller == "pyragas" else None
            ),
            "validation_metrics": selected_validation,
            "test_metrics": None,
            "best": selected_validation,
            "all_rows": validation_rows,
            "target_raw": target_raw,
            "target_norm": target_norm,
            "output_dir": summary_output_dir,
        }
        _save_csv(validation_rows, os.path.join(output_dir, "k_sweep.csv"))
        _save_json(
            validation_summary,
            os.path.join(output_dir, "control_summary.json"),
        )
        if generate_plots:
            plot_k_sweep_summary(
                validation_rows, output_dir, controller_name=controller
            )
        print("[Control] Validation-only run: controller test was not evaluated.")
        return validation_summary

    final_test_started = time.perf_counter()
    test_row, best_raw_readout, best_corrected_feedback_input, best_control_signal = (
        _evaluate_k(
            esn=esn,
            train_norm=train_norm,
            n_base=n_base,
            K=best_k,
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
            control_target_mode=canonical_target_mode,
            hr_mode=hr_mode,
            optimizer_name=optimizer_name,
            spike_threshold=spike_threshold,
            settling_tolerance=settling_tolerance,
            settling_consecutive=settling_consecutive,
            controller=controller,
            finite_s=finite_s,
            pyragas_delay=pyragas_delay,
            pyragas_sign=pyragas_sign,
            pyragas_history_signal=pyragas_history_signal,
            control_input_clip=control_input_clip,
            divergence_abs_limit=divergence_abs_limit,
            metric_start_idx=controller_test_start,
            metric_end_idx=controller_test_end,
            metric_segment="controller_test",
            discard_pyragas_transient=False,
            save_dir=os.path.join(output_dir, "best_rollout"),
        )
    )
    final_test_runtime_seconds = time.perf_counter() - final_test_started

    if controller == "pyragas":
        # This is a held-out diagnostic only; it is never used to choose K.
        _selection_score(test_row, config)
        test_row["pyragas_selection_status"] = selected_validation.get(
            "pyragas_selection_status"
        )

    raw_readout_metrics = {
        "metric_segment": "controller_test",
        "target_rmse_state": test_row.get("raw_readout_target_rmse_state"),
        "target_rmse_x": test_row.get("raw_readout_target_rmse_x"),
        "target_nrmse_x": test_row.get("raw_readout_target_nrmse_x"),
    }
    corrected_feedback_input_metrics = {
        "metric_segment": "controller_test",
        "target_rmse_state": test_row.get(
            "corrected_feedback_input_target_rmse_state"
        ),
        "target_rmse_x": test_row.get(
            "corrected_feedback_input_target_rmse_x"
        ),
        "target_nrmse_x": test_row.get(
            "corrected_feedback_input_target_nrmse_x"
        ),
    }

    if controller == "pyragas":
        final_test_metric_name = "pyragas_empirical_recurrence_error_norm"
    else:
        final_test_metric_name = (
            "corrected_feedback_input_target_rmse_state"
        )
    final_test_metric_value = test_row.get(final_test_metric_name)

    best_record = {
        **test_row,
        "selection_metric_name": "controller_validation_selection_score",
        "selection_metric_value": selection_metric_value,
        "validation_metrics": selected_validation,
        "test_metrics": test_row,
        # Compatibility: this is the validation score, never a test-derived score.
        "selection_score": selection_metric_value,
    }

    summary = {
        "schema_version": "chapter1_control_v2",
        **provenance_fields,
        "controller": controller,
        "validation_only": False,
        "controller_test_evaluated": True,
        "selection_runtime_seconds": float(selection_runtime_seconds),
        "final_test_runtime_seconds": float(final_test_runtime_seconds),
        "output_dir": summary_output_dir,
        "optimizer": optimizer_name,
        "best_k": best_k,
        "best_K": best_k,
        "stable": bool(test_row.get("stable", False)),
        "target_mode": canonical_target_mode,
        "target_state": target_raw,
        "target_state_normalized": target_norm,
        "target_metadata": target_metadata,
        "control_start_index": int(control_start_idx),
        "control_start_idx": int(control_start_idx),
        "control_start_time": control_start_time,
        "control_horizon": int(n_base),
        "controller_validation_start": int(validation_start),
        "controller_validation_end": int(validation_end),
        "controller_test_start": int(controller_test_start),
        "controller_test_end": int(controller_test_end),
        "controller_validation_frac": split[
            "controller_validation_fraction_normalized"
        ],
        "controller_test_frac": split[
            "controller_test_fraction_normalized"
        ],
        **split,
        "selection_metric_name": "controller_validation_selection_score",
        "selection_metric_value": selection_metric_value,
        "selection_metric_segment": "controller_validation",
        "final_test_metric_name": final_test_metric_name,
        "final_test_metric_value": final_test_metric_value,
        "final_test_metric_segment": "controller_test",
        "raw_readout_metrics": raw_readout_metrics,
        "corrected_feedback_input_metrics": corrected_feedback_input_metrics,
        "control_effort_mean_sq": test_row.get("control_effort_mean_sq"),
        "control_energy_dt_sum": test_row.get("control_energy_dt_sum"),
        "control_energy_dt_sum_explanation": (
            "median positive sample dt multiplied by sum_k ||u_k||^2 "
            "on the controller-test segment; null when dt is unavailable"
        ),
        "control_input_clip": control_input_clip,
        "control_divergence_abs_limit": divergence_abs_limit,
        "control_safety_coordinate_system": "normalized_esn_coordinates",
        "controller_law_coordinate_system": "normalized_esn_coordinates",
        "divergence_detected": bool(
            test_row.get("divergence_detected", False)
        ),
        "divergence_reason": test_row.get("divergence_reason"),
        "divergence_index": test_row.get("divergence_index"),
        "finite_s": float(finite_s) if controller == "finite_time" else None,
        "finite_time_feedback_law": (
            "global_piecewise_linear_and_fractional_power"
            if controller == "finite_time"
            else None
        ),
        "pyragas_delay": int(pyragas_delay) if controller == "pyragas" else None,
        "pyragas_sign": int(pyragas_sign) if controller == "pyragas" else None,
        "pyragas_history_signal": (
            pyragas_history_signal if controller == "pyragas" else None
        ),
        "validation_metrics": selected_validation,
        "test_metrics": test_row,
        "best": best_record,
        "all_rows": validation_rows,
        "target_raw": target_raw,
        "target_norm": target_norm,
        "legacy_aliases": {
            "controlled_x/y/z": "corrected_feedback_input_x/y/z",
            "u_x/y/z": "control_signal_x/y/z",
            "target_rmse_*": "corrected_feedback_input_target_rmse_*",
            "control_energy": "control_effort_mean_sq",
        },
    }

    # Flat compatibility fields are held-out controller-test values.
    summary.update(
        {
            "raw_readout_target_rmse_state": test_row.get(
                "raw_readout_target_rmse_state"
            ),
            "raw_readout_target_rmse_x": test_row.get(
                "raw_readout_target_rmse_x"
            ),
            "raw_readout_target_nrmse_x": test_row.get(
                "raw_readout_target_nrmse_x"
            ),
            "corrected_feedback_input_target_rmse_state": test_row.get(
                "corrected_feedback_input_target_rmse_state"
            ),
            "corrected_feedback_input_target_rmse_x": test_row.get(
                "corrected_feedback_input_target_rmse_x"
            ),
            "corrected_feedback_input_target_nrmse_x": test_row.get(
                "corrected_feedback_input_target_nrmse_x"
            ),
            "target_rmse_state": test_row.get("target_rmse_state"),
            "target_rmse_x": test_row.get("target_rmse_x"),
            "spike_reduction_percent": test_row.get(
                "spike_reduction_percent"
            ),
            "controller_test_time_to_tolerance": test_row.get(
                "evaluation_time_to_tolerance"
            ),
            "settling_time": test_row.get("evaluation_time_to_tolerance"),
            "settling_time_alias_of": "controller_test_time_to_tolerance",
            "control_energy": test_row.get("control_effort_mean_sq"),
            "best_target_rmse_state": test_row.get("target_rmse_state"),
            "best_target_rmse_x": test_row.get("target_rmse_x"),
            "best_spike_reduction_percent": test_row.get(
                "spike_reduction_percent"
            ),
            "best_control_energy": test_row.get("control_effort_mean_sq"),
            "best_settling_time": test_row.get("evaluation_time_to_tolerance"),
            "best_stable": test_row.get("stable"),
        }
    )
    if controller == "pyragas":
        for key, value in test_row.items():
            if key.startswith("pyragas_"):
                summary[key] = value
                summary[f"best_{key}"] = value

    _save_csv(validation_rows, os.path.join(output_dir, "k_sweep.csv"))
    _save_json(
        best_record,
        os.path.join(output_dir, "best_rollout", "metrics.json"),
    )
    _save_json(summary, os.path.join(output_dir, "control_summary.json"))

    if not summary["stable"]:
        reason = summary.get("divergence_reason") or "nonfinite_final_rollout"
        raise RuntimeError(
            f"Selected {controller} controller failed on controller test: {reason}"
        )

    if generate_plots:
        plot_metrics = {
            **test_row,
            "K": best_k,
            "selection_metric_value": selection_metric_value,
            "controller_validation_start": validation_start,
            "controller_validation_end": validation_end,
            "controller_test_start": controller_test_start,
            "controller_test_end": controller_test_end,
        }
        plot_controlled_vs_uncontrolled_x(
            test_times_aligned,
            test_aligned,
            uncontrolled,
            best_corrected_feedback_input,
            target_raw,
            control_start_idx,
            plot_metrics,
            output_dir,
            controller_name=controller,
        )
        plot_raw_readout_vs_corrected_feedback_input_x(
            test_times_aligned,
            best_raw_readout,
            best_corrected_feedback_input,
            control_start_idx,
            output_dir,
            controller_name=controller,
            metrics=plot_metrics,
        )
        plot_controlled_all_states(
            test_times_aligned,
            test_aligned,
            uncontrolled,
            best_corrected_feedback_input,
            target_raw,
            control_start_idx,
            output_dir,
            controller_name=controller,
            metrics=plot_metrics,
        )
        plot_control_signal(
            test_times_aligned,
            best_control_signal,
            control_start_idx,
            output_dir,
            controller_name=controller,
            metrics=plot_metrics,
        )

        uncontrolled_error = _compute_error_norms(uncontrolled, target_raw)
        corrected_feedback_error = _compute_error_norms(
            best_corrected_feedback_input, target_raw
        )
        plot_control_error(
            test_times_aligned,
            uncontrolled_error,
            corrected_feedback_error,
            control_start_idx,
            settling_tolerance,
            output_dir,
            controller_name=controller,
            metrics=plot_metrics,
        )
        plot_k_sweep_summary(
            validation_rows, output_dir, controller_name=controller
        )

    global_row = {
        "regime": hr_mode,
        "controller": controller,
        "optimizer": optimizer_name,
        "metric_segment": "controller_test",
        "best_k": best_k,
        "best_K": best_k,
        "selection_metric_name": summary["selection_metric_name"],
        "selection_metric_value": selection_metric_value,
        "final_test_metric_name": final_test_metric_name,
        "final_test_metric_value": final_test_metric_value,
        "raw_readout_target_rmse_state": test_row.get(
            "raw_readout_target_rmse_state"
        ),
        "raw_readout_target_rmse_x": test_row.get(
            "raw_readout_target_rmse_x"
        ),
        "corrected_feedback_input_target_rmse_state": test_row.get(
            "corrected_feedback_input_target_rmse_state"
        ),
        "corrected_feedback_input_target_rmse_x": test_row.get(
            "corrected_feedback_input_target_rmse_x"
        ),
        "control_effort_mean_sq": test_row.get("control_effort_mean_sq"),
        "control_energy_dt_sum": test_row.get("control_energy_dt_sum"),
        "controller_test_time_to_tolerance": test_row.get(
            "evaluation_time_to_tolerance"
        ),
        "divergence_detected": summary["divergence_detected"],
        "divergence_reason": summary["divergence_reason"],
        "finite_s": summary["finite_s"],
        "pyragas_delay": summary["pyragas_delay"],
        "pyragas_sign": summary["pyragas_sign"],
        "pyragas_history_signal": summary["pyragas_history_signal"],
        "controller_validation_start": validation_start,
        "controller_validation_end": validation_end,
        "controller_test_start": controller_test_start,
        "controller_test_end": controller_test_end,
        # Historical table aliases now explicitly map to held-out test metrics.
        "best_target_rmse_state": test_row.get("target_rmse_state"),
        "best_target_rmse_x": test_row.get("target_rmse_x"),
        "best_spike_reduction_percent": test_row.get(
            "spike_reduction_percent"
        ),
        "best_control_energy": test_row.get("control_effort_mean_sq"),
        "best_settling_time": test_row.get("evaluation_time_to_tolerance"),
        "best_stable": test_row.get("stable"),
        "output_dir": summary_output_dir,
    }
    if controller == "pyragas":
        for key, value in test_row.items():
            if key.startswith("pyragas_"):
                global_row[f"best_{key}"] = value

    if append_global_comparison:
        _append_global_control_comparison(
            os.path.dirname(base_output_dir), global_row
        )
    return summary


def run_linear_feedback_control_experiment(**kwargs):
    """
    Backward-compatible wrapper for your current main.py.
    """
    kwargs["controller"] = "linear_feedback"
    return run_control_experiment(**kwargs)
