from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


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


def _plot_controlled_vs_uncontrolled_x(
    times,
    truth,
    uncontrolled,
    controlled,
    target_state,
    control_start_idx,
    metrics,
    output_dir,
):
    times = _as_1d(times)
    truth = _as_2d(truth)
    uncontrolled = _as_2d(uncontrolled)
    controlled = _as_2d(controlled)
    target_state = _as_1d(target_state)

    n = min(len(times), len(truth), len(uncontrolled), len(controlled))
    times = times[:n]
    truth = truth[:n]
    uncontrolled = uncontrolled[:n]
    controlled = controlled[:n]

    control_start_idx = int(max(0, min(control_start_idx, n - 1)))
    t0 = times[control_start_idx]

    # zoom window
    left = max(0, control_start_idx - 80)
    right = min(n, control_start_idx + 120)

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=False,
        gridspec_kw={"height_ratios": [2, 1.3]}
    )

    # -------- top: full trajectory --------
    ax = axes[0]
    ax.plot(times, truth[:, 0], color="black", linewidth=1.6, label="True x")
    ax.plot(times, uncontrolled[:, 0], linewidth=1.3, label="Uncontrolled ESN x")
    ax.plot(times, controlled[:, 0], linestyle="--", linewidth=1.6, label="Controlled ESN x")
    ax.axhline(target_state[0], linestyle=":", linewidth=1.4, label="Target x")
    ax.axvline(t0, linestyle="--", linewidth=1.3, label="Control start")
    ax.axvspan(t0, times[-1], alpha=0.08)

    txt = (
        f"K = {metrics.get('K'):.4f}\n"
        f"Target RMSE = {metrics.get('target_rmse_state'):.3e}\n"
        f"Spike reduction = {metrics.get('spike_reduction_percent'):.1f}%\n"
        f"Energy = {metrics.get('control_energy'):.3e}"
    )

    ax.text(
        0.015, 0.97, txt,
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    ax.set_title("Linear feedback control: x-state comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("x state")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    # -------- bottom: zoom near control start --------
    ax2 = axes[1]
    ax2.plot(times[left:right], truth[left:right, 0], color="black", linewidth=1.6, label="True x")
    ax2.plot(times[left:right], uncontrolled[left:right, 0], linewidth=1.3, label="Uncontrolled ESN x")
    ax2.plot(times[left:right], controlled[left:right, 0], linestyle="--", linewidth=1.6, label="Controlled ESN x")
    ax2.axhline(target_state[0], linestyle=":", linewidth=1.4, label="Target x")
    ax2.axvline(t0, linestyle="--", linewidth=1.3, label="Control start")
    ax2.axvspan(t0, times[right - 1], alpha=0.08)

    ax2.set_title("Zoom around control start", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("x state")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(output_dir, "controlled_vs_uncontrolled_x_better.png")
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()

    print(f"[Plot] Saved -> {path}")


def _plot_controlled_all_states(
    times,
    truth,
    uncontrolled,
    controlled,
    target_state,
    control_start_idx,
    output_dir,
):
    times = _as_1d(times)
    truth = _as_2d(truth)
    uncontrolled = _as_2d(uncontrolled)
    controlled = _as_2d(controlled)
    target_state = _as_1d(target_state)

    n = min(len(times), len(truth), len(uncontrolled), len(controlled))
    times = times[:n]
    truth = truth[:n]
    uncontrolled = uncontrolled[:n]
    controlled = controlled[:n]

    control_start_idx = int(max(0, min(control_start_idx, n - 1)))

    names = [
        "x: membrane voltage / spike variable",
        "y: recovery variable",
        "z: slow adaptation variable",
    ]

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(times, truth[:, i], color="black", linewidth=1.2, label=f"True {i}")
        ax.plot(times, uncontrolled[:, i], linewidth=1.1, label=f"Uncontrolled {i}")
        ax.plot(times, controlled[:, i], linestyle="--", linewidth=1.3, label=f"Controlled {i}")
        ax.axhline(target_state[i], linestyle=":", linewidth=1.3, label=f"Target {i}")
        ax.axvline(times[control_start_idx], linestyle="--", linewidth=1.2, label="Control start")

        ax.set_ylabel(["x", "y", "z"][i])
        ax.set_title(names[i], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Linear feedback control: all HR states", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(output_dir, "controlled_all_states.png")
    plt.savefig(path, dpi=180)
    plt.close()

    print(f"[Plot] Saved -> {path}")


def _plot_control_signal(times, control_signal, control_start_idx, output_dir):
    times = _as_1d(times)
    control_signal = _as_2d(control_signal)

    n = min(len(times), len(control_signal))
    times = times[:n]
    control_signal = control_signal[:n]

    control_start_idx = int(max(0, min(control_start_idx, n - 1)))
    control_norm = np.linalg.norm(control_signal, axis=1)

    plt.figure(figsize=(15, 5))

    plt.plot(times, control_norm, linewidth=1.4, label="||u(t)||")
    plt.plot(times, control_signal[:, 0], linestyle="--", linewidth=1.2, label="u_x(t)")
    plt.axvline(times[control_start_idx], linestyle="--", linewidth=1.3, label="Control start")

    plt.title("Linear feedback control signal", fontsize=14, fontweight="bold")
    plt.xlabel("Time (s)")
    plt.ylabel("Control signal")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()

    path = os.path.join(output_dir, "control_signal.png")
    plt.savefig(path, dpi=180)
    plt.close()

    print(f"[Plot] Saved -> {path}")


def _plot_error_over_time(
    times,
    uncontrolled_error_norm,
    controlled_error_norm,
    control_start_idx,
    settling_tolerance,
    output_dir,
):
    times = _as_1d(times)
    uncontrolled_error_norm = _as_1d(uncontrolled_error_norm)
    controlled_error_norm = _as_1d(controlled_error_norm)

    n = min(len(times), len(uncontrolled_error_norm), len(controlled_error_norm))
    times = times[:n]
    uncontrolled_error_norm = uncontrolled_error_norm[:n]
    controlled_error_norm = controlled_error_norm[:n]

    control_start_idx = int(max(0, min(control_start_idx, n - 1)))

    plt.figure(figsize=(15, 5))

    plt.plot(times, uncontrolled_error_norm, linewidth=1.2, label="Uncontrolled error")
    plt.plot(times, controlled_error_norm, linestyle="--", linewidth=1.4, label="Controlled error")
    plt.axhline(settling_tolerance, linestyle=":", linewidth=1.3, label="Settling tolerance")
    plt.axvline(times[control_start_idx], linestyle="--", linewidth=1.3, label="Control start")

    plt.title("Target-tracking error", fontsize=14, fontweight="bold")
    plt.xlabel("Time (s)")
    plt.ylabel("||state - target||")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()

    path = os.path.join(output_dir, "control_error.png")
    plt.savefig(path, dpi=180)
    plt.close()

    print(f"[Plot] Saved -> {path}")


def _plot_k_sweep_summary(rows, output_dir):
    if not rows:
        return

    stable_rows = [r for r in rows if bool(r.get("stable", False))]

    if not stable_rows:
        stable_rows = rows

    k = np.array([float(r.get("K", np.nan)) for r in stable_rows], dtype=float)
    rmse = np.array([float(r.get("target_rmse_state", np.nan)) for r in stable_rows], dtype=float)
    spike = np.array([float(r.get("spike_reduction_percent", np.nan)) for r in stable_rows], dtype=float)
    energy = np.array([float(r.get("control_energy", np.nan)) for r in stable_rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(k, rmse, marker="o")
    axes[0].set_ylabel("Target RMSE")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(k, spike, marker="o")
    axes[1].set_ylabel("Spike reduction (%)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(k, energy, marker="o")
    axes[2].set_ylabel("Control energy")
    axes[2].set_xlabel("K")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("Linear-feedback K sweep summary", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(output_dir, "control_sweep_summary.png")
    plt.savefig(path, dpi=180)
    plt.close()

    print(f"[Plot] Saved -> {path}")


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
    settling_tol,
    settling_consecutive,
    max_abs_value,
    run_dir,
    save_outputs=False,
):
    control_rollout = esn.predict_controlled(
        train_sequence=train_norm,
        horizon_steps=n_base,
        target=target_norm,
        K=K,
        control_start_idx=control_start_idx,
        max_abs_value=max_abs_value,
    )

    controlled_norm = control_rollout["controlled_output_norm"][:n_base]
    control_signal_norm = control_rollout["control_signal_norm"][:n_base]

    controlled = _denormalize(controlled_norm, mean, std)

    # Convert normalized control signal approximately to original scale
    control_signal_raw = control_signal_norm * np.asarray(std, dtype=float)

    stable = (
        bool(control_rollout.get("stable", False))
        and np.all(np.isfinite(controlled))
        and np.all(np.isfinite(control_signal_raw))
    )

    if stable:
        metrics = _summarize_control_metrics(
            times=test_times_aligned,
            uncontrolled=uncontrolled,
            controlled=controlled,
            target_state=target_raw,
            control_signal=control_signal_raw,
            control_start_idx=control_start_idx,
            spike_threshold=spike_threshold,
            settling_tolerance=settling_tol,
            settling_consecutive=settling_consecutive,
        )

        row = {
            "regime": hr_mode,
            "optimizer": optimizer_name,
            "K": float(K),
            "stable": True,
            "target_mode": str(control_target_mode),
            "control_start_frac": float(control_start_frac),
            "control_start_idx": int(control_start_idx),
            "control_start_time": float(control_start_time),
            "target_x": float(target_raw[0]),
            "target_y": float(target_raw[1]),
            "target_z": float(target_raw[2]),
            "run_dir": run_dir,
            **metrics,
        }

        uncontrolled_error_norm = _compute_error_norms(uncontrolled, target_raw)
        controlled_error_norm = _compute_error_norms(controlled, target_raw)

        if save_outputs:
            os.makedirs(run_dir, exist_ok=True)

            _save_json(row, os.path.join(run_dir, "control_metrics.json"))
            _save_csv([row], os.path.join(run_dir, "control_metrics.csv"))

            _save_rollout_csv(
                path=os.path.join(run_dir, "controlled_rollout.csv"),
                times=test_times_aligned,
                truth=test_aligned,
                uncontrolled=uncontrolled,
                controlled=controlled,
                control_signal=control_signal_raw,
                target_state=target_raw,
            )

            _plot_controlled_vs_uncontrolled_x(
                times=test_times_aligned,
                truth=test_aligned,
                uncontrolled=uncontrolled,
                controlled=controlled,
                target_state=target_raw,
                control_start_idx=control_start_idx,
                metrics=row,
                output_dir=run_dir,
            )

            _plot_controlled_all_states(
                times=test_times_aligned,
                truth=test_aligned,
                uncontrolled=uncontrolled,
                controlled=controlled,
                target_state=target_raw,
                control_start_idx=control_start_idx,
                output_dir=run_dir,
            )

            _plot_control_signal(
                times=test_times_aligned,
                control_signal=control_signal_raw,
                control_start_idx=control_start_idx,
                output_dir=run_dir,
            )

            _plot_error_over_time(
                times=test_times_aligned,
                uncontrolled_error_norm=uncontrolled_error_norm,
                controlled_error_norm=controlled_error_norm,
                control_start_idx=control_start_idx,
                settling_tolerance=settling_tol,
                output_dir=run_dir,
            )

    else:
        row = {
            "regime": hr_mode,
            "optimizer": optimizer_name,
            "K": float(K),
            "stable": False,
            "target_mode": str(control_target_mode),
            "control_start_frac": float(control_start_frac),
            "control_start_idx": int(control_start_idx),
            "control_start_time": float(control_start_time),
            "target_x": float(target_raw[0]),
            "target_y": float(target_raw[1]),
            "target_z": float(target_raw[2]),
            "run_dir": run_dir,
            "target_rmse_state": float("inf"),
            "target_rmse_x": float("inf"),
            "uncontrolled_target_rmse_state": float("inf"),
            "uncontrolled_target_rmse_x": float("inf"),
            "spike_count_before": 0,
            "spike_count_after": 0,
            "uncontrolled_spike_count_after": 0,
            "spike_reduction_percent": float("nan"),
            "control_energy": float("inf"),
            "mean_control_norm": float("inf"),
            "max_control_norm": float("inf"),
            "settling_time": float("nan"),
            "mean_error_norm_post": float("inf"),
            "max_error_norm_post": float("inf"),
        }

        if save_outputs:
            os.makedirs(run_dir, exist_ok=True)
            _save_json(row, os.path.join(run_dir, "control_metrics.json"))
            _save_csv([row], os.path.join(run_dir, "control_metrics.csv"))

    return row


def run_linear_feedback_control_experiment(
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
    best_params,
    optimizer_name,
    control_k=None,
    control_start_frac=0.20,
    control_target_mode="rest_state",
    auto_control_k=False,
    k_min=None,
    k_max=None,
    k_num=None,
    k_refine_num=None,
):
    train = _as_2d(train)
    test = _as_2d(test)
    train_norm = _as_2d(train_norm)
    test_norm = _as_2d(test_norm)
    times = _as_1d(times)

    if not hasattr(esn, "predict_controlled"):
        raise AttributeError(
            "Your model.py does not contain predict_controlled(). "
            "Replace model.py with the updated version first."
        )

    control_root = os.path.join(base_output_dir, "control", "linear_feedback")
    os.makedirs(control_root, exist_ok=True)

    test_times = times[len(train) : len(train) + len(test)]

    if len(test_times) != len(test):
        test_times = np.arange(len(test), dtype=float)

    horizon_steps = len(test)

    control_start_frac = float(np.clip(control_start_frac, 0.0, 1.0))
    control_start_idx = int(round(control_start_frac * horizon_steps))
    control_start_idx = max(0, min(control_start_idx, horizon_steps - 1))
    control_start_time = float(test_times[control_start_idx])

    target_raw, target_norm = _choose_target_state(
        train_raw=train,
        mean=mean,
        std=std,
        target_mode=control_target_mode,
        config=config,
    )

    auto_control_k = bool(auto_control_k)

    if auto_control_k:
        coarse_k_values = _auto_k_coarse_values(
            config=config,
            k_min=k_min,
            k_max=k_max,
            k_num=k_num,
        )
        refine_count = int(
            k_refine_num
            if k_refine_num is not None
            else getattr(config, "CONTROL_AUTO_K_REFINE_NUM", 15)
        )
        k_values_print = f"auto coarse={len(coarse_k_values)}, refine={refine_count}"
    else:
        coarse_k_values = _get_k_values(control_k, config)
        k_values_print = str(coarse_k_values)

    spike_threshold = float(getattr(config, "SPIKE_THRESHOLD", 2.0))
    settling_tol = float(getattr(config, "CONTROL_SETTLING_TOL", 0.15))

    settling_consecutive = int(
        getattr(
            config,
            "CONTROL_SETTLING_HOLD_STEPS",
            getattr(config, "CONTROL_SETTLING_CONSECUTIVE", 50),
        )
    )

    max_abs_value = float(
        getattr(
            config,
            "CONTROL_DIVERGENCE_ABS_LIMIT",
            getattr(config, "CONTROL_MAX_ABS_VALUE", 1e6),
        )
    )

    print("\n" + "=" * 72)
    print("LINEAR FEEDBACK CONTROL EXPERIMENT")
    print("=" * 72)
    print(f"[Control] output folder       = {control_root}")
    print(f"[Control] HR mode             = {hr_mode}")
    print(f"[Control] target mode         = {control_target_mode}")
    print(f"[Control] target raw state    = {np.round(target_raw, 6).tolist()}")
    print(
        f"[Control] control start       = step {control_start_idx} / {horizon_steps} "
        f"(t = {control_start_time:.4f} s)"
    )
    print(f"[Control] spike threshold     = {spike_threshold:.4f}")
    print(f"[Control] K search            = {k_values_print}")

    _save_json(
        {
            "target_mode": control_target_mode,
            "target_raw_state": target_raw,
            "target_norm_state": target_norm,
            "control_start_frac": control_start_frac,
            "control_start_idx": control_start_idx,
            "control_start_time": control_start_time,
        },
        os.path.join(control_root, "target_state.json"),
    )

    # Uncontrolled baseline using normal ESN predict()
    eval_norm = np.vstack([train_norm, test_norm])
    warmup_steps = len(train_norm) - 1

    uncontrolled_norm, _ = esn.predict(eval_norm, n_warmup=warmup_steps)
    uncontrolled = _denormalize(uncontrolled_norm, mean, std)

    n_base = min(len(uncontrolled), len(test), len(test_times))
    uncontrolled = uncontrolled[:n_base]
    test_aligned = test[:n_base]
    test_times_aligned = test_times[:n_base]

    def eval_k(K, save_outputs=False):
        run_dir = os.path.join(control_root, f"K_{_slug_k(K)}")

        row = _evaluate_linear_feedback_k(
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
            settling_tol=settling_tol,
            settling_consecutive=settling_consecutive,
            max_abs_value=max_abs_value,
            run_dir=run_dir,
            save_outputs=save_outputs,
        )

        row["selection_score"] = _selection_score(row, config)
        return row

    rows = []

    if auto_control_k:
        print(f"[Control] Auto K coarse search: testing {len(coarse_k_values)} values...")

        coarse_rows = [eval_k(K, save_outputs=False) for K in coarse_k_values]
        best_coarse = _best_row(coarse_rows, config)
        rows.extend(coarse_rows)

        refined_k_values = _auto_k_refined_values(
            best_k=best_coarse.get("K"),
            config=config,
            k_min=k_min,
            k_max=k_max,
            k_refine_num=k_refine_num,
        )

        already = {round(float(r.get("K")), 10) for r in rows}
        refined_k_values = [
            K
            for K in refined_k_values
            if round(float(K), 10) not in already
        ]

        print(
            f"[Control] Auto K refine search around K={float(best_coarse.get('K')):.6f}: "
            f"testing {len(refined_k_values)} values..."
        )

        refined_rows = [eval_k(K, save_outputs=False) for K in refined_k_values]
        rows.extend(refined_rows)

        rows = sorted(rows, key=lambda r: float(r.get("K", 0.0)))
        best_row = _best_row(rows, config)

        # Save detailed files and plots only for the final best K.
        best_row = eval_k(best_row.get("K"), save_outputs=True)
        best_row["selection_score"] = _selection_score(best_row, config)

        # Replace same K row in summary with the saved detailed version.
        updated_rows = []
        replaced = False

        for row in rows:
            if round(float(row.get("K")), 10) == round(float(best_row.get("K")), 10):
                updated_rows.append(best_row)
                replaced = True
            else:
                updated_rows.append(row)

        if not replaced:
            updated_rows.append(best_row)

        rows = sorted(updated_rows, key=lambda r: float(r.get("K", 0.0)))

    else:
        # Original behavior: test the manual K or config sweep and save all outputs.
        for K in coarse_k_values:
            row = eval_k(K, save_outputs=True)
            rows.append(row)

            print(
                f"[Control] K={float(K):.4f} | "
                f"stable={row['stable']} | "
                f"score={_fmt_num(row.get('selection_score'))} | "
                f"target_rmse={_fmt_num(row.get('target_rmse_state'))} | "
                f"spike_reduction={_fmt_num(row.get('spike_reduction_percent'))}% | "
                f"energy={_fmt_num(row.get('control_energy'))} | "
                f"settling={_fmt_num(row.get('settling_time'))}"
            )

        rows = sorted(rows, key=lambda r: float(r.get("K", 0.0)))
        best_row = _best_row(rows, config)

    _save_csv(rows, os.path.join(control_root, "linear_feedback_metrics_summary.csv"))

    summary_payload = {
        "hr_mode": hr_mode,
        "optimizer": optimizer_name,
        "best_params": best_params,
        "target_mode": control_target_mode,
        "target_raw_state": target_raw,
        "target_norm_state": target_norm,
        "control_start_frac": control_start_frac,
        "control_start_idx": control_start_idx,
        "control_start_time": control_start_time,
        "auto_control_k": auto_control_k,
        "selection_rule": {
            "score": (
                "target_rmse_state "
                "+ energy_weight*control_energy "
                "+ settling_weight*settling_time "
                "- spike_weight*(spike_reduction_percent/100)"
            ),
            "energy_weight": float(getattr(config, "CONTROL_SCORE_ENERGY_WEIGHT", 0.01)),
            "settling_weight": float(getattr(config, "CONTROL_SCORE_SETTLING_WEIGHT", 0.001)),
            "spike_weight": float(getattr(config, "CONTROL_SCORE_SPIKE_WEIGHT", 0.0)),
        },
        "results": rows,
        "best_result": best_row,
    }

    _save_json(
        summary_payload,
        os.path.join(control_root, "linear_feedback_metrics_summary.json"),
    )

    _save_json(
        best_row,
        os.path.join(control_root, "best_linear_feedback_result.json"),
    )

    _plot_k_sweep_summary(rows, control_root)

    output_root = os.path.dirname(base_output_dir)

    global_row = {
        "regime": hr_mode,
        "optimizer": optimizer_name,
        "best_K": best_row.get("K"),
        "selection_score": best_row.get("selection_score"),
        "target_mode": control_target_mode,
        "target_rmse_state": best_row.get("target_rmse_state"),
        "target_rmse_x": best_row.get("target_rmse_x"),
        "spike_reduction_percent": best_row.get("spike_reduction_percent"),
        "control_energy": best_row.get("control_energy"),
        "settling_time": best_row.get("settling_time"),
        "control_start_frac": control_start_frac,
        "control_start_time": control_start_time,
        "output_dir": control_root,
    }

    _append_global_control_comparison(output_root, global_row)

    print("\n" + "=" * 72)
    print("AUTOMATIC BEST K RESULT" if auto_control_k else "BEST K RESULT")
    print("=" * 72)
    print(f"Best K              : {float(best_row.get('K')):.6f}")
    print(f"Selection score     : {_fmt_num(best_row.get('selection_score'))}")
    print(f"Target RMSE state   : {_fmt_num(best_row.get('target_rmse_state'))}")
    print(f"Target RMSE x       : {_fmt_num(best_row.get('target_rmse_x'))}")
    print(f"Spike reduction     : {_fmt_num(best_row.get('spike_reduction_percent'))}%")
    print(f"Control energy      : {_fmt_num(best_row.get('control_energy'))}")
    print(f"Settling time       : {_fmt_num(best_row.get('settling_time'))}")
    print(f"Stable              : {best_row.get('stable')}")
    print("=" * 72)
    print(f"[Control] Outputs saved in = {control_root}")

    return {
        "best_K": float(best_row.get("K", float("nan"))),
        "best_selection_score": float(best_row.get("selection_score", float("inf"))),
        "best_target_rmse_state": float(best_row.get("target_rmse_state", float("inf"))),
        "best_target_rmse_x": float(best_row.get("target_rmse_x", float("inf"))),
        "best_spike_reduction_percent": float(
            best_row.get("spike_reduction_percent", float("nan"))
        ),
        "best_control_energy": float(best_row.get("control_energy", float("inf"))),
        "best_settling_time": float(best_row.get("settling_time", float("nan"))),
        "best_stable": bool(best_row.get("stable", False)),
        "target_mode": str(control_target_mode),
        "target_raw_state": _json_safe(target_raw),
        "control_start_frac": float(control_start_frac),
        "control_start_idx": int(control_start_idx),
        "control_start_time": float(control_start_time),
        "output_dir": control_root,
        "best_run_dir": str(best_row.get("run_dir")),
    }