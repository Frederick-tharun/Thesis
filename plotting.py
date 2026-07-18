from __future__ import annotations

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    import config
except Exception:
    config = None


# -----------------------------------------------------------------------------
# Publication-quality plotting defaults
# -----------------------------------------------------------------------------
# This file is still only a plotting/evaluation utility. It does not change the
# ESN model, the controller formulas, or the optimization logic. The styling below
# only improves readability of the generated PNG figures.
_PLOT_DPI = 240
_COLOR_TRUE = "#111827"          # near black
_COLOR_UNCONTROLLED = "#2563eb"  # blue
_COLOR_CONTROLLED = "#dc2626"    # red
_COLOR_TARGET = "#059669"        # green
_COLOR_EVENT = "#7c3aed"         # purple
_COLOR_THRESHOLD = "#f59e0b"     # amber
_COLOR_GRID = "#d1d5db"


def _apply_publication_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": _PLOT_DPI,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#111827",
            "axes.edgecolor": "#9ca3af",
            "axes.linewidth": 0.8,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "grid.color": _COLOR_GRID,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "#d1d5db",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


_apply_publication_style()


def _style_axis(ax):
    """Apply consistent thesis/report style to one axis."""
    ax.grid(True, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(axis="both", labelsize=9, width=0.8, length=4)


def _clean_legend(ax, fontsize=8, loc="best", ncol=1):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc=loc, fontsize=fontsize, ncol=ncol, frameon=True, fancybox=True)


def _controller_label(controller_name=None, metrics=None, output_dir=None, rows=None):
    """
    Return a display label for the active controller.

    Backward compatible: existing callers do not have to pass controller_name.
    The function tries, in this order:
    1) explicit controller_name argument
    2) metrics dictionaries
    3) rows from K sweep
    4) output_dir path tokens such as 'finite_time' or 'pyragas'
    """
    candidates = []

    if controller_name:
        candidates.append(controller_name)

    if isinstance(metrics, dict):
        for key in ("controller", "controller_name", "control_method", "method"):
            if metrics.get(key):
                candidates.append(metrics.get(key))

    if rows:
        try:
            for row in rows:
                if isinstance(row, dict):
                    for key in ("controller", "controller_name", "control_method", "method"):
                        if row.get(key):
                            candidates.append(row.get(key))
                            raise StopIteration
        except StopIteration:
            pass

    if output_dir:
        lower_path = str(output_dir).replace("\\", "/").lower()
        for token in ("linear_feedback", "finite_time", "pyragas"):
            if token in lower_path:
                candidates.append(token)

    raw = str(candidates[0]).strip().lower() if candidates else "linear_feedback"
    raw = raw.replace("-", "_").replace(" ", "_")

    labels = {
        "linear": "Linear feedback control",
        "linear_feedback": "Linear feedback control",
        "feedback": "Linear feedback control",
        "finite": "Finite-time control",
        "finite_time": "Finite-time control",
        "finite_time_control": "Finite-time control",
        "pyragas": "Pyragas delayed-feedback control",
        "pyragas_control": "Pyragas delayed-feedback control",
        "tdfc": "Pyragas delayed-feedback control",
        "time_delay": "Pyragas delayed-feedback control",
        "time_delay_feedback": "Pyragas delayed-feedback control",
    }
    return labels.get(raw, raw.replace("_", " ").title())


def _format_optional_metric(name, value, fmt=".4f", suffix=""):
    value = _safe_float(value)
    if not np.isfinite(value):
        return None
    return f"{name} = {value:{fmt}}{suffix}"


def _control_effort_value(metrics):
    """Prefer the canonical mean-squared effort key with legacy fallback."""
    metrics = metrics or {}
    value = metrics.get("control_effort_mean_sq")
    return metrics.get("control_energy") if value is None else value


def _controller_test_boundary_idx(metrics, control_start_idx, n):
    """Return the held-out controller-test boundary in rollout coordinates."""
    metrics = metrics or {}
    value = metrics.get("controller_test_start")
    if value is None:
        value = metrics.get("controller_test_start_idx")
    if value is None:
        value = metrics.get("pyragas_evaluation_start_idx")
    boundary = _safe_float(value, control_start_idx)
    if not np.isfinite(boundary):
        boundary = control_start_idx
    return int(max(control_start_idx, min(boundary, n - 1)))


def _add_controller_segment_guides(ax, times, control_start_idx, metrics=None):
    """Shade controller validation separately from the held-out test segment."""
    boundary = _controller_test_boundary_idx(
        metrics, control_start_idx, len(times)
    )
    if boundary > control_start_idx:
        ax.axvspan(
            times[control_start_idx],
            times[boundary],
            color=_COLOR_EVENT,
            alpha=0.055,
            zorder=0,
            label="Controller validation (not used for held-out metrics)",
        )
        ax.axvline(
            times[boundary],
            color=_COLOR_TARGET,
            linestyle="-.",
            linewidth=1.25,
            label="Held-out controller-test start",
            zorder=1,
        )
        ax.axvspan(
            times[boundary],
            times[-1],
            color=_COLOR_TARGET,
            alpha=0.035,
            zorder=0,
        )
    else:
        ax.axvspan(
            times[control_start_idx],
            times[-1],
            color=_COLOR_EVENT,
            alpha=0.045,
            zorder=0,
        )
    return boundary


def _is_pyragas_context(metrics=None, output_dir=None, controller_name=None, rows=None):
    """Return True when the active plot belongs to Pyragas delayed feedback."""
    candidates = []

    if controller_name:
        candidates.append(controller_name)

    if isinstance(metrics, dict):
        for key in ("controller", "controller_name", "control_method", "method"):
            if metrics.get(key):
                candidates.append(metrics.get(key))

    if rows:
        for row in rows:
            if isinstance(row, dict):
                for key in ("controller", "controller_name", "control_method", "method"):
                    if row.get(key):
                        candidates.append(row.get(key))
                        break
                if candidates:
                    break

    if output_dir:
        candidates.append(str(output_dir))

    text = " ".join(str(c).lower().replace("-", "_") for c in candidates)
    return "pyragas" in text or "time_delay" in text or "tdfc" in text


def _format_count(value):
    value = _safe_float(value)
    if not np.isfinite(value):
        return "-"
    return str(int(round(value)))


def _metric_box_text(metrics):
    metrics = metrics or {}
    lines = []
    controller = str(metrics.get("controller", "")).strip().lower()

    if metrics.get("K") is not None:
        lines.append(f"K = {_safe_float(metrics.get('K')):.4g}")

    corrected_metrics = metrics.get("corrected_feedback_input_metrics", {})
    if not isinstance(corrected_metrics, dict):
        corrected_metrics = {}
    raw_metrics = metrics.get("raw_readout_metrics", {})
    if not isinstance(raw_metrics, dict):
        raw_metrics = {}

    corrected_rmse = metrics.get("corrected_feedback_input_target_rmse_state")
    if corrected_rmse is None:
        corrected_rmse = corrected_metrics.get("target_rmse_state")
    if corrected_rmse is None:
        corrected_rmse = metrics.get("target_rmse_state")

    raw_rmse = metrics.get("raw_readout_target_rmse_state")
    if raw_rmse is None:
        raw_rmse = raw_metrics.get("target_rmse_state")

    effort = _control_effort_value(metrics)

    if _is_pyragas_context(metrics=metrics):
        delay = metrics.get("pyragas_delay", metrics.get("delay_steps"))
        if delay is not None:
            lines.append(f"delay = {int(_safe_float(delay, 0))} steps")
        if metrics.get("pyragas_sign") is not None:
            lines.append(
                f"sign = {int(_safe_float(metrics.get('pyragas_sign'), 0)):+d}"
            )
        history_signal = metrics.get("pyragas_history_signal")
        if history_signal:
            lines.append(f"delay history = {history_signal}")

        quality_pass = metrics.get("pyragas_quality_pass")
        if quality_pass is not None:
            lines.append(f"quality = {'PASS' if bool(quality_pass) else 'FAIL'}")

        evaluated_peaks = metrics.get("pyragas_detected_peak_count")
        if evaluated_peaks is not None:
            lines.append(f"evaluated peaks = {_format_count(evaluated_peaks)}")

        rhythm_type = metrics.get("pyragas_rhythm_type")
        if rhythm_type:
            lines.append(f"rhythm = {rhythm_type}")
        cycle_count = metrics.get("pyragas_detected_cycle_count")
        if cycle_count is not None:
            lines.append(f"evaluated cycles = {_format_count(cycle_count)}")

        empirical_steps = _safe_float(
            metrics.get("pyragas_empirical_period_steps")
        )
        empirical_time = _safe_float(
            metrics.get("pyragas_empirical_period_time")
        )
        if np.isfinite(empirical_steps) and empirical_steps > 0:
            if np.isfinite(empirical_time):
                lines.append(
                    f"measured period = {int(round(empirical_steps))} steps "
                    f"({empirical_time:.3g} time units)"
                )
            else:
                lines.append(
                    f"measured period = {int(round(empirical_steps))} steps"
                )

        for item in (
            _format_optional_metric(
                "rhythm CV",
                metrics.get("pyragas_rhythm_interval_cv"),
                ".4f",
            ),
            _format_optional_metric(
                "amplitude ratio",
                metrics.get("pyragas_x_amplitude_ratio"),
                ".3f",
            ),
            _format_optional_metric(
                "cycle recurrence error",
                metrics.get("pyragas_empirical_recurrence_error_norm"),
                ".4f",
            ),
            _format_optional_metric(
                "cycle correlation",
                metrics.get("pyragas_empirical_recurrence_correlation"),
                ".3f",
            ),
            _format_optional_metric(
                "cycle-window coverage",
                metrics.get("pyragas_cycle_window_coverage"),
                ".2f",
            ),
            _format_optional_metric(
                "drift ratio", metrics.get("pyragas_drift_ratio"), ".3f"
            ),
            _format_optional_metric(
                "empirical closure",
                metrics.get("pyragas_empirical_tail_closure_error_norm"),
                ".3f",
            ),
            _format_optional_metric("mean-squared effort", effort, ".4g"),
        ):
            if item is not None:
                lines.append(item)

        return "\n".join(lines) if lines else "No Pyragas metrics available"

    if controller == "finite_time" and metrics.get("finite_s") is not None:
        lines.append(f"finite_s = {_safe_float(metrics.get('finite_s')):.3f}")
    if metrics.get("delay_steps") is not None:
        lines.append(
            f"delay = {int(_safe_float(metrics.get('delay_steps'), 0))} steps"
        )

    for item in (
        _format_optional_metric(
            "Corrected-feedback target RMSE", corrected_rmse, ".4f"
        ),
        _format_optional_metric(
            "Raw-readout target RMSE", raw_rmse, ".4f"
        ),
        _format_optional_metric(
            "Spike reduction (corrected feedback)",
            metrics.get("spike_reduction_percent"),
            ".2f",
            "%",
        ),
        _format_optional_metric("Mean-squared control effort", effort, ".4g"),
        _format_optional_metric(
            "Controller-test time to tolerance",
            metrics.get(
                "controller_test_time_to_tolerance",
                metrics.get(
                    "evaluation_time_to_tolerance", metrics.get("settling_time")
                ),
            ),
            ".4f",
        ),
    ):
        if item is not None:
            lines.append(item)

    return "\n".join(lines) if lines else "No metrics available"

def _output_dir():
    if config is None:
        return "outputs"
    return getattr(config, "OUTPUT_DIR", "outputs")


def _savefig(filename):
    os.makedirs(_output_dir(), exist_ok=True)
    path = os.path.join(_output_dir(), filename)
    plt.savefig(path, dpi=_PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved -> {path}")


def _as_1d(x):
    return np.asarray(x, dtype=float).reshape(-1)


def _as_2d(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _spike_threshold():
    if config is None:
        return 2.0

    return float(
        getattr(
            config,
            "SPIKE_THRESHOLD",
            getattr(config, "HR_SPIKE_THRESHOLD", 2.0),
        )
    )


def _spike_match_tolerance_steps():
    if config is None:
        return 20

    return int(
        getattr(
            config,
            "SPIKE_MATCH_TOLERANCE_STEPS",
            getattr(config, "SPIKE_TOLERANCE_STEPS", 20),
        )
    )


def _rmse(y_pred, y_true):
    y_pred = _as_1d(y_pred)
    y_true = _as_1d(y_true)

    n = min(len(y_pred), len(y_true))
    if n == 0:
        return np.nan

    return float(np.sqrt(np.mean((y_pred[:n] - y_true[:n]) ** 2)))


def _nrmse(y_pred, y_true):
    y_pred = _as_1d(y_pred)
    y_true = _as_1d(y_true)

    n = min(len(y_pred), len(y_true))
    if n == 0:
        return np.nan

    denom = float(np.std(y_true[:n]))

    if denom < 1e-12:
        denom = float(np.max(y_true[:n]) - np.min(y_true[:n]))

    if denom < 1e-12:
        denom = 1.0

    return _rmse(y_pred[:n], y_true[:n]) / denom


def _r2(y_pred, y_true):
    y_pred = _as_1d(y_pred)
    y_true = _as_1d(y_true)

    n = min(len(y_pred), len(y_true))
    if n == 0:
        return np.nan

    y_pred = y_pred[:n]
    y_true = y_true[:n]

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot < 1e-12:
        return np.nan

    return float(1.0 - ss_res / ss_tot)


def _corr(y_pred, y_true):
    y_pred = _as_1d(y_pred)
    y_true = _as_1d(y_true)

    n = min(len(y_pred), len(y_true))
    if n < 2:
        return np.nan

    y_pred = y_pred[:n]
    y_true = y_true[:n]

    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        return np.nan

    return float(np.corrcoef(y_pred, y_true)[0, 1])


def _find_peaks(signal, threshold=None, min_distance=5):
    signal = _as_1d(signal)

    if threshold is None:
        threshold = _spike_threshold()

    peaks = []

    last_peak = -10**9

    for i in range(1, len(signal) - 1):
        if signal[i] < threshold:
            continue

        if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
            if i - last_peak >= min_distance:
                peaks.append(i)
                last_peak = i

    return np.asarray(peaks, dtype=int)


def _match_spikes(true_peaks, pred_peaks, tolerance_steps):
    true_peaks = list(map(int, true_peaks))
    pred_peaks = list(map(int, pred_peaks))

    matched_true = set()
    matched_pred = set()

    for pi, p in enumerate(pred_peaks):
        best_ti = None
        best_dist = None

        for ti, t in enumerate(true_peaks):
            if ti in matched_true:
                continue

            dist = abs(p - t)

            if dist <= tolerance_steps and (best_dist is None or dist < best_dist):
                best_ti = ti
                best_dist = dist

        if best_ti is not None:
            matched_pred.add(pi)
            matched_true.add(best_ti)

    tp_true_idx = [true_peaks[i] for i in sorted(matched_true)]
    fp_pred_idx = [pred_peaks[i] for i in range(len(pred_peaks)) if i not in matched_pred]
    fn_true_idx = [true_peaks[i] for i in range(len(true_peaks)) if i not in matched_true]

    tp = len(tp_true_idx)
    fp = len(fp_pred_idx)
    fn = len(fn_true_idx)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "tp_true_idx": np.asarray(tp_true_idx, dtype=int),
        "fp_pred_idx": np.asarray(fp_pred_idx, dtype=int),
        "fn_true_idx": np.asarray(fn_true_idx, dtype=int),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _metrics_text(y_pred, y_true):
    return (
        f"RMSE   : {_rmse(y_pred, y_true):.4f}\n"
        f"NRMSE  : {_nrmse(y_pred, y_true):.4f}\n"
        f"R²     : {_r2(y_pred, y_true):.4f}\n"
        f"Pearson: {_corr(y_pred, y_true):.4f}"
    )


def plot_results(split, y_pred, tag="ESN"):
    full_time = _as_1d(split["full_time"])
    full_signal = _as_1d(split["full_signal"])

    train_time = _as_1d(split.get("train_time", []))
    test_time = _as_1d(split.get("test_time", []))

    y_true = _as_1d(split["y_test"])
    t_pred = _as_1d(split["t_test_y"])
    y_pred = _as_1d(y_pred)

    n = min(len(t_pred), len(y_true), len(y_pred))
    t_pred = t_pred[:n]
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    threshold = _spike_threshold()
    min_distance = int(getattr(config, "SPIKE_MIN_DISTANCE", 5)) if config is not None else 5

    full_peaks = _find_peaks(full_signal, threshold=threshold, min_distance=min_distance)
    true_peaks = _find_peaks(y_true, threshold=threshold, min_distance=min_distance)
    pred_peaks = _find_peaks(y_pred, threshold=threshold, min_distance=min_distance)

    matches = _match_spikes(
        true_peaks=true_peaks,
        pred_peaks=pred_peaks,
        tolerance_steps=_spike_match_tolerance_steps(),
    )

    precision = matches["precision"]
    recall = matches["recall"]
    f1 = matches["f1"]

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=False)

    ax = axes[0]
    ax.plot(full_time, full_signal, color="black", linewidth=1.0, label="True x signal")

    if len(train_time) > 0:
        ax.axvspan(
            train_time[0],
            train_time[-1],
            alpha=0.12,
            label="Training region",
        )

    if len(test_time) > 0:
        ax.axvspan(
            test_time[0],
            test_time[-1],
            alpha=0.10,
            label="Test region",
        )

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        label="Spike threshold",
    )

    if len(full_peaks) > 0:
        ax.scatter(
            full_time[full_peaks],
            full_signal[full_peaks],
            s=12,
            label="Detected spike peaks",
            zorder=5,
        )

    ax.set_title("1) Full recording with train/test split", fontsize=12, fontweight="bold")
    ax.set_ylabel("x state")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.plot(t_pred, y_true, color="black", linewidth=1.3, label="True x")
    ax.plot(t_pred, y_pred, linestyle="--", linewidth=1.3, label="Recursive ESN prediction")
    ax.axhline(threshold, linestyle=":", linewidth=1.0, alpha=0.75, label="Spike threshold")

    if len(true_peaks) > 0:
        ax.scatter(
            t_pred[true_peaks],
            y_true[true_peaks],
            s=16,
            label="True spike peaks",
            zorder=6,
        )

    txt = (
        _metrics_text(y_pred, y_true)
        + f"\nPeak F1 : {f1:.4f}"
        + f"\nPrecision: {precision:.4f}"
        + f"\nRecall   : {recall:.4f}"
    )

    ax.text(
        0.015,
        0.96,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.90),
    )

    ax.set_title("2) Recursive prediction on held-out test segment", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("x state")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"{tag} forecast for Hindmarsh-Rose state x", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig("results_full_zoom.png")

    plot_zoom_comparison(t_pred, y_true, y_pred, tag=tag)
    plot_spike_event_comparison(t_pred, y_true, y_pred, tag=tag)


def plot_zoom_comparison(t, y_true, y_pred, tag="ESN"):
    t = _as_1d(t)
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)

    n = min(len(t), len(y_true), len(y_pred))
    if n == 0:
        return

    t = t[:n]
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    max_points = min(n, max(300, int(0.25 * n)))

    t_zoom = t[:max_points]
    true_zoom = y_true[:max_points]
    pred_zoom = y_pred[:max_points]

    plt.figure(figsize=(15, 5))
    plt.plot(t_zoom, true_zoom, color="black", linewidth=1.5, label="True x")
    plt.plot(t_zoom, pred_zoom, linestyle="--", linewidth=1.5, label="Recursive ESN prediction")
    plt.axhline(_spike_threshold(), linestyle=":", linewidth=1.0, alpha=0.75, label="Spike threshold")

    plt.title(f"{tag} recursive prediction zoom", fontsize=14, fontweight="bold")
    plt.xlabel("Time")
    plt.ylabel("x state")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    _savefig("results_zoom_comparison.png")


def plot_spike_event_comparison(t, y_true, y_pred, tag="ESN"):
    t = _as_1d(t)
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)

    n = min(len(t), len(y_true), len(y_pred))
    if n == 0:
        return

    t = t[:n]
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    threshold = _spike_threshold()
    min_distance = int(getattr(config, "SPIKE_MIN_DISTANCE", 5)) if config is not None else 5

    true_peaks = _find_peaks(y_true, threshold=threshold, min_distance=min_distance)
    pred_peaks = _find_peaks(y_pred, threshold=threshold, min_distance=min_distance)

    matches = _match_spikes(
        true_peaks=true_peaks,
        pred_peaks=pred_peaks,
        tolerance_steps=_spike_match_tolerance_steps(),
    )

    tp = matches["tp_true_idx"]
    fp = matches["fp_pred_idx"]
    fn = matches["fn_true_idx"]

    precision = matches["precision"]
    recall = matches["recall"]
    f1 = matches["f1"]

    plt.figure(figsize=(15, 4.8))

    if len(tp) > 0:
        plt.vlines(t[tp], 2.65, 3.15, linewidth=2.0, label="True positive")

    if len(fp) > 0:
        plt.vlines(t[fp], 1.65, 2.15, linewidth=2.0, label="False positive")

    if len(fn) > 0:
        plt.vlines(t[fn], 0.65, 1.15, linewidth=2.0, label="False negative")

    if len(tp) == 0 and len(fp) == 0 and len(fn) == 0:
        plt.text(
            0.5,
            0.5,
            "No spike events detected in this segment",
            transform=plt.gca().transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )

    plt.yticks([0.9, 1.9, 2.9], ["FN", "FP", "TP"])
    plt.xlabel("Time")
    plt.ylabel("Event type")
    plt.title(
        f"Spike event comparison | Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.25)

    if len(tp) > 0 or len(fp) > 0 or len(fn) > 0:
        plt.legend(loc="best")

    plt.tight_layout()
    _savefig("spike_event_comparison.png")


def plot_all_states(t, truth, pred, tag="ESN"):
    t = _as_1d(t)
    truth = _as_2d(truth)
    pred = _as_2d(pred)

    n = min(len(t), len(truth), len(pred))
    if n == 0:
        return

    t = t[:n]
    truth = truth[:n]
    pred = pred[:n]

    names = [
        "x: membrane voltage / spike variable",
        "y: recovery variable",
        "z: slow adaptation variable",
    ]

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    for i, ax in enumerate(axes):
        if i >= truth.shape[1] or i >= pred.shape[1]:
            ax.axis("off")
            continue

        ax.plot(t, truth[:, i], color="black", linewidth=1.2, label=f"True {['x', 'y', 'z'][i]}")
        ax.plot(t, pred[:, i], linestyle="--", linewidth=1.2, label=f"Predicted {['x', 'y', 'z'][i]}")

        ax.set_ylabel(["x", "y", "z"][i])
        ax.set_title(names[i], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"{tag} - full-state recursive forecast", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig("results_all_states.png")


def plot_optimizer_convergence(rows, filename="optimizer_convergence.png"):
    if not rows:
        print("[Plot] Optimizer convergence skipped: empty rows")
        return

    grouped = {}

    for i, row in enumerate(rows):
        opt = str(row.get("optimizer", "unknown")).lower()

        score = _safe_float(
            row.get(
                "best_score",
                row.get(
                    "score",
                    row.get("validation_score", np.nan),
                ),
            )
        )

        call = row.get(
            "iteration",
            row.get(
                "iter",
                row.get(
                    "call",
                    row.get("n_call", i + 1),
                ),
            ),
        )

        call = int(_safe_float(call, i + 1))

        if np.isfinite(score):
            grouped.setdefault(opt, []).append((call, score))

    if not grouped:
        print("[Plot] Optimizer convergence skipped: no valid rows")
        return

    plt.figure(figsize=(14, 5.5))

    plotted = False

    for opt, values in grouped.items():
        values = sorted(values, key=lambda x: x[0])
        calls = np.asarray([v[0] for v in values], dtype=int)
        scores = np.asarray([v[1] for v in values], dtype=float)

        best_so_far = np.minimum.accumulate(scores)

        if len(calls) > 0:
            plt.plot(calls, best_so_far, marker="o", markersize=3, linewidth=1.5, label=opt)
            plotted = True

    plt.title("Optimizer comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Number of optimizer calls")
    plt.ylabel("Best objective value so far")
    plt.grid(True, alpha=0.25)

    if plotted:
        positive_values = []
        for values in grouped.values():
            positive_values.extend([v[1] for v in values if v[1] > 0 and np.isfinite(v[1])])

        if positive_values:
            if max(positive_values) / max(min(positive_values), 1e-12) > 100:
                plt.yscale("log")

        plt.legend(loc="best")

    plt.tight_layout()
    _savefig(filename)


def _normalize_columns(mat):
    mat = np.asarray(mat, dtype=float).copy()
    out = np.full_like(mat, np.nan, dtype=float)

    for j in range(mat.shape[1]):
        col = mat[:, j]
        mask = np.isfinite(col)

        if not np.any(mask):
            continue

        cmin = np.min(col[mask])
        cmax = np.max(col[mask])

        if abs(cmax - cmin) < 1e-12:
            out[mask, j] = 0.5
        else:
            out[mask, j] = (col[mask] - cmin) / (cmax - cmin)

    return out


def plot_optimizer_heatmap(rows, filename="optimizer_heatmap.png"):
    if not rows:
        print("[Plot] Heatmap skipped: empty rows")
        return

    preferred_order = ["gp", "dummy", "forest", "gbrt"]

    best_by_optimizer = {}

    for row in rows:
        optimizer = str(row.get("optimizer", "unknown")).strip().lower()

        score = _safe_float(
            row.get(
                "best_score",
                row.get(
                    "score",
                    row.get("validation_score", np.nan),
                ),
            )
        )

        if optimizer not in best_by_optimizer:
            best_by_optimizer[optimizer] = dict(row)
            best_by_optimizer[optimizer]["_cmp_score"] = score
        else:
            old_score = _safe_float(best_by_optimizer[optimizer].get("_cmp_score", np.inf))
            if score < old_score:
                best_by_optimizer[optimizer] = dict(row)
                best_by_optimizer[optimizer]["_cmp_score"] = score

    if not best_by_optimizer:
        print("[Plot] Heatmap skipped: no valid optimizer points")
        return

    optimizers = [o for o in preferred_order if o in best_by_optimizer]
    optimizers += sorted([o for o in best_by_optimizer if o not in preferred_order])

    best_rows = [best_by_optimizer[o] for o in optimizers]

    metrics = [
        ("best_score", "Best score"),
        ("validation_nrmse_x", "Val NRMSE x"),
        ("N_res", "N_res"),
        ("p", "density p"),
        ("spectral_radius", "rho"),
        ("leaky_coefficient", "leak"),
        ("input_scaling", "scale"),
        ("regularization", "ridge"),
        ("washout", "washout"),
    ]

    actual = []

    for row in best_rows:
        values = []

        for key, _ in metrics:
            if key == "best_score":
                val = row.get("best_score", row.get("score", row.get("validation_score", np.nan)))
            elif key == "validation_nrmse_x":
                val = row.get("validation_nrmse_x", row.get("x_nrmse", np.nan))
            else:
                val = row.get(key, np.nan)

            values.append(_safe_float(val))

        actual.append(values)

    actual = np.asarray(actual, dtype=float)

    if actual.size == 0 or not np.any(np.isfinite(actual)):
        print("[Plot] Heatmap skipped: no valid numeric values")
        return

    normalized = _normalize_columns(actual)

    fig, ax = plt.subplots(figsize=(15, max(4, 1.2 + 0.8 * len(optimizers))))

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")

    im = ax.imshow(normalized, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics], rotation=30, ha="right", fontsize=10)

    ax.set_yticks(np.arange(len(optimizers)))
    ax.set_yticklabels([opt.upper() for opt in optimizers], fontsize=11)

    for i in range(len(optimizers)):
        for j in range(len(metrics)):
            val = actual[i, j]

            if np.isfinite(val):
                if abs(val) >= 1000 or (abs(val) > 0 and abs(val) < 1e-3):
                    txt = f"{val:.1e}"
                elif abs(val) >= 10:
                    txt = f"{val:.2f}"
                else:
                    txt = f"{val:.4f}"
            else:
                txt = "-"

            color = "white" if np.isfinite(normalized[i, j]) and normalized[i, j] > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)

    ax.set_title("Optimizer heatmap: best result per optimizer", fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Column-wise normalized value")

    plt.tight_layout()
    _savefig(filename)


def _parameter_label_map():
    return {
        "N_res": r"$N_{res}$",
        "p": r"$p$",
        "spectral_radius": r"$\rho$",
        "leaky_coefficient": r"$\alpha$",
        "input_scaling": r"$\epsilon$",
        "regularization": r"$\lambda$",
        "washout": "washout",
    }


def _score_from_row(row):
    """Return the candidate objective value. Smaller is better."""
    return _safe_float(
        row.get(
            "score",
            row.get("validation_score", row.get("best_score", np.nan)),
        )
    )


def _candidate_matrix(rows, optimizer, params):
    """Collect valid BO samples for one optimizer."""
    X, y = [], []
    optimizer = str(optimizer).strip().lower()

    for row in rows:
        if optimizer and str(row.get("optimizer", "")).strip().lower() != optimizer:
            continue

        score = _score_from_row(row)

        # Failed/exploded candidates are normally stored near 1e6.
        # Keep them out of the colour scale because they make every useful region flat.
        if not np.isfinite(score) or score >= 1e5:
            continue

        values = []
        ok = True
        for p in params:
            v = _safe_float(row.get(p, np.nan))
            if not np.isfinite(v):
                ok = False
                break
            values.append(v)

        if ok:
            X.append(values)
            y.append(score)

    if len(y) == 0:
        return np.empty((0, len(params))), np.asarray([], dtype=float)

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _choose_optimizer_for_landscape(rows, requested_optimizer, params):
    requested_optimizer = str(requested_optimizer or "").strip().lower()
    if requested_optimizer:
        X, y = _candidate_matrix(rows, requested_optimizer, params)
        if len(y) >= max(6, len(params) + 2):
            return requested_optimizer, X, y

    # Fallback: choose the optimizer with the largest number of valid BO points.
    names = sorted({str(r.get("optimizer", "")).strip().lower() for r in rows if r.get("optimizer")})
    best = None
    for name in names:
        X, y = _candidate_matrix(rows, name, params)
        if best is None or len(y) > len(best[2]):
            best = (name, X, y)

    if best is None:
        return requested_optimizer, np.empty((0, len(params))), np.asarray([], dtype=float)
    return best


def _normalize_objective(y):
    """0 = best candidate, 1 = worst visible candidate."""
    y = np.asarray(y, dtype=float)
    y_best = float(np.min(y))
    y_worst = float(np.percentile(y, 95))

    if abs(y_worst - y_best) < 1e-15:
        return np.zeros_like(y), y_best, y_worst

    yn = (y - y_best) / (y_worst - y_best)
    return np.clip(yn, 0.0, 1.0), y_best, y_worst


def _padded_limits(values, pad_frac=0.04):
    values = np.asarray(values, dtype=float)
    lo, hi = float(np.min(values)), float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if abs(hi - lo) < 1e-12:
        pad = 0.5 if abs(lo) < 1e-12 else 0.05 * abs(lo)
        return lo - pad, hi + pad
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def _smooth_curve(xs, ys, n_grid=160):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 2:
        return xs, ys

    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    # Average duplicate x values.
    unique_x = []
    unique_y = []
    for x in np.unique(xs):
        m = xs == x
        unique_x.append(float(x))
        unique_y.append(float(np.mean(ys[m])))
    xs = np.asarray(unique_x, dtype=float)
    ys = np.asarray(unique_y, dtype=float)

    if len(xs) < 3:
        return xs, ys

    grid = np.linspace(xs.min(), xs.max(), n_grid)
    try:
        from scipy.interpolate import PchipInterpolator
        curve = PchipInterpolator(xs, ys, extrapolate=True)(grid)
    except Exception:
        curve = np.interp(grid, xs, ys)

    # Gentle moving average so the diagonal looks thesis/paper friendly.
    if len(curve) >= 9:
        window = max(5, int(len(curve) * 0.04))
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window) / window
        pad = window // 2
        curve = np.convolve(np.pad(curve, pad, mode="edge"), kernel, mode="valid")

    return grid, np.clip(curve, 0.0, 1.0)


def _binned_lower_curve(x, y_norm, n_bins=15):
    x = np.asarray(x, dtype=float)
    y_norm = np.asarray(y_norm, dtype=float)
    if len(x) == 0:
        return x, y_norm

    if len(np.unique(x)) < 3:
        return _smooth_curve(x, y_norm)

    n_bins = int(max(5, min(n_bins, len(x))))
    edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    xs, ys = [], []

    for left, right in zip(edges[:-1], edges[1:]):
        if right == edges[-1]:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)
        if np.any(mask):
            xs.append(float(np.mean(x[mask])))
            # Lower envelope: BO is interested in the best value found in that region.
            ys.append(float(np.min(y_norm[mask])))

    return _smooth_curve(np.asarray(xs), np.asarray(ys))


def _interpolated_surface(x, y, z, grid_size=180):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    xlim = _padded_limits(x, pad_frac=0.0)
    ylim = _padded_limits(y, pad_frac=0.0)
    gx = np.linspace(xlim[0], xlim[1], grid_size)
    gy = np.linspace(ylim[0], ylim[1], grid_size)
    GX, GY = np.meshgrid(gx, gy)

    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2 or len(z) < 4:
        return GX, GY, None

    try:
        from scipy.interpolate import griddata
        points = np.column_stack([x, y])
        Z = griddata(points, z, (GX, GY), method="linear")
        # Fill convex-hull holes with nearest-neighbour values so the heatmap is a full rectangle.
        if np.any(~np.isfinite(Z)):
            Z_nearest = griddata(points, z, (GX, GY), method="nearest")
            Z = np.where(np.isfinite(Z), Z, Z_nearest)
        return GX, GY, np.clip(Z, 0.0, 1.0)
    except Exception:
        return GX, GY, None


def plot_bo_objective_landscape(
    rows,
    optimizer="forest",
    params=("input_scaling", "spectral_radius", "leaky_coefficient"),
    labels=None,
    filename=None,
):
    """
    Paper-style BO objective landscape.

    Layout is intentionally similar to common Bayesian-optimization papers:
    - diagonal: 1D partial-dependence-like objective curve
    - lower triangle: 2D objective landscape for each hyperparameter pair
    - black dots: BO samples tested by the optimizer
    - red star / red dashed line: best sampled hyperparameter value

    Objective value is normalized so 0 is best and 1 is worst in the visible range.
    """
    if not rows:
        print("[Plot] BO objective landscape skipped: empty rows")
        return

    params = tuple(params)
    if labels is None:
        labels = _parameter_label_map()

    optimizer, X, y = _choose_optimizer_for_landscape(rows, optimizer, params)
    min_needed = max(6, len(params) + 2)
    if len(y) < min_needed:
        print(
            f"[Plot] BO objective landscape skipped: not enough valid points "
            f"for optimizer='{optimizer}'. Need at least {min_needed}, found {len(y)}."
        )
        return

    y_norm, y_best, _ = _normalize_objective(y)
    best_idx = int(np.argmin(y))
    d = len(params)

    cmap = plt.cm.viridis
    fig, axes = plt.subplots(
        d,
        d,
        figsize=(2.25 * d + 1.35, 2.05 * d + 0.7),
        squeeze=False,
    )

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]

            if i < j:
                ax.axis("off")
                continue

            xj = X[:, j]
            yi = X[:, i]

            if i == j:
                curve_x, curve_y = _binned_lower_curve(yi, y_norm, n_bins=15)
                ax.plot(curve_x, curve_y, color="#1f77b4", linewidth=1.35)
                ax.scatter(yi, y_norm, s=9, color="#1f77b4", alpha=0.30)
                ax.axvline(X[best_idx, i], color="red", linestyle="--", linewidth=1.0, alpha=0.80)
                ax.scatter(
                    [X[best_idx, i]],
                    [y_norm[best_idx]],
                    marker="*",
                    s=70,
                    color="red",
                    edgecolor="red",
                    zorder=5,
                )
                ax.set_ylim(-0.04, 1.04)
                ax.set_title(labels.get(params[i], params[i]), fontsize=11, pad=5)
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.set_ylabel("Partial dep.\nobj. func.", fontsize=8, labelpad=6)
                ax.set_xlim(*_padded_limits(yi))
            else:
                GX, GY, Z = _interpolated_surface(xj, yi, y_norm)
                if Z is not None:
                    ax.imshow(
                        Z,
                        origin="lower",
                        extent=[GX.min(), GX.max(), GY.min(), GY.max()],
                        aspect="auto",
                        cmap=cmap,
                        vmin=0,
                        vmax=1,
                        interpolation="bilinear",
                    )
                    ax.scatter(xj, yi, s=8, color="black", edgecolor="none", alpha=0.90, zorder=4)
                else:
                    ax.scatter(xj, yi, c=y_norm, cmap=cmap, vmin=0, vmax=1, s=18, edgecolor="black", linewidth=0.25)

                ax.scatter(
                    [X[best_idx, j]],
                    [X[best_idx, i]],
                    marker="*",
                    s=80,
                    color="red",
                    edgecolor="red",
                    zorder=6,
                )
                ax.set_xlim(*_padded_limits(xj))
                ax.set_ylim(*_padded_limits(yi))

            # Only show the outer labels, like a compact paper figure.
            if i == d - 1:
                ax.set_xlabel(labels.get(params[j], params[j]), fontsize=10)
            else:
                ax.set_xticklabels([])

            if j == 0 and i != j:
                ax.set_ylabel(labels.get(params[i], params[i]), fontsize=10)
            elif i != j:
                ax.set_yticklabels([])

            ax.tick_params(axis="both", labelsize=8, direction="out", length=3)
            ax.grid(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cax = fig.add_axes([0.885, 0.17, 0.026, 0.68])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Value of obj. func.", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    best_text = ", ".join(
        f"{labels.get(p, p)}={X[best_idx, k]:.4g}" for k, p in enumerate(params)
    )
    fig.suptitle(
        f"BO objective landscape ({optimizer})\nBest score={y_best:.3e}; {best_text}",
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )

    fig.subplots_adjust(left=0.10, right=0.84, bottom=0.10, top=0.88, wspace=0.18, hspace=0.12)

    if filename is None:
        filename = f"bo_objective_landscape_{optimizer}.png"

    _savefig(filename)


def _savefig_to_path(path, fig=None, dpi=_PLOT_DPI):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved -> {path}")


def plot_controlled_vs_uncontrolled_x(
    times,
    truth,
    uncontrolled,
    controlled,
    target_state,
    control_start_idx,
    metrics,
    output_dir,
    controller_name=None,
):
    """Plot true, uncontrolled, and corrected-feedback x trajectories."""
    times = _as_1d(times)
    truth = _as_2d(truth)
    uncontrolled = _as_2d(uncontrolled)
    controlled = _as_2d(controlled)
    target_state = _as_1d(target_state)

    n = min(len(times), len(truth), len(uncontrolled), len(controlled))
    if n == 0:
        return
    times = times[:n]
    truth = truth[:n]
    uncontrolled = uncontrolled[:n]
    controlled = controlled[:n]
    control_start_idx = int(max(0, min(control_start_idx, n - 1)))

    label = _controller_label(controller_name, metrics=metrics, output_dir=output_dir)

    fig, ax = plt.subplots(figsize=(16, 6.2))
    ax.plot(times, truth[:, 0], color=_COLOR_TRUE, linewidth=1.6, label="True x", zorder=4)
    ax.plot(
        times,
        uncontrolled[:, 0],
        color=_COLOR_UNCONTROLLED,
        linewidth=1.25,
        alpha=0.90,
        label="Uncontrolled ESN x",
        zorder=2,
    )
    ax.plot(
        times,
        controlled[:, 0],
        color=_COLOR_CONTROLLED,
        linestyle="--",
        linewidth=1.65,
        label="Corrected feedback input x (controlled trajectory)",
        zorder=5,
    )
    is_pyragas = _is_pyragas_context(
        metrics=metrics,
        output_dir=output_dir,
        controller_name=controller_name,
    )
    if not is_pyragas:
        ax.axhline(
            target_state[0],
            color=_COLOR_TARGET,
            linestyle=":",
            linewidth=1.7,
            label="Empirical quiet-state reference x",
            zorder=1,
        )
    ax.axvline(
        times[control_start_idx],
        color=_COLOR_EVENT,
        linestyle="--",
        linewidth=1.35,
        label="Control start",
        zorder=1,
    )
    _add_controller_segment_guides(
        ax, times, control_start_idx, metrics=metrics
    )

    txt = _metric_box_text(metrics)
    ax.text(
        0.012,
        0.975,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#d1d5db", alpha=0.94),
    )

    comparison_title = (
        f"{label}: x-state comparison"
        if is_pyragas
        else (
            f"{label}: regulation toward an empirical quiet-state "
            "reference"
        )
    )
    ax.set_title(
        comparison_title, fontsize=15, fontweight="bold", pad=12
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Hindmarsh-Rose x state")
    _style_axis(ax)
    _clean_legend(ax, fontsize=9, loc="upper right", ncol=2)
    fig.tight_layout()
    _savefig_to_path(os.path.join(output_dir, "controlled_vs_uncontrolled_x.png"), fig=fig)
    # Easy-to-find duplicate name for Pyragas outputs.
    if _is_pyragas_context(metrics=metrics, output_dir=output_dir, controller_name=controller_name):
        # Recreate the same figure would be wasteful; this line intentionally keeps
        # the standard name as the main plot. The Pyragas-specific figures below
        # use explicit pyragas_* filenames.
        pass



def plot_raw_readout_vs_corrected_feedback_input_x(
    times,
    raw_readout,
    corrected_feedback_input,
    control_start_idx,
    output_dir,
    controller_name=None,
    metrics=None,
):
    """Plot the closed-loop raw ESN readout against its corrected feedback input."""
    times = _as_1d(times)
    raw_readout = _as_2d(raw_readout)
    corrected_feedback_input = _as_2d(corrected_feedback_input)

    n = min(len(times), len(raw_readout), len(corrected_feedback_input))
    if n == 0:
        return

    times = times[:n]
    raw_readout = raw_readout[:n]
    corrected_feedback_input = corrected_feedback_input[:n]
    control_start_idx = int(max(0, min(control_start_idx, n - 1)))
    label = _controller_label(
        controller_name,
        metrics=metrics,
        output_dir=output_dir,
    )

    fig, ax = plt.subplots(figsize=(16, 5.8))
    ax.plot(
        times,
        raw_readout[:, 0],
        color=_COLOR_TRUE,
        linewidth=1.45,
        label="Raw ESN readout x (closed loop)",
        zorder=4,
    )
    ax.plot(
        times,
        corrected_feedback_input[:, 0],
        color=_COLOR_CONTROLLED,
        linestyle="--",
        linewidth=1.55,
        label="Corrected feedback input x",
        zorder=5,
    )
    ax.axvline(
        times[control_start_idx],
        color=_COLOR_EVENT,
        linestyle="--",
        linewidth=1.3,
        label="Control start",
        zorder=2,
    )
    _add_controller_segment_guides(
        ax, times, control_start_idx, metrics=metrics
    )

    ax.set_title(
        f"{label}: raw ESN readout vs corrected feedback input",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Hindmarsh-Rose x state")
    _style_axis(ax)
    _clean_legend(ax, fontsize=9, loc="upper right")
    fig.tight_layout()
    _savefig_to_path(
        os.path.join(
            output_dir,
            "raw_readout_vs_corrected_feedback_input_x.png",
        ),
        fig=fig,
    )


def _limit_trajectory_points(arrays, max_points=3500):
    """Downsample long trajectories while preserving the overall orbit shape."""
    lengths = [len(a) for a in arrays]
    n = min(lengths) if lengths else 0
    if n <= 0:
        return arrays
    if n <= max_points:
        return [a[:n] for a in arrays]
    idx = np.linspace(0, n - 1, int(max_points)).astype(int)
    return [a[:n][idx] for a in arrays]


def plot_pyragas_phase_trajectory_3d(
    times,
    truth,
    uncontrolled,
    controlled,
    control_start_idx,
    output_dir,
    controller_name=None,
    metrics=None,
):
    """
    Paper-style 3D phase-space view for Pyragas control.

    Saved as:
        pyragas_phase_trajectory_3d.png

    This figure is meant to show whether the controlled HR trajectory forms a
    repeated orbit in (x, y, z), which is the key visual evidence for Pyragas
    delayed-feedback regularization.
    """
    if not _is_pyragas_context(output_dir=output_dir, controller_name=controller_name):
        return

    times = _as_1d(times)
    truth = _as_2d(truth)
    uncontrolled = _as_2d(uncontrolled)
    controlled = _as_2d(controlled)

    n = min(len(times), len(truth), len(uncontrolled), len(controlled))
    if n == 0 or min(truth.shape[1], uncontrolled.shape[1], controlled.shape[1]) < 3:
        return

    times = times[:n]
    truth = truth[:n]
    uncontrolled = uncontrolled[:n]
    controlled = controlled[:n]
    start = int(max(0, min(control_start_idx, n - 1)))
    evaluation_start = int(
        max(
            start,
            min(
                int(_safe_float((metrics or {}).get("pyragas_evaluation_start_idx"), start)),
                n - 1,
            ),
        )
    )

    truth_post = truth[evaluation_start:, :3]
    unctrl_post = uncontrolled[evaluation_start:, :3]
    ctrl_transient = controlled[start:evaluation_start, :3]
    ctrl_post_full = controlled[evaluation_start:, :3]

    empirical_period = int(
        max(
            0,
            round(
                _safe_float(
                    (metrics or {}).get("pyragas_empirical_period_steps"),
                    0,
                )
            ),
        )
    )
    previous_cycle = None
    final_cycle = None
    if empirical_period > 0 and len(ctrl_post_full) >= 2 * empirical_period:
        previous_cycle = ctrl_post_full[-2 * empirical_period:-empirical_period]
        final_cycle = ctrl_post_full[-empirical_period:]
        previous_cycle, final_cycle = _limit_trajectory_points(
            [previous_cycle, final_cycle],
            max_points=2500,
        )

    truth_post, unctrl_post = _limit_trajectory_points(
        [truth_post, unctrl_post],
        max_points=3500,
    )
    ctrl_post = _limit_trajectory_points([ctrl_post_full], max_points=3500)[0]
    if len(ctrl_transient):
        ctrl_transient = _limit_trajectory_points([ctrl_transient], max_points=1200)[0]

    fig = plt.figure(figsize=(15.5, 6.6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # Left: reference chaotic/test trajectory and uncontrolled ESN.
    ax1.plot(truth_post[:, 0], truth_post[:, 1], truth_post[:, 2], color=_COLOR_TRUE, linewidth=1.0, alpha=0.75, label="True trajectory")
    ax1.plot(unctrl_post[:, 0], unctrl_post[:, 1], unctrl_post[:, 2], color=_COLOR_UNCONTROLLED, linewidth=1.2, alpha=0.85, label="Uncontrolled ESN")
    ax1.set_title("Held-out controller-test reference and uncontrolled ESN", fontsize=11, fontweight="bold", pad=10)

    # Right: distinguish controller validation from the held-out trajectory
    # that is used only for final periodic-orbit reporting.
    if len(ctrl_transient):
        ax2.plot(
            ctrl_transient[:, 0],
            ctrl_transient[:, 1],
            ctrl_transient[:, 2],
            color=_COLOR_EVENT,
            linewidth=0.9,
            alpha=0.30,
            label="Controller validation (not used for held-out metrics)",
        )
    if previous_cycle is not None and final_cycle is not None:
        ax2.plot(
            ctrl_post[:, 0],
            ctrl_post[:, 1],
            ctrl_post[:, 2],
            color=_COLOR_CONTROLLED,
            linewidth=0.8,
            alpha=0.18,
            label="Held-out corrected-feedback trajectory",
        )
        ax2.plot(
            previous_cycle[:, 0],
            previous_cycle[:, 1],
            previous_cycle[:, 2],
            color=_COLOR_EVENT,
            linewidth=1.2,
            alpha=0.80,
            label="Previous measured cycle",
        )
        ax2.plot(
            final_cycle[:, 0],
            final_cycle[:, 1],
            final_cycle[:, 2],
            color=_COLOR_CONTROLLED,
            linewidth=1.6,
            alpha=0.98,
            label="Final measured cycle",
        )
        ax2.scatter(
            final_cycle[0, 0],
            final_cycle[0, 1],
            final_cycle[0, 2],
            color=_COLOR_EVENT,
            s=28,
            label="Final cycle start",
            depthshade=True,
        )
        ax2.scatter(
            final_cycle[-1, 0],
            final_cycle[-1, 1],
            final_cycle[-1, 2],
            color=_COLOR_TARGET,
            s=28,
            label="Final cycle end",
            depthshade=True,
        )
    else:
        ax2.plot(ctrl_post[:, 0], ctrl_post[:, 1], ctrl_post[:, 2], color=_COLOR_CONTROLLED, linewidth=1.35, alpha=0.95, label="Held-out corrected-feedback trajectory")
        ax2.scatter(ctrl_post[0, 0], ctrl_post[0, 1], ctrl_post[0, 2], color=_COLOR_EVENT, s=28, label="Held-out controller-test start", depthshade=True)
        ax2.scatter(ctrl_post[-1, 0], ctrl_post[-1, 1], ctrl_post[-1, 2], color=_COLOR_TARGET, s=28, label="End", depthshade=True)
    quality_pass = (metrics or {}).get("pyragas_quality_pass")
    status = "PASS" if quality_pass is True else "FAIL" if quality_pass is False else "not assessed"
    rhythm_type = str((metrics or {}).get("pyragas_rhythm_type", "undetermined"))
    ax2.set_title(
        f"Held-out controller-test {rhythm_type} trajectory: {status}",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    for ax in (ax1, ax2):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(True, alpha=0.30)
        ax.view_init(elev=24, azim=-58)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Pyragas delayed-feedback control: 3D HR phase trajectory",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig_to_path(os.path.join(output_dir, "pyragas_phase_trajectory_3d.png"), fig=fig)


def plot_pyragas_periodic_zoom(
    times,
    truth,
    uncontrolled,
    controlled,
    target_state,
    control_start_idx,
    output_dir,
    controller_name=None,
    metrics=None,
):
    """
    Zoomed held-out controller-test plot for Pyragas periodic rhythm.

    Saved as:
        pyragas_periodic_zoom.png
    """
    if not _is_pyragas_context(output_dir=output_dir, controller_name=controller_name):
        return

    times = _as_1d(times)
    truth = _as_2d(truth)
    uncontrolled = _as_2d(uncontrolled)
    controlled = _as_2d(controlled)
    target_state = _as_1d(target_state)

    n = min(len(times), len(truth), len(uncontrolled), len(controlled))
    if n == 0:
        return

    times = times[:n]
    truth = truth[:n]
    uncontrolled = uncontrolled[:n]
    controlled = controlled[:n]
    control_start = int(max(0, min(control_start_idx, n - 1)))
    start = int(
        max(
            control_start,
            min(
                int(_safe_float((metrics or {}).get("pyragas_evaluation_start_idx"), control_start)),
                n - 1,
            ),
        )
    )

    post_len = n - start
    if post_len <= 10:
        return

    empirical_period = int(
        max(
            0,
            round(
                _safe_float(
                    (metrics or {}).get("pyragas_empirical_period_steps"),
                    0,
                )
            ),
        )
    )
    if empirical_period > 0:
        zoom_len = int(min(post_len, max(800, 3 * empirical_period)))
    else:
        zoom_len = int(min(post_len, max(800, 0.35 * post_len)))
    end = min(n, start + zoom_len)

    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.plot(times[start:end], truth[start:end, 0], color=_COLOR_TRUE, linewidth=1.35, label="True x")
    ax.plot(times[start:end], uncontrolled[start:end, 0], color=_COLOR_UNCONTROLLED, linewidth=1.15, alpha=0.85, label="Uncontrolled ESN x")
    ax.plot(times[start:end], controlled[start:end, 0], color=_COLOR_CONTROLLED, linestyle="--", linewidth=1.55, label="Corrected feedback input x (Pyragas)")
    ax.axvline(times[start], color=_COLOR_TARGET, linestyle="-.", linewidth=1.25, label="Held-out controller-test start")
    quality_pass = (metrics or {}).get("pyragas_quality_pass")
    status = "PASS" if quality_pass is True else "FAIL" if quality_pass is False else "not assessed"
    rhythm_type = str((metrics or {}).get("pyragas_rhythm_type", "undetermined"))
    period_text = (
        f", measured period={empirical_period} steps"
        if empirical_period > 0
        else ""
    )
    ax.set_title(
        f"Held-out controller-test Pyragas {rhythm_type} rhythm: {status}{period_text}",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Hindmarsh-Rose x state")
    _style_axis(ax)
    _clean_legend(ax, fontsize=9, loc="upper right", ncol=2)
    fig.tight_layout()
    _savefig_to_path(os.path.join(output_dir, "pyragas_periodic_zoom.png"), fig=fig)

def plot_controlled_all_states(
    times,
    truth,
    uncontrolled,
    controlled,
    target_state,
    control_start_idx,
    output_dir,
    controller_name=None,
    metrics=None,
):
    """Plot all available HR states with the correct controller label."""
    times = _as_1d(times)
    truth = _as_2d(truth)
    uncontrolled = _as_2d(uncontrolled)
    controlled = _as_2d(controlled)
    target_state = _as_1d(target_state)

    n = min(len(times), len(truth), len(uncontrolled), len(controlled))
    if n == 0:
        return
    times = times[:n]
    truth = truth[:n]
    uncontrolled = uncontrolled[:n]
    controlled = controlled[:n]
    control_start_idx = int(max(0, min(control_start_idx, n - 1)))

    label = _controller_label(controller_name, metrics=metrics, output_dir=output_dir)
    is_pyragas = _is_pyragas_context(
        metrics=metrics,
        output_dir=output_dir,
        controller_name=controller_name,
    )

    labels_short = ["x", "y", "z"]
    names = [
        "x: membrane voltage / spike variable",
        "y: recovery variable",
        "z: slow adaptation variable",
    ]

    n_states = min(3, truth.shape[1], uncontrolled.shape[1], controlled.shape[1])
    fig, axes = plt.subplots(n_states, 1, figsize=(16, 3.45 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(times, truth[:, i], color=_COLOR_TRUE, linewidth=1.45, label=f"True {labels_short[i]}", zorder=4)
        ax.plot(
            times,
            uncontrolled[:, i],
            color=_COLOR_UNCONTROLLED,
            linewidth=1.15,
            alpha=0.88,
            label=f"Uncontrolled {labels_short[i]}",
            zorder=2,
        )
        ax.plot(
            times,
            controlled[:, i],
            color=_COLOR_CONTROLLED,
            linestyle="--",
            linewidth=1.45,
            label=f"Corrected feedback input {labels_short[i]}",
            zorder=5,
        )
        if not is_pyragas:
            ax.axhline(
                target_state[i],
                color=_COLOR_TARGET,
                linestyle=":",
                linewidth=1.35,
                label=(
                    "Empirical quiet-state reference "
                    f"{labels_short[i]}"
                ),
                zorder=1,
            )
        ax.axvline(
            times[control_start_idx],
            color=_COLOR_EVENT,
            linestyle="--",
            linewidth=1.15,
            label="Control start",
            zorder=1,
        )
        _add_controller_segment_guides(
            ax, times, control_start_idx, metrics=metrics
        )
        ax.set_ylabel(labels_short[i])
        ax.set_title(names[i], fontsize=11, fontweight="bold", pad=7)
        _style_axis(ax)
        _clean_legend(ax, fontsize=8, loc="upper right", ncol=3)

    axes[-1].set_xlabel("Time")
    states_title = (
        f"{label}: all HR states"
        if is_pyragas
        else (
            f"{label}: regulation toward an empirical quiet-state "
            "reference"
        )
    )
    fig.suptitle(
        states_title, fontsize=16, fontweight="bold", y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _savefig_to_path(os.path.join(output_dir, "controlled_all_states.png"), fig=fig)

    if is_pyragas:
        plot_pyragas_phase_trajectory_3d(
            times,
            truth,
            uncontrolled,
            controlled,
            control_start_idx,
            output_dir,
            controller_name=controller_name,
            metrics=metrics,
        )
        plot_pyragas_periodic_zoom(
            times,
            truth,
            uncontrolled,
            controlled,
            target_state,
            control_start_idx,
            output_dir,
            controller_name=controller_name,
            metrics=metrics,
        )


def plot_control_signal(
    times,
    control_signal,
    control_start_idx,
    output_dir,
    controller_name=None,
    metrics=None,
):
    """Plot the control signal with the correct controller label."""
    times = _as_1d(times)
    control_signal = _as_2d(control_signal)

    n = min(len(times), len(control_signal))
    if n == 0:
        return
    times = times[:n]
    control_signal = control_signal[:n]
    control_start_idx = int(max(0, min(control_start_idx, n - 1)))
    control_norm = np.linalg.norm(control_signal, axis=1)
    label = _controller_label(
        controller_name, metrics=metrics, output_dir=output_dir
    )

    fig, ax = plt.subplots(figsize=(16, 5.4))
    ax.plot(times, control_norm, color=_COLOR_CONTROLLED, linewidth=1.55, label=r"$||u(t)||$", zorder=4)
    ax.plot(times, control_signal[:, 0], color=_COLOR_UNCONTROLLED, linestyle="--", linewidth=1.25, label=r"$u_x(t)$", zorder=3)
    ax.axvline(times[control_start_idx], color=_COLOR_EVENT, linestyle="--", linewidth=1.3, label="Control start", zorder=2)
    _add_controller_segment_guides(ax, times, control_start_idx, metrics=metrics)
    ax.set_title(f"{label}: control signal", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Control magnitude")
    _style_axis(ax)
    _clean_legend(ax, fontsize=9, loc="upper right")
    fig.tight_layout()
    _savefig_to_path(os.path.join(output_dir, "control_signal.png"), fig=fig)


def plot_control_error(
    times,
    uncontrolled_error_norm,
    controlled_error_norm,
    control_start_idx,
    settling_tolerance,
    output_dir,
    controller_name=None,
    metrics=None,
):
    """Plot target-tracking error with readable styling and correct controller label."""
    times = _as_1d(times)
    uncontrolled_error_norm = _as_1d(uncontrolled_error_norm)
    controlled_error_norm = _as_1d(controlled_error_norm)

    n = min(len(times), len(uncontrolled_error_norm), len(controlled_error_norm))
    if n == 0:
        return
    times = times[:n]
    uncontrolled_error_norm = uncontrolled_error_norm[:n]
    controlled_error_norm = controlled_error_norm[:n]
    control_start_idx = int(max(0, min(control_start_idx, n - 1)))
    label = _controller_label(
        controller_name, metrics=metrics, output_dir=output_dir
    )
    is_pyragas = _is_pyragas_context(
        metrics=metrics,
        output_dir=output_dir,
        controller_name=controller_name,
    )

    fig, ax = plt.subplots(figsize=(16, 5.4))
    ax.plot(times, uncontrolled_error_norm, color=_COLOR_UNCONTROLLED, linewidth=1.25, label="Uncontrolled error")
    ax.plot(times, controlled_error_norm, color=_COLOR_CONTROLLED, linestyle="--", linewidth=1.55, label="Corrected feedback-input error")
    ax.axhline(settling_tolerance, color=_COLOR_TARGET, linestyle=":", linewidth=1.35, label="Settling tolerance")
    ax.axvline(times[control_start_idx], color=_COLOR_EVENT, linestyle="--", linewidth=1.3, label="Control start")
    _add_controller_segment_guides(ax, times, control_start_idx, metrics=metrics)
    error_title = (
        "reference-state distance (diagnostic only)"
        if is_pyragas
        else "empirical quiet-reference tracking error"
    )
    ax.set_title(f"{label}: {error_title}", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$||state - empirical\ quiet\ reference||$")
    _style_axis(ax)
    _clean_legend(ax, fontsize=9, loc="upper right")
    fig.tight_layout()
    _savefig_to_path(os.path.join(output_dir, "control_error.png"), fig=fig)


def plot_k_sweep_summary(rows, output_dir, controller_name=None):
    """Plot controller-validation metrics used to select the feedback gain."""
    if not rows:
        return

    stable_rows = [r for r in rows if bool(r.get("stable", False))]
    if not stable_rows:
        stable_rows = rows

    label = _controller_label(
        controller_name,
        output_dir=output_dir,
        rows=stable_rows,
    )

    # Pyragas selects periodic-orbit regularity rather than fixed-target error.
    if _is_pyragas_context(
        output_dir=output_dir,
        controller_name=controller_name,
        rows=stable_rows,
    ):
        k = np.array(
            [_safe_float(r.get("K", np.nan)) for r in stable_rows],
            dtype=float,
        )
        cv = np.array(
            [
                _safe_float(r.get("pyragas_rhythm_interval_cv", np.nan))
                for r in stable_rows
            ],
            dtype=float,
        )
        amp = np.array(
            [
                _safe_float(r.get("pyragas_x_amplitude_ratio", np.nan))
                for r in stable_rows
            ],
            dtype=float,
        )
        periodicity = np.array(
            [
                _safe_float(
                    r.get("pyragas_empirical_recurrence_error_norm", np.nan)
                )
                for r in stable_rows
            ],
            dtype=float,
        )
        effort = np.array(
            [_safe_float(_control_effort_value(r)) for r in stable_rows],
            dtype=float,
        )
        selection = np.array(
            [
                _safe_float(
                    r.get(
                        "selection_metric_value",
                        r.get("selection_score", np.nan),
                    )
                )
                for r in stable_rows
            ],
            dtype=float,
        )
        quality = np.array(
            [
                bool(r.get("pyragas_quality_pass", False))
                for r in stable_rows
            ],
            dtype=bool,
        )

        valid = np.isfinite(k)
        if not np.any(valid):
            return

        k, cv, amp, periodicity, effort, selection, quality = (
            k[valid],
            cv[valid],
            amp[valid],
            periodicity[valid],
            effort[valid],
            selection[valid],
            quality[valid],
        )
        order = np.argsort(k)
        k, cv, amp, periodicity, effort, selection, quality = (
            k[order],
            cv[order],
            amp[order],
            periodicity[order],
            effort[order],
            selection[order],
            quality[order],
        )

        fig, axes = plt.subplots(4, 1, figsize=(12.5, 13.0), sharex=True)
        series = [
            (
                cv,
                "Validation rhythm CV",
                "lower = more regular validation cycles",
                "min",
            ),
            (
                amp,
                "Validation x amplitude ratio",
                "closer to 1 on validation segment",
                "closest1",
            ),
            (
                periodicity,
                "Validation cycle recurrence error",
                "lower = stronger validation recurrence",
                "min",
            ),
            (
                effort,
                "Validation mean-squared control effort",
                "lower = less validation control effort",
                "min",
            ),
        ]

        best_idx = None
        selected_label = "Best validation candidate (no quality pass)"
        finite_selection = np.isfinite(selection)
        valid_selection = finite_selection & quality
        if np.any(valid_selection):
            candidates = np.flatnonzero(valid_selection)
            best_idx = int(candidates[np.argmin(selection[candidates])])
            selected_label = "Validation-selected quality-passing K"
        elif np.any(finite_selection):
            best_idx = int(np.nanargmin(selection))

        for ax, (vals, ylabel, legend_label, mode) in zip(axes, series):
            ax.plot(
                k,
                vals,
                color=_COLOR_CONTROLLED,
                marker="o",
                markersize=4.5,
                linewidth=1.55,
                label=legend_label,
            )
            finite_mask = np.isfinite(vals)
            if np.any(finite_mask):
                if (
                    best_idx is not None
                    and best_idx < len(k)
                    and np.isfinite(vals[best_idx])
                ):
                    idx = best_idx
                    star_label = selected_label
                elif mode == "closest1":
                    idx = int(np.nanargmin(np.abs(vals - 1.0)))
                    star_label = "Best visible validation point"
                else:
                    idx = int(np.nanargmin(vals))
                    star_label = "Best visible validation point"
                ax.scatter(
                    [k[idx]],
                    [vals[idx]],
                    color=_COLOR_EVENT,
                    s=70,
                    marker="*",
                    zorder=5,
                    label=star_label,
                )
            if mode == "closest1":
                ax.axhline(
                    1.0,
                    color=_COLOR_TARGET,
                    linestyle=":",
                    linewidth=1.1,
                    label="ratio = 1",
                )
            ax.set_ylabel(ylabel)
            _style_axis(ax)
            _clean_legend(ax, fontsize=8, loc="best")

        axes[-1].set_xlabel(
            "Feedback gain K (selected on controller-validation segment)"
        )
        quality_summary = (
            f"{int(np.sum(quality))} validation quality-passing candidate(s)"
            if np.any(quality)
            else "NO VALIDATION QUALITY-PASSING CANDIDATE"
        )
        fig.suptitle(
            f"{label}: controller-validation Pyragas K sweep\n"
            f"{quality_summary}; controller-test segment not used",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        pyragas_path = os.path.join(
            output_dir,
            "pyragas_periodic_k_sweep_summary.png",
        )
        legacy_path = os.path.join(output_dir, "control_sweep_summary.png")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(pyragas_path, dpi=_PLOT_DPI, bbox_inches="tight")
        fig.savefig(legacy_path, dpi=_PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plot] Saved -> {pyragas_path}")
        print(f"[Plot] Saved -> {legacy_path}")
        return

    k = np.array(
        [_safe_float(r.get("K", np.nan)) for r in stable_rows],
        dtype=float,
    )
    rmse_vals = np.array(
        [
            _safe_float(
                r.get(
                    "corrected_feedback_input_target_rmse_state",
                    r.get("target_rmse_state", np.nan),
                )
            )
            for r in stable_rows
        ],
        dtype=float,
    )
    spike = np.array(
        [
            _safe_float(r.get("spike_reduction_percent", np.nan))
            for r in stable_rows
        ],
        dtype=float,
    )
    effort = np.array(
        [_safe_float(_control_effort_value(r)) for r in stable_rows],
        dtype=float,
    )

    valid = np.isfinite(k)
    if not np.any(valid):
        return

    k, rmse_vals, spike, effort = (
        k[valid],
        rmse_vals[valid],
        spike[valid],
        effort[valid],
    )
    order = np.argsort(k)
    k, rmse_vals, spike, effort = (
        k[order],
        rmse_vals[order],
        spike[order],
        effort[order],
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True)
    series = [
        (
            rmse_vals,
            "Validation corrected-feedback target RMSE",
            "Validation corrected-feedback target RMSE",
            "min",
        ),
        (
            spike,
            "Validation spike reduction (%)",
            "Validation spike reduction",
            "max",
        ),
        (
            effort,
            "Validation mean-squared control effort",
            "Validation mean-squared control effort",
            "min",
        ),
    ]

    for ax, (vals, ylabel, legend_label, mode) in zip(axes, series):
        ax.plot(
            k,
            vals,
            color=_COLOR_CONTROLLED,
            marker="o",
            markersize=4.5,
            linewidth=1.55,
            label=legend_label,
        )
        finite_mask = np.isfinite(vals)
        if np.any(finite_mask):
            if mode == "max":
                best_idx = int(np.nanargmax(vals))
            else:
                best_idx = int(np.nanargmin(vals))
            ax.scatter(
                [k[best_idx]],
                [vals[best_idx]],
                color=_COLOR_EVENT,
                s=70,
                marker="*",
                zorder=5,
                label="Best visible validation point",
            )
        ax.set_ylabel(ylabel)
        _style_axis(ax)
        _clean_legend(ax, fontsize=8, loc="best")

    axes[-1].set_xlabel(
        "Feedback gain K (selected on controller-validation segment)"
    )
    fig.suptitle(
        f"{label}: controller-validation K sweep "
        "(controller-test segment not used)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _savefig_to_path(
        os.path.join(output_dir, "control_sweep_summary.png"),
        fig=fig,
    )

def _default_table_format(value):
    if value == "" or value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    try:
        v = float(value)
        if not np.isfinite(v):
            return ""
        if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 1e-4):
            return f"{v:.2e}"
        if abs(v) >= 10:
            return f"{v:.3f}"
        return f"{v:.6f}"
    except Exception:
        return str(value)


def plot_final_comparison_table(path, rows, formatter=None):
    if not rows:
        return
    formatter = formatter or _default_table_format

    effort_column = next(
        (
            name
            for name in (
                "Control_effort_mean_sq",
                "control_effort_mean_sq",
                "Control_energy_legacy_alias",
            )
            if any(name in row for row in rows)
        ),
        "Control_effort_mean_sq",
    )
    columns = [
        "Regime",
        "Optimizer",
        "Pred_NRMSE_x",
        "Pred_NRMSE_all",
        "Best_K",
        "Final_test_metric_name",
        "Final_test_metric_value",
        "Control_target_RMSE_state",
        "Spike_reduction_percent",
        effort_column,
        "Controller_test_time_to_tolerance",
        "Control_stable",
    ]
    columns = [c for c in columns if c in rows[0]]
    cell_text = [[formatter(row.get(c, "")) for c in columns] for row in rows]

    fig_height = max(2.2, 1.0 + 0.45 * len(rows))
    fig_width = max(12.0, 1.3 * len(columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
        cell.set_linewidth(0.4)

    ax.set_title(
        "Final ESN + BO + Control comparison",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    _savefig_to_path(path, fig=fig, dpi=220)
