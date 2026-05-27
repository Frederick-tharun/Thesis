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


def _output_dir():
    if config is None:
        return "outputs"
    return getattr(config, "OUTPUT_DIR", "outputs")


def _savefig(filename):
    os.makedirs(_output_dir(), exist_ok=True)
    path = os.path.join(_output_dir(), filename)
    plt.savefig(path, dpi=180, bbox_inches="tight")
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
    ax.set_xlabel("Time (s)")
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
    plt.xlabel("Time (s)")
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
    plt.xlabel("Time (s)")
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

    axes[-1].set_xlabel("Time (s)")
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


def _savefig_to_path(path, fig=None, dpi=220):
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
):
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

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(times, truth[:, 0], color="black", linewidth=1.4, label="True x")
    ax.plot(times, uncontrolled[:, 0], linewidth=1.2, label="Uncontrolled ESN x")
    ax.plot(times, controlled[:, 0], linestyle="--", linewidth=1.4, label="Controlled ESN x")
    ax.axhline(target_state[0], linestyle=":", linewidth=1.4, label="Target x")
    ax.axvline(times[control_start_idx], linestyle="--", linewidth=1.4, label="Control start")

    txt = (
        f"K = {metrics.get('K')}\n"
        f"Target RMSE = {_safe_float(metrics.get('target_rmse_state')):.4f}\n"
        f"Spike reduction = {_safe_float(metrics.get('spike_reduction_percent')):.2f}%\n"
        f"Energy = {_safe_float(metrics.get('control_energy')):.4f}"
    )
    ax.text(
        0.01,
        0.97,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )

    ax.set_title("Linear feedback control: x-state comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("x state")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    _savefig_to_path(os.path.join(output_dir, "controlled_vs_uncontrolled_x.png"), fig=fig, dpi=220)


def plot_controlled_all_states(
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
    if n == 0:
        return
    times = times[:n]
    truth = truth[:n]
    uncontrolled = uncontrolled[:n]
    controlled = controlled[:n]
    control_start_idx = int(max(0, min(control_start_idx, n - 1)))

    labels_short = ["x", "y", "z"]
    names = [
        "x: membrane voltage / spike variable",
        "y: recovery variable",
        "z: slow adaptation variable",
    ]

    n_states = min(3, truth.shape[1], uncontrolled.shape[1], controlled.shape[1])
    fig, axes = plt.subplots(n_states, 1, figsize=(15, 3.2 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(times, truth[:, i], color="black", linewidth=1.2, label=f"True {labels_short[i]}")
        ax.plot(times, uncontrolled[:, i], linewidth=1.1, label=f"Uncontrolled {labels_short[i]}")
        ax.plot(times, controlled[:, i], linestyle="--", linewidth=1.3, label=f"Controlled {labels_short[i]}")
        ax.axhline(target_state[i], linestyle=":", linewidth=1.2, label=f"Target {labels_short[i]}")
        ax.axvline(times[control_start_idx], linestyle="--", linewidth=1.1, label="Control start")
        ax.set_ylabel(labels_short[i])
        ax.set_title(names[i], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Linear feedback control: all HR states", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig_to_path(os.path.join(output_dir, "controlled_all_states.png"), fig=fig, dpi=220)


def plot_control_signal(times, control_signal, control_start_idx, output_dir):
    times = _as_1d(times)
    control_signal = _as_2d(control_signal)

    n = min(len(times), len(control_signal))
    if n == 0:
        return
    times = times[:n]
    control_signal = control_signal[:n]
    control_start_idx = int(max(0, min(control_start_idx, n - 1)))
    control_norm = np.linalg.norm(control_signal, axis=1)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(times, control_norm, linewidth=1.4, label=r"$||u(t)||$")
    ax.plot(times, control_signal[:, 0], linestyle="--", linewidth=1.2, label=r"$u_x(t)$")
    ax.axvline(times[control_start_idx], linestyle="--", linewidth=1.3, label="Control start")
    ax.set_title("Linear feedback control signal", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control signal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    _savefig_to_path(os.path.join(output_dir, "control_signal.png"), fig=fig, dpi=220)


def plot_control_error(
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
    if n == 0:
        return
    times = times[:n]
    uncontrolled_error_norm = uncontrolled_error_norm[:n]
    controlled_error_norm = controlled_error_norm[:n]
    control_start_idx = int(max(0, min(control_start_idx, n - 1)))

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(times, uncontrolled_error_norm, linewidth=1.2, label="Uncontrolled error")
    ax.plot(times, controlled_error_norm, linestyle="--", linewidth=1.4, label="Controlled error")
    ax.axhline(settling_tolerance, linestyle=":", linewidth=1.3, label="Settling tolerance")
    ax.axvline(times[control_start_idx], linestyle="--", linewidth=1.3, label="Control start")
    ax.set_title("Target-tracking error", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$||state - target||$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    _savefig_to_path(os.path.join(output_dir, "control_error.png"), fig=fig, dpi=220)


def plot_k_sweep_summary(rows, output_dir):
    if not rows:
        return

    stable_rows = [r for r in rows if bool(r.get("stable", False))]
    if not stable_rows:
        stable_rows = rows

    k = np.array([_safe_float(r.get("K", np.nan)) for r in stable_rows], dtype=float)
    rmse_vals = np.array([_safe_float(r.get("target_rmse_state", np.nan)) for r in stable_rows], dtype=float)
    spike = np.array([_safe_float(r.get("spike_reduction_percent", np.nan)) for r in stable_rows], dtype=float)
    energy = np.array([_safe_float(r.get("control_energy", np.nan)) for r in stable_rows], dtype=float)

    order = np.argsort(k)
    k, rmse_vals, spike, energy = k[order], rmse_vals[order], spike[order], energy[order]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(k, rmse_vals, marker="o")
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
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig_to_path(os.path.join(output_dir, "control_sweep_summary.png"), fig=fig, dpi=220)


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

    columns = [
        "Regime",
        "Optimizer",
        "Pred_NRMSE_x",
        "Pred_NRMSE_all",
        "Best_K",
        "Control_target_RMSE_state",
        "Spike_reduction_percent",
        "Control_energy",
        "Settling_time",
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
