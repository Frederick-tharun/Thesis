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

def plot_bo_objective_landscape(
    rows,
    optimizer="forest",
    params=("spectral_radius", "leaky_coefficient", "input_scaling"),
    labels=None,
    filename=None,
):
    """
    Paper-style Bayesian Optimization objective landscape / partial-dependence plot.

    This is intended to replace the simple optimizer heatmap for thesis figures.
    It shows:
      - diagonal: 1D objective sensitivity for each hyperparameter
      - lower triangle: 2D objective landscape for hyperparameter pairs
      - black dots: sampled BO points
      - red star/dotted line: best sampled hyperparameter value

    Lower objective value is better. For visualization, objective values are normalized
    from 0 to 1, where 0 is best and 1 is worst among valid points.
    """
    if not rows:
        print("[Plot] BO objective landscape skipped: empty rows")
        return

    if labels is None:
        labels = {
            "N_res": r"$N_{res}$",
            "p": r"density $p$",
            "spectral_radius": r"$\rho$",
            "leaky_coefficient": r"leak $\alpha$",
            "input_scaling": "input scale",
            "regularization": "ridge",
            "washout": "washout",
        }

    optimizer = str(optimizer).strip().lower()
    params = tuple(params)

    clean_values = []
    clean_scores = []

    for row in rows:
        if str(row.get("optimizer", "")).strip().lower() != optimizer:
            continue

        score = _safe_float(
            row.get("score", row.get("validation_score", row.get("best_score", np.nan)))
        )

        # Failed/exploded ESN candidates are usually stored as 1e6.
        # They destroy the colour scale, so keep only meaningful BO points.
        if not np.isfinite(score) or score >= 1e5:
            continue

        vals = []
        ok = True
        for name in params:
            v = _safe_float(row.get(name, np.nan))
            if not np.isfinite(v):
                ok = False
                break
            vals.append(v)

        if ok:
            clean_values.append(vals)
            clean_scores.append(score)

    if len(clean_scores) < max(6, len(params) + 2):
        print(
            f"[Plot] BO objective landscape skipped: not enough valid points for {optimizer}"
        )
        return

    X = np.asarray(clean_values, dtype=float)
    y = np.asarray(clean_scores, dtype=float)
    best_idx = int(np.argmin(y))

    # Robust normalization: 0=best, 1=bad. Use 95th percentile so one bad point
    # does not flatten all meaningful colour differences.
    y_best = float(np.min(y))
    y_worst = float(np.percentile(y, 95))
    if abs(y_worst - y_best) < 1e-15:
        y_norm = np.zeros_like(y)
    else:
        y_norm = np.clip((y - y_best) / (y_worst - y_best), 0.0, 1.0)

    d = len(params)
    fig, axes = plt.subplots(d, d, figsize=(3.6 * d + 1.6, 3.35 * d))
    if d == 1:
        axes = np.asarray([[axes]])

    cmap = plt.cm.viridis

    def _binned_curve(x_values, scores, n_bins=18):
        """Return a simple 1D partial-dependence-like curve from sampled points."""
        x_values = np.asarray(x_values, dtype=float)
        scores = np.asarray(scores, dtype=float)
        if len(np.unique(x_values)) < 2:
            return x_values, scores

        edges = np.linspace(np.min(x_values), np.max(x_values), n_bins + 1)
        xs, ys = [], []
        for left, right in zip(edges[:-1], edges[1:]):
            if right == edges[-1]:
                mask = (x_values >= left) & (x_values <= right)
            else:
                mask = (x_values >= left) & (x_values < right)
            if np.any(mask):
                xs.append(float(np.mean(x_values[mask])))
                # Use the lower envelope, because BO cares about the best achievable value.
                ys.append(float(np.min(scores[mask])))
        return np.asarray(xs), np.asarray(ys)

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]

            if i < j:
                ax.axis("off")
                continue

            xj = X[:, j]
            xi = X[:, i]

            if i == j:
                xs, ys = _binned_curve(xi, y_norm, n_bins=min(18, max(5, len(y_norm)//2)))
                order = np.argsort(xs)
                xs = xs[order]
                ys = ys[order]

                ax.plot(xs, ys, linewidth=1.4)
                ax.scatter(xi, y_norm, s=12, alpha=0.35)
                ax.axvline(X[best_idx, i], color="red", linestyle=":", linewidth=1.4)
                ax.scatter(
                    [X[best_idx, i]],
                    [y_norm[best_idx]],
                    color="red",
                    marker="*",
                    s=80,
                    zorder=5,
                )
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel("Partial dep.\nobj. func.", fontsize=9)
                ax.set_title(labels.get(params[i], params[i]), fontsize=11)

            else:
                # 2D objective landscape. Interpolate only if enough unique points exist.
                drew_surface = False
                try:
                    from scipy.interpolate import griddata

                    if len(np.unique(xj)) >= 3 and len(np.unique(xi)) >= 3:
                        gx = np.linspace(np.min(xj), np.max(xj), 120)
                        gy = np.linspace(np.min(xi), np.max(xi), 120)
                        GX, GY = np.meshgrid(gx, gy)
                        Z = griddata(
                            points=np.column_stack([xj, xi]),
                            values=y_norm,
                            xi=(GX, GY),
                            method="linear",
                        )

                        ax.imshow(
                            Z,
                            origin="lower",
                            aspect="auto",
                            extent=[gx.min(), gx.max(), gy.min(), gy.max()],
                            cmap=cmap,
                            vmin=0,
                            vmax=1,
                            alpha=0.88,
                        )
                        drew_surface = True
                except Exception:
                    drew_surface = False

                if not drew_surface:
                    ax.scatter(xj, xi, c=y_norm, cmap=cmap, vmin=0, vmax=1, s=22)
                else:
                    ax.scatter(xj, xi, c="black", s=9, alpha=0.85)

                ax.scatter(
                    X[best_idx, j],
                    X[best_idx, i],
                    color="red",
                    marker="*",
                    s=90,
                    zorder=6,
                )

            if i == d - 1:
                ax.set_xlabel(labels.get(params[j], params[j]), fontsize=10)
            if j == 0 and i != j:
                ax.set_ylabel(labels.get(params[i], params[i]), fontsize=10)

            ax.grid(True, alpha=0.15)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(y_norm)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.78, pad=0.02)
    cbar.set_label("Value of obj. func. (normalized)", fontsize=10)

    best_text = ", ".join(
        f"{labels.get(p, p)}={X[best_idx, k]:.4g}" for k, p in enumerate(params)
    )
    fig.suptitle(
        f"BO objective landscape ({optimizer})\nBest score={y_best:.3e}; {best_text}",
        fontsize=14,
        fontweight="bold",
    )

    if filename is None:
        filename = f"bo_objective_landscape_{optimizer}.png"

    plt.tight_layout(rect=[0, 0, 0.90, 0.92])
    _savefig(filename)
