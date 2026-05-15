import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _load_metrics(metrics):
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        return metrics
    if isinstance(metrics, str) and os.path.isfile(metrics):
        import json
        with open(metrics, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def plot_controlled_vs_uncontrolled_x_better(
    rollout_csv,
    output_dir,
    metrics=None,
    zoom_before=300,
    zoom_after=700,
):
    """
    Creates a better x-state plot:
    - full trajectory
    - zoom near control start
    """
    _ensure_dir(output_dir)
    metrics = _load_metrics(metrics)
    df = pd.read_csv(rollout_csv)

    required = [
        "time",
        "true_x",
        "uncontrolled_x",
        "controlled_x",
        "target_x",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in rollout csv: {missing}")

    times = df["time"].to_numpy()
    true_x = df["true_x"].to_numpy()
    uncontrolled_x = df["uncontrolled_x"].to_numpy()
    controlled_x = df["controlled_x"].to_numpy()
    target_x = df["target_x"].to_numpy()

    if "control_start_time" in metrics:
        t0 = float(metrics["control_start_time"])
        control_start_idx = int(np.argmin(np.abs(times - t0)))
    else:
        # fallback: first row where control signal becomes nonzero
        if {"u_x", "u_y", "u_z"}.issubset(df.columns):
            u_norm = np.sqrt(df["u_x"] ** 2 + df["u_y"] ** 2 + df["u_z"] ** 2).to_numpy()
            nz = np.where(u_norm > 1e-12)[0]
            control_start_idx = int(nz[0]) if len(nz) else 0
        else:
            control_start_idx = 0
        t0 = float(times[control_start_idx])

    left = max(0, control_start_idx - zoom_before)
    right = min(len(df), control_start_idx + zoom_after)

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2.2, 1.3]}
    )

    # full plot
    ax = axes[0]
    ax.plot(times, true_x, linewidth=1.5, label="True x")
    ax.plot(times, uncontrolled_x, linewidth=1.4, label="Uncontrolled ESN x")
    ax.plot(times, controlled_x, linestyle="--", linewidth=1.8, label="Controlled ESN x")
    ax.axhline(target_x[0], linestyle=":", linewidth=1.5, label="Target x")
    ax.axvline(t0, linestyle="--", linewidth=1.3, label="Control start")
    ax.axvspan(t0, times[-1], alpha=0.08)

    txt_lines = []
    if "K" in metrics:
        txt_lines.append(f"K = {metrics['K']:.4f}")
    if "target_rmse_state" in metrics:
        txt_lines.append(f"Target RMSE = {metrics['target_rmse_state']:.3e}")
    if "spike_reduction_percent" in metrics:
        txt_lines.append(f"Spike reduction = {metrics['spike_reduction_percent']:.1f}%")
    if "control_energy" in metrics:
        txt_lines.append(f"Energy = {metrics['control_energy']:.3e}")

    if txt_lines:
        ax.text(
            0.015,
            0.97,
            "\n".join(txt_lines),
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

    ax.set_title("Linear feedback control: x-state comparison", fontweight="bold")
    ax.set_ylabel("x state")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    # zoom plot
    ax2 = axes[1]
    ax2.plot(times[left:right], true_x[left:right], linewidth=1.5, label="True x")
    ax2.plot(times[left:right], uncontrolled_x[left:right], linewidth=1.4, label="Uncontrolled ESN x")
    ax2.plot(times[left:right], controlled_x[left:right], linestyle="--", linewidth=1.8, label="Controlled ESN x")
    ax2.axhline(target_x[0], linestyle=":", linewidth=1.5, label="Target x")
    ax2.axvline(t0, linestyle="--", linewidth=1.3, label="Control start")
    ax2.axvspan(t0, times[right - 1], alpha=0.08)

    ax2.set_title("Zoom around control start", fontweight="bold")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("x state")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "controlled_vs_uncontrolled_x_better.png")
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path}")


def plot_error_and_signal_better(rollout_csv, output_dir, metrics=None):
    """
    Creates one figure with:
    - state error norm over time
    - control signal norm over time
    """
    _ensure_dir(output_dir)
    metrics = _load_metrics(metrics)
    df = pd.read_csv(rollout_csv)

    required = [
        "time",
        "controlled_x", "controlled_y", "controlled_z",
        "target_x", "target_y", "target_z",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in rollout csv: {missing}")

    times = df["time"].to_numpy()

    err = np.sqrt(
        (df["controlled_x"] - df["target_x"]) ** 2 +
        (df["controlled_y"] - df["target_y"]) ** 2 +
        (df["controlled_z"] - df["target_z"]) ** 2
    ).to_numpy()

    if {"u_x", "u_y", "u_z"}.issubset(df.columns):
        u_norm = np.sqrt(df["u_x"] ** 2 + df["u_y"] ** 2 + df["u_z"] ** 2).to_numpy()
    else:
        u_norm = np.zeros(len(df))

    if "control_start_time" in metrics:
        t0 = float(metrics["control_start_time"])
    else:
        nz = np.where(u_norm > 1e-12)[0]
        t0 = float(times[nz[0]]) if len(nz) else float(times[0])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(times, err, linewidth=1.8)
    axes[0].axvline(t0, linestyle="--", linewidth=1.3, label="Control start")
    axes[0].axvspan(t0, times[-1], alpha=0.08)
    axes[0].set_title("Controlled state error norm", fontweight="bold")
    axes[0].set_ylabel("||controlled - target||")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(times, u_norm, linewidth=1.8, label="||u(t)||")
    if "u_x" in df.columns:
        axes[1].plot(times, df["u_x"].to_numpy(), linestyle="--", linewidth=1.2, label="u_x")
    axes[1].axvline(t0, linestyle="--", linewidth=1.3, label="Control start")
    axes[1].axvspan(t0, times[-1], alpha=0.08)
    axes[1].set_title("Control signal", fontweight="bold")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Control")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "control_error_and_signal_better.png")
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path}")


def plot_k_sweep_results_better(summary_csv, output_dir):
    """
    Better 2x2 K-sweep summary plot from linear_feedback_metrics_summary.csv
    """
    _ensure_dir(output_dir)
    df = pd.read_csv(summary_csv)

    if "K" not in df.columns:
        raise ValueError("Summary CSV must contain 'K' column.")

    df = df.sort_values("K").reset_index(drop=True)

    if "selection_score" in df.columns and df["selection_score"].notna().any():
        best_idx = int(df["selection_score"].idxmin())
    elif "target_rmse_state" in df.columns and df["target_rmse_state"].notna().any():
        best_idx = int(df["target_rmse_state"].idxmin())
    else:
        best_idx = 0

    best_k = float(df.loc[best_idx, "K"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    panels = [
        ("target_rmse_state", "Target RMSE vs K", "Target RMSE", True),
        ("spike_reduction_percent", "Spike Reduction vs K", "Spike Reduction (%)", False),
        ("control_energy", "Control Energy vs K", "Energy", True),
        ("settling_time", "Settling Time vs K", "Settling Time", False),
    ]

    for ax, (col, title, ylabel, sci) in zip(axes, panels):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"{col} missing", ha="center", va="center")
            ax.set_title(title, fontweight="bold")
            ax.grid(True, alpha=0.25)
            continue

        x = df["K"].to_numpy()
        y = df[col].to_numpy()

        ax.plot(x, y, marker="o", linewidth=2.0, markersize=6)
        ax.axvline(best_k, linestyle="--", linewidth=1.3, alpha=0.85)

        if np.isfinite(df.loc[best_idx, col]):
            ax.scatter([best_k], [df.loc[best_idx, col]], s=110, marker="*", label=f"Best K = {best_k:.4g}")

        if np.all(x > 0) and len(np.unique(x)) > 2:
            ax.set_xscale("log")

        if sci:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    summary_lines = [f"Best K = {best_k:.6g}"]
    if "target_rmse_state" in df.columns:
        summary_lines.append(f"RMSE = {df.loc[best_idx, 'target_rmse_state']:.3e}")
    if "spike_reduction_percent" in df.columns:
        summary_lines.append(f"Spike reduction = {df.loc[best_idx, 'spike_reduction_percent']:.1f}%")
    if "control_energy" in df.columns:
        summary_lines.append(f"Energy = {df.loc[best_idx, 'control_energy']:.3e}")

    fig.suptitle("K-Sweep Summary", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        " | ".join(summary_lines),
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_path = os.path.join(output_dir, "k_sweep_summary_better.png")
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path}")

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_bo_heatmap_better(
    results_csv,
    output_dir,
    x_col,
    y_col,
    value_col,
    filename="bo_heatmap_better.png",
    agg="min",
    x_log=False,
    y_log=False,
    annotate_best=True,
):
    """
    Create a better 2D heatmap from saved hyperparameter search results.

    Parameters
    ----------
    results_csv : str
        Path to CSV containing BO / optimization results.
    output_dir : str
        Folder to save the plot.
    x_col, y_col : str
        Hyperparameter column names for x and y axes.
    value_col : str
        Metric column to visualize (e.g. objective, recursive_nrmse, score).
    filename : str
        Output image name.
    agg : str
        Aggregation for repeated (x,y) points: "min", "mean", or "median".
    x_log, y_log : bool
        Whether to display axis labels as log-scale values.
    annotate_best : bool
        Whether to mark the best point on the heatmap.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)

    needed = [x_col, y_col, value_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df[[x_col, y_col, value_col]].copy()
    df = df.dropna()

    if df.empty:
        print("[Plot] No valid rows after dropping NaNs.")
        return

    # choose aggregation
    if agg == "min":
        pivot_df = df.groupby([y_col, x_col], as_index=False)[value_col].min()
    elif agg == "mean":
        pivot_df = df.groupby([y_col, x_col], as_index=False)[value_col].mean()
    elif agg == "median":
        pivot_df = df.groupby([y_col, x_col], as_index=False)[value_col].median()
    else:
        raise ValueError("agg must be one of: 'min', 'mean', 'median'")

    heatmap = pivot_df.pivot(index=y_col, columns=x_col, values=value_col)

    # sort axes numerically
    heatmap = heatmap.sort_index(axis=0).sort_index(axis=1)

    x_vals = heatmap.columns.to_numpy()
    y_vals = heatmap.index.to_numpy()
    z = heatmap.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(
        z,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    # axis ticks
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_yticks(np.arange(len(y_vals)))

    def _fmt(vals, log_flag):
        out = []
        for v in vals:
            if log_flag:
                out.append(f"{v:.1e}")
            else:
                if abs(v) >= 100 or abs(v) < 0.01:
                    out.append(f"{v:.2e}")
                else:
                    out.append(f"{v:.4g}")
        return out

    ax.set_xticklabels(_fmt(x_vals, x_log), rotation=45, ha="right")
    ax.set_yticklabels(_fmt(y_vals, y_log))

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Heatmap of {value_col} over {x_col} and {y_col}", fontweight="bold")

    # Annotate best cell
    if annotate_best and np.isfinite(z).any():
        best_flat = np.nanargmin(z)  # assumes lower is better
        best_row, best_col = np.unravel_index(best_flat, z.shape)
        best_x = x_vals[best_col]
        best_y = y_vals[best_row]
        best_val = z[best_row, best_col]

        ax.scatter(best_col, best_row, marker="*", s=220, edgecolors="black", label="Best")
        ax.legend(loc="upper right")

        ax.text(
            0.5,
            -0.18,
            f"Best point: {x_col}={best_x:.4g}, {y_col}={best_y:.4g}, {value_col}={best_val:.4e}",
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path}")