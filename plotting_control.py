from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


def _save(fig, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved -> {path}")


def plot_control_timeseries(
    *,
    times: np.ndarray,
    uncontrolled: np.ndarray,
    controlled: np.ndarray,
    target_state: np.ndarray,
    control_start_time: float,
    output_dir: str,
    tag: str = "Linear feedback control",
):
    times = np.asarray(times, dtype=float)
    uncontrolled = np.asarray(uncontrolled, dtype=float)
    controlled = np.asarray(controlled, dtype=float)
    target_state = np.asarray(target_state, dtype=float).reshape(-1)

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.set_title(f"{tag} | x state overlay", fontsize=15, fontweight="bold")

    ax.plot(times, uncontrolled[:, 0], linewidth=1.2, label="Uncontrolled ESN x")
    ax.plot(times, controlled[:, 0], linewidth=1.4, linestyle="--", label="Controlled ESN x")
    ax.axhline(target_state[0], linewidth=1.1, linestyle=":", label="Target x")
    ax.axvline(control_start_time, linewidth=1.2, linestyle="--", label="Control start")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("x (original scale)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")

    plt.tight_layout()
    _save(fig, output_dir, "control_timeseries_overlay.png")


def plot_control_all_states(
    *,
    times: np.ndarray,
    uncontrolled: np.ndarray,
    controlled: np.ndarray,
    target_state: np.ndarray,
    control_start_time: float,
    output_dir: str,
    tag: str = "Linear feedback control",
):
    times = np.asarray(times, dtype=float)
    uncontrolled = np.asarray(uncontrolled, dtype=float)
    controlled = np.asarray(controlled, dtype=float)
    target_state = np.asarray(target_state, dtype=float).reshape(-1)

    n_states = min(controlled.shape[1], 3)
    labels = ["x", "y", "z"]

    fig, axes = plt.subplots(n_states, 1, figsize=(15, 8), sharex=True)

    if n_states == 1:
        axes = [axes]

    fig.suptitle(f"{tag} | all states", fontsize=16, fontweight="bold", y=0.98)

    for i in range(n_states):
        ax = axes[i]
        ax.plot(times, uncontrolled[:, i], linewidth=1.1, label=f"Uncontrolled {labels[i]}")
        ax.plot(
            times,
            controlled[:, i],
            linewidth=1.3,
            linestyle="--",
            label=f"Controlled {labels[i]}",
        )
        ax.axhline(target_state[i], linewidth=1.0, linestyle=":", label=f"Target {labels[i]}")
        ax.axvline(control_start_time, linewidth=1.1, linestyle="--", label="Control start")
        ax.set_ylabel(labels[i])
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, output_dir, "control_all_states.png")


def plot_control_signal(
    *,
    times: np.ndarray,
    control_signal: np.ndarray,
    control_start_time: float,
    output_dir: str,
    tag: str = "Linear feedback control",
):
    times = np.asarray(times, dtype=float)
    control_signal = np.asarray(control_signal, dtype=float)

    norm_u = np.linalg.norm(control_signal, axis=1)

    fig, ax = plt.subplots(figsize=(15, 4.5))

    ax.set_title(f"{tag} | control signal", fontsize=15, fontweight="bold")
    ax.plot(times, norm_u, linewidth=1.4, label="||u(t)||")

    if control_signal.shape[1] > 0:
        ax.plot(times, control_signal[:, 0], linewidth=1.0, linestyle="--", label="u_x(t)")

    ax.axvline(control_start_time, linewidth=1.1, linestyle="--", label="Control start")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control signal (original scale)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")

    plt.tight_layout()
    _save(fig, output_dir, "control_signal.png")


def plot_control_error(
    *,
    times: np.ndarray,
    controlled: np.ndarray,
    target_state: np.ndarray,
    control_start_time: float,
    output_dir: str,
    tag: str = "Linear feedback control",
):
    times = np.asarray(times, dtype=float)
    controlled = np.asarray(controlled, dtype=float)
    target_state = np.asarray(target_state, dtype=float).reshape(-1)

    err_norm = np.linalg.norm(controlled - target_state.reshape(1, -1), axis=1)

    fig, ax = plt.subplots(figsize=(15, 4.5))

    ax.set_title(f"{tag} | target-tracking error", fontsize=15, fontweight="bold")
    ax.plot(times, err_norm, linewidth=1.4, label="||y(t) - target||")
    ax.axvline(control_start_time, linewidth=1.1, linestyle="--", label="Control start")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error norm")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")

    plt.tight_layout()
    _save(fig, output_dir, "control_error.png")


def plot_control_sweep_summary(
    rows: list[dict],
    output_dir: str,
    tag: str = "Linear feedback sweep",
):
    if not rows:
        return

    rows = sorted(rows, key=lambda r: float(r.get("K", 0.0)))

    ks = np.asarray([float(r.get("K", 0.0)) for r in rows], dtype=float)
    rmse_state = np.asarray([float(r.get("target_rmse_state", np.nan)) for r in rows], dtype=float)
    spike_red = np.asarray([float(r.get("spike_reduction_percent", np.nan)) for r in rows], dtype=float)
    energy = np.asarray([float(r.get("control_energy_mean", np.nan)) for r in rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(tag, fontsize=16, fontweight="bold", y=0.98)

    axes[0].plot(ks, rmse_state, marker="o", linewidth=1.5)
    axes[0].set_ylabel("Target RMSE")
    axes[0].grid(True, linestyle="--", alpha=0.25)

    axes[1].plot(ks, spike_red, marker="o", linewidth=1.5)
    axes[1].set_ylabel("Spike reduction (%)")
    axes[1].grid(True, linestyle="--", alpha=0.25)

    axes[2].plot(ks, energy, marker="o", linewidth=1.5)
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Mean control energy")
    axes[2].grid(True, linestyle="--", alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, output_dir, "control_sweep_summary.png")
