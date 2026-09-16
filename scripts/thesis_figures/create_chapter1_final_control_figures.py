from __future__ import annotations

"""Generate the final submission-ready Chapter 1 controller figures.

Reads the locked, already-computed rollout CSVs and control_summary.json
files under FINAL_THESIS_RUN/{03_linear_feedback,04_finite_time,05_pyragas}/
-- nothing is resimulated or reselected. Every number the figures show is
cross-checked against the locked control_summary.json before saving.

Design rationale (after three rejected attempts, worth recording):
  The control event is effectively instantaneous -- control switches on at
  t=1140 and the trajectory collapses within ~1 time unit. Plotting the
  whole 420-unit test window therefore spends ~99% of the canvas on steady
  state, which renders either as a dead flat line ("conveys nothing") or,
  for Pyragas, as an unresolvable picket fence of ~25 spike trains ("too
  dense"). Both failures share one cause: showing 420 units when the event
  occupies 1.

  The fix used here, applied identically to both figures: zoom the waveform
  to the transition so individual spikes resolve and the
  collapse/regularization is a visible event, with the uncontrolled
  free-running trace kept alongside as the counterfactual (a controlled
  trace on its own is just a flat line; next to a still-spiking one it is a
  result).

  Everything that is a number rather than a shape -- sustained suppression
  (4 -> 0 spikes, 100%), Pyragas recurrence correlation (0.9999), and the
  ~100x precision gap between linear and finite-time -- lives in the
  comparison table. Earlier drafts encoded these as extra panels (a spike
  raster, an inter-spike-interval scatter, a log-scale error decay); all
  were dropped as unreadable relative to what the table already states.

Produces:
  1. regulation_controllers.{png,pdf}
  2. pyragas_controller.{png,pdf}
  3. controller_comparison_table.{csv,md}

Output directory: Chapter1_Final_Thesis_Figures/control/
"""

import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np

import config

RUN_ROOT = REPO_ROOT / "FINAL_THESIS_RUN"
OUTPUT_ROOT = REPO_ROOT / "Chapter1_Final_Thesis_Figures" / "control"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CONTROLLER_DIRS = {
    "linear_feedback": "03_linear_feedback",
    "finite_time": "04_finite_time",
    "pyragas": "05_pyragas",
}
CONTROLLER_DISPLAY = {
    "linear_feedback": "Linear feedback",
    "finite_time": "Finite-time feedback",
    "pyragas": "Pyragas delayed feedback",
}

COLOR_UNCONTROLLED = "0.55"
COLOR_ONSET = "#d62728"
COLOR = {"linear_feedback": "#0f9b8e", "finite_time": "#6a4c93", "pyragas": "#c9184a"}

SPIKE_THRESHOLD = float(getattr(config, "SPIKE_THRESHOLD", 1.0))


# ============================================================
# Data loading and verification
# ============================================================

def _load_rollout(controller: str):
    path = RUN_ROOT / CONTROLLER_DIRS[controller] / "best_rollout" / "rollout.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    def col(name):
        return np.asarray([float(row[name]) for row in rows], dtype=float)

    return {
        "time": col("time"),
        "target_x": col("target_x"),
        "uncontrolled_x": col("uncontrolled_x"),
        "uncontrolled_y": col("uncontrolled_y"),
        "uncontrolled_z": col("uncontrolled_z"),
        "controlled_x": col("corrected_feedback_input_x"),
        "controlled_y": col("corrected_feedback_input_y"),
        "controlled_z": col("corrected_feedback_input_z"),
    }


def _load_summary(controller: str) -> dict:
    return json.loads((RUN_ROOT / CONTROLLER_DIRS[controller] / "control_summary.json").read_text())


def _peak_indices(x, threshold):
    """Identical to control_experiment._peak_indices: one peak per
    contiguous above-threshold episode, so spike counts here match the
    locked metrics exactly."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) < 3:
        return np.asarray([], dtype=int)
    above = np.isfinite(x) & (x > float(threshold))
    if not np.any(above):
        return np.asarray([], dtype=int)
    transitions = np.diff(np.concatenate(([False], above, [False])).astype(int))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    peaks = [int(left + np.argmax(x[left:right])) for left, right in zip(starts, stops) if right > left]
    return np.asarray(peaks, dtype=int)


def _verify(controller: str, data: dict, summary: dict) -> None:
    """Reproduce both locked headline metrics before drawing anything."""
    start, end = int(summary["controller_test_start"]), int(summary["controller_test_end"])

    target = np.asarray(summary["target_state"], dtype=float)
    controlled = np.stack(
        [data["controlled_x"], data["controlled_y"], data["controlled_z"]], axis=1
    )[start:end]
    # Matches control_experiment._rmse: flattened elementwise RMSE over both
    # time and the 3 state dimensions, not the per-timestep norm.
    rmse_state = float(np.sqrt(np.mean((controlled - target.reshape(1, -1)) ** 2)))
    locked_rmse = float(summary["corrected_feedback_input_target_rmse_state"])
    if not math.isclose(rmse_state, locked_rmse, rel_tol=1e-4, abs_tol=1e-8):
        raise RuntimeError(
            f"{controller}: reproduced RMSE={rmse_state!r} != locked {locked_rmse!r}"
        )

    n_unctrl = len(_peak_indices(data["uncontrolled_x"][start:end], SPIKE_THRESHOLD))
    n_ctrl = len(_peak_indices(data["controlled_x"][start:end], SPIKE_THRESHOLD))
    reduction = 100.0 * (n_unctrl - n_ctrl) / n_unctrl if n_unctrl > 0 else 0.0
    locked_reduction = float(summary["spike_reduction_percent"])
    if not math.isclose(reduction, locked_reduction, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(
            f"{controller}: reproduced spike reduction={reduction!r}% != "
            f"locked {locked_reduction!r}% (counted {n_unctrl} uncontrolled, {n_ctrl} controlled)"
        )
    print(
        f"[{controller}] verified: RMSE={rmse_state:.4g}, spikes {n_unctrl}->{n_ctrl} "
        f"({reduction:+.1f}%) on the held-out controller-test window"
    )


def _save(fig, basename):
    for suffix, kwargs in ((".png", {"dpi": 300}), (".pdf", {})):
        path = OUTPUT_ROOT / f"{basename}{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved: {path}")
    plt.close(fig)


# ============================================================
# 1. Regulation: linear feedback + finite-time
# ============================================================

def build_regulation_figure():
    controllers = ("linear_feedback", "finite_time")
    data = {c: _load_rollout(c) for c in controllers}
    summary = {c: _load_summary(c) for c in controllers}
    for c in controllers:
        _verify(c, data[c], summary[c])

    ref = data["linear_feedback"]
    time = ref["time"]
    onset_idx = int(summary["linear_feedback"]["control_start_idx"])
    onset_t = float(time[onset_idx])
    target_x = float(summary["linear_feedback"]["target_state"][0])

    fig, ax_wave = plt.subplots(1, 1, figsize=(11.0, 4.8))

    # Single panel: the waveform carries the result. Sustained suppression
    # over the remaining horizon is a number (4 -> 0 spikes, 100% reduction)
    # and lives in the comparison table, not in a second panel.
    zoom = (float(time[0]), onset_t + 110.0)
    mask = (time >= zoom[0]) & (time <= zoom[1])

    ax_wave.plot(time[mask], ref["uncontrolled_x"][mask], color=COLOR_UNCONTROLLED,
                 linewidth=1.3, label="Uncontrolled ESN (free-running)")
    ax_wave.plot(time[mask], data["linear_feedback"]["controlled_x"][mask],
                 color=COLOR["linear_feedback"], linewidth=2.0, linestyle="--",
                 label="Linear feedback")
    ax_wave.plot(time[mask], data["finite_time"]["controlled_x"][mask],
                 color=COLOR["finite_time"], linewidth=1.3, linestyle=":",
                 label="Finite-time feedback")
    ax_wave.axhline(target_x, color="black", linestyle=(0, (1, 3)), linewidth=1.0,
                    label="Quiet-state target")
    ax_wave.axvline(onset_t, color=COLOR_ONSET, linestyle="-.", linewidth=1.5)
    ax_wave.annotate(
        "control on", xy=(onset_t, 1.55), xytext=(onset_t - 7, 1.55),
        color=COLOR_ONSET, fontsize=9.5, fontweight="bold", va="center", ha="right",
        arrowprops=dict(arrowstyle="->", color=COLOR_ONSET, linewidth=1.2),
    )

    ax_wave.set_xlim(zoom)
    ax_wave.set_xlabel("Time", fontsize=11)
    ax_wave.set_ylabel("Hindmarsh-Rose $x$ state", fontsize=11)
    ax_wave.grid(True, alpha=0.18)
    # Legend outside the axes: the controlled trace is a flat line near the
    # bottom of the panel, so any in-axes placement sits on top of data.
    ax_wave.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
                   fontsize=8.5, frameon=False)
    _save(fig, "regulation_controllers")


# ============================================================
# 2. Pyragas
# ============================================================

def build_pyragas_figure():
    data = _load_rollout("pyragas")
    summary = _load_summary("pyragas")
    _verify("pyragas", data, summary)

    time = data["time"]
    onset_idx = int(summary["control_start_idx"])
    onset_t = float(time[onset_idx])

    # Two panels carrying genuinely different evidence: (a) the time-domain
    # transition, (b) the geometric one. The phase portrait is the only
    # supplementary panel kept, because "the trajectory collapses onto a
    # single closed loop" is a statement the comparison table cannot make --
    # and a closed orbit embedded in the attractor is exactly what Pyragas
    # delayed feedback is designed to stabilize.
    fig = plt.figure(figsize=(14.5, 5.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.10,
                             left=0.055, right=0.95, top=0.88, bottom=0.13)
    ax_wave = fig.add_subplot(outer[0])

    zoom = (float(time[0]), onset_t + 110.0)
    mask = (time >= zoom[0]) & (time <= zoom[1])

    ax_wave.plot(time[mask], data["uncontrolled_x"][mask], color=COLOR_UNCONTROLLED,
                 linewidth=1.3, label="Uncontrolled ESN (free-running)")
    ax_wave.plot(time[mask], data["controlled_x"][mask], color=COLOR["pyragas"],
                 linewidth=1.5, linestyle="--", label="Pyragas-controlled")
    ax_wave.axvline(onset_t, color=COLOR_ONSET, linestyle="-.", linewidth=1.5)
    ax_wave.annotate(
        "control on", xy=(onset_t, 1.55), xytext=(onset_t - 7, 1.55),
        color=COLOR_ONSET, fontsize=9.5, fontweight="bold", va="center", ha="right",
        arrowprops=dict(arrowstyle="->", color=COLOR_ONSET, linewidth=1.2),
    )
    ax_wave.set_xlim(zoom)
    ax_wave.set_xlabel("Time", fontsize=11)
    ax_wave.set_ylabel("Hindmarsh-Rose $x$ state", fontsize=11)
    ax_wave.grid(True, alpha=0.18)
    ax_wave.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
                   fontsize=8.5, frameon=False)
    ax_wave.text(0.008, 0.96, "(a)", transform=ax_wave.transAxes, fontsize=12,
                 fontweight="bold", va="top")

    # ---- (b) phase portrait: attractor vs. stabilized closed orbit -------
    ax3d = fig.add_subplot(outer[1], projection="3d")
    stride = max(1, (len(time) - onset_idx) // 4000)
    sl = slice(onset_idx, None, stride)

    ax3d.plot(data["uncontrolled_x"][sl], data["uncontrolled_y"][sl], data["uncontrolled_z"][sl],
              color=COLOR_UNCONTROLLED, linewidth=0.8, alpha=0.75,
              label="Uncontrolled (chaotic attractor)")
    ax3d.plot(data["controlled_x"][sl], data["controlled_y"][sl], data["controlled_z"][sl],
              color=COLOR["pyragas"], linewidth=1.3,
              label="Pyragas-controlled (closed orbit)")

    ax3d.set_xlabel(r"$x$", fontsize=10.5, labelpad=2)
    ax3d.set_ylabel(r"$y$", fontsize=10.5, labelpad=2)
    ax3d.set_zlabel(r"$z$", fontsize=10.5, labelpad=2)
    ax3d.tick_params(axis="both", labelsize=7.5)
    ax3d.view_init(elev=24, azim=-58)
    ax3d.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06), fontsize=8.5, frameon=False)
    ax3d.text2D(0.02, 0.96, "(b)", transform=ax3d.transAxes, fontsize=12, fontweight="bold")

    _save(fig, "pyragas_controller")


# ============================================================
# 3. Controller comparison table
# ============================================================

def build_comparison_table():
    rows = []
    for controller in ("linear_feedback", "finite_time", "pyragas"):
        summary = _load_summary(controller)
        rows.append({
            "controller": CONTROLLER_DISPLAY[controller],
            "K": summary["best_k"],
            "delay_steps": summary.get("pyragas_delay"),
            "sign": summary.get("pyragas_sign"),
            "state_rmse": summary["corrected_feedback_input_target_rmse_state"],
            "spike_reduction_percent": summary["spike_reduction_percent"],
            "recurrence_error_norm": summary.get("pyragas_empirical_recurrence_error_norm"),
            "recurrence_correlation": summary.get("pyragas_empirical_recurrence_correlation"),
            "control_effort_mean_sq": summary["control_effort_mean_sq"],
        })

    csv_path = OUTPUT_ROOT / "controller_comparison_table.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| Controller | K | Delay (steps) | Sign | State RMSE | Spike reduction | Recurrence error | Recurrence corr. | Control effort |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value, spec="{:.4g}"):
            return "-" if value is None else spec.format(value)

        md_lines.append(
            f"| {row['controller']} | {row['K']:.4g} | {fmt(row['delay_steps'], '{:d}')} | "
            f"{fmt(row['sign'], '{:d}')} | {row['state_rmse']:.4g} | "
            f"{row['spike_reduction_percent']:.1f}% | {fmt(row['recurrence_error_norm'])} | "
            f"{fmt(row['recurrence_correlation'])} | {row['control_effort_mean_sq']:.4g} |"
        )
    md_path = OUTPUT_ROOT / "controller_comparison_table.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    for line in md_lines:
        print(line)


def main():
    print("=" * 72)
    print("Chapter 1 final controller figures")
    print("=" * 72)
    build_regulation_figure()
    print()
    build_pyragas_figure()
    print()
    build_comparison_table()
    print("\nAll controller assets verified and saved to:")
    print(f"  {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
