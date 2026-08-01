#!/usr/bin/env python3
"""Generate Chapter 2 HR diagnostics and cautious classifications."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .config_ch2 import (
        BURST_MIN_GAP_PROMINENCE,
        BURST_MIN_INTERVALS_PER_TIMESCALE,
        BURST_MIN_LOG_ISI_GAP,
        BURST_MIN_SPIKES,
        CONTINUOUS_CURRENT_SEQUENCE,
        DT,
        FIXED_COMPARISON_DURATION,
        FIXED_CURRENTS,
        HALF_WINDOW_CONSISTENCY_TOLERANCE,
        HR_PARAMETERS,
        INITIAL_STATE,
        INITIAL_TRANSIENT_STEPS,
        LYAPUNOV_CHECKPOINT_STEPS,
        LYAPUNOV_CONVERGENCE_ABSOLUTE_TOLERANCE,
        LYAPUNOV_CONVERGENCE_RELATIVE_TOLERANCE,
        LYAPUNOV_ESTIMATION_STEPS,
        LYAPUNOV_RENORMALIZATION_STEPS,
        REGULAR_INTERBURST_INTERVAL_CV_MAX,
        REGULAR_SPIKES_PER_BURST_CV_MAX,
        REGULAR_WITHIN_BURST_ISI_CV_MAX,
        RETAINED_SAMPLES_PER_CURRENT,
        SPIKE_HEIGHT,
        SPIKE_MIN_DISTANCE_STEPS,
        SPIKE_PROMINENCE,
        SWITCH_POST_DURATION,
        SWITCH_PRE_DURATION,
    )
    from .dynamics_analysis_ch2 import (
        LyapunovAnalysis,
        HalfWindowConsistency,
        SpikeBurstAnalysis,
        analyze_spikes_and_bursts,
        estimate_lyapunov,
        evaluate_half_window_consistency,
        preliminary_regime,
    )
    from .hr_data_ch2 import (
        HRTrajectory,
        load_trajectory_npz,
        save_trajectory_npz,
        simulate_continuous_currents,
        simulate_fixed_current,
    )
except ImportError:  # Support ``python chapter2/generate_diagnostics.py``.
    from config_ch2 import (
        BURST_MIN_GAP_PROMINENCE,
        BURST_MIN_INTERVALS_PER_TIMESCALE,
        BURST_MIN_LOG_ISI_GAP,
        BURST_MIN_SPIKES,
        CONTINUOUS_CURRENT_SEQUENCE,
        DT,
        FIXED_COMPARISON_DURATION,
        FIXED_CURRENTS,
        HALF_WINDOW_CONSISTENCY_TOLERANCE,
        HR_PARAMETERS,
        INITIAL_STATE,
        INITIAL_TRANSIENT_STEPS,
        LYAPUNOV_CHECKPOINT_STEPS,
        LYAPUNOV_CONVERGENCE_ABSOLUTE_TOLERANCE,
        LYAPUNOV_CONVERGENCE_RELATIVE_TOLERANCE,
        LYAPUNOV_ESTIMATION_STEPS,
        LYAPUNOV_RENORMALIZATION_STEPS,
        REGULAR_INTERBURST_INTERVAL_CV_MAX,
        REGULAR_SPIKES_PER_BURST_CV_MAX,
        REGULAR_WITHIN_BURST_ISI_CV_MAX,
        RETAINED_SAMPLES_PER_CURRENT,
        SPIKE_HEIGHT,
        SPIKE_MIN_DISTANCE_STEPS,
        SPIKE_PROMINENCE,
        SWITCH_POST_DURATION,
        SWITCH_PRE_DURATION,
    )
    from dynamics_analysis_ch2 import (
        LyapunovAnalysis,
        HalfWindowConsistency,
        SpikeBurstAnalysis,
        analyze_spikes_and_bursts,
        estimate_lyapunov,
        evaluate_half_window_consistency,
        preliminary_regime,
    )
    from hr_data_ch2 import (
        HRTrajectory,
        load_trajectory_npz,
        save_trajectory_npz,
        simulate_continuous_currents,
        simulate_fixed_current,
    )


FIGURE_DPI = 180
LINE_WIDTH = 0.75
SUMMARY_FIELDS = (
    "current_I",
    "retained_samples",
    "transient_steps",
    "spike_count",
    "mean_isi",
    "isi_std",
    "isi_cv",
    "burst_structure",
    "burst_count",
    "mean_spikes_per_burst",
    "std_spikes_per_burst",
    "mean_within_burst_isi",
    "within_burst_isi_cv",
    "mean_interburst_interval",
    "interburst_interval_cv",
    "largest_lyapunov_exponent",
    "lyapunov_convergence",
    "lyapunov_classification",
    "half_window_consistency",
    "preliminary_regime",
    "notes",
)


def current_token(current: float) -> str:
    return f"{current:.2f}".replace(".", "p")


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def clear_old_png_figures(figure_dir: Path) -> list[Path]:
    """Delete only reproducible PNG files in the exact figure directory."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    removed = sorted(figure_dir.glob("*.png"))
    for path in removed:
        path.unlink()
    return removed


def plot_fixed_comparison(
    trajectories: dict[float, HRTrajectory],
    path: Path,
) -> None:
    duration_samples = int(round(FIXED_COMPARISON_DURATION / DT))
    starts = {
        current: max(0, len(trajectory.t) - duration_samples)
        for current, trajectory in trajectories.items()
    }
    visible_x = np.concatenate(
        [
            trajectories[current].x[starts[current] :]
            for current in FIXED_CURRENTS
        ]
    )
    padding = 0.05 * float(np.ptp(visible_x))
    y_limits = (float(np.min(visible_x) - padding), float(np.max(visible_x) + padding))
    reference = trajectories[FIXED_CURRENTS[0]]
    x_limits = (
        float(reference.t[-duration_samples]),
        float(reference.t[-1]),
    )

    fig, axes = plt.subplots(
        len(FIXED_CURRENTS),
        1,
        figsize=(13, 12),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for ax, current in zip(axes, FIXED_CURRENTS):
        trajectory = trajectories[current]
        start = starts[current]
        ax.plot(
            trajectory.t[start:],
            trajectory.x[start:],
            color="#1f77b4",
            linewidth=LINE_WIDTH,
        )
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_ylabel("x(t)")
        ax.text(
            0.012,
            0.88,
            f"I = {current:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.grid(alpha=0.22)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Settled fixed-current Hindmarsh–Rose dynamics", fontsize=15)
    _save_figure(fig, path)


def _draw_switch_lines(
    axes: Iterable[plt.Axes],
    trajectory: HRTrajectory,
    switch_indices: np.ndarray,
) -> None:
    for switch_index in switch_indices:
        switch_time = trajectory.t[int(switch_index)]
        for ax in axes:
            ax.axvline(
                switch_time,
                color="black",
                linestyle="--",
                linewidth=0.8,
                alpha=0.8,
            )


def plot_continuous_combined(
    trajectory: HRTrajectory,
    switch_indices: np.ndarray,
    path: Path,
) -> None:
    fig, (ax_i, ax_x) = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
        constrained_layout=True,
    )
    ax_i.step(trajectory.t, trajectory.I, where="post", color="#d62728")
    ax_i.set_ylabel("I(t)")
    ax_i.grid(alpha=0.25)
    ax_x.plot(trajectory.t, trajectory.x, linewidth=0.55, color="#1f77b4")
    ax_x.set(xlabel="Time", ylabel="x(t)")
    ax_x.grid(alpha=0.25)
    _draw_switch_lines((ax_i, ax_x), trajectory, switch_indices)

    boundaries = np.concatenate(
        ([0], switch_indices, [len(trajectory.t)])
    )
    for segment, current in enumerate(CONTINUOUS_CURRENT_SEQUENCE):
        start = int(boundaries[segment])
        end = int(boundaries[segment + 1]) - 1
        midpoint_time = 0.5 * (trajectory.t[start] + trajectory.t[end])
        ax_i.text(
            midpoint_time,
            current + 0.035,
            f"I={current:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_x.text(
        0.01,
        0.04,
        "Single continuous state trajectory; no state resets",
        transform=ax_x.transAxes,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    fig.suptitle("Continuous current protocol and state response", fontsize=15)
    _save_figure(fig, path)


def plot_switches_comparison(
    trajectory: HRTrajectory,
    switch_indices: np.ndarray,
    path: Path,
) -> None:
    before = int(round(SWITCH_PRE_DURATION / DT))
    after = int(round(SWITCH_POST_DURATION / DT))
    fig, axes = plt.subplots(
        len(switch_indices),
        1,
        figsize=(13, 12),
        sharex=True,
        constrained_layout=True,
    )
    for number, (ax_x, switch_index) in enumerate(
        zip(axes, switch_indices), start=1
    ):
        switch = int(switch_index)
        start = max(0, switch - before)
        end = min(len(trajectory.t), switch + after + 1)
        switch_time = trajectory.t[switch]
        relative_time = trajectory.t[start:end] - switch_time
        old_current = trajectory.I[switch - 1]
        new_current = trajectory.I[switch]

        ax_i = ax_x.twinx()
        ax_x.plot(
            relative_time,
            trajectory.x[start:end],
            color="#1f77b4",
            linewidth=LINE_WIDTH,
            label="x(t)",
        )
        ax_i.step(
            relative_time,
            trajectory.I[start:end],
            where="post",
            color="#d62728",
            linewidth=1.1,
            alpha=0.8,
            label="I(t)",
        )
        ax_x.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax_x.set_ylabel("x(t)", color="#1f77b4")
        ax_i.set_ylabel("I(t)", color="#d62728")
        ax_x.set_title(
            f"Switch {number}: I={old_current:.2f} → I={new_current:.2f}",
            loc="left",
            fontsize=10,
        )
        ax_x.grid(alpha=0.22)
    axes[-1].set_xlabel("Time relative to switch")
    axes[-1].set_xlim(-SWITCH_PRE_DURATION, SWITCH_POST_DURATION)
    fig.suptitle(
        "State-continuous responses around all current switches",
        fontsize=15,
    )
    _save_figure(fig, path)


def plot_consistency_check(
    consistency: dict[float, HalfWindowConsistency],
    path: Path,
) -> None:
    labels = [f"{current:.2f}" for current in FIXED_CURRENTS]
    positions = np.arange(len(FIXED_CURRENTS), dtype=float)
    width = 0.23
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for state_index, (state_name, color) in enumerate(zip("xyz", colors)):
        axes[0].bar(
            positions + (state_index - 1) * width,
            [
                consistency[current].mean_shifts[state_index]
                for current in FIXED_CURRENTS
            ],
            width,
            label=state_name,
            color=color,
        )
        axes[1].bar(
            positions + (state_index - 1) * width,
            [
                consistency[current].std_shifts[state_index]
                for current in FIXED_CURRENTS
            ],
            width,
            label=state_name,
            color=color,
        )
    axes[2].bar(
        positions,
        [consistency[current].isi_relative_shift for current in FIXED_CURRENTS],
        width=0.55,
        color="#9467bd",
    )
    axes[0].set_ylabel("Normalized mean shift")
    axes[1].set_ylabel("Relative std. shift")
    axes[2].set_ylabel("Relative mean-ISI shift")
    axes[2].set_xlabel("Applied current I")
    axes[0].legend(ncol=3, title="State")
    for ax in axes:
        ax.axhline(
            HALF_WINDOW_CONSISTENCY_TOLERANCE,
            color="black",
            linestyle="--",
            linewidth=0.9,
            label="10% tolerance",
        )
        ax.set_xticks(positions, labels)
        ax.grid(axis="y", alpha=0.22)
    axes[2].legend(loc="upper right")
    fig.suptitle(
        "Half-window consistency check: first versus second retained half",
        fontsize=15,
    )
    _save_figure(fig, path)


def _fixed_statistics_row(
    current: float,
    trajectory: HRTrajectory,
    consistency: HalfWindowConsistency,
) -> dict[str, object]:
    midpoint = len(trajectory.t) // 2
    row: dict[str, object] = {
        "current": current,
        "n_samples": len(trajectory.t),
        "transient_steps": INITIAL_TRANSIENT_STEPS,
        "t_start": trajectory.t[0],
        "t_end": trajectory.t[-1],
    }
    for name in ("x", "y", "z"):
        values = getattr(trajectory, name)
        row[f"{name}_min"] = float(np.min(values))
        row[f"{name}_max"] = float(np.max(values))
        row[f"{name}_mean"] = float(np.mean(values))
        row[f"{name}_std"] = float(np.std(values, ddof=0))
        row[f"{name}_early_mean"] = float(np.mean(values[:midpoint]))
        row[f"{name}_late_mean"] = float(np.mean(values[midpoint:]))
        row[f"{name}_early_std"] = float(np.std(values[:midpoint], ddof=0))
        row[f"{name}_late_std"] = float(np.std(values[midpoint:], ddof=0))
    row["early_mean_isi"] = consistency.early_mean_isi
    row["late_mean_isi"] = consistency.late_mean_isi
    row["isi_relative_shift"] = consistency.isi_relative_shift
    row["half_window_consistency"] = consistency.result
    return row


def write_fixed_statistics(
    path: Path,
    trajectories: dict[float, HRTrajectory],
    consistency: dict[float, HalfWindowConsistency],
) -> None:
    rows = [
        _fixed_statistics_row(current, trajectories[current], consistency[current])
        for current in FIXED_CURRENTS
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(
    trajectories: dict[float, HRTrajectory],
    spikes: dict[float, SpikeBurstAnalysis],
    consistency: dict[float, HalfWindowConsistency],
    lyapunov: dict[float, LyapunovAnalysis],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for current in FIXED_CURRENTS:
        spike_result = spikes[current]
        consistency_result = consistency[current]
        lyapunov_result = lyapunov[current]
        notes = (
            f"{spike_result.notes}; {consistency_result.notes}; "
            f"Lyapunov {lyapunov_result.convergence}, "
            f"final-checkpoint tolerance={lyapunov_result.convergence_tolerance:.4g}"
        )
        rows.append(
            {
                "current_I": f"{current:.2f}",
                "retained_samples": len(trajectories[current].t),
                "transient_steps": INITIAL_TRANSIENT_STEPS,
                "spike_count": len(spike_result.spike_indices),
                "mean_isi": spike_result.mean_isi,
                "isi_std": spike_result.std_isi,
                "isi_cv": spike_result.isi_cv,
                "burst_structure": spike_result.burst_structure,
                "burst_count": (
                    float("nan")
                    if spike_result.burst_count is None
                    else spike_result.burst_count
                ),
                "mean_spikes_per_burst": spike_result.mean_spikes_per_burst,
                "std_spikes_per_burst": spike_result.std_spikes_per_burst,
                "mean_within_burst_isi": spike_result.mean_within_burst_isi,
                "within_burst_isi_cv": spike_result.within_burst_isi_cv,
                "mean_interburst_interval": spike_result.mean_interburst_interval,
                "interburst_interval_cv": spike_result.interburst_interval_cv,
                "largest_lyapunov_exponent": lyapunov_result.exponent,
                "lyapunov_convergence": lyapunov_result.convergence,
                "lyapunov_classification": lyapunov_result.classification,
                "half_window_consistency": consistency_result.result,
                "preliminary_regime": preliminary_regime(
                    spike_result,
                    lyapunov_result,
                ),
                "notes": notes,
            }
        )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_lyapunov_convergence_csv(
    path: Path,
    lyapunov: dict[float, LyapunovAnalysis],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("current_I", "evaluation_steps", "running_lle"),
            lineterminator="\n",
        )
        writer.writeheader()
        for current in FIXED_CURRENTS:
            result = lyapunov[current]
            for steps, running_lle in zip(
                result.checkpoint_steps, result.checkpoint_lle
            ):
                writer.writerow(
                    {
                        "current_I": f"{current:.2f}",
                        "evaluation_steps": int(steps),
                        "running_lle": float(running_lle),
                    }
                )


def _display(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.8g}" if np.isfinite(value) else "NaN"
    if value == "":
        return "NaN"
    return str(value).replace("|", "\\|")


def write_summary_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Chapter 2 Hindmarsh–Rose dynamics summary",
        "",
        "These labels are preliminary diagnostics, not final scientific classifications.",
        "",
        "| " + " | ".join(SUMMARY_FIELDS) + " |",
        "| " + " | ".join("---" for _ in SUMMARY_FIELDS) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_display(row[field]) for field in SUMMARY_FIELDS) + " |"
        )
    lines.extend(
        [
            "",
            "## Methods",
            "",
            f"Spikes are x peaks with height >= {SPIKE_HEIGHT}, prominence >= "
            f"{SPIKE_PROMINENCE}, and minimum distance "
            f"{SPIKE_MIN_DISTANCE_STEPS} steps ({SPIKE_MIN_DISTANCE_STEPS * DT:g} "
            "model-time units).",
            "",
            "The largest adjacent gap in sorted log-ISI values defines the adaptive "
            f"candidate split. It must be >= {BURST_MIN_LOG_ISI_GAP:g} log units, "
            f">= {BURST_MIN_GAP_PROMINENCE:g} times the median other positive gap, "
            f"and leave at least {BURST_MIN_INTERVALS_PER_TIMESCALE} intervals on "
            f"each side. Each accepted burst contains at least {BURST_MIN_SPIKES} "
            "spikes. A regular tonic train is not split; ambiguous structure remains "
            "uncertain.",
            "",
            "Periodic bursting requires the CV of within-burst ISIs, inter-burst "
            "intervals, and spikes per burst to be no greater than "
            f"{REGULAR_WITHIN_BURST_ISI_CV_MAX:.0%}, "
            f"{REGULAR_INTERBURST_INTERVAL_CV_MAX:.0%}, and "
            f"{REGULAR_SPIKES_PER_BURST_CV_MAX:.0%}, respectively. Overall ISI CV "
            "is not used to decide whether bursting is periodic.",
            "",
            "The half-window consistency check compares state means, state standard "
            f"deviations, and mean ISI using a {HALF_WINDOW_CONSISTENCY_TOLERANCE:.0%} "
            "tolerance. 'consistent' only means that the two retained halves have "
            "similar measurements. 'inconsistent' can also result from chaotic "
            "fluctuations or incomplete burst cycles and does not by itself show "
            "that the initial transient was insufficient.",
            "",
            f"Lyapunov estimates use a {INITIAL_TRANSIENT_STEPS}-step transient, "
            f"{LYAPUNOV_ESTIMATION_STEPS} evaluation steps, and tangent "
            f"renormalization every {LYAPUNOV_RENORMALIZATION_STEPS} steps. Running "
            f"estimates are retained at {', '.join(map(str, LYAPUNOV_CHECKPOINT_STEPS))}. "
            "The estimate is converged when both consecutive changes among the last "
            "three checkpoints are within the larger of "
            f"{LYAPUNOV_CONVERGENCE_ABSOLUTE_TOLERANCE:g} and "
            f"{LYAPUNOV_CONVERGENCE_RELATIVE_TOLERANCE:.0%} of the final estimate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diagnostic_files(output_dir: Path) -> list[Path]:
    """Return the complete canonical output set, excluding the manifest."""
    data_dir = output_dir / "data"
    figure_dir = output_dir / "figures"
    files = [
        data_dir / "continuous_switched_currents.npz",
        *(
            data_dir / f"fixed_I_{current_token(current)}.npz"
            for current in FIXED_CURRENTS
        ),
        output_dir / "fixed_current_statistics.csv",
        output_dir / "dynamics_summary.csv",
        output_dir / "dynamics_summary.md",
        output_dir / "lyapunov_convergence.csv",
        figure_dir / "fixed_currents_x_comparison.png",
        figure_dir / "continuous_I_and_x.png",
        figure_dir / "continuous_switches_comparison.png",
        figure_dir / "transient_settling_check.png",
    ]
    return sorted(files)


def write_manifest(
    path: Path,
    generated_files: Iterable[Path],
    output_dir: Path,
    switch_indices: np.ndarray,
    continuous_samples: int,
) -> None:
    files = sorted(Path(item) for item in generated_files)
    manifest = {
        "description": "Chapter 2 diagnostics; not a locked final test dataset",
        "equations_and_rk4": "Matched to Chapter 1 data_loader.py",
        "parameters": {
            "a": HR_PARAMETERS.a,
            "b": HR_PARAMETERS.b,
            "c": HR_PARAMETERS.c,
            "d": HR_PARAMETERS.d,
            "r": HR_PARAMETERS.r,
            "s": HR_PARAMETERS.s,
            "x_r": HR_PARAMETERS.x_r,
        },
        "initial_state": list(INITIAL_STATE),
        "dt": DT,
        "initial_transient_steps": INITIAL_TRANSIENT_STEPS,
        "transient_steps": INITIAL_TRANSIENT_STEPS,
        "retained_samples_per_current": RETAINED_SAMPLES_PER_CURRENT,
        "continuous_samples": int(continuous_samples),
        "fixed_currents": list(FIXED_CURRENTS),
        "continuous_current_sequence": list(CONTINUOUS_CURRENT_SEQUENCE),
        "continuous_switch_indices": switch_indices.tolist(),
        "switch_indices": switch_indices.tolist(),
        "lyapunov_evaluation_steps": LYAPUNOV_ESTIMATION_STEPS,
        "lyapunov_renormalization_interval": LYAPUNOV_RENORMALIZATION_STEPS,
        "lyapunov_checkpoints": list(LYAPUNOV_CHECKPOINT_STEPS),
        "spike_detection": {
            "height": SPIKE_HEIGHT,
            "prominence": SPIKE_PROMINENCE,
            "minimum_distance_steps": SPIKE_MIN_DISTANCE_STEPS,
        },
        "burst_detection": (
            f"largest adjacent gap in sorted log-ISIs; gap >= "
            f"{BURST_MIN_LOG_ISI_GAP:g}; prominence >= "
            f"{BURST_MIN_GAP_PROMINENCE:g}; at least "
            f"{BURST_MIN_INTERVALS_PER_TIMESCALE} intervals per timescale; "
            f"at least {BURST_MIN_SPIKES} spikes per burst"
        ),
        "consistency_relative_tolerance": HALF_WINDOW_CONSISTENCY_TOLERANCE,
        "lyapunov": {
            "method": "Chapter 1 Benettin tangent-linear RK4 implementation",
            "estimation_steps": LYAPUNOV_ESTIMATION_STEPS,
            "renormalization_steps": LYAPUNOV_RENORMALIZATION_STEPS,
            "checkpoint_steps": list(LYAPUNOV_CHECKPOINT_STEPS),
            "convergence_rule": (
                "both final checkpoint changes <= max(absolute tolerance, "
                "relative tolerance * abs(final estimate))"
            ),
        },
        "files": {
            str(file.relative_to(output_dir)): file_sha256(file) for file in files
        },
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")


def rebuild_manifest(output_dir: Path, continuous: HRTrajectory) -> Path:
    """Hash the finalized canonical outputs and write the manifest last."""
    files = diagnostic_files(output_dir)
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing diagnostic outputs: {missing}")
    switch_indices = np.flatnonzero(np.diff(continuous.I) != 0.0) + 1
    path = output_dir / "diagnostic_manifest.json"
    write_manifest(
        path,
        files,
        output_dir,
        switch_indices,
        len(continuous.t),
    )
    return path


def analyze_fixed_trajectories(
    fixed: dict[float, HRTrajectory],
) -> tuple[
    dict[float, SpikeBurstAnalysis],
    dict[float, HalfWindowConsistency],
    dict[float, LyapunovAnalysis],
]:
    spikes = {
        current: analyze_spikes_and_bursts(fixed[current])
        for current in FIXED_CURRENTS
    }
    consistency = {
        current: evaluate_half_window_consistency(fixed[current])
        for current in FIXED_CURRENTS
    }
    lyapunov: dict[float, LyapunovAnalysis] = {}
    for current in FIXED_CURRENTS:
        print(f"Estimating Lyapunov exponent for I={current:.2f}...", flush=True)
        lyapunov[current] = estimate_lyapunov(current)
    return spikes, consistency, lyapunov


def write_analysis_outputs(
    output_dir: Path,
    fixed: dict[float, HRTrajectory],
    spikes: dict[float, SpikeBurstAnalysis],
    consistency: dict[float, HalfWindowConsistency],
    lyapunov: dict[float, LyapunovAnalysis],
) -> list[Path]:
    rows = build_summary_rows(fixed, spikes, consistency, lyapunov)
    statistics_csv = output_dir / "fixed_current_statistics.csv"
    summary_csv = output_dir / "dynamics_summary.csv"
    summary_md = output_dir / "dynamics_summary.md"
    convergence_csv = output_dir / "lyapunov_convergence.csv"
    write_fixed_statistics(statistics_csv, fixed, consistency)
    write_summary_csv(summary_csv, rows)
    write_summary_markdown(summary_md, rows)
    write_lyapunov_convergence_csv(convergence_csv, lyapunov)
    return [statistics_csv, summary_csv, summary_md, convergence_csv]


def regenerate_analysis_outputs(output_dir: Path) -> list[Path]:
    """Regenerate analysis products from existing trajectories only."""
    data_dir = output_dir / "data"
    fixed = {
        current: load_trajectory_npz(
            data_dir / f"fixed_I_{current_token(current)}.npz"
        )
        for current in FIXED_CURRENTS
    }
    continuous = load_trajectory_npz(data_dir / "continuous_switched_currents.npz")
    spikes, consistency, lyapunov = analyze_fixed_trajectories(fixed)
    generated = write_analysis_outputs(
        output_dir, fixed, spikes, consistency, lyapunov
    )
    generated.append(rebuild_manifest(output_dir, continuous))
    return generated


def generate_all(output_dir: Path) -> tuple[list[Path], list[Path]]:
    data_dir = output_dir / "data"
    figure_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    removed_figures = clear_old_png_figures(figure_dir)

    generated: list[Path] = []
    fixed: dict[float, HRTrajectory] = {}
    for current in FIXED_CURRENTS:
        print(f"Simulating fixed current I={current:.2f}...", flush=True)
        trajectory = simulate_fixed_current(
            current,
            retained_samples=RETAINED_SAMPLES_PER_CURRENT,
            transient_steps=INITIAL_TRANSIENT_STEPS,
        )
        fixed[current] = trajectory
        data_path = data_dir / f"fixed_I_{current_token(current)}.npz"
        save_trajectory_npz(data_path, trajectory)
        generated.append(data_path)

    spikes, consistency, lyapunov = analyze_fixed_trajectories(fixed)
    generated.extend(
        write_analysis_outputs(output_dir, fixed, spikes, consistency, lyapunov)
    )

    fixed_figure = figure_dir / "fixed_currents_x_comparison.png"
    consistency_figure = figure_dir / "transient_settling_check.png"
    plot_fixed_comparison(fixed, fixed_figure)
    plot_consistency_check(consistency, consistency_figure)
    generated.extend((fixed_figure, consistency_figure))

    print("Simulating continuous switched-current trajectory...", flush=True)
    continuous, switch_indices = simulate_continuous_currents(
        CONTINUOUS_CURRENT_SEQUENCE,
        samples_per_segment=RETAINED_SAMPLES_PER_CURRENT,
        transient_steps=INITIAL_TRANSIENT_STEPS,
    )
    continuous_path = data_dir / "continuous_switched_currents.npz"
    save_trajectory_npz(continuous_path, continuous)
    generated.append(continuous_path)

    combined_figure = figure_dir / "continuous_I_and_x.png"
    switches_figure = figure_dir / "continuous_switches_comparison.png"
    plot_continuous_combined(continuous, switch_indices, combined_figure)
    plot_switches_comparison(continuous, switch_indices, switches_figure)
    generated.extend((combined_figure, switches_figure))

    manifest_path = rebuild_manifest(output_dir, continuous)
    generated.append(manifest_path)
    return generated, removed_figures


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Output directory (default: {default_output})",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Read existing NPZ files and regenerate only analysis outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.analysis_only:
        generated = regenerate_analysis_outputs(output_dir)
        print("Existing trajectory files were read but not regenerated.")
    else:
        generated, removed = generate_all(output_dir)
        print(f"Removed {len(removed)} old PNG files from {output_dir / 'figures'}")
    print(f"Generated {len(generated)} current diagnostic files in {output_dir}")
    for path in generated:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
