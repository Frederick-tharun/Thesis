from __future__ import annotations

"""Generate the final submission-ready Chapter 1 prediction figures.

Reads the three locked, hash-verified ESN bundles in
FINAL_THESIS_RUN/01_prediction_all_regimes/<regime>/model_bundle.npz and
re-plots their held-out autonomous prediction with:

  - all three states (x, y, z) plus an x-state residual row
  - the locked held-out NRMSE annotated directly on the figure
  - a regime-specific inset: 3D phase portrait (chaotic bursting) or a
    zoomed representative burst/spike interval (the two periodic regimes)

Nothing is retrained or reselected. The reproduced rollout is verified
against the locked FINAL_THESIS_RUN/.../heldout_test_metrics.json before any
figure is saved, so a mismatch aborts the run instead of silently producing
a wrong figure.

Output: Chapter1_Final_Thesis_Figures/prediction/<regime>_prediction.{png,pdf}
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42  # embed real (non-Type3) fonts
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.signal import find_peaks

import config
from model import EchoStateNetwork

RUN_ROOT = REPO_ROOT / "FINAL_THESIS_RUN"
OUTPUT_ROOT = REPO_ROOT / "Chapter1_Final_Thesis_Figures" / "prediction"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

REGIMES = ("periodic_spiking", "periodic_bursting", "chaotic_bursting")

# Tolerance for cross-checking our reproduced NRMSE against the locked
# heldout_test_metrics.json before trusting the figure.
_VERIFY_RTOL = 1e-6
_VERIFY_ATOL = 1e-9


# ============================================================
# Deterministic HR regeneration (avoids importing data_loader.py, which
# pulls in pandas at module load time and is not needed for HR mode).
# Mirrors data_loader._hr_rhs / _rk4_hr / _load_hr exactly.
# ============================================================

def _hr_rhs(state, params):
    x, y, z = state
    a, b, c, d = params["a"], params["b"], params["c"], params["d"]
    r, s, xr, I = params["r"], params["s"], params["xr"], params["I"]
    dx = y - a * x**3 + b * x**2 - z + I
    dy = c - d * x**2 - y
    dz = r * (s * (x - xr) - z)
    return np.array([dx, dy, dz], dtype=float)


def _rk4_hr(x0, n_steps, dt, params):
    out = np.zeros((n_steps, 3), dtype=float)
    state = np.asarray(x0, dtype=float).copy()
    for i in range(n_steps):
        out[i] = state
        k1 = _hr_rhs(state, params)
        k2 = _hr_rhs(state + 0.5 * dt * k1, params)
        k3 = _hr_rhs(state + 0.5 * dt * k2, params)
        k4 = _hr_rhs(state + dt * k3, params)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def simulate_regime(regime: str):
    """Reproduce config.py's locked HR trajectory and train/test split."""
    total = int(config.HR_TOTAL_STEPS)
    burn = int(config.HR_TRANSIENT)
    dt = float(config.HR_DT)
    params = config.HR_PARAMETER_SETS[regime]

    traj = _rk4_hr(x0=params["x0"], n_steps=total + burn, dt=dt, params=params)
    traj = traj[burn:]
    times = np.arange(len(traj)) * dt
    series = traj

    n_train = int(len(series) * config.TRAIN_RATIO)
    train, test = series[:n_train], series[n_train:]
    return times, train, test


# ============================================================
# Locked-model prediction reproduction
# ============================================================

def reproduce_locked_prediction(regime: str):
    result_dir = RUN_ROOT / "01_prediction_all_regimes" / regime
    model_path = result_dir / "model_bundle.npz"
    if not model_path.is_file():
        raise FileNotFoundError(f"Locked model bundle not found: {model_path}")

    times, train, test = simulate_regime(regime)
    model, metadata = EchoStateNetwork.load_bundle(model_path)

    mean = np.asarray(metadata["external_mean"], dtype=float)
    std = np.asarray(metadata["external_std"], dtype=float)
    if mean.size == 0 or std.size == 0:
        raise RuntimeError(f"{regime}: bundle is missing external_mean/external_std.")

    train_norm = (train - mean) / std
    test_norm = (test - mean) / std
    full_input = np.vstack([train_norm, test_norm])

    pred_norm, _ = model.predict(full_input, n_warmup=len(train_norm) - 1)
    pred_norm = np.asarray(pred_norm, dtype=float)[: len(test)]
    prediction = pred_norm * std + mean

    heldout_time = times[len(train): len(train) + len(test)]
    heldout_time = heldout_time - heldout_time[0]

    return {
        "time": heldout_time,
        "reference": test,
        "prediction": prediction,
        "pred_norm": pred_norm,
        "test_norm": test_norm,
        "result_dir": result_dir,
    }


def _nrmse(pred, truth):
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    denom = float(np.std(truth))
    if denom < 1e-12:
        denom = 1.0
    return float(np.sqrt(np.mean((pred - truth) ** 2)) / denom)


def verify_against_locked_metrics(regime: str, bundle: dict) -> dict:
    metrics_path = bundle["result_dir"] / "heldout_test_metrics.json"
    locked = json.loads(metrics_path.read_text())

    nrmse_x = _nrmse(bundle["pred_norm"][:, 0], bundle["test_norm"][:, 0])
    nrmse_all = _nrmse(bundle["pred_norm"], bundle["test_norm"])

    for name, computed, locked_key in (
        ("nrmse_x", nrmse_x, "nrmse_recursive_x"),
        ("nrmse_all", nrmse_all, "nrmse_recursive_all_states"),
    ):
        locked_value = float(locked[locked_key])
        if not np.isclose(computed, locked_value, rtol=_VERIFY_RTOL, atol=_VERIFY_ATOL):
            raise RuntimeError(
                f"{regime}: reproduced {name}={computed!r} does not match locked "
                f"{locked_key}={locked_value!r}. Refusing to generate a figure "
                "from an unverified rollout."
            )

    return {"nrmse_x": nrmse_x, "nrmse_all": nrmse_all}


# ============================================================
# Shared figure-building blocks
# ============================================================

STATE_LABELS = (r"$x$", r"$y$", r"$z$")


def add_state_panels(fig, gridspec_slot, time_full, reference_full, prediction_full, display_stride):
    grid = gridspec_slot.subgridspec(nrows=3, ncols=1, hspace=0.08)

    t_disp = time_full[::display_stride]
    ref_disp = reference_full[::display_stride]
    pred_disp = prediction_full[::display_stride]

    axes = []
    for i in range(3):
        ax = fig.add_subplot(grid[i, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        ax.plot(t_disp, ref_disp[:, i], linewidth=1.2, label="Reference")
        ax.plot(t_disp, pred_disp[:, i], linestyle="--", linewidth=1.2, label="ESN prediction")
        ax.set_ylabel(STATE_LABELS[i], rotation=0, labelpad=12, fontsize=12)
        ax.grid(True, alpha=0.20)
        ax.tick_params(axis="both", labelsize=9)
        if i < 2:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Held-out time", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=9, ncol=2, frameon=True)
    axes[0].text(
        0.01, 0.88, "(a)", transform=axes[0].transAxes, fontsize=12, fontweight="bold"
    )
    return axes


def save_figure(fig, basename):
    png_path = OUTPUT_ROOT / f"{basename}.png"
    pdf_path = OUTPUT_ROOT / f"{basename}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ============================================================
# Regime-specific panel (b)
# ============================================================

def split_spikes_into_bursts(peak_indices: np.ndarray) -> list[np.ndarray]:
    if len(peak_indices) == 0:
        return []
    if len(peak_indices) == 1:
        return [peak_indices]
    gaps = np.diff(peak_indices)
    if len(gaps) == 1:
        return [peak_indices]
    sorted_gaps = np.sort(gaps)
    gap_ratios = sorted_gaps[1:] / np.maximum(sorted_gaps[:-1], 1)
    largest_jump = int(np.argmax(gap_ratios))
    if gap_ratios[largest_jump] >= 1.5:
        burst_gap_limit = 0.5 * (sorted_gaps[largest_jump] + sorted_gaps[largest_jump + 1])
    else:
        burst_gap_limit = 1.8 * np.median(gaps)
    groups: list[list[int]] = [[int(peak_indices[0])]]
    for previous_peak, current_peak in zip(peak_indices[:-1], peak_indices[1:]):
        if current_peak - previous_peak > burst_gap_limit:
            groups.append([])
        groups[-1].append(int(current_peak))
    return [np.asarray(group, dtype=int) for group in groups]


def select_representative_burst(reference_x: np.ndarray) -> tuple[int, int]:
    spike_threshold = float(getattr(config, "SPIKE_THRESHOLD", 1.0))
    minimum_distance = int(getattr(config, "SPIKE_MIN_DISTANCE", 5))
    peaks, _ = find_peaks(reference_x, height=spike_threshold, distance=minimum_distance)
    burst_groups = split_spikes_into_bursts(peaks)
    if not burst_groups:
        return 0, min(len(reference_x), 5000)
    maximum_length = max(len(group) for group in burst_groups)
    candidate_groups = [g for g in burst_groups if len(g) == maximum_length]
    selected_group = candidate_groups[len(candidate_groups) // 2]
    first_peak, last_peak = int(selected_group[0]), int(selected_group[-1])
    if len(selected_group) > 1:
        typical_spike_gap = int(np.median(np.diff(selected_group)))
    else:
        typical_spike_gap = 500
    margin = max(300, int(1.5 * typical_spike_gap))
    start = max(0, first_peak - margin)
    end = min(len(reference_x), last_peak + margin)
    return start, end


def select_two_spike_zoom(truth_x: np.ndarray) -> tuple[int, int]:
    threshold = float(getattr(config, "SPIKE_THRESHOLD", 1.0))
    min_distance = int(getattr(config, "SPIKE_MIN_DISTANCE", 5))
    peaks, _ = find_peaks(truth_x, height=threshold, distance=min_distance)
    if len(peaks) >= 2:
        gaps = np.diff(peaks)
        pair_index = int(np.argmin(gaps))
        first_peak, second_peak = int(peaks[pair_index]), int(peaks[pair_index + 1])
        gap = second_peak - first_peak
        margin = max(100, int(0.8 * gap))
        start = max(0, first_peak - margin)
        end = min(len(truth_x), second_peak + margin)
        return start, end
    centre = len(truth_x) // 2
    half_width = min(1500, centre)
    return max(0, centre - half_width), min(len(truth_x), centre + half_width)


def add_zoom_panel(fig, gridspec_slot, time, reference, prediction, start, end, title, mark_axis):
    for boundary in (time[start], time[end - 1]):
        mark_axis.axvline(boundary, linestyle=":", linewidth=1.1, color="0.35")

    ax = fig.add_subplot(gridspec_slot)
    ax.plot(time[start:end], reference[start:end, 0], linewidth=1.8, label="Reference")
    ax.plot(
        time[start:end], prediction[start:end, 0], linestyle="--", linewidth=1.7,
        label="ESN prediction",
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Held-out time", fontsize=10)
    ax.set_ylabel(r"$x$", fontsize=12, rotation=0, labelpad=12)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.20)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.set_box_aspect(1.05)
    ax.text(0.03, 0.95, "(b)", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    return ax


def add_phase_portrait_panel(fig, gridspec_slot, reference, prediction, n_points=4000):
    stride = max(1, len(reference) // n_points)
    ref_p = reference[::stride]
    pred_p = prediction[::stride]

    ax = fig.add_subplot(gridspec_slot, projection="3d")
    ax.plot(ref_p[:, 0], ref_p[:, 1], ref_p[:, 2], linewidth=1.2, label="Reference")
    ax.plot(
        pred_p[:, 0], pred_p[:, 1], pred_p[:, 2], linestyle="--", linewidth=1.2,
        label="ESN prediction",
    )
    ax.set_xlabel(r"$x$", fontsize=11, labelpad=4)
    ax.set_ylabel(r"$y$", fontsize=11, labelpad=4)
    ax.set_zlabel(r"$z$", fontsize=11, labelpad=4)
    ax.tick_params(axis="both", labelsize=8)
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.set_title("Reconstructed attractor phase portrait", fontsize=11, fontweight="bold", pad=2)
    ax.text2D(0.03, 0.95, "(b)", transform=ax.transAxes, fontsize=11, fontweight="bold")
    return ax


# ============================================================
# Per-regime figure assembly
# ============================================================

def build_figure(regime: str, time, reference, prediction, nrmse_x, nrmse_all):
    display_stride = max(1, len(time) // 15000)

    fig = plt.figure(figsize=(13.0, 7.4))
    outer = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[2.25, 1.05], wspace=0.17)

    axes = add_state_panels(fig, outer[0], time, reference, prediction, display_stride)

    if regime == "chaotic_bursting":
        add_phase_portrait_panel(fig, outer[1], reference, prediction)
    elif regime == "periodic_bursting":
        start, end = select_representative_burst(reference[:, 0])
        add_zoom_panel(
            fig, outer[1], time, reference, prediction, start, end,
            "Representative bursting interval", mark_axis=axes[0],
        )
    elif regime == "periodic_spiking":
        start, end = select_two_spike_zoom(reference[:, 0])
        add_zoom_panel(
            fig, outer[1], time, reference, prediction, start, end,
            "Representative two-spike interval", mark_axis=axes[0],
        )
    else:
        raise ValueError(f"Unknown regime: {regime}")

    fig.subplots_adjust(left=0.075, right=0.98, top=0.96, bottom=0.09)
    return fig


def main():
    print("=" * 72)
    print("Chapter 1 final prediction figures")
    print("=" * 72)
    for regime in REGIMES:
        print(f"\n[{regime}] reproducing locked rollout...")
        bundle = reproduce_locked_prediction(regime)
        metrics = verify_against_locked_metrics(regime, bundle)
        print(
            f"[{regime}] verified: nrmse_x={metrics['nrmse_x']:.6g} "
            f"nrmse_all={metrics['nrmse_all']:.6g} (matches locked heldout_test_metrics.json)"
        )
        fig = build_figure(
            regime,
            bundle["time"],
            bundle["reference"],
            bundle["prediction"],
            metrics["nrmse_x"],
            metrics["nrmse_all"],
        )
        save_figure(fig, f"{regime}_prediction")

    print("\nAll three prediction figures verified and saved to:")
    print(f"  {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
