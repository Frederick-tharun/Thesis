"""Held-out chaotic-bursting figure for Section 2.4 of the thesis.

(a) Reference and ESN prediction for x, y, z over the full test horizon.
(b) Zoom on the final two spikes, where the traces visibly separate.
(c) Spike-timing offset of every detected spike, with the valid-prediction
    horizon marked.

The chaotic rollout is sensitive to floating-point differences between CPU
types. The script therefore checks the reproduced rollout against the saved
held-out metrics and stops if they do not match. Run it on a node of the
`work` partition (see run_final_thesis_pipeline.slurm).
"""

from __future__ import annotations

from pathlib import Path
import json
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np

import config
from final_pipeline import _heldout_metrics, _load_regime
from model import EchoStateNetwork


REGIME = "chaotic_bursting"
RESULT_DIR = REPO_ROOT / "FINAL_THESIS_RUN" / "01_prediction_all_regimes" / REGIME
MODEL_PATH = RESULT_DIR / "model_bundle.npz"
METRICS_PATH = RESULT_DIR / "heldout_test_metrics.json"

OUTPUT_PDF = REPO_ROOT / "Report" / "figures" / "chaotic_bursting_heldout.pdf"
OUTPUT_PNG = REPO_ROOT / "Chapter1_Final_Thesis_Figures" / "prediction" / "chaotic_bursting_heldout.png"

ZOOM_WINDOW = (1470.0, 1500.0)


def reproduce_locked_prediction():
    """Reproduce the held-out rollout and check it against the saved metrics."""
    _, _, times, train, test = _load_regime(REGIME)
    model, metadata = EchoStateNetwork.load_bundle(MODEL_PATH)

    mean = np.asarray(metadata["external_mean"], dtype=float)
    std = np.asarray(metadata["external_std"], dtype=float)

    train_norm = (train - mean) / std
    test_norm = (test - mean) / std

    pred_norm, _ = model.predict(
        np.vstack([train_norm, test_norm]),
        n_warmup=len(train_norm) - 1,
    )
    pred_norm = np.asarray(pred_norm, dtype=float)[: len(test)]
    prediction = pred_norm * std + mean

    threshold_norm = (float(config.SPIKE_THRESHOLD) - mean[0, 0]) / std[0, 0]
    metrics = _heldout_metrics(pred_norm, test_norm, prediction, test, threshold_norm)
    saved = json.loads(METRICS_PATH.read_text())
    for key in ("nrmse_recursive_x", "nrmse_recursive_all_states"):
        if not np.isclose(metrics[key], saved[key], rtol=1e-9, atol=0.0):
            raise RuntimeError(
                f"Reproduced {key}={metrics[key]:.6g} differs from the saved "
                f"value {saved[key]:.6g}. Run this script on a `work` node."
            )

    heldout_time = times[len(train) : len(train) + len(test)]
    return heldout_time, test, prediction, saved


def spike_peaks(x: np.ndarray, threshold: float) -> np.ndarray:
    """Maximum of each contiguous episode with x >= threshold."""
    above = np.r_[False, x >= threshold, False].astype(int)
    starts = np.flatnonzero(np.diff(above) == 1)
    ends = np.flatnonzero(np.diff(above) == -1)
    return np.array(
        [start + int(np.argmax(x[start:end])) for start, end in zip(starts, ends)],
        dtype=int,
    )


FONT_SCALE = 2.1  # same text scaling as create_chapter1_final_prediction_figures.py


def fs(points: float) -> float:
    return round(points * FONT_SCALE, 1)


def create_figure(time, reference, prediction, saved) -> None:
    threshold = float(config.SPIKE_THRESHOLD)
    ref_peaks = spike_peaks(reference[:, 0], threshold)
    pred_peaks = spike_peaks(prediction[:, 0], threshold)
    if len(ref_peaks) != len(pred_peaks):
        raise RuntimeError("Spike counts differ; offsets cannot be paired.")
    offsets = time[pred_peaks] - time[ref_peaks]

    # Held-out time axis starting at zero, as in the periodic figures.
    time = time - time[0]
    vph_time = saved["chaotic_valid_prediction_horizon"]["horizon_time"]
    zoom_window = (ZOOM_WINDOW[0] - 1050.0, ZOOM_WINDOW[1] - 1050.0)

    figure = plt.figure(figsize=(13.0, 7.4))
    outer = figure.add_gridspec(1, 2, width_ratios=[1.5, 1.8], wspace=0.24)
    left = outer[0].subgridspec(3, 1, hspace=0.10)
    right = outer[1].subgridspec(2, 1, hspace=1.0)

    state_axes = []
    for index, label in enumerate((r"$x$", r"$y$", r"$z$")):
        axis = figure.add_subplot(
            left[index, 0], sharex=state_axes[0] if state_axes else None
        )
        state_axes.append(axis)
        axis.plot(time, reference[:, index], linewidth=2.0, label="Reference")
        axis.plot(time, prediction[:, index], linestyle="--", linewidth=2.0,
                  label="ESN prediction")
        axis.axvline(vph_time, color="0.35", linestyle=":", linewidth=1.8)
        axis.set_ylabel(label, rotation=0, labelpad=22, fontsize=fs(12))
        axis.grid(True, alpha=0.20)
        axis.tick_params(axis="both", labelsize=fs(9))
        if index < 2:
            axis.tick_params(labelbottom=False)
    state_axes[0].axvspan(*zoom_window, color="0.85", zorder=0)
    state_axes[0].set_title("(a)", fontsize=fs(9), fontweight="bold", pad=8, loc="left")
    state_axes[-1].set_xlabel("Held-out time", fontsize=fs(11))
    state_axes[-1].set_xlim(time[0], time[-1])

    zoom = (time >= zoom_window[0]) & (time < zoom_window[1])
    zoom_axis = figure.add_subplot(right[0, 0])
    zoom_axis.plot(time[zoom], reference[zoom, 0], linewidth=2.8)
    zoom_axis.plot(time[zoom], prediction[zoom, 0], linestyle="--", linewidth=2.6)
    zoom_axis.set_title("(b) Final two spikes", fontsize=fs(9), fontweight="bold",
                        pad=8, loc="left")
    zoom_axis.set_xlabel("Held-out time", fontsize=fs(10))
    zoom_axis.set_ylabel(r"$x$", fontsize=fs(12), rotation=0, labelpad=22)
    zoom_axis.tick_params(axis="both", labelsize=fs(9))
    zoom_axis.grid(True, alpha=0.20)

    ref_spike_times = time[ref_peaks]
    offset_axis = figure.add_subplot(right[1, 0])
    offset_axis.axhline(0.0, color="0.6", linewidth=1.2)
    offset_axis.axvline(vph_time, color="0.35", linestyle=":", linewidth=1.8)
    offset_axis.plot(ref_spike_times, offsets, marker="o", markersize=9,
                     linewidth=1.8, color="C2")
    offset_axis.set_title("(c) Spike-timing offset", fontsize=fs(9),
                          fontweight="bold", pad=8, loc="left")
    offset_axis.set_xlabel("Reference spike time", fontsize=fs(10))
    offset_axis.set_ylabel(r"$\Delta t$", fontsize=fs(12), rotation=0, labelpad=22)
    offset_axis.set_xlim(time[0] - 10.0, time[-1] + 10.0)
    offset_axis.tick_params(axis="both", labelsize=fs(9))
    offset_axis.grid(True, alpha=0.20)

    handles, labels = state_axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, fontsize=fs(10),
                  frameon=True, bbox_to_anchor=(0.5, 1.0))
    figure.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.09)

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(figure)

    for k, (r, o) in enumerate(zip(ref_spike_times, offsets), start=1):
        print(f"spike {k:2d}  t_heldout={r:7.2f}  offset={o:+.2f}")
    print(f"Saved {OUTPUT_PDF}\nSaved {OUTPUT_PNG}")


def main() -> None:
    time, reference, prediction, saved = reproduce_locked_prediction()
    create_figure(time, reference, prediction, saved)


if __name__ == "__main__":
    main()
