from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks

import config
from final_pipeline import _load_regime
from model import EchoStateNetwork


RUN_ROOT = REPO_ROOT / "FINAL_THESIS_RUN"
REGIME = "periodic_spiking"

PREDICTION_DIR = (
    RUN_ROOT
    / "01_prediction_all_regimes"
    / REGIME
)

MODEL_PATH = PREDICTION_DIR / "model_bundle.npz"

OUTPUT_PATH = (
    PREDICTION_DIR
    / "periodic_spiking_thesis_figure.png"
)


def reproduce_locked_prediction():
    """Reproduce the held-out rollout using the locked ESN bundle."""

    _, _, times, train, test = _load_regime(REGIME)

    model, metadata = EchoStateNetwork.load_bundle(
        MODEL_PATH
    )

    mean = np.asarray(
        metadata["external_mean"],
        dtype=float,
    )

    std = np.asarray(
        metadata["external_std"],
        dtype=float,
    )

    if mean.size == 0 or std.size == 0:
        raise RuntimeError(
            "External training mean and standard deviation "
            "were not found in the model bundle."
        )

    train_norm = (train - mean) / std
    test_norm = (test - mean) / std

    evaluation_input = np.vstack(
        [train_norm, test_norm]
    )

    pred_norm, _ = model.predict(
        evaluation_input,
        n_warmup=len(train_norm) - 1,
    )

    pred_norm = np.asarray(
        pred_norm,
        dtype=float,
    )[: len(test)]

    pred_raw = pred_norm * std + mean

    pred_time = times[
        len(train) : len(train) + len(pred_raw)
    ]

    # Use time relative to the beginning of the held-out rollout.
    pred_time = pred_time - pred_time[0]

    return pred_time, test, pred_raw


def select_zoom_window(
    truth_x: np.ndarray,
) -> tuple[int, int]:
    """Select a compact interval containing two neighbouring spikes."""

    threshold = float(
        getattr(config, "SPIKE_THRESHOLD", 1.0)
    )

    min_distance = int(
        getattr(config, "SPIKE_MIN_DISTANCE", 5)
    )

    peaks, _ = find_peaks(
        truth_x,
        height=threshold,
        distance=min_distance,
    )

    if len(peaks) >= 2:
        gaps = np.diff(peaks)

        # Select the closest pair so that the zoom remains compact.
        pair_index = int(np.argmin(gaps))

        first_peak = int(peaks[pair_index])
        second_peak = int(peaks[pair_index + 1])

        gap = second_peak - first_peak
        margin = max(100, int(0.8 * gap))

        start = max(0, first_peak - margin)
        end = min(
            len(truth_x),
            second_peak + margin,
        )

        return start, end

    # Fallback when fewer than two spikes are detected.
    centre = len(truth_x) // 2
    half_width = min(1500, centre)

    return (
        max(0, centre - half_width),
        min(len(truth_x), centre + half_width),
    )


def create_figure(
    time: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> None:
    """Create the final periodic-spiking thesis figure."""

    start, end = select_zoom_window(
        truth[:, 0]
    )

    # Reduce plotting density without changing the underlying results.
    display_stride = max(
        1,
        len(time) // 15000,
    )

    time_display = time[::display_stride]
    truth_display = truth[::display_stride]
    prediction_display = prediction[::display_stride]

    figure = plt.figure(
        figsize=(12.5, 6.5),
    )

    outer_grid = figure.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[2.35, 1.0],
        wspace=0.16,
    )

    left_grid = outer_grid[0].subgridspec(
        nrows=3,
        ncols=1,
        hspace=0.08,
    )

    state_names = (
        r"$x$",
        r"$y$",
        r"$z$",
    )

    axes = []

    for state_index, state_name in enumerate(
        state_names
    ):
        axis = figure.add_subplot(
            left_grid[state_index, 0],
            sharex=axes[0] if axes else None,
        )

        axes.append(axis)

        axis.plot(
            time_display,
            truth_display[:, state_index],
            linewidth=1.25,
            label="Reference",
        )

        axis.plot(
            time_display,
            prediction_display[:, state_index],
            linestyle="--",
            linewidth=1.15,
            label="ESN prediction",
        )

        axis.set_ylabel(
            state_name,
            fontsize=12,
            rotation=0,
            labelpad=14,
        )

        axis.tick_params(
            axis="both",
            labelsize=9,
        )

        axis.grid(
            True,
            alpha=0.20,
        )

        if state_index < 2:
            axis.tick_params(
                labelbottom=False,
            )

    axes[-1].set_xlabel(
        "Held-out time",
        fontsize=11,
    )

    axes[0].legend(
        loc="upper right",
        fontsize=9,
        ncol=2,
        frameon=True,
    )

    axes[0].text(
        0.01,
        0.88,
        "(a)",
        transform=axes[0].transAxes,
        fontsize=11,
        fontweight="bold",
    )

    # Highlight only the x-state interval used in the zoom.
    x_lower, x_upper = axes[0].get_ylim()

    zoom_rectangle = Rectangle(
        (
            time[start],
            x_lower,
        ),
        time[end - 1] - time[start],
        x_upper - x_lower,
        fill=False,
        linestyle=":",
        linewidth=1.2,
    )

    axes[0].add_patch(zoom_rectangle)

    # Compact zoom comparison on the right.
    zoom_axis = figure.add_subplot(
        outer_grid[1]
    )

    zoom_axis.plot(
        time[start:end],
        truth[start:end, 0],
        linewidth=1.8,
        label="Reference",
    )

    zoom_axis.plot(
        time[start:end],
        prediction[start:end, 0],
        linestyle="--",
        linewidth=1.7,
        label="ESN prediction",
    )

    zoom_axis.set_title(
        "Representative two-spike interval",
        fontsize=11,
        pad=8,
    )

    zoom_axis.set_xlabel(
        "Held-out time",
        fontsize=10,
    )

    zoom_axis.set_ylabel(
        r"$x$",
        fontsize=12,
        rotation=0,
        labelpad=12,
    )

    zoom_axis.tick_params(
        axis="both",
        labelsize=9,
    )

    zoom_axis.grid(
        True,
        alpha=0.20,
    )

    zoom_axis.legend(
        loc="best",
        fontsize=8,
        frameon=True,
    )

    zoom_axis.set_box_aspect(1.05)

    zoom_axis.text(
        0.03,
        0.94,
        "(b)",
        transform=zoom_axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.11,
        top=0.96,
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved thesis figure: {OUTPUT_PATH}"
    )


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model bundle not found: {MODEL_PATH}"
        )

    time, truth, prediction = (
        reproduce_locked_prediction()
    )

    create_figure(
        time,
        truth,
        prediction,
    )


if __name__ == "__main__":
    main()