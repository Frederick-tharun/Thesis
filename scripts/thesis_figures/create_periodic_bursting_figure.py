from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import config
from final_pipeline import _load_regime
from model import EchoStateNetwork


RUN_ROOT = REPO_ROOT / "FINAL_THESIS_RUN"
REGIME = "periodic_bursting"

RESULT_DIR = (
    RUN_ROOT
    / "01_prediction_all_regimes"
    / REGIME
)

MODEL_PATH = RESULT_DIR / "model_bundle.npz"

OUTPUT_PNG = (
    RESULT_DIR
    / "periodic_bursting_thesis_figure.png"
)

OUTPUT_PDF = (
    RESULT_DIR
    / "periodic_bursting_thesis_figure.pdf"
)


def reproduce_locked_prediction():
    """
    Load the locked periodic-bursting ESN and reproduce its
    autonomous held-out prediction.
    """

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
            "The model bundle does not contain the "
            "training-derived normalisation values."
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

    prediction = pred_norm * std + mean

    heldout_time = times[
        len(train) : len(train) + len(prediction)
    ]

    # Start the held-out time axis from zero.
    heldout_time = heldout_time - heldout_time[0]

    return heldout_time, test, prediction


def split_spikes_into_bursts(
    peak_indices: np.ndarray,
) -> list[np.ndarray]:
    """
    Group nearby spikes into bursts by separating short
    intra-burst gaps from longer inter-burst gaps.
    """

    if len(peak_indices) == 0:
        return []

    if len(peak_indices) == 1:
        return [peak_indices]

    gaps = np.diff(peak_indices)

    if len(gaps) == 1:
        return [peak_indices]

    sorted_gaps = np.sort(gaps)

    gap_ratios = (
        sorted_gaps[1:]
        / np.maximum(sorted_gaps[:-1], 1)
    )

    largest_jump = int(np.argmax(gap_ratios))

    if gap_ratios[largest_jump] >= 1.5:
        burst_gap_limit = 0.5 * (
            sorted_gaps[largest_jump]
            + sorted_gaps[largest_jump + 1]
        )
    else:
        burst_gap_limit = 1.8 * np.median(gaps)

    groups: list[list[int]] = [
        [int(peak_indices[0])]
    ]

    for previous_peak, current_peak in zip(
        peak_indices[:-1],
        peak_indices[1:],
    ):
        if (
            current_peak - previous_peak
            > burst_gap_limit
        ):
            groups.append([])

        groups[-1].append(int(current_peak))

    return [
        np.asarray(group, dtype=int)
        for group in groups
    ]


def select_representative_burst(
    reference_x: np.ndarray,
) -> tuple[int, int]:
    """
    Select one complete burst with a short quiet interval
    before and after it.
    """

    spike_threshold = float(
        getattr(
            config,
            "SPIKE_THRESHOLD",
            1.0,
        )
    )

    minimum_distance = int(
        getattr(
            config,
            "SPIKE_MIN_DISTANCE",
            5,
        )
    )

    peaks, _ = find_peaks(
        reference_x,
        height=spike_threshold,
        distance=minimum_distance,
    )

    burst_groups = split_spikes_into_bursts(
        peaks
    )

    if not burst_groups:
        return 0, min(
            len(reference_x),
            5000,
        )

    # Select the burst containing the largest number of spikes.
    maximum_length = max(
        len(group)
        for group in burst_groups
    )

    candidate_groups = [
        group
        for group in burst_groups
        if len(group) == maximum_length
    ]

    # Choose a central burst if several bursts have the same size.
    selected_group = candidate_groups[
        len(candidate_groups) // 2
    ]

    first_peak = int(selected_group[0])
    last_peak = int(selected_group[-1])

    if len(selected_group) > 1:
        typical_spike_gap = int(
            np.median(
                np.diff(selected_group)
            )
        )
    else:
        typical_spike_gap = 500

    margin = max(
        300,
        int(1.5 * typical_spike_gap),
    )

    start = max(
        0,
        first_peak - margin,
    )

    end = min(
        len(reference_x),
        last_peak + margin,
    )

    return start, end


def create_figure(
    time: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
) -> None:
    """
    Create the periodic-bursting thesis figure.
    """

    zoom_start, zoom_end = (
        select_representative_burst(
            reference[:, 0]
        )
    )

    # Reduce only the displayed density of the full-horizon plots.
    # The prediction itself is not altered.
    display_stride = max(
        1,
        len(time) // 15000,
    )

    display_time = time[::display_stride]
    display_reference = reference[
        ::display_stride
    ]
    display_prediction = prediction[
        ::display_stride
    ]

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

    state_labels = (
        r"$x$",
        r"$y$",
        r"$z$",
    )

    state_axes = []

    for state_index, state_label in enumerate(
        state_labels
    ):
        axis = figure.add_subplot(
            left_grid[state_index, 0],
            sharex=(
                state_axes[0]
                if state_axes
                else None
            ),
        )

        state_axes.append(axis)

        axis.plot(
            display_time,
            display_reference[:, state_index],
            linewidth=1.25,
            label="Reference",
        )

        axis.plot(
            display_time,
            display_prediction[:, state_index],
            linestyle="--",
            linewidth=1.15,
            label="ESN prediction",
        )

        axis.set_ylabel(
            state_label,
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

    state_axes[-1].set_xlabel(
        "Held-out time",
        fontsize=11,
    )

    state_axes[0].legend(
        loc="upper right",
        fontsize=9,
        ncol=2,
        frameon=True,
    )

    state_axes[0].text(
        0.01,
        0.88,
        "(a)",
        transform=state_axes[0].transAxes,
        fontsize=11,
        fontweight="bold",
    )

    # Mark the representative burst in the x-state panel.
    for boundary in (
        time[zoom_start],
        time[zoom_end - 1],
    ):
        state_axes[0].axvline(
            boundary,
            linestyle=":",
            linewidth=1.2,
        )

    # Detailed x-state bursting interval.
    zoom_axis = figure.add_subplot(
        outer_grid[1]
    )

    zoom_axis.plot(
        time[zoom_start:zoom_end],
        reference[
            zoom_start:zoom_end,
            0,
        ],
        linewidth=1.8,
        label="Reference",
    )

    zoom_axis.plot(
        time[zoom_start:zoom_end],
        prediction[
            zoom_start:zoom_end,
            0,
        ],
        linestyle="--",
        linewidth=1.7,
        label="ESN prediction",
    )

    zoom_axis.set_title(
        "Representative bursting interval",
        fontsize=11,
        fontweight="bold",
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
        0.95,
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
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved PNG: {OUTPUT_PNG}"
    )

    print(
        f"Saved PDF: {OUTPUT_PDF}"
    )


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Locked model bundle not found: "
            f"{MODEL_PATH}"
        )

    time, reference, prediction = (
        reproduce_locked_prediction()
    )

    create_figure(
        time,
        reference,
        prediction,
    )


if __name__ == "__main__":
    main()