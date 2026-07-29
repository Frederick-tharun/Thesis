from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from final_pipeline import _load_regime
from model import EchoStateNetwork


RUN_ROOT = REPO_ROOT / "FINAL_THESIS_RUN"
REGIME = "chaotic_bursting"

RESULT_DIR = RUN_ROOT / "01_prediction_all_regimes" / REGIME
MODEL_PATH = RESULT_DIR / "model_bundle.npz"

OUTPUT_PNG = RESULT_DIR / "chaotic_bursting_thesis_figure.png"
OUTPUT_PDF = RESULT_DIR / "chaotic_bursting_thesis_figure.pdf"


def reproduce_locked_prediction():
    """
    Reproduce the locked held-out autonomous prediction
    using the saved chaotic-bursting model bundle.
    """
    _, _, times, train, test = _load_regime(REGIME)

    model, metadata = EchoStateNetwork.load_bundle(MODEL_PATH)

    mean = np.asarray(metadata["external_mean"], dtype=float)
    std = np.asarray(metadata["external_std"], dtype=float)

    if mean.size == 0 or std.size == 0:
        raise RuntimeError(
            "external_mean or external_std not found in model bundle."
        )

    train_norm = (train - mean) / std
    test_norm = (test - mean) / std

    full_input = np.vstack([train_norm, test_norm])

    pred_norm, _ = model.predict(
        full_input,
        n_warmup=len(train_norm) - 1,
    )

    pred_norm = np.asarray(pred_norm, dtype=float)[: len(test)]
    prediction = pred_norm * std + mean

    heldout_time = times[len(train): len(train) + len(test)]
    heldout_time = heldout_time - heldout_time[0]

    return heldout_time, test, prediction


def create_figure(time, reference, prediction):
    """
    Create the thesis figure:
    (a) x,y,z time series
    (b) 3D phase-space comparison
    """

    # optional display downsampling for clarity
    stride_time = max(1, len(time) // 12000)
    stride_phase = max(1, len(time) // 4000)

    t_plot = time[::stride_time]
    ref_plot = reference[::stride_time]
    pred_plot = prediction[::stride_time]

    ref_phase = reference[::stride_phase]
    pred_phase = prediction[::stride_phase]

    fig = plt.figure(figsize=(13.5, 6.8))
    outer = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[2.2, 1.2],
        wspace=0.18
    )

    # -------------------------------
    # Left side: time-series panels
    # -------------------------------
    left = outer[0].subgridspec(3, 1, hspace=0.08)

    labels = [r"$x$", r"$y$", r"$z$"]
    axes = []

    for i in range(3):
        ax = fig.add_subplot(left[i, 0], sharex=axes[0] if axes else None)
        axes.append(ax)

        ax.plot(
            t_plot,
            ref_plot[:, i],
            linewidth=1.2,
            label="Reference"
        )
        ax.plot(
            t_plot,
            pred_plot[:, i],
            linestyle="--",
            linewidth=1.2,
            label="ESN prediction"
        )

        ax.set_ylabel(labels[i], rotation=0, labelpad=12, fontsize=12)
        ax.grid(True, alpha=0.20)
        ax.tick_params(axis="both", labelsize=9)

        if i < 2:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Held-out time", fontsize=11)

    axes[0].legend(
        loc="upper right",
        fontsize=9,
        ncol=2,
        frameon=True
    )

    axes[0].text(
        0.01, 0.88, "(a)",
        transform=axes[0].transAxes,
        fontsize=12,
        fontweight="bold"
    )

    # --------------------------------
    # Right side: 3D phase-space plot
    # --------------------------------
    ax3d = fig.add_subplot(outer[1], projection="3d")

    ax3d.plot(
        ref_phase[:, 0],
        ref_phase[:, 1],
        ref_phase[:, 2],
        linewidth=1.2,
        label="Reference"
    )

    ax3d.plot(
        pred_phase[:, 0],
        pred_phase[:, 1],
        pred_phase[:, 2],
        linestyle="--",
        linewidth=1.2,
        label="ESN prediction"
    )

    ax3d.set_xlabel(r"$x$", fontsize=11, labelpad=4)
    ax3d.set_ylabel(r"$y$", fontsize=11, labelpad=4)
    ax3d.set_zlabel(r"$z$", fontsize=11, labelpad=4)
    ax3d.tick_params(axis="both", labelsize=8)
    ax3d.view_init(elev=24, azim=-58)

    ax3d.legend(
        loc="upper right",
        fontsize=8,
        frameon=True
    )

    ax3d.text2D(
        0.03, 0.95, "(b)",
        transform=ax3d.transAxes,
        fontsize=12,
        fontweight="bold"
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.96,
        bottom=0.11
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved PNG: {OUTPUT_PNG}")
    print(f"Saved PDF: {OUTPUT_PDF}")


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model bundle not found: {MODEL_PATH}")

    time, reference, prediction = reproduce_locked_prediction()
    create_figure(time, reference, prediction)


if __name__ == "__main__":
    main()