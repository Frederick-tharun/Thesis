from __future__ import annotations

"""Generate the Section 2.1 data-allocation timeline figure.

Single split bar over the retained per-regime record t in [0, 1500),
showing the primary [0,1050) train+select / [1050,1500) held-out-test
split, with a zoomed inset underneath showing how the search stage
subdivides [0,1050) into an unused prefix, the 500-time-unit
candidate-training slice, and the three 80-time-unit validation
windows.

All boundaries are computed from config.py constants (mirroring
optimize_model.prediction_validation_spec's arithmetic), not
hardcoded, so the figure cannot drift from the actual protocol.

Output: Report/figures/data_split_timeline.pdf (and .png for preview).
"""

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
from matplotlib.patches import Patch, Rectangle

import config

OUTPUT_DIR = REPO_ROOT / "Report" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DT = float(config.HR_DT)
N_TOTAL = int(config.HR_TOTAL_STEPS)


def steps_to_time(steps: float) -> float:
    return steps * DT


def compute_boundaries() -> dict:
    """Mirror optimize_model.prediction_validation_spec's boundary arithmetic
    in steps, using only config.py constants. Importing that function directly
    pulls in scikit-optimize (an unrelated BO-search dependency) purely to
    reach this deterministic arithmetic, so the derivation is inlined here
    instead; every constant below is read from config.py, not hand-typed.
    """
    n_final_train = int(N_TOTAL * float(getattr(config, "TRAIN_RATIO", 0.70)))
    n_final_train = max(10, min(n_final_train, N_TOTAL - 1))
    heldout_length = N_TOTAL - n_final_train

    num_windows = int(getattr(config, "PREDICTION_VALIDATION_NUM_WINDOWS", 3))
    window_length = int(getattr(config, "PREDICTION_VALIDATION_WINDOW_LENGTH", 8000))
    total_validation = num_windows * window_length
    first_start = n_final_train - total_validation
    window_starts = [first_start + i * window_length for i in range(num_windows)]

    max_train_steps = int(
        getattr(config, "OPT_TRAIN_MAX_STEPS", getattr(config, "BO_TRAIN_MAX_STEPS", 50000))
    )
    training_end = window_starts[0]
    training_start = max(0, training_end - max_train_steps)

    heldout_start = n_final_train
    heldout_end = n_final_train + heldout_length

    windows = [
        (steps_to_time(s), steps_to_time(s + window_length)) for s in window_starts
    ]
    return {
        "record_end": steps_to_time(N_TOTAL),
        "candidate_train_start": steps_to_time(training_start),
        "candidate_train_end": steps_to_time(training_end),
        "windows": windows,
        "final_train_end": steps_to_time(n_final_train),
        "heldout_start": steps_to_time(heldout_start),
        "heldout_end": steps_to_time(heldout_end),
    }


COLOR_UNUSED = "#e7e9ee"
COLOR_TRAIN = "#2a5c8a"
COLOR_TRAIN_SOFT = "#5b8fb9"
COLOR_VALID = "#e0952b"
COLOR_TEST = "#2f9169"
EDGE = "#20242b"

# The thesis page places this figure at \textwidth = 15 cm (~5.9 in);
# rendered here at 7.0 in wide, that is a mild ~0.84x shrink. FONT_SCALE
# compensates so on-page text lands around 11-14pt, matching the other
# Section 2.4 figures.
FONT_SCALE = 1.55


def fs(points: float) -> float:
    return round(points * FONT_SCALE, 1)


def bar(ax, x0, x1, y0, height, color, hatch=None):
    ax.add_patch(
        Rectangle(
            (x0, y0), x1 - x0, height, facecolor=color, edgecolor=EDGE,
            linewidth=1.0, hatch=hatch, zorder=2,
        )
    )


def build_figure(boundaries: dict):
    record_end = boundaries["record_end"]
    ct_start = boundaries["candidate_train_start"]
    ct_end = boundaries["candidate_train_end"]
    windows = boundaries["windows"]
    final_train_end = boundaries["final_train_end"]
    heldout_start = boundaries["heldout_start"]
    heldout_end = boundaries["heldout_end"]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    main_y, main_h = 0.84, 0.12
    inset_y, inset_h = 0.46, 0.24

    # Main bar: the primary train+select / held-out-test split.
    bar(ax, 0.0, final_train_end, main_y, main_h, COLOR_TRAIN)
    bar(ax, heldout_start, heldout_end, main_y, main_h, COLOR_TEST)
    ax.text(
        final_train_end / 2, main_y + main_h / 2,
        "Train + select on this interval", ha="center", va="center",
        fontsize=fs(9.5), color="white", fontweight="bold", zorder=3,
    )
    ax.text(
        (heldout_start + heldout_end) / 2, main_y + main_h / 2,
        "Held-out test", ha="center", va="center",
        fontsize=fs(9.5), color="white", fontweight="bold", zorder=3,
    )

    # Inset bar: zoomed breakdown of the search stage inside [0, 1050).
    bar(ax, 0.0, ct_start, inset_y, inset_h, COLOR_UNUSED, "....")
    bar(ax, ct_start, ct_end, inset_y, inset_h, COLOR_TRAIN_SOFT)
    for w_start, w_end in windows:
        bar(ax, w_start, w_end, inset_y, inset_h, COLOR_VALID)
    for w_start, w_end in windows[:-1]:
        ax.plot([w_end, w_end], [inset_y + 0.02, inset_y + inset_h - 0.02],
                 color="white", linewidth=1.2, zorder=3)
    ax.add_patch(
        Rectangle(
            (0.0, inset_y), final_train_end, inset_h, facecolor="none",
            edgecolor="0.45", linewidth=1.1, linestyle="--", zorder=4,
        )
    )

    # Zoom-callout connectors from the main bar's train segment down to the inset.
    for x in (0.0, final_train_end):
        ax.plot([x, x], [main_y, inset_y + inset_h], color="0.55",
                 linewidth=0.9, linestyle=":", zorder=1)

    ax.text(
        final_train_end / 2, (main_y + inset_y + inset_h) / 2,
        "how this interval is used during hyperparameter search",
        ha="center", va="center", fontsize=fs(8.3), color="0.35",
        style="italic",
    )

    ax.set_xlim(-15, record_end + 15)
    ax.set_ylim(0.40, 1.02)
    ax.set_yticks([])

    tick_positions = sorted({0.0, ct_start, ct_end, final_train_end, record_end})
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{t:g}" for t in tick_positions], fontsize=fs(9))
    ax.set_xlabel(
        "Time $t$ (from end of discarded transient)",
        fontsize=fs(10), labelpad=fs(6),
    )

    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    legend_elements = [
        Patch(facecolor=COLOR_TRAIN_SOFT, edgecolor=EDGE, label="Candidate training"),
        Patch(facecolor=COLOR_VALID, edgecolor=EDGE, label="Validation windows (scored)"),
        Patch(facecolor=COLOR_UNUSED, edgecolor=EDGE, hatch="....", label="Not used in search"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=3, fontsize=fs(8), frameon=False, handlelength=1.3, columnspacing=1.3,
    )

    fig.subplots_adjust(left=0.03, right=0.98, top=0.97, bottom=0.32)
    return fig


def main():
    boundaries = compute_boundaries()
    print("Verified interval boundaries (time units):")
    for key, value in boundaries.items():
        print(f"  {key}: {value}")

    fig = build_figure(boundaries)
    png_path = OUTPUT_DIR / "data_split_timeline.png"
    pdf_path = OUTPUT_DIR / "data_split_timeline.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
