"""Continuous current-switching figures: one for the regular-trained model,
one for the chaotic-trained model. Each is standalone (own title, own scale).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from . import config

RESULTS_DIR = config.PACKAGE_ROOT / "results" / "final_evaluation_long"

TRUTH_COLOR = "#222222"
CURRENT_COLOR = "#009E73"
REGULAR_COLOR = "#0072B2"
CHAOTIC_COLOR = "#D55E00"

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def build_figure(npz, pred_color, title, model_word, segment_labels, out_path):
    time = npz["time"]
    current = npz["current"]
    truth = npz["truth"]
    pred = npz["prediction"]
    switches = npz["switch_indices"]

    fig, (ax_i, ax_t, ax_p) = plt.subplots(
        3, 1, figsize=(14, 8), sharex=True, height_ratios=[1, 1, 1.3]
    )

    ax_i.plot(time, current, color=CURRENT_COLOR, lw=1.4)
    ax_i.set_ylabel("Current I(t)")
    ax_i.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax_t.plot(time, truth[:, 0], color=TRUTH_COLOR, lw=0.7)
    ax_t.set_ylabel("True x(t)")
    ax_t.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax_p.plot(time, truth[:, 0], color=TRUTH_COLOR, lw=0.6, alpha=0.5, label="truth")
    ax_p.plot(time, pred[:, 0], color=pred_color, lw=0.7, label=f"{model_word} model prediction")
    ax_p.set_ylabel("x(t)")
    ax_p.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_p.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_p.set_xlabel("time")

    ymin = min(truth[:, 0].min(), pred[:, 0].min())
    ymax = max(truth[:, 0].max(), pred[:, 0].max())
    pad = 0.08 * (ymax - ymin)
    ax_t.set_ylim(ymin - pad, ymax + pad)
    ax_p.set_ylim(ymin - pad, ymax + pad)

    for ax in (ax_i, ax_t, ax_p):
        for b in switches:
            ax.axvline(time[b], color="grey", lw=0.7, ls="--", alpha=0.7)

    i_lo, i_hi = ax_i.get_ylim()
    ax_i.set_ylim(i_lo, i_hi + 0.22 * (i_hi - i_lo))
    boundaries = [0] + switches.tolist() + [len(time)]
    for k, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mid = time[(start + min(stop, len(time) - 1)) // 2]
        ax_i.text(mid, i_hi + 0.06 * (i_hi - i_lo), segment_labels[k],
                   ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    reg = np.load(RESULTS_DIR / "continuous_schedule_regular.npz")
    cha = np.load(RESULTS_DIR / "continuous_schedule_chaotic.npz")

    build_figure(
        reg, REGULAR_COLOR,
        "Regular-trained model: tracking a changing current",
        "Regular-trained",
        [f"I={c:.3g}" for c in reg["schedule_currents"]],
        RESULTS_DIR / "fig0a_continuous_regular_model.png",
    )
    build_figure(
        cha, CHAOTIC_COLOR,
        "Chaotic-trained model: tracking a changing current",
        "Chaotic-trained",
        [f"I={c:.3g}" for c in cha["schedule_currents"]],
        RESULTS_DIR / "fig0b_continuous_chaotic_model.png",
    )


if __name__ == "__main__":
    main()
