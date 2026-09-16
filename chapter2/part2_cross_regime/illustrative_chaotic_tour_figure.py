"""Figure for the illustrative chaotic-range tour. Clearly labelled as
supplementary/illustrative -- not the official CC/CR benchmark figure.
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
PRED_COLOR = "#D55E00"
TEST_MARK_COLOR = "#7A0DA6"

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def main() -> None:
    npz = np.load(RESULTS_DIR / "illustrative_chaotic_tour.npz")
    time, current, truth, pred = npz["time"], npz["current"], npz["truth"], npz["prediction"]
    switches = npz["switch_indices"]
    schedule = npz["schedule_currents"]
    is_test = npz["is_test_current"]

    fig, (ax_i, ax_t, ax_p) = plt.subplots(3, 1, figsize=(15, 8), sharex=True, height_ratios=[1, 1, 1.3])

    ax_i.plot(time, current, color=CURRENT_COLOR, lw=1.4)
    ax_i.set_ylabel("Current I(t)")
    ax_i.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax_t.plot(time, truth[:, 0], color=TRUTH_COLOR, lw=0.7)
    ax_t.set_ylabel("True x(t)")
    ax_t.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax_p.plot(time, truth[:, 0], color=TRUTH_COLOR, lw=0.6, alpha=0.5, label="truth")
    ax_p.plot(time, pred[:, 0], color=PRED_COLOR, lw=0.7, label="chaotic-trained model prediction")
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
    ax_i.set_ylim(i_lo, i_hi + 0.24 * (i_hi - i_lo))
    boundaries = [0] + switches.tolist() + [len(time)]
    for k, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mid = time[(start + min(stop, len(time) - 1)) // 2]
        role = "TEST" if is_test[k] else "training"
        color = TEST_MARK_COLOR if is_test[k] else "#444444"
        ax_i.text(mid, i_hi + 0.06 * (i_hi - i_lo), f"I={schedule[k]:.4g}\n({role})",
                   ha="center", va="bottom", fontsize=8, color=color)

    fig.suptitle(
        "Illustrative only: chaotic model across its full confirmed chaotic range (2.98–3.40)\n"
        "Not the official CC/CR benchmark — purely a wider visual tour combining training and test currents",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path = RESULTS_DIR / "fig_illustrative_chaotic_tour.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
