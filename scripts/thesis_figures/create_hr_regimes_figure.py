from __future__ import annotations

"""Generate the Section 2.1 figure showing the three Hindmarsh-Rose regimes.

Three stacked panels, one per regime, each showing the fast membrane
potential x(t) together with the slow adaptation current z(t) over the
full retained record. The z overlay is what makes the
fast-slow mechanism visible: z is almost constant under tonic spiking,
sweeps up and down once per burst under periodic bursting, and repeats
irregularly under chaotic bursting.

Parameters, integration step and transient length are read from
config.py and the trajectory is produced by the same RK4 routine the
data loader uses, so the figure cannot drift from the simulated records
that the rest of the chapter reports on.

Output: Report/figures/hr_regimes.pdf (and .png for preview).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # embed real (non-Type3) fonts
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np

import config
from data_loader import _rk4_hr

OUTPUT_DIR = REPO_ROOT / "Report" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DT = float(config.HR_DT)

# The full retained record (t in [0, 1500)), i.e. exactly the data the
# network is trained and tested on. Individual tonic spikes stay
# resolvable at this span, and it shows six periodic-burst cycles
# against the irregular chaotic record.
WINDOW_TIME = float(config.HR_TOTAL_STEPS) * DT
WINDOW_STEPS = int(round(WINDOW_TIME / DT))

REGIMES = [
    ("periodic_spiking", "(a) Periodic spiking"),
    ("periodic_bursting", "(b) Periodic bursting"),
    ("chaotic_bursting", "(c) Chaotic bursting"),
]

COLOR_X = "#2a5c8a"
COLOR_Z = "#e0952b"

# Rendered at 7.0 in wide for a 15 cm (~5.9 in) text block, a ~0.84x
# shrink; FONT_SCALE compensates so on-page text lands near 9-10 pt
# against the 12 pt body text.
FONT_SCALE = 1.2


def fs(points: float) -> float:
    return round(points * FONT_SCALE, 1)


def simulate(regime: str) -> np.ndarray:
    """Retained record of `regime`, from config, for the plotted window."""
    parameter_set = config.HR_PARAMETER_SETS[regime]
    parameters = {
        name: float(parameter_set[name])
        for name in ("a", "b", "c", "d", "r", "s", "xr", "I")
    }
    transient = int(parameter_set["transient_steps"])
    trajectory = _rk4_hr(
        parameter_set["x0"],
        transient + WINDOW_STEPS,
        DT,
        parameters,
    )
    return trajectory[transient:]


def build_figure() -> plt.Figure:
    records = {regime: simulate(regime) for regime, _ in REGIMES}

    # One z scale for all three panels. Auto-scaling each panel would
    # stretch tonic spiking's nearly constant z into an apparent sweep
    # as large as the bursting regimes', which is the opposite of what
    # separates them.
    z_values = np.concatenate([record[:, 2] for record in records.values()])
    z_low, z_high = float(np.min(z_values)), float(np.max(z_values))
    z_pad = 0.08 * (z_high - z_low)

    fig, axes = plt.subplots(
        3, 1, figsize=(7.0, 4.9), sharex=True,
    )

    times = np.arange(WINDOW_STEPS) * DT

    for ax, (regime, title) in zip(axes, REGIMES):
        record = records[regime]
        parameter_set = config.HR_PARAMETER_SETS[regime]

        twin = ax.twinx()
        twin.plot(
            times, record[:, 2], color=COLOR_Z, linewidth=1.4, zorder=2,
        )
        twin.set_ylim(z_low - z_pad, z_high + z_pad)
        twin.set_ylabel(r"$z$", fontsize=fs(10), rotation=0, labelpad=10)
        twin.tick_params(axis="y", labelsize=fs(8), colors=COLOR_Z)
        twin.spines["right"].set_color(COLOR_Z)
        twin.spines["top"].set_visible(False)

        ax.plot(times, record[:, 0], color=COLOR_X, linewidth=0.8, zorder=3)
        ax.set_ylim(-2.2, 2.6)
        ax.set_ylabel(r"$x$", fontsize=fs(10), rotation=0, labelpad=10)
        ax.tick_params(axis="both", labelsize=fs(8))
        ax.set_yticks([-2, 0, 2])
        ax.spines["top"].set_visible(False)
        ax.set_zorder(twin.get_zorder() + 1)
        ax.patch.set_visible(False)

        ax.set_title(
            f"{title}   ($r={parameter_set['r']:g}$, $I={parameter_set['I']:g}$)",
            fontsize=fs(9), fontweight="bold", loc="left", pad=5,
        )

    axes[-1].set_xlabel(
        "Time $t$ (from end of discarded transient)", fontsize=fs(10),
        labelpad=6,
    )
    axes[-1].set_xlim(0, WINDOW_TIME)

    handles = [
        plt.Line2D([], [], color=COLOR_X, linewidth=1.5,
                   label=r"$x$, fast membrane potential"),
        plt.Line2D([], [], color=COLOR_Z, linewidth=1.5,
                   label=r"$z$, slow adaptation current (shared scale)"),
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=2, fontsize=fs(8.5), frameon=False, handlelength=1.8,
        columnspacing=2.2,
    )

    fig.subplots_adjust(
        left=0.085, right=0.915, top=0.945, bottom=0.175, hspace=0.5,
    )
    return fig


def main() -> None:
    fig = build_figure()
    png_path = OUTPUT_DIR / "hr_regimes.png"
    pdf_path = OUTPUT_DIR / "hr_regimes.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
