"""Create the thesis comparison of linear and finite-time feedback sweeps.

The layout uses one gain-versus-error panel per controller. The repository
stores full-state RMSE for each validation gain,
so the average MSE plotted here is RMSE squared. Unlike the reference paper,
this run has one realization per gain; consequently no uncertainty band is
drawn.

Run from any directory with:

    python3 scripts/thesis_figures/create_linear_finite_comparison.py

The script writes a vector PDF and a 600-dpi PNG to
Figures/chapter1/control_results.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "FINAL_THESIS_RUN"
OUTPUT_DIR = REPO_ROOT / "Figures" / "chapter1" / "control_results"
OUTPUT_STEM = OUTPUT_DIR / "linear_finite_average_mse_comparison"

RMSE_COLUMN = "corrected_feedback_input_target_rmse_state"
VALIDATION_SEGMENT = "controller_validation"


@dataclass(frozen=True)
class ControllerSpec:
    """Paths, labels, and presentation limits for one controller."""

    name: str
    panel_title: str
    sweep_path: Path
    summary_path: Path
    gain_limits: tuple[float, float]
    finite_s: float | None = None


CONTROLLERS = (
    ControllerSpec(
        name="linear_feedback",
        panel_title="Linear feedback",
        sweep_path=RUN_ROOT / "03_linear_feedback" / "k_sweep.csv",
        summary_path=RUN_ROOT / "03_linear_feedback" / "control_summary.json",
        # Show the refined-search neighbourhood plus its bracketing coarse
        # points. The complete 0.01--2.00 sweep remains in the source CSV.
        gain_limits=(0.65, 1.35),
    ),
    ControllerSpec(
        name="finite_time",
        panel_title=r"Global finite-time feedback ($s=0.9$)",
        sweep_path=(
            RUN_ROOT
            / "04_finite_time"
            / "candidates"
            / "s_0p9"
            / "k_sweep.csv"
        ),
        summary_path=RUN_ROOT / "04_finite_time" / "control_summary.json",
        # As above, focus on the refined validation neighbourhood. The
        # complete 0.01--1.50 sweep remains in the source CSV.
        gain_limits=(0.55, 1.14),
        finite_s=0.9,
    ),
)


@dataclass(frozen=True)
class SweepData:
    gains: np.ndarray
    average_mse: np.ndarray
    selected_gain: float
    evaluation_sample_count: int
    state_dimension: int


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def _load_selected_gain(spec: ControllerSpec) -> tuple[float, int]:
    if not spec.summary_path.is_file():
        raise FileNotFoundError(
            f"Missing controller summary: {spec.summary_path}"
        )

    with spec.summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    if summary.get("controller") != spec.name:
        raise ValueError(
            f"Expected controller {spec.name!r} in {spec.summary_path}, "
            f"found {summary.get('controller')!r}"
        )

    if spec.finite_s is not None and not np.isclose(
        float(summary.get("finite_s", np.nan)), spec.finite_s
    ):
        raise ValueError(
            f"Expected finite_s={spec.finite_s} in {spec.summary_path}"
        )

    target_state = summary.get("target_state")
    if not isinstance(target_state, list) or not target_state:
        raise ValueError(
            f"Missing target_state in {spec.summary_path}"
        )

    return float(summary["best_k"]), len(target_state)


def load_sweep(spec: ControllerSpec) -> SweepData:
    """Load stable controller-validation points and convert RMSE to MSE."""

    if not spec.sweep_path.is_file():
        raise FileNotFoundError(f"Missing sweep CSV: {spec.sweep_path}")

    selected_gain, state_dimension = _load_selected_gain(spec)
    mse_by_gain: dict[float, list[float]] = {}
    sample_counts: set[int] = set()

    with spec.sweep_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {
            "controller",
            "K",
            "stable",
            "metric_segment",
            "evaluation_sample_count",
            RMSE_COLUMN,
        }
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise KeyError(
                f"{spec.sweep_path} is missing columns: {sorted(missing)}"
            )

        for row in reader:
            if row["controller"] != spec.name:
                continue
            if row["metric_segment"] != VALIDATION_SEGMENT:
                continue
            if not _parse_bool(row["stable"]):
                continue

            gain = float(row["K"])
            rmse = float(row[RMSE_COLUMN])
            sample_count = int(float(row["evaluation_sample_count"]))

            if not np.isfinite(gain) or not np.isfinite(rmse):
                continue
            if rmse < 0.0 or sample_count <= 0:
                continue

            # Full-state RMSE is sqrt(mean(error**2)) across validation
            # samples and state variables; its square is average MSE.
            mse_by_gain.setdefault(gain, []).append(rmse**2)
            sample_counts.add(sample_count)

    if not mse_by_gain:
        raise ValueError(
            f"No stable {VALIDATION_SEGMENT} rows found in {spec.sweep_path}"
        )
    if len(sample_counts) != 1:
        raise ValueError(
            f"Inconsistent validation sample counts in {spec.sweep_path}: "
            f"{sorted(sample_counts)}"
        )

    gains = np.array(sorted(mse_by_gain), dtype=float)
    average_mse = np.array(
        [np.mean(mse_by_gain[gain]) for gain in gains], dtype=float
    )

    if not np.any(np.isclose(gains, selected_gain, rtol=0.0, atol=1e-12)):
        raise ValueError(
            f"Selected K={selected_gain} is absent from {spec.sweep_path}"
        )

    return SweepData(
        gains=gains,
        average_mse=average_mse,
        selected_gain=selected_gain,
        evaluation_sample_count=sample_counts.pop(),
        state_dimension=state_dimension,
    )


def _presentation_points(
    data: SweepData,
    gain_limits: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = gain_limits
    mask = (data.gains >= lower) & (data.gains <= upper)
    gains = data.gains[mask]
    average_mse = data.average_mse[mask]

    if gains.size < 3:
        raise ValueError(
            f"Fewer than three sweep points fall inside K={gain_limits}"
        )
    if not (gains[0] <= data.selected_gain <= gains[-1]):
        raise ValueError(
            f"Selected K={data.selected_gain} falls outside K={gain_limits}"
        )

    return gains, average_mse


def _selected_mse(data: SweepData) -> float:
    index = int(np.argmin(np.abs(data.gains - data.selected_gain)))
    return float(data.average_mse[index])


def _draw_panel(
    axis: plt.Axes,
    spec: ControllerSpec,
    data: SweepData,
    panel_label: str,
) -> None:
    gains, average_mse = _presentation_points(data, spec.gain_limits)
    selected_mse = _selected_mse(data)
    colour = "#1f77b4"

    axis.plot(
        gains,
        average_mse,
        color=colour,
        linewidth=1.8,
        marker="o",
        markersize=4.0,
        markerfacecolor=colour,
        markeredgewidth=0.0,
        zorder=2,
    )

    axis.axvline(
        data.selected_gain,
        color=colour,
        linestyle="--",
        linewidth=1.15,
        alpha=0.9,
        zorder=1,
    )
    axis.scatter(
        [data.selected_gain],
        [selected_mse],
        marker="*",
        s=150,
        facecolor=colour,
        edgecolor="black",
        linewidth=0.7,
        clip_on=False,
        zorder=4,
    )
    axis.annotate(
        rf"$K^\ast={data.selected_gain:.3f}$",
        xy=(data.selected_gain, selected_mse),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="bottom",
    )

    axis.set_title(
        f"({panel_label}) {spec.panel_title}",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=6,
    )
    axis.set_xlabel(r"Feedback gain $K$", fontsize=10)
    axis.set_ylabel("Average full-state MSE", fontsize=10)
    axis.ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )
    axis.tick_params(axis="both", labelsize=9)
    axis.grid(True, color="#b0b0b0", linewidth=0.65, alpha=0.30)
    axis.set_axisbelow(True)
    axis.set_ylim(bottom=0.0)

    span = float(gains[-1] - gains[0])
    axis.set_xlim(gains[0] - 0.025 * span, gains[-1] + 0.025 * span)

    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def create_figure() -> tuple[Path, Path]:
    datasets = [load_sweep(spec) for spec in CONTROLLERS]

    sample_counts = {data.evaluation_sample_count for data in datasets}
    state_dimensions = {data.state_dimension for data in datasets}
    if len(sample_counts) != 1 or len(state_dimensions) != 1:
        raise ValueError(
            "The two controllers do not use the same validation dimensions"
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7.9, 6.8),
        sharex=False,
    )

    for axis, spec, data, panel_label in zip(
        axes, CONTROLLERS, datasets, ("a", "b")
    ):
        _draw_panel(axis, spec, data, panel_label)

    figure.suptitle(
        "Effect of feedback gain on average regulation error",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )

    sample_count = sample_counts.pop()
    state_dimension = state_dimensions.pop()
    figure.text(
        0.5,
        0.018,
        (
            f"Validation sweep: MSE averaged over {sample_count:,} samples "
            f"and {state_dimension} state variables; one realization per "
            r"$K$ (no uncertainty band)."
        ),
        ha="center",
        va="bottom",
        fontsize=7.5,
    )

    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.095,
        top=0.875,
        hspace=0.48,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_pdf = OUTPUT_STEM.with_suffix(".pdf")
    output_png = OUTPUT_STEM.with_suffix(".png")

    figure.savefig(output_pdf, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(
        output_png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)

    return output_pdf, output_png


def main() -> None:
    output_pdf, output_png = create_figure()
    print(f"Saved vector PDF: {output_pdf}")
    print(f"Saved 600-dpi PNG: {output_png}")


if __name__ == "__main__":
    main()
