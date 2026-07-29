from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_ROOT = (
    REPO_ROOT
    / "FINAL_THESIS_RUN"
    / "04_finite_time"
    / "candidates"
)

EXPONENTS = {
    0.3: "s_0p3",
    0.5: "s_0p5",
    0.7: "s_0p7",
    0.8: "s_0p8",
    0.9: "s_0p9",
}

FINAL_SELECTED_S = 0.9

OUTPUT_DIR = REPO_ROOT / "Figures" / "chapter1" / "control_results"
OUTPUT_PNG = OUTPUT_DIR / "finite_time_parameter_sensitivity_attractive.png"
OUTPUT_PDF = OUTPUT_DIR / "finite_time_parameter_sensitivity_attractive.pdf"


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def load_sweep(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load stable validation scores from one exponent-specific gain sweep.
    Duplicate gains from the coarse and refined stages are averaged.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing sweep file: {path}")

    values_by_gain: dict[float, list[float]] = {}

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "K",
            "stable",
            "metric_segment",
            "selection_score",
        }

        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise KeyError(
                f"{path} is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            if row["metric_segment"] != "controller_validation":
                continue

            if not parse_bool(row["stable"]):
                continue

            gain = float(row["K"])
            score = float(row["selection_score"])

            if not np.isfinite(gain) or not np.isfinite(score):
                continue

            values_by_gain.setdefault(gain, []).append(score)

    if not values_by_gain:
        raise ValueError(f"No stable validation results found in {path}")

    gains = np.array(sorted(values_by_gain), dtype=float)

    scores = np.array(
        [
            np.mean(values_by_gain[gain])
            for gain in gains
        ],
        dtype=float,
    )

    return gains, scores


def load_selected_result(path: Path) -> tuple[float, float]:
    """Load the selected gain and validation score for one exponent."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary file: {path}")

    with path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    selected_gain = float(summary["best_k"])

    validation = summary.get("validation_metrics", {})
    selected_score = float(
        validation.get(
            "selection_score",
            summary["selection_metric_value"],
        )
    )

    return selected_gain, selected_score


def nearest_point(
    gains: np.ndarray,
    scores: np.ndarray,
    selected_gain: float,
) -> tuple[float, float]:
    index = int(np.argmin(np.abs(gains - selected_gain)))
    return float(gains[index]), float(scores[index])


def main() -> None:
    sweep_data: dict[
        float,
        tuple[np.ndarray, np.ndarray, float, float],
    ] = {}

    selected_exponents: list[float] = []
    selected_gains: list[float] = []
    selected_scores: list[float] = []

    for exponent, folder in EXPONENTS.items():
        candidate_dir = CANDIDATE_ROOT / folder

        gains, scores = load_sweep(
            candidate_dir / "k_sweep.csv"
        )

        selected_gain, selected_score = load_selected_result(
            candidate_dir / "control_summary.json"
        )

        sweep_data[exponent] = (
            gains,
            scores,
            selected_gain,
            selected_score,
        )

        selected_exponents.append(exponent)
        selected_gains.append(selected_gain)
        selected_scores.append(selected_score)

    figure = plt.figure(
        figsize=(10.8, 6.8),
        constrained_layout=True,
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.75, 1.0],
    )

    landscape_axis = figure.add_subplot(grid[:, 0])
    gain_axis = figure.add_subplot(grid[0, 1])
    score_axis = figure.add_subplot(grid[1, 1])

    # -------------------------------------------------------------
    # Panel (a): validation landscape
    # -------------------------------------------------------------
    scatter_for_colourbar = None

    for exponent in EXPONENTS:
        gains, scores, selected_gain, _ = sweep_data[exponent]

        best_score_for_exponent = float(np.min(scores))

        relative_score = scores / best_score_for_exponent
        log_relative_score = np.log10(relative_score)

        # Better candidates appear larger.
        marker_sizes = 30.0 + 115.0 / (1.0 + log_relative_score)

        y_values = np.full_like(
            gains,
            exponent,
            dtype=float,
        )

        landscape_axis.plot(
            [0.0, 1.5],
            [exponent, exponent],
            linewidth=0.7,
            alpha=0.25,
        )

        scatter_for_colourbar = landscape_axis.scatter(
            gains,
            y_values,
            c=log_relative_score,
            s=marker_sizes,
            cmap="viridis_r",
            edgecolors="none",
            alpha=0.9,
        )

        plotted_gain, _ = nearest_point(
            gains,
            scores,
            selected_gain,
        )

        selected_marker_size = (
            270 if np.isclose(exponent, FINAL_SELECTED_S) else 170
        )

        landscape_axis.scatter(
            plotted_gain,
            exponent,
            marker="*",
            s=selected_marker_size,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

        landscape_axis.annotate(
            rf"$K^\ast={selected_gain:.3f}$",
            xy=(plotted_gain, exponent),
            xytext=(6, 9),
            textcoords="offset points",
            fontsize=8,
        )

    landscape_axis.set_title(
        "(a) Validation landscape",
        loc="left",
        fontweight="bold",
    )

    landscape_axis.set_xlabel(r"Feedback gain $K$")
    landscape_axis.set_ylabel(r"Fractional-power exponent $s$")
    landscape_axis.set_xlim(-0.02, 1.52)
    landscape_axis.set_ylim(0.25, 0.95)
    landscape_axis.set_yticks(list(EXPONENTS.keys()))
    landscape_axis.grid(True, axis="x", alpha=0.25)

    if scatter_for_colourbar is not None:
        colourbar = figure.colorbar(
            scatter_for_colourbar,
            ax=landscape_axis,
            pad=0.02,
        )

        colourbar.set_label(
            r"$\log_{10}\!\left(J/J_{\min,s}\right)$"
        )

    landscape_axis.text(
        0.02,
        0.02,
        "Larger and darker points indicate better gains\n"
        "within the corresponding exponent.",
        transform=landscape_axis.transAxes,
        fontsize=8,
        verticalalignment="bottom",
    )

    # -------------------------------------------------------------
    # Panel (b): selected gain versus exponent
    # -------------------------------------------------------------
    exponent_array = np.asarray(selected_exponents, dtype=float)
    gain_array = np.asarray(selected_gains, dtype=float)

    gain_axis.plot(
        exponent_array,
        gain_array,
        marker="o",
        linewidth=1.8,
    )

    for exponent, gain in zip(
        selected_exponents,
        selected_gains,
    ):
        marker = "*" if np.isclose(exponent, FINAL_SELECTED_S) else "o"
        marker_size = 160 if marker == "*" else 65

        gain_axis.scatter(
            exponent,
            gain,
            marker=marker,
            s=marker_size,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

        gain_axis.annotate(
            f"{gain:.3f}",
            xy=(exponent, gain),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
        )

    gain_axis.set_title(
        r"(b) Selected gain $K^\ast$",
        loc="left",
        fontweight="bold",
    )

    gain_axis.set_xlabel(r"Exponent $s$")
    gain_axis.set_ylabel(r"Selected gain $K^\ast$")
    gain_axis.set_xticks(exponent_array)
    gain_axis.grid(True, alpha=0.25)

    # -------------------------------------------------------------
    # Panel (c): best score versus exponent
    # -------------------------------------------------------------
    score_array = np.asarray(selected_scores, dtype=float)

    score_axis.plot(
        exponent_array,
        score_array,
        marker="o",
        linewidth=1.8,
    )

    for exponent, score in zip(
        selected_exponents,
        selected_scores,
    ):
        marker = "*" if np.isclose(exponent, FINAL_SELECTED_S) else "o"
        marker_size = 160 if marker == "*" else 65

        score_axis.scatter(
            exponent,
            score,
            marker=marker,
            s=marker_size,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

        score_axis.annotate(
            f"{score:.2e}",
            xy=(exponent, score),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=7,
        )

    score_axis.set_yscale("log")

    score_axis.set_title(
        "(c) Best validation score",
        loc="left",
        fontweight="bold",
    )

    score_axis.set_xlabel(r"Exponent $s$")
    score_axis.set_ylabel(r"Minimum score $J_{\mathrm{reg}}$")
    score_axis.set_xticks(exponent_array)
    score_axis.grid(True, which="both", alpha=0.25)

    figure.suptitle(
        r"Interaction between gain $K$ and exponent $s$ "
        r"in global finite-time feedback",
        fontweight="bold",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    print(f"Saved PNG: {OUTPUT_PNG}")
    print(f"Saved PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
