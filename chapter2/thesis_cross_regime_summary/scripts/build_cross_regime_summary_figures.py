#!/usr/bin/env python3
"""Build the presentation-only Chapter 2 cross-regime summary package.

This script reads frozen JSON/CSV/NPZ evaluation artifacts and creates four
figures plus two table formats. It does not import model, training,
optimisation, or evaluation code and cannot generate new predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "chapter2" / "cross_regime_improvement" / "results"
RECORDS_PATH = SOURCE_ROOT / "evaluation_records.json"
AGGREGATE_PATH = SOURCE_ROOT / "aggregate_results.json"
SOURCE_TABLE_PATH = SOURCE_ROOT / "train_test_summary.csv"
RAW_ARRAY_ROOT = SOURCE_ROOT / "raw_arrays"
FIGURE_ROOT = PACKAGE_ROOT / "figures"
TABLE_ROOT = PACKAGE_ROOT / "tables"

DIRECTIONS = ("RR", "RC", "CC", "CR")
TRAIN_REGIME = {"RR": "Regular", "RC": "Regular", "CC": "Chaotic", "CR": "Chaotic"}
TEST_REGIME = {"RR": "Regular", "RC": "Chaotic", "CC": "Chaotic", "CR": "Regular"}
TEST_CURRENTS = {
    "RR": (1.67, 3.29, 3.50),
    "RC": (3.20, 3.34),
    "CC": (3.20, 3.34),
    "CR": (1.67, 3.29, 3.50),
}
TRAIN_CURRENTS = {
    "Regular": (1.67, 3.29, 3.50),
    "Chaotic": (3.20, 3.34),
}
EXPECTED_COUNTS = {"RR": 45, "RC": 30, "CC": 30, "CR": 45}

TRUTH_COLOUR = "#222222"
PREDICTION_COLOUR = "#0072B2"
REGULAR_COLOUR = "#4C78A8"
CHAOTIC_COLOUR = "#F28E2B"
DIRECTION_COLOURS = {
    "RR": "#4C78A8",
    "RC": "#F2A541",
    "CC": "#59A14F",
    "CR": "#E15759",
}

FIGURE_STEMS = (
    "01_cross_regime_task_setup",
    "02_regular_trained_results",
    "03_chaotic_trained_results",
    "04_cross_regime_summary_metrics",
)


class SummaryBuildError(RuntimeError):
    """Raised when frozen inputs or generated outputs fail validation."""


def load_json(path: Path) -> dict[str, Any]:
    """Load a strict JSON object, rejecting NaN and Infinity tokens."""

    def reject_constant(token: str) -> None:
        raise SummaryBuildError(f"non-standard JSON constant {token!r} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise SummaryBuildError(f"expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def direction_records(records: Sequence[Mapping[str, Any]], code: str) -> list[Mapping[str, Any]]:
    return [
        record
        for record in records
        if record.get("family") == "fixed_short" and record.get("matrix_code") == code
    ]


def summarise(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    finite_nrmse = [
        float(record["metrics"]["nrmse_state"])
        for record in records
        if record["metrics"].get("nrmse_state") is not None
    ]
    if not finite_nrmse:
        raise SummaryBuildError("a direction contains no finite NRMSE values")
    vpt = [float(record["metrics"]["valid_prediction_time"]) for record in records]
    return {
        "record_count": len(records),
        "finite_nrmse_count": len(finite_nrmse),
        "median_nrmse": statistics.median(finite_nrmse),
        "iqr_nrmse": float(np.percentile(finite_nrmse, 75) - np.percentile(finite_nrmse, 25)),
        "median_vpt": statistics.median(vpt),
        "divergence_count": sum(bool(record["metrics"]["diverged"]) for record in records),
        "divergence_rate": statistics.fmean(bool(record["metrics"]["diverged"]) for record in records),
        "numerical_failure_count": sum(bool(record["numerical_failure"]) for record in records),
        "numerical_failure_rate": statistics.fmean(bool(record["numerical_failure"]) for record in records),
    }


def close_enough(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-14)


def validate_source_table() -> None:
    with SOURCE_TABLE_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for code in DIRECTIONS:
        selected = [row for row in rows if row["matrix_code"] == code]
        actual_currents = tuple(sorted(float(row["current"]) for row in selected))
        if actual_currents != tuple(sorted(TEST_CURRENTS[code])):
            raise SummaryBuildError(
                f"source CSV currents for {code} are {actual_currents}, expected {TEST_CURRENTS[code]}"
            )
        if sum(int(row["record_count"]) for row in selected) != EXPECTED_COUNTS[code]:
            raise SummaryBuildError(f"source CSV record count mismatch for {code}")


def load_and_validate() -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, float | int]],
    dict[str, dict[str, Any]],
]:
    evaluation = load_json(RECORDS_PATH)
    aggregate = load_json(AGGREGATE_PATH)
    if evaluation.get("status") != "complete":
        raise SummaryBuildError("the final evaluation is not marked complete")
    records = evaluation.get("records")
    if not isinstance(records, list) or len(records) != 345:
        raise SummaryBuildError("expected the complete 345-record final evaluation")
    record_ids = [str(record.get("record_id")) for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise SummaryBuildError("final evaluation record IDs are not unique")
    if aggregate.get("record_count") != 345 or aggregate.get("schema") != "chapter2_cross_regime_improvement_aggregate_v1":
        raise SummaryBuildError("unexpected aggregate-results schema or record count")

    summaries: dict[str, dict[str, float | int]] = {}
    representatives: dict[str, dict[str, Any]] = {}
    for code in DIRECTIONS:
        selected = direction_records(records, code)
        if len(selected) != EXPECTED_COUNTS[code]:
            raise SummaryBuildError(
                f"{code} has {len(selected)} fixed-short records; expected {EXPECTED_COUNTS[code]}"
            )
        actual_currents = tuple(sorted({float(record["current"]) for record in selected}))
        if actual_currents != tuple(sorted(TEST_CURRENTS[code])):
            raise SummaryBuildError(f"unexpected tested currents for {code}: {actual_currents}")

        computed = summarise(selected)
        stored = aggregate.get("matrix", {}).get(code)
        if not isinstance(stored, dict):
            raise SummaryBuildError(f"aggregate results have no {code} cell")
        for key in (
            "record_count",
            "finite_nrmse_count",
            "median_nrmse",
            "iqr_nrmse",
            "median_vpt",
            "divergence_count",
            "divergence_rate",
            "numerical_failure_count",
            "numerical_failure_rate",
        ):
            if isinstance(computed[key], int):
                matches = computed[key] == stored.get(key)
            else:
                matches = stored.get(key) is not None and close_enough(
                    float(computed[key]), float(stored[key])
                )
            if not matches:
                raise SummaryBuildError(
                    f"recomputed {code} {key}={computed[key]!r} does not match aggregate {stored.get(key)!r}"
                )
        summaries[code] = computed

        median = float(computed["median_nrmse"])
        representative = min(
            selected,
            key=lambda record: (
                abs(float(record["aggregate_nrmse_value"]) - median),
                str(record["record_id"]),
            ),
        )
        metric_nrmse = representative["metrics"].get("nrmse_state")
        if metric_nrmse is None or not close_enough(
            float(representative["aggregate_nrmse_value"]), float(metric_nrmse)
        ):
            raise SummaryBuildError(f"representative NRMSE fields disagree for {code}")
        expected_name = f"{representative['record_id']}.npz"
        array_path = RAW_ARRAY_ROOT / expected_name
        if not array_path.is_file():
            raise SummaryBuildError(f"missing saved arrays for {code}: {array_path}")
        if sha256(array_path) != representative["raw_arrays_sha256"]:
            raise SummaryBuildError(f"saved-array hash mismatch for {code}: {array_path}")
        representatives[code] = {**representative, "validated_array_path": array_path}

    validate_source_table()
    return records, summaries, representatives


def load_arrays(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = Path(record["validated_array_path"])
    with np.load(path, allow_pickle=False) as archive:
        required = {"time", "targets", "predictions"}
        if not required.issubset(archive.files):
            raise SummaryBuildError(f"{path} lacks required arrays: {sorted(required - set(archive.files))}")
        arrays = {name: archive[name].copy() for name in archive.files}
    if arrays["targets"].ndim != 2 or arrays["targets"].shape[1] < 1:
        raise SummaryBuildError(f"unexpected target shape in {path}")
    if arrays["predictions"].shape != arrays["targets"].shape:
        raise SummaryBuildError(f"target/prediction shape mismatch in {path}")
    if arrays["time"].shape[0] != arrays["targets"].shape[0]:
        raise SummaryBuildError(f"time/trajectory length mismatch in {path}")
    return arrays


def save_figure(fig: Any, stem: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_ROOT / f"{stem}.png"
    pdf_path = FIGURE_ROOT / f"{stem}.pdf"
    fig.savefig(
        png_path,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": stem, "Software": "Matplotlib"},
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": stem,
            "Creator": "build_cross_regime_summary_figures.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def rounded_box(
    axis: Any,
    xy: tuple[float, float],
    width: float,
    height: float,
    colour: str,
    title: str,
    subtitle: str,
) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5,
        edgecolor=colour,
        facecolor=colour,
        alpha=0.12,
    )
    axis.add_patch(box)
    axis.text(x + width / 2, y + height * 0.63, title, ha="center", va="center", fontsize=14, weight="bold", color=colour)
    axis.text(x + width / 2, y + height * 0.28, subtitle, ha="center", va="center", fontsize=11, color="#333333")


def task_setup_figure() -> None:
    fig, axis = plt.subplots(figsize=(11.2, 6.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    fig.suptitle("Chapter 2 cross-regime prediction task", fontsize=18, weight="bold", y=0.98)
    axis.text(
        0.5,
        0.91,
        "Can an ESN trained in one dynamical regime predict the other regime?",
        ha="center",
        va="center",
        fontsize=13,
        color="#333333",
    )

    rounded_box(axis, (0.05, 0.57), 0.31, 0.22, REGULAR_COLOUR, "Train on REGULAR", "currents 1.67, 3.29, 3.50")
    rounded_box(axis, (0.64, 0.57), 0.31, 0.22, CHAOTIC_COLOUR, "Test on CHAOTIC", "target currents 3.20, 3.34")
    rounded_box(axis, (0.05, 0.24), 0.31, 0.22, CHAOTIC_COLOUR, "Train on CHAOTIC", "currents 3.20, 3.34")
    rounded_box(axis, (0.64, 0.24), 0.31, 0.22, REGULAR_COLOUR, "Test on REGULAR", "target currents 1.67, 3.29, 3.50")

    for y, label, colour in ((0.68, "RC", DIRECTION_COLOURS["RC"]), (0.35, "CR", DIRECTION_COLOURS["CR"])):
        arrow = FancyArrowPatch(
            (0.38, y),
            (0.62, y),
            arrowstyle="-|>",
            mutation_scale=24,
            linewidth=2.4,
            color=colour,
        )
        axis.add_patch(arrow)
        axis.text(0.50, y + 0.045, label, ha="center", va="center", fontsize=15, weight="bold", color=colour)

    axis.text(
        0.5,
        0.10,
        "Within-regime controls:  RR = Regular → Regular     •     CC = Chaotic → Chaotic",
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F4F4F4", "edgecolor": "#C9C9C9"},
    )
    save_figure(fig, FIGURE_STEMS[0])


def divergence_label(record: Mapping[str, Any]) -> str:
    return "Yes" if bool(record["metrics"]["diverged"]) else "No"


def trajectory_panel(axis: Any, code: str, record: Mapping[str, Any], panel_label: str) -> None:
    arrays = load_arrays(record)
    time = arrays["time"] - arrays["time"][0]
    axis.plot(time, arrays["targets"][:, 0], color=TRUTH_COLOUR, lw=1.05, label="True x(t)", zorder=2)
    axis.plot(
        time,
        arrays["predictions"][:, 0],
        color=PREDICTION_COLOUR,
        lw=0.90,
        alpha=0.90,
        label="Predicted x(t)",
        zorder=3,
    )
    axis.set_xlim(float(time[0]), float(time[-1]))
    axis.set_ylabel("x(t)")
    axis.set_title(
        f"{panel_label}  {code}: {TRAIN_REGIME[code]} → {TEST_REGIME[code]}"
        f"  |  target current I = {float(record['current']):.2f}",
        loc="left",
        weight="bold",
    )
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    metrics = record["metrics"]
    metric_text = (
        f"NRMSE = {float(metrics['nrmse_state']):.4g}\n"
        f"VPT = {float(metrics['valid_prediction_time']):.2f}\n"
        f"Diverged: {divergence_label(record)}"
    )
    axis.text(
        0.985,
        0.94,
        metric_text,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "white", "edgecolor": "#BFBFBF", "alpha": 0.93},
        zorder=5,
    )


def trained_results_figure(
    representatives: Mapping[str, Mapping[str, Any]],
    codes: tuple[str, str],
    training_regime: str,
    stem: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.1), sharex=False)
    for axis, code, panel_label in zip(axes, codes, ("A", "B")):
        trajectory_panel(axis, code, representatives[code], panel_label)
    axes[0].legend(loc="upper left", ncol=2, frameon=False)
    axes[-1].set_xlabel("Forecast time")
    axes[0].set_xlabel("Forecast time")
    currents = ", ".join(f"{current:.2f}" for current in TRAIN_CURRENTS[training_regime])
    fig.suptitle(
        f"ESN trained on {training_regime.lower()} currents: {currents}",
        fontsize=16,
        weight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.8)
    save_figure(fig, stem)


def annotate_bars(axis: Any, bars: Sequence[Any], values: Sequence[float], formatter: Any) -> None:
    for bar, value in zip(bars, values):
        axis.annotate(
            formatter(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )


def summary_metrics_figure(summaries: Mapping[str, Mapping[str, float | int]]) -> None:
    codes = list(DIRECTIONS)
    colours = [DIRECTION_COLOURS[code] for code in codes]
    nrmse = [float(summaries[code]["median_nrmse"]) for code in codes]
    vpt = [float(summaries[code]["median_vpt"]) for code in codes]
    divergence = [100.0 * float(summaries[code]["divergence_rate"]) for code in codes]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.8))
    panels = (
        (axes[0], nrmse, "A  Median NRMSE (log scale)", "NRMSE (log scale)"),
        (axes[1], vpt, "B  Median valid prediction time", "VPT (time units)"),
        (axes[2], divergence, "C  Divergence rate", "Divergent rollouts (%)"),
    )
    for axis, values, title, ylabel in panels:
        bars = axis.bar(codes, values, color=colours, width=0.67, edgecolor="white", linewidth=0.7)
        axis.set_title(title, weight="bold", pad=10)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)
        if axis is axes[0]:
            axis.set_yscale("log")
            axis.set_ylim(min(nrmse) / 2.5, max(nrmse) * 2.5)
            annotate_bars(axis, bars, values, lambda value: f"{value:.4g}")
        elif axis is axes[1]:
            axis.set_ylim(0, max(vpt) * 1.18)
            annotate_bars(axis, bars, values, lambda value: f"{value:.2f}")
        else:
            axis.set_ylim(0, 100)
            annotate_bars(axis, bars, values, lambda value: f"{value:.0f}%")
    fig.suptitle("Cross-regime performance summary", fontsize=16, weight="bold", y=0.995)
    fig.text(
        0.5,
        0.015,
        "RR: regular→regular   RC: regular→chaotic   CC: chaotic→chaotic   CR: chaotic→regular",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.95), w_pad=2.2)
    save_figure(fig, FIGURE_STEMS[3])


def compact_number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 10:
        return f"{value:.4g}"
    return f"{value:.6g}"


def table_rows(
    summaries: Mapping[str, Mapping[str, float | int]],
    representatives: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    rows = []
    for code in DIRECTIONS:
        tested = "; ".join(f"{current:.2f}" for current in TEST_CURRENTS[code])
        representative = float(representatives[code]["current"])
        summary = summaries[code]
        rows.append(
            {
                "Direction": code,
                "Train regime": TRAIN_REGIME[code],
                "Test regime": TEST_REGIME[code],
                "Representative current(s) or tested currents": (
                    f"Tested: {tested}; representative: {representative:.2f}"
                ),
                "Median NRMSE": compact_number(float(summary["median_nrmse"])),
                "IQR NRMSE": compact_number(float(summary["iqr_nrmse"])),
                "Median VPT": f"{float(summary['median_vpt']):.2f}",
                "Divergence rate": f"{100.0 * float(summary['divergence_rate']):.1f}%",
                "Numerical failure rate": f"{100.0 * float(summary['numerical_failure_rate']):.1f}%",
            }
        )
    return rows


def write_tables(
    summaries: Mapping[str, Mapping[str, float | int]],
    representatives: Mapping[str, Mapping[str, Any]],
) -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = table_rows(summaries, representatives)
    fieldnames = list(rows[0])
    csv_path = TABLE_ROOT / "01_cross_regime_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    md_path = TABLE_ROOT / "01_cross_regime_summary.md"
    markdown = [
        "# Cross-regime summary",
        "",
        "All metrics are aggregated over the saved fixed-short final-evaluation rollouts. IQR is Q3 − Q1.",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    markdown.extend("| " + " | ".join(row[field] for field in fieldnames) + " |" for row in rows)
    markdown.append("")
    md_path.write_text("\n".join(markdown), encoding="utf-8")


def validate_outputs() -> None:
    expected = [FIGURE_ROOT / f"{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("png", "pdf")]
    expected += [
        TABLE_ROOT / "01_cross_regime_summary.csv",
        TABLE_ROOT / "01_cross_regime_summary.md",
    ]
    missing = [path for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SummaryBuildError(f"missing or empty generated outputs: {missing}")


def report(
    summaries: Mapping[str, Mapping[str, float | int]],
    representatives: Mapping[str, Mapping[str, Any]],
) -> None:
    print("Validated source artifacts:")
    for path in (RECORDS_PATH, AGGREGATE_PATH, SOURCE_TABLE_PATH):
        print(f"  {path.relative_to(PROJECT_ROOT)}  sha256={sha256(path)}")
    print("Deterministic representatives:")
    for code in DIRECTIONS:
        record = representatives[code]
        print(
            f"  {code}: {record['record_id']} | current={float(record['current']):.2f} "
            f"seed={record['seed']} window={record['window']} "
            f"NRMSE={float(record['aggregate_nrmse_value']):.12g}"
        )
    print("Direction summaries:")
    for code in DIRECTIONS:
        item = summaries[code]
        print(
            f"  {code}: median NRMSE={float(item['median_nrmse']):.12g}, "
            f"IQR={float(item['iqr_nrmse']):.12g}, median VPT={float(item['median_vpt']):.12g}, "
            f"divergence={100.0 * float(item['divergence_rate']):.1f}%, "
            f"numerical failure={100.0 * float(item['numerical_failure_rate']):.1f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate frozen sources and report selections without writing outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summaries, representatives = load_and_validate()
    report(summaries, representatives)
    if args.check_only:
        return
    style()
    task_setup_figure()
    trained_results_figure(
        representatives,
        ("RR", "RC"),
        "Regular",
        FIGURE_STEMS[1],
    )
    trained_results_figure(
        representatives,
        ("CC", "CR"),
        "Chaotic",
        FIGURE_STEMS[2],
    )
    summary_metrics_figure(summaries)
    write_tables(summaries, representatives)
    validate_outputs()
    print(f"Wrote summary package outputs under {PACKAGE_ROOT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
