"""Thesis-quality figures for locked cross-regime improvement results."""

from __future__ import annotations

from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chapter2.esn_data import load_fixed_trajectory
from chapter2.esn_optimisation import load_strict_json

from .config import (
    ALL_CURRENTS,
    CHAOTIC_CURRENTS,
    CORRECTION_ROOT,
    FIGURE_ROOT,
    MATRIX_CODES,
    REGULAR_CURRENTS,
    SCENARIO_LABELS,
    SCENARIO_TRAINING_CURRENTS,
    SEEDS,
    regime_for_current,
)
from .experiment import RAW_RESULTS
from .protection import refuse_existing


STATE_NAMES = ("x", "y", "z")
SCENARIO_ORDER = tuple(SCENARIO_TRAINING_CURRENTS)
REGIME_ORDER = ("regular", "chaotic")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: Any, stem: str, output_root: Path) -> list[str]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = [output_root / f"{stem}.png", output_root / f"{stem}.pdf"]
    for path in paths:
        refuse_existing(path)
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [str(path) for path in paths]


def _arrays(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = Path(str(record["raw_arrays_path"]))
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _representative(
    records: Sequence[Mapping[str, Any]], scenario: str, current: float,
    *, family: str = "fixed_short", window: int | None = 1, seed: int = 42,
) -> Mapping[str, Any]:
    return next(
        item for item in records
        if item["family"] == family and item["scenario"] == scenario
        and int(item["seed"]) == seed and item.get("current") == current
        and (window is None or item.get("window") == window)
    )


def regime_overview(output_root: Path = FIGURE_ROOT) -> list[str]:
    fig, axes = plt.subplots(5, 1, figsize=(9, 8), sharex=False)
    for axis, current in zip(axes, ALL_CURRENTS):
        trajectory = load_fixed_trajectory(current)
        section = slice(70_000, 72_000)
        colour = "tab:blue" if current in REGULAR_CURRENTS else "tab:orange"
        axis.plot(trajectory.time[section], trajectory.states[section, 0], color=colour, lw=0.8)
        axis.set_ylabel("x")
        axis.set_title(f"I={current:.2f} — {regime_for_current(current)}", loc="left")
    axes[-1].set_xlabel("time")
    fig.suptitle("Fixed-current Hindmarsh–Rose regime overview", y=1.01)
    fig.tight_layout()
    return _save(fig, "01_regime_overview", output_root)


def prediction_figure(
    records: Sequence[Mapping[str, Any]], scenario: str, current: float,
    stem: str, title: str, output_root: Path = FIGURE_ROOT,
) -> list[str]:
    arrays = _arrays(_representative(records, scenario, current))
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(arrays["time"], arrays["targets"][:, index], color="black", lw=0.8, label="truth")
        axis.plot(arrays["time"], arrays["predictions"][:, index], color="tab:blue", lw=0.75, alpha=0.9, label="prediction")
        axis.set_ylabel(STATE_NAMES[index])
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("time")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, stem, output_root)


def mixed_prediction_figure(
    records: Sequence[Mapping[str, Any]], output_root: Path = FIGURE_ROOT
) -> list[str]:
    choices = ((1.67, "regular"), (3.34, "chaotic"))
    fig, axes = plt.subplots(3, 2, figsize=(11, 7), sharex="col")
    for column, (current, regime) in enumerate(choices):
        arrays = _arrays(_representative(records, "mixed_shuffled", current))
        for state, axis in enumerate(axes[:, column]):
            axis.plot(arrays["time"], arrays["targets"][:, state], color="black", lw=0.75)
            axis.plot(arrays["time"], arrays["predictions"][:, state], color="tab:green", lw=0.7)
            axis.set_ylabel(STATE_NAMES[state])
        axes[0, column].set_title(f"{regime.title()} target, I={current:.2f}")
        axes[-1, column].set_xlabel("time")
    fig.suptitle("Mixed-shuffled-trained ESN: regular and chaotic targets")
    fig.tight_layout()
    return _save(fig, "06_mixed_predictions", output_root)


def _cell_records(records: Sequence[Mapping[str, Any]], scenario: str, regime: str) -> list[Mapping[str, Any]]:
    return [
        item for item in records
        if item["family"] == "fixed_short" and item["scenario"] == scenario
        and regime_for_current(float(item["current"])) == regime
    ]


def _heatmap(
    values: np.ndarray, title: str, label: str, stem: str,
    output_root: Path, *, fmt: str = ".3g", cmap: str = "viridis",
) -> list[str]:
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    image = axis.imshow(values, aspect="auto", cmap=cmap)
    axis.set_xticks(range(2), ["Regular target", "Chaotic target"])
    axis.set_yticks(range(3), [SCENARIO_LABELS[x] for x in SCENARIO_ORDER])
    for row in range(3):
        for column in range(2):
            axis.text(column, row, format(values[row, column], fmt), ha="center", va="center", color="white" if values[row, column] > np.nanmedian(values) else "black")
    fig.colorbar(image, ax=axis, label=label)
    axis.set_title(title)
    fig.tight_layout()
    return _save(fig, stem, output_root)


def matrix_heatmaps(
    records: Sequence[Mapping[str, Any]], output_root: Path = FIGURE_ROOT
) -> list[str]:
    nrmse = np.zeros((3, 2))
    divergence = np.zeros((3, 2))
    failure = np.zeros((3, 2))
    for row, scenario in enumerate(SCENARIO_ORDER):
        for column, regime in enumerate(REGIME_ORDER):
            items = _cell_records(records, scenario, regime)
            finite = [float(x["metrics"]["nrmse_state"]) for x in items if x["metrics"]["nrmse_state"] is not None]
            nrmse[row, column] = np.log10(max(statistics.median(finite), 1e-12)) if finite else 6.0
            divergence[row, column] = np.mean([bool(x["metrics"]["diverged"]) for x in items])
            failure[row, column] = np.mean([bool(x["numerical_failure"]) for x in items])
    paths = []
    paths += _heatmap(nrmse, "Train–test matrix: median NRMSE", "log10 median NRMSE", "07_train_test_nrmse_heatmap", output_root)
    paths += _heatmap(divergence, "Train–test matrix: divergence", "divergence rate", "08_divergence_heatmap", output_root, fmt=".1%", cmap="magma")
    paths += _heatmap(failure, "Train–test matrix: numerical failure", "failure rate", "09_numerical_failure_heatmap", output_root, fmt=".1%", cmap="magma")
    return paths


def vpt_comparison(records: Sequence[Mapping[str, Any]], output_root: Path = FIGURE_ROOT) -> list[str]:
    codes = ("RR", "RC", "CC", "CR", "MR", "MC")
    values = []
    for code in codes:
        items = [x for x in records if x["family"] == "fixed_short" and x["matrix_code"] == code]
        values.append(statistics.median(float(x["metrics"]["valid_prediction_time"]) for x in items))
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(codes, values, color=["tab:blue", "tab:orange", "tab:orange", "tab:blue", "tab:green", "tab:green"])
    axis.set_ylabel("median valid prediction time")
    axis.set_xlabel("train–test cell")
    axis.set_title("Valid prediction time across regimes")
    fig.tight_layout()
    return _save(fig, "10_vpt_comparison", output_root)


def seed_robustness(records: Sequence[Mapping[str, Any]], output_root: Path = FIGURE_ROOT) -> list[str]:
    codes = ("RR", "RC", "CC", "CR", "MR", "MC")
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for x_index, code in enumerate(codes):
        for seed_index, seed in enumerate(SEEDS):
            items = [item for item in records if item["family"] == "fixed_short" and item["matrix_code"] == code and int(item["seed"]) == seed]
            finite = [float(item["metrics"]["nrmse_state"]) for item in items if item["metrics"]["nrmse_state"] is not None]
            value = statistics.median(finite) if finite else 1_000_000.0
            axis.scatter(x_index + (seed_index - 2) * 0.045, value, s=24, label=str(seed) if x_index == 0 else None)
    axis.set_yscale("log")
    axis.set_xticks(range(len(codes)), codes)
    axis.set_ylabel("per-seed median NRMSE (log scale)")
    axis.set_title("Five-seed robustness")
    axis.legend(title="seed", ncol=5, loc="upper center")
    fig.tight_layout()
    return _save(fig, "11_seed_robustness", output_root)


def phase_portraits(records: Sequence[Mapping[str, Any]], output_root: Path = FIGURE_ROOT) -> list[str]:
    choices = (
        ("regular_to_chaotic", 3.20, "RC"),
        ("chaotic_to_regular", 3.29, "CR"),
        ("mixed_shuffled", 3.34, "MC"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for column, (scenario, current, code) in enumerate(choices):
        arrays = _arrays(_representative(records, scenario, current))
        stride = max(1, len(arrays["targets"]) // 5000)
        for row, z_index in enumerate((1, 2)):
            axis = axes[row, column]
            axis.plot(arrays["targets"][::stride, 0], arrays["targets"][::stride, z_index], color="black", lw=0.6, label="truth")
            axis.plot(arrays["predictions"][::stride, 0], arrays["predictions"][::stride, z_index], color="tab:blue", lw=0.55, alpha=0.8, label="prediction")
            axis.set_xlabel("x")
            axis.set_ylabel(STATE_NAMES[z_index])
        axes[0, column].set_title(f"{code}, I={current:.2f}")
    axes[0, 0].legend()
    fig.suptitle("True and predicted phase portraits")
    fig.tight_layout()
    return _save(fig, "12_phase_portraits", output_root)


def continuous_switch_figure(records: Sequence[Mapping[str, Any]], output_root: Path = FIGURE_ROOT) -> list[str]:
    schedules = ("regular_then_chaotic", "chaotic_then_regular", "alternating_mixed")
    fig, axes = plt.subplots(4, 3, figsize=(13, 9), sharex="col")
    for column, schedule in enumerate(schedules):
        record = next(x for x in records if x["family"] == "continuous" and x["scenario"] == "mixed_shuffled" and int(x["seed"]) == 42 and x["schedule"] == schedule)
        arrays = _arrays(record)
        stride = max(1, len(arrays["time"]) // 10_000)
        time = arrays["time"][::stride]
        axes[0, column].plot(time, arrays["current"][::stride], color="tab:purple", lw=0.75)
        axes[1, column].plot(time, arrays["targets"][::stride, 0], color="black", lw=0.65)
        axes[2, column].plot(time, arrays["predictions"][::stride, 0], color="tab:green", lw=0.65)
        axes[3, column].plot(time, arrays["pointwise_normalised_error"][::stride], color="tab:red", lw=0.65)
        changes = np.flatnonzero(np.diff(arrays["current"]) != 0) + 1
        for axis in axes[:, column]:
            for index in changes:
                axis.axvline(arrays["time"][index], color="grey", ls="--", lw=0.55)
        axes[0, column].set_title(schedule.replace("_", " "))
        axes[-1, column].set_xlabel("time")
    for axis, label in zip(axes[:, 0], ("I(t)", "true x(t)", "predicted x(t)", "normalised error")):
        axis.set_ylabel(label)
    fig.suptitle("Continuous-current switch experiments: mixed-shuffled model, seed 42")
    fig.tight_layout()
    return _save(fig, "13_continuous_switches", output_root)


def _overall_summary(records: Sequence[Mapping[str, Any]], scenario: str) -> dict[str, float]:
    items = [x for x in records if x["scenario"] == scenario]
    finite = [float(x["metrics"]["nrmse_state"]) for x in items if x["metrics"]["nrmse_state"] is not None]
    return {
        "numerical failure rate": np.mean([bool(x["numerical_failure"]) for x in items]),
        "divergence rate": np.mean([bool(x["metrics"]["diverged"]) for x in items]),
        "median NRMSE": statistics.median(finite) if finite else 1_000_000.0,
        "median VPT": statistics.median(float(x["metrics"]["valid_prediction_time"]) for x in items),
    }


def baseline_vs_improved(records: Sequence[Mapping[str, Any]], output_root: Path = FIGURE_ROOT) -> list[str]:
    baseline = load_strict_json(CORRECTION_ROOT / "corrected_results.json")["records"]
    metrics = ("numerical failure rate", "divergence rate", "median NRMSE", "median VPT")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    x = np.arange(3)
    for axis, metric in zip(axes, metrics):
        old = [_overall_summary(baseline, scenario)[metric] for scenario in SCENARIO_ORDER]
        new = [_overall_summary(records, scenario)[metric] for scenario in SCENARIO_ORDER]
        axis.bar(x - 0.18, old, width=0.36, label="baseline", color="0.65")
        axis.bar(x + 0.18, new, width=0.36, label="improved", color="tab:blue")
        axis.set_xticks(x, ["regular", "chaotic", "mixed"], rotation=25)
        axis.set_title(metric)
        if metric == "median NRMSE":
            axis.set_yscale("log")
    axes[0].legend()
    fig.suptitle("Frozen baseline versus scenario-optimised ESN")
    fig.tight_layout()
    return _save(fig, "14_baseline_vs_improved", output_root)


def generate_all_figures(
    *, records_path: Path = RAW_RESULTS, output_root: Path = FIGURE_ROOT
) -> list[str]:
    _style()
    raw = load_strict_json(records_path)
    if raw.get("status") != "complete" or len(raw.get("records", [])) != 345:
        raise ValueError("complete 345-record improved evaluation is required")
    records = raw["records"]
    generated = []
    generated += regime_overview(output_root)
    generated += prediction_figure(records, "regular_to_chaotic", 1.67, "02_rr_prediction", "RR: regular-trained model on regular target", output_root)
    generated += prediction_figure(records, "regular_to_chaotic", 3.20, "03_rc_prediction", "RC: regular-trained model on chaotic target", output_root)
    generated += prediction_figure(records, "chaotic_to_regular", 3.34, "04_cc_prediction", "CC: chaotic-trained model on chaotic target", output_root)
    generated += prediction_figure(records, "chaotic_to_regular", 3.29, "05_cr_prediction", "CR: chaotic-trained model on regular target", output_root)
    generated += mixed_prediction_figure(records, output_root)
    generated += matrix_heatmaps(records, output_root)
    generated += vpt_comparison(records, output_root)
    generated += seed_robustness(records, output_root)
    generated += phase_portraits(records, output_root)
    generated += continuous_switch_figure(records, output_root)
    generated += baseline_vs_improved(records, output_root)
    return generated
