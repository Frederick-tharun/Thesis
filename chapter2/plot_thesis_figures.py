"""Generate the four frozen, thesis-facing Chapter 2 figures.

The module is deliberately presentation-only.  It loads the verified Step 8
JSON and NPZ artifacts, validates the predetermined selections, and never
trains a model or produces a new prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from chapter2.config_ch2 import DT
from chapter2.esn_config import FINAL_SEEDS, TRAIN_CURRENTS, UNSEEN_CURRENTS
from chapter2.esn_data import file_sha256
from chapter2.esn_optimisation import ORDINARY_BASELINE, PARAMETER_AWARE


CHAPTER2_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CHAPTER2_ROOT.parent
FINAL_RESULTS = CHAPTER2_ROOT / "final_results"
RAW_RESULTS_PATH = FINAL_RESULTS / "step8_raw_results.json"
AGGREGATE_RESULTS_PATH = FINAL_RESULTS / "step8_aggregate_results.json"
EVALUATION_MANIFEST_PATH = FINAL_RESULTS / "step8_evaluation_manifest.json"
SELECTION_LOCK_PATH = FINAL_RESULTS / "selected_model.json"
AUDIT_PATH = FINAL_RESULTS / "step8_seed_stability_audit.json"
OUTPUT_DIR = FINAL_RESULTS / "figures_thesis"

REPRESENTATIVE_SEED = 42
REPRESENTATIVE_WINDOW = 1
TRANSITION_HALF_WIDTH_TIME = 10.0
MODEL_TYPES = (PARAMETER_AWARE, ORDINARY_BASELINE)
FAMILY_ORDER = (
    "known_short",
    "known_long",
    "unseen_short",
    "unseen_long",
    "continuous",
)
FAMILY_LABELS = {
    "known_short": "Known-current short",
    "known_long": "Known-current long",
    "unseen_short": "Unseen-current short",
    "unseen_long": "Unseen-current long",
    "continuous": "Changing current",
}
EXPECTED_COUNTS = {
    "known_short": 90,
    "known_long": 30,
    "unseen_short": 60,
    "unseen_long": 20,
    "continuous": 10,
}

TRUTH = "#222222"
AWARE = "#0072B2"
BASELINE = "#D55E00"
CURRENT = "#009E73"
BOUNDARY = "#8A8A8A"
MODEL_COLORS = {PARAMETER_AWARE: AWARE, ORDINARY_BASELINE: BASELINE}
MODEL_LABELS = {
    PARAMETER_AWARE: "Parameter-aware ESN",
    ORDINARY_BASELINE: "Ordinary ESN",
}

FIGURE_STEMS = (
    "01_fixed_current_predictions_known_and_unseen",
    "02_continuous_current_prediction",
    "03_current_transition_tracking",
    "04_overall_predictive_performance",
)
EXPECTED_OUTPUT_NAMES = frozenset(
    [f"{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("pdf", "png")]
    + ["figure_manifest.json"]
)


class ThesisFigureError(RuntimeError):
    """Raised when verified inputs or deterministic figure selections fail."""


@dataclass(frozen=True)
class FixedCase:
    current: float
    family: str
    classification: str
    aware_record: Mapping[str, Any]
    baseline_record: Mapping[str, Any]


def strict_json_text(value: Any) -> str:
    """Return indented standards-compliant JSON and reject non-finite values."""
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def load_strict_json(path: Path) -> dict[str, Any]:
    """Load JSON while rejecting the non-standard NaN and Infinity tokens."""

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token!r} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ThesisFigureError(f"expected a JSON object in {path}")
    return value


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _expected_record_ids() -> set[str]:
    def tag(current: float) -> str:
        return f"{current:.2f}".replace(".", "p")

    expected: set[str] = set()
    for model_type in MODEL_TYPES:
        for seed in FINAL_SEEDS:
            for family, currents in (
                ("known_short", TRAIN_CURRENTS),
                ("unseen_short", UNSEEN_CURRENTS),
            ):
                for current in currents:
                    for window in (1, 2, 3):
                        expected.add(
                            f"{family}__{model_type}__seed_{seed}__I_{tag(current)}"
                            f"__window_{window}"
                        )
            for family, currents in (
                ("known_long", TRAIN_CURRENTS),
                ("unseen_long", UNSEEN_CURRENTS),
            ):
                for current in currents:
                    expected.add(
                        f"{family}__{model_type}__seed_{seed}__I_{tag(current)}"
                    )
            expected.add(f"continuous__{model_type}__seed_{seed}")
    return expected


def validate_record_matrix(records: Sequence[Mapping[str, Any]]) -> None:
    """Require the exact frozen 210-record Step 8 matrix."""
    identifiers = [str(item.get("record_id")) for item in records]
    if len(identifiers) != 210 or len(set(identifiers)) != 210:
        raise ThesisFigureError("Step 8 must contain 210 unique record identifiers")
    expected = _expected_record_ids()
    actual = set(identifiers)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ThesisFigureError(f"record matrix mismatch; missing={missing}, extra={extra}")
    counts = {
        family: sum(item.get("family") == family for item in records)
        for family in EXPECTED_COUNTS
    }
    if counts != EXPECTED_COUNTS:
        raise ThesisFigureError(f"record family counts mismatch: {counts}")


def _verified_hash(
    audit: Mapping[str, Any], group: str, path: Path
) -> str:
    hashes = audit["artifact_hashes_before_and_after_audit"]["hashes"][group]
    key = project_relative(path)
    if key not in hashes:
        raise ThesisFigureError(f"audit contains no verified hash for {key}")
    return str(hashes[key])


def _require_verified_file(
    audit: Mapping[str, Any], group: str, path: Path
) -> None:
    expected = _verified_hash(audit, group, path)
    actual = file_sha256(path)
    if actual != expected:
        raise ThesisFigureError(
            f"verified input hash mismatch for {path}: expected {expected}, got {actual}"
        )


def load_verified_results() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load the frozen results after checking their audit and selection provenance."""
    audit = load_strict_json(AUDIT_PATH)
    if audit.get("verdict") != "AUDIT PASSED — SCIENTIFIC SEED INSTABILITY PRESERVED":
        raise ThesisFigureError("the required Step 8 audit verdict is absent")
    if not audit.get("artifact_hashes_before_and_after_audit", {}).get("unchanged"):
        raise ThesisFigureError("the Step 8 audit did not verify immutable artifacts")

    for group, path in (
        ("selection_lock", SELECTION_LOCK_PATH),
        ("evaluation_manifest", EVALUATION_MANIFEST_PATH),
        ("aggregate_json_and_csv", RAW_RESULTS_PATH),
        ("aggregate_json_and_csv", AGGREGATE_RESULTS_PATH),
    ):
        _require_verified_file(audit, group, path)

    evaluation = load_strict_json(EVALUATION_MANIFEST_PATH)
    selection = evaluation.get("plot_selection", {})
    if selection != {
        "representative_seed": REPRESENTATIVE_SEED,
        "representative_window": REPRESENTATIVE_WINDOW,
        "cherry_picking_allowed": False,
        "formats": ["png", "pdf"],
    }:
        raise ThesisFigureError(f"unexpected locked plot selection: {selection}")
    if tuple(int(value) for value in evaluation.get("seeds", ())) != tuple(FINAL_SEEDS):
        raise ThesisFigureError("evaluation manifest seeds do not match the frozen seeds")

    raw = load_strict_json(RAW_RESULTS_PATH)
    records = raw.get("records")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ThesisFigureError("raw-results JSON has no valid records list")
    validate_record_matrix(records)
    return records, evaluation, audit


def select_exact_record(
    records: Sequence[Mapping[str, Any]],
    *,
    family: str,
    model_type: str,
    seed: int,
    current: float | None,
    window: int | None,
) -> Mapping[str, Any]:
    """Select exactly one record and never substitute a nearby case."""
    selected = [
        item
        for item in records
        if item.get("family") == family
        and item.get("model_type") == model_type
        and int(item.get("seed")) == seed
        and (
            (current is None and item.get("current") is None)
            or (
                current is not None
                and item.get("current") is not None
                and math.isclose(float(item["current"]), current, rel_tol=0.0, abs_tol=1e-12)
            )
        )
        and item.get("window") == window
    ]
    if len(selected) != 1:
        raise ThesisFigureError(
            "expected exactly one record for "
            f"family={family}, model={model_type}, seed={seed}, "
            f"current={current}, window={window}; found {len(selected)}"
        )
    return selected[0]


def select_fixed_cases(
    records: Sequence[Mapping[str, Any]], evaluation: Mapping[str, Any]
) -> list[FixedCase]:
    """Return the five predetermined seed-42, first-window current cases."""
    windows = evaluation.get("fixed_short_windows")
    if not isinstance(windows, list):
        raise ThesisFigureError("locked fixed-current windows are absent")
    first = [item for item in windows if item.get("window") == REPRESENTATIVE_WINDOW]
    if len(first) != 1 or first[0].get("start") != 70_000:
        raise ThesisFigureError("the predetermined first evaluation window is not locked")
    expected_forecast = [72_000, 80_000]
    if first[0].get("forecast_range") != expected_forecast:
        raise ThesisFigureError("the first locked forecast range is not [72,000, 80,000)")

    specifications = [
        *( (float(current), "known_short", "used during training") for current in TRAIN_CURRENTS ),
        *( (float(current), "unseen_short", "unseen current") for current in UNSEEN_CURRENTS ),
    ]
    cases: list[FixedCase] = []
    for current, family, classification in specifications:
        aware = select_exact_record(
            records,
            family=family,
            model_type=PARAMETER_AWARE,
            seed=REPRESENTATIVE_SEED,
            current=current,
            window=REPRESENTATIVE_WINDOW,
        )
        baseline = select_exact_record(
            records,
            family=family,
            model_type=ORDINARY_BASELINE,
            seed=REPRESENTATIVE_SEED,
            current=current,
            window=REPRESENTATIVE_WINDOW,
        )
        for item in (aware, baseline):
            if item.get("forecast_range") != expected_forecast:
                raise ThesisFigureError(f"wrong fixed forecast range in {item['record_id']}")
            if item.get("warmup_range") != [70_000, 72_000]:
                raise ThesisFigureError(f"wrong fixed warm-up range in {item['record_id']}")
        cases.append(FixedCase(current, family, classification, aware, baseline))
    if [case.current for case in cases] != [1.67, 3.20, 3.50, 3.29, 3.34]:
        raise ThesisFigureError("fixed-current cases are not in the required order")
    return cases


def select_continuous_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        model_type: select_exact_record(
            records,
            family="continuous",
            model_type=model_type,
            seed=REPRESENTATIVE_SEED,
            current=None,
            window=None,
        )
        for model_type in MODEL_TYPES
    }


def load_record_arrays(item: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Load one verified raw NPZ without pickle or object arrays."""
    path = PROJECT_ROOT / str(item["raw_arrays_path"])
    actual_hash = file_sha256(path)
    if actual_hash != item.get("raw_arrays_sha256"):
        raise ThesisFigureError(f"raw-array hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        expected_names = (
            "predictions",
            "targets",
            "pointwise_normalised_error",
            "time",
            "current",
        )
        if tuple(saved.files) != expected_names:
            raise ThesisFigureError(f"raw-array schema mismatch: {path}")
        if any(saved[name].dtype.kind == "O" for name in saved.files):
            raise ThesisFigureError(f"unsafe object array in {path}")
        arrays = {name: np.asarray(saved[name]).copy() for name in saved.files}

    count = int(item["forecast_range"][1]) - int(item["forecast_range"][0])
    if arrays["predictions"].shape != (count, 3):
        raise ThesisFigureError(f"prediction shape mismatch in {path}")
    if arrays["targets"].shape != (count, 3):
        raise ThesisFigureError(f"target shape mismatch in {path}")
    for name in ("time", "current", "pointwise_normalised_error"):
        if arrays[name].shape != (count,):
            raise ThesisFigureError(f"{name} shape mismatch in {path}")
    if count > 1 and not np.allclose(np.diff(arrays["time"]), DT, rtol=0.0, atol=2e-10):
        raise ThesisFigureError(f"time step mismatch in {path}")
    return arrays


def require_finite(label: str, *arrays: np.ndarray) -> None:
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ThesisFigureError(f"unexpected non-finite plotting values in {label}")


def validate_array_pair(
    aware: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray], label: str
) -> None:
    for name in ("targets", "time", "current"):
        if not np.array_equal(aware[name], baseline[name]):
            raise ThesisFigureError(f"aware/baseline {name} alignment mismatch for {label}")


def derive_change_indices(current: np.ndarray) -> np.ndarray:
    """Derive current boundaries solely from the stored evaluated current array."""
    values = np.asarray(current)
    if values.ndim != 1:
        raise ThesisFigureError("continuous current must be one-dimensional")
    change_indices = np.flatnonzero(values[1:] != values[:-1]) + 1
    if len(change_indices) != 4:
        raise ThesisFigureError(
            f"expected exactly four continuous current changes, found {len(change_indices)}"
        )
    return change_indices.astype(int, copy=False)


def transition_slices(
    current: np.ndarray,
    *,
    dt: float = DT,
    half_width_time: float = TRANSITION_HALF_WIDTH_TIME,
) -> list[tuple[int, int, int]]:
    """Return four exact, inclusive-endpoint ±10-time-unit transition slices."""
    steps_float = half_width_time / dt
    half_steps = int(round(steps_float))
    if not math.isclose(steps_float, half_steps, rel_tol=0.0, abs_tol=1e-10):
        raise ThesisFigureError("transition half-width is not an integer sample count")
    slices = []
    for boundary in derive_change_indices(current):
        start = int(boundary) - half_steps
        stop = int(boundary) + half_steps + 1
        if start < 0 or stop > len(current):
            raise ThesisFigureError(
                f"cannot form ±{half_width_time:g} interval at boundary {boundary}"
            )
        slices.append((start, stop, int(boundary)))
    return slices


def performance_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute Figure 4 medians from every frozen record in each family/model."""
    result: dict[str, dict[str, dict[str, Any]]] = {}
    expected_per_model = {family: count // 2 for family, count in EXPECTED_COUNTS.items()}
    for family in FAMILY_ORDER:
        result[family] = {}
        for model_type in MODEL_TYPES:
            group = [
                item
                for item in records
                if item.get("family") == family and item.get("model_type") == model_type
            ]
            if len(group) != expected_per_model[family]:
                raise ThesisFigureError(
                    f"wrong Figure 4 record count for {family}/{model_type}: {len(group)}"
                )
            seeds = sorted({int(item["seed"]) for item in group})
            if seeds != sorted(FINAL_SEEDS):
                raise ThesisFigureError(f"Figure 4 seed set mismatch for {family}/{model_type}")
            nrmse = np.asarray([float(item["aggregate_nrmse_value"]) for item in group])
            normalized_vpt = []
            horizons = []
            for item in group:
                start, stop = (int(value) for value in item["forecast_range"])
                horizon = stop - start
                valid = int(item["metrics"]["valid_prediction_steps"])
                if horizon <= 0 or valid < 0 or valid > horizon:
                    raise ThesisFigureError(f"invalid VPT horizon in {item['record_id']}")
                horizons.append(horizon)
                normalized_vpt.append(100.0 * valid / horizon)
            vpt = np.asarray(normalized_vpt, dtype=float)
            require_finite(f"Figure 4 {family}/{model_type}", nrmse, vpt)
            result[family][model_type] = {
                "median_nrmse": float(np.median(nrmse)),
                "median_normalized_vpt_percent": float(np.median(vpt)),
                "record_count": len(group),
                "seed_values": seeds,
                "divergence_count": sum(bool(item["metrics"]["diverged"]) for item in group),
                "evaluated_prediction_steps": sorted(set(horizons)),
                "record_ids": [str(item["record_id"]) for item in group],
            }
    return result


def _select_font() -> str:
    installed = {item.name for item in font_manager.fontManager.ttflist}
    for candidate in ("Times New Roman", "STIX Two Text", "STIXGeneral", "DejaVu Serif"):
        if candidate in installed:
            return candidate
    raise ThesisFigureError("none of the approved thesis fonts is installed")


def apply_thesis_style() -> str:
    """Apply the one shared, colour-blind-friendly style for all four figures."""
    font = _select_font()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 10.5,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.formatter.useoffset": False,
            "path.simplify": True,
            "path.simplify_threshold": 0.05,
        }
    )
    return font


def _style_axis(axis: Any, *, grid: str | None = None) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid is not None:
        axis.grid(axis=grid, color="#D8D8D8", linewidth=0.45, alpha=0.55)
        axis.set_axisbelow(True)


def common_limits(arrays: Iterable[np.ndarray], *, margin_fraction: float = 0.04) -> tuple[float, float]:
    flattened = [np.ravel(np.asarray(values, dtype=float)) for values in arrays]
    require_finite("common axis limits", *flattened)
    low = min(float(np.min(values)) for values in flattened)
    high = max(float(np.max(values)) for values in flattened)
    span = high - low
    margin = margin_fraction * span if span > 0.0 else max(abs(low) * margin_fraction, 0.1)
    return low - margin, high + margin


def _legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=TRUTH, linewidth=1.1, label="Ground truth"),
        Line2D([0], [0], color=AWARE, linewidth=1.1, label=MODEL_LABELS[PARAMETER_AWARE]),
        Line2D([0], [0], color=BASELINE, linewidth=1.1, label=MODEL_LABELS[ORDINARY_BASELINE]),
    ]


def _save_figure(fig: Any, output_dir: Path, stem: str) -> list[dict[str, Any]]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.045,
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "Chapter 2 thesis plotter"},
    )
    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.045,
        facecolor="white",
    )
    plt.close(fig)
    return [
        {
            "filename": path.name,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in (pdf, png)
    ]


def plot_fixed_current_predictions(
    cases: Sequence[FixedCase], output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    loaded: list[dict[str, Any]] = []
    limit_inputs: list[np.ndarray] = []
    for case in cases:
        aware = load_record_arrays(case.aware_record)
        baseline = load_record_arrays(case.baseline_record)
        validate_array_pair(aware, baseline, f"I={case.current:.2f}")
        if len(aware["time"]) != 8_000:
            raise ThesisFigureError(f"Figure 1 requires 8,000 predictions at I={case.current:.2f}")
        require_finite(
            f"Figure 1 I={case.current:.2f}",
            aware["targets"],
            aware["predictions"],
            baseline["predictions"],
        )
        loaded.append({"case": case, "aware": aware, "baseline": baseline})
        limit_inputs.extend(
            (aware["targets"][:, 0], aware["predictions"][:, 0], baseline["predictions"][:, 0])
        )
    y_limits = common_limits(limit_inputs)

    fig = plt.figure(figsize=(6.5, 8.4), constrained_layout=True)
    grid = fig.add_gridspec(7, 1, height_ratios=(0.23, 1, 1, 1, 0.18, 1, 1))
    heading_known = fig.add_subplot(grid[0])
    heading_unseen = fig.add_subplot(grid[4])
    for axis in (heading_known, heading_unseen):
        axis.set_axis_off()
    heading_known.text(
        0.0, 0.5, "Currents used during training", transform=heading_known.transAxes,
        ha="left", va="center", color="#4A4A4A", fontweight="semibold"
    )
    heading_known.legend(
        handles=_legend_handles(), loc="center right", ncol=3, frameon=False,
        handlelength=2.0, columnspacing=1.1
    )
    heading_unseen.text(
        0.0, 0.5, "Unseen currents", transform=heading_unseen.transAxes,
        ha="left", va="center", color="#4A4A4A", fontweight="semibold"
    )

    positions = (1, 2, 3, 5, 6)
    axes = []
    for index, position in enumerate(positions):
        share = axes[0] if axes else None
        axes.append(fig.add_subplot(grid[position], sharex=share, sharey=share))
    letters = "abcde"
    for index, (axis, item) in enumerate(zip(axes, loaded)):
        case = item["case"]
        aware = item["aware"]
        baseline = item["baseline"]
        relative_time = np.arange(len(aware["time"]), dtype=float) * DT
        axis.plot(
            relative_time, baseline["predictions"][:, 0], color=BASELINE,
            linewidth=0.62, alpha=0.9, zorder=1
        )
        axis.plot(
            relative_time, aware["predictions"][:, 0], color=AWARE,
            linewidth=0.66, alpha=0.95, zorder=2
        )
        axis.plot(
            relative_time, aware["targets"][:, 0], color=TRUTH,
            linewidth=0.58, alpha=0.92, zorder=3
        )
        axis.set_ylim(y_limits)
        axis.set_title(
            f"({letters[index]})  $I={case.current:.2f}$ — {case.classification}",
            loc="left", pad=2.5
        )
        _style_axis(axis, grid="y")
        if index < len(axes) - 1:
            axis.tick_params(labelbottom=False)
    fig.supxlabel("Time since prediction start, $t$")
    fig.supylabel("Membrane potential, $x(t)$")
    artifacts = _save_figure(fig, output_dir, FIGURE_STEMS[0])
    return artifacts, loaded


def _load_continuous_pair(
    continuous_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    arrays = {
        model_type: load_record_arrays(continuous_records[model_type])
        for model_type in MODEL_TYPES
    }
    validate_array_pair(
        arrays[PARAMETER_AWARE], arrays[ORDINARY_BASELINE], "continuous seed 42"
    )
    require_finite(
        "continuous trajectory",
        arrays[PARAMETER_AWARE]["targets"],
        arrays[PARAMETER_AWARE]["predictions"],
        arrays[ORDINARY_BASELINE]["predictions"],
        arrays[PARAMETER_AWARE]["current"],
        arrays[PARAMETER_AWARE]["time"],
    )
    return arrays


def plot_continuous_prediction(
    continuous_records: Mapping[str, Mapping[str, Any]], output_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]], np.ndarray]:
    arrays = _load_continuous_pair(continuous_records)
    aware = arrays[PARAMETER_AWARE]
    baseline = arrays[ORDINARY_BASELINE]
    changes = derive_change_indices(aware["current"])
    time = aware["time"]
    boundary_times = time[changes]
    y_limits = common_limits(
        (aware["targets"][:, 0], aware["predictions"][:, 0], baseline["predictions"][:, 0])
    )

    fig, axes = plt.subplots(
        4, 1, figsize=(6.5, 7.2), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": (0.75, 1, 1, 1)}
    )
    axes[0].step(time, aware["current"], where="post", color=CURRENT, linewidth=0.9)
    axes[0].set_ylabel("External current, $I(t)$")
    axes[0].set_title("(a)  Supplied changing current", loc="left")
    segment_starts = np.r_[0, changes]
    segment_stops = np.r_[changes, len(time)]
    for start, stop in zip(segment_starts, segment_stops):
        midpoint = (int(start) + int(stop) - 1) // 2
        level = float(aware["current"][int(start)])
        axes[0].text(
            time[midpoint], level + 0.08, f"$I={level:.2f}$", color=CURRENT,
            ha="center", va="bottom", fontsize=8
        )

    panels = (
        (axes[1], aware["targets"][:, 0], TRUTH, "(b)  Ground-truth membrane potential"),
        (axes[2], aware["predictions"][:, 0], AWARE, "(c)  Parameter-aware ESN prediction"),
        (axes[3], baseline["predictions"][:, 0], BASELINE, "(d)  Ordinary ESN prediction"),
    )
    for axis, values, color, title in panels:
        axis.plot(time, values, color=color, linewidth=0.52)
        axis.set_ylim(y_limits)
        axis.set_ylabel("Membrane potential, $x(t)$")
        axis.set_title(title, loc="left")
    for axis in axes:
        for boundary_time in boundary_times:
            axis.axvline(boundary_time, color=BOUNDARY, linewidth=0.55, alpha=0.55)
        _style_axis(axis, grid=None)
    axes[-1].set_xlabel("Time, $t$")
    artifacts = _save_figure(fig, output_dir, FIGURE_STEMS[1])
    return artifacts, arrays, changes


def plot_transition_tracking(
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    changes: np.ndarray,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[tuple[int, int, int]]]:
    aware = arrays[PARAMETER_AWARE]
    baseline = arrays[ORDINARY_BASELINE]
    slices = transition_slices(aware["current"])
    if not np.array_equal(changes, np.asarray([item[2] for item in slices])):
        raise ThesisFigureError("continuous boundary selection changed between figures")
    limit_inputs = []
    for start, stop, _ in slices:
        limit_inputs.extend(
            (
                aware["targets"][start:stop, 0],
                aware["predictions"][start:stop, 0],
                baseline["predictions"][start:stop, 0],
            )
        )
    y_limits = common_limits(limit_inputs)

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.2), sharex=True, sharey=True, constrained_layout=True)
    letters = "abcd"
    for index, (axis, (start, stop, boundary)) in enumerate(zip(axes.flat, slices)):
        relative_time = (np.arange(start, stop, dtype=float) - boundary) * DT
        axis.plot(
            relative_time, baseline["predictions"][start:stop, 0], color=BASELINE,
            linewidth=0.78, alpha=0.9, zorder=1
        )
        axis.plot(
            relative_time, aware["predictions"][start:stop, 0], color=AWARE,
            linewidth=0.82, alpha=0.95, zorder=2
        )
        axis.plot(
            relative_time, aware["targets"][start:stop, 0], color=TRUTH,
            linewidth=0.72, alpha=0.95, zorder=3
        )
        axis.axvline(0.0, color=BOUNDARY, linestyle="--", linewidth=0.8)
        old = float(aware["current"][boundary - 1])
        new = float(aware["current"][boundary])
        axis.set_title(f"({letters[index]})  $I={old:.2f} \\rightarrow {new:.2f}$", loc="left")
        axis.set_xlim(-TRANSITION_HALF_WIDTH_TIME, TRANSITION_HALF_WIDTH_TIME)
        axis.set_ylim(y_limits)
        _style_axis(axis, grid="y")
    fig.supxlabel("Time relative to current change")
    fig.supylabel("Membrane potential, $x(t)$")
    fig.legend(
        handles=_legend_handles(), loc="outside upper center", ncol=3,
        frameon=False, handlelength=2.2, columnspacing=1.2
    )
    artifacts = _save_figure(fig, output_dir, FIGURE_STEMS[2])
    return artifacts, slices


def _bar_label(axis: Any, bar: Any, value: float, *, percent: bool) -> None:
    label = f"{value:.1f}%" if percent else f"{value:.3f}"
    offset = 0.012 * axis.get_xlim()[1]
    axis.text(
        bar.get_width() + offset,
        bar.get_y() + bar.get_height() / 2,
        label,
        ha="left",
        va="center",
        fontsize=7.8,
        color="#333333",
    )


def plot_overall_performance(
    summary: Mapping[str, Mapping[str, Mapping[str, Any]]], output_dir: Path
) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 5.0), sharey=True, constrained_layout=True)
    y = np.arange(len(FAMILY_ORDER), dtype=float)
    height = 0.34
    panels = (
        (axes[0], "median_nrmse", "(a) Median all-state NRMSE\nLower is better", False),
        (
            axes[1],
            "median_normalized_vpt_percent",
            "(b) Median valid prediction horizon\nHigher is better",
            True,
        ),
    )
    for axis, field, title, percent in panels:
        aware_values = np.asarray([summary[family][PARAMETER_AWARE][field] for family in FAMILY_ORDER])
        baseline_values = np.asarray([summary[family][ORDINARY_BASELINE][field] for family in FAMILY_ORDER])
        maximum = max(float(np.max(aware_values)), float(np.max(baseline_values)))
        axis.set_xlim(0.0, maximum * 1.20 if maximum > 0.0 else 1.0)
        baseline_bars = axis.barh(
            y + height / 2, baseline_values, height, color=BASELINE,
            label=MODEL_LABELS[ORDINARY_BASELINE]
        )
        aware_bars = axis.barh(
            y - height / 2, aware_values, height, color=AWARE,
            label=MODEL_LABELS[PARAMETER_AWARE]
        )
        for bar, value in zip(baseline_bars, baseline_values):
            _bar_label(axis, bar, float(value), percent=percent)
        for bar, value in zip(aware_bars, aware_values):
            _bar_label(axis, bar, float(value), percent=percent)
        axis.set_title(title, loc="left")
        axis.set_yticks(y, [FAMILY_LABELS[family] for family in FAMILY_ORDER])
        axis.set_xlabel(
            "Valid prediction horizon (%)" if percent else "All-state NRMSE"
        )
        _style_axis(axis, grid="x")
    axes[0].invert_yaxis()
    fig.legend(
        handles=[
            Line2D([0], [0], color=AWARE, linewidth=6, label=MODEL_LABELS[PARAMETER_AWARE]),
            Line2D([0], [0], color=BASELINE, linewidth=6, label=MODEL_LABELS[ORDINARY_BASELINE]),
        ],
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    return _save_figure(fig, output_dir, FIGURE_STEMS[3])


def clear_output_directory(output_dir: Path) -> None:
    """Clear only the dedicated thesis-figure directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(output_dir.iterdir()):
        if path.is_dir():
            raise ThesisFigureError(f"unexpected directory inside figure output: {path}")
        path.unlink()


def _record_hashes(items: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {str(item["record_id"]): str(item["raw_arrays_sha256"]) for item in items}


def _output_record(
    stem: str,
    files: Sequence[Mapping[str, Any]],
    *,
    purpose: str,
    families: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    seed: int | Sequence[int],
    current_values: Sequence[float] | None,
    classification: Mapping[str, Sequence[float]] | str,
    window_identifier: Any,
    window_start: Any,
    forecast_length: Any,
    change_indices: Sequence[int] | None,
    aggregate_field: Any,
) -> dict[str, Any]:
    return {
        "figure": stem,
        "files": list(files),
        "scientific_purpose": purpose,
        "source_result_families": list(families),
        "record_identifiers": [str(item["record_id"]) for item in records],
        "model_names": [MODEL_LABELS[PARAMETER_AWARE], MODEL_LABELS[ORDINARY_BASELINE]],
        "seed": seed,
        "current_values": list(current_values) if current_values is not None else None,
        "known_unseen_classification": classification,
        "window_identifier": window_identifier,
        "window_start": window_start,
        "forecast_length": forecast_length,
        "dt": DT,
        "current_change_indices_in_scored_array": list(change_indices) if change_indices is not None else None,
        "aggregate_field_used": aggregate_field,
        "raw_result_hashes": _record_hashes(records),
        "selection_statement": (
            "Record selection was fixed before benchmark access and was not based on performance."
        ),
    }


def build_figure_manifest(
    *,
    font: str,
    fixed_cases: Sequence[FixedCase],
    fixed_loaded: Sequence[Mapping[str, Any]],
    continuous_records: Mapping[str, Mapping[str, Any]],
    changes: np.ndarray,
    transition_windows: Sequence[tuple[int, int, int]],
    performance: Mapping[str, Mapping[str, Mapping[str, Any]]],
    all_records: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    fixed_records = [
        record
        for case in fixed_cases
        for record in (case.aware_record, case.baseline_record)
    ]
    continuous_items = [continuous_records[model] for model in MODEL_TYPES]
    performance_values = {
        family: {
            model: {
                key: performance[family][model][key]
                for key in (
                    "median_nrmse",
                    "median_normalized_vpt_percent",
                    "record_count",
                    "seed_values",
                    "divergence_count",
                    "evaluated_prediction_steps",
                )
            }
            for model in MODEL_TYPES
        }
        for family in FAMILY_ORDER
    }
    figures = [
        _output_record(
            FIGURE_STEMS[0], artifacts[FIGURE_STEMS[0]],
            purpose=(
                "Compare complete held-out autonomous membrane-potential forecasts at "
                "currents used during training and at unseen currents."
            ),
            families=("known_short", "unseen_short"),
            records=fixed_records,
            seed=REPRESENTATIVE_SEED,
            current_values=[case.current for case in fixed_cases],
            classification={
                "used_during_training": [float(value) for value in TRAIN_CURRENTS],
                "unseen": [float(value) for value in UNSEEN_CURRENTS],
            },
            window_identifier=REPRESENTATIVE_WINDOW,
            window_start=70_000,
            forecast_length=8_000,
            change_indices=None,
            aggregate_field=None,
        ),
        _output_record(
            FIGURE_STEMS[1], artifacts[FIGURE_STEMS[1]],
            purpose="Show full-trajectory response to the supplied changing current without boundary resets.",
            families=("continuous",),
            records=continuous_items,
            seed=REPRESENTATIVE_SEED,
            current_values=[1.67, 3.29, 3.50, 3.34, 3.20],
            classification="predefined supporting changing-current benchmark",
            window_identifier=None,
            window_start=None,
            forecast_length=497_999,
            change_indices=[int(value) for value in changes],
            aggregate_field=None,
        ),
        _output_record(
            FIGURE_STEMS[2], artifacts[FIGURE_STEMS[2]],
            purpose="Compare both ESNs locally across each of the four supplied-current changes.",
            families=("continuous",),
            records=continuous_items,
            seed=REPRESENTATIVE_SEED,
            current_values=[1.67, 3.29, 3.50, 3.34, 3.20],
            classification="predefined supporting changing-current benchmark",
            window_identifier="four predetermined ±10-time-unit boundary views",
            window_start=[start for start, _, _ in transition_windows],
            forecast_length=[stop - start for start, stop, _ in transition_windows],
            change_indices=[int(value) for value in changes],
            aggregate_field=None,
        ),
        _output_record(
            FIGURE_STEMS[3], artifacts[FIGURE_STEMS[3]],
            purpose=(
                "Compare overall predictive accuracy and normalized valid horizon across "
                "all frozen benchmark records."
            ),
            families=FAMILY_ORDER,
            records=all_records,
            seed=[int(value) for value in FINAL_SEEDS],
            current_values=[1.67, 3.20, 3.50, 3.29, 3.34],
            classification={
                "known": [float(value) for value in TRAIN_CURRENTS],
                "unseen": [float(value) for value in UNSEEN_CURRENTS],
                "changing_current": [1.67, 3.29, 3.50, 3.34, 3.20],
            },
            window_identifier="all frozen windows and horizons",
            window_start=[70_000, 80_000, 89_999],
            forecast_length={
                "known_short": 8_000,
                "known_long": 27_999,
                "unseen_short": 8_000,
                "unseen_long": 27_999,
                "continuous": 497_999,
            },
            change_indices=[int(value) for value in changes],
            aggregate_field=(
                "family/model median of aggregate_nrmse_value; family/model median of "
                "100 * metrics.valid_prediction_steps / evaluated prediction steps"
            ),
        ),
    ]
    figures[3]["verified_performance_values"] = performance_values
    return {
        "schema": "chapter2_thesis_figures_v1",
        "output_directory": project_relative(OUTPUT_DIR),
        "figure_count": 4,
        "image_file_count": 8,
        "font": font,
        "colors": {
            "truth": TRUTH,
            "parameter_aware": AWARE,
            "ordinary_baseline": BASELINE,
            "current": CURRENT,
            "boundary": BOUNDARY,
        },
        "selection_policy": {
            "representative_seed": REPRESENTATIVE_SEED,
            "representative_window": REPRESENTATIVE_WINDOW,
            "fixed_forecast_range": [72_000, 80_000],
            "performance_selected": False,
            "all_divergent_records_retained_in_figure_4": True,
            "known_and_unseen_combined_visually_but_aggregated_separately": True,
        },
        "source_artifacts": {
            project_relative(RAW_RESULTS_PATH): file_sha256(RAW_RESULTS_PATH),
            project_relative(AGGREGATE_RESULTS_PATH): file_sha256(AGGREGATE_RESULTS_PATH),
            project_relative(EVALUATION_MANIFEST_PATH): file_sha256(EVALUATION_MANIFEST_PATH),
            project_relative(SELECTION_LOCK_PATH): file_sha256(SELECTION_LOCK_PATH),
            project_relative(AUDIT_PATH): file_sha256(AUDIT_PATH),
        },
        "figures": figures,
    }


def generate_thesis_figures(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """Validate the frozen results and generate exactly four PDF/PNG figure pairs."""
    records, evaluation, _ = load_verified_results()
    protected_before = {
        path: file_sha256(path)
        for path in (RAW_RESULTS_PATH, AGGREGATE_RESULTS_PATH, AUDIT_PATH)
    }
    fixed_cases = select_fixed_cases(records, evaluation)
    continuous_records = select_continuous_records(records)
    performance = performance_summary(records)
    font = apply_thesis_style()

    clear_output_directory(output_dir)
    artifacts: dict[str, Sequence[Mapping[str, Any]]] = {}
    fixed_artifacts, fixed_loaded = plot_fixed_current_predictions(fixed_cases, output_dir)
    artifacts[FIGURE_STEMS[0]] = fixed_artifacts
    continuous_artifacts, continuous_arrays, changes = plot_continuous_prediction(
        continuous_records, output_dir
    )
    artifacts[FIGURE_STEMS[1]] = continuous_artifacts
    transition_artifacts, windows = plot_transition_tracking(
        continuous_arrays, changes, output_dir
    )
    artifacts[FIGURE_STEMS[2]] = transition_artifacts
    artifacts[FIGURE_STEMS[3]] = plot_overall_performance(performance, output_dir)

    manifest = build_figure_manifest(
        font=font,
        fixed_cases=fixed_cases,
        fixed_loaded=fixed_loaded,
        continuous_records=continuous_records,
        changes=changes,
        transition_windows=windows,
        performance=performance,
        all_records=records,
        artifacts=artifacts,
    )
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(strict_json_text(manifest), encoding="utf-8")

    actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_names != EXPECTED_OUTPUT_NAMES:
        raise ThesisFigureError(
            f"unexpected thesis-figure output set: {sorted(actual_names)}"
        )
    protected_after = {path: file_sha256(path) for path in protected_before}
    if protected_after != protected_before:
        raise ThesisFigureError("a protected scientific result changed during plotting")
    return manifest


def main() -> None:
    manifest = generate_thesis_figures()
    print(
        f"Generated {manifest['figure_count']} thesis figures "
        f"({manifest['image_file_count']} image files) in {OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
