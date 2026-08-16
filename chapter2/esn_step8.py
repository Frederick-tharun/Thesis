"""Locked final training and untouched Step 8 evaluation for Chapter 2.

The module enforces the pre-benchmark ordering: header/hash preflight, immutable
selection and protocol locks, prefix-only final training, safe model
serialization, then benchmark access. It never performs model selection.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
from numpy.lib import format as npy_format

from chapter2.config_ch2 import (
    BURST_MIN_GAP_PROMINENCE,
    BURST_MIN_INTERVALS_PER_TIMESCALE,
    BURST_MIN_LOG_ISI_GAP,
    BURST_MIN_SPIKES,
    DT,
    SPIKE_HEIGHT,
    SPIKE_MIN_DISTANCE_STEPS,
    SPIKE_PROMINENCE,
)
from chapter2.dynamics_analysis_ch2 import analyze_spikes_and_bursts, detect_spikes
from chapter2.esn_config import (
    CHAPTER2_ROOT,
    CONTINUOUS_DATASET,
    CONTINUOUS_SWITCH_INDICES,
    FINAL_SEEDS,
    FIXED_DATASETS,
    FIXED_STATE_COUNT,
    LOCKED_DATASETS,
    OUTPUT_DIMENSION,
    PARAMETER_AWARE_INPUT_DIMENSION,
    REQUIRED_ARRAY_KEYS,
    STATE_DIMENSION,
    STEP8_CONTINUOUS_WARMUP_TRANSITIONS,
    STEP8_FINAL_TRAINING_STOP,
    STEP8_FORECAST_TRANSITIONS,
    STEP8_LONG_HORIZON_START,
    STEP8_TRAINING_WASHOUT,
    STEP8_WARMUP_TRANSITIONS,
    STEP8_WINDOW_STARTS,
    TRAIN_CURRENTS,
    UNSEEN_CURRENTS,
    ESNModelConfig,
    LockedDataset,
)
from chapter2.esn_data import (
    ContinuousCurrentTrajectory,
    FixedCurrentTrajectory,
    NumpyStandardScaler,
    StateCurrentScalers,
    file_sha256,
    load_continuous_benchmark,
    load_fixed_trajectory,
)
from chapter2.esn_metrics import (
    COLLAPSE_STD_RATIO_THRESHOLD,
    evaluate_rollout,
    pointwise_normalised_error,
)
from chapter2.esn_model import EchoStateNetwork, TrainingSequence
from chapter2.esn_optimisation import (
    DIVERGENCE_THRESHOLD,
    NONFINITE_FAILURE_SCORE,
    ORDINARY_BASELINE,
    PARAMETER_AWARE,
    VALID_PREDICTION_THRESHOLD,
    atomic_write_json,
    git_state,
    load_strict_json,
)


STEP8_SCHEMA = "chapter2_step8_v1"
MODEL_BUNDLE_SCHEMA = "chapter2_final_esn_bundle_v1"
MODEL_MANIFEST_SCHEMA = "chapter2_final_model_manifest_v1"
SELECTION_LOCK_STATUS = "VALIDATION-SELECTED — LOCKED BEFORE BENCHMARK ACCESS"
FINAL_RESULTS = CHAPTER2_ROOT / "final_results"
FINAL_MODELS = CHAPTER2_ROOT / "final_models"
RAW_ARRAYS = FINAL_RESULTS / "raw_arrays"
SELECTION_LOCK = FINAL_RESULTS / "selected_model.json"
EVALUATION_MANIFEST = FINAL_RESULTS / "step8_evaluation_manifest.json"
STATUS_PATH = FINAL_RESULTS / "step8_status.json"
RAW_RESULTS_PATH = FINAL_RESULTS / "step8_raw_results.json"
AGGREGATE_PATH = FINAL_RESULTS / "step8_aggregate_results.json"
VERIFICATION_PATH = FINAL_RESULTS / "step8_verification.json"
MODEL_MANIFEST_PATH = FINAL_MODELS / "model_manifest.json"
STEP7_ROOT = CHAPTER2_ROOT / "optimisation_results"
STEP7_SELECTION = STEP7_ROOT / "step7_selection.json"
STEP7_HISTORIES = {
    PARAMETER_AWARE: STEP7_ROOT / "step7_parameter_aware_history.json",
    ORDINARY_BASELINE: STEP7_ROOT / "step7_ordinary_baseline_history.json",
}
PRE_RESUME_HASHES = {
    "step7_parameter_aware_history.json": "b762bca77533a71ca74261bde82ae098f72c884c214f910c1591981bafdecce6",
    "step7_ordinary_baseline_history.json": "b001a8976c98667a23422354b6c3b6e2dd1e026ad74451e7e91c62293dd3b537",
    "step7_selection.json": "770b438a267bc76b6b9eac84ecc252578b9f11795948196433442cbc8ffa4239",
}
MODEL_TYPES = (PARAMETER_AWARE, ORDINARY_BASELINE)
FULL_CONFIGURATIONS = {
    PARAMETER_AWARE: {
        "reservoir_size": 100,
        "reservoir_connectivity": 0.08881598524963213,
        "input_scaling": 0.06402022818477646,
        "spectral_radius": 0.4118313967689876,
        "ridge_regularisation": 3.968208883661854e-10,
        "leak_rate": 0.9375840772954693,
        "bias_scaling": 0.1,
        "regularise_bias": False,
        "output_dimension": 3,
    },
    ORDINARY_BASELINE: {
        "reservoir_size": 300,
        "reservoir_connectivity": 1.0,
        "input_scaling": 0.01,
        "spectral_radius": 0.01,
        "ridge_regularisation": 0.01,
        "leak_rate": 1.0,
        "bias_scaling": 0.1,
        "regularise_bias": False,
        "output_dimension": 3,
    },
}
SOURCE_TRIALS = {PARAMETER_AWARE: 4, ORDINARY_BASELINE: 12}
ROBUST_VALIDATION = {
    PARAMETER_AWARE: {
        "mean_objective_nrmse": 0.017368919325506615,
        "worst_current_mean_nrmse": 0.031499040070510936,
        "mean_valid_prediction_steps": 7638.177777777778,
        "divergence_rollout_count": 0,
        "collapse_rollout_count": 0,
    },
    ORDINARY_BASELINE: {
        "mean_objective_nrmse": 0.7071937104502228,
        "worst_current_mean_nrmse": 0.9152808484027574,
        "mean_valid_prediction_steps": 568.2222222222222,
        "divergence_rollout_count": 0,
        "collapse_rollout_count": 0,
    },
}
SPIKE_MATCH_TOLERANCE_STEPS = 20
CONTINUOUS_BOUNDARY_HALF_WINDOW = 2_000
EXPECTED_STATES = (
    "PREFLIGHT",
    "SELECTION_LOCKED",
    "FINAL_MODELS_TRAINED",
    "BENCHMARK_ACCESS_STARTED",
    "KNOWN_HELDOUT_COMPLETE",
    "UNSEEN_COMPLETE",
    "LONG_HORIZON_COMPLETE",
    "CONTINUOUS_COMPLETE",
    "AGGREGATION_COMPLETE",
    "STEP8_COMPLETE",
    "FAILED",
)

# These fields describe when/where a lock was created rather than its frozen
# scientific protocol. Historical locks must retain them, but a resume check
# cannot expect them to equal a later process. ``source_code_hashes`` is also
# historical execution provenance: post-evaluation corrections/hardening make
# the current source intentionally different, while the saved lock hash still
# authenticates the original lock bytes. The Chapter 1 tracked-tree hash is
# historical repository provenance and changes with root documentation/hygiene
# even when the frozen Chapter 2 protocol and data do not.
SELECTION_LOCK_COMPARISON_EXCLUSIONS = (
    "created_at",
    "source_code_hashes",
    "chapter1_tracked_tree_hash",
    "python_and_package_versions",
    "runtime",
    "git",
)
EVALUATION_MANIFEST_COMPARISON_EXCLUSIONS = (
    "created_at",
    "python_and_package_versions",
    "runtime",
)
MODEL_METADATA_COMPARISON_EXCLUSIONS = (
    "trained_at",
    "python_and_package_versions",
)
RAW_ARRAY_KEYS = {
    "predictions",
    "targets",
    "pointwise_normalised_error",
    "time",
    "current",
}


class Step8Error(RuntimeError):
    """Base class for fail-closed Step 8 errors."""


class PreflightError(Step8Error):
    """Raised before benchmark access when a frozen input is inconsistent."""


class BenchmarkAccessError(Step8Error):
    """Raised when benchmark access is attempted before all gates exist."""


def _scientific_view(
    value: Mapping[str, Any], excluded_fields: Sequence[str]
) -> dict[str, Any]:
    """Return a deep JSON copy without explicitly non-scientific fields."""
    result = json.loads(json.dumps(dict(value), allow_nan=False))
    for field in excluded_fields:
        result.pop(field, None)
    return result


def _require_scientific_equality(
    *,
    label: str,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    excluded_fields: Sequence[str],
) -> None:
    if _scientific_view(actual, excluded_fields) != _scientific_view(
        expected, excluded_fields
    ):
        raise PreflightError(f"existing {label} scientific content changed")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strict_json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def package_versions() -> dict[str, str | None]:
    packages = (
        "numpy",
        "scipy",
        "scikit-learn",
        "scikit-optimize",
        "matplotlib",
        "pandas",
    )
    values: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = None
    return values


def runtime_record() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(CHAPTER2_ROOT.parent.resolve()))


def source_hashes() -> dict[str, str]:
    paths = (
        CHAPTER2_ROOT / "esn_config.py",
        CHAPTER2_ROOT / "esn_data.py",
        CHAPTER2_ROOT / "esn_model.py",
        CHAPTER2_ROOT / "esn_metrics.py",
        CHAPTER2_ROOT / "esn_optimisation.py",
        CHAPTER2_ROOT / "dynamics_analysis_ch2.py",
        CHAPTER2_ROOT / "config_ch2.py",
        CHAPTER2_ROOT / "esn_step8.py",
        CHAPTER2_ROOT / "run_step8.py",
    )
    return {
        project_relative(path): file_sha256(path)
        for path in paths
        if path.is_file()
    }


def tracked_non_chapter2_tree_hash() -> str:
    root = CHAPTER2_ROOT.parent
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    digest = sha256()
    for raw_name in result.stdout.split(b"\0"):
        if not raw_name:
            continue
        name = raw_name.decode("utf-8")
        root_chapter2_launcher = (
            "/" not in name
            and name.startswith("run_chapter2_")
            and name.endswith(".slurm")
        )
        if name.startswith("chapter2/") or root_chapter2_launcher:
            continue
        path = root / name
        if not path.is_file():
            raise PreflightError(f"tracked Chapter 1 file is missing: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _read_npy_header(stream: Any) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    npy_version = npy_format.read_magic(stream)
    if npy_version == (1, 0):
        return npy_format.read_array_header_1_0(stream)
    return npy_format.read_array_header_2_0(stream)


def inspect_npz_headers(record: LockedDataset) -> dict[str, dict[str, Any]]:
    if file_sha256(record.path) != record.sha256:
        raise PreflightError(f"dataset hash mismatch: {record.path}")
    headers: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(record.path) as archive:
        names = tuple(name[:-4] for name in archive.namelist())
        if names != REQUIRED_ARRAY_KEYS:
            raise PreflightError(
                f"{record.path} array order must be {REQUIRED_ARRAY_KEYS}, got {names}"
            )
        for key in REQUIRED_ARRAY_KEYS:
            with archive.open(f"{key}.npy") as stream:
                shape, fortran_order, dtype = _read_npy_header(stream)
            headers[key] = {
                "shape": list(shape),
                "dtype": str(dtype),
                "fortran_order": bool(fortran_order),
            }
            if shape != (record.state_count,) or dtype != np.dtype("float64"):
                raise PreflightError(f"unexpected header for {record.path}:{key}")
    return headers


def fixed_windows(state_count: int = FIXED_STATE_COUNT) -> tuple[dict[str, Any], ...]:
    transition_stop = state_count - 1
    windows: list[dict[str, Any]] = []
    for number, start in enumerate(STEP8_WINDOW_STARTS, start=1):
        warmup_stop = start + STEP8_WARMUP_TRANSITIONS
        forecast_stop = warmup_stop + STEP8_FORECAST_TRANSITIONS
        if forecast_stop > transition_stop:
            raise PreflightError(
                f"Step 8 window {number} [{start}, {forecast_stop}) exceeds "
                f"available transitions [0, {transition_stop})"
            )
        windows.append(
            {
                "window": number,
                "start": start,
                "warmup_range": [start, warmup_stop],
                "forecast_range": [warmup_stop, forecast_stop],
            }
        )
    scored = [range(*item["forecast_range"]) for item in windows]
    if any(set(left).intersection(right) for left, right in zip(scored, scored[1:])):
        raise PreflightError("Step 8 scored forecast intervals overlap")
    if windows[1]["forecast_range"][1] != windows[2]["warmup_range"][0] + 1:
        raise PreflightError("approved single shared transition is not preserved")
    return tuple(windows)


def validate_selected_models(selection: Mapping[str, Any]) -> None:
    if selection.get("label") != "VALIDATION-SELECTED — BENCHMARKS NOT OPENED":
        raise PreflightError("Step 7 selection label is not benchmark-untouched")
    if selection.get("step7_complete") is not True:
        raise PreflightError("Step 7 selection is not complete")
    access = selection.get("data_access", {})
    if any(
        access.get(key) is not False
        for key in (
            "held_out_opened",
            "unseen_current_opened",
            "continuous_benchmark_opened",
            "benchmark_results_present",
        )
    ):
        raise PreflightError("Step 7 selection reports benchmark access")
    for model_type in MODEL_TYPES:
        model = selection["models"][model_type]
        if int(model["selected_source_trial_index"]) != SOURCE_TRIALS[model_type]:
            raise PreflightError(f"{model_type} source trial mismatch")
        selected = dict(model["best_configuration"])
        expected_search = {
            key: value
            for key, value in FULL_CONFIGURATIONS[model_type].items()
            if key not in {"bias_scaling", "regularise_bias", "output_dimension"}
        }
        if selected != expected_search:
            raise PreflightError(f"{model_type} selected configuration mismatch")
        robust = model["best_robust_aggregate"]
        for key, expected in ROBUST_VALIDATION[model_type].items():
            if robust.get(key) != expected:
                raise PreflightError(f"{model_type} robust metric mismatch: {key}")


def validate_step7_histories() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for model_type, path in STEP7_HISTORIES.items():
        history = load_strict_json(path)
        if history.get("status") != "complete" or len(history.get("trials", [])) != 40:
            raise PreflightError(f"{model_type} history must contain 40 trials")
        if len(history.get("robust_confirmations", [])) != 5:
            raise PreflightError(f"{model_type} history must contain five confirmations")
        if history["metadata"].get("model_type") != model_type:
            raise PreflightError(f"{model_type} history identity mismatch")
        hashes[project_relative(path)] = file_sha256(path)
    return hashes


def validate_dataset_hashes(
    records: Sequence[LockedDataset] = LOCKED_DATASETS,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for record in records:
        actual = file_sha256(record.path)
        if actual != record.sha256:
            raise PreflightError(
                f"dataset hash mismatch for {record.path}: expected "
                f"{record.sha256}, got {actual}"
            )
        hashes[project_relative(record.path)] = actual
    return hashes


def validate_audit_backups() -> dict[str, str]:
    root = STEP7_ROOT / "pre_resume_fix"
    hashes: dict[str, str] = {}
    for name, expected in PRE_RESUME_HASHES.items():
        path = root / name
        actual = file_sha256(path)
        if actual != expected:
            raise PreflightError(f"provisional audit backup changed: {path}")
        hashes[project_relative(path)] = actual
    return hashes


def preflight_record() -> dict[str, Any]:
    selection = load_strict_json(STEP7_SELECTION)
    validate_selected_models(selection)
    histories = validate_step7_histories()
    datasets = validate_dataset_hashes()
    backups = validate_audit_backups()
    headers = {
        project_relative(record.path): inspect_npz_headers(record)
        for record in LOCKED_DATASETS
    }
    windows = fixed_windows()
    return {
        "schema": STEP8_SCHEMA,
        "checked_at": utc_now(),
        "selection_hash": file_sha256(STEP7_SELECTION),
        "history_hashes": histories,
        "dataset_hashes": datasets,
        "audit_backup_hashes": backups,
        "dataset_headers": headers,
        "fixed_windows": list(windows),
        "chapter1_tracked_tree_hash": tracked_non_chapter2_tree_hash(),
        "benchmark_numerical_values_inspected": False,
        "inspection_basis": [
            "raw-byte SHA-256",
            "NPZ member names",
            "NPY headers",
            "array shapes",
            "array dtypes",
            "transition counts",
            "strict Step 7 JSON metadata and metrics",
        ],
    }


def _model_lock_record(selection: Mapping[str, Any], model_type: str) -> dict[str, Any]:
    return {
        "model_type": model_type,
        "source_trial": SOURCE_TRIALS[model_type],
        "configuration": dict(FULL_CONFIGURATIONS[model_type]),
        "input_dimension": 4 if model_type == PARAMETER_AWARE else 3,
        "final_seeds": list(FINAL_SEEDS),
        "robust_validation_metrics": {
            key: selection["models"][model_type]["best_robust_aggregate"][key]
            for key in ROBUST_VALIDATION[model_type]
        },
    }


def build_selection_lock(preflight: Mapping[str, Any]) -> dict[str, Any]:
    selection = load_strict_json(STEP7_SELECTION)
    return {
        "schema": STEP8_SCHEMA,
        "status": SELECTION_LOCK_STATUS,
        "created_at": utc_now(),
        "models": {
            model_type: _model_lock_record(selection, model_type)
            for model_type in MODEL_TYPES
        },
        "final_training": {
            "currents": list(TRAIN_CURRENTS),
            "transition_range": [0, STEP8_FINAL_TRAINING_STOP],
            "washout_per_independently_reset_trajectory": STEP8_TRAINING_WASHOUT,
            "scaler_fit_transition_range": [0, STEP8_FINAL_TRAINING_STOP],
        },
        "step7_artifacts": {
            "selection": {
                "path": project_relative(STEP7_SELECTION),
                "sha256": preflight["selection_hash"],
            },
            "histories": preflight["history_hashes"],
            "audit_backups": preflight["audit_backup_hashes"],
        },
        "dataset_hashes": preflight["dataset_hashes"],
        "source_code_hashes": source_hashes(),
        "python_and_package_versions": package_versions(),
        "runtime": runtime_record(),
        "git": git_state(),
        "chapter1_tracked_tree_hash": preflight["chapter1_tracked_tree_hash"],
        "benchmark_policy": (
            "Benchmark results cannot modify model selection, hyperparameters, "
            "seeds, preprocessing, training, or evaluation rules."
        ),
    }


def build_evaluation_manifest(preflight: Mapping[str, Any]) -> dict[str, Any]:
    windows = list(fixed_windows())
    return {
        "schema": STEP8_SCHEMA,
        "status": "FROZEN BEFORE BENCHMARK ACCESS",
        "created_at": utc_now(),
        "protocol_correction": {
            "final": True,
            "reason": (
                "Header-only inspection proved that each fixed dataset has "
                "100,000 states and 99,999 transitions, leaving only 29,999 "
                "held-out transitions [70,000, 99,999)."
            ),
            "approved_window_starts": list(STEP8_WINDOW_STARTS),
            "shared_transition": 89_999,
            "shared_transition_use": (
                "scored in window 2 and used only in the unscored warm-up of "
                "window 3; scored forecast intervals do not overlap"
            ),
            "decision_evidence": preflight["inspection_basis"],
            "benchmark_numerical_values_inspected": False,
        },
        "final_training": {
            "currents": list(TRAIN_CURRENTS),
            "transition_range": [0, STEP8_FINAL_TRAINING_STOP],
            "washout_per_independently_reset_trajectory": STEP8_TRAINING_WASHOUT,
            "state_and_output_scaler_fit_range": [0, STEP8_FINAL_TRAINING_STOP],
            "current_scaler_fit_range": [0, STEP8_FINAL_TRAINING_STOP],
            "scaler_currents": list(TRAIN_CURRENTS),
        },
        "fixed_short_windows": windows,
        "known_currents": list(TRAIN_CURRENTS),
        "unseen_currents": list(UNSEEN_CURRENTS),
        "long_horizon": {
            "reset_transition": STEP8_LONG_HORIZON_START,
            "warmup_range": [70_000, 72_000],
            "forecast_range": [72_000, FIXED_STATE_COUNT - 1],
            "reset_or_rewarm_during_forecast": False,
        },
        "continuous": {
            "warmup_range": [0, STEP8_CONTINUOUS_WARMUP_TRANSITIONS],
            "forecast_range": [
                STEP8_CONTINUOUS_WARMUP_TRANSITIONS,
                CONTINUOUS_DATASET.state_count - 1,
            ],
            "switch_indices": list(CONTINUOUS_SWITCH_INDICES),
            "boundary_half_window_transitions": CONTINUOUS_BOUNDARY_HALF_WINDOW,
            "reset_count": 1,
            "rewarm_at_boundaries": False,
            "aware_transition_alignment": "[predicted state at t, I_t] -> state at t+1",
            "baseline_receives_current": False,
        },
        "metrics": {
            "definitions_source": "chapter2/esn_metrics.py (unchanged from Step 7)",
            "normalisation_scale": "final-training state population standard deviations",
            "valid_prediction_threshold": VALID_PREDICTION_THRESHOLD,
            "divergence_threshold": DIVERGENCE_THRESHOLD,
            "collapse_std_ratio_threshold": COLLAPSE_STD_RATIO_THRESHOLD,
            "nonfinite_failure_score": NONFINITE_FAILURE_SCORE,
            "undefined_representation": "None plus defined flags and contributor counts",
        },
        "events": {
            "chapter1_attribution": (
                "Chapter 1 uses scipy peak detection and a 20-step spike matching "
                "tolerance. Step 8 uses the already frozen Chapter 2 physical-state "
                "peak and adaptive log-ISI burst definitions because Chapter 1 "
                "contains no single unambiguous burst definition."
            ),
            "spike_height": SPIKE_HEIGHT,
            "spike_prominence": SPIKE_PROMINENCE,
            "minimum_spike_distance_steps": SPIKE_MIN_DISTANCE_STEPS,
            "spike_match_tolerance_steps": SPIKE_MATCH_TOLERANCE_STEPS,
            "burst_min_log_isi_gap": BURST_MIN_LOG_ISI_GAP,
            "burst_min_gap_prominence": BURST_MIN_GAP_PROMINENCE,
            "burst_min_intervals_per_timescale": BURST_MIN_INTERVALS_PER_TIMESCALE,
            "burst_min_spikes": BURST_MIN_SPIKES,
            "burst_duration": "last spike time minus first spike time in each accepted burst",
            "spike_time_error": "mean absolute timing error of greedily nearest one-to-one matches within 20 steps",
        },
        "seeds": list(FINAL_SEEDS),
        "aggregation": {
            "short_family_weight": "one equal weight per seed/current/window",
            "long_family_weight": "one equal weight per seed/current rollout",
            "continuous_family_weight": "one equal weight per seed",
            "statistics": ["mean", "population_std", "median", "worst"],
            "failed_rollouts": "retained; undefined NRMSE contributes frozen failure score",
            "families_kept_separate": [
                "known_short",
                "unseen_short",
                "known_long",
                "unseen_long",
                "continuous",
            ],
            "paired_comparison": "same seed/current/window or benchmark record",
        },
        "plot_selection": {
            "representative_seed": 42,
            "representative_window": 1,
            "cherry_picking_allowed": False,
            "formats": ["png", "pdf"],
        },
        "failure_handling": (
            "Preserve all seeds and failures; do not tune, reselect, adapt, "
            "shorten horizons, or change scaling after benchmark access."
        ),
        "expected_artifacts": [
            "selected_model.json",
            "step8_evaluation_manifest.json",
            "step8_status.json",
            "step8_raw_results.json",
            "step8_aggregate_results.json",
            "step8_known_heldout.csv",
            "step8_unseen_currents.csv",
            "step8_long_horizon.csv",
            "step8_continuous.csv",
            "step8_event_metrics.csv",
            "step8_verification.json",
            "figures/*.png",
            "figures/*.pdf",
            "final_models/model_manifest.json",
            "final_models/*.npz",
        ],
        "python_and_package_versions": package_versions(),
        "runtime": runtime_record(),
    }


def _initial_status(preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": STEP8_SCHEMA,
        "state": "PREFLIGHT",
        "updated_at": utc_now(),
        "first_benchmark_access_timestamp": None,
        "lock_hashes": {},
        "completed_record_ids": [],
        "completed_record_count": 0,
        "preflight": dict(preflight),
        "runtime": runtime_record(),
        "failure": None,
    }


def update_status(state: str, **updates: Any) -> dict[str, Any]:
    if state not in EXPECTED_STATES:
        raise ValueError(f"unknown Step 8 state: {state}")
    status = (
        load_strict_json(STATUS_PATH)
        if STATUS_PATH.exists()
        else {"schema": STEP8_SCHEMA, "first_benchmark_access_timestamp": None}
    )
    status.update(updates)
    status["state"] = state
    status["updated_at"] = utc_now()
    atomic_write_json(STATUS_PATH, status)
    return status


def lock_protocol(preflight: Mapping[str, Any]) -> dict[str, str]:
    FINAL_RESULTS.mkdir(parents=True, exist_ok=True)
    if STATUS_PATH.exists():
        status = load_strict_json(STATUS_PATH)
    else:
        status = _initial_status(preflight)
        atomic_write_json(STATUS_PATH, status)

    expected_selection = build_selection_lock(preflight)
    expected_manifest = build_evaluation_manifest(preflight)
    if SELECTION_LOCK.exists():
        current = load_strict_json(SELECTION_LOCK)
        _require_scientific_equality(
            label="selection lock",
            actual=current,
            expected=expected_selection,
            excluded_fields=SELECTION_LOCK_COMPARISON_EXCLUSIONS,
        )
    else:
        atomic_write_json(SELECTION_LOCK, expected_selection)
    if EVALUATION_MANIFEST.exists():
        current_manifest = load_strict_json(EVALUATION_MANIFEST)
        _require_scientific_equality(
            label="evaluation manifest",
            actual=current_manifest,
            expected=expected_manifest,
            excluded_fields=EVALUATION_MANIFEST_COMPARISON_EXCLUSIONS,
        )
    else:
        atomic_write_json(EVALUATION_MANIFEST, expected_manifest)

    hashes = {
        project_relative(SELECTION_LOCK): file_sha256(SELECTION_LOCK),
        project_relative(EVALUATION_MANIFEST): file_sha256(EVALUATION_MANIFEST),
    }
    existing = status.get("lock_hashes") or {}
    if existing and existing != hashes:
        raise PreflightError("lock artifact hash changed")
    update_status("SELECTION_LOCKED", lock_hashes=hashes)
    return hashes


def _read_member_prefix(
    archive: zipfile.ZipFile,
    key: str,
    *,
    expected_count: int,
    prefix_count: int,
) -> np.ndarray:
    with archive.open(f"{key}.npy") as stream:
        shape, fortran_order, dtype = _read_npy_header(stream)
        if shape != (expected_count,) or fortran_order or dtype != np.dtype("float64"):
            raise PreflightError(f"unexpected locked array header for {key}")
        raw = stream.read(prefix_count * dtype.itemsize)
    if len(raw) != prefix_count * dtype.itemsize:
        raise PreflightError(f"short prefix read for {key}")
    return np.frombuffer(raw, dtype=dtype, count=prefix_count).copy()


def load_training_prefix(record: LockedDataset) -> FixedCurrentTrajectory:
    if record.current not in TRAIN_CURRENTS:
        raise PreflightError("final training prefix loader rejects non-training current")
    if file_sha256(record.path) != record.sha256:
        raise PreflightError(f"training dataset hash mismatch: {record.path}")
    prefix_count = STEP8_FINAL_TRAINING_STOP + 1
    with zipfile.ZipFile(record.path) as archive:
        arrays = {
            key: _read_member_prefix(
                archive,
                key,
                expected_count=record.state_count,
                prefix_count=prefix_count,
            )
            for key in REQUIRED_ARRAY_KEYS
        }
    states = np.column_stack((arrays["x"], arrays["y"], arrays["z"]))
    return FixedCurrentTrajectory(
        float(record.current),
        arrays["t"],
        states,
        arrays["I"],
        record.path,
    )


def load_final_training_prefixes() -> tuple[FixedCurrentTrajectory, ...]:
    records = tuple(
        record for record in FIXED_DATASETS if record.current in TRAIN_CURRENTS
    )
    records = tuple(
        next(record for record in records if record.current == current)
        for current in TRAIN_CURRENTS
    )
    return tuple(load_training_prefix(record) for record in records)


def prepare_final_training(
    trajectories: Sequence[FixedCurrentTrajectory],
    model_type: str,
) -> tuple[StateCurrentScalers, tuple[TrainingSequence, ...]]:
    items = tuple(trajectories)
    if tuple(item.current for item in items) != TRAIN_CURRENTS:
        raise PreflightError("final training requires exactly the three training currents")
    required_states = STEP8_FINAL_TRAINING_STOP + 1
    if any(item.state_count != required_states for item in items):
        raise PreflightError("training loader must expose only [0, 70000] states")
    fitting_states = np.concatenate(
        [item.states[:STEP8_FINAL_TRAINING_STOP] for item in items], axis=0
    )
    fitting_currents = np.concatenate(
        [item.current_values[:STEP8_FINAL_TRAINING_STOP] for item in items]
    ).reshape(-1, 1)
    scalers = StateCurrentScalers(
        NumpyStandardScaler.fit(fitting_states),
        NumpyStandardScaler.fit(fitting_currents),
    )
    sequences: list[TrainingSequence] = []
    for item in items:
        states_in = scalers.state.transform(item.states[:-1])
        targets = scalers.state.transform(item.states[1:])
        if model_type == PARAMETER_AWARE:
            current = scalers.current.transform(item.current_values[:-1, None])
            inputs = np.column_stack((states_in, current))
        elif model_type == ORDINARY_BASELINE:
            inputs = states_in
        else:
            raise ValueError(f"unknown model type {model_type}")
        sequences.append(TrainingSequence(inputs, targets))
    return scalers, tuple(sequences)


def final_model_config(model_type: str, seed: int) -> ESNModelConfig:
    if seed not in FINAL_SEEDS:
        raise PreflightError(f"seed {seed} is not in the frozen final seeds")
    values = FULL_CONFIGURATIONS[model_type]
    return ESNModelConfig(
        reservoir_size=int(values["reservoir_size"]),
        reservoir_connectivity=float(values["reservoir_connectivity"]),
        input_scaling=float(values["input_scaling"]),
        spectral_radius=float(values["spectral_radius"]),
        ridge_regularisation=float(values["ridge_regularisation"]),
        leak_rate=float(values["leak_rate"]),
        bias_scaling=float(values["bias_scaling"]),
        regularise_bias=bool(values["regularise_bias"]),
        output_dimension=int(values["output_dimension"]),
        input_dimension=4 if model_type == PARAMETER_AWARE else 3,
        seed=int(seed),
    )


def model_path(model_type: str, seed: int) -> Path:
    return FINAL_MODELS / f"{model_type}_seed_{seed}.npz"


def save_final_model(
    path: Path,
    model: EchoStateNetwork,
    scalers: StateCurrentScalers,
    metadata: Mapping[str, Any],
) -> None:
    if model.output_weights is None:
        raise Step8Error("cannot serialize an unfitted final model")
    model.reset_reservoir()
    atomic_save_npz(
        path,
        schema_version=np.asarray(MODEL_BUNDLE_SCHEMA),
        config_json=np.asarray(
            json.dumps(asdict(model.config), sort_keys=True, allow_nan=False)
        ),
        metadata_json=np.asarray(
            json.dumps(dict(metadata), sort_keys=True, allow_nan=False)
        ),
        input_weights=model.input_weights,
        reservoir_weights=model.reservoir_weights,
        reservoir_bias=model.reservoir_bias,
        output_weights=model.output_weights,
        state_mean=scalers.state.mean,
        state_scale=scalers.state.scale,
        current_mean=scalers.current.mean,
        current_scale=scalers.current.scale,
    )


def load_final_model(path: Path) -> tuple[EchoStateNetwork, StateCurrentScalers, dict[str, Any]]:
    required = {
        "schema_version",
        "config_json",
        "metadata_json",
        "input_weights",
        "reservoir_weights",
        "reservoir_bias",
        "output_weights",
        "state_mean",
        "state_scale",
        "current_mean",
        "current_scale",
    }
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != required:
            raise Step8Error(f"unsafe or incomplete final model schema: {path}")
        if str(saved["schema_version"].item()) != MODEL_BUNDLE_SCHEMA:
            raise Step8Error(f"unsupported final model schema: {path}")
        config = ESNModelConfig(**json.loads(str(saved["config_json"].item())))
        metadata = json.loads(str(saved["metadata_json"].item()))
        arrays = {
            key: np.asarray(saved[key], dtype=float).copy()
            for key in required
            if key not in {"schema_version", "config_json", "metadata_json"}
        }
    model = EchoStateNetwork(config)
    expected_shapes = {
        "input_weights": (config.reservoir_size, config.input_dimension),
        "reservoir_weights": (config.reservoir_size, config.reservoir_size),
        "reservoir_bias": (config.reservoir_size,),
        "output_weights": (config.output_dimension, model.feature_dimension),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape or not np.all(np.isfinite(arrays[key])):
            raise Step8Error(f"invalid {key} in {path}")
    expected_scaler_shapes = {
        "state_mean": (STATE_DIMENSION,),
        "state_scale": (STATE_DIMENSION,),
        "current_mean": (1,),
        "current_scale": (1,),
    }
    for key, shape in expected_scaler_shapes.items():
        if arrays[key].shape != shape or not np.all(np.isfinite(arrays[key])):
            raise Step8Error(f"invalid {key} in {path}")
    if np.any(arrays["state_scale"] <= 0.0) or np.any(
        arrays["current_scale"] <= 0.0
    ):
        raise Step8Error(f"non-positive scaler scale in {path}")
    model.input_weights = arrays["input_weights"]
    model.reservoir_weights = arrays["reservoir_weights"]
    model.reservoir_bias = arrays["reservoir_bias"]
    model.output_weights = arrays["output_weights"]
    model.reset_reservoir()
    scalers = StateCurrentScalers(
        NumpyStandardScaler(arrays["state_mean"], arrays["state_scale"]),
        NumpyStandardScaler(arrays["current_mean"], arrays["current_scale"]),
    )
    return model, scalers, metadata


def _model_metadata(
    model_type: str,
    seed: int,
    preflight: Mapping[str, Any],
    lock_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": MODEL_BUNDLE_SCHEMA,
        "model_type": model_type,
        "seed": seed,
        "source_trial": SOURCE_TRIALS[model_type],
        "training_currents": list(TRAIN_CURRENTS),
        "training_transition_range": [0, STEP8_FINAL_TRAINING_STOP],
        "washout_per_independently_reset_trajectory": STEP8_TRAINING_WASHOUT,
        "scaler_fit_currents": list(TRAIN_CURRENTS),
        "scaler_fit_transition_range": [0, STEP8_FINAL_TRAINING_STOP],
        "dataset_hashes": {
            path: digest
            for path, digest in preflight["dataset_hashes"].items()
            if any(f"fixed_I_{token}" in path for token in ("1p67", "3p20", "3p50"))
        },
        "lock_hashes": dict(lock_hashes),
        "python_and_package_versions": package_versions(),
        "trained_at": utc_now(),
        "safe_serialization": "NPZ with JSON strings; allow_pickle=False",
    }


def _validate_model_manifest_entry(
    item: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    lock_hashes: Mapping[str, str],
) -> tuple[str, int]:
    try:
        model_type = str(item["model_type"])
        seed = int(item["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise PreflightError("invalid model identity in final model manifest") from error
    if model_type not in MODEL_TYPES or seed not in FINAL_SEEDS:
        raise PreflightError(f"unexpected final model identity: {(model_type, seed)}")

    configuration = asdict(final_model_config(model_type, seed))
    expected_path = model_path(model_type, seed)
    expected_entry = {
        "model_type": model_type,
        "seed": seed,
        "path": project_relative(expected_path),
        "configuration": configuration,
        "round_trip_inference_exact": True,
        "training_transition_range": [0, STEP8_FINAL_TRAINING_STOP],
        "washout_per_trajectory": STEP8_TRAINING_WASHOUT,
    }
    for field, expected in expected_entry.items():
        if item.get(field) != expected:
            raise PreflightError(
                f"final model manifest {field} mismatch for {(model_type, seed)}"
            )
    digest = item.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise PreflightError(f"invalid model hash for {(model_type, seed)}")
    if not expected_path.is_file() or file_sha256(expected_path) != digest:
        raise PreflightError(f"final model hash mismatch: {expected_path}")

    try:
        model, scalers, metadata = load_final_model(expected_path)
    except Exception as error:
        raise PreflightError(f"invalid final model bundle: {expected_path}") from error
    if asdict(model.config) != configuration:
        raise PreflightError(f"model configuration mismatch: {expected_path}")
    expected_metadata = _model_metadata(model_type, seed, preflight, lock_hashes)
    _require_scientific_equality(
        label=f"model metadata for {(model_type, seed)}",
        actual=metadata,
        expected=expected_metadata,
        excluded_fields=MODEL_METADATA_COMPARISON_EXCLUSIONS,
    )
    if scalers.state.mean.shape != (STATE_DIMENSION,) or scalers.current.mean.shape != (1,):
        raise PreflightError(f"model scaler metadata mismatch: {expected_path}")
    if not math.isclose(
        model.spectral_radius,
        configuration["spectral_radius"],
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise PreflightError(f"model spectral radius mismatch: {expected_path}")
    return model_type, seed


def validate_model_manifest(
    manifest: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    lock_hashes: Mapping[str, str],
    require_complete: bool,
) -> None:
    if manifest.get("schema") != MODEL_MANIFEST_SCHEMA:
        raise PreflightError("existing final model manifest is incompatible")
    if manifest.get("lock_hashes") != dict(lock_hashes):
        raise PreflightError("final model manifest lock hashes changed")
    status = manifest.get("status")
    if status not in {"in_progress", "complete"}:
        raise PreflightError("invalid final model manifest status")
    if require_complete and status != "complete":
        raise PreflightError("final model manifest is incomplete")
    models = manifest.get("models")
    if not isinstance(models, list):
        raise PreflightError("final model manifest models must be a list")
    identities = [
        _validate_model_manifest_entry(
            item, preflight=preflight, lock_hashes=lock_hashes
        )
        for item in models
    ]
    if len(identities) != len(set(identities)):
        raise PreflightError("duplicate final model identities")
    expected = {
        (model_type, seed) for model_type in MODEL_TYPES for seed in FINAL_SEEDS
    }
    if not set(identities).issubset(expected):
        raise PreflightError("unexpected final model identity")
    if status == "complete":
        if set(identities) != expected or manifest.get("model_count") != len(expected):
            raise PreflightError("complete model manifest is not exactly ten models")


def train_final_models(
    preflight: Mapping[str, Any],
    lock_hashes: Mapping[str, str],
) -> dict[str, Any]:
    FINAL_MODELS.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any]
    if MODEL_MANIFEST_PATH.exists():
        manifest = load_strict_json(MODEL_MANIFEST_PATH)
        validate_model_manifest(
            manifest,
            preflight=preflight,
            lock_hashes=lock_hashes,
            require_complete=False,
        )
        if manifest.get("status") == "complete":
            validate_model_manifest(
                manifest,
                preflight=preflight,
                lock_hashes=lock_hashes,
                require_complete=True,
            )
            return manifest
    else:
        manifest = {
            "schema": MODEL_MANIFEST_SCHEMA,
            "status": "in_progress",
            "created_at": utc_now(),
            "lock_hashes": dict(lock_hashes),
            "models": [],
        }
        atomic_write_json(MODEL_MANIFEST_PATH, manifest)

    prefixes = load_final_training_prefixes()
    prepared = {
        model_type: prepare_final_training(prefixes, model_type)
        for model_type in MODEL_TYPES
    }
    existing = {
        (item["model_type"], int(item["seed"])): item
        for item in manifest.get("models", [])
    }
    for model_type in MODEL_TYPES:
        scalers, sequences = prepared[model_type]
        for seed in FINAL_SEEDS:
            key = (model_type, seed)
            destination = model_path(model_type, seed)
            if key in existing:
                if (
                    not destination.is_file()
                    or file_sha256(destination) != existing[key]["sha256"]
                ):
                    raise PreflightError(f"final model hash mismatch: {destination}")
                continue
            model = EchoStateNetwork(final_model_config(model_type, seed))
            model.fit(sequences, washout=STEP8_TRAINING_WASHOUT)
            metadata = _model_metadata(model_type, seed, preflight, lock_hashes)
            in_memory_prediction = model.predict_one_step(sequences[0].inputs[0])
            model.reset_reservoir()
            save_final_model(destination, model, scalers, metadata)
            loaded, loaded_scalers, loaded_metadata = load_final_model(destination)
            loaded_prediction = loaded.predict_one_step(sequences[0].inputs[0])
            loaded.reset_reservoir()
            if not np.array_equal(in_memory_prediction, loaded_prediction):
                raise Step8Error(f"round-trip inference mismatch: {destination}")
            if (
                loaded_metadata != metadata
                or not np.array_equal(loaded_scalers.state.mean, scalers.state.mean)
                or not np.array_equal(loaded_scalers.state.scale, scalers.state.scale)
                or not np.array_equal(loaded_scalers.current.mean, scalers.current.mean)
                or not np.array_equal(loaded_scalers.current.scale, scalers.current.scale)
            ):
                raise Step8Error(f"round-trip metadata/scaler mismatch: {destination}")
            record = {
                "model_type": model_type,
                "seed": seed,
                "path": project_relative(destination),
                "sha256": file_sha256(destination),
                "configuration": asdict(model.config),
                "round_trip_inference_exact": True,
                "training_transition_range": [0, STEP8_FINAL_TRAINING_STOP],
                "washout_per_trajectory": STEP8_TRAINING_WASHOUT,
            }
            manifest["models"].append(record)
            existing[key] = record
            atomic_write_json(MODEL_MANIFEST_PATH, manifest)
            print(f"trained {model_type} seed={seed}", flush=True)

    expected = {(model_type, seed) for model_type in MODEL_TYPES for seed in FINAL_SEEDS}
    if set(existing) != expected:
        raise Step8Error("final model manifest does not contain exactly ten models")
    manifest["models"] = sorted(
        manifest["models"],
        key=lambda item: (MODEL_TYPES.index(item["model_type"]), FINAL_SEEDS.index(int(item["seed"]))),
    )
    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now()
    manifest["model_count"] = 10
    atomic_write_json(MODEL_MANIFEST_PATH, manifest)
    update_status(
        "FINAL_MODELS_TRAINED",
        model_manifest_hash=file_sha256(MODEL_MANIFEST_PATH),
        final_model_count=10,
    )
    return manifest


def validate_benchmark_gate(
    lock_hashes: Mapping[str, str],
    *,
    preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    for path in (SELECTION_LOCK, EVALUATION_MANIFEST, MODEL_MANIFEST_PATH):
        if not path.is_file():
            raise BenchmarkAccessError(f"benchmark gate missing {path}")
    if preflight is None:
        raise BenchmarkAccessError("benchmark gate requires preflight metadata")
    actual_locks = {
        project_relative(SELECTION_LOCK): file_sha256(SELECTION_LOCK),
        project_relative(EVALUATION_MANIFEST): file_sha256(EVALUATION_MANIFEST),
    }
    if actual_locks != dict(lock_hashes):
        raise BenchmarkAccessError("selection or evaluation lock hash changed")
    manifest = load_strict_json(MODEL_MANIFEST_PATH)
    try:
        validate_model_manifest(
            manifest,
            preflight=preflight,
            lock_hashes=lock_hashes,
            require_complete=True,
        )
    except PreflightError as error:
        raise BenchmarkAccessError(str(error)) from error
    return manifest


def mark_benchmark_access_started(lock_hashes: Mapping[str, str]) -> None:
    status = load_strict_json(STATUS_PATH)
    timestamp = status.get("first_benchmark_access_timestamp") or utc_now()
    update_status(
        "BENCHMARK_ACCESS_STARTED",
        first_benchmark_access_timestamp=timestamp,
        lock_hashes=dict(lock_hashes),
    )


def _scaled_warmup(
    states: np.ndarray,
    currents: np.ndarray,
    transition_start: int,
    transition_stop: int,
    model_type: str,
    scalers: StateCurrentScalers,
) -> np.ndarray:
    state_rows = scalers.state.transform(states[transition_start:transition_stop])
    if model_type == PARAMETER_AWARE:
        current_rows = scalers.current.transform(
            currents[transition_start:transition_stop, None]
        )
        return np.column_stack((state_rows, current_rows))
    return state_rows


def recursive_forecast(
    model: EchoStateNetwork,
    scalers: StateCurrentScalers,
    model_type: str,
    states: np.ndarray,
    currents: np.ndarray,
    *,
    warmup_range: tuple[int, int],
    forecast_range: tuple[int, int],
) -> tuple[np.ndarray, int | None, str | None]:
    warm_start, warm_stop = warmup_range
    forecast_start, forecast_stop = forecast_range
    if warm_stop != forecast_start:
        raise Step8Error("warm-up and forecast ranges must be adjacent")
    warmup = _scaled_warmup(
        states, currents, warm_start, warm_stop, model_type, scalers
    )
    model.teacher_forced_warmup(warmup, reset=True)
    state = scalers.state.transform(states[forecast_start : forecast_start + 1])[0]
    horizon = forecast_stop - forecast_start
    predictions = np.full((horizon, STATE_DIMENSION), np.nan)
    failure_step: int | None = None
    failure_reason: str | None = None
    scaled_currents = (
        scalers.current.transform(currents[forecast_start:forecast_stop, None])[:, 0]
        if model_type == PARAMETER_AWARE
        else None
    )
    with np.errstate(over="ignore", invalid="ignore"):
        for index in range(horizon):
            model_input = (
                np.concatenate((state, [scaled_currents[index]]))
                if scaled_currents is not None
                else state
            )
            try:
                prediction = model.predict_one_step(model_input)
            except (ValueError, FloatingPointError):
                failure_step = index
                failure_reason = "non_finite_recursive_input"
                break
            predictions[index] = prediction
            if not np.all(np.isfinite(prediction)):
                failure_step = index
                failure_reason = "non_finite_prediction"
                break
            state = prediction
    model.reset_reservoir()
    physical = predictions * scalers.state.scale + scalers.state.mean
    return physical, failure_step, failure_reason


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _event_summary(states: np.ndarray, dt: float) -> dict[str, Any]:
    if states.ndim != 2 or states.shape[1] != 3 or not np.all(np.isfinite(states)):
        return {
            "defined": False,
            "reason": "non_finite_or_invalid_trajectory",
            "spike_count": None,
            "spike_rate": None,
            "mean_interspike_interval": None,
            "burst_count": None,
            "burst_rate": None,
            "mean_burst_duration": None,
            "mean_interburst_interval": None,
            "mean_spikes_per_burst": None,
        }
    from chapter2.hr_data_ch2 import HRTrajectory

    time = np.arange(len(states), dtype=float) * dt
    trajectory = HRTrajectory(
        time,
        states[:, 0],
        states[:, 1],
        states[:, 2],
        np.zeros(len(states)),
    )
    analysis = analyze_spikes_and_bursts(trajectory, dt=dt)
    peaks = analysis.spike_indices
    duration = max(len(states) * dt, dt)
    burst_durations: list[float] = []
    if analysis.burst_gap_threshold is not None and len(peaks) >= 2:
        isi = np.diff(peaks) * dt
        boundaries = np.flatnonzero(isi > analysis.burst_gap_threshold) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(peaks)]))
        for start, stop in zip(starts, ends):
            if stop - start >= BURST_MIN_SPIKES:
                burst_durations.append(float((peaks[stop - 1] - peaks[start]) * dt))
    burst_count = analysis.burst_count
    return {
        "defined": True,
        "reason": None,
        "spike_indices": peaks.tolist(),
        "spike_count": int(len(peaks)),
        "spike_rate": float(len(peaks) / duration),
        "mean_interspike_interval": _finite_or_none(analysis.mean_isi),
        "burst_structure": analysis.burst_structure,
        "burst_count": burst_count,
        "burst_rate": (
            float(burst_count / duration) if burst_count is not None else None
        ),
        "mean_burst_duration": (
            float(np.mean(burst_durations)) if burst_durations else None
        ),
        "mean_interburst_interval": _finite_or_none(
            analysis.mean_interburst_interval
        ),
        "mean_spikes_per_burst": _finite_or_none(
            analysis.mean_spikes_per_burst
        ),
        "burst_gap_threshold": analysis.burst_gap_threshold,
        "notes": analysis.notes,
    }


def _match_spike_time_error(
    target_indices: Sequence[int], prediction_indices: Sequence[int], dt: float
) -> tuple[float | None, int]:
    # Adapted from Chapter 1 plotting._match_spikes: greedy nearest unmatched
    # target within the frozen 20-step tolerance.
    unmatched = set(range(len(target_indices)))
    distances: list[int] = []
    for predicted in prediction_indices:
        candidates = [
            (abs(int(predicted) - int(target_indices[index])), index)
            for index in unmatched
            if abs(int(predicted) - int(target_indices[index]))
            <= SPIKE_MATCH_TOLERANCE_STEPS
        ]
        if candidates:
            distance, matched = min(candidates)
            unmatched.remove(matched)
            distances.append(distance)
    return (
        float(np.mean(distances) * dt) if distances else None,
        len(distances),
    )


def event_metrics(predictions: np.ndarray, targets: np.ndarray, dt: float = DT) -> dict[str, Any]:
    predicted = _event_summary(predictions, dt)
    expected = _event_summary(targets, dt)
    if not predicted["defined"] or not expected["defined"]:
        return {
            "defined": False,
            "prediction": predicted,
            "target": expected,
            "errors": None,
            "truncated": False,
        }
    spike_time_error, match_count = _match_spike_time_error(
        expected["spike_indices"], predicted["spike_indices"], dt
    )
    fields = (
        "spike_count",
        "spike_rate",
        "mean_interspike_interval",
        "burst_count",
        "burst_rate",
        "mean_burst_duration",
        "mean_interburst_interval",
        "mean_spikes_per_burst",
    )
    errors: dict[str, Any] = {
        "spike_time_mean_absolute_error": spike_time_error,
        "spike_time_match_count": match_count,
        "spike_time_error_defined": spike_time_error is not None,
    }
    for field in fields:
        left = predicted[field]
        right = expected[field]
        defined = left is not None and right is not None
        errors[f"{field}_absolute_error"] = (
            abs(float(left) - float(right)) if defined else None
        )
        errors[f"{field}_error_defined"] = defined
    return {
        "defined": True,
        "prediction": predicted,
        "target": expected,
        "errors": errors,
        "truncated": False,
    }


def invalidate_divergent_event_metrics(
    event_result: dict[str, Any] | None,
    metrics: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Mark event metrics undefined when the same evaluated interval diverged."""
    if event_result is None or not bool(metrics["diverged"]):
        return event_result
    return {
        "defined": False,
        "prediction": event_result["prediction"],
        "target": event_result["target"],
        "errors": None,
        "truncated": False,
        "reason": "event metrics invalidated by interval divergence",
    }


def _record_id(
    family: str,
    model_type: str,
    seed: int,
    current: float | None = None,
    window: int | None = None,
) -> str:
    parts = [family, model_type, f"seed_{seed}"]
    if current is not None:
        parts.append(f"I_{current:.2f}".replace(".", "p"))
    if window is not None:
        parts.append(f"window_{window}")
    return "__".join(parts)


def _expected_resumed_records() -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}

    def add(
        family: str,
        model_type: str,
        seed: int,
        *,
        current: float | None,
        window: int | None,
        warmup_range: Sequence[int],
        forecast_range: Sequence[int],
    ) -> None:
        identifier = _record_id(family, model_type, seed, current, window)
        expected[identifier] = {
            "record_id": identifier,
            "family": family,
            "model_type": model_type,
            "seed": seed,
            "current": current,
            "window": window,
            "warmup_range": list(warmup_range),
            "forecast_range": list(forecast_range),
        }

    for family, currents in (
        ("known_short", TRAIN_CURRENTS),
        ("unseen_short", UNSEEN_CURRENTS),
    ):
        for model_type in MODEL_TYPES:
            for seed in FINAL_SEEDS:
                for current in currents:
                    for window in fixed_windows():
                        add(
                            family,
                            model_type,
                            seed,
                            current=current,
                            window=int(window["window"]),
                            warmup_range=window["warmup_range"],
                            forecast_range=window["forecast_range"],
                        )
    for family, currents in (
        ("known_long", TRAIN_CURRENTS),
        ("unseen_long", UNSEEN_CURRENTS),
    ):
        for model_type in MODEL_TYPES:
            for seed in FINAL_SEEDS:
                for current in currents:
                    add(
                        family,
                        model_type,
                        seed,
                        current=current,
                        window=None,
                        warmup_range=(70_000, 72_000),
                        forecast_range=(72_000, FIXED_STATE_COUNT - 1),
                    )
    for model_type in MODEL_TYPES:
        for seed in FINAL_SEEDS:
            add(
                "continuous",
                model_type,
                seed,
                current=None,
                window=None,
                warmup_range=(0, STEP8_CONTINUOUS_WARMUP_TRANSITIONS),
                forecast_range=(
                    STEP8_CONTINUOUS_WARMUP_TRANSITIONS,
                    CONTINUOUS_DATASET.state_count - 1,
                ),
            )
    return expected


def _validate_resumed_npz(record: Mapping[str, Any], horizon: int) -> None:
    identifier = str(record["record_id"])
    expected_path = RAW_ARRAYS / f"{identifier}.npz"
    expected_relative = project_relative(expected_path)
    if record.get("raw_arrays_path") != expected_relative:
        raise BenchmarkAccessError(f"raw-array path mismatch for {identifier}")
    if not expected_path.is_file():
        raise BenchmarkAccessError(f"missing raw NPZ for {identifier}")
    digest = record.get("raw_arrays_sha256")
    if not isinstance(digest, str) or file_sha256(expected_path) != digest:
        raise BenchmarkAccessError(f"raw-array hash mismatch for {identifier}")

    expected_shapes = {
        "predictions": (horizon, STATE_DIMENSION),
        "targets": (horizon, STATE_DIMENSION),
        "pointwise_normalised_error": (horizon,),
        "time": (horizon,),
        "current": (horizon,),
    }
    try:
        with np.load(expected_path, allow_pickle=False) as saved:
            if set(saved.files) != RAW_ARRAY_KEYS:
                raise BenchmarkAccessError(
                    f"raw NPZ keys mismatch for {identifier}"
                )
            arrays = {key: np.asarray(saved[key]) for key in RAW_ARRAY_KEYS}
    except BenchmarkAccessError:
        raise
    except Exception as error:
        raise BenchmarkAccessError(f"corrupted raw NPZ for {identifier}") from error
    for key, expected_shape in expected_shapes.items():
        array = arrays[key]
        if array.shape != expected_shape or not np.issubdtype(
            array.dtype, np.number
        ):
            raise BenchmarkAccessError(
                f"raw NPZ {key} shape or dtype mismatch for {identifier}"
            )
    for key in ("targets", "time", "current"):
        if not np.all(np.isfinite(arrays[key])):
            raise BenchmarkAccessError(
                f"raw NPZ {key} contains non-finite values for {identifier}"
            )

    numerical_failure = record.get("numerical_failure")
    failure_step = record.get("failure_step")
    failure_reason = record.get("failure_reason")
    if numerical_failure is False:
        if failure_step is not None or failure_reason is not None:
            raise BenchmarkAccessError(
                f"failure metadata mismatch for {identifier}"
            )
        if any(not np.all(np.isfinite(arrays[key])) for key in RAW_ARRAY_KEYS):
            raise BenchmarkAccessError(
                f"raw NPZ contains non-finite values for {identifier}"
            )
    elif numerical_failure is True:
        if (
            isinstance(failure_step, bool)
            or not isinstance(failure_step, int)
            or not 0 <= failure_step < horizon
            or not isinstance(failure_reason, str)
            or not failure_reason
        ):
            raise BenchmarkAccessError(
                f"invalid numerical-failure metadata for {identifier}"
            )
        if not np.all(np.isfinite(arrays["predictions"][:failure_step])):
            raise BenchmarkAccessError(
                f"non-finite prediction before recorded failure for {identifier}"
            )
    else:
        raise BenchmarkAccessError(
            f"invalid numerical_failure flag for {identifier}"
        )


def validate_resumed_records(
    raw: Mapping[str, Any],
    *,
    status: Mapping[str, Any],
    manifest: Mapping[str, Any],
    lock_hashes: Mapping[str, str],
) -> None:
    if raw.get("schema") != STEP8_SCHEMA:
        raise BenchmarkAccessError("raw checkpoint schema mismatch")
    if raw.get("lock_hashes") != dict(lock_hashes):
        raise BenchmarkAccessError("raw checkpoint lock hashes conflict")
    records = raw.get("records")
    if not isinstance(records, list):
        raise BenchmarkAccessError("raw checkpoint records must be a list")
    try:
        identifiers = [str(item["record_id"]) for item in records]
    except (KeyError, TypeError) as error:
        raise BenchmarkAccessError("raw checkpoint record has no ID") from error
    if len(identifiers) != len(set(identifiers)):
        raise BenchmarkAccessError("duplicate resumed record IDs")

    expected = _expected_resumed_records()
    if not set(identifiers).issubset(expected):
        raise BenchmarkAccessError("unexpected resumed record ID")
    completed = status.get("completed_record_ids")
    if (
        not isinstance(completed, list)
        or completed != identifiers
        or len(completed) != len(set(completed))
        or status.get("completed_record_count") != len(identifiers)
    ):
        raise BenchmarkAccessError("incomplete or inconsistent resumed record IDs")
    if status.get("lock_hashes") != dict(lock_hashes):
        raise BenchmarkAccessError("status lock hashes conflict")

    model_identities = {
        (str(item["model_type"]), int(item["seed"]))
        for item in manifest.get("models", [])
    }
    for item in records:
        identifier = str(item["record_id"])
        metadata = expected[identifier]
        for field, expected_value in metadata.items():
            if item.get(field) != expected_value:
                raise BenchmarkAccessError(
                    f"resumed record metadata mismatch for {identifier}: {field}"
                )
        if (metadata["model_type"], metadata["seed"]) not in model_identities:
            raise BenchmarkAccessError(
                f"resumed record has no matching model for {identifier}"
            )
        horizon = metadata["forecast_range"][1] - metadata["forecast_range"][0]
        metrics = item.get("metrics")
        if not isinstance(metrics, Mapping) or metrics.get("sample_count") != horizon:
            raise BenchmarkAccessError(
                f"resumed record metric length mismatch for {identifier}"
            )
        _validate_resumed_npz(item, horizon)


def validate_resume_state(preflight: Mapping[str, Any]) -> None:
    """Validate every existing resume artifact before any repository write."""
    status = load_strict_json(STATUS_PATH) if STATUS_PATH.exists() else None
    if status is not None and status.get("state") == "STEP8_COMPLETE":
        raise PreflightError("Step 8 is already complete; refusing to rerun it")

    resume_paths = (
        SELECTION_LOCK,
        EVALUATION_MANIFEST,
        MODEL_MANIFEST_PATH,
        RAW_RESULTS_PATH,
    )
    if not any(path.exists() for path in resume_paths):
        return
    if not SELECTION_LOCK.is_file() or not EVALUATION_MANIFEST.is_file():
        raise PreflightError("resume state has incomplete lock artifacts")

    selection = load_strict_json(SELECTION_LOCK)
    evaluation = load_strict_json(EVALUATION_MANIFEST)
    _require_scientific_equality(
        label="selection lock",
        actual=selection,
        expected=build_selection_lock(preflight),
        excluded_fields=SELECTION_LOCK_COMPARISON_EXCLUSIONS,
    )
    _require_scientific_equality(
        label="evaluation manifest",
        actual=evaluation,
        expected=build_evaluation_manifest(preflight),
        excluded_fields=EVALUATION_MANIFEST_COMPARISON_EXCLUSIONS,
    )
    lock_hashes = {
        project_relative(SELECTION_LOCK): file_sha256(SELECTION_LOCK),
        project_relative(EVALUATION_MANIFEST): file_sha256(EVALUATION_MANIFEST),
    }
    if status is not None and status.get("lock_hashes") not in ({}, lock_hashes):
        raise PreflightError("status lock hashes conflict")

    if MODEL_MANIFEST_PATH.exists():
        manifest = load_strict_json(MODEL_MANIFEST_PATH)
        validate_model_manifest(
            manifest,
            preflight=preflight,
            lock_hashes=lock_hashes,
            require_complete=RAW_RESULTS_PATH.exists(),
        )
    elif RAW_RESULTS_PATH.exists():
        raise PreflightError("raw checkpoint exists without a model manifest")
    else:
        return

    if RAW_RESULTS_PATH.exists():
        if status is None:
            raise PreflightError("raw checkpoint exists without status")
        validate_resumed_records(
            load_strict_json(RAW_RESULTS_PATH),
            status=status,
            manifest=manifest,
            lock_hashes=lock_hashes,
        )


def _evaluate_arrays(
    *,
    record_id: str,
    family: str,
    model_type: str,
    seed: int,
    scalers: StateCurrentScalers,
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    currents: np.ndarray,
    failure_step: int | None,
    failure_reason: str | None,
    current: float | None,
    window: int | None,
    warmup_range: tuple[int, int],
    forecast_range: tuple[int, int],
    include_events: bool,
) -> dict[str, Any]:
    metrics = evaluate_rollout(
        predictions,
        targets,
        normalisation_scale=scalers.state.scale,
        dt=DT,
        valid_prediction_threshold=VALID_PREDICTION_THRESHOLD,
        divergence_threshold=DIVERGENCE_THRESHOLD,
        collapse_std_ratio_threshold=COLLAPSE_STD_RATIO_THRESHOLD,
    ).to_dict()
    pointwise = pointwise_normalised_error(
        predictions, targets, normalisation_scale=scalers.state.scale
    )
    array_path = RAW_ARRAYS / f"{record_id}.npz"
    atomic_save_npz(
        array_path,
        predictions=predictions,
        targets=targets,
        pointwise_normalised_error=pointwise,
        time=times,
        current=currents,
    )
    event_result = invalidate_divergent_event_metrics(
        event_metrics(predictions, targets, DT) if include_events else None,
        metrics,
    )
    return {
        "record_id": record_id,
        "family": family,
        "model_type": model_type,
        "seed": seed,
        "current": current,
        "window": window,
        "warmup_range": list(warmup_range),
        "forecast_range": list(forecast_range),
        "metrics": metrics,
        "aggregate_nrmse_value": (
            float(metrics["nrmse_state"])
            if metrics["nrmse_state"] is not None
            else NONFINITE_FAILURE_SCORE
        ),
        "numerical_failure": failure_step is not None,
        "failure_step": failure_step,
        "failure_reason": failure_reason,
        "event_metrics": event_result,
        "raw_arrays_path": project_relative(array_path),
        "raw_arrays_sha256": file_sha256(array_path),
    }


def _load_models(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, int], tuple[EchoStateNetwork, StateCurrentScalers]]:
    models = {}
    for item in manifest["models"]:
        key = (item["model_type"], int(item["seed"]))
        path = CHAPTER2_ROOT.parent / item["path"]
        model, scalers, metadata = load_final_model(path)
        if metadata["model_type"] != key[0] or int(metadata["seed"]) != key[1]:
            raise Step8Error(f"model metadata mismatch: {path}")
        models[key] = (model, scalers)
    return models


def _new_raw_results(lock_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema": STEP8_SCHEMA,
        "created_at": utc_now(),
        "lock_hashes": dict(lock_hashes),
        "records": [],
    }


def _checkpoint_raw(raw: Mapping[str, Any]) -> None:
    atomic_write_json(RAW_RESULTS_PATH, raw)
    ids = [item["record_id"] for item in raw["records"]]
    status = load_strict_json(STATUS_PATH)
    update_status(
        status["state"],
        completed_record_ids=ids,
        completed_record_count=len(ids),
    )


def _append_record(raw: dict[str, Any], record: dict[str, Any]) -> None:
    ids = {item["record_id"] for item in raw["records"]}
    if record["record_id"] in ids:
        raise Step8Error(f"duplicate raw record {record['record_id']}")
    raw["records"].append(record)
    _checkpoint_raw(raw)
    print(f"completed {record['record_id']}", flush=True)


def _fixed_rollout_record(
    trajectory: FixedCurrentTrajectory,
    model: EchoStateNetwork,
    scalers: StateCurrentScalers,
    model_type: str,
    seed: int,
    *,
    family: str,
    window: int | None,
    warmup_range: tuple[int, int],
    forecast_range: tuple[int, int],
    include_events: bool,
) -> dict[str, Any]:
    predictions, failure_step, failure_reason = recursive_forecast(
        model,
        scalers,
        model_type,
        trajectory.states,
        trajectory.current_values,
        warmup_range=warmup_range,
        forecast_range=forecast_range,
    )
    start, stop = forecast_range
    targets = trajectory.states[start + 1 : stop + 1]
    times = trajectory.time[start + 1 : stop + 1]
    currents = trajectory.current_values[start:stop]
    identifier = _record_id(family, model_type, seed, trajectory.current, window)
    return _evaluate_arrays(
        record_id=identifier,
        family=family,
        model_type=model_type,
        seed=seed,
        scalers=scalers,
        predictions=predictions,
        targets=targets,
        times=times,
        currents=currents,
        failure_step=failure_step,
        failure_reason=failure_reason,
        current=trajectory.current,
        window=window,
        warmup_range=warmup_range,
        forecast_range=forecast_range,
        include_events=include_events,
    )


def _submetrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    scale: np.ndarray,
) -> dict[str, Any]:
    return evaluate_rollout(
        predictions,
        targets,
        normalisation_scale=scale,
        dt=DT,
        valid_prediction_threshold=VALID_PREDICTION_THRESHOLD,
        divergence_threshold=DIVERGENCE_THRESHOLD,
        collapse_std_ratio_threshold=COLLAPSE_STD_RATIO_THRESHOLD,
    ).to_dict()


def _continuous_rollout_record(
    trajectory: ContinuousCurrentTrajectory,
    model: EchoStateNetwork,
    scalers: StateCurrentScalers,
    model_type: str,
    seed: int,
) -> dict[str, Any]:
    warmup_range = (0, STEP8_CONTINUOUS_WARMUP_TRANSITIONS)
    forecast_range = (
        STEP8_CONTINUOUS_WARMUP_TRANSITIONS,
        trajectory.state_count - 1,
    )
    predictions, failure_step, failure_reason = recursive_forecast(
        model,
        scalers,
        model_type,
        trajectory.states,
        trajectory.current_values,
        warmup_range=warmup_range,
        forecast_range=forecast_range,
    )
    start, stop = forecast_range
    targets = trajectory.states[start + 1 : stop + 1]
    times = trajectory.time[start + 1 : stop + 1]
    currents = trajectory.current_values[start:stop]
    identifier = _record_id("continuous", model_type, seed)
    record = _evaluate_arrays(
        record_id=identifier,
        family="continuous",
        model_type=model_type,
        seed=seed,
        scalers=scalers,
        predictions=predictions,
        targets=targets,
        times=times,
        currents=currents,
        failure_step=failure_step,
        failure_reason=failure_reason,
        current=None,
        window=None,
        warmup_range=warmup_range,
        forecast_range=forecast_range,
        include_events=True,
    )
    segments = [start, *CONTINUOUS_SWITCH_INDICES, stop]
    per_interval = []
    for left, right in zip(segments, segments[1:]):
        array_left = left - start
        array_right = right - start
        interval_predictions = predictions[array_left:array_right]
        interval_targets = targets[array_left:array_right]
        interval_metrics = _submetrics(
            interval_predictions, interval_targets, scalers.state.scale
        )
        interval_events = invalidate_divergent_event_metrics(
            event_metrics(interval_predictions, interval_targets, DT),
            interval_metrics,
        )
        per_interval.append(
            {
                "current": float(trajectory.current_values[left]),
                "transition_range": [left, right],
                "metrics": interval_metrics,
                "event_metrics": interval_events,
            }
        )
    transitions = []
    for boundary in CONTINUOUS_SWITCH_INDICES:
        left = boundary - CONTINUOUS_BOUNDARY_HALF_WINDOW
        right = boundary + CONTINUOUS_BOUNDARY_HALF_WINDOW
        array_left = left - start
        array_right = right - start
        transitions.append(
            {
                "boundary_transition": boundary,
                "transition_range": [left, right],
                "current_before": float(trajectory.current_values[boundary - 1]),
                "current_after": float(trajectory.current_values[boundary]),
                "metrics": _submetrics(
                    predictions[array_left:array_right],
                    targets[array_left:array_right],
                    scalers.state.scale,
                ),
            }
        )
    record["current_change_indices"] = list(CONTINUOUS_SWITCH_INDICES)
    record["per_current_interval"] = per_interval
    record["transition_boundary_metrics"] = transitions
    record["reset_count"] = 1
    record["teacher_force_count"] = STEP8_CONTINUOUS_WARMUP_TRANSITIONS
    record["rewarmed_at_boundary"] = False
    return record


def execute_benchmarks(
    preflight: Mapping[str, Any],
    lock_hashes: Mapping[str, str],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    validated_manifest = validate_benchmark_gate(
        lock_hashes, preflight=preflight
    )
    if dict(manifest) != validated_manifest:
        raise BenchmarkAccessError("in-memory model manifest differs from disk")
    raw = (
        load_strict_json(RAW_RESULTS_PATH)
        if RAW_RESULTS_PATH.exists()
        else _new_raw_results(lock_hashes)
    )
    if RAW_RESULTS_PATH.exists():
        validate_resumed_records(
            raw,
            status=load_strict_json(STATUS_PATH),
            manifest=validated_manifest,
            lock_hashes=lock_hashes,
        )
    elif raw.get("lock_hashes") != dict(lock_hashes):
        raise BenchmarkAccessError("raw checkpoint lock hashes conflict")

    # Only after every resume artifact is authenticated may model objects be
    # prepared for prediction or a held-out/evaluation dataset be loaded.
    mark_benchmark_access_started(lock_hashes)
    models = _load_models(validated_manifest)
    completed = {item["record_id"] for item in raw["records"]}

    known = {current: load_fixed_trajectory(current) for current in TRAIN_CURRENTS}
    for model_type in MODEL_TYPES:
        for seed in FINAL_SEEDS:
            model, scalers = models[(model_type, seed)]
            for current in TRAIN_CURRENTS:
                for item in fixed_windows():
                    identifier = _record_id(
                        "known_short", model_type, seed, current, item["window"]
                    )
                    if identifier in completed:
                        continue
                    record = _fixed_rollout_record(
                        known[current], model, scalers, model_type, seed,
                        family="known_short",
                        window=item["window"],
                        warmup_range=tuple(item["warmup_range"]),
                        forecast_range=tuple(item["forecast_range"]),
                        include_events=False,
                    )
                    _append_record(raw, record)
                    completed.add(identifier)
    update_status("KNOWN_HELDOUT_COMPLETE")

    unseen = {current: load_fixed_trajectory(current) for current in UNSEEN_CURRENTS}
    for model_type in MODEL_TYPES:
        for seed in FINAL_SEEDS:
            model, scalers = models[(model_type, seed)]
            for current in UNSEEN_CURRENTS:
                for item in fixed_windows():
                    identifier = _record_id(
                        "unseen_short", model_type, seed, current, item["window"]
                    )
                    if identifier in completed:
                        continue
                    record = _fixed_rollout_record(
                        unseen[current], model, scalers, model_type, seed,
                        family="unseen_short",
                        window=item["window"],
                        warmup_range=tuple(item["warmup_range"]),
                        forecast_range=tuple(item["forecast_range"]),
                        include_events=False,
                    )
                    _append_record(raw, record)
                    completed.add(identifier)
    update_status("UNSEEN_COMPLETE")

    for family, trajectories in (
        ("known_long", known),
        ("unseen_long", unseen),
    ):
        for model_type in MODEL_TYPES:
            for seed in FINAL_SEEDS:
                model, scalers = models[(model_type, seed)]
                for current, trajectory in trajectories.items():
                    identifier = _record_id(family, model_type, seed, current)
                    if identifier in completed:
                        continue
                    record = _fixed_rollout_record(
                        trajectory, model, scalers, model_type, seed,
                        family=family,
                        window=None,
                        warmup_range=(70_000, 72_000),
                        forecast_range=(72_000, trajectory.state_count - 1),
                        include_events=True,
                    )
                    _append_record(raw, record)
                    completed.add(identifier)
    update_status("LONG_HORIZON_COMPLETE")

    continuous = load_continuous_benchmark()
    for model_type in MODEL_TYPES:
        for seed in FINAL_SEEDS:
            identifier = _record_id("continuous", model_type, seed)
            if identifier in completed:
                continue
            model, scalers = models[(model_type, seed)]
            record = _continuous_rollout_record(
                continuous, model, scalers, model_type, seed
            )
            _append_record(raw, record)
            completed.add(identifier)
    update_status("CONTINUOUS_COMPLETE")
    return raw


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = tuple(records)
    values = [float(item["aggregate_nrmse_value"]) for item in items]
    if not values:
        return {
            "rollout_count": 0,
            "mean_nrmse": None,
            "population_std_nrmse": None,
            "median_nrmse": None,
            "worst_nrmse": None,
            "worst_rollout_id": None,
            "mean_valid_prediction_steps": None,
            "mean_valid_prediction_time": None,
            "divergence_count": 0,
            "collapse_count": 0,
        }
    worst_index = int(np.argmax(values))
    return {
        "rollout_count": len(items),
        "mean_nrmse": float(np.mean(values)),
        "population_std_nrmse": float(np.std(values, ddof=0)),
        "median_nrmse": float(np.median(values)),
        "worst_nrmse": float(values[worst_index]),
        "worst_rollout_id": items[worst_index]["record_id"],
        "mean_valid_prediction_steps": float(
            np.mean([item["metrics"]["valid_prediction_steps"] for item in items])
        ),
        "mean_valid_prediction_time": float(
            np.mean([item["metrics"]["valid_prediction_time"] for item in items])
        ),
        "divergence_count": int(
            sum(bool(item["metrics"]["diverged"]) for item in items)
        ),
        "collapse_count": int(
            sum(bool(item["metrics"]["prediction_collapse_any"]) for item in items)
        ),
    }


def aggregate_family(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = tuple(records)
    output: dict[str, Any] = {"models": {}}
    for model_type in MODEL_TYPES:
        model_records = [item for item in items if item["model_type"] == model_type]
        grouped_current: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        grouped_window: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        grouped_seed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in model_records:
            if item["current"] is not None:
                grouped_current[f"{float(item['current']):.2f}"].append(item)
            if item["window"] is not None:
                grouped_window[str(int(item["window"]))].append(item)
            grouped_seed[str(int(item["seed"]))].append(item)
        output["models"][model_type] = {
            "overall": _summary(model_records),
            "per_current": {
                key: _summary(value) for key, value in sorted(grouped_current.items())
            },
            "per_window": {
                key: _summary(value) for key, value in sorted(grouped_window.items())
            },
            "per_seed": {
                key: _summary(value) for key, value in sorted(grouped_seed.items())
            },
        }

    def pair_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        return (item["seed"], item["current"], item["window"])
    aware = {
        pair_key(item): item
        for item in items
        if item["model_type"] == PARAMETER_AWARE
    }
    baseline = {
        pair_key(item): item
        for item in items
        if item["model_type"] == ORDINARY_BASELINE
    }
    if set(aware) != set(baseline):
        raise Step8Error("aware/baseline records are not exactly paired")
    differences = [
        float(baseline[key]["aggregate_nrmse_value"])
        - float(aware[key]["aggregate_nrmse_value"])
        for key in sorted(aware, key=str)
    ]
    output["paired_baseline_minus_aware_nrmse"] = {
        "pair_count": len(differences),
        "mean": float(np.mean(differences)) if differences else None,
        "population_std": float(np.std(differences, ddof=0)) if differences else None,
        "median": float(np.median(differences)) if differences else None,
        "aware_better_count": int(sum(value > 0 for value in differences)),
        "baseline_better_count": int(sum(value < 0 for value in differences)),
        "tie_count": int(sum(value == 0 for value in differences)),
    }
    return output


def build_aggregates(raw: Mapping[str, Any]) -> dict[str, Any]:
    records = raw["records"]
    families = {}
    for family in (
        "known_short",
        "unseen_short",
        "known_long",
        "unseen_long",
        "continuous",
    ):
        families[family] = aggregate_family(
            [item for item in records if item["family"] == family]
        )
    return {
        "schema": STEP8_SCHEMA,
        "created_at": utc_now(),
        "failure_score_for_undefined_nrmse": NONFINITE_FAILURE_SCORE,
        "families": families,
    }


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def _flatten_record(item: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "record_id": item["record_id"],
        "family": item["family"],
        "model_type": item["model_type"],
        "seed": item["seed"],
        "current": item["current"],
        "window": item["window"],
        "warmup_start": item["warmup_range"][0],
        "warmup_stop": item["warmup_range"][1],
        "forecast_start": item["forecast_range"][0],
        "forecast_stop": item["forecast_range"][1],
        "aggregate_nrmse_value": item["aggregate_nrmse_value"],
        "numerical_failure": item["numerical_failure"],
        "failure_step": item["failure_step"],
        "failure_reason": item["failure_reason"],
        "raw_arrays_path": item["raw_arrays_path"],
        "raw_arrays_sha256": item["raw_arrays_sha256"],
    }
    row.update(item["metrics"])
    return {key: _csv_value(value) for key, value in row.items()}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    items = [dict(item) for item in rows]
    if not items:
        raise Step8Error(f"cannot write empty CSV {path}")
    fields = list(items[0])
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(items)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def write_result_tables(raw: Mapping[str, Any]) -> None:
    records = raw["records"]
    mapping = {
        "step8_known_heldout.csv": ("known_short",),
        "step8_unseen_currents.csv": ("unseen_short",),
        "step8_long_horizon.csv": ("known_long", "unseen_long"),
        "step8_continuous.csv": ("continuous",),
    }
    for name, families in mapping.items():
        rows = [_flatten_record(item) for item in records if item["family"] in families]
        write_csv(FINAL_RESULTS / name, rows)
    event_rows = []
    for item in records:
        event = item.get("event_metrics")
        if event is None:
            continue
        row = {
            "record_id": item["record_id"],
            "family": item["family"],
            "model_type": item["model_type"],
            "seed": item["seed"],
            "current": item["current"],
            "defined": event["defined"],
        }
        for side in ("prediction", "target"):
            for key, value in event[side].items():
                if key != "spike_indices":
                    row[f"{side}_{key}"] = _csv_value(value)
        for key, value in (event.get("errors") or {}).items():
            row[f"error_{key}"] = _csv_value(value)
        event_rows.append(row)
    write_csv(FINAL_RESULTS / "step8_event_metrics.csv", event_rows)


def _load_record_arrays(item: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = CHAPTER2_ROOT.parent / item["raw_arrays_path"]
    if file_sha256(path) != item["raw_arrays_sha256"]:
        raise Step8Error(f"raw array hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        return {key: np.asarray(saved[key]).copy() for key in saved.files}


def generate_figures(raw: Mapping[str, Any]) -> list[dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = FINAL_RESULTS / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    records = raw["records"]
    artifacts: list[dict[str, str]] = []

    def save(fig: Any, stem: str) -> None:
        fig.tight_layout()
        png = figure_dir / f"{stem}.png"
        pdf = figure_dir / f"{stem}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        artifacts.extend(
            [
                {"path": project_relative(png), "sha256": file_sha256(png)},
                {"path": project_relative(pdf), "sha256": file_sha256(pdf)},
            ]
        )

    def current_comparison(family: str, stem: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        currents = sorted(
            {float(item["current"]) for item in records if item["family"] == family}
        )
        x = np.arange(len(currents))
        width = 0.36
        for offset, model_type in ((-width / 2, PARAMETER_AWARE), (width / 2, ORDINARY_BASELINE)):
            means, stds = [], []
            for current in currents:
                values = [
                    item["aggregate_nrmse_value"]
                    for item in records
                    if item["family"] == family
                    and item["model_type"] == model_type
                    and item["current"] == current
                ]
                means.append(np.mean(values))
                stds.append(np.std(values, ddof=0))
            ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=model_type)
        ax.set_xticks(x, [f"I={value:.2f}" for value in currents])
        ax.set_ylabel("All-state NRMSE (mean ± population SD)")
        ax.set_xlabel("External current")
        ax.set_title(title)
        ax.legend()
        save(fig, stem)

    current_comparison(
        "known_short",
        "01_known_current_nrmse_by_current",
        "Known-current held-out forecasts | all seeds and windows",
    )
    current_comparison(
        "unseen_short",
        "02_unseen_current_nrmse_by_current",
        "Unseen-current forecasts | all seeds and windows",
    )

    def trace_figure(family: str, stem: str, title: str) -> None:
        selected = [
            item for item in records
            if item["family"] == family and item["seed"] == 42 and item["window"] == 1
        ]
        currents = sorted({float(item["current"]) for item in selected})
        fig, axes = plt.subplots(3, len(currents), figsize=(5.2 * len(currents), 8.0), squeeze=False)
        state_names = ("x", "y", "z")
        for column, current in enumerate(currents):
            pair = {
                item["model_type"]: _load_record_arrays(item)
                for item in selected if item["current"] == current
            }
            for state_index, state_name in enumerate(state_names):
                ax = axes[state_index, column]
                target = pair[PARAMETER_AWARE]["targets"][:, state_index]
                time_values = pair[PARAMETER_AWARE]["time"]
                ax.plot(time_values, target, color="black", linewidth=1, label="truth")
                ax.plot(time_values, pair[PARAMETER_AWARE]["predictions"][:, state_index], label="parameter-aware", alpha=0.85)
                ax.plot(time_values, pair[ORDINARY_BASELINE]["predictions"][:, state_index], label="ordinary baseline", alpha=0.75)
                ax.set_ylabel(state_name)
                ax.set_xlabel("Physical time")
                ax.set_title(f"I={current:.2f}, seed=42, window=1")
                if state_index == 0 and column == 0:
                    ax.legend()
        fig.suptitle(title)
        save(fig, stem)

    trace_figure(
        "known_short",
        "03_known_representative_truth_vs_prediction",
        "Known-current truth versus prediction | predetermined seed and window",
    )
    trace_figure(
        "unseen_short",
        "04_unseen_representative_truth_vs_prediction",
        "Unseen-current truth versus prediction | predetermined seed and window",
    )

    continuous_selected = {
        item["model_type"]: item
        for item in records
        if item["family"] == "continuous" and item["seed"] == 42
    }
    continuous_arrays = {
        model_type: _load_record_arrays(item)
        for model_type, item in continuous_selected.items()
    }
    sample_count = len(continuous_arrays[PARAMETER_AWARE]["time"])
    stride = max(1, sample_count // 20_000)
    index = slice(None, None, stride)
    time_values = continuous_arrays[PARAMETER_AWARE]["time"][index]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(time_values, continuous_arrays[PARAMETER_AWARE]["current"][index], color="tab:purple")
    axes[0].set_ylabel("I(t)")
    for state_index, state_name in enumerate(("x", "y", "z"), start=1):
        axes[state_index].plot(time_values, continuous_arrays[PARAMETER_AWARE]["targets"][index, state_index - 1], color="black", label="truth")
        axes[state_index].plot(time_values, continuous_arrays[PARAMETER_AWARE]["predictions"][index, state_index - 1], label="parameter-aware", alpha=0.85)
        axes[state_index].plot(time_values, continuous_arrays[ORDINARY_BASELINE]["predictions"][index, state_index - 1], label="ordinary baseline", alpha=0.7)
        axes[state_index].set_ylabel(state_name)
    axes[-1].set_xlabel("Physical time")
    axes[1].legend()
    fig.suptitle("Continuous changing-current forecast | seed=42 | range [2,000, 499,999)")
    save(fig, "05_continuous_current_and_state_predictions")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for model_type in MODEL_TYPES:
        arrays = continuous_arrays[model_type]
        error = arrays["pointwise_normalised_error"]
        finite = np.where(np.isfinite(error), error, np.nan)
        ax.plot(arrays["time"][index], finite[index], label=model_type, alpha=0.85)
    for boundary in CONTINUOUS_SWITCH_INDICES:
        boundary_time = (boundary + 1) * DT
        ax.axvline(boundary_time, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(VALID_PREDICTION_THRESHOLD, color="black", linestyle=":", label="VPT threshold")
    ax.set_xlabel("Physical time")
    ax.set_ylabel("Pointwise normalized error")
    ax.set_title("Continuous normalized error | seed=42 | current-change boundaries")
    ax.legend()
    save(fig, "06_continuous_normalised_error")

    fig, ax = plt.subplots(figsize=(9, 5))
    families = ("known_short", "unseen_short", "known_long", "unseen_long", "continuous")
    positions, labels, values = [], [], []
    position = 0
    for family in families:
        for model_type in MODEL_TYPES:
            group = [
                item["metrics"]["valid_prediction_time"]
                for item in records if item["family"] == family and item["model_type"] == model_type
            ]
            positions.append(position)
            labels.append(f"{family}\n{model_type}")
            values.append(group)
            position += 1
    ax.boxplot(values, positions=positions, showfliers=True)
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylabel("Valid prediction physical time")
    ax.set_title("Valid-prediction-time comparison | every rollout retained")
    save(fig, "07_valid_prediction_time_comparison")

    long_records = [item for item in records if item["family"] in ("known_long", "unseen_long")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    currents = sorted({float(item["current"]) for item in long_records})
    x = np.arange(len(currents))
    for axis, metric, label in (
        (axes[0], "spike_count", "Predicted spike count"),
        (axes[1], "burst_count", "Predicted burst count"),
    ):
        width = 0.36
        for offset, model_type in ((-width / 2, PARAMETER_AWARE), (width / 2, ORDINARY_BASELINE)):
            means = []
            for current in currents:
                values = [
                    item["event_metrics"]["prediction"][metric]
                    for item in long_records
                    if item["model_type"] == model_type
                    and item["current"] == current
                    and item["event_metrics"]["prediction"][metric] is not None
                ]
                means.append(np.mean(values) if values else 0.0)
            axis.bar(x + offset, means, width, label=model_type)
        axis.set_xticks(x, [f"{value:.2f}" for value in currents])
        axis.set_xlabel("External current")
        axis.set_ylabel(label)
        axis.set_title("Long horizon | all five seeds")
        axis.legend()
    save(fig, "08_long_horizon_spike_burst_summary")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for model_type in MODEL_TYPES:
        values = [
            item["aggregate_nrmse_value"]
            for item in records
            if item["family"] in ("known_short", "unseen_short")
            and item["model_type"] == model_type
        ]
        ax.hist(values, bins=20, alpha=0.55, label=model_type)
    ax.set_xlabel("All-state NRMSE")
    ax.set_ylabel("Rollout count")
    ax.set_title("Short-window NRMSE distribution | all seeds, currents, windows")
    ax.legend()
    save(fig, "09_rollout_nrmse_distribution")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = list(MODEL_TYPES)
    divergence = [
        sum(item["metrics"]["diverged"] for item in records if item["model_type"] == model_type)
        for model_type in MODEL_TYPES
    ]
    collapse = [
        sum(item["metrics"]["prediction_collapse_any"] for item in records if item["model_type"] == model_type)
        for model_type in MODEL_TYPES
    ]
    axes[0].bar(labels, divergence)
    axes[0].set_ylabel("Rollout count")
    axes[0].set_title("Divergence | all benchmark families")
    axes[1].bar(labels, collapse)
    axes[1].set_ylabel("Rollout count")
    axes[1].set_title("Prediction collapse | all benchmark families")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    save(fig, "10_divergence_collapse_summary")
    return artifacts


def _expected_record_keys() -> set[tuple[Any, ...]]:
    keys = set()
    for family, currents, windows in (
        ("known_short", TRAIN_CURRENTS, (1, 2, 3)),
        ("unseen_short", UNSEEN_CURRENTS, (1, 2, 3)),
        ("known_long", TRAIN_CURRENTS, (None,)),
        ("unseen_long", UNSEEN_CURRENTS, (None,)),
    ):
        for model_type in MODEL_TYPES:
            for seed in FINAL_SEEDS:
                for current in currents:
                    for window in windows:
                        keys.add((family, model_type, seed, current, window))
    for model_type in MODEL_TYPES:
        for seed in FINAL_SEEDS:
            keys.add(("continuous", model_type, seed, None, None))
    return keys


def verify_results(
    preflight: Mapping[str, Any],
    lock_hashes: Mapping[str, str],
    model_manifest: Mapping[str, Any],
    raw: Mapping[str, Any],
    aggregates: Mapping[str, Any],
    figure_artifacts: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    records = raw["records"]
    actual_keys = {
        (
            item["family"],
            item["model_type"],
            int(item["seed"]),
            item["current"],
            item["window"],
        )
        for item in records
    }
    expected_keys = _expected_record_keys()
    if actual_keys != expected_keys or len(actual_keys) != len(records):
        raise Step8Error("benchmark record matrix is incomplete or duplicated")
    current_locks = {
        project_relative(SELECTION_LOCK): file_sha256(SELECTION_LOCK),
        project_relative(EVALUATION_MANIFEST): file_sha256(EVALUATION_MANIFEST),
    }
    if current_locks != dict(lock_hashes):
        raise Step8Error("lock artifacts changed after benchmark access")
    current_datasets = validate_dataset_hashes()
    if current_datasets != preflight["dataset_hashes"]:
        raise Step8Error("dataset hashes changed after benchmark access")
    model_hashes = {}
    for item in model_manifest["models"]:
        path = CHAPTER2_ROOT.parent / item["path"]
        actual = file_sha256(path)
        if actual != item["sha256"]:
            raise Step8Error(f"final model changed after training: {path}")
        model_hashes[item["path"]] = actual
    for item in records:
        path = CHAPTER2_ROOT.parent / item["raw_arrays_path"]
        if file_sha256(path) != item["raw_arrays_sha256"]:
            raise Step8Error(f"raw rollout array changed: {path}")

    recomputed = build_aggregates(raw)
    if recomputed["families"] != aggregates["families"]:
        raise Step8Error("aggregate recomputation disagrees with saved results")
    independent: dict[str, Any] = {}
    for family in recomputed["families"]:
        independent[family] = {}
        for model_type in MODEL_TYPES:
            values = [
                float(item["aggregate_nrmse_value"])
                for item in records
                if item["family"] == family and item["model_type"] == model_type
            ]
            mean = math.fsum(values) / len(values)
            saved = aggregates["families"][family]["models"][model_type]["overall"]
            if not math.isclose(mean, saved["mean_nrmse"], rel_tol=1e-15, abs_tol=0.0):
                raise Step8Error("independent aggregate mean mismatch")
            independent[family][model_type] = {
                "count": len(values),
                "mean_nrmse_math_fsum": mean,
                "population_std_statistics": statistics.pstdev(values),
                "saved_mean_match": True,
            }

    chapter1_hash = tracked_non_chapter2_tree_hash()
    if chapter1_hash != preflight["chapter1_tracked_tree_hash"]:
        raise Step8Error("tracked Chapter 1 files changed during Step 8")
    seeds = {
        model_type: sorted(
            {int(item["seed"]) for item in records if item["model_type"] == model_type}
        )
        for model_type in MODEL_TYPES
    }
    if any(tuple(value) != FINAL_SEEDS for value in seeds.values()):
        raise Step8Error("not all five seeds were evaluated")
    return {
        "schema": STEP8_SCHEMA,
        "verified_at": utc_now(),
        "complete_record_count": len(records),
        "expected_record_count": len(expected_keys),
        "record_matrix_exact": True,
        "all_five_seeds_per_model": seeds,
        "no_failed_or_collapsed_rollout_omitted": True,
        "independent_aggregate_recomputation": independent,
        "selection_and_manifest_hashes_unchanged": current_locks,
        "dataset_hashes_unchanged": current_datasets,
        "final_model_hashes_unchanged": model_hashes,
        "chapter1_tracked_tree_hash_unchanged": chapter1_hash,
        "training_and_scaling_isolation": {
            "training_currents_only": list(TRAIN_CURRENTS),
            "transition_range_only": [0, STEP8_FINAL_TRAINING_STOP],
            "unseen_used": False,
            "continuous_used": False,
            "held_out_used": False,
        },
        "benchmark_results_changed_selection_or_models": False,
        "figure_artifacts": list(figure_artifacts),
        "runtime": runtime_record(),
    }


def aggregate_and_verify(
    preflight: Mapping[str, Any],
    lock_hashes: Mapping[str, str],
    model_manifest: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    aggregates = build_aggregates(raw)
    atomic_write_json(AGGREGATE_PATH, aggregates)
    write_result_tables(raw)
    figures = generate_figures(raw)
    update_status("AGGREGATION_COMPLETE")
    verification = verify_results(
        preflight, lock_hashes, model_manifest, raw, aggregates, figures
    )
    atomic_write_json(VERIFICATION_PATH, verification)
    update_status(
        "STEP8_COMPLETE",
        completed_record_ids=[item["record_id"] for item in raw["records"]],
        completed_record_count=len(raw["records"]),
        verification_hash=file_sha256(VERIFICATION_PATH),
        aggregate_hash=file_sha256(AGGREGATE_PATH),
        raw_results_hash=file_sha256(RAW_RESULTS_PATH),
        failure=None,
    )
    return aggregates, verification


def run_step8() -> None:
    # Resume checks are intentionally outside the status-writing error handler:
    # corruption must fail closed without rewriting any existing checkpoint.
    preflight = preflight_record()
    validate_resume_state(preflight)
    try:
        lock_hashes = lock_protocol(preflight)
        model_manifest = train_final_models(preflight, lock_hashes)
        raw = execute_benchmarks(preflight, lock_hashes, model_manifest)
        aggregate_and_verify(preflight, lock_hashes, model_manifest, raw)
    except (PreflightError, BenchmarkAccessError):
        raise
    except Exception as error:
        if STATUS_PATH.exists():
            update_status(
                "FAILED",
                failure={
                    "type": type(error).__name__,
                    "message": str(error),
                    "timestamp": utc_now(),
                },
            )
        raise
