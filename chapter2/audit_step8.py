"""Post-benchmark software-integrity and reservoir-seed stability audit.

This module is deliberately separate from :mod:`chapter2.esn_step8`.  It
does not train models, change predictions, select hyperparameters, or rewrite
the locked Step 8 artifacts.  It independently checks those artifacts, replays
the parameter-aware seed-456 records, recomputes metrics and aggregates, and
writes verification tables. Thesis-facing figures are generated separately by
``chapter2.plot_thesis_figures``; the audit does not regenerate figures by
default.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence, TextIO

import numpy as np

from chapter2.config_ch2 import DT
from chapter2.esn_config import (
    CHAPTER2_ROOT,
    FINAL_SEEDS,
    FIXED_DATASETS,
    LOCKED_DATASETS,
    STEP8_FINAL_TRAINING_STOP,
    STEP8_TRAINING_WASHOUT,
    TRAIN_CURRENTS,
    UNSEEN_CURRENTS,
)
from chapter2.esn_data import file_sha256, load_continuous_benchmark, load_fixed_trajectory
from chapter2.esn_metrics import evaluate_rollout, pointwise_normalised_error
from chapter2.esn_optimisation import (
    NONFINITE_FAILURE_SCORE,
    ORDINARY_BASELINE,
    PARAMETER_AWARE,
    load_strict_json,
)
from chapter2.esn_step8 import (
    AGGREGATE_PATH,
    EVALUATION_MANIFEST,
    FINAL_MODELS,
    FINAL_RESULTS,
    FULL_CONFIGURATIONS,
    MODEL_MANIFEST_PATH,
    MODEL_TYPES,
    RAW_RESULTS_PATH,
    SELECTION_LOCK,
    SOURCE_TRIALS,
    STATUS_PATH,
    VERIFICATION_PATH,
    final_model_config,
    load_final_model,
    load_final_training_prefixes,
    prepare_final_training,
    project_relative,
    reference_chapter1_tree_hash,
    save_final_model,
    tracked_non_chapter2_tree_hash,
)


AUDIT_SCHEMA = "chapter2_step8_seed_stability_audit_v1"
# Historical protected audit used by thesis figures and provenance checks.
AUDIT_PATH = FINAL_RESULTS / "step8_seed_stability_audit.json"
DEFAULT_FUTURE_AUDIT_PATH = (
    FINAL_RESULTS / "step8_seed_stability_audit_v2.json"
)
TABLE_DIR = FINAL_RESULTS / "tables_final"
SEEDS = tuple(int(seed) for seed in FINAL_SEEDS)
REPLAY_RELATIVE_TOLERANCE = 1.0e-7
REPLAY_ABSOLUTE_TOLERANCE = 1.0e-8
FAMILIES = (
    "known_short",
    "unseen_short",
    "known_long",
    "unseen_long",
    "continuous",
)


class Step8AuditError(RuntimeError):
    """Raised when an immutable Step 8 artifact fails the audit."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strict_json_text(value: Any) -> str:
    """Serialize a strict JSON document; non-finite values are forbidden."""
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_write(
    path: Path,
    writer: Callable[[TextIO], None],
) -> None:
    """Write a derived audit artifact atomically in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_strict_json(path: Path, value: Any) -> None:
    text = strict_json_text(value)
    _atomic_write(path, lambda stream: stream.write(text))


def validate_future_audit_path(path: Path) -> Path:
    """Require a new versioned JSON path and protect the historical audit."""
    destination = Path(path)
    if destination.resolve() == AUDIT_PATH.resolve():
        raise Step8AuditError("the historical audit path is immutable")
    if not re.fullmatch(r".+_v[0-9]+\.json", destination.name):
        raise Step8AuditError("future audit output must use a versioned filename")
    if destination.exists():
        raise Step8AuditError(f"refusing to overwrite audit output: {destination}")
    return destination


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _finite_or_none(value: float | np.floating[Any]) -> float | None:
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def _close(left: Any, right: Any, *, tolerance: float = 1.0e-12) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _close(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, Sequence) and isinstance(right, Sequence) and not isinstance(
        left, (str, bytes)
    ) and not isinstance(right, (str, bytes)):
        return len(left) == len(right) and all(
            _close(a, b, tolerance=tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    return left == right


def expected_record_keys() -> set[tuple[Any, ...]]:
    """Build the frozen record matrix without using the Step 8 verifier."""
    keys: set[tuple[Any, ...]] = set()
    specifications = (
        ("known_short", TRAIN_CURRENTS, (1, 2, 3)),
        ("unseen_short", UNSEEN_CURRENTS, (1, 2, 3)),
        ("known_long", TRAIN_CURRENTS, (None,)),
        ("unseen_long", UNSEEN_CURRENTS, (None,)),
    )
    for family, currents, windows in specifications:
        for model_type in MODEL_TYPES:
            for seed in SEEDS:
                for current in currents:
                    for window in windows:
                        keys.add((family, model_type, seed, current, window))
    for model_type in MODEL_TYPES:
        for seed in SEEDS:
            keys.add(("continuous", model_type, seed, None, None))
    return keys


def record_matrix_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    actual_list = [
        (
            item["family"],
            item["model_type"],
            int(item["seed"]),
            item["current"],
            item["window"],
        )
        for item in records
    ]
    expected = expected_record_keys()
    actual = set(actual_list)
    if actual != expected or len(actual_list) != len(actual):
        raise Step8AuditError("record matrix is incomplete or contains duplicates")
    family_counts = {
        family: sum(item["family"] == family for item in records)
        for family in FAMILIES
    }
    expected_counts = {
        "known_short": 90,
        "unseen_short": 60,
        "known_long": 30,
        "unseen_long": 20,
        "continuous": 10,
    }
    if family_counts != expected_counts:
        raise Step8AuditError(f"unexpected family counts: {family_counts}")
    return {
        "exact": True,
        "total": len(records),
        "family_counts_both_models": family_counts,
        "duplicate_count": 0,
        "missing_count": 0,
        "seeds_by_model": {
            model_type: sorted(
                {int(item["seed"]) for item in records if item["model_type"] == model_type}
            )
            for model_type in MODEL_TYPES
        },
    }


def artifact_hash_inventory(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Hash every immutable scientific input and Step 8 numerical output."""
    aggregate_files = (
        AGGREGATE_PATH,
        RAW_RESULTS_PATH,
        STATUS_PATH,
        VERIFICATION_PATH,
        FINAL_RESULTS / "step8_known_heldout.csv",
        FINAL_RESULTS / "step8_unseen_currents.csv",
        FINAL_RESULTS / "step8_long_horizon.csv",
        FINAL_RESULTS / "step8_continuous.csv",
        FINAL_RESULTS / "step8_event_metrics.csv",
    )
    raw_paths = sorted(
        {CHAPTER2_ROOT.parent / str(item["raw_arrays_path"]) for item in records}
    )
    if len(raw_paths) != 210:
        raise Step8AuditError(f"expected 210 raw arrays, found {len(raw_paths)}")
    return {
        "selection_lock": {
            project_relative(SELECTION_LOCK): file_sha256(SELECTION_LOCK)
        },
        "evaluation_manifest": {
            project_relative(EVALUATION_MANIFEST): file_sha256(EVALUATION_MANIFEST)
        },
        "model_manifest": {
            project_relative(MODEL_MANIFEST_PATH): file_sha256(MODEL_MANIFEST_PATH)
        },
        "final_model_npz": {
            project_relative(path): file_sha256(path)
            for path in sorted(FINAL_MODELS.glob("*_seed_*.npz"))
        },
        "datasets": {
            project_relative(record.path): file_sha256(record.path)
            for record in LOCKED_DATASETS
        },
        "raw_step8_arrays": {
            project_relative(path): file_sha256(path) for path in raw_paths
        },
        "aggregate_json_and_csv": {
            project_relative(path): file_sha256(path) for path in aggregate_files
        },
    }


def validate_model_identity(
    model_type: str,
    seed: int,
    configuration: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    """Validate one manifest/config/metadata identity tuple."""
    expected = final_model_config(model_type, seed)
    if dict(configuration) != expected.__dict__:
        raise Step8AuditError(f"configuration mismatch for {model_type} seed {seed}")
    if metadata.get("model_type") != model_type or int(metadata.get("seed", -1)) != seed:
        raise Step8AuditError(f"metadata mismatch for {model_type} seed {seed}")
    if int(metadata.get("source_trial", -1)) != SOURCE_TRIALS[model_type]:
        raise Step8AuditError(f"source-trial mismatch for {model_type} seed {seed}")
    if metadata.get("training_transition_range") != [0, STEP8_FINAL_TRAINING_STOP]:
        raise Step8AuditError(f"training-range mismatch for {model_type} seed {seed}")
    if metadata.get("scaler_fit_transition_range") != [0, STEP8_FINAL_TRAINING_STOP]:
        raise Step8AuditError(f"scaler-range mismatch for {model_type} seed {seed}")


def model_integrity_audit() -> tuple[dict[str, Any], dict[tuple[str, int], Any]]:
    manifest = load_strict_json(MODEL_MANIFEST_PATH)
    selection = load_strict_json(SELECTION_LOCK)
    if manifest.get("model_count") != 10 or len(manifest.get("models", [])) != 10:
        raise Step8AuditError("final model manifest does not contain ten models")

    prefixes = load_final_training_prefixes()
    expected_scalers = {
        model_type: prepare_final_training(prefixes, model_type)[0]
        for model_type in MODEL_TYPES
    }
    models: dict[tuple[str, int], Any] = {}
    records: list[dict[str, Any]] = []
    weight_digests: dict[str, list[str]] = defaultdict(list)
    for item in manifest["models"]:
        model_type = str(item["model_type"])
        seed = int(item["seed"])
        path = CHAPTER2_ROOT.parent / item["path"]
        actual_hash = file_sha256(path)
        if actual_hash != item["sha256"]:
            raise Step8AuditError(f"model hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as saved:
            if any(saved[name].dtype.kind == "O" for name in saved.files):
                raise Step8AuditError(f"object array in model bundle: {path}")
        model, scalers, metadata = load_final_model(path)
        validate_model_identity(model_type, seed, item["configuration"], metadata)
        if dict(selection["models"][model_type]["configuration"]) != dict(
            FULL_CONFIGURATIONS[model_type]
        ):
            raise Step8AuditError(f"selection lock mismatch for {model_type}")
        expected = expected_scalers[model_type]
        scaler_pairs = (
            (scalers.state.mean, expected.state.mean),
            (scalers.state.scale, expected.state.scale),
            (scalers.current.mean, expected.current.mean),
            (scalers.current.scale, expected.current.scale),
        )
        scaler_max_absolute_difference = max(
            float(np.max(np.abs(left - right))) for left, right in scaler_pairs
        )
        scaler_tolerance = 2.0 * np.finfo(np.float64).eps
        scaler_match = all(
            np.allclose(left, right, rtol=0.0, atol=scaler_tolerance)
            for left, right in scaler_pairs
        )
        if not scaler_match:
            raise Step8AuditError(f"training-prefix scaler mismatch: {path}")
        arrays = (
            model.input_weights,
            model.reservoir_weights,
            model.reservoir_bias,
            model.output_weights,
        )
        if not all(np.all(np.isfinite(values)) for values in arrays):
            raise Step8AuditError(f"non-finite model weights: {path}")
        requested_radius = float(model.config.spectral_radius)
        realised_radius = float(model.spectral_radius)
        radius_match = math.isclose(
            realised_radius, requested_radius, rel_tol=1.0e-8, abs_tol=1.0e-10
        )
        if not radius_match:
            raise Step8AuditError(f"spectral-radius mismatch: {path}")
        edge_count = model.config.reservoir_size * (model.config.reservoir_size - 1)
        probability = float(model.config.reservoir_connectivity)
        connectivity_sd = math.sqrt(probability * (1.0 - probability) / edge_count)
        connectivity_tolerance = 5.0 * connectivity_sd + 1.0 / edge_count
        connectivity_match = abs(model.realised_connectivity - probability) <= connectivity_tolerance
        if not connectivity_match:
            raise Step8AuditError(f"realised connectivity is implausible: {path}")

        for name, values in (
            ("input", model.input_weights),
            ("reservoir", model.reservoir_weights),
            ("readout", model.output_weights),
        ):
            weight_digests[f"{model_type}:{name}"].append(_array_digest(values))

        round_trip_exact = False
        if (model_type, seed) in {
            (PARAMETER_AWARE, 456),
            (PARAMETER_AWARE, 42),
            (ORDINARY_BASELINE, 42),
        }:
            test_input = np.zeros(model.config.input_dimension, dtype=float)
            model.reset_reservoir()
            expected_prediction = model.predict_one_step(test_input)
            model.reset_reservoir()
            with tempfile.TemporaryDirectory(prefix="chapter2-audit-") as directory:
                temporary = Path(directory) / "round_trip.npz"
                save_final_model(temporary, model, scalers, metadata)
                loaded, loaded_scalers, loaded_metadata = load_final_model(temporary)
                actual_prediction = loaded.predict_one_step(test_input)
                round_trip_exact = bool(
                    np.array_equal(expected_prediction, actual_prediction)
                    and loaded_metadata == metadata
                    and np.array_equal(loaded_scalers.state.mean, scalers.state.mean)
                    and np.array_equal(loaded_scalers.state.scale, scalers.state.scale)
                    and np.array_equal(loaded_scalers.current.mean, scalers.current.mean)
                    and np.array_equal(loaded_scalers.current.scale, scalers.current.scale)
                )
            if not round_trip_exact:
                raise Step8AuditError(f"model round-trip mismatch: {path}")

        models[(model_type, seed)] = (model, scalers)
        records.append(
            {
                "model_type": model_type,
                "seed": seed,
                "path": project_relative(path),
                "sha256": actual_hash,
                "safe_allow_pickle_false_load": True,
                "all_weights_finite": True,
                "input_dimension": model.config.input_dimension,
                "output_dimension": model.config.output_dimension,
                "configuration_and_metadata_match_lock": True,
                "training_and_scaler_range": [0, STEP8_FINAL_TRAINING_STOP],
                "scalers_match_training_prefix_recomputation": True,
                "scaler_comparison_absolute_tolerance": scaler_tolerance,
                "scaler_maximum_absolute_difference": scaler_max_absolute_difference,
                "input_weight_frobenius_norm": float(np.linalg.norm(model.input_weights)),
                "reservoir_weight_frobenius_norm": float(
                    np.linalg.norm(model.reservoir_weights)
                ),
                "readout_weight_frobenius_norm": float(np.linalg.norm(model.output_weights)),
                "requested_connectivity": probability,
                "realised_connectivity": model.realised_connectivity,
                "connectivity_five_sigma_tolerance": connectivity_tolerance,
                "connectivity_match": True,
                "requested_spectral_radius": requested_radius,
                "realised_spectral_radius": realised_radius,
                "spectral_radius_match": True,
                "round_trip_prediction_exact_when_checked": (
                    round_trip_exact
                    if (model_type, seed)
                    in {
                        (PARAMETER_AWARE, 456),
                        (PARAMETER_AWARE, 42),
                        (ORDINARY_BASELINE, 42),
                    }
                    else None
                ),
            }
        )

    expected_keys = {(model_type, seed) for model_type in MODEL_TYPES for seed in SEEDS}
    if set(models) != expected_keys:
        raise Step8AuditError("model/seed matrix is incomplete")
    accidental_sharing = {
        key: len(values) != len(set(values)) for key, values in weight_digests.items()
    }
    if any(accidental_sharing.values()):
        raise Step8AuditError(f"duplicate seed-specific weights detected: {accidental_sharing}")
    return (
        {
            "model_count": 10,
            "five_seeds_per_model": True,
            "records": sorted(
                records, key=lambda item: (MODEL_TYPES.index(item["model_type"]), SEEDS.index(item["seed"]))
            ),
            "seed_specific_weight_digests_all_unique_within_model": True,
            "accidental_cross_seed_artifact_sharing": False,
            "shared_scalers_are_intentional_and_match_training_prefixes": True,
            "scaler_recomputation_note": (
                "Saved values match a fresh permitted-prefix recomputation within "
                "two binary64 epsilons; the largest observed difference is one ULP."
            ),
        },
        models,
    )


def _load_raw_arrays(item: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = CHAPTER2_ROOT.parent / item["raw_arrays_path"]
    actual_hash = file_sha256(path)
    if actual_hash != item["raw_arrays_sha256"]:
        raise Step8AuditError(f"raw-array hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        if tuple(saved.files) != (
            "predictions",
            "targets",
            "pointwise_normalised_error",
            "time",
            "current",
        ):
            raise Step8AuditError(f"raw-array schema mismatch: {path}")
        if any(saved[name].dtype.kind == "O" for name in saved.files):
            raise Step8AuditError(f"unsafe raw-array dtype: {path}")
        return {name: np.asarray(saved[name]).copy() for name in saved.files}


def _trajectory_for_record(
    item: Mapping[str, Any],
    fixed: Mapping[float, Any],
    continuous: Any,
) -> Any:
    return continuous if item["family"] == "continuous" else fixed[float(item["current"])]


def _aligned_expected_arrays(item: Mapping[str, Any], trajectory: Any) -> dict[str, np.ndarray]:
    start, stop = (int(value) for value in item["forecast_range"])
    return {
        "targets": trajectory.states[start + 1 : stop + 1],
        "time": trajectory.time[start + 1 : stop + 1],
        "current": trajectory.current_values[start:stop],
    }


def replay_record(
    item: Mapping[str, Any],
    model: Any,
    scalers: Any,
    trajectory: Any,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    """Independently replay one record with explicit recursive feedback."""
    warm_start, warm_stop = (int(value) for value in item["warmup_range"])
    forecast_start, forecast_stop = (int(value) for value in item["forecast_range"])
    if warm_stop != forecast_start:
        raise Step8AuditError(f"non-adjacent ranges in {item['record_id']}")
    state_rows = scalers.state.transform(trajectory.states[warm_start:warm_stop])
    if item["model_type"] == PARAMETER_AWARE:
        current_rows = scalers.current.transform(
            trajectory.current_values[warm_start:warm_stop, None]
        )
        warmup = np.column_stack((state_rows, current_rows))
    else:
        warmup = state_rows
    model.teacher_forced_warmup(warmup, reset=True)
    state = scalers.state.transform(
        trajectory.states[forecast_start : forecast_start + 1]
    )[0]
    horizon = forecast_stop - forecast_start
    predictions_scaled = np.full((horizon, 3), np.nan, dtype=float)
    scaled_currents = (
        scalers.current.transform(
            trajectory.current_values[forecast_start:forecast_stop, None]
        )[:, 0]
        if item["model_type"] == PARAMETER_AWARE
        else None
    )
    failure_step: int | None = None
    failure_reason: str | None = None
    with np.errstate(over="ignore", invalid="ignore"):
        for index in range(horizon):
            model_input = (
                np.concatenate((state, [scaled_currents[index]]))
                if scaled_currents is not None
                else state
            )
            prediction = model.predict_one_step(model_input)
            predictions_scaled[index] = prediction
            if not np.all(np.isfinite(prediction)):
                failure_step = index
                failure_reason = "non_finite_prediction"
                break
            state = prediction
            if progress and (index + 1) % 50_000 == 0:
                print(
                    f"replay {item['record_id']}: {index + 1}/{horizon}",
                    flush=True,
                )
    model.reset_reservoir()
    predictions = predictions_scaled * scalers.state.scale + scalers.state.mean
    raw = _load_raw_arrays(item)
    difference = np.abs(predictions - raw["predictions"])
    finite_difference = difference[np.isfinite(difference)]
    max_difference = float(np.max(finite_difference)) if finite_difference.size else 0.0
    exact = bool(np.array_equal(predictions, raw["predictions"], equal_nan=True))
    within_tolerance = bool(
        np.allclose(
            predictions,
            raw["predictions"],
            rtol=REPLAY_RELATIVE_TOLERANCE,
            atol=REPLAY_ABSOLUTE_TOLERANCE,
            equal_nan=True,
        )
    )
    if not within_tolerance:
        raise Step8AuditError(f"forecast replay mismatch: {item['record_id']}")
    if failure_step != item["failure_step"] or failure_reason != item["failure_reason"]:
        raise Step8AuditError(f"failure metadata replay mismatch: {item['record_id']}")
    return {
        "record_id": item["record_id"],
        "model_type": item["model_type"],
        "seed": int(item["seed"]),
        "family": item["family"],
        "current": item["current"],
        "window": item["window"],
        "prediction_dtype": str(raw["predictions"].dtype),
        "comparison_relative_tolerance": REPLAY_RELATIVE_TOLERANCE,
        "comparison_absolute_tolerance": REPLAY_ABSOLUTE_TOLERANCE,
        "tolerance_rationale": (
            "Accumulated binary64 BLAS roundoff across recursive steps on the "
            "audit node; substantially tighter than forecast or metric scales."
        ),
        "exact_array_match": exact,
        "within_dtype_tolerance": True,
        "maximum_absolute_difference": max_difference,
        "failure_step_match": True,
    }


def replay_audit(
    records: Sequence[Mapping[str, Any]],
    models: Mapping[tuple[str, int], Any],
) -> dict[str, Any]:
    fixed = {
        current: load_fixed_trajectory(current)
        for current in (*TRAIN_CURRENTS, *UNSEEN_CURRENTS)
    }
    continuous = load_continuous_benchmark()
    seed456 = [
        item
        for item in records
        if item["model_type"] == PARAMETER_AWARE and int(item["seed"]) == 456
    ]
    controls = [
        item
        for item in records
        if item["family"] == "known_short"
        and float(item["current"]) == 3.20
        and int(item["window"]) == 1
        and int(item["seed"]) == 42
        and item["model_type"] in (PARAMETER_AWARE, ORDINARY_BASELINE)
    ]
    if len(seed456) != 21 or len(controls) != 2:
        raise Step8AuditError("replay scope is not the expected 21+2 records")
    selected = seed456 + controls
    output = []
    for number, item in enumerate(selected, start=1):
        print(f"replaying {number}/{len(selected)} {item['record_id']}", flush=True)
        model, scalers = models[(item["model_type"], int(item["seed"]))]
        output.append(
            replay_record(
                item,
                model,
                scalers,
                _trajectory_for_record(item, fixed, continuous),
                progress=item["family"] == "continuous",
            )
        )
    return {
        "scope": (
            "All 21 parameter-aware seed-456 benchmark records plus the "
            "predetermined known-current seed-42/window-1 control for each model."
        ),
        "seed456_record_count": len(seed456),
        "stable_control_record_count": len(controls),
        "total_replayed": len(output),
        "all_within_saved_float64_tolerance": all(
            item["within_dtype_tolerance"] for item in output
        ),
        "all_exact": all(item["exact_array_match"] for item in output),
        "records": output,
    }


def teacher_forced_case(
    model: Any,
    scalers: Any,
    trajectory: Any,
    model_type: str,
    *,
    warmup_range: tuple[int, int] = (70_000, 72_000),
    forecast_range: tuple[int, int] = (72_000, 80_000),
) -> dict[str, Any]:
    warm_start, warm_stop = warmup_range
    start, stop = forecast_range
    warm_states = scalers.state.transform(trajectory.states[warm_start:warm_stop])
    if model_type == PARAMETER_AWARE:
        warm_currents = scalers.current.transform(
            trajectory.current_values[warm_start:warm_stop, None]
        )
        warmup = np.column_stack((warm_states, warm_currents))
    else:
        warmup = warm_states
    model.teacher_forced_warmup(warmup, reset=True)
    true_states = scalers.state.transform(trajectory.states[start:stop])
    scaled_currents = scalers.current.transform(
        trajectory.current_values[start:stop, None]
    )[:, 0]
    predictions = np.empty((stop - start, 3), dtype=float)
    for index, true_state in enumerate(true_states):
        model_input = (
            np.concatenate((true_state, [scaled_currents[index]]))
            if model_type == PARAMETER_AWARE
            else true_state
        )
        predictions[index] = model.predict_one_step(model_input)
    model.reset_reservoir()
    physical = predictions * scalers.state.scale + scalers.state.mean
    targets = trajectory.states[start + 1 : stop + 1]
    metrics = evaluate_rollout(
        physical,
        targets,
        normalisation_scale=scalers.state.scale,
        dt=DT,
        valid_prediction_threshold=0.4,
        divergence_threshold=5.0,
    ).to_dict()
    return {
        "nrmse_state": metrics["nrmse_state"],
        "valid_prediction_steps": metrics["valid_prediction_steps"],
        "diverged": metrics["diverged"],
        "divergence_index": metrics["divergence_index"],
    }


def stability_diagnostics(
    records: Sequence[Mapping[str, Any]],
    models: Mapping[tuple[str, int], Any],
) -> dict[str, Any]:
    fixed = {
        current: load_fixed_trajectory(current)
        for current in (*TRAIN_CURRENTS, *UNSEEN_CURRENTS)
    }
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        print(f"teacher-forced diagnostic parameter-aware seed={seed}", flush=True)
        model, scalers = models[(PARAMETER_AWARE, seed)]
        cases = {
            f"I={current:.2f}": teacher_forced_case(
                model, scalers, fixed[current], PARAMETER_AWARE
            )
            for current in (*TRAIN_CURRENTS, *UNSEEN_CURRENTS)
        }
        teacher_values = [float(item["nrmse_state"]) for item in cases.values()]
        autonomous = [
            item
            for item in records
            if item["model_type"] == PARAMETER_AWARE
            and int(item["seed"]) == seed
            and item["family"] in ("known_short", "unseen_short")
            and int(item["window"]) == 1
        ]
        per_seed[str(seed)] = {
            "teacher_forced_window1_by_current": cases,
            "teacher_forced_mean_nrmse": float(statistics.fmean(teacher_values)),
            "teacher_forced_max_nrmse": float(max(teacher_values)),
            "teacher_forced_divergence_count": sum(
                bool(item["diverged"]) for item in cases.values()
            ),
            "autonomous_window1_mean_nrmse": float(
                statistics.fmean(float(item["aggregate_nrmse_value"]) for item in autonomous)
            ),
            "autonomous_window1_divergence_count": sum(
                bool(item["metrics"]["diverged"]) for item in autonomous
            ),
            "all_family_divergence_count": sum(
                bool(item["metrics"]["diverged"])
                for item in records
                if item["model_type"] == PARAMETER_AWARE and int(item["seed"]) == seed
            ),
        }

    seed456_records = [
        item
        for item in records
        if item["model_type"] == PARAMETER_AWARE and int(item["seed"]) == 456
    ]
    rollout_details = []
    for item in seed456_records:
        arrays = _load_raw_arrays(item)
        pointwise = arrays["pointwise_normalised_error"]
        first_vpt = np.flatnonzero(pointwise >= float(item["metrics"]["valid_prediction_threshold"]))
        first_vpt_index = int(first_vpt[0]) if len(first_vpt) else None
        divergence_index = item["metrics"]["divergence_index"]
        stop = int(divergence_index) if divergence_index is not None else len(arrays["predictions"])
        before = arrays["predictions"][:stop]
        maximum = float(np.max(np.abs(before))) if before.size else None
        rollout_details.append(
            {
                "record_id": item["record_id"],
                "first_valid_prediction_threshold_crossing_index": first_vpt_index,
                "divergence_index": divergence_index,
                "divergence_reason": item["metrics"]["divergence_reason"],
                "maximum_absolute_physical_state_before_divergence": maximum,
                "maximum_before_divergence_defined": maximum is not None,
                "numerical_failure": item["numerical_failure"],
            }
        )

    model456, _ = models[(PARAMETER_AWARE, 456)]
    return {
        "diagnostic_status": "post_hoc_software_integrity_and_stability_audit_only",
        "not_a_primary_benchmark_metric": True,
        "comparison_with_all_parameter_aware_seeds": per_seed,
        "seed456_rollouts": rollout_details,
        "seed456_weight_norms": {
            "norm": "Frobenius",
            "reservoir": float(np.linalg.norm(model456.reservoir_weights)),
            "input": float(np.linalg.norm(model456.input_weights)),
            "readout": float(np.linalg.norm(model456.output_weights)),
            "realised_spectral_radius": float(model456.spectral_radius),
        },
    }


def ridge_condition_diagnostic(models: Mapping[tuple[str, int], Any]) -> dict[str, Any]:
    """Recompute the exact seed-456 final-training ridge system condition."""
    print("recomputing seed-456 final-training ridge system", flush=True)
    prefixes = load_final_training_prefixes()
    _, sequences = prepare_final_training(prefixes, PARAMETER_AWARE)
    model, _ = models[(PARAMETER_AWARE, 456)]
    statistics_record = model.accumulate_ridge_statistics(
        sequences, washout=STEP8_TRAINING_WASHOUT
    )
    system = (
        statistics_record.gram
        + model.config.ridge_regularisation * model.ridge_penalty_matrix
    )
    condition = float(np.linalg.cond(system))
    model.reset_reservoir()
    return {
        "defined": math.isfinite(condition),
        "value": condition if math.isfinite(condition) else None,
        "matrix": "exact final-training ridge system G + lambda*P",
        "sample_count": statistics_record.sample_count,
        "feature_dimension": model.feature_dimension,
        "washout_per_trajectory": STEP8_TRAINING_WASHOUT,
        "interpretation_scope": "diagnostic only; no threshold or selection use",
    }


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(records)
    if not items:
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
    values = [float(item["aggregate_nrmse_value"]) for item in items]
    worst = max(range(len(items)), key=lambda index: values[index])
    return {
        "rollout_count": len(items),
        "mean_nrmse": float(statistics.fmean(values)),
        "population_std_nrmse": float(statistics.pstdev(values)),
        "median_nrmse": float(statistics.median(values)),
        "worst_nrmse": values[worst],
        "worst_rollout_id": items[worst]["record_id"],
        "mean_valid_prediction_steps": float(
            statistics.fmean(float(item["metrics"]["valid_prediction_steps"]) for item in items)
        ),
        "mean_valid_prediction_time": float(
            statistics.fmean(float(item["metrics"]["valid_prediction_time"]) for item in items)
        ),
        "divergence_count": sum(bool(item["metrics"]["diverged"]) for item in items),
        "collapse_count": sum(
            bool(item["metrics"]["prediction_collapse_any"]) for item in items
        ),
    }


def independent_family_aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"models": {}}
    for model_type in MODEL_TYPES:
        model_records = [item for item in records if item["model_type"] == model_type]
        current_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        window_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        seed_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in model_records:
            if item["current"] is not None:
                current_groups[f"{float(item['current']):.2f}"].append(item)
            if item["window"] is not None:
                window_groups[str(int(item["window"]))].append(item)
            seed_groups[str(int(item["seed"]))].append(item)
        output["models"][model_type] = {
            "overall": _summary(model_records),
            "per_current": {key: _summary(value) for key, value in sorted(current_groups.items())},
            "per_window": {key: _summary(value) for key, value in sorted(window_groups.items())},
            "per_seed": {key: _summary(value) for key, value in sorted(seed_groups.items())},
        }

    def pair_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        return int(item["seed"]), item["current"], item["window"]

    aware = {pair_key(item): item for item in records if item["model_type"] == PARAMETER_AWARE}
    baseline = {
        pair_key(item): item for item in records if item["model_type"] == ORDINARY_BASELINE
    }
    if set(aware) != set(baseline):
        raise Step8AuditError("model records are not exactly paired")
    differences = [
        float(baseline[key]["aggregate_nrmse_value"])
        - float(aware[key]["aggregate_nrmse_value"])
        for key in sorted(aware, key=str)
    ]
    output["paired_baseline_minus_aware_nrmse"] = {
        "pair_count": len(differences),
        "mean": float(statistics.fmean(differences)),
        "population_std": float(statistics.pstdev(differences)),
        "median": float(statistics.median(differences)),
        "aware_better_count": sum(value > 0 for value in differences),
        "baseline_better_count": sum(value < 0 for value in differences),
        "tie_count": sum(value == 0 for value in differences),
    }
    return output


def independent_recomputation(
    records: Sequence[Mapping[str, Any]],
    models: Mapping[tuple[str, int], Any],
) -> dict[str, Any]:
    fixed = {
        current: load_fixed_trajectory(current)
        for current in (*TRAIN_CURRENTS, *UNSEEN_CURRENTS)
    }
    continuous = load_continuous_benchmark()
    metric_matches = 0
    alignment_matches = 0
    hash_matches = 0
    maximum_aggregate_input_difference = 0.0
    for number, item in enumerate(records, start=1):
        arrays = _load_raw_arrays(item)
        hash_matches += 1
        trajectory = _trajectory_for_record(item, fixed, continuous)
        aligned = _aligned_expected_arrays(item, trajectory)
        if not all(np.array_equal(arrays[name], values) for name, values in aligned.items()):
            raise Step8AuditError(f"raw target/time/current alignment mismatch: {item['record_id']}")
        alignment_matches += 1
        _, scalers = models[(item["model_type"], int(item["seed"]))]
        recomputed_metrics = evaluate_rollout(
            arrays["predictions"],
            arrays["targets"],
            normalisation_scale=scalers.state.scale,
            dt=DT,
            valid_prediction_threshold=float(item["metrics"]["valid_prediction_threshold"]),
            divergence_threshold=float(item["metrics"]["divergence_threshold"]),
            collapse_std_ratio_threshold=float(
                item["metrics"]["collapse_std_ratio_threshold"]
            ),
        ).to_dict()
        recomputed_pointwise = pointwise_normalised_error(
            arrays["predictions"], arrays["targets"], normalisation_scale=scalers.state.scale
        )
        if not np.array_equal(
            recomputed_pointwise,
            arrays["pointwise_normalised_error"],
            equal_nan=True,
        ):
            raise Step8AuditError(f"pointwise error mismatch: {item['record_id']}")
        if not _close(recomputed_metrics, item["metrics"], tolerance=1.0e-13):
            raise Step8AuditError(f"metric recomputation mismatch: {item['record_id']}")
        expected_aggregate = (
            float(recomputed_metrics["nrmse_state"])
            if recomputed_metrics["nrmse_state"] is not None
            else NONFINITE_FAILURE_SCORE
        )
        aggregate_difference = abs(
            expected_aggregate - float(item["aggregate_nrmse_value"])
        )
        maximum_aggregate_input_difference = max(
            maximum_aggregate_input_difference, aggregate_difference
        )
        if not math.isclose(expected_aggregate, float(item["aggregate_nrmse_value"]), rel_tol=1.0e-13, abs_tol=1.0e-15):
            raise Step8AuditError(f"aggregate input mismatch: {item['record_id']}")
        metric_matches += 1
        if number % 25 == 0 or number == len(records):
            print(f"independent metric recomputation {number}/{len(records)}", flush=True)

    independent = {
        family: independent_family_aggregate(
            [item for item in records if item["family"] == family]
        )
        for family in FAMILIES
    }
    saved = load_strict_json(AGGREGATE_PATH)["families"]
    if not _close(independent, saved, tolerance=1.0e-12):
        raise Step8AuditError("independent aggregates disagree with saved aggregates")
    overall = {
        family: {
            model_type: independent[family]["models"][model_type]["overall"]
            for model_type in MODEL_TYPES
        }
        for family in FAMILIES
    }
    return {
        "raw_array_hash_matches": hash_matches,
        "target_time_current_alignment_matches": alignment_matches,
        "metric_recomputations_match": metric_matches,
        "aggregate_input_comparison_relative_tolerance": 1.0e-13,
        "maximum_aggregate_input_absolute_difference": maximum_aggregate_input_difference,
        "aggregate_recomputation_matches": True,
        "equal_record_weighting_verified": True,
        "population_standard_deviation_verified": True,
        "divergent_records_included": True,
        "collapsed_records_included": True,
        "short_long_known_unseen_families_separate": True,
        "validation_metrics_reported_as_step8_benchmark": False,
        "overall": overall,
    }


def event_validity_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correction_root = FINAL_RESULTS / "post_benchmark_event_correction"
    manifest = load_strict_json(correction_root / "correction_manifest.json")
    pre = load_strict_json(correction_root / "pre_correction" / "step8_raw_results.json")
    post = load_strict_json(correction_root / "post_correction" / "step8_raw_results.json")
    current = load_strict_json(RAW_RESULTS_PATH)
    if file_sha256(correction_root / "post_correction" / "step8_raw_results.json") != file_sha256(
        RAW_RESULTS_PATH
    ):
        raise Step8AuditError("current raw JSON is not the preserved post-correction copy")
    if file_sha256(correction_root / "pre_correction" / "step8_aggregate_results.json") != file_sha256(
        correction_root / "post_correction" / "step8_aggregate_results.json"
    ):
        raise Step8AuditError("state aggregates changed during event correction")

    pre_by_id = {item["record_id"]: item for item in pre["records"]}
    post_by_id = {item["record_id"]: item for item in post["records"]}
    changed_intervals: list[dict[str, Any]] = []
    for record_id in sorted(pre_by_id):
        before_record = pre_by_id[record_id]
        after_record = post_by_id[record_id]
        before_copy = json.loads(json.dumps(before_record))
        after_copy = json.loads(json.dumps(after_record))
        if before_record["family"] == "continuous":
            before_intervals = before_copy.pop("per_current_interval")
            after_intervals = after_copy.pop("per_current_interval")
            for before_interval, after_interval in zip(before_intervals, after_intervals):
                before_base = dict(before_interval)
                after_base = dict(after_interval)
                before_event = before_base.pop("event_metrics")
                after_event = after_base.pop("event_metrics")
                if before_base != after_base:
                    raise Step8AuditError("continuous state metrics changed during correction")
                if before_event != after_event:
                    if not (
                        after_event.get("defined") is False
                        and after_event.get("errors") is None
                        and "divergence" in str(after_event.get("reason", ""))
                    ):
                        raise Step8AuditError("event invalidation schema is incomplete")
                    changed_intervals.append(
                        {
                            "record_id": record_id,
                            "current": after_interval["current"],
                            "transition_range": after_interval["transition_range"],
                        }
                    )
        if before_copy != after_copy:
            raise Step8AuditError("non-event raw record content changed during correction")
    if len(changed_intervals) != 6:
        raise Step8AuditError(f"expected six corrected intervals, got {len(changed_intervals)}")

    invalid_primary = 0
    invalid_intervals = 0
    for item in records:
        event = item.get("event_metrics")
        if event is not None and bool(item["metrics"]["diverged"]):
            if event.get("defined") is not False or event.get("errors") is not None:
                raise Step8AuditError(f"divergent event remains defined: {item['record_id']}")
            invalid_primary += 1
        for interval in item.get("per_current_interval", []):
            if bool(interval["metrics"]["diverged"]):
                event = interval["event_metrics"]
                if event.get("defined") is not False or event.get("errors") is not None:
                    raise Step8AuditError(
                        f"divergent interval event remains defined: {item['record_id']}"
                    )
                invalid_intervals += 1
    if len(current.get("post_benchmark_corrections", [])) != 1:
        raise Step8AuditError("post-benchmark correction provenance is missing")
    return {
        "verified": True,
        "correction_manifest_sha256": file_sha256(
            correction_root / "correction_manifest.json"
        ),
        "pre_correction_raw_sha256": file_sha256(
            correction_root / "pre_correction" / "step8_raw_results.json"
        ),
        "post_correction_raw_sha256": file_sha256(
            correction_root / "post_correction" / "step8_raw_results.json"
        ),
        "current_matches_post_correction_copy": True,
        "changed_continuous_interval_count": len(changed_intervals),
        "changed_intervals": changed_intervals,
        "divergent_primary_event_records_invalid": invalid_primary,
        "divergent_continuous_interval_events_invalid": invalid_intervals,
        "invalid_schema": {
            "defined": False,
            "errors": None,
            "reason_semantics": "prediction diverged",
        },
        "predictions_changed": False,
        "state_metrics_changed": False,
        "nrmse_changed": False,
        "valid_prediction_time_changed": False,
        "models_scalers_locks_or_protocol_changed": False,
        "primary_state_aggregates_changed": False,
        "event_csv_note": (
            "The corrected objects are per-current continuous subintervals; the "
            "existing event CSV contains only primary-record events. Its hash "
            "therefore correctly remains unchanged, and undefined primary rows "
            "have empty error columns."
        ),
        "manifest_affected_count_match": manifest["correction"]["affected_interval_count"]
        == len(changed_intervals),
    }


def divergence_collapse_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in FAMILIES:
        output[family] = {}
        for model_type in MODEL_TYPES:
            items = [
                item
                for item in records
                if item["family"] == family and item["model_type"] == model_type
            ]
            divergence = sum(bool(item["metrics"]["diverged"]) for item in items)
            collapse = sum(
                bool(item["metrics"]["prediction_collapse_any"]) for item in items
            )
            output[family][model_type] = {
                "record_count": len(items),
                "divergence_count": divergence,
                "divergence_percent": 100.0 * divergence / len(items),
                "collapse_count": collapse,
                "collapse_percent": 100.0 * collapse / len(items),
            }
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    items = [dict(row) for row in rows]
    if not items:
        raise Step8AuditError(f"cannot write empty table: {path}")
    def write(stream: TextIO) -> None:
        writer = csv.DictWriter(stream, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)

    _atomic_write(path, write)


def event_error_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    long_records = [item for item in records if item["family"] in ("known_long", "unseen_long")]
    for current in (*TRAIN_CURRENTS, *UNSEEN_CURRENTS):
        for model_type in MODEL_TYPES:
            items = [
                item
                for item in long_records
                if float(item["current"]) == current and item["model_type"] == model_type
            ]
            row: dict[str, Any] = {
                "current": current,
                "current_status": "known" if current in TRAIN_CURRENTS else "unseen",
                "model_type": model_type,
                "rollout_count": len(items),
                "event_record_contributor_count": sum(
                    bool(item["event_metrics"]["defined"]) for item in items
                ),
            }
            for metric in ("spike_count", "burst_count"):
                key = f"{metric}_absolute_error"
                flag = f"{metric}_error_defined"
                values = [
                    float(item["event_metrics"]["errors"][key])
                    for item in items
                    if item["event_metrics"]["defined"]
                    and item["event_metrics"]["errors"].get(flag) is True
                ]
                row[f"{metric}_error_mean"] = (
                    float(statistics.fmean(values)) if values else None
                )
                row[f"{metric}_error_median"] = (
                    float(statistics.median(values)) if values else None
                )
                row[f"{metric}_error_contributor_count"] = len(values)
            rows.append(row)
    return rows


def write_final_tables(
    records: Sequence[Mapping[str, Any]],
    recomputation: Mapping[str, Any],
    counts: Mapping[str, Any],
) -> list[dict[str, str]]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    aggregate_rows = []
    for family in FAMILIES:
        for model_type in MODEL_TYPES:
            summary = recomputation["overall"][family][model_type]
            aggregate_rows.append({"family": family, "model_type": model_type, **summary})
    per_seed_rows = []
    for family in FAMILIES:
        for model_type in MODEL_TYPES:
            for seed in SEEDS:
                items = [
                    item
                    for item in records
                    if item["family"] == family
                    and item["model_type"] == model_type
                    and int(item["seed"]) == seed
                ]
                values = [float(item["aggregate_nrmse_value"]) for item in items]
                vpt = [float(item["metrics"]["valid_prediction_time"]) for item in items]
                per_seed_rows.append(
                    {
                        "family": family,
                        "model_type": model_type,
                        "seed": seed,
                        "record_count": len(items),
                        "mean_nrmse": statistics.fmean(values),
                        "median_nrmse": statistics.median(values),
                        "worst_nrmse": max(values),
                        "mean_valid_prediction_time": statistics.fmean(vpt),
                        "median_valid_prediction_time": statistics.median(vpt),
                        "divergence_count": sum(
                            bool(item["metrics"]["diverged"]) for item in items
                        ),
                        "collapse_count": sum(
                            bool(item["metrics"]["prediction_collapse_any"])
                            for item in items
                        ),
                    }
                )
    count_rows = [
        {"family": family, "model_type": model_type, **counts[family][model_type]}
        for family in FAMILIES
        for model_type in MODEL_TYPES
    ]
    collapse_rows = [
        {
            "family": family,
            "parameter_aware_collapse_count": counts[family][PARAMETER_AWARE][
                "collapse_count"
            ],
            "ordinary_baseline_collapse_count": counts[family][ORDINARY_BASELINE][
                "collapse_count"
            ],
        }
        for family in FAMILIES
    ]
    paths_and_rows = (
        (TABLE_DIR / "01_recomputed_aggregate_summary.csv", aggregate_rows),
        (TABLE_DIR / "02_per_seed_nrmse_vpt_summary.csv", per_seed_rows),
        (TABLE_DIR / "03_divergence_rate_summary.csv", count_rows),
        (TABLE_DIR / "04_collapse_summary.csv", collapse_rows),
        (TABLE_DIR / "05_event_error_contributors.csv", event_error_rows(records)),
    )
    artifacts = []
    for path, rows in paths_and_rows:
        _write_csv(path, rows)
        artifacts.append({"path": project_relative(path), "sha256": file_sha256(path)})
    return artifacts


def representative_records(
    records: Sequence[Mapping[str, Any]], family: str
) -> list[Mapping[str, Any]]:
    """Return the manifest-preselected seed-42/window-1 records."""
    selected = [
        item
        for item in records
        if item["family"] == family
        and int(item["seed"]) == 42
        and int(item["window"]) == 1
    ]
    return sorted(selected, key=lambda item: (float(item["current"]), item["model_type"]))


def strict_json_audit(paths: Iterable[Path]) -> dict[str, Any]:
    checked = []
    for path in paths:
        load_strict_json(path)
        checked.append(project_relative(path))
    return {"all_strict": True, "checked_count": len(checked), "paths": checked}


def run_audit(
    *,
    compute_condition: bool = True,
    output_path: Path = DEFAULT_FUTURE_AUDIT_PATH,
) -> dict[str, Any]:
    output_path = validate_future_audit_path(output_path)
    raw = load_strict_json(RAW_RESULTS_PATH)
    records = raw["records"]
    status = load_strict_json(STATUS_PATH)
    verification = load_strict_json(VERIFICATION_PATH)
    baseline_hashes = artifact_hash_inventory(records)
    matrix = record_matrix_audit(records)
    model_audit, models = model_integrity_audit()
    replay = replay_audit(records, models)
    diagnostics = stability_diagnostics(records, models)
    diagnostics["ridge_system_condition"] = (
        ridge_condition_diagnostic(models)
        if compute_condition
        else {
            "defined": False,
            "value": None,
            "reason": "explicitly skipped by command-line option",
        }
    )
    recomputation = independent_recomputation(records, models)
    events = event_validity_audit(records)
    counts = divergence_collapse_counts(records)
    tables = write_final_tables(records, recomputation, counts)
    final_hashes = artifact_hash_inventory(records)
    if final_hashes != baseline_hashes:
        raise Step8AuditError("an immutable scientific artifact changed during audit")

    chapter1_hash = tracked_non_chapter2_tree_hash()
    reference_chapter1_hash = reference_chapter1_tree_hash()
    legacy_chapter1_hash = status["preflight"]["chapter1_tracked_tree_hash"]
    if chapter1_hash != reference_chapter1_hash:
        raise Step8AuditError("tracked Chapter 1 tree changed")
    if status.get("state") != "STEP8_COMPLETE" or status.get("completed_record_count") != 210:
        raise Step8AuditError("Step 8 status is not complete")
    if verification.get("complete_record_count") != 210:
        raise Step8AuditError("saved verification is incomplete")

    aware_divergence = {
        family: counts[family][PARAMETER_AWARE]["divergence_count"] for family in FAMILIES
    }
    expected_aware = {
        "known_short": 9,
        "unseen_short": 6,
        "known_long": 3,
        "unseen_long": 2,
        "continuous": 2,
    }
    if aware_divergence != expected_aware:
        raise Step8AuditError(f"unexpected aware divergence distribution: {aware_divergence}")
    if any(counts[family][ORDINARY_BASELINE]["divergence_count"] for family in FAMILIES):
        raise Step8AuditError("baseline divergence count is not zero")
    if any(
        counts[family][model_type]["collapse_count"]
        for family in FAMILIES
        for model_type in MODEL_TYPES
    ):
        raise Step8AuditError("primary collapse count is not zero")

    seed456 = diagnostics["comparison_with_all_parameter_aware_seeds"]["456"]
    genuine_closed_loop = (
        seed456["teacher_forced_divergence_count"] == 0
        and seed456["autonomous_window1_divergence_count"] == 5
        and seed456["all_family_divergence_count"] == 21
        and replay["all_within_saved_float64_tolerance"]
    )
    if not genuine_closed_loop:
        raise Step8AuditError("seed-456 evidence does not support the expected classification")

    json_paths = sorted(FINAL_RESULTS.glob("*.json")) + sorted(
        (FINAL_RESULTS / "post_benchmark_event_correction").rglob("*.json")
    ) + [MODEL_MANIFEST_PATH]
    strict = strict_json_audit(json_paths)
    audit = {
        "schema": AUDIT_SCHEMA,
        "created_at": utc_now(),
        "producing_audit_source_sha256": file_sha256(Path(__file__)),
        "scope": "post_hoc software-integrity and stability audit; not model selection",
        "classification": {
            "seed": 456,
            "model_type": PARAMETER_AWARE,
            "software_defect": False,
            "genuine_closed_loop_reservoir_seed_instability": True,
            "basis": (
                "The saved model and scalers are internally consistent, all stored "
                "forecasts replay from the bundle, teacher-forced one-step forecasts "
                "remain finite and accurate, and instability appears only after "
                "recursive feedback."
            ),
        },
        "generic_implementation_defect_found": False,
        "benchmark_rerun_required": False,
        "original_scientific_outputs_changed": False,
        "frozen_design_changed": False,
        "record_matrix": matrix,
        "artifact_hashes_before_and_after_audit": {
            "unchanged": True,
            "hashes": baseline_hashes,
        },
        "model_integrity": model_audit,
        "forecast_replay": replay,
        "stability_diagnostics": diagnostics,
        "independent_recomputation": recomputation,
        "divergence_and_collapse": counts,
        "event_validity": events,
        "strict_json": strict,
        "chapter1_integrity": {
            "unchanged": True,
            "legacy_preflight_tracked_tree_sha256": legacy_chapter1_hash,
            "reference_tracked_tree_sha256": reference_chapter1_hash,
            "current_tracked_tree_sha256": chapter1_hash,
        },
        "presentation": {
            "figure_generation_separated_from_audit": True,
            "thesis_figure_directory": "chapter2/final_results/figures_thesis",
            "thesis_figure_command": "python3 -m chapter2.plot_thesis_figures",
            "revised_table_directory": project_relative(TABLE_DIR),
            "figure_artifacts": [],
            "table_artifacts": tables,
        },
        "scientific_interpretation": (
            "Supplying the external current can produce exceptionally accurate and "
            "long-lasting forecasts for stable reservoir realizations. However, the "
            "parameter-aware ESN is sensitive to reservoir initialization, as "
            "demonstrated by the repeated divergence of seed 456, and it does not "
            "show consistently reliable long-horizon generalisation to unseen currents."
        ),
        "limitations": [
            "Only five frozen reservoir seeds were evaluated.",
            "Teacher-forced errors are post-hoc integrity diagnostics, not new primary metrics.",
            "Divergence denotes the locked normalized-error threshold, not necessarily non-finite numerical overflow.",
            "The continuous-current dataset is a supporting transition benchmark, not the primary unseen-current test.",
        ],
        "verdict": "AUDIT PASSED — SCIENTIFIC SEED INSTABILITY PRESERVED",
    }
    write_strict_json(output_path, audit)
    return audit


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-condition",
        action="store_true",
        help="Skip the exact ridge-system condition diagnostic.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FUTURE_AUDIT_PATH,
        help="new versioned audit JSON path (existing files are never overwritten)",
    )
    args = parser.parse_args()
    audit = run_audit(
        compute_condition=not args.skip_condition,
        output_path=args.output,
    )
    print(audit["verdict"], flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
