"""Leakage-safe mechanics for the frozen Chapter 2 cross-regime experiment.

This module composes the existing Chapter 2 simulator, ESN, scalers, metrics,
event definitions, and atomic writers.  Existing scientific files are read but
never rewritten.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence, TextIO
import zipfile

import numpy as np
from numpy.lib import format as npy_format

from chapter2.cross_regime_config import (
    ALL_CURRENTS,
    CHAOTIC_CURRENTS,
    CONTINUOUS_SCHEDULES,
    CONTINUOUS_STATE_COUNT,
    CONTINUOUS_SWITCH_INDICES,
    EFFECTIVE_TRAINING_BUDGET,
    EXPECTED_MODELS,
    EXPECTED_RECORDS,
    RAW_TRAINING_TRANSITIONS,
    REGULAR_CURRENTS,
    SCENARIO_TRAINING_CURRENTS,
    SEEDS,
    TRAINING_WASHOUT,
    block_order,
    model_config,
)
from chapter2.esn_config import FIXED_DATASETS, REQUIRED_ARRAY_KEYS, fixed_dataset
from chapter2.esn_data import (
    ContinuousCurrentTrajectory,
    FixedCurrentTrajectory,
    NumpyStandardScaler,
    StateCurrentScalers,
    file_sha256,
)
from chapter2.cross_regime_numerics import (
    evaluate_predictions,
    first_nonfinite_prediction_step,
    metric_prediction_view,
)
from chapter2.esn_model import EchoStateNetwork, TrainingSequence
from chapter2.esn_optimisation import NONFINITE_FAILURE_SCORE
from chapter2.esn_step8 import (
    MODEL_BUNDLE_SCHEMA,
    atomic_save_npz,
    event_metrics,
    invalidate_divergent_event_metrics,
    load_final_model,
    save_final_model,
)


SCHEMA = "chapter2_cross_regime_v1"
RAW_ARRAY_KEYS = (
    "predictions",
    "targets",
    "pointwise_normalised_error",
    "time",
    "current",
)


class CrossRegimeError(RuntimeError):
    """Raised when a frozen cross-regime contract is violated."""


def strict_load_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=reject)
    if not isinstance(value, dict):
        raise CrossRegimeError(f"JSON root is not an object: {path}")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(path, text)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    items = [dict(row) for row in rows]
    if not items:
        raise CrossRegimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_npy_header(stream: Any) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = npy_format.read_magic(stream)
    if version == (1, 0):
        return npy_format.read_array_header_1_0(stream)
    if version == (2, 0):
        return npy_format.read_array_header_2_0(stream)
    raise CrossRegimeError(f"unsupported NPY header version {version}")


def _read_member_prefix(
    archive: zipfile.ZipFile,
    key: str,
    *,
    expected_count: int,
    prefix_count: int,
) -> np.ndarray:
    """Read exactly ``prefix_count`` values without loading a held-out suffix."""
    with archive.open(f"{key}.npy") as stream:
        shape, fortran_order, dtype = _read_npy_header(stream)
        if shape != (expected_count,) or fortran_order or dtype != np.dtype("float64"):
            raise CrossRegimeError(f"unexpected locked array header for {key}")
        raw = stream.read(prefix_count * dtype.itemsize)
    if len(raw) != prefix_count * dtype.itemsize:
        raise CrossRegimeError(f"short prefix read for {key}")
    return np.frombuffer(raw, dtype=dtype, count=prefix_count).copy()


def load_training_prefix(current: float, transition_count: int) -> FixedCurrentTrajectory:
    """Load only state samples ``[0, transition_count]`` for one training block."""
    record = fixed_dataset(float(current))
    if transition_count <= TRAINING_WASHOUT or transition_count > 70_000:
        raise CrossRegimeError("invalid training transition count")
    if file_sha256(record.path) != record.sha256:
        raise CrossRegimeError(f"training dataset hash mismatch: {record.path}")
    prefix_count = transition_count + 1
    with zipfile.ZipFile(record.path) as archive:
        if tuple(name[:-4] for name in archive.namelist()) != REQUIRED_ARRAY_KEYS:
            raise CrossRegimeError(f"dataset member schema mismatch: {record.path}")
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
    trajectory = FixedCurrentTrajectory(
        float(current), arrays["t"], states, arrays["I"], record.path
    )
    if trajectory.state_count != prefix_count:
        raise CrossRegimeError("prefix loader exposed an incorrect state count")
    return trajectory


def prepare_training(
    scenario: str,
    seed: int,
    *,
    trajectories: Mapping[float, FixedCurrentTrajectory] | None = None,
    washout: int = TRAINING_WASHOUT,
) -> tuple[StateCurrentScalers, tuple[TrainingSequence, ...], dict[str, Any]]:
    """Fit scenario-only scalers and construct independent chronological blocks."""
    if scenario not in SCENARIO_TRAINING_CURRENTS or seed not in SEEDS:
        raise CrossRegimeError("unknown scenario or seed")
    allocations = RAW_TRAINING_TRANSITIONS[scenario]
    if trajectories is None:
        blocks = {
            current: load_training_prefix(current, transitions)
            for current, transitions in allocations.items()
        }
    else:
        blocks = {float(current): value for current, value in trajectories.items()}
        if set(blocks) != set(allocations):
            raise CrossRegimeError("training trajectory membership mismatch")
        for current, trajectory in blocks.items():
            expected_states = allocations[current] + 1
            if trajectory.state_count != expected_states:
                raise CrossRegimeError(
                    f"training prefix for I={current} must expose {expected_states} states"
                )

    fitting_states = np.concatenate(
        [blocks[current].states[:-1] for current in allocations], axis=0
    )
    fitting_currents = np.concatenate(
        [blocks[current].current_values[:-1] for current in allocations]
    ).reshape(-1, 1)
    scalers = StateCurrentScalers(
        NumpyStandardScaler.fit(fitting_states),
        NumpyStandardScaler.fit(fitting_currents),
    )
    sequences: list[TrainingSequence] = []
    for current in block_order(scenario, seed):
        trajectory = blocks[current]
        state_inputs = scalers.state.transform(trajectory.states[:-1])
        targets = scalers.state.transform(trajectory.states[1:])
        current_inputs = scalers.current.transform(
            trajectory.current_values[:-1, None]
        )
        sequences.append(
            TrainingSequence(np.column_stack((state_inputs, current_inputs)), targets)
        )

    effective = sum(len(sequence.inputs) - washout for sequence in sequences)
    expected_effective = sum(length - washout for length in allocations.values())
    if washout == TRAINING_WASHOUT and effective != EFFECTIVE_TRAINING_BUDGET:
        raise CrossRegimeError("effective training budget is not 130,000")
    provenance = {
        "scenario": scenario,
        "seed": seed,
        "training_currents": list(SCENARIO_TRAINING_CURRENTS[scenario]),
        "block_order": list(block_order(scenario, seed)),
        "raw_transitions": {f"{key:.2f}": value for key, value in allocations.items()},
        "washout_per_independently_reset_block": washout,
        "effective_samples": effective,
        "expected_effective_samples_for_supplied_washout": expected_effective,
        "scaler_fit_scope": "state and current inputs from scenario training prefixes only",
        "state_scaler_mean": scalers.state.mean.tolist(),
        "state_scaler_scale": scalers.state.scale.tolist(),
        "current_scaler_mean": scalers.current.mean.tolist(),
        "current_scaler_scale": scalers.current.scale.tolist(),
        "artificial_cross_trajectory_transitions": False,
        "reservoir_reset_between_blocks": True,
    }
    return scalers, tuple(sequences), provenance


def train_scenario_model(
    scenario: str,
    seed: int,
) -> tuple[EchoStateNetwork, StateCurrentScalers, dict[str, Any]]:
    scalers, sequences, provenance = prepare_training(scenario, seed)
    model = EchoStateNetwork(model_config(seed))
    model.fit(sequences, washout=TRAINING_WASHOUT)
    return model, scalers, provenance


def save_model_bundle(
    path: Path,
    model: EchoStateNetwork,
    scalers: StateCurrentScalers,
    metadata: Mapping[str, Any],
) -> None:
    save_final_model(path, model, scalers, metadata)


def load_model_bundle(
    path: Path,
) -> tuple[EchoStateNetwork, StateCurrentScalers, dict[str, Any]]:
    return load_final_model(path)


def recursive_forecast(
    model: EchoStateNetwork,
    scalers: StateCurrentScalers,
    states: np.ndarray,
    currents: np.ndarray,
    *,
    warmup_range: tuple[int, int],
    forecast_range: tuple[int, int],
    reset: bool = True,
) -> tuple[np.ndarray, int | None, str | None]:
    """Warm once, then feed back predictions while supplying only true current."""
    warm_start, warm_stop = warmup_range
    forecast_start, forecast_stop = forecast_range
    if warm_stop != forecast_start:
        raise CrossRegimeError("warm-up and forecast ranges must be adjacent")
    if forecast_stop > len(states) - 1:
        raise CrossRegimeError("forecast exceeds available transitions")
    warm_inputs = np.column_stack(
        (states[warm_start:warm_stop], currents[warm_start:warm_stop])
    )
    model.teacher_forced_warmup(scalers.transform_inputs(warm_inputs), reset=reset)
    state = scalers.state.transform(states[forecast_start : forecast_start + 1])[0]
    scaled_currents = scalers.current.transform(
        currents[forecast_start:forecast_stop, None]
    )[:, 0]
    predictions = np.full((forecast_stop - forecast_start, 3), np.nan)
    failure_step: int | None = None
    failure_reason: str | None = None
    with np.errstate(over="ignore", invalid="ignore"):
        for index, supplied_current in enumerate(scaled_currents):
            try:
                prediction = model.predict_one_step(
                    np.concatenate((state, [supplied_current]))
                )
            except (ValueError, FloatingPointError):
                failure_step = index
                failure_reason = "non_finite_recursive_input"
                break
            if not np.all(np.isfinite(prediction)):
                failure_step = index
                failure_reason = "non_finite_prediction"
                break
            physical_prediction = (
                prediction * scalers.state.scale + scalers.state.mean
            )
            if not np.all(np.isfinite(physical_prediction)):
                failure_step = index
                failure_reason = "non_finite_physical_prediction"
                break
            predictions[index] = physical_prediction
            state = prediction
    model.reset_reservoir()
    return predictions, failure_step, failure_reason


def evaluation_classification(scenario: str, current: float) -> str:
    if scenario == "mixed_shuffled":
        return "mixed-regime temporal holdout"
    training = set(SCENARIO_TRAINING_CURRENTS[scenario])
    if current in training:
        return "training-current temporal holdout"
    source_regular = scenario == "regular_to_chaotic"
    target_regular = current in REGULAR_CURRENTS
    if source_regular != target_regular:
        return "cross-regime generalization"
    return "same-regime evaluation"


def record_id(
    family: str,
    scenario: str,
    seed: int,
    *,
    current: float | None = None,
    window: int | None = None,
    schedule: str | None = None,
) -> str:
    parts = [family, scenario, f"seed_{seed}"]
    if current is not None:
        parts.append(f"I_{current:.2f}".replace(".", "p"))
    if window is not None:
        parts.append(f"window_{window}")
    if schedule is not None:
        parts.append(schedule)
    return "__".join(parts)


def expected_record_ids() -> set[str]:
    identifiers: set[str] = set()
    for scenario in SCENARIO_TRAINING_CURRENTS:
        for seed in SEEDS:
            for current in ALL_CURRENTS:
                for window in (1, 2, 3):
                    identifiers.add(
                        record_id("fixed_short", scenario, seed, current=current, window=window)
                    )
                identifiers.add(record_id("fixed_long", scenario, seed, current=current))
            for schedule in CONTINUOUS_SCHEDULES:
                identifiers.add(
                    record_id("continuous", scenario, seed, schedule=schedule)
                )
    if len(identifiers) != EXPECTED_RECORDS:
        raise CrossRegimeError("expected-record construction is not 345")
    return identifiers


def expected_model_keys() -> set[tuple[str, int]]:
    keys = {
        (scenario, seed)
        for scenario in SCENARIO_TRAINING_CURRENTS
        for seed in SEEDS
    }
    if len(keys) != EXPECTED_MODELS:
        raise CrossRegimeError("expected-model construction is not 15")
    return keys


def validate_record_matrix(records: Sequence[Mapping[str, Any]]) -> None:
    identifiers = [str(item["record_id"]) for item in records]
    if len(identifiers) != len(set(identifiers)):
        raise CrossRegimeError("duplicate evaluation record")
    expected = expected_record_ids()
    if set(identifiers) != expected:
        missing = sorted(expected - set(identifiers))
        unexpected = sorted(set(identifiers) - expected)
        raise CrossRegimeError(
            f"evaluation matrix mismatch; missing={missing[:3]}, unexpected={unexpected[:3]}"
        )


def transition_boundary_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    currents: np.ndarray,
    scale: np.ndarray,
    *,
    forecast_start: int,
    half_window: int = 2_000,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for boundary in CONTINUOUS_SWITCH_INDICES:
        local = boundary - forecast_start
        start = max(0, local - half_window)
        stop = min(len(predictions), local + half_window)
        fields, _, _ = evaluate_predictions(
            predictions[start:stop],
            targets[start:stop],
            scale,
        )
        results.append(
            {
                "boundary_state_index": boundary,
                "current_before": float(currents[local - 1]),
                "current_after": float(currents[local]),
                "raw_array_range": [start, stop],
                "transition_range": [forecast_start + start, forecast_start + stop],
                "metrics": fields["metrics"],
            }
        )
    return results


def pointwise_sha256(pointwise: np.ndarray) -> str:
    """Hash derived errors independently of the immutable historical archive."""
    return sha256(np.ascontiguousarray(pointwise, dtype="<f8").tobytes()).hexdigest()


def derive_record_fields(
    record: Mapping[str, Any], arrays: Mapping[str, np.ndarray], scale: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """Derive metadata only; never forecast, read a model, or write an archive."""
    fields, physical, pointwise = evaluate_predictions(
        arrays["predictions"], arrays["targets"], scale
    )
    metric_predictions = metric_prediction_view(physical, fields["failure_step"])
    fields["event_metrics"] = invalidate_divergent_event_metrics(
        event_metrics(metric_predictions, arrays["targets"], 0.01), fields["metrics"]
    )
    if record["family"] == "continuous":
        fields["transition_boundaries"] = transition_boundary_metrics(
            metric_predictions, arrays["targets"], arrays["current"], scale,
            forecast_start=record["forecast_range"][0],
        )
    return fields, pointwise


def validate_record_source(
    record: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
    trajectory: FixedCurrentTrajectory | ContinuousCurrentTrajectory,
) -> None:
    """Verify frozen identity, window and exact target/time/current alignment."""
    expected_id = record_id(
        record["family"], record["scenario"], int(record["seed"]),
        current=record["current"], window=record["window"], schedule=record["schedule"],
    )
    if record["record_id"] != expected_id:
        raise CrossRegimeError("record identity mismatch")
    if record["family"] == "fixed_short":
        if record["window"] not in (1, 2, 3):
            raise CrossRegimeError("evaluation window mismatch")
        start = (70_000, 80_000, 89_999)[record["window"] - 1]
        warmup, forecast = [start, start + 2_000], [start + 2_000, start + 10_000]
    elif record["family"] == "fixed_long":
        warmup, forecast = [70_000, 72_000], [72_000, 99_999]
    elif record["family"] == "continuous":
        warmup, forecast = [0, 2_000], [2_000, 499_999]
    else:
        raise CrossRegimeError("unknown evaluation family")
    if list(record["warmup_range"]) != warmup or list(record["forecast_range"]) != forecast:
        raise CrossRegimeError("evaluation window mismatch")
    begin, stop = forecast
    expected = {
        "targets": trajectory.states[begin + 1:stop + 1],
        "time": trajectory.time[begin + 1:stop + 1],
        "current": trajectory.current_values[begin:stop],
    }
    if not all(np.array_equal(arrays[key], value) for key, value in expected.items()):
        raise CrossRegimeError("raw targets/time/current differ from source trajectory")
    expected_class = (
        "continuous mixed-regime transition schedule" if record["family"] == "continuous"
        else evaluation_classification(record["scenario"], record["current"])
    )
    if record["evaluation_class"] != expected_class:
        raise CrossRegimeError("evaluation classification mismatch")


def build_evaluation_record(
    *,
    identifier: str,
    family: str,
    scenario: str,
    seed: int,
    scalers: StateCurrentScalers,
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    currents: np.ndarray,
    raw_path: Path,
    warmup_range: tuple[int, int],
    forecast_range: tuple[int, int],
    failure_step: int | None,
    failure_reason: str | None,
    current: float | None = None,
    window: int | None = None,
    schedule: str | None = None,
) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=float).copy()
    physical_failure_step = first_nonfinite_prediction_step(predictions)
    if physical_failure_step is not None:
        predictions[physical_failure_step:] = np.nan
    record: dict[str, Any] = {
        "record_id": identifier,
        "family": family,
        "scenario": scenario,
        "seed": seed,
        "current": current,
        "window": window,
        "schedule": schedule,
        "evaluation_class": (
            evaluation_classification(scenario, current)
            if current is not None
            else "continuous mixed-regime transition schedule"
        ),
        "warmup_range": list(warmup_range),
        "forecast_range": list(forecast_range),
        "generation_failure_step": failure_step,
        "generation_failure_reason": failure_reason,
        "raw_arrays_path": str(raw_path),
    }
    fields, pointwise = derive_record_fields(
        record, {"predictions": predictions, "targets": targets, "current": currents},
        scalers.state.scale,
    )
    record.update(fields)
    atomic_save_npz(
        raw_path, predictions=predictions, targets=targets,
        pointwise_normalised_error=pointwise, time=times, current=currents,
    )
    record["raw_arrays_sha256"] = file_sha256(raw_path)
    return record


def validate_raw_array(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = Path(str(record["raw_arrays_path"]))
    if file_sha256(path) != record["raw_arrays_sha256"]:
        raise CrossRegimeError(f"raw-array hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        if tuple(saved.files) != RAW_ARRAY_KEYS:
            raise CrossRegimeError(f"raw-array schema mismatch: {path}")
        if any(saved[key].dtype.kind == "O" for key in saved.files):
            raise CrossRegimeError(f"unsafe raw-array dtype: {path}")
        arrays = {key: np.asarray(saved[key]).copy() for key in saved.files}
    horizon = record["forecast_range"][1] - record["forecast_range"][0]
    if arrays["predictions"].shape != (horizon, 3) or arrays["targets"].shape != (horizon, 3):
        raise CrossRegimeError(f"raw-array state shape mismatch: {path}")
    if any(arrays[key].shape != (horizon,) for key in ("pointwise_normalised_error", "time", "current")):
        raise CrossRegimeError(f"raw-array vector shape mismatch: {path}")
    return arrays


def file_hash_inventory(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): file_sha256(path) for path in sorted(set(paths))}


def regime_evidence() -> list[dict[str, Any]]:
    """Return the prespecified mapping with the stored numerical evidence."""
    rows = {
        1.67: (0.00019531963917531855, "not_converged", "periodic bursting", "regular"),
        3.20: (0.011420290964722529, "converged positive", "chaotic bursting", "chaotic"),
        3.29: (0.0006856194984316945, "converged near zero", "uncertain", "regular"),
        3.34: (0.01224151011021557, "converged positive", "chaotic bursting", "chaotic"),
        3.50: (-0.0007120309203858866, "not_converged non-positive", "periodic spiking", "regular"),
    }
    return [
        {
            "current": current,
            "largest_lyapunov_exponent": values[0],
            "lyapunov_evidence": values[1],
            "qualitative_regime": values[2],
            "experiment_classification": values[3],
            "sources": [
                "chapter2/outputs/dynamics_summary.csv",
                "chapter2/outputs/dynamics_summary.md",
                "chapter2/outputs/lyapunov_convergence.csv",
            ],
            "classification_rule": (
                "converged clearly positive LLE supports chaotic; periodic waveform "
                "evidence or converged near-zero/non-positive LLE supports the "
                "prespecified non-chaotic group; I=3.29 remains qualitatively uncertain"
            ),
        }
        for current, values in rows.items()
    ]
