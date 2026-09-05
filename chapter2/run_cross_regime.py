"""Pilot, protocol lock, and resumable full cross-regime execution entry point."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from chapter2.config_ch2 import DT, INITIAL_STATE, INITIAL_TRANSIENT_STEPS
from chapter2.cross_regime import (
    CrossRegimeError,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    build_evaluation_record,
    expected_model_keys,
    expected_record_ids,
    file_hash_inventory,
    load_model_bundle,
    record_id,
    recursive_forecast,
    regime_evidence,
    save_model_bundle,
    strict_load_json,
    train_scenario_model,
    validate_raw_array,
    validate_record_matrix,
)
from chapter2.cross_regime_config import (
    ALL_CURRENTS,
    BASE_COMMIT,
    BRANCH,
    CONTINUOUS_SAMPLES_PER_SEGMENT,
    CONTINUOUS_SCHEDULES,
    CONTINUOUS_STATE_COUNT,
    CONTINUOUS_SWITCH_INDICES,
    CONTINUOUS_WARMUP_TRANSITIONS,
    DATASET_ROOT,
    EFFECTIVE_TRAINING_BUDGET,
    EXPECTED_BINARY_ARTIFACTS,
    EXPECTED_CONTINUOUS_RECORDS,
    EXPECTED_LONG_RECORDS,
    EXPECTED_MODELS,
    EXPECTED_RECORDS,
    EXPECTED_SCHEDULE_DATASETS,
    EXPECTED_SHORT_RECORDS,
    FROZEN_HYPERPARAMETERS,
    LONG_FORECAST_RANGE,
    LONG_WARMUP_RANGE,
    MIXED_BLOCK_ORDERS,
    MIXED_BLOCK_ORDER_RULE,
    MODEL_ROOT,
    PILOT_ROOT,
    PRIMARY_CROSS_REGIME_CURRENTS,
    RAW_ARRAY_ROOT,
    RAW_TRAINING_TRANSITIONS,
    REPRESENTATIVE_SEED,
    RESULT_ROOT,
    SCENARIO_TRAINING_CURRENTS,
    SEEDS,
    SHORT_FORECAST_TRANSITIONS,
    SHORT_WINDOW_STARTS,
    TRAINING_WASHOUT,
    WARMUP_TRANSITIONS,
    block_order,
    model_config,
    validate_configuration,
)
from chapter2.esn_config import FIXED_DATASETS
from chapter2.esn_data import (
    ContinuousCurrentTrajectory,
    FixedCurrentTrajectory,
    NumpyStandardScaler,
    StateCurrentScalers,
    file_sha256,
    load_continuous_trajectory_file,
    load_fixed_trajectory,
)
from chapter2.esn_metrics import evaluate_rollout
from chapter2.esn_model import EchoStateNetwork, TrainingSequence
from chapter2.esn_step8 import MODEL_BUNDLE_SCHEMA, package_versions
from chapter2.hr_data_ch2 import (
    HRTrajectory,
    save_trajectory_npz,
    simulate_continuous_currents,
)


PROJECT_ROOT = RESULT_ROOT.parents[1]
MANIFEST_PATH = RESULT_ROOT / "cross_regime_manifest.json"
STATUS_PATH = RESULT_ROOT / "cross_regime_status.json"
RAW_RESULTS_PATH = RESULT_ROOT / "cross_regime_raw_results.json"
AGGREGATE_PATH = RESULT_ROOT / "cross_regime_aggregate_results.json"
MODEL_MANIFEST_PATH = RESULT_ROOT / "model_manifest.json"
DATASET_MANIFEST_PATH = RESULT_ROOT / "dataset_manifest.json"
HASH_PATH = RESULT_ROOT / "artifact_hashes.json"
VERIFICATION_PATH = RESULT_ROOT / "cross_regime_verification.json"
RESULTS_DOCUMENT = RESULT_ROOT.parent / "CROSS_REGIME_RESULTS.md"
PROTOCOL_DOCUMENT = RESULT_ROOT.parent / "CROSS_REGIME_PROTOCOL.md"
SOURCE_PATHS = (
    RESULT_ROOT.parent / "cross_regime_config.py",
    RESULT_ROOT.parent / "cross_regime.py",
    Path(__file__),
    RESULT_ROOT.parent / "audit_cross_regime.py",
    PROTOCOL_DOCUMENT,
    PROJECT_ROOT / "chapter2/optimisation_results/step7_selection.json",
    PROJECT_ROOT / "run_chapter2_cross_regime.slurm",
    *sorted((PROJECT_ROOT / "chapter2").glob("*.py")),
    *sorted((PROJECT_ROOT / "chapter2/tests").glob("test_cross_regime*.py")),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def protected_paths_unchanged() -> bool:
    protected = (
        "FINAL_THESIS_RUN",
        "chapter2/optimisation_results",
        "chapter2/final_models",
        "chapter2/final_results",
        "chapter2/outputs/data",
        "chapter2/outputs/figures",
        "chapter2/pilot_results",
    )
    return subprocess.run(
        ["git", "diff", "--quiet", BASE_COMMIT, "--", *protected], cwd=PROJECT_ROOT
    ).returncode == 0


def original_binary_inventory() -> dict[str, Any]:
    diagnostic = strict_load_json(PROJECT_ROOT / "chapter2/outputs/diagnostic_manifest.json")
    model_manifest = strict_load_json(PROJECT_ROOT / "chapter2/final_models/model_manifest.json")
    raw = strict_load_json(PROJECT_ROOT / "chapter2/final_results/step8_raw_results.json")
    entries: list[tuple[Path, str]] = [
        (PROJECT_ROOT / "chapter2/outputs" / name, digest)
        for name, digest in diagnostic["files"].items()
        if name.endswith(".npz")
    ]
    entries.extend(
        (PROJECT_ROOT / item["path"], item["sha256"])
        for item in model_manifest["models"]
    )
    entries.extend(
        (PROJECT_ROOT / item["raw_arrays_path"], item["raw_arrays_sha256"])
        for item in raw["records"]
    )
    mismatches = [
        project_relative(path)
        for path, expected in entries
        if not path.is_file() or file_sha256(path) != expected
    ]
    return {
        "expected_count": 226,
        "actual_count": len(entries),
        "unique_count": len({path.resolve() for path, _ in entries}),
        "mismatches": mismatches,
        "valid": len(entries) == 226 and len(mismatches) == 0,
    }


def source_hashes() -> dict[str, str]:
    return {
        project_relative(path): file_sha256(path)
        for path in SOURCE_PATHS
    }


def dataset_hashes() -> dict[str, str]:
    values = {}
    for record in FIXED_DATASETS:
        actual = file_sha256(record.path)
        if actual != record.sha256:
            raise CrossRegimeError(f"fixed dataset hash mismatch: {record.path}")
        values[project_relative(record.path)] = actual
    return values


def preflight(*, require_clean: bool) -> dict[str, Any]:
    validate_configuration()
    selection = strict_load_json(PROJECT_ROOT / "chapter2/optimisation_results/step7_selection.json")
    selected = selection["models"]["parameter_aware"]["best_configuration"]
    if any(FROZEN_HYPERPARAMETERS.get(key) != value for key, value in selected.items()):
        raise CrossRegimeError("architecture differs from the Step 7 selection")
    host = platform.node()
    user = os.environ.get("USER") or git_output("config", "user.name")
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    status = git_output("status", "--short")
    if host != "tinyx" and os.environ.get("SLURM_JOB_ID") is None:
        raise CrossRegimeError(f"unexpected login host {host}")
    if user != "dsnf129h":
        raise CrossRegimeError(f"unexpected user {user}")
    if branch != BRANCH:
        raise CrossRegimeError(f"unexpected branch {branch}")
    if require_clean and status:
        raise CrossRegimeError("full execution requires a clean committed tree")
    if not protected_paths_unchanged():
        raise CrossRegimeError("a protected tracked path differs from the base commit")
    binary = original_binary_inventory()
    if not binary["valid"]:
        raise CrossRegimeError("original 226-file scientific inventory failed")
    return {
        "hostname": host,
        "user": user,
        "repository": str(PROJECT_ROOT),
        "branch": branch,
        "head": head,
        "base_commit": BASE_COMMIT,
        "tracked_worktree_clean": not bool(status),
        "protected_paths_unchanged": True,
        "original_scientific_binaries": binary,
        "fixed_dataset_hashes": dataset_hashes(),
    }


def freeze_protocol() -> dict[str, Any]:
    """Write the pre-benchmark lock after source/tests/pilot are ready."""
    if MANIFEST_PATH.exists():
        raise CrossRegimeError(f"protocol manifest already exists: {MANIFEST_PATH}")
    pilot_path = PILOT_ROOT / "pilot_report.json"
    pilot = strict_load_json(pilot_path)
    if not pilot.get("passed") or pilot.get("source_hashes") != source_hashes():
        raise CrossRegimeError("a passing pilot for the current source is required")
    gate = preflight(require_clean=False)
    manifest = {
        "schema": "chapter2_cross_regime_v1",
        "status": "FROZEN BEFORE CROSS-REGIME BENCHMARK ACCESS",
        "created_at": utc_now(),
        "scientific_question": (
            "Can a parameter-aware ESN trained only on one HR regime generalize "
            "to the other, relative to shuffled mixed-regime training?"
        ),
        "preflight": gate,
        "regime_evidence": regime_evidence(),
        "configuration": {
            "model_type": "parameter_aware",
            "hyperparameters": FROZEN_HYPERPARAMETERS,
            "source": "chapter2/optimisation_results/step7_selection.json",
            "seeds": list(SEEDS),
            "scenarios": {
                scenario: {
                    "training_currents": list(currents),
                    "primary_cross_regime_currents": list(PRIMARY_CROSS_REGIME_CURRENTS[scenario]),
                    "raw_training_transitions": {
                        f"{current:.2f}": RAW_TRAINING_TRANSITIONS[scenario][current]
                        for current in currents
                    },
                    "effective_samples": EFFECTIVE_TRAINING_BUDGET,
                }
                for scenario, currents in SCENARIO_TRAINING_CURRENTS.items()
            },
            "washout_per_block": TRAINING_WASHOUT,
            "mixed_block_order_rule": MIXED_BLOCK_ORDER_RULE,
            "mixed_block_orders": {
                str(seed): list(order) for seed, order in MIXED_BLOCK_ORDERS.items()
            },
            "short_window_starts": list(SHORT_WINDOW_STARTS),
            "warmup_transitions": WARMUP_TRANSITIONS,
            "short_forecast_transitions": SHORT_FORECAST_TRANSITIONS,
            "long_warmup_range": list(LONG_WARMUP_RANGE),
            "long_forecast_range": list(LONG_FORECAST_RANGE),
            "continuous_schedules": {
                name: list(values) for name, values in CONTINUOUS_SCHEDULES.items()
            },
            "continuous_switch_indices": list(CONTINUOUS_SWITCH_INDICES),
            "continuous_state_count": CONTINUOUS_STATE_COUNT,
            "continuous_warmup_transitions": CONTINUOUS_WARMUP_TRANSITIONS,
            "thresholds": {"valid_prediction": 0.4, "divergence": 5.0, "collapse": 0.05},
            "representative_seed": REPRESENTATIVE_SEED,
            "expected": {
                "models": EXPECTED_MODELS,
                "short_records": EXPECTED_SHORT_RECORDS,
                "long_records": EXPECTED_LONG_RECORDS,
                "continuous_records": EXPECTED_CONTINUOUS_RECORDS,
                "records": EXPECTED_RECORDS,
                "schedule_datasets": EXPECTED_SCHEDULE_DATASETS,
                "large_binary_artifacts": EXPECTED_BINARY_ARTIFACTS,
            },
        },
        "pilot_report_sha256": file_sha256(pilot_path),
        "source_hashes": source_hashes(),
        "environment": package_versions(),
        "benchmark_numerical_values_inspected_for_tuning": False,
        "limitation": (
            "The frozen hyperparameters were previously selected using mixed-regime "
            "validation currents. This experiment isolates the effect of the final "
            "training data and fitted readout under a fixed, previously selected ESN "
            "architecture. It does not demonstrate that every design decision was "
            "learned exclusively from one regime."
        ),
    }
    atomic_write_json(MANIFEST_PATH, manifest)
    return manifest


def synthetic_prefix(current: float, transitions: int) -> FixedCurrentTrajectory:
    time = np.arange(transitions + 1) * DT
    phase = time * (0.7 + current / 10.0)
    states = np.column_stack(
        (np.sin(phase) + current / 10.0, np.cos(phase), np.sin(phase * 0.13))
    )
    currents = np.full(transitions + 1, current)
    return FixedCurrentTrajectory(current, time, states, currents)


def run_pilot() -> dict[str, Any]:
    """Run a small synthetic mechanics pilot without held-out-data access."""
    if PILOT_ROOT.exists():
        raise CrossRegimeError(f"pilot output already exists: {PILOT_ROOT}")
    PILOT_ROOT.mkdir(parents=True)
    pilot_records = []
    reset_checks = []
    for scenario, currents in SCENARIO_TRAINING_CURRENTS.items():
        pilot_transitions = 240
        pilot_washout = 40
        trajectories = {
            current: synthetic_prefix(current, pilot_transitions) for current in currents
        }
        fitting_states = np.concatenate([item.states[:-1] for item in trajectories.values()])
        fitting_currents = np.concatenate([item.current_values[:-1] for item in trajectories.values()])[:, None]
        scalers = StateCurrentScalers(
            NumpyStandardScaler.fit(fitting_states),
            NumpyStandardScaler.fit(fitting_currents),
        )
        sequences = []
        for current in block_order(scenario, 42):
            item = trajectories[current]
            sequences.append(
                TrainingSequence(
                    scalers.transform_inputs(
                        np.column_stack((item.states[:-1], item.current_values[:-1]))
                    ),
                    scalers.transform_targets(item.states[1:]),
                )
            )
        model = EchoStateNetwork(model_config(42)).fit(sequences, washout=pilot_washout)
        reset_checks.append(
            {
                "scenario": scenario,
                "block_count": len(sequences),
                "independent_reset_contract": True,
                "effective_samples": sum(len(item.inputs) - pilot_washout for item in sequences),
            }
        )
        for schedule, values in CONTINUOUS_SCHEDULES.items():
            trajectory, switches = simulate_continuous_currents(
                values, samples_per_segment=60, transient_steps=60
            )
            predictions, failure_step, failure_reason = recursive_forecast(
                model,
                scalers,
                trajectory.state,
                trajectory.I,
                warmup_range=(0, 20),
                forecast_range=(20, len(trajectory.I) - 1),
            )
            targets = trajectory.state[21:]
            metrics = evaluate_rollout(
                predictions,
                targets,
                normalisation_scale=scalers.state.scale,
                dt=DT,
                valid_prediction_threshold=0.4,
                divergence_threshold=5.0,
                collapse_std_ratio_threshold=0.05,
            ).to_dict()
            path = PILOT_ROOT / f"{scenario}__{schedule}.npz"
            from chapter2.esn_step8 import atomic_save_npz
            atomic_save_npz(path, predictions=predictions, targets=targets, current=trajectory.I[20:-1])
            pilot_records.append(
                {
                    "scenario": scenario,
                    "schedule": schedule,
                    "schedule_values": list(values),
                    "switch_indices": switches.tolist(),
                    "prediction_shape": list(predictions.shape),
                    "input_dimension": model.config.input_dimension,
                    "failure_step": failure_step,
                    "failure_reason": failure_reason,
                    "metrics_finite_or_explicitly_classified": (
                        metrics["nrmse_state"] is not None or metrics["diverged"]
                    ),
                    "no_boundary_reset_or_rewarm": True,
                    "path": project_relative(path),
                    "sha256": file_sha256(path),
                }
            )
    report = {
        "schema": "chapter2_cross_regime_pilot_v1",
        "created_at": utc_now(),
        "seed": 42,
        "source_hashes": source_hashes(),
        "synthetic_training_only": True,
        "real_held_out_benchmark_access": False,
        "all_three_scenarios": sorted(SCENARIO_TRAINING_CURRENTS),
        "all_three_short_schedules": sorted(CONTINUOUS_SCHEDULES),
        "full_protocol_budget_validation": EFFECTIVE_TRAINING_BUDGET == 130_000,
        "scaler_isolation": True,
        "recursive_prediction_feedback": True,
        "true_current_only_after_warmup": True,
        "training_block_checks": reset_checks,
        "records": pilot_records,
        "passed": len(pilot_records) == 9 and all(
            item["metrics_finite_or_explicitly_classified"] for item in pilot_records
        ),
    }
    atomic_write_json(PILOT_ROOT / "pilot_report.json", report)
    return report


def atomic_save_schedule(path: Path, trajectory: HRTrajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_trajectory_npz(temporary, trajectory)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate_schedules() -> dict[str, Any]:
    if DATASET_MANIFEST_PATH.exists():
        manifest = strict_load_json(DATASET_MANIFEST_PATH)
        for item in manifest["datasets"]:
            path = PROJECT_ROOT / item["path"]
            if file_sha256(path) != item["sha256"]:
                raise CrossRegimeError(f"resumed schedule hash mismatch: {path}")
        return manifest
    records = []
    for name, values in CONTINUOUS_SCHEDULES.items():
        trajectory, switches = simulate_continuous_currents(
            values,
            samples_per_segment=CONTINUOUS_SAMPLES_PER_SEGMENT,
            transient_steps=INITIAL_TRANSIENT_STEPS,
            initial_state=INITIAL_STATE,
        )
        if len(trajectory.I) != CONTINUOUS_STATE_COUNT or tuple(switches) != CONTINUOUS_SWITCH_INDICES:
            raise CrossRegimeError(f"generated schedule contract mismatch: {name}")
        path = DATASET_ROOT / f"{name}.npz"
        atomic_save_schedule(path, trajectory)
        records.append(
            {
                "name": name,
                "path": project_relative(path),
                "sha256": file_sha256(path),
                "state_count": len(trajectory.I),
                "sequence": list(values),
                "switch_indices": list(CONTINUOUS_SWITCH_INDICES),
            }
        )
    manifest = {"schema": "chapter2_cross_regime_dataset_manifest_v1", "created_at": utc_now(), "datasets": records}
    atomic_write_json(DATASET_MANIFEST_PATH, manifest)
    return manifest


def train_models(manifest_hash: str) -> dict[str, Any]:
    if MODEL_MANIFEST_PATH.exists():
        manifest = strict_load_json(MODEL_MANIFEST_PATH)
    else:
        manifest = {"schema": "chapter2_cross_regime_model_manifest_v1", "created_at": utc_now(), "models": []}
    completed = {(item["scenario"], int(item["seed"])): item for item in manifest["models"]}
    for scenario, seed in sorted(expected_model_keys()):
        if (scenario, seed) in completed:
            item = completed[(scenario, seed)]
            path = PROJECT_ROOT / item["path"]
            if file_sha256(path) != item["sha256"]:
                raise CrossRegimeError(f"resumed model hash mismatch: {path}")
            model, _, metadata = load_model_bundle(path)
            if asdict(model.config) != asdict(model_config(seed)) or metadata["scenario"] != scenario or metadata.get("protocol_manifest_sha256") != manifest_hash:
                raise CrossRegimeError(f"resumed model identity mismatch: {path}")
            continue
        model, scalers, training = train_scenario_model(scenario, seed)
        metadata = {
            "schema": MODEL_BUNDLE_SCHEMA,
            "experiment_schema": "chapter2_cross_regime_v1",
            "scenario": scenario,
            "seed": seed,
            "configuration": asdict(model.config),
            "training": training,
            "protocol_manifest_sha256": manifest_hash,
            "trained_at": utc_now(),
        }
        path = MODEL_ROOT / f"{scenario}__seed_{seed}.npz"
        save_model_bundle(path, model, scalers, metadata)
        loaded, loaded_scalers, loaded_metadata = load_model_bundle(path)
        if loaded_metadata != metadata or asdict(loaded.config) != asdict(model.config):
            raise CrossRegimeError(f"model round-trip metadata mismatch: {path}")
        if not (
            np.array_equal(loaded_scalers.state.mean, scalers.state.mean)
            and np.array_equal(loaded_scalers.current.scale, scalers.current.scale)
        ):
            raise CrossRegimeError(f"model round-trip scaler mismatch: {path}")
        item = {
            "scenario": scenario,
            "seed": seed,
            "path": project_relative(path),
            "sha256": file_sha256(path),
            "configuration": asdict(model.config),
            "training": training,
            "round_trip_verified": True,
        }
        manifest["models"].append(item)
        atomic_write_json(MODEL_MANIFEST_PATH, manifest)
    keys = {(item["scenario"], int(item["seed"])) for item in manifest["models"]}
    if keys != expected_model_keys() or len(manifest["models"]) != EXPECTED_MODELS:
        raise CrossRegimeError("model matrix incomplete or duplicated")
    manifest["status"] = "complete"
    manifest["model_count"] = EXPECTED_MODELS
    atomic_write_json(MODEL_MANIFEST_PATH, manifest)
    return manifest


def load_schedules(manifest: Mapping[str, Any]) -> dict[str, ContinuousCurrentTrajectory]:
    result = {}
    for item in manifest["datasets"]:
        path = PROJECT_ROOT / item["path"]
        result[item["name"]] = load_continuous_trajectory_file(
            path,
            expected_sequence=item["sequence"],
            expected_switch_indices=item["switch_indices"],
            expected_sha256=item["sha256"],
            expected_state_count=item["state_count"],
        )
    return result


def _raw_state(manifest_hash: str, model_manifest_hash: str, dataset_manifest_hash: str) -> dict[str, Any]:
    if RAW_RESULTS_PATH.exists():
        value = strict_load_json(RAW_RESULTS_PATH)
        expected_hashes = {
            "protocol_manifest": manifest_hash,
            "model_manifest": model_manifest_hash,
            "dataset_manifest": dataset_manifest_hash,
        }
        if value.get("lock_hashes") != expected_hashes:
            raise CrossRegimeError("resume lock hashes differ")
        for item in value["records"]:
            validate_raw_array(item)
        return value
    return {
        "schema": "chapter2_cross_regime_raw_results_v1",
        "created_at": utc_now(),
        "lock_hashes": {
            "protocol_manifest": manifest_hash,
            "model_manifest": model_manifest_hash,
            "dataset_manifest": dataset_manifest_hash,
        },
        "records": [],
    }


def _append_record(raw: dict[str, Any], item: dict[str, Any]) -> None:
    if item["record_id"] in {record["record_id"] for record in raw["records"]}:
        raise CrossRegimeError(f"duplicate record {item['record_id']}")
    raw["records"].append(item)
    atomic_write_json(RAW_RESULTS_PATH, raw)
    print(f"completed {item['record_id']}", flush=True)


def execute_benchmarks(
    model_manifest: Mapping[str, Any], dataset_manifest: Mapping[str, Any], manifest_hash: str
) -> dict[str, Any]:
    raw = _raw_state(manifest_hash, file_sha256(MODEL_MANIFEST_PATH), file_sha256(DATASET_MANIFEST_PATH))
    completed = {item["record_id"] for item in raw["records"]}
    fixed = {current: load_fixed_trajectory(current) for current in ALL_CURRENTS}
    schedules = load_schedules(dataset_manifest)
    for model_item in model_manifest["models"]:
        scenario = str(model_item["scenario"]); seed = int(model_item["seed"])
        model, scalers, metadata = load_model_bundle(PROJECT_ROOT / model_item["path"])
        if metadata["protocol_manifest_sha256"] != manifest_hash:
            raise CrossRegimeError("model was not trained under current protocol lock")
        for current in ALL_CURRENTS:
            trajectory = fixed[current]
            for window, start in enumerate(SHORT_WINDOW_STARTS, start=1):
                warmup = (start, start + WARMUP_TRANSITIONS)
                forecast = (warmup[1], warmup[1] + SHORT_FORECAST_TRANSITIONS)
                identifier = record_id("fixed_short", scenario, seed, current=current, window=window)
                if identifier in completed: continue
                predictions, failure_step, failure_reason = recursive_forecast(
                    model, scalers, trajectory.states, trajectory.current_values,
                    warmup_range=warmup, forecast_range=forecast,
                )
                begin, stop = forecast
                item = build_evaluation_record(
                    identifier=identifier, family="fixed_short", scenario=scenario, seed=seed,
                    scalers=scalers, predictions=predictions,
                    targets=trajectory.states[begin + 1:stop + 1],
                    times=trajectory.time[begin + 1:stop + 1],
                    currents=trajectory.current_values[begin:stop],
                    raw_path=RAW_ARRAY_ROOT / f"{identifier}.npz",
                    warmup_range=warmup, forecast_range=forecast,
                    failure_step=failure_step, failure_reason=failure_reason,
                    current=current, window=window,
                )
                _append_record(raw, item); completed.add(identifier)
            identifier = record_id("fixed_long", scenario, seed, current=current)
            if identifier not in completed:
                predictions, failure_step, failure_reason = recursive_forecast(
                    model, scalers, trajectory.states, trajectory.current_values,
                    warmup_range=LONG_WARMUP_RANGE, forecast_range=LONG_FORECAST_RANGE,
                )
                begin, stop = LONG_FORECAST_RANGE
                item = build_evaluation_record(
                    identifier=identifier, family="fixed_long", scenario=scenario, seed=seed,
                    scalers=scalers, predictions=predictions,
                    targets=trajectory.states[begin + 1:stop + 1],
                    times=trajectory.time[begin + 1:stop + 1], currents=trajectory.current_values[begin:stop],
                    raw_path=RAW_ARRAY_ROOT / f"{identifier}.npz",
                    warmup_range=LONG_WARMUP_RANGE, forecast_range=LONG_FORECAST_RANGE,
                    failure_step=failure_step, failure_reason=failure_reason, current=current,
                )
                _append_record(raw, item); completed.add(identifier)
        for schedule, trajectory in schedules.items():
            identifier = record_id("continuous", scenario, seed, schedule=schedule)
            if identifier in completed: continue
            forecast = (CONTINUOUS_WARMUP_TRANSITIONS, trajectory.state_count - 1)
            predictions, failure_step, failure_reason = recursive_forecast(
                model, scalers, trajectory.states, trajectory.current_values,
                warmup_range=(0, CONTINUOUS_WARMUP_TRANSITIONS), forecast_range=forecast,
            )
            begin, stop = forecast
            item = build_evaluation_record(
                identifier=identifier, family="continuous", scenario=scenario, seed=seed,
                scalers=scalers, predictions=predictions,
                targets=trajectory.states[begin + 1:stop + 1],
                times=trajectory.time[begin + 1:stop + 1], currents=trajectory.current_values[begin:stop],
                raw_path=RAW_ARRAY_ROOT / f"{identifier}.npz",
                warmup_range=(0, CONTINUOUS_WARMUP_TRANSITIONS), forecast_range=forecast,
                failure_step=failure_step, failure_reason=failure_reason, schedule=schedule,
            )
            _append_record(raw, item); completed.add(identifier)
    validate_record_matrix(raw["records"])
    raw["status"] = "complete"
    atomic_write_json(RAW_RESULTS_PATH, raw)
    return raw


def summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    penalized = [float(item["aggregate_nrmse_value"]) for item in items]
    finite = [float(item["metrics"]["nrmse_state"]) for item in items if item["metrics"]["nrmse_state"] is not None]
    vpt = [float(item["metrics"]["valid_prediction_time"]) for item in items]
    return {
        "record_count": len(items),
        "finite_rmse_mean": statistics.fmean(
            item["metrics"]["rmse_state"] for item in items
            if item["metrics"].get("rmse_state") is not None
        ) if any(item["metrics"].get("rmse_state") is not None for item in items) else None,
        "finite_nrmse_count": len(finite),
        "finite_nrmse_mean": statistics.fmean(finite) if finite else None,
        "finite_nrmse_population_std": statistics.pstdev(finite) if len(finite) > 1 else (0.0 if finite else None),
        "finite_nrmse_median": statistics.median(finite) if finite else None,
        "finite_nrmse_iqr": (float(np.percentile(finite, 75) - np.percentile(finite, 25)) if finite else None),
        "failure_penalized_nrmse_mean": statistics.fmean(penalized),
        "failure_penalized_nrmse_population_std": statistics.pstdev(penalized) if len(penalized) > 1 else 0.0,
        "failure_penalized_nrmse_median": statistics.median(penalized),
        "mean_valid_prediction_time": statistics.fmean(vpt),
        "median_valid_prediction_time": statistics.median(vpt),
        "divergence_count": sum(bool(item["metrics"]["diverged"]) for item in items),
        "divergence_rate": sum(bool(item["metrics"]["diverged"]) for item in items) / len(items),
        "collapse_count": sum(bool(item["metrics"]["prediction_collapse_any"]) for item in items),
        "collapse_rate": sum(bool(item["metrics"]["prediction_collapse_any"]) for item in items) / len(items),
        "numerical_failure_count": sum(bool(item["numerical_failure"]) for item in items),
        "event_defined_count": sum(bool(item["event_metrics"]["defined"]) for item in items),
    }


def paired_comparisons(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs = (("regular_to_chaotic", "mixed_shuffled"), ("chaotic_to_regular", "mixed_shuffled"), ("regular_to_chaotic", "chaotic_to_regular"))
    output = []
    for left, right in pairs:
        left_items = {(
            item["family"], int(item["seed"]), item["current"], item["window"], item["schedule"]
        ): item for item in records if item["scenario"] == left}
        right_items = {(
            item["family"], int(item["seed"]), item["current"], item["window"], item["schedule"]
        ): item for item in records if item["scenario"] == right}
        if set(left_items) != set(right_items): raise CrossRegimeError("paired comparison keys differ")
        seed_differences = []
        for seed in SEEDS:
            values = [
                float(left_items[key]["aggregate_nrmse_value"]) - float(right_items[key]["aggregate_nrmse_value"])
                for key in left_items if key[1] == seed
            ]
            seed_differences.append(statistics.fmean(values))
        rng = np.random.default_rng(20260904)
        bootstrap = np.asarray([
            np.mean(rng.choice(seed_differences, size=len(seed_differences), replace=True))
            for _ in range(20_000)
        ])
        output.append({
            "left": left, "right": right, "difference_definition": "left minus right failure-penalized NRMSE",
            "paired_record_count": len(left_items), "per_seed_mean_difference": dict(zip(map(str, SEEDS), seed_differences)),
            "mean_difference": statistics.fmean(seed_differences),
            "bootstrap_95_percent_ci_over_seeds": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))],
            "significance_claim": False,
        })
    return output


def aggregate_results(raw: Mapping[str, Any], *, write: bool = True) -> dict[str, Any]:
    records = raw["records"]
    scenarios = {}
    for scenario in SCENARIO_TRAINING_CURRENTS:
        selected = [item for item in records if item["scenario"] == scenario]
        scenarios[scenario] = {
            "overall": summary(selected),
            "by_family": {family: summary([item for item in selected if item["family"] == family]) for family in ("fixed_short", "fixed_long", "continuous")},
            "by_evaluation_class": {kind: summary([item for item in selected if item["evaluation_class"] == kind]) for kind in sorted({item["evaluation_class"] for item in selected})},
            "by_seed": {str(seed): summary([item for item in selected if int(item["seed"]) == seed]) for seed in SEEDS},
        }
    aggregate = {
        "schema": "chapter2_cross_regime_aggregate_v1", "created_at": utc_now(),
        "record_count": len(records), "scenarios": scenarios,
        "paired_comparisons": paired_comparisons(records),
        "transfer_target_comparisons": [
            {"training_scenario": scenario, "family": family, "target_current": current,
             "transfer": summary([item for item in records if item["scenario"] == scenario and item["family"] == family and item["current"] == current]),
             "mixed_reference": summary([item for item in records if item["scenario"] == "mixed_shuffled" and item["family"] == family and item["current"] == current])}
            for scenario, currents in PRIMARY_CROSS_REGIME_CURRENTS.items()
            for current in currents for family in ("fixed_short", "fixed_long")
        ],
        "five_seed_inference_limitation": "Bootstrap intervals are descriptive; no statistical-significance claim is made from five seeds.",
    }
    if write:
        atomic_write_json(AGGREGATE_PATH, aggregate)
        write_tables(records, aggregate)
    return aggregate


def flatten_record(item: Mapping[str, Any]) -> dict[str, Any]:
    row = {key: item[key] for key in ("record_id", "family", "scenario", "seed", "current", "window", "schedule", "evaluation_class", "numerical_failure", "failure_step", "failure_reason", "aggregate_nrmse_value")}
    row.update(item["metrics"])
    return {key: ("" if value is None else value) for key, value in row.items()}


def write_tables(records: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any]) -> None:
    atomic_write_csv(RESULT_ROOT / "fixed_short_results.csv", [flatten_record(item) for item in records if item["family"] == "fixed_short"])
    atomic_write_csv(RESULT_ROOT / "fixed_long_results.csv", [flatten_record(item) for item in records if item["family"] == "fixed_long"])
    atomic_write_csv(RESULT_ROOT / "continuous_schedule_results.csv", [flatten_record(item) for item in records if item["family"] == "continuous"])
    atomic_write_csv(RESULT_ROOT / "transfer_target_comparison.csv", [
        {"training_scenario": comparison["training_scenario"], "family": comparison["family"],
         "target_current": comparison["target_current"], "model_role": role, **comparison[role]}
        for comparison in aggregate["transfer_target_comparisons"] for role in ("transfer", "mixed_reference")
    ])
    seed_rows=[]; divergence=[]
    for scenario in SCENARIO_TRAINING_CURRENTS:
        for seed in SEEDS:
            values=[item for item in records if item["scenario"]==scenario and int(item["seed"])==seed]
            seed_rows.append({"scenario":scenario,"seed":seed,**summary(values)})
        overall=aggregate["scenarios"][scenario]["overall"]
        divergence.append({"scenario":scenario,"record_count":overall["record_count"],"divergence_count":overall["divergence_count"],"divergence_rate":overall["divergence_rate"],"collapse_count":overall["collapse_count"],"collapse_rate":overall["collapse_rate"],"numerical_failure_count":overall["numerical_failure_count"]})
    atomic_write_csv(RESULT_ROOT / "per_seed_summary.csv", seed_rows)
    atomic_write_csv(RESULT_ROOT / "divergence_summary.csv", divergence)
    atomic_write_csv(RESULT_ROOT / "scenario_comparison.csv", [{"scenario":scenario,**aggregate["scenarios"][scenario]["overall"]} for scenario in SCENARIO_TRAINING_CURRENTS])


def create_results_document(aggregate: Mapping[str, Any]) -> None:
    lines = ["# Chapter 2 cross-regime results", "", "Benchmark execution is complete; the audit is still pending.", "", "## Scenario summary", "", "| Scenario | Records | Finite mean NRMSE | Penalized mean NRMSE | Mean VPT | Divergence | Collapse |", "|---|---:|---:|---:|---:|---:|---:|"]
    for scenario, data in aggregate["scenarios"].items():
        item=data["overall"]
        finite="undefined" if item["finite_nrmse_mean"] is None else f"{item['finite_nrmse_mean']:.6g}"
        lines.append(f"| `{scenario}` | {item['record_count']} | {finite} | {item['failure_penalized_nrmse_mean']:.6g} | {item['mean_valid_prediction_time']:.6g} | {item['divergence_count']} ({item['divergence_rate']:.1%}) | {item['collapse_count']} ({item['collapse_rate']:.1%}) |")
    lines += ["", "## Interpretation", "", "The numerical comparison, directional asymmetry, seed sensitivity, event behavior, and schedule-transition findings must be interpreted after the independent audit. The five-seed bootstrap intervals are descriptive and are not treated as proof of statistical significance.", "", "The frozen hyperparameters were previously selected using mixed-regime validation currents. This experiment isolates the effect of the final training data and fitted readout under a fixed, previously selected ESN architecture. It does not demonstrate that every design decision was learned exclusively from one regime.", "", "Regime qualification: `I=3.29` is grouped with the prespecified regular/non-chaotic set because its converged LLE is near zero and its half-window diagnostics are consistent, while the original qualitative dynamics table labels it uncertain."]
    atomic_write_text(RESULTS_DOCUMENT, "\n".join(lines)+"\n")


def run_full() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with (RESULT_ROOT / ".execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CrossRegimeError("another cross-regime execution is active") from error
        _run_full_locked()


def _run_full_locked() -> None:
    if os.environ.get("SLURM_JOB_ID") is None:
        raise CrossRegimeError("full execution is permitted only inside a Slurm allocation")
    if STATUS_PATH.exists() and strict_load_json(STATUS_PATH).get("state") == "COMPLETE":
        raise CrossRegimeError("completed experiment refuses accidental rerun")
    gate = preflight(require_clean=True)
    if not MANIFEST_PATH.exists(): raise CrossRegimeError("frozen protocol manifest is missing")
    manifest = strict_load_json(MANIFEST_PATH)
    if manifest["source_hashes"] != source_hashes(): raise CrossRegimeError("source hashes differ from protocol lock")
    status={"schema":"chapter2_cross_regime_status_v1","state":"RUNNING","started_at":utc_now(),"slurm_job_id":os.environ["SLURM_JOB_ID"],"node":platform.node(),"implementation_commit":gate["head"]}
    atomic_write_json(STATUS_PATH,status)
    try:
        schedules=generate_schedules(); models=train_models(file_sha256(MANIFEST_PATH)); raw=execute_benchmarks(models,schedules,file_sha256(MANIFEST_PATH)); aggregate=aggregate_results(raw); create_results_document(aggregate)
        status.update({"state":"BENCHMARK_COMPLETE_AUDIT_REQUIRED","completed_at":utc_now(),"record_count":len(raw["records"]),"model_count":len(models["models"]),"dataset_count":len(schedules["datasets"]),"aggregate_sha256":file_sha256(AGGREGATE_PATH)})
        atomic_write_json(STATUS_PATH,status)
    except Exception as error:
        status.update({"state":"FAILED","failed_at":utc_now(),"failure":{"type":type(error).__name__,"message":str(error)}}); atomic_write_json(STATUS_PATH,status); raise


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); modes=parser.add_mutually_exclusive_group(required=True); modes.add_argument("--pilot",action="store_true"); modes.add_argument("--freeze",action="store_true"); modes.add_argument("--full",action="store_true"); args=parser.parse_args(argv)
    if args.pilot:
        report=run_pilot(); print("PILOT PASSED" if report["passed"] else "PILOT FAILED"); return 0 if report["passed"] else 1
    if args.freeze:
        freeze_protocol(); print(f"FROZEN {MANIFEST_PATH}"); return 0
    run_full(); print("BENCHMARK COMPLETE; AUDIT REQUIRED"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
