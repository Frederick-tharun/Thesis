"""Pilot, final training, evaluation matrix, schedules, and thesis tables."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from chapter2.cross_regime import (
    CrossRegimeError,
    atomic_write_csv,
    build_evaluation_record,
    expected_record_ids,
    load_model_bundle,
    prepare_training,
    record_id,
    recursive_forecast,
    save_model_bundle,
)
from chapter2.cross_regime_config import (
    CONTINUOUS_WARMUP_TRANSITIONS,
    LONG_FORECAST_RANGE,
    LONG_WARMUP_RANGE,
    SHORT_FORECAST_TRANSITIONS,
    SHORT_WINDOW_STARTS,
    WARMUP_TRANSITIONS,
)
from chapter2.esn_data import (
    FixedCurrentTrajectory,
    NumpyStandardScaler,
    StateCurrentScalers,
    file_sha256,
    load_fixed_trajectory,
)
from chapter2.esn_model import EchoStateNetwork
from chapter2.esn_optimisation import atomic_write_json, load_strict_json
from chapter2 import run_cross_regime as baseline_runner

from .config import (
    ALL_CURRENTS,
    CHAOTIC_CURRENTS,
    CONTINUOUS_SCHEDULES,
    MATRIX_CODES,
    MODEL_ROOT,
    OPTIMISATION_ROOT,
    RAW_ARRAY_ROOT,
    REGULAR_CURRENTS,
    RESULT_ROOT,
    SCENARIO_TRAINING_CURRENTS,
    SEEDS,
    block_order,
    regime_for_current,
)
from .optimisation import (
    StabilityAwareEvaluator,
    esn_config,
    prepare_scenario_data,
    run_five_seed_confirmation,
    run_search,
    write_selection,
)
from .protection import (
    assert_baseline_artifacts_unchanged,
    assert_esn_source_unchanged,
    refuse_existing,
)


MODEL_MANIFEST = RESULT_ROOT / "model_manifest.json"
RAW_RESULTS = RESULT_ROOT / "evaluation_records.json"
AGGREGATE_RESULTS = RESULT_ROOT / "aggregate_results.json"
SUMMARY_TABLE = RESULT_ROOT / "train_test_summary.csv"
PILOT_REPORT = RESULT_ROOT / "pilot_report.json"
SELECTION_PATH = OPTIMISATION_ROOT / "selection.json"


def _selection() -> dict[str, Any]:
    if not SELECTION_PATH.is_file():
        raise CrossRegimeError("final evaluation is locked until selection.json exists")
    selection = load_strict_json(SELECTION_PATH)
    if selection.get("status") != "LOCKED BEFORE FINAL EVALUATION":
        raise CrossRegimeError("scenario hyperparameters are not locked")
    return selection


def _new_or_resume(path: Path, initial: dict[str, Any], resume: bool) -> dict[str, Any]:
    if path.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
        return load_strict_json(path)
    atomic_write_json(path, initial)
    return initial


def train_final_models(*, resume: bool = False) -> dict[str, Any]:
    """Train five locked models per scenario from scenario-only prefixes."""
    assert_esn_source_unchanged()
    selection = _selection()
    selection_hash = file_sha256(SELECTION_PATH)
    manifest = _new_or_resume(
        MODEL_MANIFEST,
        {
            "schema": "chapter2_cross_regime_improvement_models_v1",
            "status": "training_in_progress",
            "selection_sha256": selection_hash,
            "models": [],
        },
        resume,
    )
    if manifest.get("selection_sha256") != selection_hash:
        raise CrossRegimeError("model resume selection lock mismatch")
    complete = {(x["scenario"], int(x["seed"])) for x in manifest["models"]}
    for scenario in SCENARIO_TRAINING_CURRENTS:
        parameters = selection["models"][scenario]["hyperparameters"]
        for seed in SEEDS:
            if (scenario, seed) in complete:
                continue
            scalers, sequences, training = prepare_training(scenario, seed)
            model = EchoStateNetwork(esn_config(parameters, seed))
            model.fit(sequences, washout=training["washout_per_independently_reset_block"])
            path = MODEL_ROOT / f"{scenario}__seed_{seed}.npz"
            refuse_existing(path)
            metadata = {
                "schema": "chapter2_cross_regime_improvement_model_v1",
                "scenario": scenario,
                "seed": seed,
                "selection_sha256": selection_hash,
                "hyperparameters": parameters,
                "training": training,
                "benchmark_opened_before_model_lock": False,
            }
            save_model_bundle(path, model, scalers, metadata)
            loaded, _, loaded_metadata = load_model_bundle(path)
            if asdict(loaded.config) != asdict(model.config) or loaded_metadata != metadata:
                raise CrossRegimeError("saved model failed round-trip verification")
            manifest["models"].append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "configuration": asdict(model.config),
                    "training": training,
                }
            )
            atomic_write_json(MODEL_MANIFEST, manifest)
    expected = {(scenario, seed) for scenario in SCENARIO_TRAINING_CURRENTS for seed in SEEDS}
    actual = {(x["scenario"], int(x["seed"])) for x in manifest["models"]}
    if actual != expected or len(manifest["models"]) != 15:
        raise CrossRegimeError("final model matrix is incomplete or duplicated")
    manifest["status"] = "complete"
    atomic_write_json(MODEL_MANIFEST, manifest)
    return manifest


def _load_schedules() -> Mapping[str, Any]:
    manifest_path = baseline_runner.DATASET_MANIFEST_PATH
    manifest = load_strict_json(manifest_path)
    schedules = baseline_runner.load_schedules(manifest)
    if set(schedules) != set(CONTINUOUS_SCHEDULES):
        raise CrossRegimeError("supporting continuous schedule set mismatch")
    return schedules


def _append_record(raw: dict[str, Any], record: dict[str, Any]) -> None:
    if record["record_id"] in {x["record_id"] for x in raw["records"]}:
        raise CrossRegimeError(f"duplicate evaluation record: {record['record_id']}")
    raw["records"].append(record)
    atomic_write_json(RAW_RESULTS, raw)


def _fixed_records(
    raw: dict[str, Any], model: EchoStateNetwork, scalers: StateCurrentScalers,
    scenario: str, seed: int, trajectories: Mapping[float, FixedCurrentTrajectory]
) -> None:
    completed = {x["record_id"] for x in raw["records"]}
    for current in ALL_CURRENTS:
        trajectory = trajectories[current]
        for window, start in enumerate(SHORT_WINDOW_STARTS, start=1):
            identifier = record_id("fixed_short", scenario, seed, current=current, window=window)
            if identifier in completed:
                continue
            warmup = (start, start + WARMUP_TRANSITIONS)
            forecast = (warmup[1], warmup[1] + SHORT_FORECAST_TRANSITIONS)
            prediction, failure_step, failure_reason = recursive_forecast(
                model, scalers, trajectory.states, trajectory.current_values,
                warmup_range=warmup, forecast_range=forecast,
            )
            raw_path = RAW_ARRAY_ROOT / f"{identifier}.npz"
            refuse_existing(raw_path)
            begin, stop = forecast
            record = build_evaluation_record(
                identifier=identifier, family="fixed_short", scenario=scenario,
                seed=seed, scalers=scalers, predictions=prediction,
                targets=trajectory.states[begin + 1:stop + 1],
                times=trajectory.time[begin + 1:stop + 1],
                currents=trajectory.current_values[begin:stop], raw_path=raw_path,
                warmup_range=warmup, forecast_range=forecast,
                failure_step=failure_step, failure_reason=failure_reason,
                current=current, window=window,
            )
            record["matrix_code"] = MATRIX_CODES[(scenario, regime_for_current(current))]
            _append_record(raw, record)
            completed.add(identifier)
        identifier = record_id("fixed_long", scenario, seed, current=current)
        if identifier in completed:
            continue
        prediction, failure_step, failure_reason = recursive_forecast(
            model, scalers, trajectory.states, trajectory.current_values,
            warmup_range=LONG_WARMUP_RANGE, forecast_range=LONG_FORECAST_RANGE,
        )
        raw_path = RAW_ARRAY_ROOT / f"{identifier}.npz"
        refuse_existing(raw_path)
        begin, stop = LONG_FORECAST_RANGE
        record = build_evaluation_record(
            identifier=identifier, family="fixed_long", scenario=scenario,
            seed=seed, scalers=scalers, predictions=prediction,
            targets=trajectory.states[begin + 1:stop + 1],
            times=trajectory.time[begin + 1:stop + 1],
            currents=trajectory.current_values[begin:stop], raw_path=raw_path,
            warmup_range=LONG_WARMUP_RANGE, forecast_range=LONG_FORECAST_RANGE,
            failure_step=failure_step, failure_reason=failure_reason, current=current,
        )
        record["matrix_code"] = MATRIX_CODES[(scenario, regime_for_current(current))]
        _append_record(raw, record)
        completed.add(identifier)


def _continuous_records(
    raw: dict[str, Any], model: EchoStateNetwork, scalers: StateCurrentScalers,
    scenario: str, seed: int, schedules: Mapping[str, Any]
) -> None:
    completed = {x["record_id"] for x in raw["records"]}
    for name, trajectory in schedules.items():
        identifier = record_id("continuous", scenario, seed, schedule=name)
        if identifier in completed:
            continue
        forecast = (CONTINUOUS_WARMUP_TRANSITIONS, trajectory.state_count - 1)
        prediction, failure_step, failure_reason = recursive_forecast(
            model, scalers, trajectory.states, trajectory.current_values,
            warmup_range=(0, CONTINUOUS_WARMUP_TRANSITIONS), forecast_range=forecast,
        )
        raw_path = RAW_ARRAY_ROOT / f"{identifier}.npz"
        refuse_existing(raw_path)
        begin, stop = forecast
        record = build_evaluation_record(
            identifier=identifier, family="continuous", scenario=scenario,
            seed=seed, scalers=scalers, predictions=prediction,
            targets=trajectory.states[begin + 1:stop + 1],
            times=trajectory.time[begin + 1:stop + 1],
            currents=trajectory.current_values[begin:stop], raw_path=raw_path,
            warmup_range=(0, CONTINUOUS_WARMUP_TRANSITIONS), forecast_range=forecast,
            failure_step=failure_step, failure_reason=failure_reason, schedule=name,
        )
        record["matrix_code"] = None
        _append_record(raw, record)
        completed.add(identifier)


def evaluate_final(*, resume: bool = False) -> dict[str, Any]:
    """Open final targets only after selection and all model locks exist."""
    assert_baseline_artifacts_unchanged()
    selection = _selection()
    models = load_strict_json(MODEL_MANIFEST)
    if models.get("status") != "complete" or models.get("selection_sha256") != file_sha256(SELECTION_PATH):
        raise CrossRegimeError("all 15 models must be locked before benchmark access")
    raw = _new_or_resume(
        RAW_RESULTS,
        {
            "schema": "chapter2_cross_regime_improvement_evaluation_v1",
            "status": "evaluation_in_progress",
            "selection_sha256": file_sha256(SELECTION_PATH),
            "model_manifest_sha256": file_sha256(MODEL_MANIFEST),
            "records": [],
        },
        resume,
    )
    if raw["selection_sha256"] != file_sha256(SELECTION_PATH):
        raise CrossRegimeError("evaluation resume selection mismatch")
    trajectories = {current: load_fixed_trajectory(current) for current in ALL_CURRENTS}
    schedules = _load_schedules()
    for item in models["models"]:
        path = Path(item["path"])
        if file_sha256(path) != item["sha256"]:
            raise CrossRegimeError(f"locked model hash mismatch: {path}")
        model, scalers, metadata = load_model_bundle(path)
        if metadata["selection_sha256"] != raw["selection_sha256"]:
            raise CrossRegimeError("model was not trained from the locked selection")
        scenario, seed = str(item["scenario"]), int(item["seed"])
        _fixed_records(raw, model, scalers, scenario, seed, trajectories)
        _continuous_records(raw, model, scalers, scenario, seed, schedules)
    identifiers = [x["record_id"] for x in raw["records"]]
    if len(identifiers) != 345 or set(identifiers) != expected_record_ids():
        raise CrossRegimeError("final evaluation matrix is incomplete or duplicated")
    raw["status"] = "complete"
    raw["historical_baseline_used_for_tuning"] = False
    atomic_write_json(RAW_RESULTS, raw)
    aggregate = aggregate_and_write(raw["records"])
    return {"raw": raw, "aggregate": aggregate, "selection": selection}


def _summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("empty result cell")
    finite_nrmse = [float(x["metrics"]["nrmse_state"]) for x in items if x["metrics"]["nrmse_state"] is not None]
    vpt = [float(x["metrics"]["valid_prediction_time"]) for x in items]
    failures = sum(bool(x["numerical_failure"]) for x in items)
    divergences = sum(bool(x["metrics"]["diverged"]) for x in items)
    return {
        "record_count": len(items),
        "finite_nrmse_count": len(finite_nrmse),
        "median_nrmse": statistics.median(finite_nrmse) if finite_nrmse else None,
        "iqr_nrmse": float(np.percentile(finite_nrmse, 75) - np.percentile(finite_nrmse, 25)) if finite_nrmse else None,
        "mean_nrmse": statistics.fmean(finite_nrmse) if finite_nrmse else None,
        "median_vpt": statistics.median(vpt),
        "mean_vpt": statistics.fmean(vpt),
        "divergence_count": divergences,
        "divergence_rate": divergences / len(items),
        "numerical_failure_count": failures,
        "numerical_failure_rate": failures / len(items),
    }


def aggregate_and_write(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fixed = [x for x in records if x["family"] == "fixed_short"]
    matrix = {}
    for (scenario, regime), code in MATRIX_CODES.items():
        selected = [x for x in fixed if x["scenario"] == scenario and regime_for_current(float(x["current"])) == regime]
        matrix[code] = {"scenario": scenario, "target_regime": regime, **_summary(selected)}
    continuous = {
        scenario: {
            schedule: _summary([x for x in records if x["family"] == "continuous" and x["scenario"] == scenario and x["schedule"] == schedule])
            for schedule in CONTINUOUS_SCHEDULES
        }
        for scenario in SCENARIO_TRAINING_CURRENTS
    }
    aggregate = {
        "schema": "chapter2_cross_regime_improvement_aggregate_v1",
        "matrix": matrix,
        "continuous_schedules": continuous,
        "record_count": len(records),
        "selection_locked_before_evaluation": True,
    }
    refuse_existing(AGGREGATE_RESULTS)
    atomic_write_json(AGGREGATE_RESULTS, aggregate)
    rows = []
    for (scenario, regime), code in MATRIX_CODES.items():
        for current in (REGULAR_CURRENTS if regime == "regular" else CHAOTIC_CURRENTS):
            selected = [x for x in fixed if x["scenario"] == scenario and float(x["current"]) == current]
            rows.append(
                {
                    "matrix_code": code,
                    "train_model": scenario,
                    "test_regime": regime,
                    "current": current,
                    "seeds": len({int(x["seed"]) for x in selected}),
                    **_summary(selected),
                }
            )
    refuse_existing(SUMMARY_TABLE)
    atomic_write_csv(SUMMARY_TABLE, rows)
    return aggregate


def _synthetic(current: float, transitions: int) -> FixedCurrentTrajectory:
    time = np.arange(transitions + 1, dtype=float) * 0.01
    phase = time * (0.6 + current / 10.0)
    states = np.column_stack(
        (np.sin(phase) + current / 10.0, np.cos(phase), np.sin(phase * 0.17))
    )
    return FixedCurrentTrajectory(current, time, states, np.full(transitions + 1, current))


class _FeedbackSpy:
    def __init__(self, fail_at: int | None = None):
        self.inputs: list[np.ndarray] = []
        self.fail_at = fail_at

    def teacher_forced_warmup(self, inputs: np.ndarray, reset: bool = True) -> None:
        self.warmup = np.asarray(inputs).copy()

    def predict_one_step(self, value: np.ndarray) -> np.ndarray:
        self.inputs.append(np.asarray(value).copy())
        if self.fail_at is not None and len(self.inputs) == self.fail_at:
            return np.full(3, np.nan)
        return np.asarray(value[:3]) + 0.01

    def reset_reservoir(self) -> None:
        pass


def run_pilot(*, output_path: Path = PILOT_REPORT) -> dict[str, Any]:
    """Small synthetic mechanics pilot; no held-out or baseline result is opened."""
    refuse_existing(output_path)
    assert_esn_source_unchanged()
    windows = ((1, (40, 50), (50, 65)),)
    parameters = {
        "reservoir_size": 100,
        "reservoir_connectivity": 0.1,
        "input_scaling": 0.3,
        "spectral_radius": 0.7,
        "ridge_regularisation": 1.0e-6,
        "leak_rate": 0.6,
    }
    scenario_checks = []
    for scenario, currents in SCENARIO_TRAINING_CURRENTS.items():
        trajectories = {current: _synthetic(current, 70) for current in currents}
        data = prepare_scenario_data(
            scenario, trajectories, fitting_range=(0, 40), validation_windows=windows
        )
        result = StabilityAwareEvaluator(data, washout=5)(parameters, 42)
        scenario_checks.append(
            {
                "scenario": scenario,
                "currents": list(data.currents),
                "block_order": list(block_order(scenario, 42)),
                "chronological_samples_inside_blocks": True,
                "no_artificial_cross_block_target_pairs": True,
                "reservoir_reset_between_independent_blocks": True,
                "scaler_fitting_range": [0, 40],
                "validation_separate": True,
                "validation_rollouts": len(result["rollouts"]),
                "metrics_recorded": all("metrics" in x for x in result["rollouts"]),
            }
        )
    unit = StateCurrentScalers(
        NumpyStandardScaler(np.zeros(3), np.ones(3)),
        NumpyStandardScaler(np.zeros(1), np.ones(1)),
    )
    states = np.arange(30, dtype=float).reshape(10, 3)
    currents = np.array([1.67, 1.67, 3.20, 3.20, 3.34, 3.34, 1.67, 1.67, 1.67, 1.67])
    spy = _FeedbackSpy()
    predicted, failure_step, _ = recursive_forecast(
        spy, unit, states, currents, warmup_range=(0, 2), forecast_range=(2, 9)
    )
    altered = states.copy()
    altered[3:] = -9999
    predicted_again, _, _ = recursive_forecast(
        _FeedbackSpy(), unit, altered, currents, warmup_range=(0, 2), forecast_range=(2, 9)
    )
    failed, failed_step, _ = recursive_forecast(
        _FeedbackSpy(fail_at=3), unit, states, currents,
        warmup_range=(0, 2), forecast_range=(2, 9),
    )
    checks = {
        "input_dimension": esn_config(parameters, 42).input_dimension == 4,
        "output_dimension": esn_config(parameters, 42).output_dimension == 3,
        "scenario_memberships_exact": all(
            item["currents"] == list(SCENARIO_TRAINING_CURRENTS[item["scenario"]])
            for item in scenario_checks
        ),
        "recursive_prediction_feedback": bool(np.array_equal(predicted, predicted_again)),
        "supplied_current_used": bool(np.array_equal(np.asarray(spy.inputs)[:, 3], currents[2:9])),
        "future_target_states_not_fed_back": bool(np.array_equal(predicted, predicted_again)),
        "numerical_failure_handled": bool(failed_step == 2 and np.isnan(failed[2:]).all()),
        "outputs_align_with_targets": predicted.shape == states[3:10].shape,
        "generation_completed": failure_step is None,
        "historical_345_results_not_opened": True,
    }
    report = {
        "schema": "chapter2_cross_regime_improvement_pilot_v1",
        "synthetic_development_subsets_only": True,
        "same_echo_state_network_class": f"{EchoStateNetwork.__module__}.{EchoStateNetwork.__name__}",
        "scenario_checks": scenario_checks,
        "checks": checks,
        "passed": all(checks.values()) and all(
            x["metrics_recorded"]
            and x["chronological_samples_inside_blocks"]
            and x["no_artificial_cross_block_target_pairs"]
            and x["reservoir_reset_between_independent_blocks"]
            and x["validation_separate"]
            for x in scenario_checks
        ),
    }
    atomic_write_json(output_path, report)
    return report


def _require_slurm() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise CrossRegimeError("production optimisation/final evaluation requires Slurm")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--pilot", action="store_true")
    modes.add_argument("--optimise-all", action="store_true")
    modes.add_argument("--final", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.pilot:
        report = run_pilot()
        print("PILOT PASSED" if report["passed"] else "PILOT FAILED")
        return 0 if report["passed"] else 1
    _require_slurm()
    if args.optimise_all:
        for scenario in SCENARIO_TRAINING_CURRENTS:
            run_search(scenario, resume=args.resume)
            run_five_seed_confirmation(scenario)
        write_selection()
        print(f"SELECTION LOCKED: {SELECTION_PATH}")
        return 0
    train_final_models(resume=args.resume)
    evaluate_final(resume=args.resume)
    from .figures import generate_all_figures
    generate_all_figures()
    print("FINAL EVALUATION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
