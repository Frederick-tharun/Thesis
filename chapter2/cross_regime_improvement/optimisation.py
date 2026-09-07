"""Leakage-safe scenario optimisation and five-seed robust confirmation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import statistics
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from chapter2.cross_regime import load_training_prefix, recursive_forecast
from chapter2.cross_regime_numerics import evaluate_predictions
from chapter2.esn_config import ESNModelConfig
from chapter2.esn_data import (
    FixedCurrentTrajectory,
    NumpyStandardScaler,
    StateCurrentScalers,
    file_sha256,
)
from chapter2.esn_model import EchoStateNetwork, TrainingSequence
from chapter2.esn_optimisation import (
    PARAMETER_AWARE,
    SearchSettings,
    atomic_write_json,
    hyperparameters_to_point,
    load_strict_json,
    optimizer_runtime_record,
    optimizer_settings_record,
    point_to_hyperparameters,
    run_bayesian_search,
    search_dimensions,
    serialized_hyperparameters,
    software_versions,
)
from chapter2.esn_config import FIXED_DATASETS

from .config import (
    BAYESIAN_CALLS,
    BIAS_SCALING,
    CANDIDATE_MODEL_SEED,
    COLLAPSE_THRESHOLD,
    DIVERGENCE_THRESHOLD,
    FITTING_RANGE,
    INITIAL_RANDOM_CALLS,
    INPUT_DIMENSION,
    NONFINITE_FAILURE_SCORE,
    OPTIMISATION_ROOT,
    OPTIMISER_SEEDS,
    OUTPUT_DIMENSION,
    REGULARISE_BIAS,
    SCENARIO_TRAINING_CURRENTS,
    SEARCH_PARAMETER_ORDER,
    SEARCH_SPACE_DEFINITION,
    SEEDS,
    TOP_CANDIDATE_COUNT,
    TRAINING_WASHOUT,
    VALIDATION_RANGE,
    VALIDATION_WINDOWS,
    VALID_PREDICTION_THRESHOLD,
    block_order,
)
from .protection import assert_baseline_artifacts_unchanged, assert_esn_source_unchanged


HISTORY_SCHEMA = "chapter2_cross_regime_improvement_optimisation_v1"
SELECTION_SCHEMA = "chapter2_cross_regime_improvement_selection_v1"
OBJECTIVE_DEFINITION = (
    "Validation-only bounded scalar proxy: 1e6*numerical_failures + "
    "1e4*divergences + 100*b(worst NRMSE) + 10*b(median NRMSE) + "
    "b(mean NRMSE) - 0.1*b(median VPT), b(x)=x/(1+x). Exact final "
    "candidate ranking remains lexicographic and is authoritative."
)
SELECTION_RULES = (
    "lowest numerical-failure count",
    "lowest divergence count",
    "lowest worst-case validation NRMSE",
    "lowest median validation NRMSE",
    "lowest mean validation NRMSE",
    "highest median validation VPT",
    "highest mean validation VPT",
    "lexicographically serialized hyperparameters",
)


@dataclass(frozen=True)
class ScenarioOptimisationData:
    scenario: str
    trajectories: Mapping[float, FixedCurrentTrajectory]
    scalers: StateCurrentScalers
    fitting_range: tuple[int, int]
    validation_windows: tuple[
        tuple[int, tuple[int, int], tuple[int, int]], ...
    ]

    @property
    def currents(self) -> tuple[float, ...]:
        return tuple(SCENARIO_TRAINING_CURRENTS[self.scenario])


def _authorised_load(
    scenario: str,
    current: float,
    transition_count: int,
    loader: Callable[[float, int], FixedCurrentTrajectory],
) -> FixedCurrentTrajectory:
    if scenario not in SCENARIO_TRAINING_CURRENTS:
        raise ValueError(f"unknown scenario: {scenario}")
    if float(current) not in SCENARIO_TRAINING_CURRENTS[scenario]:
        raise PermissionError(
            f"I={current:.2f} is forbidden during {scenario} optimisation"
        )
    return loader(float(current), int(transition_count))


def load_scenario_optimisation_data(
    scenario: str,
    *,
    loader: Callable[[float, int], FixedCurrentTrajectory] = load_training_prefix,
) -> ScenarioOptimisationData:
    """Load only scenario currents, and only through validation transition 69,999."""
    if scenario not in SCENARIO_TRAINING_CURRENTS:
        raise ValueError(f"unknown scenario: {scenario}")
    stop = VALIDATION_RANGE[1]
    trajectories = {
        float(current): _authorised_load(scenario, current, stop, loader)
        for current in SCENARIO_TRAINING_CURRENTS[scenario]
    }
    return prepare_scenario_data(scenario, trajectories)


def prepare_scenario_data(
    scenario: str,
    trajectories: Mapping[float, FixedCurrentTrajectory],
    *,
    fitting_range: tuple[int, int] = FITTING_RANGE,
    validation_windows: Sequence[
        tuple[int, tuple[int, int], tuple[int, int]]
    ] = VALIDATION_WINDOWS,
) -> ScenarioOptimisationData:
    """Fit scalers on fitting inputs only and preserve each trajectory as a block."""
    if scenario not in SCENARIO_TRAINING_CURRENTS:
        raise ValueError(f"unknown scenario: {scenario}")
    expected = tuple(float(x) for x in SCENARIO_TRAINING_CURRENTS[scenario])
    blocks = {float(key): value for key, value in trajectories.items()}
    if set(blocks) != set(expected):
        raise ValueError("optimisation trajectory membership mismatch")
    fit_start, fit_stop = fitting_range
    windows = tuple(validation_windows)
    max_stop = max(scored[1] for _, _, scored in windows)
    if not (0 <= fit_start < fit_stop <= min(warm[0] for _, warm, _ in windows)):
        raise ValueError("fitting and validation must be chronological and disjoint")
    for current, trajectory in blocks.items():
        if trajectory.current != current or trajectory.state_count < max_stop + 1:
            raise ValueError("trajectory identity or length mismatch")
        if not np.all(trajectory.current_values == current):
            raise ValueError("fixed-current trajectory contains a current transition")
    fitting_states = np.concatenate(
        [blocks[current].states[fit_start:fit_stop] for current in expected]
    )
    fitting_currents = np.concatenate(
        [blocks[current].current_values[fit_start:fit_stop] for current in expected]
    )[:, None]
    scalers = StateCurrentScalers(
        NumpyStandardScaler.fit(fitting_states),
        NumpyStandardScaler.fit(fitting_currents),
    )
    return ScenarioOptimisationData(
        scenario,
        blocks,
        scalers,
        (fit_start, fit_stop),
        windows,
    )


def training_sequences(
    data: ScenarioOptimisationData, seed: int
) -> tuple[TrainingSequence, ...]:
    """Create chronological independent blocks; never create a cross-block pair."""
    start, stop = data.fitting_range
    sequences = []
    for current in block_order(data.scenario, int(seed)):
        trajectory = data.trajectories[current]
        inputs = np.column_stack(
            (
                trajectory.states[start:stop],
                trajectory.current_values[start:stop],
            )
        )
        targets = trajectory.states[start + 1 : stop + 1]
        sequences.append(
            TrainingSequence(
                data.scalers.transform_inputs(inputs),
                data.scalers.transform_targets(targets),
            )
        )
    return tuple(sequences)


def esn_config(parameters: Mapping[str, Any], seed: int) -> ESNModelConfig:
    """Map the unchanged Chapter 2 search family to the unchanged ESN class."""
    return ESNModelConfig(
        reservoir_size=int(parameters["reservoir_size"]),
        reservoir_connectivity=float(parameters["reservoir_connectivity"]),
        input_scaling=float(parameters["input_scaling"]),
        spectral_radius=float(parameters["spectral_radius"]),
        ridge_regularisation=float(parameters["ridge_regularisation"]),
        leak_rate=float(parameters["leak_rate"]),
        bias_scaling=BIAS_SCALING,
        regularise_bias=REGULARISE_BIAS,
        seed=int(seed),
        input_dimension=INPUT_DIMENSION,
        output_dimension=OUTPUT_DIMENSION,
    )


def _bounded(value: float) -> float:
    resolved = max(0.0, float(value))
    return resolved / (1.0 + resolved)


def aggregate_rollouts(rollouts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = tuple(rollouts)
    if not items:
        raise ValueError("cannot aggregate zero validation rollouts")
    scores = [float(item["aggregate_nrmse_value"]) for item in items]
    vpts = [float(item["metrics"]["valid_prediction_time"]) for item in items]
    finite_rmse = [
        float(item["metrics"]["rmse_state"])
        for item in items
        if item["metrics"]["rmse_state"] is not None
    ]
    r2 = [
        float(item["metrics"]["r2_macro"])
        for item in items
        if item["metrics"]["r2_macro"] is not None
    ]
    correlations = [
        float(item["metrics"]["correlation_macro"])
        for item in items
        if item["metrics"]["correlation_macro"] is not None
    ]
    per_current = {}
    for current in sorted({float(item["current"]) for item in items}):
        selected = [item for item in items if float(item["current"]) == current]
        per_current[f"{current:.2f}"] = {
            "rollout_count": len(selected),
            "numerical_failure_count": sum(bool(x["numerical_failure"]) for x in selected),
            "divergence_count": sum(bool(x["metrics"]["diverged"]) for x in selected),
            "worst_nrmse": max(float(x["aggregate_nrmse_value"]) for x in selected),
            "median_nrmse": statistics.median(float(x["aggregate_nrmse_value"]) for x in selected),
            "median_vpt": statistics.median(float(x["metrics"]["valid_prediction_time"]) for x in selected),
        }
    return {
        "rollout_count": len(items),
        "numerical_failure_count": sum(bool(x["numerical_failure"]) for x in items),
        "divergence_count": sum(bool(x["metrics"]["diverged"]) for x in items),
        "prediction_collapse_count": sum(bool(x["metrics"]["prediction_collapse_any"]) for x in items),
        "worst_nrmse": max(scores),
        "median_nrmse": statistics.median(scores),
        "mean_nrmse": statistics.fmean(scores),
        "median_rmse": statistics.median(finite_rmse) if finite_rmse else None,
        "mean_rmse": statistics.fmean(finite_rmse) if finite_rmse else None,
        "median_vpt": statistics.median(vpts),
        "mean_vpt": statistics.fmean(vpts),
        "median_r2": statistics.median(r2) if r2 else None,
        "mean_r2": statistics.fmean(r2) if r2 else None,
        "median_pearson_correlation": statistics.median(correlations) if correlations else None,
        "mean_pearson_correlation": statistics.fmean(correlations) if correlations else None,
        "per_current": per_current,
    }


def stability_objective(aggregate: Mapping[str, Any]) -> float:
    """Bounded Bayesian proxy whose priority bands cannot overlap."""
    return float(
        1_000_000 * int(aggregate["numerical_failure_count"])
        + 10_000 * int(aggregate["divergence_count"])
        + 100 * _bounded(float(aggregate["worst_nrmse"]))
        + 10 * _bounded(float(aggregate["median_nrmse"]))
        + _bounded(float(aggregate["mean_nrmse"]))
        - 0.1 * _bounded(float(aggregate["median_vpt"]))
    )


def candidate_rank_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    aggregate = item["aggregate"]
    parameters = item.get("hyperparameters", {})
    return (
        int(aggregate["numerical_failure_count"]),
        int(aggregate["divergence_count"]),
        float(aggregate["worst_nrmse"]),
        float(aggregate["median_nrmse"]),
        float(aggregate["mean_nrmse"]),
        -float(aggregate["median_vpt"]),
        -float(aggregate["mean_vpt"]),
        serialized_hyperparameters(parameters),
    )


class StabilityAwareEvaluator:
    """Fit one candidate and score only its scenario's validation windows."""

    def __init__(self, data: ScenarioOptimisationData, *, washout: int = TRAINING_WASHOUT):
        self.data = data
        self.washout = int(washout)

    def _failure_rollout(
        self,
        current: float,
        window: int,
        warmup: tuple[int, int],
        scored: tuple[int, int],
        seed: int,
        reason: str,
    ) -> dict[str, Any]:
        horizon = scored[1] - scored[0]
        trajectory = self.data.trajectories[current]
        predictions = np.full((horizon, OUTPUT_DIMENSION), np.nan)
        targets = trajectory.states[scored[0] + 1 : scored[1] + 1]
        fields, _, _ = evaluate_predictions(predictions, targets, self.data.scalers.state.scale)
        return {
            "current": current,
            "window": window,
            "warmup_range": list(warmup),
            "scored_range": list(scored),
            "model_seed": seed,
            "fit_failure": True,
            "generation_failure_reason": reason,
            **fields,
        }

    def __call__(self, parameters: dict[str, Any], model_seed: int) -> dict[str, Any]:
        assert_esn_source_unchanged()
        cases = [
            (current, number, warmup, scored)
            for current in self.data.currents
            for number, warmup, scored in self.data.validation_windows
        ]
        try:
            model = EchoStateNetwork(esn_config(parameters, model_seed))
            model.fit(training_sequences(self.data, model_seed), washout=self.washout)
        except (ArithmeticError, RuntimeError, ValueError) as error:
            rollouts = [
                self._failure_rollout(*case, model_seed, f"model_fit_failure:{type(error).__name__}")
                for case in cases
            ]
            aggregate = aggregate_rollouts(rollouts)
            return {"objective": stability_objective(aggregate), "rollouts": rollouts, "aggregate": aggregate}

        rollouts = []
        for current, number, warmup, scored in cases:
            trajectory = self.data.trajectories[current]
            try:
                predictions, generation_step, generation_reason = recursive_forecast(
                    model,
                    self.data.scalers,
                    trajectory.states,
                    trajectory.current_values,
                    warmup_range=warmup,
                    forecast_range=scored,
                )
                targets = trajectory.states[scored[0] + 1 : scored[1] + 1]
                fields, _, _ = evaluate_predictions(
                    predictions, targets, self.data.scalers.state.scale
                )
                rollouts.append(
                    {
                        "current": current,
                        "window": number,
                        "warmup_range": list(warmup),
                        "scored_range": list(scored),
                        "model_seed": int(model_seed),
                        "fit_failure": False,
                        "generation_failure_step": generation_step,
                        "generation_failure_reason": generation_reason,
                        **fields,
                    }
                )
            except (ArithmeticError, RuntimeError, ValueError) as error:
                rollouts.append(
                    self._failure_rollout(
                        current,
                        number,
                        warmup,
                        scored,
                        model_seed,
                        f"rollout_failure:{type(error).__name__}",
                    )
                )
        aggregate = aggregate_rollouts(rollouts)
        return {
            "objective": stability_objective(aggregate),
            "rollouts": rollouts,
            "aggregate": aggregate,
        }


def history_path(scenario: str) -> Path:
    if scenario not in SCENARIO_TRAINING_CURRENTS:
        raise ValueError(f"unknown scenario: {scenario}")
    return OPTIMISATION_ROOT / f"{scenario}_history.json"


def _dataset_hashes(scenario: str) -> dict[str, str]:
    allowed = set(SCENARIO_TRAINING_CURRENTS[scenario])
    return {
        str(record.path): file_sha256(record.path)
        for record in FIXED_DATASETS
        if record.current in allowed
    }


def checkpoint_metadata(
    data: ScenarioOptimisationData, settings: SearchSettings
) -> dict[str, Any]:
    return {
        "schema": HISTORY_SCHEMA,
        "label": "VALIDATION-ONLY; FINAL TARGET-REGIME BENCHMARKS UNOPENED",
        "scenario": data.scenario,
        "model_type": PARAMETER_AWARE,
        "input_dimension": INPUT_DIMENSION,
        "output_dimension": OUTPUT_DIMENSION,
        "search_space": SEARCH_SPACE_DEFINITION,
        "parameter_order": list(SEARCH_PARAMETER_ORDER),
        "optimizer_settings": optimizer_settings_record(settings),
        "software_versions": software_versions(),
        "optimizer_runtime": optimizer_runtime_record(),
        "objective": OBJECTIVE_DEFINITION,
        "authoritative_selection_rules": list(SELECTION_RULES),
        "replay_verification": {
            "point_order": list(SEARCH_PARAMETER_ORDER),
            "integer_and_categorical_comparison": "exact",
            "float_relative_tolerance": 1.0e-15,
            "float_absolute_tolerance": 0.0,
        },
        "thresholds": {
            "valid_prediction": VALID_PREDICTION_THRESHOLD,
            "divergence": DIVERGENCE_THRESHOLD,
            "collapse": COLLAPSE_THRESHOLD,
            "numerical_failure_score": NONFINITE_FAILURE_SCORE,
        },
        "training_washout_per_independently_reset_block": TRAINING_WASHOUT,
        "training_currents": list(data.currents),
        "block_order_seed_42": list(block_order(data.scenario, CANDIDATE_MODEL_SEED)),
        "fitting_transition_range": list(data.fitting_range),
        "validation_transition_range": list(VALIDATION_RANGE),
        "validation_windows": [
            {"window": n, "warmup": list(w), "scored": list(s)}
            for n, w, s in data.validation_windows
        ],
        "dataset_hashes": _dataset_hashes(data.scenario),
        "preprocessing": {
            "fitted_from": "scenario training-current fitting-transition inputs only",
            "state_mean": data.scalers.state.mean.tolist(),
            "state_scale": data.scalers.state.scale.tolist(),
            "current_mean": data.scalers.current.mean.tolist(),
            "current_scale": data.scalers.current.scale.tolist(),
        },
        "data_access": {
            "allowed_currents": list(data.currents),
            "held_out_evaluation_opened": False,
            "target_regime_opened_for_selection": False,
            "historical_345_results_opened": False,
            "continuous_schedules_opened": False,
        },
    }


def run_search(
    scenario: str,
    *,
    resume: bool = False,
    n_calls: int = BAYESIAN_CALLS,
    n_initial_calls: int = INITIAL_RANDOM_CALLS,
    data: ScenarioOptimisationData | None = None,
) -> dict[str, Any]:
    assert_baseline_artifacts_unchanged()
    prepared = data or load_scenario_optimisation_data(scenario)
    settings = SearchSettings(
        n_calls=int(n_calls),
        n_initial_calls=int(n_initial_calls),
        acquisition_function="EI",
        optimizer_seed=OPTIMISER_SEEDS[scenario],
        candidate_model_seed=CANDIDATE_MODEL_SEED,
    )
    return run_bayesian_search(
        checkpoint_path=history_path(scenario),
        model_type=PARAMETER_AWARE,
        evaluator=StabilityAwareEvaluator(prepared),
        metadata=checkpoint_metadata(prepared, settings),
        settings=settings,
        resume=resume,
    )


def _unique_top_trials(history: Mapping[str, Any], count: int) -> list[Mapping[str, Any]]:
    seen = set()
    selected = []
    for trial in sorted(history["trials"], key=candidate_rank_key):
        key = serialized_hyperparameters(trial["hyperparameters"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(trial)
        if len(selected) == min(count, len(history["trials"])):
            break
    return selected


def _confirmation_aggregate(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rollouts = [rollout for result in seed_results for rollout in result["rollouts"]]
    aggregate = aggregate_rollouts(rollouts)
    per_seed = {str(item["model_seed"]): item["aggregate"] for item in seed_results}
    worst_seed = max(
        seed_results,
        key=lambda item: candidate_rank_key(
            {"aggregate": item["aggregate"], "hyperparameters": {}}
        )[:-1],
    )["model_seed"]
    aggregate.update(
        {
            "seed_count": len(seed_results),
            "per_seed": per_seed,
            "worst_seed": int(worst_seed),
            "across_seed_summaries": {
                key: {
                    "median": statistics.median(
                        float(item["aggregate"][key]) for item in seed_results
                    ),
                    "mean": statistics.fmean(
                        float(item["aggregate"][key]) for item in seed_results
                    ),
                }
                for key in ("numerical_failure_count", "divergence_count", "median_nrmse", "mean_nrmse", "median_vpt", "mean_vpt")
            },
        }
    )
    return aggregate


def run_five_seed_confirmation(
    scenario: str,
    *,
    data: ScenarioOptimisationData | None = None,
    top_count: int = TOP_CANDIDATE_COUNT,
) -> dict[str, Any]:
    path = history_path(scenario)
    history = load_strict_json(path)
    if history["status"] not in {"search_complete", "robust_confirmation_in_progress", "complete"}:
        raise ValueError("scenario search must complete before confirmation")
    prepared = data or load_scenario_optimisation_data(scenario)
    evaluator = StabilityAwareEvaluator(prepared)
    top = _unique_top_trials(history, top_count)
    confirmations = {
        serialized_hyperparameters(x["hyperparameters"]): x
        for x in history.get("robust_confirmations", [])
    }
    for rank, trial in enumerate(top, start=1):
        parameters = trial["hyperparameters"]
        key = serialized_hyperparameters(parameters)
        confirmation = confirmations.get(
            key,
            {
                "validation_rank": rank,
                "source_trial_index": trial["trial_index"],
                "hyperparameters": parameters,
                "seed_results": [],
                "complete": False,
                "aggregate": None,
            },
        )
        completed = {int(x["model_seed"]) for x in confirmation["seed_results"]}
        for seed in SEEDS:
            if seed in completed:
                continue
            evaluated = (
                {
                    "objective": trial["objective"],
                    "rollouts": trial["rollouts"],
                    "aggregate": trial["aggregate"],
                }
                if seed == CANDIDATE_MODEL_SEED
                else evaluator(parameters, seed)
            )
            confirmation["seed_results"].append(
                {"model_seed": seed, "reused_search_result": seed == CANDIDATE_MODEL_SEED, **evaluated}
            )
            confirmations[key] = confirmation
            history["robust_confirmations"] = list(confirmations.values())
            history["status"] = "robust_confirmation_in_progress"
            atomic_write_json(path, history)
        confirmation["seed_results"] = sorted(
            confirmation["seed_results"], key=lambda x: SEEDS.index(int(x["model_seed"]))
        )
        confirmation["aggregate"] = _confirmation_aggregate(confirmation["seed_results"])
        confirmation["complete"] = True
        confirmations[key] = confirmation
        history["robust_confirmations"] = list(confirmations.values())
        atomic_write_json(path, history)
    history["robust_confirmations"] = sorted(
        confirmations.values(), key=lambda x: int(x["validation_rank"])
    )
    history["status"] = "complete"
    atomic_write_json(path, history)
    return history


def selected_confirmation(history: Mapping[str, Any]) -> Mapping[str, Any]:
    complete = [item for item in history.get("robust_confirmations", []) if item.get("complete")]
    if not complete:
        raise ValueError("history has no complete five-seed confirmation")
    return min(complete, key=candidate_rank_key)


def write_selection(*, output_path: Path | None = None) -> dict[str, Any]:
    destination = output_path or OPTIMISATION_ROOT / "selection.json"
    if destination.exists():
        raise FileExistsError(f"selection lock already exists: {destination}")
    histories = {scenario: load_strict_json(history_path(scenario)) for scenario in SCENARIO_TRAINING_CURRENTS}
    if any(item["status"] != "complete" for item in histories.values()):
        raise ValueError("all three scenario searches and confirmations must be complete")
    models = {}
    for scenario, history in histories.items():
        selected = selected_confirmation(history)
        models[scenario] = {
            "training_currents": list(SCENARIO_TRAINING_CURRENTS[scenario]),
            "hyperparameters": selected["hyperparameters"],
            "five_seed_validation": selected["aggregate"],
            "source_trial_index": selected["source_trial_index"],
            "history_sha256": file_sha256(history_path(scenario)),
        }
    artifact = {
        "schema": SELECTION_SCHEMA,
        "status": "LOCKED BEFORE FINAL EVALUATION",
        "same_echo_state_network_architecture": True,
        "input_dimension": INPUT_DIMENSION,
        "output_dimension": OUTPUT_DIMENSION,
        "authoritative_selection_rules": list(SELECTION_RULES),
        "models": models,
        "data_access": {
            "historical_baseline_used_for_tuning": False,
            "final_target_regime_used_for_tuning": False,
            "continuous_schedules_used_for_tuning": False,
        },
    }
    atomic_write_json(destination, artifact)
    return artifact


def describe_design() -> dict[str, Any]:
    """Small, data-free protocol record used by tests and the mechanics pilot."""
    return {
        "scenarios": {key: list(value) for key, value in SCENARIO_TRAINING_CURRENTS.items()},
        "mixed_orders": {str(seed): list(block_order("mixed_shuffled", seed)) for seed in SEEDS},
        "search_space": SEARCH_SPACE_DEFINITION,
        "selection_rules": list(SELECTION_RULES),
        "thresholds": {
            "valid_prediction": VALID_PREDICTION_THRESHOLD,
            "divergence": DIVERGENCE_THRESHOLD,
            "collapse": COLLAPSE_THRESHOLD,
            "numerical_failure_score": NONFINITE_FAILURE_SCORE,
        },
        "model_dimensions": [INPUT_DIMENSION, OUTPUT_DIMENSION],
    }
