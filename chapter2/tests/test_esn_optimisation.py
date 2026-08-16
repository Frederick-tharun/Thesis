"""Synthetic tests for leakage-safe Chapter 2 Bayesian optimisation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from skopt.space import Categorical, Real

from chapter2.esn_config import TRAIN_CURRENTS
from chapter2.esn_data import (
    NumpyStandardScaler,
    OneStepPairs,
    StateCurrentScalers,
    ValidationWindowView,
)
from chapter2.esn_metrics import COLLAPSE_STD_RATIO_THRESHOLD
from chapter2.esn_model import TrainingSequence
from chapter2 import esn_optimisation as optimisation
from chapter2.esn_optimisation import (
    ACQUISITION_FUNCTION,
    BAYESIAN_CALLS_PER_MODEL,
    CANDIDATE_MODEL_SEED,
    CHECKPOINT_SCHEMA,
    DIVERGENCE_THRESHOLD,
    INITIAL_RANDOM_CALLS,
    MODEL_INPUT_DIMENSIONS,
    NONFINITE_FAILURE_SCORE,
    ORDINARY_BASELINE,
    PARAMETER_AWARE,
    SEARCH_SEEDS,
    SEARCH_SPACE_DEFINITION,
    TRAINING_WASHOUT,
    VALID_PREDICTION_THRESHOLD,
    ModelOptimisationData,
    RealCandidateEvaluator,
    SearchSettings,
    ValidationCase,
    aggregate_rollouts,
    atomic_write_json,
    build_selection_artifact,
    load_strict_json,
    model_config,
    prepare_model_data,
    robust_tie_key,
    rollout_objective,
    run_bayesian_search,
    run_robust_confirmation,
    search_dimensions,
)


POINTS = [
    [100, 0.10, 0.20, 0.80, 1.0e-6, 0.50],
    [200, 0.20, 0.30, 0.90, 2.0e-6, 0.40],
    [300, 0.30, 0.40, 1.00, 3.0e-6, 0.30],
    [100, 0.40, 0.50, 1.10, 4.0e-6, 0.20],
    [200, 0.50, 0.60, 1.20, 5.0e-6, 0.10],
    [300, 0.60, 0.70, 1.30, 6.0e-6, 0.60],
]


class FakeOptimizer:
    def __init__(self) -> None:
        self.told: list[tuple[list[float], float]] = []

    def ask(self) -> list[float]:
        return list(POINTS[len(self.told)])

    def tell(self, point: list[float], score: float) -> None:
        self.told.append((list(point), float(score)))


def _optimizer_factory(
    model_type: str, settings: SearchSettings
) -> FakeOptimizer:
    assert model_type in (PARAMETER_AWARE, ORDINARY_BASELINE)
    return FakeOptimizer()


def _rollouts(base_score: float, seed: int = 42) -> list[dict]:
    rows = []
    for current in TRAIN_CURRENTS:
        for window in (1, 2, 3):
            score = base_score + current * 1.0e-3 + window * 1.0e-2
            rows.append(
                {
                    "current": current,
                    "window": window,
                    "model_seed": seed,
                    "objective_nrmse": score,
                    "metrics": {
                        "nrmse_state": score,
                        "valid_prediction_steps": 100 - window,
                        "diverged": False,
                        "prediction_collapse_any": False,
                        "r2_macro": 0.5,
                        "correlation_macro": 0.8,
                    },
                }
            )
    return rows


def _evaluation(parameters: dict, seed: int) -> dict:
    base = float(parameters["reservoir_size"]) / 1_000.0 + seed * 1.0e-7
    rollouts = _rollouts(base, seed)
    return {
        "objective": rollout_objective(rollouts),
        "rollouts": rollouts,
        "aggregate": aggregate_rollouts(rollouts),
    }


def _real_skopt_evaluation(parameters: dict, seed: int) -> dict:
    values = np.asarray(
        [
            float(parameters["reservoir_size"]) / 300.0,
            float(parameters["reservoir_connectivity"]),
            float(parameters["input_scaling"]) / 3.0,
            float(parameters["spectral_radius"]) / 3.0,
            (np.log10(float(parameters["ridge_regularisation"])) + 10.0) / 8.0,
            float(parameters["leak_rate"]),
        ]
    )
    base = float(np.mean(np.square(values - 0.37))) + seed * 1.0e-9
    rollouts = _rollouts(base, seed)
    return {
        "objective": rollout_objective(rollouts),
        "rollouts": rollouts,
        "aggregate": aggregate_rollouts(rollouts),
    }


def _metadata(model_type: str, settings: SearchSettings) -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "label": optimisation.RESULT_LABEL,
        "model_type": model_type,
        "input_dimension": MODEL_INPUT_DIMENSIONS[model_type],
        "output_dimension": 3,
        "search_space": SEARCH_SPACE_DEFINITION,
        "optimizer_settings": optimisation.optimizer_settings_record(settings),
        "software_versions": optimisation.software_versions(),
        "objective": optimisation.OBJECTIVE_DEFINITION,
        "thresholds": {
            "valid_prediction": VALID_PREDICTION_THRESHOLD,
            "divergence": DIVERGENCE_THRESHOLD,
            "collapse_std_ratio": COLLAPSE_STD_RATIO_THRESHOLD,
            "nonfinite_failure_score": NONFINITE_FAILURE_SCORE,
        },
        "training_washout_per_trajectory": TRAINING_WASHOUT,
        "training_currents": list(TRAIN_CURRENTS),
        "fitting_transition_range": [0, 40_000],
        "validation_transition_range": [40_000, 70_000],
        "dataset_hashes": {"training.npz": "a" * 64},
        "preprocessing": {
            "state_mean": [0.0, 0.0, 0.0],
            "state_scale": [1.0, 1.0, 1.0],
            "current_mean": [0.0],
            "current_scale": [1.0],
        },
        "git": {"commit": "test", "dirty": True, "status_short": []},
    }


def _settings(model_type: str, n_calls: int = 5) -> SearchSettings:
    return SearchSettings(
        n_calls=n_calls,
        n_initial_calls=min(2, n_calls),
        acquisition_function=ACQUISITION_FUNCTION,
        optimizer_seed=SEARCH_SEEDS[model_type],
        candidate_model_seed=CANDIDATE_MODEL_SEED,
    )


def _pair(
    current: float,
    start: int,
    rows: int,
    *,
    first_state: np.ndarray | None = None,
) -> OneStepPairs:
    state = (
        np.array([current, 2.0 * current, -current])
        if first_state is None
        else np.asarray(first_state, dtype=float)
    )
    inputs = np.column_stack(
        (
            np.repeat(state.reshape(1, 3), rows, axis=0),
            np.full(rows, current),
        )
    )
    targets = inputs[:, :3] + 0.01
    return OneStepPairs(
        inputs,
        targets,
        np.arange(start, start + rows),
        current,
    )


def _small_prepared() -> tuple[SimpleNamespace, ...]:
    items = []
    for current in TRAIN_CURRENTS:
        fitting = _pair(current, 0, 4)
        windows = []
        for number in (1, 2, 3):
            warmup = _pair(current, number * 10, 2)
            scored = _pair(
                current,
                number * 10 + 2,
                3,
                first_state=warmup.targets[-1],
            )
            definition = SimpleNamespace(number=number)
            windows.append(ValidationWindowView(definition, warmup, scored))
        items.append(
            SimpleNamespace(
                current=current,
                fitting=fitting,
                validation_windows=tuple(windows),
            )
        )
    return tuple(items)


def _identity_scalers() -> StateCurrentScalers:
    return StateCurrentScalers(
        NumpyStandardScaler(np.zeros(3), np.ones(3)),
        NumpyStandardScaler(np.zeros(1), np.ones(1)),
    )


def _small_cases(model_type: str) -> tuple[ValidationCase, ...]:
    cases = []
    input_dimension = MODEL_INPUT_DIMENSIONS[model_type]
    for current in TRAIN_CURRENTS:
        for window in (1, 2, 3):
            cases.append(
                ValidationCase(
                    current=current,
                    window=window,
                    warmup_inputs=np.zeros((2, input_dimension)),
                    initial_state=np.zeros(3),
                    current_values=(
                        np.full(3, current)
                        if model_type == PARAMETER_AWARE
                        else None
                    ),
                    targets_physical=np.zeros((3, 3)),
                    warmup_range=(0, 2),
                    scored_range=(2, 5),
                )
            )
    return tuple(cases)


def test_frozen_search_space_bounds_types_and_order() -> None:
    dimensions = search_dimensions()

    assert [dimension.name for dimension in dimensions] == list(
        optimisation.SEARCH_PARAMETER_ORDER
    )
    assert isinstance(dimensions[0], Categorical)
    assert tuple(dimensions[0].categories) == (100, 200, 300)
    for dimension, low, high, prior in (
        (dimensions[1], 0.01, 1.0, "uniform"),
        (dimensions[2], 0.01, 3.0, "uniform"),
        (dimensions[3], 0.01, 3.0, "uniform"),
        (dimensions[4], 1.0e-10, 1.0e-2, "log-uniform"),
        (dimensions[5], 0.01, 1.0, "uniform"),
    ):
        assert isinstance(dimension, Real)
        assert dimension.low == low
        assert dimension.high == high
        assert dimension.prior == prior


def test_frozen_budgets_seeds_acquisition_thresholds_and_washout() -> None:
    aware = SearchSettings.frozen(PARAMETER_AWARE)
    baseline = SearchSettings.frozen(ORDINARY_BASELINE)

    assert aware.n_calls == baseline.n_calls == BAYESIAN_CALLS_PER_MODEL == 40
    assert aware.n_initial_calls == baseline.n_initial_calls == INITIAL_RANDOM_CALLS == 10
    assert aware.acquisition_function == baseline.acquisition_function == "EI"
    assert aware.optimizer_seed == 2026
    assert baseline.optimizer_seed == 2027
    assert aware.candidate_model_seed == baseline.candidate_model_seed == 42
    assert TRAINING_WASHOUT == 2_000
    assert VALID_PREDICTION_THRESHOLD == 0.4
    assert DIVERGENCE_THRESHOLD == 5.0
    assert COLLAPSE_STD_RATIO_THRESHOLD == 0.05


def test_skopt_numpy_scalars_are_canonicalized_for_strict_json() -> None:
    point = [
        np.int64(100),
        np.float64(0.1),
        np.float64(0.2),
        np.float64(0.8),
        np.float64(1.0e-6),
        np.float64(0.5),
    ]

    parameters = optimisation.point_to_hyperparameters(point)
    stored_point = optimisation.hyperparameters_to_point(parameters)

    json.dumps(stored_point, allow_nan=False)
    assert type(stored_point[0]) is int
    assert all(type(value) is float for value in stored_point[1:])


def test_model_dimensions_are_fair_and_output_is_always_three() -> None:
    parameters = optimisation.point_to_hyperparameters(POINTS[0])
    aware = model_config(parameters, PARAMETER_AWARE, 42)
    baseline = model_config(parameters, ORDINARY_BASELINE, 42)

    assert aware.input_dimension == 4
    assert baseline.input_dimension == 3
    assert aware.output_dimension == baseline.output_dimension == 3
    assert aware.bias_scaling == baseline.bias_scaling == 0.1
    assert not aware.regularise_bias
    assert not baseline.regularise_bias


def test_objective_is_hand_calculated_equal_mean_of_nine_windows() -> None:
    values = iter(range(1, 10))
    rollouts = []
    for current in TRAIN_CURRENTS:
        for window in (1, 2, 3):
            rollouts.append(
                {
                    "current": current,
                    "window": window,
                    "objective_nrmse": float(next(values)),
                }
            )

    assert rollout_objective(rollouts) == 5.0


def test_objective_rejects_missing_or_duplicated_window() -> None:
    rollouts = _rollouts(0.1)
    with pytest.raises(ValueError, match="exactly"):
        rollout_objective(rollouts[:-1])
    duplicated = rollouts[:-1] + [rollouts[0]]
    with pytest.raises(ValueError, match="exactly"):
        rollout_objective(duplicated)


def test_preparation_uses_fitting_only_and_baseline_has_no_current_input() -> None:
    prepared = _small_prepared()
    original = copy.deepcopy(prepared)

    aware = prepare_model_data(
        prepared, PARAMETER_AWARE, enforce_frozen_lengths=False
    )
    baseline = prepare_model_data(
        prepared, ORDINARY_BASELINE, enforce_frozen_lengths=False
    )

    assert aware.input_dimension == 4
    assert baseline.input_dimension == 3
    assert all(item.inputs.shape[1] == 4 for item in aware.training_sequences)
    assert all(item.inputs.shape[1] == 3 for item in baseline.training_sequences)
    assert all(case.current_values is not None for case in aware.validation_cases)
    assert all(case.current_values is None for case in baseline.validation_cases)
    expected_states = np.concatenate(
        [item.fitting.inputs[:, :3] for item in prepared]
    )
    np.testing.assert_allclose(
        aware.scalers.state.mean, np.mean(expected_states, axis=0)
    )
    for before, after in zip(original, prepared):
        np.testing.assert_array_equal(before.fitting.inputs, after.fitting.inputs)


def test_shared_loader_is_called_once_and_no_benchmark_loader_is_needed() -> None:
    calls = 0

    def loader(*, include_current: bool):
        nonlocal calls
        calls += 1
        assert include_current
        return _small_prepared()

    data = optimisation.prepare_both_model_data(
        loader, enforce_frozen_lengths=False
    )

    assert calls == 1
    assert set(data) == {PARAMETER_AWARE, ORDINARY_BASELINE}


class RecordingModel:
    def __init__(self, input_dimension: int, *, nonfinite_step: int | None = None):
        self.input_dimension = input_dimension
        self.nonfinite_step = nonfinite_step
        self.reset_count = 0
        self.warmups: list[np.ndarray] = []
        self.inputs: list[np.ndarray] = []

    def reset_reservoir(self) -> None:
        self.reset_count += 1

    def teacher_forced_warmup(
        self, inputs: np.ndarray, *, reset: bool
    ) -> np.ndarray:
        assert not reset
        self.warmups.append(np.asarray(inputs).copy())
        return np.zeros(2)

    def predict_one_step(self, input_value: np.ndarray) -> np.ndarray:
        self.inputs.append(np.asarray(input_value).copy())
        step = len(self.inputs) - 1
        if self.nonfinite_step == step:
            return np.full(3, np.nan)
        return np.asarray(input_value[:3]) + 1.0


def test_rollout_resets_warms_exactly_and_recursively_feeds_predictions() -> None:
    case = ValidationCase(
        current=3.20,
        window=1,
        warmup_inputs=np.full((2_000, 4), 7.0),
        initial_state=np.array([0.0, 1.0, 2.0]),
        current_values=np.full(8_000, -0.25),
        targets_physical=np.full((8_000, 3), 999.0),
        warmup_range=(40_000, 42_000),
        scored_range=(42_000, 50_000),
    )
    model = RecordingModel(4)

    predictions, failure_step, _ = optimisation._recursive_case_rollout(
        model, case, PARAMETER_AWARE
    )

    assert model.reset_count == 1
    assert model.warmups[0].shape == (2_000, 4)
    assert predictions.shape == (8_000, 3)
    np.testing.assert_array_equal(model.inputs[0], [0.0, 1.0, 2.0, -0.25])
    np.testing.assert_array_equal(model.inputs[1], [1.0, 2.0, 3.0, -0.25])
    assert failure_step is None
    assert not np.any(model.inputs[1][:3] == 999.0)


def test_baseline_rollout_never_receives_current() -> None:
    case = _small_cases(ORDINARY_BASELINE)[0]
    model = RecordingModel(3)

    optimisation._recursive_case_rollout(model, case, ORDINARY_BASELINE)

    assert all(input_value.shape == (3,) for input_value in model.inputs)


def test_nonfinite_candidate_rollouts_receive_frozen_failure_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[RecordingModel] = []

    class FakeESN(RecordingModel):
        def __init__(self, config):
            super().__init__(config.input_dimension, nonfinite_step=1)
            instances.append(self)

        def fit(self, sequences, *, washout):
            assert len(sequences) == 3
            assert washout == 2_000
            return self

        def reset_reservoir(self) -> None:
            super().reset_reservoir()
            self.inputs.clear()

    monkeypatch.setattr(optimisation, "EchoStateNetwork", FakeESN)
    training = tuple(
        TrainingSequence(np.zeros((2, 4)), np.zeros((2, 3)))
        for _ in TRAIN_CURRENTS
    )
    data = ModelOptimisationData(
        PARAMETER_AWARE,
        4,
        training,
        _small_cases(PARAMETER_AWARE),
        _identity_scalers(),
    )

    result = RealCandidateEvaluator(data)(
        optimisation.point_to_hyperparameters(POINTS[0]), 42
    )

    assert len(result["rollouts"]) == 9
    assert result["objective"] == NONFINITE_FAILURE_SCORE
    assert all(
        item["objective_nrmse"] == NONFINITE_FAILURE_SCORE
        for item in result["rollouts"]
    )
    assert all(item["failure_step"] == 1 for item in result["rollouts"])
    assert instances[0].reset_count == 9


def test_finite_prediction_overflow_receives_frozen_failure_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OverflowESN(RecordingModel):
        def __init__(self, config):
            super().__init__(config.input_dimension)

        def fit(self, sequences, *, washout):
            return self

        def reset_reservoir(self) -> None:
            super().reset_reservoir()
            self.inputs.clear()

        def predict_one_step(self, input_value: np.ndarray) -> np.ndarray:
            self.inputs.append(np.asarray(input_value).copy())
            return np.full(3, 1.0e308)

    monkeypatch.setattr(optimisation, "EchoStateNetwork", OverflowESN)
    training = tuple(
        TrainingSequence(np.zeros((2, 4)), np.zeros((2, 3)))
        for _ in TRAIN_CURRENTS
    )
    data = ModelOptimisationData(
        PARAMETER_AWARE,
        4,
        training,
        _small_cases(PARAMETER_AWARE),
        _identity_scalers(),
    )

    result = RealCandidateEvaluator(data)(
        optimisation.point_to_hyperparameters(POINTS[0]), 42
    )

    assert result["objective"] == NONFINITE_FAILURE_SCORE
    assert all(item["failure_step"] == 0 for item in result["rollouts"])
    assert all(item["metrics"]["diverged"] for item in result["rollouts"])
    assert all(
        item["metrics"]["divergence_reason"]
        == "normalised_error_threshold_reached"
        for item in result["rollouts"]
    )
    assert all(
        item["failure_reason"] == "non_finite_normalised_error"
        for item in result["rollouts"]
    )
    json.dumps(result, allow_nan=False)


def test_checkpoint_json_is_strict_deterministic_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    payload = {"z": 1, "a": {"value": 2.5}}

    atomic_write_json(path, payload)
    first = path.read_bytes()
    atomic_write_json(path, payload)
    second = path.read_bytes()

    assert first == second
    assert load_strict_json(path) == payload
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError):
        atomic_write_json(path, {"bad": np.nan})


def test_resume_matches_uninterrupted_history(tmp_path: Path) -> None:
    settings = _settings(PARAMETER_AWARE, n_calls=5)
    metadata = _metadata(PARAMETER_AWARE, settings)
    uninterrupted_path = tmp_path / "uninterrupted.json"
    resumed_path = tmp_path / "resumed.json"

    uninterrupted = run_bayesian_search(
        checkpoint_path=uninterrupted_path,
        model_type=PARAMETER_AWARE,
        evaluator=_evaluation,
        metadata=metadata,
        settings=settings,
        resume=False,
        optimizer_factory=_optimizer_factory,
        clock=lambda: 0.0,
    )

    calls = 0

    def interrupted(parameters: dict, seed: int) -> dict:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("synthetic interruption")
        return _evaluation(parameters, seed)

    with pytest.raises(RuntimeError, match="interruption"):
        run_bayesian_search(
            checkpoint_path=resumed_path,
            model_type=PARAMETER_AWARE,
            evaluator=interrupted,
            metadata=metadata,
            settings=settings,
            resume=False,
            optimizer_factory=_optimizer_factory,
            clock=lambda: 0.0,
        )

    resumed_evaluations: list[str] = []

    def resumed_evaluator(parameters: dict, seed: int) -> dict:
        resumed_evaluations.append(
            optimisation.serialized_hyperparameters(parameters)
        )
        return _evaluation(parameters, seed)

    resumed = run_bayesian_search(
        checkpoint_path=resumed_path,
        model_type=PARAMETER_AWARE,
        evaluator=resumed_evaluator,
        metadata=metadata,
        settings=settings,
        resume=True,
        optimizer_factory=_optimizer_factory,
        clock=lambda: 0.0,
    )

    assert resumed["trials"] == uninterrupted["trials"]
    assert len(resumed_evaluations) == 3
    assert resumed_evaluations[0] == optimisation.serialized_hyperparameters(
        optimisation.point_to_hyperparameters(POINTS[2])
    )


def _real_test_settings(n_calls: int) -> SearchSettings:
    return SearchSettings(
        n_calls=n_calls,
        n_initial_calls=min(3, n_calls),
        acquisition_function=ACQUISITION_FUNCTION,
        optimizer_seed=SEARCH_SEEDS[PARAMETER_AWARE],
        candidate_model_seed=CANDIDATE_MODEL_SEED,
    )


@pytest.mark.parametrize(
    "completed_before_interruption",
    (1, 4),
    ids=("initial-random-phase", "after-initial-random-phase"),
)
def test_real_skopt_resume_exactly_matches_uninterrupted_run(
    tmp_path: Path,
    completed_before_interruption: int,
) -> None:
    settings = _real_test_settings(n_calls=6)
    metadata = _metadata(PARAMETER_AWARE, settings)
    uninterrupted_path = (
        tmp_path / f"uninterrupted-{completed_before_interruption}.json"
    )
    resumed_path = tmp_path / f"resumed-{completed_before_interruption}.json"

    uninterrupted = run_bayesian_search(
        checkpoint_path=uninterrupted_path,
        model_type=PARAMETER_AWARE,
        evaluator=_real_skopt_evaluation,
        metadata=metadata,
        settings=settings,
        resume=False,
        clock=lambda: 0.0,
    )

    interrupted_calls = 0

    def interrupting_evaluator(parameters: dict, seed: int) -> dict:
        nonlocal interrupted_calls
        interrupted_calls += 1
        if interrupted_calls == completed_before_interruption + 1:
            raise RuntimeError("synthetic real-optimizer interruption")
        return _real_skopt_evaluation(parameters, seed)

    with pytest.raises(RuntimeError, match="interruption"):
        run_bayesian_search(
            checkpoint_path=resumed_path,
            model_type=PARAMETER_AWARE,
            evaluator=interrupting_evaluator,
            metadata=metadata,
            settings=settings,
            resume=False,
            clock=lambda: 0.0,
        )

    resumed_evaluations: list[str] = []

    def resumed_evaluator(parameters: dict, seed: int) -> dict:
        resumed_evaluations.append(
            optimisation.serialized_hyperparameters(parameters)
        )
        return _real_skopt_evaluation(parameters, seed)

    resumed = run_bayesian_search(
        checkpoint_path=resumed_path,
        model_type=PARAMETER_AWARE,
        evaluator=resumed_evaluator,
        metadata=metadata,
        settings=settings,
        resume=True,
        clock=lambda: 0.0,
    )

    uninterrupted_points = [trial["point"] for trial in uninterrupted["trials"]]
    resumed_points = [trial["point"] for trial in resumed["trials"]]
    uninterrupted_objectives = [
        trial["objective"] for trial in uninterrupted["trials"]
    ]
    resumed_objectives = [trial["objective"] for trial in resumed["trials"]]
    assert resumed_points == uninterrupted_points
    assert resumed_objectives == uninterrupted_objectives
    assert len(resumed_evaluations) == settings.n_calls - completed_before_interruption
    assert resumed_evaluations[0] == optimisation.serialized_hyperparameters(
        uninterrupted["trials"][completed_before_interruption]["hyperparameters"]
    )

    best_uninterrupted = min(
        uninterrupted["trials"],
        key=lambda trial: (
            trial["objective"],
            optimisation.serialized_hyperparameters(trial["hyperparameters"]),
        ),
    )
    best_resumed = min(
        resumed["trials"],
        key=lambda trial: (
            trial["objective"],
            optimisation.serialized_hyperparameters(trial["hyperparameters"]),
        ),
    )
    assert best_resumed["point"] == best_uninterrupted["point"]
    assert best_resumed["objective"] == best_uninterrupted["objective"]


def test_real_skopt_replay_rejects_an_altered_saved_point(
    tmp_path: Path,
) -> None:
    settings = _real_test_settings(n_calls=3)
    metadata = _metadata(PARAMETER_AWARE, settings)
    path = tmp_path / "altered-point.json"
    calls = 0

    def interrupting_evaluator(parameters: dict, seed: int) -> dict:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return _real_skopt_evaluation(parameters, seed)

    with pytest.raises(RuntimeError, match="interruption"):
        run_bayesian_search(
            checkpoint_path=path,
            model_type=PARAMETER_AWARE,
            evaluator=interrupting_evaluator,
            metadata=metadata,
            settings=settings,
            resume=False,
            clock=lambda: 0.0,
        )

    checkpoint = load_strict_json(path)
    checkpoint["trials"][0]["point"][1] = 0.123456789
    atomic_write_json(path, checkpoint)
    resumed_calls = 0

    def forbidden_evaluator(parameters: dict, seed: int) -> dict:
        nonlocal resumed_calls
        resumed_calls += 1
        return _real_skopt_evaluation(parameters, seed)

    with pytest.raises(
        optimisation.OptimizerReplayMismatchError,
        match="model_type=parameter_aware.*trial_number=1.*differing_fields",
    ) as error:
        run_bayesian_search(
            checkpoint_path=path,
            model_type=PARAMETER_AWARE,
            evaluator=forbidden_evaluator,
            metadata=metadata,
            settings=settings,
            resume=True,
            clock=lambda: 0.0,
        )
    assert "package_versions" in str(error.value)
    assert resumed_calls == 0


@pytest.mark.parametrize(
    "conflict",
    ("optimizer_settings", "optimizer_seed", "search_space", "software_versions"),
)
def test_real_skopt_resume_rejects_optimizer_contract_conflicts(
    tmp_path: Path,
    conflict: str,
) -> None:
    settings = _real_test_settings(n_calls=1)
    metadata = _metadata(PARAMETER_AWARE, settings)
    path = tmp_path / f"conflict-{conflict}.json"
    run_bayesian_search(
        checkpoint_path=path,
        model_type=PARAMETER_AWARE,
        evaluator=_real_skopt_evaluation,
        metadata=metadata,
        settings=settings,
        resume=False,
        clock=lambda: 0.0,
    )
    checkpoint = load_strict_json(path)
    if conflict == "optimizer_settings":
        checkpoint["metadata"]["optimizer_settings"][
            "acquisition_optimizer"
        ] = "sampling"
    elif conflict == "optimizer_seed":
        checkpoint["metadata"]["optimizer_settings"]["optimizer_seed"] += 1
    elif conflict == "search_space":
        checkpoint["metadata"]["search_space"][
            "reservoir_connectivity"
        ]["high"] = 0.99
    else:
        checkpoint["metadata"]["software_versions"]["scikit_optimize"] = "0.0"
    atomic_write_json(path, checkpoint)

    with pytest.raises(ValueError, match="metadata conflicts"):
        run_bayesian_search(
            checkpoint_path=path,
            model_type=PARAMETER_AWARE,
            evaluator=_real_skopt_evaluation,
            metadata=metadata,
            settings=settings,
            resume=True,
            clock=lambda: 0.0,
        )


def test_conflicting_checkpoint_metadata_is_rejected(tmp_path: Path) -> None:
    settings = _settings(PARAMETER_AWARE, n_calls=2)
    metadata = _metadata(PARAMETER_AWARE, settings)
    path = tmp_path / "history.json"
    run_bayesian_search(
        checkpoint_path=path,
        model_type=PARAMETER_AWARE,
        evaluator=_evaluation,
        metadata=metadata,
        settings=settings,
        resume=False,
        optimizer_factory=_optimizer_factory,
        clock=lambda: 0.0,
    )
    conflicting = copy.deepcopy(metadata)
    conflicting["dataset_hashes"]["training.npz"] = "b" * 64

    with pytest.raises(ValueError, match="conflicts"):
        run_bayesian_search(
            checkpoint_path=path,
            model_type=PARAMETER_AWARE,
            evaluator=_evaluation,
            metadata=conflicting,
            settings=settings,
            resume=True,
            optimizer_factory=_optimizer_factory,
            clock=lambda: 0.0,
        )


def test_completed_trials_are_not_re_evaluated_on_resume(tmp_path: Path) -> None:
    settings = _settings(PARAMETER_AWARE, n_calls=3)
    metadata = _metadata(PARAMETER_AWARE, settings)
    path = tmp_path / "history.json"
    first_calls = 0

    def evaluator(parameters: dict, seed: int) -> dict:
        nonlocal first_calls
        first_calls += 1
        return _evaluation(parameters, seed)

    run_bayesian_search(
        checkpoint_path=path,
        model_type=PARAMETER_AWARE,
        evaluator=evaluator,
        metadata=metadata,
        settings=settings,
        resume=False,
        optimizer_factory=_optimizer_factory,
        clock=lambda: 0.0,
    )
    second_calls = 0

    def forbidden_repeat(parameters: dict, seed: int) -> dict:
        nonlocal second_calls
        second_calls += 1
        return _evaluation(parameters, seed)

    run_bayesian_search(
        checkpoint_path=path,
        model_type=PARAMETER_AWARE,
        evaluator=forbidden_repeat,
        metadata=metadata,
        settings=settings,
        resume=True,
        optimizer_factory=_optimizer_factory,
        clock=lambda: 0.0,
    )

    assert first_calls == 3
    assert second_calls == 0


def _complete_history(
    tmp_path: Path, model_type: str
) -> dict:
    settings = _settings(model_type, n_calls=5)
    path = tmp_path / f"{model_type}.json"
    run_bayesian_search(
        checkpoint_path=path,
        model_type=model_type,
        evaluator=_evaluation,
        metadata=_metadata(model_type, settings),
        settings=settings,
        resume=False,
        optimizer_factory=_optimizer_factory,
        clock=lambda: 0.0,
    )
    return run_robust_confirmation(
        checkpoint_path=path,
        evaluator=_evaluation,
        final_seeds=(42, 123, 456, 789, 2026),
        top_count=5,
    )


def test_top_five_confirmation_uses_all_five_seeds_and_reuses_seed42(
    tmp_path: Path,
) -> None:
    history = _complete_history(tmp_path, PARAMETER_AWARE)

    assert len(history["robust_confirmations"]) == 5
    for confirmation in history["robust_confirmations"]:
        assert [item["model_seed"] for item in confirmation["seed_results"]] == [
            42,
            123,
            456,
            789,
            2026,
        ]
        assert confirmation["seed_results"][0]["reused_seed42_search_result"]
        assert confirmation["aggregate"]["robust_seed_count"] == 5
        assert confirmation["aggregate"]["robust_rollout_count"] == 45


def test_robust_confirmation_accepts_an_in_progress_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / f"{PARAMETER_AWARE}.json"
    history = _complete_history(tmp_path, PARAMETER_AWARE)
    history["status"] = "robust_confirmation_in_progress"
    atomic_write_json(path, history)

    resumed = run_robust_confirmation(
        checkpoint_path=path, evaluator=_evaluation
    )
    assert resumed["status"] == "complete"


def test_tie_breaking_is_deterministic_in_frozen_order() -> None:
    base = {
        "hyperparameters": {"reservoir_size": 100},
        "aggregate": {
            "mean_objective_nrmse": 1.0,
            "worst_current_mean_nrmse": 2.0,
            "mean_valid_prediction_steps": 10.0,
        },
    }
    lower_worst = copy.deepcopy(base)
    lower_worst["aggregate"]["worst_current_mean_nrmse"] = 1.5
    higher_vpt = copy.deepcopy(lower_worst)
    higher_vpt["aggregate"]["mean_valid_prediction_steps"] = 20.0

    assert robust_tie_key(lower_worst) < robust_tie_key(base)
    assert robust_tie_key(higher_vpt) < robust_tie_key(lower_worst)


def test_selection_requires_and_records_equal_budgets_and_full_diagnostics(
    tmp_path: Path,
) -> None:
    aware = _complete_history(tmp_path, PARAMETER_AWARE)
    baseline = _complete_history(tmp_path, ORDINARY_BASELINE)

    selection = build_selection_artifact(aware, baseline)

    assert selection["label"] == "VALIDATION-SELECTED — BENCHMARKS NOT OPENED"
    assert selection["equal_bayesian_budgets"]
    assert not selection["data_access"]["benchmark_results_present"]
    assert set(selection["models"]) == {PARAMETER_AWARE, ORDINARY_BASELINE}
    for result in selection["models"].values():
        assert len(result["top_five_seed42_candidates"]) == 5
        assert len(result["five_seed_robust_confirmation_results"]) == 5
        assert "best_configuration" in result
        assert "best_robust_aggregate" in result


def test_unequal_model_budgets_are_rejected(tmp_path: Path) -> None:
    aware = _complete_history(tmp_path, PARAMETER_AWARE)
    baseline = _complete_history(tmp_path, ORDINARY_BASELINE)
    baseline["metadata"]["optimizer_settings"]["n_calls"] = 4

    with pytest.raises(ValueError, match="equal"):
        build_selection_artifact(aware, baseline)
