"""Leakage-safe Bayesian model selection for Chapter 2.

Only training-current fitting transitions and the nine predefined validation
windows enter this module. Held-out, unseen-current, and continuous-benchmark
loaders are intentionally not imported.
"""

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import asdict, dataclass
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from skopt import Optimizer
from skopt.space import Categorical, Real

try:
    from .esn_config import (
        CHAPTER2_ROOT,
        FINAL_SEEDS,
        FIXED_DATASETS,
        OUTPUT_DIMENSION,
        TRAIN_CURRENTS,
        ESNModelConfig,
    )
    from .esn_data import (
        NumpyStandardScaler,
        PreparedOptimisationTrajectory,
        StateCurrentScalers,
        file_sha256,
        load_optimisation_data,
        scale_one_step_pairs,
    )
    from .esn_metrics import (
        COLLAPSE_STD_RATIO_THRESHOLD,
        evaluate_rollout,
    )
    from .esn_model import EchoStateNetwork, TrainingSequence
except ImportError:  # Support direct execution from the chapter2 directory.
    from esn_config import (
        CHAPTER2_ROOT,
        FINAL_SEEDS,
        FIXED_DATASETS,
        OUTPUT_DIMENSION,
        TRAIN_CURRENTS,
        ESNModelConfig,
    )
    from esn_data import (
        NumpyStandardScaler,
        PreparedOptimisationTrajectory,
        StateCurrentScalers,
        file_sha256,
        load_optimisation_data,
        scale_one_step_pairs,
    )
    from esn_metrics import COLLAPSE_STD_RATIO_THRESHOLD, evaluate_rollout
    from esn_model import EchoStateNetwork, TrainingSequence


FloatArray = NDArray[np.float64]
PARAMETER_AWARE = "parameter_aware"
ORDINARY_BASELINE = "ordinary_baseline"
MODEL_TYPES = (PARAMETER_AWARE, ORDINARY_BASELINE)
MODEL_INPUT_DIMENSIONS = {PARAMETER_AWARE: 4, ORDINARY_BASELINE: 3}

BAYESIAN_CALLS_PER_MODEL = 40
INITIAL_RANDOM_CALLS = 10
ACQUISITION_FUNCTION = "EI"
SEARCH_SEEDS = {PARAMETER_AWARE: 2026, ORDINARY_BASELINE: 2027}
CANDIDATE_MODEL_SEED = 42
TRAINING_WASHOUT = 2_000
VALID_PREDICTION_THRESHOLD = 0.4
DIVERGENCE_THRESHOLD = 5.0
NONFINITE_FAILURE_SCORE = 1_000_000.0
TOP_CANDIDATE_COUNT = 5

SEARCH_PARAMETER_ORDER = (
    "reservoir_size",
    "reservoir_connectivity",
    "input_scaling",
    "spectral_radius",
    "ridge_regularisation",
    "leak_rate",
)
SEARCH_SPACE_DEFINITION: dict[str, dict[str, Any]] = {
    "reservoir_size": {
        "type": "categorical",
        "values": [100, 200, 300],
    },
    "reservoir_connectivity": {
        "type": "real",
        "low": 0.01,
        "high": 1.0,
        "prior": "uniform",
    },
    "input_scaling": {
        "type": "real",
        "low": 0.01,
        "high": 3.0,
        "prior": "uniform",
    },
    "spectral_radius": {
        "type": "real",
        "low": 0.01,
        "high": 3.0,
        "prior": "uniform",
    },
    "ridge_regularisation": {
        "type": "real",
        "low": 1.0e-10,
        "high": 1.0e-2,
        "prior": "log-uniform",
    },
    "leak_rate": {
        "type": "real",
        "low": 0.01,
        "high": 1.0,
        "prior": "uniform",
    },
}
OBJECTIVE_DEFINITION = (
    "Arithmetic mean of physical-unit all-state NRMSE across all nine equally "
    "weighted training-current validation rollouts."
)
SELECTION_RULES = (
    "Lower robust mean NRMSE",
    "Lower worst-current mean NRMSE",
    "Higher mean valid-prediction steps",
    "Lexicographically ordered serialized hyperparameters",
)
CHECKPOINT_SCHEMA = "chapter2_step7_bayesian_optimisation_v2"
SELECTION_SCHEMA = "chapter2_step7_validation_selection_v1"
RESULT_LABEL = "VALIDATION-SELECTED — BENCHMARKS NOT OPENED"
# These values spell out every public Optimizer constructor setting used by
# Step 7. Keeping defaults explicit makes checkpoint compatibility auditable
# and prevents an upstream default change from silently changing a resumed run.
OPTIMIZER_BASE_ESTIMATOR = "GP"
OPTIMIZER_INITIAL_POINT_GENERATOR = "random"
OPTIMIZER_N_JOBS = 1
OPTIMIZER_ACQUISITION_OPTIMIZER = "auto"
OPTIMIZER_RESOLVED_ACQUISITION_OPTIMIZER = "lbfgs"
OPTIMIZER_MODEL_QUEUE_SIZE = None
OPTIMIZER_SPACE_CONSTRAINT = None
OPTIMIZER_ACQUISITION_FUNCTION_KWARGS = None
OPTIMIZER_ACQUISITION_OPTIMIZER_KWARGS = None
OPTIMIZER_AVOID_DUPLICATES = True
FLOAT_REPLAY_REL_TOLERANCE = 1.0e-15
FLOAT_REPLAY_ABS_TOLERANCE = 0.0


class OptimizerReplayMismatchError(ValueError):
    """Raised when a checkpoint is not the history of the fresh optimiser."""



@dataclass(frozen=True)
class SearchSettings:
    """Deterministic scikit-optimize settings for one model type."""

    n_calls: int
    n_initial_calls: int
    acquisition_function: str
    optimizer_seed: int
    candidate_model_seed: int

    @classmethod
    def frozen(cls, model_type: str) -> "SearchSettings":
        _validate_model_type(model_type)
        return cls(
            BAYESIAN_CALLS_PER_MODEL,
            INITIAL_RANDOM_CALLS,
            ACQUISITION_FUNCTION,
            SEARCH_SEEDS[model_type],
            CANDIDATE_MODEL_SEED,
        )

    def __post_init__(self) -> None:
        if self.n_calls <= 0:
            raise ValueError("n_calls must be positive")
        if not 0 < self.n_initial_calls <= self.n_calls:
            raise ValueError("n_initial_calls must be in [1, n_calls]")
        if self.acquisition_function != ACQUISITION_FUNCTION:
            raise ValueError("acquisition_function must be EI")


@dataclass(frozen=True)
class ValidationCase:
    """One aligned warm-up and autonomous scored validation case."""

    current: float
    window: int
    warmup_inputs: FloatArray
    initial_state: FloatArray
    current_values: FloatArray | None
    targets_physical: FloatArray
    warmup_range: tuple[int, int]
    scored_range: tuple[int, int]


@dataclass(frozen=True)
class ModelOptimisationData:
    """Prepared fitting sequences and nine validation cases for one model."""

    model_type: str
    input_dimension: int
    training_sequences: tuple[TrainingSequence, ...]
    validation_cases: tuple[ValidationCase, ...]
    scalers: StateCurrentScalers


CandidateEvaluator = Callable[[dict[str, Any], int], dict[str, Any]]
OptimizerFactory = Callable[[str, SearchSettings], Any]


def _validate_model_type(model_type: str) -> None:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}")


def search_dimensions() -> list[Any]:
    """Return the frozen scikit-optimize dimensions in canonical order."""
    return [
        Categorical([100, 200, 300], name="reservoir_size"),
        Real(0.01, 1.0, prior="uniform", name="reservoir_connectivity"),
        Real(0.01, 3.0, prior="uniform", name="input_scaling"),
        Real(0.01, 3.0, prior="uniform", name="spectral_radius"),
        Real(
            1.0e-10,
            1.0e-2,
            prior="log-uniform",
            name="ridge_regularisation",
        ),
        Real(0.01, 1.0, prior="uniform", name="leak_rate"),
    ]


def point_to_hyperparameters(point: Sequence[Any]) -> dict[str, Any]:
    if len(point) != len(SEARCH_PARAMETER_ORDER):
        raise ValueError("Bayesian point has the wrong dimension")
    values = dict(zip(SEARCH_PARAMETER_ORDER, point))
    return {
        "reservoir_size": int(values["reservoir_size"]),
        "reservoir_connectivity": float(values["reservoir_connectivity"]),
        "input_scaling": float(values["input_scaling"]),
        "spectral_radius": float(values["spectral_radius"]),
        "ridge_regularisation": float(values["ridge_regularisation"]),
        "leak_rate": float(values["leak_rate"]),
    }


def hyperparameters_to_point(parameters: Mapping[str, Any]) -> list[Any]:
    return [parameters[name] for name in SEARCH_PARAMETER_ORDER]


def serialized_hyperparameters(parameters: Mapping[str, Any]) -> str:
    return json.dumps(
        {name: parameters[name] for name in sorted(parameters)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def make_optimizer(model_type: str, settings: SearchSettings) -> Optimizer:
    _validate_model_type(model_type)
    return Optimizer(
        dimensions=search_dimensions(),
        base_estimator=OPTIMIZER_BASE_ESTIMATOR,
        n_random_starts=None,
        n_initial_points=settings.n_initial_calls,
        initial_point_generator=OPTIMIZER_INITIAL_POINT_GENERATOR,
        n_jobs=OPTIMIZER_N_JOBS,
        acq_func=settings.acquisition_function,
        acq_optimizer=OPTIMIZER_ACQUISITION_OPTIMIZER,
        random_state=settings.optimizer_seed,
        model_queue_size=OPTIMIZER_MODEL_QUEUE_SIZE,
        space_constraint=OPTIMIZER_SPACE_CONSTRAINT,
        acq_func_kwargs=OPTIMIZER_ACQUISITION_FUNCTION_KWARGS,
        acq_optimizer_kwargs=OPTIMIZER_ACQUISITION_OPTIMIZER_KWARGS,
        avoid_duplicates=OPTIMIZER_AVOID_DUPLICATES,
    )


def optimizer_settings_record(settings: SearchSettings) -> dict[str, Any]:
    """Return the complete public Optimizer construction contract."""
    return {
        **asdict(settings),
        "base_estimator": OPTIMIZER_BASE_ESTIMATOR,
        "n_random_starts": None,
        "initial_point_generator": OPTIMIZER_INITIAL_POINT_GENERATOR,
        "n_jobs": OPTIMIZER_N_JOBS,
        "acquisition_optimizer": OPTIMIZER_ACQUISITION_OPTIMIZER,
        "resolved_acquisition_optimizer": (
            OPTIMIZER_RESOLVED_ACQUISITION_OPTIMIZER
        ),
        "model_queue_size": OPTIMIZER_MODEL_QUEUE_SIZE,
        "space_constraint": OPTIMIZER_SPACE_CONSTRAINT,
        "acquisition_function_kwargs": (
            OPTIMIZER_ACQUISITION_FUNCTION_KWARGS
        ),
        "acquisition_optimizer_kwargs": (
            OPTIMIZER_ACQUISITION_OPTIMIZER_KWARGS
        ),
        "avoid_duplicates": OPTIMIZER_AVOID_DUPLICATES,
    }


def software_versions() -> dict[str, str]:
    """Return versions that can affect deterministic optimiser replay."""
    from importlib.metadata import version

    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
        "scikit_optimize": version("scikit-optimize"),
    }


def optimizer_runtime_record() -> dict[str, str | None]:
    """Record the numerical runtime that can affect GP replay."""
    return {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def _fit_scalers_from_permitted_views(
    prepared: Sequence[PreparedOptimisationTrajectory],
) -> StateCurrentScalers:
    fitting_states = np.concatenate(
        [item.fitting.inputs[:, :3] for item in prepared],
        axis=0,
    )
    fitting_currents = np.concatenate(
        [item.fitting.inputs[:, 3] for item in prepared],
        axis=0,
    ).reshape(-1, 1)
    return StateCurrentScalers(
        NumpyStandardScaler.fit(fitting_states),
        NumpyStandardScaler.fit(fitting_currents),
    )


def prepare_model_data(
    prepared: Sequence[PreparedOptimisationTrajectory],
    model_type: str,
    *,
    enforce_frozen_lengths: bool = True,
) -> ModelOptimisationData:
    """Create one model bundle from permitted views only."""
    _validate_model_type(model_type)
    items = tuple(prepared)
    if tuple(item.current for item in items) != TRAIN_CURRENTS:
        raise ValueError("optimisation data must contain all training currents in order")
    if any(item.fitting.inputs.shape[1] != 4 for item in items):
        raise ValueError("shared prepared data must contain parameter-aware inputs")
    if enforce_frozen_lengths and any(len(item.fitting) != 40_000 for item in items):
        raise ValueError("each fitting trajectory must contain 40,000 transitions")

    scalers = _fit_scalers_from_permitted_views(items)
    input_dimension = MODEL_INPUT_DIMENSIONS[model_type]
    training_sequences: list[TrainingSequence] = []
    validation_cases: list[ValidationCase] = []

    for item in items:
        scaled_fitting = scale_one_step_pairs(item.fitting, scalers)
        fitting_inputs = (
            scaled_fitting.inputs
            if model_type == PARAMETER_AWARE
            else scaled_fitting.inputs[:, :3]
        )
        training_sequences.append(
            TrainingSequence(fitting_inputs, scaled_fitting.targets)
        )

        if enforce_frozen_lengths and len(item.validation_windows) != 3:
            raise ValueError("each training current must expose three windows")
        for view in item.validation_windows:
            scaled_warmup = scale_one_step_pairs(view.warmup, scalers)
            scaled_scored = scale_one_step_pairs(view.scored, scalers)
            if enforce_frozen_lengths:
                if len(scaled_warmup) != 2_000:
                    raise ValueError("validation warm-up must be 2,000 transitions")
                if len(scaled_scored) != 8_000:
                    raise ValueError("validation scoring must be 8,000 transitions")
            if not np.array_equal(
                scaled_warmup.targets[-1], scaled_scored.inputs[0, :3]
            ):
                raise ValueError("validation warm-up and scored states are misaligned")
            warmup_inputs = (
                scaled_warmup.inputs
                if model_type == PARAMETER_AWARE
                else scaled_warmup.inputs[:, :3]
            )
            current_values = (
                scaled_scored.inputs[:, 3].copy()
                if model_type == PARAMETER_AWARE
                else None
            )
            validation_cases.append(
                ValidationCase(
                    current=item.current,
                    window=view.definition.number,
                    warmup_inputs=np.asarray(warmup_inputs, dtype=float).copy(),
                    initial_state=scaled_warmup.targets[-1].copy(),
                    current_values=current_values,
                    targets_physical=scalers.inverse_states(
                        scaled_scored.targets
                    ),
                    warmup_range=(
                        int(view.warmup.transition_indices[0]),
                        int(view.warmup.transition_indices[-1]) + 1,
                    ),
                    scored_range=(
                        int(view.scored.transition_indices[0]),
                        int(view.scored.transition_indices[-1]) + 1,
                    ),
                )
            )

    if enforce_frozen_lengths:
        expected = {(current, window) for current in TRAIN_CURRENTS for window in (1, 2, 3)}
        actual = {(case.current, case.window) for case in validation_cases}
        if actual != expected:
            raise ValueError("prepared data must contain exactly nine validation cases")
    return ModelOptimisationData(
        model_type=model_type,
        input_dimension=input_dimension,
        training_sequences=tuple(training_sequences),
        validation_cases=tuple(validation_cases),
        scalers=scalers,
    )


def prepare_both_model_data(
    loader: Callable[..., Sequence[PreparedOptimisationTrajectory]] = (
        load_optimisation_data
    ),
    *,
    enforce_frozen_lengths: bool = True,
) -> dict[str, ModelOptimisationData]:
    """Load permitted views once and derive fair four- and three-input bundles."""
    prepared = tuple(loader(include_current=True))
    return {
        model_type: prepare_model_data(
            prepared,
            model_type,
            enforce_frozen_lengths=enforce_frozen_lengths,
        )
        for model_type in MODEL_TYPES
    }


def training_dataset_hashes() -> dict[str, str]:
    """Hash only the three training-current files used during Step 7."""
    records = [
        record for record in FIXED_DATASETS if record.current in TRAIN_CURRENTS
    ]
    return {
        str(record.path.relative_to(CHAPTER2_ROOT.parent)): file_sha256(record.path)
        for record in records
    }


def git_state() -> dict[str, Any]:
    """Return the current commit and porcelain dirty status when available."""
    root = CHAPTER2_ROOT.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status_short": []}
    return {"commit": commit, "dirty": bool(status), "status_short": status}


def preprocessing_record(data: ModelOptimisationData) -> dict[str, Any]:
    return {
        "state_mean": data.scalers.state.mean.tolist(),
        "state_scale": data.scalers.state.scale.tolist(),
        "current_mean": data.scalers.current.mean.tolist(),
        "current_scale": data.scalers.current.scale.tolist(),
        "fitted_from": (
            "state and current inputs from fitting transitions [0, 40000) "
            "of I=(1.67, 3.20, 3.50) only"
        ),
    }


def checkpoint_metadata(
    model_type: str,
    settings: SearchSettings,
    data: ModelOptimisationData,
    dataset_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "label": RESULT_LABEL,
        "model_type": model_type,
        "input_dimension": MODEL_INPUT_DIMENSIONS[model_type],
        "output_dimension": OUTPUT_DIMENSION,
        "search_space": SEARCH_SPACE_DEFINITION,
        "optimizer_settings": optimizer_settings_record(settings),
        "software_versions": software_versions(),
        "optimizer_runtime": optimizer_runtime_record(),
        "objective": OBJECTIVE_DEFINITION,
        "replay_verification": {
            "point_order": list(SEARCH_PARAMETER_ORDER),
            "integer_and_categorical_comparison": "exact",
            "float_relative_tolerance": FLOAT_REPLAY_REL_TOLERANCE,
            "float_absolute_tolerance": FLOAT_REPLAY_ABS_TOLERANCE,
            "tolerance_rationale": "strict JSON binary64 serialization safety",
        },
        "objective_diagnostic_exclusions": [
            "r2",
            "correlation",
            "valid_prediction_time",
            "prediction_collapse",
            "spikes",
            "bursts",
        ],
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
        "validation_case_count": 9,
        "warmup_transitions_per_window": 2_000,
        "scored_transitions_per_window": 8_000,
        "dataset_hashes": dict(dataset_hashes),
        "preprocessing": preprocessing_record(data),
        "data_access": {
            "held_out_requested": False,
            "held_out_constructed": False,
            "unseen_current_requested": False,
            "continuous_benchmark_requested": False,
        },
        "git": git_state(),
    }


def model_config(
    parameters: Mapping[str, Any],
    model_type: str,
    model_seed: int,
) -> ESNModelConfig:
    return ESNModelConfig(
        reservoir_size=int(parameters["reservoir_size"]),
        spectral_radius=float(parameters["spectral_radius"]),
        leak_rate=float(parameters["leak_rate"]),
        input_scaling=float(parameters["input_scaling"]),
        bias_scaling=0.1,
        reservoir_connectivity=float(parameters["reservoir_connectivity"]),
        ridge_regularisation=float(parameters["ridge_regularisation"]),
        seed=int(model_seed),
        input_dimension=MODEL_INPUT_DIMENSIONS[model_type],
        output_dimension=OUTPUT_DIMENSION,
        regularise_bias=False,
    )


def _recursive_case_rollout(
    model: EchoStateNetwork,
    case: ValidationCase,
    model_type: str,
) -> tuple[FloatArray, int | None, str | None]:
    """Run one scored region without accepting any future true state."""
    model.reset_reservoir()
    model.teacher_forced_warmup(case.warmup_inputs, reset=False)
    state = case.initial_state.copy()
    horizon = len(case.targets_physical)
    predictions = np.full((horizon, OUTPUT_DIMENSION), np.nan, dtype=float)

    for step in range(horizon):
        if model_type == PARAMETER_AWARE:
            assert case.current_values is not None
            input_value = np.concatenate((state, [case.current_values[step]]))
        else:
            if case.current_values is not None:
                raise ValueError("ordinary baseline must not receive current values")
            input_value = state
        prediction = model.predict_one_step(input_value)
        predictions[step] = prediction
        if not np.all(np.isfinite(prediction)):
            return predictions, step, "non_finite_prediction"
        state = prediction
    return predictions, None, None


def rollout_objective(rollouts: Sequence[Mapping[str, Any]]) -> float:
    """Return the equally weighted mean of exactly nine rollout scores."""
    items = tuple(rollouts)
    expected = {(current, window) for current in TRAIN_CURRENTS for window in (1, 2, 3)}
    actual = {(float(item["current"]), int(item["window"])) for item in items}
    if len(items) != 9 or actual != expected:
        raise ValueError("objective requires exactly three currents by three windows")
    values = np.asarray([float(item["objective_nrmse"]) for item in items])
    if not np.all(np.isfinite(values)):
        raise ValueError("rollout objective values must be finite")
    return float(np.mean(values))


def aggregate_rollouts(rollouts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = tuple(rollouts)
    scores = np.asarray([float(item["objective_nrmse"]) for item in items])
    current_values: dict[float, list[float]] = defaultdict(list)
    for item in items:
        current_values[float(item["current"])].append(
            float(item["objective_nrmse"])
        )
    current_means = {
        f"{current:.2f}": float(np.mean(values))
        for current, values in sorted(current_values.items())
    }
    finite_diagnostics = [
        float(item["metrics"]["nrmse_state"])
        for item in items
        if item["metrics"]["nrmse_state"] is not None
    ]
    return {
        "mean_objective_nrmse": float(np.mean(scores)),
        "std_objective_nrmse": float(np.std(scores, ddof=0)),
        "median_objective_nrmse": float(np.median(scores)),
        "worst_objective_nrmse": float(np.max(scores)),
        "worst_current_mean_nrmse": float(max(current_means.values())),
        "per_current_mean_nrmse": current_means,
        "finite_rollout_mean_nrmse": (
            float(np.mean(finite_diagnostics)) if finite_diagnostics else None
        ),
        "mean_valid_prediction_steps": float(
            np.mean([item["metrics"]["valid_prediction_steps"] for item in items])
        ),
        "divergence_rollout_count": int(
            sum(bool(item["metrics"]["diverged"]) for item in items)
        ),
        "collapse_rollout_count": int(
            sum(bool(item["metrics"]["prediction_collapse_any"]) for item in items)
        ),
    }


class RealCandidateEvaluator:
    """Fit one candidate and evaluate its nine permitted validation rollouts."""

    def __init__(self, data: ModelOptimisationData) -> None:
        self.data = data

    def __call__(
        self,
        parameters: dict[str, Any],
        model_seed: int,
    ) -> dict[str, Any]:
        model = EchoStateNetwork(
            model_config(parameters, self.data.model_type, model_seed)
        )
        model.fit(self.data.training_sequences, washout=TRAINING_WASHOUT)
        rollouts: list[dict[str, Any]] = []

        for case in self.data.validation_cases:
            predicted_scaled, failure_step, failure_reason = (
                _recursive_case_rollout(model, case, self.data.model_type)
            )
            with np.errstate(over="ignore", invalid="ignore"):
                predictions_physical = (
                    predicted_scaled * self.data.scalers.state.scale
                    + self.data.scalers.state.mean
                )
                normalised_difference = (
                    predictions_physical - case.targets_physical
                ) / self.data.scalers.state.scale
                pointwise_error = np.sqrt(
                    np.mean(np.square(normalised_difference), axis=1)
                )
            numerical_failure_rows = np.flatnonzero(
                ~np.all(np.isfinite(predictions_physical), axis=1)
                | ~np.isfinite(pointwise_error)
            )
            if failure_step is None and len(numerical_failure_rows):
                failure_step = int(numerical_failure_rows[0])
                if not np.all(np.isfinite(predictions_physical[failure_step])):
                    failure_reason = "non_finite_physical_prediction"
                else:
                    failure_reason = "non_finite_normalised_error"
            metrics = evaluate_rollout(
                predictions_physical,
                case.targets_physical,
                normalisation_scale=self.data.scalers.state.scale,
                dt=0.01,
                valid_prediction_threshold=VALID_PREDICTION_THRESHOLD,
                divergence_threshold=DIVERGENCE_THRESHOLD,
                collapse_std_ratio_threshold=COLLAPSE_STD_RATIO_THRESHOLD,
            ).to_dict()
            if metrics["nrmse_state"] is None and failure_step is None:
                failure_step = 0
                failure_reason = "non_finite_rollout_metric"
            nonfinite_failure = (
                failure_step is not None or metrics["nrmse_state"] is None
            )
            objective_nrmse = (
                NONFINITE_FAILURE_SCORE
                if nonfinite_failure
                else float(metrics["nrmse_state"])
            )
            rollouts.append(
                {
                    "current": case.current,
                    "window": case.window,
                    "warmup_range": list(case.warmup_range),
                    "scored_range": list(case.scored_range),
                    "model_seed": int(model_seed),
                    "objective_nrmse": objective_nrmse,
                    "nonfinite_failure": nonfinite_failure,
                    "failure_step": failure_step,
                    "failure_reason": failure_reason,
                    "metrics": metrics,
                }
            )

        objective = rollout_objective(rollouts)
        return {
            "objective": objective,
            "rollouts": rollouts,
            "aggregate": aggregate_rollouts(rollouts),
        }


def _strict_json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically write strict deterministic JSON in the destination directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(_strict_json_text(value))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, destination)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def load_strict_json(path: str | Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    with Path(path).open(encoding="utf-8") as file:
        value = json.load(file, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("checkpoint root must be a JSON object")
    return value


def _resume_signature(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "model_type",
        "input_dimension",
        "output_dimension",
        "search_space",
        "optimizer_settings",
        "software_versions",
        "optimizer_runtime",
        "objective",
        "replay_verification",
        "thresholds",
        "training_washout_per_trajectory",
        "training_currents",
        "fitting_transition_range",
        "validation_transition_range",
        "dataset_hashes",
        "preprocessing",
    )
    return {key: metadata.get(key) for key in keys}


def _new_checkpoint(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metadata": dict(metadata),
        "status": "search_in_progress",
        "trials": [],
        "robust_confirmations": [],
    }


def _validate_resume(
    checkpoint: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    if "metadata" not in checkpoint:
        raise ValueError("checkpoint is missing metadata")
    if _resume_signature(checkpoint["metadata"]) != _resume_signature(metadata):
        raise ValueError("checkpoint metadata conflicts with the current run")


def verify_same_point(
    replayed_point: Sequence[Any],
    saved_point: Sequence[Any],
    *,
    model_type: str,
    trial_number: int,
    checkpoint_versions: Mapping[str, Any] | None,
) -> None:
    """Require exact equality in the canonical serialized point order.

    Compatible package versions are required before replay. Floating values
    use a 1e-15 relative, zero-absolute tolerance solely for binary64
    serialization/inverse-transform roundoff; integers and categories remain
    exact.
    """
    replayed = list(replayed_point)
    saved = list(saved_point)
    differences: list[dict[str, Any]] = []
    if len(replayed) != len(SEARCH_PARAMETER_ORDER) or len(saved) != len(
        SEARCH_PARAMETER_ORDER
    ):
        differences.append(
            {
                "field": "point_length",
                "replayed": len(replayed),
                "saved": len(saved),
            }
        )
    else:
        for index, field in enumerate(SEARCH_PARAMETER_ORDER):
            replayed_value = replayed[index]
            saved_value = saved[index]
            if field == "reservoir_size":
                same = replayed_value == saved_value
            else:
                try:
                    same = math.isclose(
                        float(replayed_value),
                        float(saved_value),
                        rel_tol=FLOAT_REPLAY_REL_TOLERANCE,
                        abs_tol=FLOAT_REPLAY_ABS_TOLERANCE,
                    )
                except (TypeError, ValueError):
                    same = False
            if not same:
                differences.append(
                    {
                        "field": field,
                        "replayed": replayed_value,
                        "saved": saved_value,
                    }
                )
    if differences:
        version_record = {
            "checkpoint": dict(checkpoint_versions or {}),
            "current": software_versions(),
        }
        raise OptimizerReplayMismatchError(
            "Bayesian optimiser replay mismatch: "
            f"model_type={model_type}; trial_number={trial_number}; "
            f"expected_point={replayed!r}; saved_point={saved!r}; "
            f"differing_fields={differences!r}; "
            f"package_versions={version_record!r}"
        )


def _replay_completed_trials(
    optimizer: Any,
    trials: Sequence[Mapping[str, Any]],
    *,
    model_type: str,
    settings: SearchSettings,
    checkpoint_versions: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    """Restore optimiser state using the original ask/evaluate/tell order."""
    if len(trials) > settings.n_calls:
        raise ValueError("checkpoint contains more trials than the search budget")
    completed_by_parameters: dict[str, Mapping[str, Any]] = {}
    for trial_number, trial in enumerate(trials, start=1):
        if int(trial.get("trial_index", -1)) != trial_number:
            raise ValueError("checkpoint trials are not in original trial order")
        objective = float(trial["objective"])
        if not np.isfinite(objective):
            raise ValueError("checkpoint contains a non-finite objective")
        replayed_point = list(optimizer.ask())
        saved_point = list(trial["point"])
        verify_same_point(
            replayed_point,
            saved_point,
            model_type=model_type,
            trial_number=trial_number,
            checkpoint_versions=checkpoint_versions,
        )
        optimizer.tell(saved_point, objective)
        completed_by_parameters[
            serialized_hyperparameters(trial["hyperparameters"])
        ] = trial
    return completed_by_parameters


def reconstruct_interruption_checkpoint(
    *,
    audit_checkpoint_path: str | Path,
    working_checkpoint_path: str | Path,
    model_type: str,
    retained_trial_count: int,
    metadata: Mapping[str, Any],
    settings: SearchSettings,
) -> dict[str, Any]:
    """Build and replay-verify a partial checkpoint from an audit history."""
    _validate_model_type(model_type)
    source_path = Path(audit_checkpoint_path)
    destination = Path(working_checkpoint_path)
    if source_path.resolve() == destination.resolve():
        raise ValueError("audit and working checkpoint paths must differ")
    source = load_strict_json(source_path)
    source_metadata = source.get("metadata", {})
    if source_metadata.get("model_type") != model_type:
        raise ValueError("audit checkpoint has the wrong model type")
    if not 0 < retained_trial_count < len(source.get("trials", [])):
        raise ValueError("retained trial count must be a strict history prefix")

    expected_legacy_settings = asdict(settings)
    source_settings = source_metadata.get("optimizer_settings", {})
    if any(
        source_settings.get(key) != value
        for key, value in expected_legacy_settings.items()
    ):
        raise ValueError("audit checkpoint optimiser settings conflict with protocol")
    protocol_keys = (
        "label",
        "model_type",
        "input_dimension",
        "output_dimension",
        "search_space",
        "objective",
        "thresholds",
        "training_washout_per_trajectory",
        "training_currents",
        "fitting_transition_range",
        "validation_transition_range",
        "dataset_hashes",
        "preprocessing",
    )
    conflicting_keys = [
        key
        for key in protocol_keys
        if source_metadata.get(key) != metadata.get(key)
    ]
    if conflicting_keys:
        raise ValueError(
            "audit checkpoint metadata conflicts with the frozen protocol: "
            f"{conflicting_keys!r}"
        )

    trials = source["trials"]
    expected_order = list(range(1, len(trials) + 1))
    actual_order = [int(trial.get("trial_index", -1)) for trial in trials]
    if actual_order != expected_order:
        raise ValueError("audit checkpoint trials are not in original order")

    retained = copy.deepcopy(trials[:retained_trial_count])
    for trial_number, trial in enumerate(retained, start=1):
        if len(trial.get("rollouts", [])) != 9:
            raise ValueError(
                f"audit trial {trial_number} does not contain nine rollouts"
            )
        canonical_point = hyperparameters_to_point(trial["hyperparameters"])
        verify_same_point(
            canonical_point,
            trial["point"],
            model_type=model_type,
            trial_number=trial_number,
            checkpoint_versions=source_metadata.get("software_versions"),
        )
        objective = float(trial["objective"])
        if not np.isfinite(objective):
            raise ValueError(f"audit trial {trial_number} objective is non-finite")
        if rollout_objective(trial["rollouts"]) != objective:
            raise ValueError(
                f"audit trial {trial_number} objective disagrees with rollouts"
            )

    checkpoint = _new_checkpoint(metadata)
    checkpoint["trials"] = retained
    checkpoint["resume_reconstruction"] = {
        "deliberately_reconstructed": True,
        "source_audit_checkpoint": str(source_path),
        "original_trial_count": len(trials),
        "retained_trial_count": retained_trial_count,
        "retained_trial_range": [1, retained_trial_count],
        "discarded_trial_range": [retained_trial_count + 1, len(trials)],
        "removed_robust_confirmation_count": len(
            source.get("robust_confirmations", [])
        ),
        "historical_objectives_reevaluated": False,
        "replay_verified": True,
        "reason": (
            "Reconstructed at the documented original interruption boundary "
            "for deterministic ask/verify/tell resume correction."
        ),
    }
    _replay_completed_trials(
        make_optimizer(model_type, settings),
        checkpoint["trials"],
        model_type=model_type,
        settings=settings,
        checkpoint_versions=metadata.get("software_versions"),
    )
    atomic_write_json(destination, checkpoint)
    return checkpoint


def run_bayesian_search(
    *,
    checkpoint_path: str | Path,
    model_type: str,
    evaluator: CandidateEvaluator,
    metadata: Mapping[str, Any],
    settings: SearchSettings,
    resume: bool,
    optimizer_factory: OptimizerFactory = make_optimizer,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run or resume one deterministic Bayesian search."""
    _validate_model_type(model_type)
    path = Path(checkpoint_path)
    if path.exists():
        if not resume:
            raise FileExistsError(
                f"checkpoint already exists; use resume to continue: {path}"
            )
        checkpoint = load_strict_json(path)
        _validate_resume(checkpoint, metadata)
    else:
        checkpoint = _new_checkpoint(metadata)
        atomic_write_json(path, checkpoint)

    optimizer = optimizer_factory(model_type, settings)
    completed_by_parameters = _replay_completed_trials(
        optimizer,
        checkpoint["trials"],
        model_type=model_type,
        settings=settings,
        checkpoint_versions=checkpoint["metadata"].get("software_versions"),
    )

    while len(checkpoint["trials"]) < settings.n_calls:
        trial_index = len(checkpoint["trials"]) + 1
        proposed_point = list(optimizer.ask())
        parameters = point_to_hyperparameters(proposed_point)
        point = hyperparameters_to_point(parameters)
        serialized = serialized_hyperparameters(parameters)
        started = clock()
        if serialized in completed_by_parameters:
            previous = completed_by_parameters[serialized]
            evaluation = {
                "objective": previous["objective"],
                "rollouts": previous["rollouts"],
                "aggregate": previous["aggregate"],
            }
            status = "completed_reused_duplicate"
        else:
            evaluation = evaluator(parameters, settings.candidate_model_seed)
            status = "completed"
        duration = float(clock() - started)
        objective = float(evaluation["objective"])
        if not np.isfinite(objective):
            raise ValueError("candidate evaluator returned a non-finite objective")
        optimizer.tell(point, objective)
        trial = {
            "trial_index": trial_index,
            "point": point,
            "hyperparameters": parameters,
            "model_seed": settings.candidate_model_seed,
            "objective": objective,
            "rollouts": evaluation["rollouts"],
            "aggregate": evaluation["aggregate"],
            "status": status,
            "duration_seconds": duration,
        }
        checkpoint["trials"].append(trial)
        completed_by_parameters[serialized] = trial
        checkpoint["status"] = (
            "search_complete"
            if len(checkpoint["trials"]) == settings.n_calls
            else "search_in_progress"
        )
        atomic_write_json(path, checkpoint)
        print(
            f"[{model_type}] trial {trial_index}/{settings.n_calls} "
            f"objective={objective:.9g} duration={duration:.1f}s",
            flush=True,
        )
    return checkpoint


def _unique_top_trials(
    checkpoint: Mapping[str, Any],
    top_count: int,
) -> list[Mapping[str, Any]]:
    ordered = sorted(
        checkpoint["trials"],
        key=lambda trial: (
            float(trial["objective"]),
            serialized_hyperparameters(trial["hyperparameters"]),
            int(trial["trial_index"]),
        ),
    )
    unique: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for trial in ordered:
        serialized = serialized_hyperparameters(trial["hyperparameters"])
        if serialized in seen:
            continue
        seen.add(serialized)
        unique.append(trial)
        if len(unique) == min(top_count, len(checkpoint["trials"])):
            break
    return unique


def _robust_aggregate(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rollouts = [
        rollout
        for seed_result in seed_results
        for rollout in seed_result["rollouts"]
    ]
    aggregate = aggregate_rollouts(rollouts)
    aggregate["robust_rollout_count"] = len(rollouts)
    aggregate["robust_seed_count"] = len(seed_results)
    return aggregate


def run_robust_confirmation(
    *,
    checkpoint_path: str | Path,
    evaluator: CandidateEvaluator,
    final_seeds: Sequence[int] = FINAL_SEEDS,
    top_count: int = TOP_CANDIDATE_COUNT,
) -> dict[str, Any]:
    """Confirm the top distinct candidates across every frozen final seed."""
    path = Path(checkpoint_path)
    checkpoint = load_strict_json(path)
    if checkpoint["status"] not in {
        "search_complete",
        "robust_confirmation_in_progress",
        "complete",
    }:
        raise ValueError("Bayesian search must complete before robust confirmation")
    top_trials = _unique_top_trials(checkpoint, top_count)
    confirmations = {
        serialized_hyperparameters(item["hyperparameters"]): item
        for item in checkpoint.get("robust_confirmations", [])
    }

    for rank, trial in enumerate(top_trials, start=1):
        parameters = trial["hyperparameters"]
        serialized = serialized_hyperparameters(parameters)
        confirmation = confirmations.get(
            serialized,
            {
                "seed42_rank": rank,
                "source_trial_index": trial["trial_index"],
                "hyperparameters": parameters,
                "seed_results": [],
                "complete": False,
                "aggregate": None,
            },
        )
        existing_seeds = {
            int(item["model_seed"]) for item in confirmation["seed_results"]
        }
        for seed in final_seeds:
            if int(seed) in existing_seeds:
                continue
            if int(seed) == CANDIDATE_MODEL_SEED:
                result = {
                    "model_seed": int(seed),
                    "objective": trial["objective"],
                    "rollouts": trial["rollouts"],
                    "aggregate": trial["aggregate"],
                    "reused_seed42_search_result": True,
                }
            else:
                evaluated = evaluator(parameters, int(seed))
                result = {
                    "model_seed": int(seed),
                    "objective": evaluated["objective"],
                    "rollouts": evaluated["rollouts"],
                    "aggregate": evaluated["aggregate"],
                    "reused_seed42_search_result": False,
                }
            confirmation["seed_results"].append(result)
            existing_seeds.add(int(seed))
            confirmations[serialized] = confirmation
            checkpoint["robust_confirmations"] = list(confirmations.values())
            checkpoint["status"] = "robust_confirmation_in_progress"
            atomic_write_json(path, checkpoint)
            print(
                f"[{checkpoint['metadata']['model_type']}] robust rank {rank} "
                f"seed={seed} objective={float(result['objective']):.9g}",
                flush=True,
            )

        confirmation["seed_results"] = sorted(
            confirmation["seed_results"],
            key=lambda item: list(final_seeds).index(int(item["model_seed"])),
        )
        confirmation["aggregate"] = _robust_aggregate(
            confirmation["seed_results"]
        )
        confirmation["complete"] = True
        confirmations[serialized] = confirmation
        checkpoint["robust_confirmations"] = list(confirmations.values())
        atomic_write_json(path, checkpoint)

    checkpoint["robust_confirmations"] = sorted(
        confirmations.values(),
        key=lambda item: int(item["seed42_rank"]),
    )
    checkpoint["status"] = "complete"
    atomic_write_json(path, checkpoint)
    return checkpoint


def robust_tie_key(confirmation: Mapping[str, Any]) -> tuple[Any, ...]:
    aggregate = confirmation["aggregate"]
    return (
        float(aggregate["mean_objective_nrmse"]),
        float(aggregate["worst_current_mean_nrmse"]),
        -float(aggregate["mean_valid_prediction_steps"]),
        serialized_hyperparameters(confirmation["hyperparameters"]),
    )


def selected_confirmation(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    complete = [
        item
        for item in checkpoint["robust_confirmations"]
        if bool(item["complete"])
    ]
    if not complete:
        raise ValueError("checkpoint has no complete robust confirmations")
    return min(complete, key=robust_tie_key)


def build_selection_artifact(
    parameter_aware_history: Mapping[str, Any],
    ordinary_history: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the Step 7 validation-only selection record."""
    histories = {
        PARAMETER_AWARE: parameter_aware_history,
        ORDINARY_BASELINE: ordinary_history,
    }
    if any(history["status"] != "complete" for history in histories.values()):
        raise ValueError("both model histories must be complete")
    budgets = {
        model_type: history["metadata"]["optimizer_settings"]["n_calls"]
        for model_type, history in histories.items()
    }
    if len(set(budgets.values())) != 1:
        raise ValueError("both models must receive equal Bayesian budgets")

    model_results: dict[str, Any] = {}
    for model_type, history in histories.items():
        selected = selected_confirmation(history)
        top_trials = _unique_top_trials(history, TOP_CANDIDATE_COUNT)
        model_results[model_type] = {
            "input_dimension": MODEL_INPUT_DIMENSIONS[model_type],
            "output_dimension": OUTPUT_DIMENSION,
            "best_configuration": selected["hyperparameters"],
            "best_robust_aggregate": selected["aggregate"],
            "selected_source_trial_index": selected["source_trial_index"],
            "top_five_seed42_candidates": [
                {
                    "rank": index,
                    "trial_index": trial["trial_index"],
                    "objective": trial["objective"],
                    "hyperparameters": trial["hyperparameters"],
                    "aggregate": trial["aggregate"],
                }
                for index, trial in enumerate(top_trials, start=1)
            ],
            "five_seed_robust_confirmation_results": history[
                "robust_confirmations"
            ],
        }

    metadata = parameter_aware_history["metadata"]
    return {
        "schema": SELECTION_SCHEMA,
        "label": RESULT_LABEL,
        "step7_complete": True,
        "search_space": SEARCH_SPACE_DEFINITION,
        "optimizer_settings": {
            model_type: histories[model_type]["metadata"]["optimizer_settings"]
            for model_type in MODEL_TYPES
        },
        "equal_bayesian_budgets": True,
        "objective": OBJECTIVE_DEFINITION,
        "selection_rules": list(SELECTION_RULES),
        "thresholds": metadata["thresholds"],
        "training_washout_per_trajectory": TRAINING_WASHOUT,
        "dataset_hashes": metadata["dataset_hashes"],
        "preprocessing": metadata["preprocessing"],
        "git": git_state(),
        "data_access": {
            "held_out_opened": False,
            "unseen_current_opened": False,
            "continuous_benchmark_opened": False,
            "benchmark_results_present": False,
        },
        "models": model_results,
        "next_step": (
            "Step 8 may lock these validation-selected configurations and train "
            "final models; no final model was trained or saved in Step 7."
        ),
    }


def write_selection_artifact(
    parameter_aware_path: str | Path,
    ordinary_path: str | Path,
    selection_path: str | Path,
) -> dict[str, Any]:
    parameter_history = load_strict_json(parameter_aware_path)
    ordinary_history = load_strict_json(ordinary_path)
    selection = build_selection_artifact(parameter_history, ordinary_history)
    atomic_write_json(selection_path, selection)
    return selection


def history_path(output_dir: str | Path, model_type: str) -> Path:
    filename = {
        PARAMETER_AWARE: "step7_parameter_aware_history.json",
        ORDINARY_BASELINE: "step7_ordinary_baseline_history.json",
    }[model_type]
    return Path(output_dir) / filename


def selection_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "step7_selection.json"
