"""Safe, leakage-aware data preparation for the Chapter 2 ESN experiments.

The functions in this module only read and transform frozen trajectories. They
do not simulate Hindmarsh--Rose dynamics, fit an ESN, or write any dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

try:
    from .esn_config import (
        CONTINUOUS_DATASET,
        FIXED_DATASETS,
        FITTING_TRANSITIONS,
        OUTPUT_DIMENSION,
        PARAMETER_AWARE_INPUT_DIMENSION,
        REQUIRED_ARRAY_KEYS,
        STATE_DIMENSION,
        TRAIN_CURRENTS,
        UNSEEN_CURRENTS,
        VALIDATION_WINDOWS,
        LockedDataset,
        TransitionRange,
        ValidationWindow,
        fixed_dataset,
        transition_split,
    )
except ImportError:  # Support direct imports from the chapter2 directory.
    from esn_config import (
        CONTINUOUS_DATASET,
        FIXED_DATASETS,
        FITTING_TRANSITIONS,
        OUTPUT_DIMENSION,
        PARAMETER_AWARE_INPUT_DIMENSION,
        REQUIRED_ARRAY_KEYS,
        STATE_DIMENSION,
        TRAIN_CURRENTS,
        UNSEEN_CURRENTS,
        VALIDATION_WINDOWS,
        LockedDataset,
        TransitionRange,
        ValidationWindow,
        fixed_dataset,
        transition_split,
    )


FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]


class DataValidationError(ValueError):
    """Raised when trajectory contents violate the frozen data contract."""


class DatasetIntegrityError(DataValidationError):
    """Raised when a locked trajectory does not match its recorded SHA-256."""


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of ``path`` without modifying it."""
    digest = sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_array(name: str, values: np.ndarray, ndim: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != ndim:
        raise DataValidationError(
            f"{name} must be {ndim}-dimensional, got shape {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise DataValidationError(f"{name} must have a numeric dtype")
    if not np.all(np.isfinite(array)):
        raise DataValidationError(f"{name} must contain only finite values")
    return array


def _validated_trajectory_arrays(
    time: np.ndarray,
    states: np.ndarray,
    currents: np.ndarray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    checked_time = _numeric_array("time", time, 1)
    checked_states = _numeric_array("states", states, 2)
    checked_currents = _numeric_array("current", currents, 1)
    if checked_states.shape[1:] != (STATE_DIMENSION,):
        raise DataValidationError(
            f"states must have shape (n, {STATE_DIMENSION}), got "
            f"{checked_states.shape}"
        )
    lengths = (len(checked_time), len(checked_states), len(checked_currents))
    if len(set(lengths)) != 1:
        raise DataValidationError(
            "time, states, and current arrays must have equal lengths"
        )
    if lengths[0] < 2:
        raise DataValidationError("a trajectory must contain at least two states")
    if not np.all(np.diff(checked_time) > 0.0):
        raise DataValidationError("time must be strictly increasing")
    return (
        np.asarray(checked_time, dtype=float).copy(),
        np.asarray(checked_states, dtype=float).copy(),
        np.asarray(checked_currents, dtype=float).copy(),
    )


@dataclass(frozen=True)
class FixedCurrentTrajectory:
    """One independent trajectory recorded under a constant supplied current."""

    current: float
    time: FloatArray
    states: FloatArray
    current_values: FloatArray
    path: Path | None = None

    def __post_init__(self) -> None:
        time, states, currents = _validated_trajectory_arrays(
            self.time, self.states, self.current_values
        )
        if not np.isfinite(self.current):
            raise DataValidationError("fixed current must be finite")
        if not np.all(currents == float(self.current)):
            raise DataValidationError(
                f"fixed trajectory must contain only I={self.current}"
            )
        object.__setattr__(self, "current", float(self.current))
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "current_values", currents)
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))

    @property
    def state_count(self) -> int:
        return len(self.time)


@dataclass(frozen=True)
class ContinuousCurrentTrajectory:
    """One state-continuous trajectory with predefined current switches."""

    current_sequence: tuple[float, ...]
    switch_indices: tuple[int, ...]
    time: FloatArray
    states: FloatArray
    current_values: FloatArray
    path: Path | None = None

    def __post_init__(self) -> None:
        time, states, currents = _validated_trajectory_arrays(
            self.time, self.states, self.current_values
        )
        detected_switches = tuple(
            (np.flatnonzero(np.diff(currents) != 0.0) + 1).tolist()
        )
        if detected_switches != tuple(self.switch_indices):
            raise DataValidationError(
                "continuous current switch indices do not match the expected "
                f"indices: expected {tuple(self.switch_indices)}, got "
                f"{detected_switches}"
            )
        segment_starts = (0,) + detected_switches
        detected_sequence = tuple(float(currents[index]) for index in segment_starts)
        if detected_sequence != tuple(self.current_sequence):
            raise DataValidationError(
                "continuous current order does not match the expected sequence: "
                f"expected {tuple(self.current_sequence)}, got {detected_sequence}"
            )
        object.__setattr__(
            self, "current_sequence", tuple(float(x) for x in self.current_sequence)
        )
        object.__setattr__(
            self, "switch_indices", tuple(int(x) for x in self.switch_indices)
        )
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "current_values", currents)
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))

    @property
    def state_count(self) -> int:
        return len(self.time)


Trajectory: TypeAlias = FixedCurrentTrajectory | ContinuousCurrentTrajectory


def _read_locked_arrays(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_state_count: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"locked trajectory does not exist: {source}")
    actual_sha256 = file_sha256(source)
    if actual_sha256 != expected_sha256:
        raise DatasetIntegrityError(
            f"SHA-256 mismatch for {source}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )

    try:
        with np.load(source, allow_pickle=False) as saved:
            keys = tuple(saved.files)
            if keys != REQUIRED_ARRAY_KEYS:
                raise DataValidationError(
                    f"{source} keys must be {REQUIRED_ARRAY_KEYS}, got {keys}"
                )
            arrays = {name: np.asarray(saved[name]).copy() for name in keys}
    except DataValidationError:
        raise
    except (OSError, ValueError) as error:
        raise DataValidationError(f"could not safely load {source}: {error}") from error

    for name in REQUIRED_ARRAY_KEYS:
        _numeric_array(name, arrays[name], 1)
    lengths = {len(arrays[name]) for name in REQUIRED_ARRAY_KEYS}
    if lengths != {expected_state_count}:
        raise DataValidationError(
            f"{source} must contain {expected_state_count} states in every array; "
            f"got lengths {sorted(lengths)}"
        )
    states = np.column_stack((arrays["x"], arrays["y"], arrays["z"]))
    return _validated_trajectory_arrays(arrays["t"], states, arrays["I"])


def load_fixed_trajectory_file(
    path: str | Path,
    *,
    expected_current: float,
    expected_sha256: str,
    expected_state_count: int,
) -> FixedCurrentTrajectory:
    """Load and validate one hash-identified fixed-current trajectory file."""
    time, states, currents = _read_locked_arrays(
        path,
        expected_sha256=expected_sha256,
        expected_state_count=expected_state_count,
    )
    return FixedCurrentTrajectory(
        float(expected_current), time, states, currents, Path(path)
    )


def load_fixed_trajectory(current: float) -> FixedCurrentTrajectory:
    """Load one configured fixed-current trajectory with integrity checking."""
    record = fixed_dataset(float(current))
    assert record.current is not None
    return load_fixed_trajectory_file(
        record.path,
        expected_current=record.current,
        expected_sha256=record.sha256,
        expected_state_count=record.state_count,
    )


def load_continuous_trajectory_file(
    path: str | Path,
    *,
    expected_sequence: Sequence[float],
    expected_switch_indices: Sequence[int],
    expected_sha256: str,
    expected_state_count: int,
) -> ContinuousCurrentTrajectory:
    """Load and validate one hash-identified continuous-current trajectory."""
    time, states, currents = _read_locked_arrays(
        path,
        expected_sha256=expected_sha256,
        expected_state_count=expected_state_count,
    )
    return ContinuousCurrentTrajectory(
        tuple(float(x) for x in expected_sequence),
        tuple(int(x) for x in expected_switch_indices),
        time,
        states,
        currents,
        Path(path),
    )


def load_continuous_benchmark() -> ContinuousCurrentTrajectory:
    """Load the separate predefined continuous transition benchmark."""
    return load_continuous_trajectory_file(
        CONTINUOUS_DATASET.path,
        expected_sequence=CONTINUOUS_DATASET.current_sequence,
        expected_switch_indices=CONTINUOUS_DATASET.switch_indices,
        expected_sha256=CONTINUOUS_DATASET.sha256,
        expected_state_count=CONTINUOUS_DATASET.state_count,
    )


def load_unseen_benchmarks() -> tuple[FixedCurrentTrajectory, ...]:
    """Load the fixed unseen-current benchmarks outside the optimisation API."""
    return tuple(load_fixed_trajectory(current) for current in UNSEEN_CURRENTS)


@dataclass(frozen=True)
class OneStepPairs:
    """Aligned one-step inputs and targets created within one trajectory."""

    inputs: FloatArray
    targets: FloatArray
    transition_indices: IntegerArray
    source_current: float | None

    def __post_init__(self) -> None:
        inputs = _numeric_array("inputs", self.inputs, 2)
        targets = _numeric_array("targets", self.targets, 2)
        indices = np.asarray(self.transition_indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise DataValidationError(
                "transition_indices must be a one-dimensional integer array"
            )
        if inputs.shape[1] not in (STATE_DIMENSION, PARAMETER_AWARE_INPUT_DIMENSION):
            raise DataValidationError("inputs must have three or four columns")
        if targets.shape[1] != OUTPUT_DIMENSION:
            raise DataValidationError(
                f"targets must have {OUTPUT_DIMENSION} columns"
            )
        if len(inputs) != len(targets) or len(inputs) != len(indices):
            raise DataValidationError(
                "inputs, targets, and transition indices must have equal lengths"
            )
        object.__setattr__(self, "inputs", np.asarray(inputs, dtype=float).copy())
        object.__setattr__(self, "targets", np.asarray(targets, dtype=float).copy())
        object.__setattr__(
            self, "transition_indices", np.asarray(indices, dtype=np.int64).copy()
        )

    def __len__(self) -> int:
        return len(self.inputs)


def create_one_step_pairs(
    trajectory: Trajectory,
    transition_range: TransitionRange | None = None,
    *,
    include_current: bool = True,
) -> OneStepPairs:
    """Create exact ``t -> t+1`` pairs inside one trajectory.

    ``transition_range.stop`` is exclusive. A range ``[a, b)`` uses state
    inputs ``a`` through ``b-1`` and targets ``a+1`` through ``b``.
    """
    transition_count = trajectory.state_count - 1
    selected = transition_range or TransitionRange(0, transition_count)
    if selected.stop > transition_count:
        raise DataValidationError(
            f"transition range [{selected.start}, {selected.stop}) exceeds "
            f"the available range [0, {transition_count})"
        )
    indices = np.arange(selected.start, selected.stop, dtype=np.int64)
    state_inputs = trajectory.states[selected.start : selected.stop]
    targets = trajectory.states[selected.start + 1 : selected.stop + 1]
    if include_current:
        inputs = np.column_stack(
            (state_inputs, trajectory.current_values[selected.start : selected.stop])
        )
    else:
        inputs = state_inputs.copy()
    source_current = (
        trajectory.current if isinstance(trajectory, FixedCurrentTrajectory) else None
    )
    return OneStepPairs(inputs, targets, indices, source_current)


def concatenate_one_step_pairs(groups: Sequence[OneStepPairs]) -> OneStepPairs:
    """Concatenate already-created pairs without inventing boundary transitions."""
    items = tuple(groups)
    if not items:
        raise DataValidationError("at least one pair group is required")
    input_dimensions = {item.inputs.shape[1] for item in items}
    if len(input_dimensions) != 1:
        raise DataValidationError("all pair groups must use the same input dimension")
    return OneStepPairs(
        np.concatenate([item.inputs for item in items], axis=0),
        np.concatenate([item.targets for item in items], axis=0),
        np.concatenate([item.transition_indices for item in items], axis=0),
        None,
    )


@dataclass(frozen=True)
class ValidationWindowView:
    """Prepared warm-up and scored pairs for one frozen validation window."""

    definition: ValidationWindow
    warmup: OneStepPairs
    scored: OneStepPairs


@dataclass(frozen=True)
class PreparedOptimisationTrajectory:
    """Leakage-safe optimisation views for one training-current trajectory.

    This deliberately has no held-out field. Optimisation receives fitting
    transitions and the predefined validation windows only.
    """

    current: float
    fitting: OneStepPairs
    validation_windows: tuple[ValidationWindowView, ...]


@dataclass(frozen=True)
class SeenCurrentHeldOut:
    """Explicit post-locking held-out view for one seen training current."""

    current: float
    pairs: OneStepPairs


def prepare_fixed_trajectory(
    trajectory: FixedCurrentTrajectory,
    *,
    include_current: bool = True,
) -> PreparedOptimisationTrajectory:
    """Create only the views permitted during optimisation."""
    split = transition_split(trajectory.state_count)
    windows = tuple(
        ValidationWindowView(
            window,
            create_one_step_pairs(
                trajectory, window.warmup, include_current=include_current
            ),
            create_one_step_pairs(
                trajectory, window.scored, include_current=include_current
            ),
        )
        for window in VALIDATION_WINDOWS
    )
    return PreparedOptimisationTrajectory(
        trajectory.current,
        create_one_step_pairs(
            trajectory, split.fitting, include_current=include_current
        ),
        windows,
    )


def _validate_training_current_request(currents: Sequence[float]) -> tuple[float, ...]:
    """Validate and normalize a request restricted to seen training currents."""
    requested = tuple(float(current) for current in currents)
    if not requested:
        raise DataValidationError("at least one training current is required")
    if len(requested) != len(set(requested)):
        raise DataValidationError("training currents must not contain duplicates")
    forbidden = tuple(current for current in requested if current not in TRAIN_CURRENTS)
    if forbidden:
        raise DataValidationError(
            "data may contain only training currents "
            f"{TRAIN_CURRENTS}; rejected {forbidden}"
        )
    return requested


def load_optimisation_data(
    currents: Sequence[float] = TRAIN_CURRENTS,
    *,
    include_current: bool = True,
) -> tuple[PreparedOptimisationTrajectory, ...]:
    """Load optimisation views for permitted training currents only."""
    requested = _validate_training_current_request(currents)
    return tuple(
        prepare_fixed_trajectory(
            load_fixed_trajectory(current), include_current=include_current
        )
        for current in requested
    )


def load_optimization_data(
    currents: Sequence[float] = TRAIN_CURRENTS,
    *,
    include_current: bool = True,
) -> tuple[PreparedOptimisationTrajectory, ...]:
    """US-spelling alias for :func:`load_optimisation_data`."""
    return load_optimisation_data(currents, include_current=include_current)


def load_seen_current_held_out(
    currents: Sequence[float] = TRAIN_CURRENTS,
    *,
    include_current: bool = True,
) -> tuple[SeenCurrentHeldOut, ...]:
    """Load seen-current held-out views through an explicit benchmark API.

    This post-locking loader is intentionally separate from
    :func:`load_optimisation_data` and is never called by it.
    """
    requested = _validate_training_current_request(currents)
    held_out: list[SeenCurrentHeldOut] = []
    for current in requested:
        trajectory = load_fixed_trajectory(current)
        split = transition_split(trajectory.state_count)
        held_out.append(
            SeenCurrentHeldOut(
                current,
                create_one_step_pairs(
                    trajectory,
                    split.held_out,
                    include_current=include_current,
                ),
            )
        )
    return tuple(held_out)


@dataclass(frozen=True)
class NumpyStandardScaler:
    """Small immutable per-column standardiser implemented with NumPy."""

    mean: FloatArray
    scale: FloatArray

    def __post_init__(self) -> None:
        mean = _numeric_array("scaler mean", self.mean, 1)
        scale = _numeric_array("scaler scale", self.scale, 1)
        if mean.shape != scale.shape:
            raise DataValidationError("scaler mean and scale shapes must match")
        if len(mean) == 0 or np.any(scale <= 0.0):
            raise DataValidationError("scaler scales must be positive")
        object.__setattr__(self, "mean", np.asarray(mean, dtype=float).copy())
        object.__setattr__(self, "scale", np.asarray(scale, dtype=float).copy())

    @classmethod
    def fit(cls, values: np.ndarray) -> "NumpyStandardScaler":
        """Fit per-column mean and population standard deviation."""
        matrix = _numeric_array("scaler fitting values", values, 2)
        if len(matrix) == 0:
            raise DataValidationError("cannot fit a scaler to an empty array")
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0, ddof=0)
        scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> FloatArray:
        matrix = _numeric_array("values to transform", values, 2)
        if matrix.shape[1] != len(self.mean):
            raise DataValidationError(
                f"expected {len(self.mean)} columns, got {matrix.shape[1]}"
            )
        return np.asarray((matrix - self.mean) / self.scale, dtype=float)

    def inverse_transform(self, values: np.ndarray) -> FloatArray:
        matrix = _numeric_array("values to inverse-transform", values, 2)
        if matrix.shape[1] != len(self.mean):
            raise DataValidationError(
                f"expected {len(self.mean)} columns, got {matrix.shape[1]}"
            )
        return np.asarray(matrix * self.scale + self.mean, dtype=float)


@dataclass(frozen=True)
class StateCurrentScalers:
    """Shared neuronal-state and supplied-current scaling convention."""

    state: NumpyStandardScaler
    current: NumpyStandardScaler

    def __post_init__(self) -> None:
        if len(self.state.mean) != STATE_DIMENSION:
            raise DataValidationError("state scaler must have three columns")
        if len(self.current.mean) != 1:
            raise DataValidationError("current scaler must have one column")

    def transform_inputs(self, inputs: np.ndarray) -> FloatArray:
        matrix = _numeric_array("model inputs", inputs, 2)
        if matrix.shape[1] == STATE_DIMENSION:
            return self.state.transform(matrix)
        if matrix.shape[1] != PARAMETER_AWARE_INPUT_DIMENSION:
            raise DataValidationError("model inputs must have three or four columns")
        return np.column_stack(
            (self.state.transform(matrix[:, :STATE_DIMENSION]),
             self.current.transform(matrix[:, STATE_DIMENSION:]))
        )

    def transform_targets(self, targets: np.ndarray) -> FloatArray:
        """Apply the same state convention used for input states."""
        return self.state.transform(targets)

    def inverse_states(self, states: np.ndarray) -> FloatArray:
        return self.state.inverse_transform(states)


def fit_training_scalers(
    trajectories: Sequence[FixedCurrentTrajectory] | None = None,
) -> StateCurrentScalers:
    """Fit shared scalers from all three permitted fitting views and no others."""
    selected = (
        tuple(load_fixed_trajectory(current) for current in TRAIN_CURRENTS)
        if trajectories is None
        else tuple(trajectories)
    )
    if any(not isinstance(item, FixedCurrentTrajectory) for item in selected):
        raise DataValidationError(
            "scaler fitting accepts fixed-current training trajectories only; "
            "continuous data is forbidden"
        )
    selected_currents = tuple(item.current for item in selected)
    if len(selected_currents) != len(set(selected_currents)) or set(
        selected_currents
    ) != set(TRAIN_CURRENTS):
        raise DataValidationError(
            "scalers must be fitted from exactly the training currents "
            f"{TRAIN_CURRENTS}; got {selected_currents}"
        )
    for trajectory in selected:
        transition_split(trajectory.state_count)

    fitting_states = np.concatenate(
        [
            trajectory.states[
                FITTING_TRANSITIONS.start : FITTING_TRANSITIONS.stop
            ]
            for trajectory in selected
        ],
        axis=0,
    )
    fitting_currents = np.concatenate(
        [
            trajectory.current_values[
                FITTING_TRANSITIONS.start : FITTING_TRANSITIONS.stop
            ]
            for trajectory in selected
        ]
    ).reshape(-1, 1)
    return StateCurrentScalers(
        NumpyStandardScaler.fit(fitting_states),
        NumpyStandardScaler.fit(fitting_currents),
    )


def scale_one_step_pairs(
    pairs: OneStepPairs,
    scalers: StateCurrentScalers,
) -> OneStepPairs:
    """Scale inputs and targets while preserving alignment and provenance."""
    return OneStepPairs(
        scalers.transform_inputs(pairs.inputs),
        scalers.transform_targets(pairs.targets),
        pairs.transition_indices,
        pairs.source_current,
    )


def locked_dataset_hashes() -> dict[Path, str]:
    """Return current digests for the six configured datasets after verification."""
    hashes: dict[Path, str] = {}
    for record in FIXED_DATASETS + (CONTINUOUS_DATASET,):
        actual = file_sha256(record.path)
        if actual != record.sha256:
            raise DatasetIntegrityError(
                f"SHA-256 mismatch for {record.path}: expected {record.sha256}, "
                f"got {actual}"
            )
        hashes[record.path] = actual
    return hashes
