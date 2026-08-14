"""Deterministic parameter-aware Echo State Network mechanics for Chapter 2.

The reservoir and readout follow the compatible Chapter 1 convention:

    r_t = (1 - alpha) r_(t-1)
          + alpha tanh(W_in u_t + W r_(t-1) + b)

The next three-state prediction uses the updated reservoir and the explicit
feature vector v_t = [1, u_t, r_t]. Parameter-aware inputs have four columns
[x_t, y_t, z_t, I_t]; baseline inputs have three [x_t, y_t, z_t]. In both
cases the readout has exactly three outputs and never predicts current I.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

try:
    from .esn_config import (
        OUTPUT_DIMENSION,
        PARAMETER_AWARE_INPUT_DIMENSION,
        STATE_DIMENSION,
        ESNModelConfig,
    )
except ImportError:  # Support direct imports from the chapter2 directory.
    from esn_config import (
        OUTPUT_DIMENSION,
        PARAMETER_AWARE_INPUT_DIMENSION,
        STATE_DIMENSION,
        ESNModelConfig,
    )


FloatArray = NDArray[np.float64]
_MODEL_SCHEMA = "chapter2_esn_model_v1"


class ModelValidationError(ValueError):
    """Raised when model inputs violate the ESN mechanics contract."""


class ModelNotFittedError(RuntimeError):
    """Raised when a learned readout is required before fitting."""


class PairSequence(Protocol):
    """Structural type implemented by ``esn_data.OneStepPairs``."""

    inputs: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class TrainingSequence:
    """One independent, aligned teacher-forced transition sequence."""

    inputs: FloatArray
    targets: FloatArray

    def __post_init__(self) -> None:
        inputs = _matrix("training inputs", self.inputs)
        targets = _matrix("training targets", self.targets)
        if inputs.shape[1] not in (
            STATE_DIMENSION,
            PARAMETER_AWARE_INPUT_DIMENSION,
        ):
            raise ModelValidationError("training inputs must have 3 or 4 columns")
        if targets.shape[1] != OUTPUT_DIMENSION:
            raise ModelValidationError("training targets must have 3 columns")
        if len(inputs) != len(targets):
            raise ModelValidationError(
                "training inputs and targets must have equal lengths"
            )
        if len(inputs) < 2:
            raise ModelValidationError(
                "a training sequence requires at least two transitions"
            )
        object.__setattr__(self, "inputs", np.asarray(inputs, dtype=float).copy())
        object.__setattr__(self, "targets", np.asarray(targets, dtype=float).copy())


SequenceLike: TypeAlias = (
    TrainingSequence | PairSequence | tuple[np.ndarray, np.ndarray]
)


@dataclass(frozen=True)
class RidgeStatistics:
    """Streaming sufficient statistics for the linear readout."""

    gram: FloatArray
    cross: FloatArray
    sample_count: int

    def __post_init__(self) -> None:
        gram = _matrix("Gram statistic", self.gram)
        cross = _matrix("cross statistic", self.cross)
        if gram.shape[0] != gram.shape[1]:
            raise ModelValidationError("Gram statistic must be square")
        if cross.shape[1] != gram.shape[0]:
            raise ModelValidationError("ridge statistic dimensions are inconsistent")
        if self.sample_count <= 0:
            raise ModelValidationError("sample_count must be positive")
        object.__setattr__(self, "gram", np.asarray(gram, dtype=float).copy())
        object.__setattr__(self, "cross", np.asarray(cross, dtype=float).copy())


def _matrix(name: str, values: np.ndarray) -> FloatArray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ModelValidationError(f"{name} must be two-dimensional")
    if not np.issubdtype(array.dtype, np.number):
        raise ModelValidationError(f"{name} must have a numeric dtype")
    if not np.all(np.isfinite(array)):
        raise ModelValidationError(f"{name} must contain only finite values")
    return np.asarray(array, dtype=float)


def _vector(name: str, values: np.ndarray, length: int) -> FloatArray:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape != (length,):
        raise ModelValidationError(f"{name} must have shape ({length},)")
    if not np.issubdtype(array.dtype, np.number):
        raise ModelValidationError(f"{name} must have a numeric dtype")
    if not np.all(np.isfinite(array)):
        raise ModelValidationError(f"{name} must contain only finite values")
    return np.asarray(array, dtype=float)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ModelValidationError(f"{name} must be positive")
    return value


class EchoStateNetwork:
    """Small deterministic ESN with separate input and output dimensions."""

    def __init__(self, config: ESNModelConfig | None = None) -> None:
        self.config = config or ESNModelConfig()
        self.input_weights: FloatArray
        self.reservoir_weights: FloatArray
        self.reservoir_bias: FloatArray
        self.output_weights: FloatArray | None = None
        self.reservoir_state = np.zeros(self.config.reservoir_size, dtype=float)
        self._initialise_reservoir()

    @property
    def is_fitted(self) -> bool:
        return self.output_weights is not None

    @property
    def feature_dimension(self) -> int:
        """Dimension of v_t = [1, u_t, r_t]."""
        return 1 + self.config.input_dimension + self.config.reservoir_size

    @property
    def spectral_radius(self) -> float:
        return float(
            np.max(np.abs(np.linalg.eigvals(self.reservoir_weights)))
        )

    @property
    def realised_connectivity(self) -> float:
        """Fraction of nonzero off-diagonal recurrent weights."""
        size = self.config.reservoir_size
        return float(np.count_nonzero(self.reservoir_weights) / (size * (size - 1)))

    @property
    def ridge_penalty_matrix(self) -> FloatArray:
        """Diagonal ridge matrix; the intercept is unregularised by default."""
        penalty = np.eye(self.feature_dimension, dtype=float)
        if not self.config.regularise_bias:
            penalty[0, 0] = 0.0
        return penalty

    def _initialise_reservoir(self) -> None:
        config = self.config
        rng = np.random.default_rng(config.seed)
        self.input_weights = rng.uniform(
            -config.input_scaling,
            config.input_scaling,
            size=(config.reservoir_size, config.input_dimension),
        )
        self.reservoir_bias = rng.uniform(
            -config.bias_scaling,
            config.bias_scaling,
            size=config.reservoir_size,
        )

        raw = np.empty((0, 0), dtype=float)
        radius = 0.0
        for _ in range(10_000):
            mask = rng.random(
                (config.reservoir_size, config.reservoir_size)
            ) < config.reservoir_connectivity
            np.fill_diagonal(mask, False)
            raw = rng.uniform(
                -1.0, 1.0, size=(config.reservoir_size, config.reservoir_size)
            )
            raw *= mask
            radius = float(np.max(np.abs(np.linalg.eigvals(raw))))
            if radius > np.finfo(float).eps:
                break
        if radius <= np.finfo(float).eps:
            raise RuntimeError(
                "could not initialise a recurrent matrix with nonzero spectral radius"
            )
        self.reservoir_weights = np.asarray(
            raw * (config.spectral_radius / radius), dtype=float
        )

    def reset_reservoir(self) -> None:
        """Restore the reservoir to its independent-trajectory zero state."""
        self.reservoir_state.fill(0.0)

    def _validated_inputs(
        self, values: np.ndarray, *, allow_empty: bool = False
    ) -> FloatArray:
        inputs = _matrix("model inputs", values)
        if inputs.shape[1] != self.config.input_dimension:
            raise ModelValidationError(
                f"model inputs must have {self.config.input_dimension} columns"
            )
        if not allow_empty and len(inputs) == 0:
            raise ModelValidationError("model inputs must not be empty")
        return inputs

    def _advance(self, input_value: FloatArray) -> FloatArray:
        previous = self.reservoir_state
        activation = (
            self.input_weights @ input_value
            + self.reservoir_weights @ previous
            + self.reservoir_bias
        )
        leak = self.config.leak_rate
        self.reservoir_state = (
            (1.0 - leak) * previous + leak * np.tanh(activation)
        )
        return self.reservoir_state

    def _readout_feature(
        self, input_value: FloatArray, reservoir_state: FloatArray
    ) -> FloatArray:
        return np.concatenate(([1.0], input_value, reservoir_state))

    def _coerce_sequences(
        self, sequences: Sequence[SequenceLike]
    ) -> tuple[TrainingSequence, ...]:
        items = tuple(sequences)
        if not items:
            raise ModelValidationError("at least one training sequence is required")
        prepared: list[TrainingSequence] = []
        for item in items:
            if isinstance(item, TrainingSequence):
                sequence = item
            elif isinstance(item, tuple) and len(item) == 2:
                sequence = TrainingSequence(item[0], item[1])
            elif hasattr(item, "inputs") and hasattr(item, "targets"):
                sequence = TrainingSequence(item.inputs, item.targets)
            else:
                raise TypeError(
                    "each sequence must provide aligned inputs and targets"
                )
            if sequence.inputs.shape[1] != self.config.input_dimension:
                raise ModelValidationError(
                    "training input dimension does not match model configuration"
                )
            if sequence.targets.shape[1] != self.config.output_dimension:
                raise ModelValidationError(
                    "training output dimension does not match model configuration"
                )
            prepared.append(sequence)
        return tuple(prepared)

    def accumulate_ridge_statistics(
        self,
        sequences: Sequence[SequenceLike],
        *,
        washout: int = 0,
    ) -> RidgeStatistics:
        """Accumulate G=sum(vv^T) and C=sum(yv^T) sequence by sequence."""
        if isinstance(washout, bool) or not isinstance(washout, int):
            raise TypeError("washout must be an integer")
        if washout < 0:
            raise ModelValidationError("washout must be non-negative")
        prepared = self._coerce_sequences(sequences)
        if any(washout >= len(sequence.inputs) for sequence in prepared):
            raise ModelValidationError(
                "washout must leave at least one fitted transition per sequence"
            )

        gram = np.zeros((self.feature_dimension, self.feature_dimension))
        cross = np.zeros((self.config.output_dimension, self.feature_dimension))
        sample_count = 0

        for sequence in prepared:
            self.reset_reservoir()
            for index, (input_value, target) in enumerate(
                zip(sequence.inputs, sequence.targets)
            ):
                state = self._advance(input_value)
                if index < washout:
                    continue
                feature = self._readout_feature(input_value, state)
                gram += np.outer(feature, feature)
                cross += np.outer(target, feature)
                sample_count += 1

        self.reset_reservoir()
        return RidgeStatistics(gram, cross, sample_count)

    def fit(
        self,
        sequences: Sequence[SequenceLike],
        *,
        washout: int = 0,
    ) -> "EchoStateNetwork":
        """Fit the three-state readout from already prepared/scaled sequences."""
        statistics = self.accumulate_ridge_statistics(
            sequences, washout=washout
        )
        system = (
            statistics.gram
            + self.config.ridge_regularisation * self.ridge_penalty_matrix
        )
        try:
            weights = np.linalg.solve(system, statistics.cross.T).T
        except np.linalg.LinAlgError as error:
            raise RuntimeError("ridge readout solve failed") from error
        if not np.all(np.isfinite(weights)):
            raise RuntimeError("ridge readout solve produced non-finite weights")
        self.output_weights = np.ascontiguousarray(weights, dtype=float)
        self.reset_reservoir()
        return self

    def _require_fitted(self) -> FloatArray:
        if self.output_weights is None:
            raise ModelNotFittedError("ESN is not fitted; call fit() first")
        return self.output_weights

    def teacher_forced_warmup(
        self,
        inputs: np.ndarray,
        *,
        reset: bool = True,
    ) -> FloatArray:
        """Advance using supplied true-state inputs and return the final state."""
        checked = self._validated_inputs(inputs, allow_empty=True)
        if reset:
            self.reset_reservoir()
        for input_value in checked:
            self._advance(input_value)
        return self.reservoir_state.copy()

    def predict_one_step(self, input_value: np.ndarray) -> FloatArray:
        """Predict s_(t+1) after consuming one supplied u_t."""
        weights = self._require_fitted()
        checked = _vector(
            "one-step input", input_value, self.config.input_dimension
        )
        state = self._advance(checked)
        prediction = weights @ self._readout_feature(checked, state)
        return np.asarray(prediction, dtype=float)

    def autonomous_rollout(
        self,
        initial_state: np.ndarray,
        *,
        steps: int | None = None,
        current_values: np.ndarray | None = None,
        warmup_inputs: np.ndarray | None = None,
        reset: bool = True,
    ) -> FloatArray:
        """Recursively predict states following ``initial_state``.

        Warm-up rows are consumed as true inputs first. Prediction row 0 is
        s_hat_(k+1) from the supplied s_k. For a parameter-aware model,
        ``current_values[j]`` is the known I at rollout transition j. The
        entire current sequence is processed without inspecting switches or
        resetting the reservoir.
        """
        self._require_fitted()
        state = _vector(
            "initial_state", initial_state, self.config.output_dimension
        ).copy()
        if warmup_inputs is None:
            warmup = np.empty((0, self.config.input_dimension), dtype=float)
        else:
            warmup = self._validated_inputs(warmup_inputs, allow_empty=True)

        if self.config.input_dimension == PARAMETER_AWARE_INPUT_DIMENSION:
            if current_values is None:
                raise ModelValidationError(
                    "parameter-aware rollout requires supplied current_values"
                )
            currents = np.asarray(current_values)
            if currents.ndim != 1 or not np.issubdtype(currents.dtype, np.number):
                raise ModelValidationError(
                    "current_values must be a one-dimensional numeric array"
                )
            if not np.all(np.isfinite(currents)):
                raise ModelValidationError(
                    "current_values must contain only finite values"
                )
            currents = np.asarray(currents, dtype=float)
            if len(currents) == 0:
                raise ModelValidationError("current_values must not be empty")
            if steps is None:
                resolved_steps = len(currents)
            else:
                resolved_steps = _positive_integer("steps", steps)
                if resolved_steps != len(currents):
                    raise ModelValidationError(
                        "steps must equal the number of supplied current values"
                    )
        else:
            if current_values is not None:
                raise ModelValidationError(
                    "baseline rollout does not accept current_values"
                )
            if steps is None:
                raise ModelValidationError("baseline rollout requires steps")
            resolved_steps = _positive_integer("steps", steps)
            currents = np.empty(0, dtype=float)

        if reset:
            self.reset_reservoir()
        for input_value in warmup:
            self._advance(input_value)

        predictions = np.empty(
            (resolved_steps, self.config.output_dimension), dtype=float
        )
        for index in range(resolved_steps):
            if self.config.input_dimension == PARAMETER_AWARE_INPUT_DIMENSION:
                input_value = np.concatenate((state, [currents[index]]))
            else:
                input_value = state
            state = self.predict_one_step(input_value)
            predictions[index] = state
        return predictions

    def save(self, path: str | Path) -> None:
        """Serialize configuration and weights to a pickle-free NPZ bundle."""
        weights = self._require_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as file:
            np.savez_compressed(
                file,
                schema_version=np.asarray(_MODEL_SCHEMA),
                config_json=np.asarray(json.dumps(asdict(self.config), sort_keys=True)),
                input_weights=self.input_weights,
                reservoir_weights=self.reservoir_weights,
                reservoir_bias=self.reservoir_bias,
                output_weights=weights,
                reservoir_state=self.reservoir_state,
            )

    @classmethod
    def load(cls, path: str | Path) -> "EchoStateNetwork":
        """Load and validate a model bundle with ``allow_pickle=False``."""
        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as saved:
                required = {
                    "schema_version",
                    "config_json",
                    "input_weights",
                    "reservoir_weights",
                    "reservoir_bias",
                    "output_weights",
                    "reservoir_state",
                }
                if set(saved.files) != required:
                    raise ModelValidationError(
                        "saved model fields do not match the expected schema"
                    )
                schema = str(saved["schema_version"].item())
                config_data = json.loads(str(saved["config_json"].item()))
                arrays = {
                    name: np.asarray(saved[name], dtype=float).copy()
                    for name in required
                    if name not in {"schema_version", "config_json"}
                }
        except (OSError, ValueError, json.JSONDecodeError) as error:
            if isinstance(error, ModelValidationError):
                raise
            raise ModelValidationError(f"could not safely load model: {error}") from error

        if schema != _MODEL_SCHEMA:
            raise ModelValidationError(f"unsupported model schema {schema!r}")
        try:
            config = ESNModelConfig(**config_data)
        except (TypeError, ValueError) as error:
            raise ModelValidationError(
                f"saved model configuration is invalid: {error}"
            ) from error
        model = cls(config)
        expected_shapes = {
            "input_weights": (
                config.reservoir_size,
                config.input_dimension,
            ),
            "reservoir_weights": (
                config.reservoir_size,
                config.reservoir_size,
            ),
            "reservoir_bias": (config.reservoir_size,),
            "output_weights": (
                config.output_dimension,
                model.feature_dimension,
            ),
            "reservoir_state": (config.reservoir_size,),
        }
        for name, shape in expected_shapes.items():
            array = arrays[name]
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ModelValidationError(
                    f"saved {name} must have finite shape {shape}"
                )

        model.input_weights = arrays["input_weights"]
        model.reservoir_weights = arrays["reservoir_weights"]
        model.reservoir_bias = arrays["reservoir_bias"]
        model.output_weights = arrays["output_weights"]
        model.reservoir_state = arrays["reservoir_state"]
        if not np.isclose(
            model.spectral_radius,
            config.spectral_radius,
            rtol=1.0e-8,
            atol=1.0e-10,
        ):
            raise ModelValidationError(
                "saved reservoir does not match the configured spectral radius"
            )
        return model
