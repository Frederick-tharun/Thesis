"""Unit tests for deterministic Chapter 2 ESN model mechanics.

All data in this module are synthetic. These tests verify mechanics and
leakage-safe interfaces; they do not establish biological prediction accuracy.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from chapter2.esn_config import ESNModelConfig
from chapter2.esn_model import (
    EchoStateNetwork,
    ModelNotFittedError,
    ModelValidationError,
    TrainingSequence,
)


def _config(
    *,
    input_dimension: int = 4,
    seed: int = 42,
    reservoir_size: int = 14,
    connectivity: float = 0.5,
    ridge: float = 1.0e-7,
    input_scaling: float = 0.4,
    bias_scaling: float = 0.1,
) -> ESNModelConfig:
    return ESNModelConfig(
        reservoir_size=reservoir_size,
        spectral_radius=0.8,
        leak_rate=0.45,
        input_scaling=input_scaling,
        bias_scaling=bias_scaling,
        reservoir_connectivity=connectivity,
        ridge_regularisation=ridge,
        seed=seed,
        input_dimension=input_dimension,
        output_dimension=3,
    )


def _synthetic_sequence(
    input_dimension: int,
    *,
    rows: int = 18,
    offset: float = 0.0,
) -> TrainingSequence:
    index = np.arange(rows, dtype=float) + offset
    states = np.column_stack(
        (
            np.sin(0.2 * index),
            np.cos(0.13 * index),
            0.03 * index,
        )
    )
    next_states = np.column_stack(
        (
            np.sin(0.2 * (index + 1.0)),
            np.cos(0.13 * (index + 1.0)),
            0.03 * (index + 1.0),
        )
    )
    if input_dimension == 4:
        currents = (1.5 + 0.02 * index).reshape(-1, 1)
        inputs = np.column_stack((states, currents))
    else:
        inputs = states
    return TrainingSequence(inputs, next_states)


def _set_direct_readout(
    model: EchoStateNetwork,
    state_multiplier: float = 1.0,
) -> None:
    weights = np.zeros((3, model.feature_dimension), dtype=float)
    weights[:, 1:4] = state_multiplier * np.eye(3)
    model.output_weights = weights
    model.reset_reservoir()


def _manual_statistics(
    model: EchoStateNetwork,
    sequences: tuple[TrainingSequence, ...],
    *,
    reset_each: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.reset_reservoir()
    for sequence in sequences:
        if reset_each:
            model.reset_reservoir()
        for input_value, target in zip(sequence.inputs, sequence.targets):
            state = model._advance(input_value)
            features.append(model._readout_feature(input_value, state))
            targets.append(target)
    feature_matrix = np.asarray(features)
    target_matrix = np.asarray(targets)
    return (
        feature_matrix.T @ feature_matrix,
        target_matrix.T @ feature_matrix,
        feature_matrix,
    )


def test_parameter_aware_dimensions_are_four_inputs_three_outputs() -> None:
    model = EchoStateNetwork(_config(input_dimension=4))
    sequence = _synthetic_sequence(4)

    model.fit((sequence,))

    assert model.input_weights.shape == (14, 4)
    assert model.output_weights is not None
    assert model.output_weights.shape == (3, 1 + 4 + 14)
    assert model.predict_one_step(sequence.inputs[0]).shape == (3,)


def test_baseline_dimensions_are_three_inputs_three_outputs() -> None:
    model = EchoStateNetwork(_config(input_dimension=3))
    sequence = _synthetic_sequence(3)

    model.fit((sequence,))

    assert model.input_weights.shape == (14, 3)
    assert model.output_weights is not None
    assert model.output_weights.shape == (3, 1 + 3 + 14)
    assert model.predict_one_step(sequence.inputs[0]).shape == (3,)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("input_dimension", 2, "input_dimension"),
        ("input_dimension", 5, "input_dimension"),
        ("output_dimension", 4, "output_dimension"),
        ("reservoir_size", 1, "reservoir_size"),
        ("spectral_radius", 0.0, "spectral_radius"),
        ("leak_rate", 0.0, "leak_rate"),
        ("leak_rate", 1.1, "leak_rate"),
        ("reservoir_connectivity", 0.0, "reservoir_connectivity"),
        ("ridge_regularisation", 0.0, "ridge_regularisation"),
        ("input_scaling", -0.1, "input_scaling"),
        ("bias_scaling", -0.1, "bias_scaling"),
    ),
)
def test_invalid_configuration_values_raise_clear_errors(
    field: str, value: float, message: str
) -> None:
    values = {
        "reservoir_size": 10,
        "spectral_radius": 0.8,
        "leak_rate": 0.5,
        "input_scaling": 0.4,
        "bias_scaling": 0.1,
        "reservoir_connectivity": 0.5,
        "ridge_regularisation": 1.0e-6,
        "seed": 42,
        "input_dimension": 4,
        "output_dimension": 3,
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        ESNModelConfig(**values)


def test_configuration_is_frozen() -> None:
    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.seed = 123  # type: ignore[misc]


def test_model_rejects_data_dimensions_that_do_not_match_configuration() -> None:
    parameter_aware = EchoStateNetwork(_config(input_dimension=4))
    baseline_sequence = _synthetic_sequence(3)

    with pytest.raises(ModelValidationError, match="dimension"):
        parameter_aware.fit((baseline_sequence,))
    parameter_aware.fit((_synthetic_sequence(4),))
    with pytest.raises(ModelValidationError, match="shape"):
        parameter_aware.predict_one_step(np.zeros(3))


def test_same_seed_and_configuration_produce_identical_weights() -> None:
    config = _config(seed=123)
    first = EchoStateNetwork(config)
    second = EchoStateNetwork(config)
    sequence = _synthetic_sequence(4)

    first.fit((sequence,))
    second.fit((sequence,))

    np.testing.assert_array_equal(first.input_weights, second.input_weights)
    np.testing.assert_array_equal(first.reservoir_bias, second.reservoir_bias)
    np.testing.assert_array_equal(first.reservoir_weights, second.reservoir_weights)
    np.testing.assert_array_equal(first.output_weights, second.output_weights)


def test_different_seeds_change_reservoir_initialisation() -> None:
    first = EchoStateNetwork(_config(seed=42))
    second = EchoStateNetwork(_config(seed=43))

    assert not np.array_equal(first.input_weights, second.input_weights)
    assert not np.array_equal(first.reservoir_weights, second.reservoir_weights)


def test_reservoir_spectral_radius_matches_configuration() -> None:
    model = EchoStateNetwork(_config(seed=456, reservoir_size=24))

    assert model.spectral_radius == pytest.approx(
        model.config.spectral_radius, rel=1.0e-10, abs=1.0e-12
    )


def test_bernoulli_connectivity_and_zero_diagonal_are_respected() -> None:
    requested = 0.2
    model = EchoStateNetwork(
        _config(seed=789, reservoir_size=80, connectivity=requested)
    )

    np.testing.assert_array_equal(
        np.diag(model.reservoir_weights), np.zeros(model.config.reservoir_size)
    )
    assert model.realised_connectivity == pytest.approx(requested, abs=0.035)


def test_reservoir_reset_restores_exact_zero_state() -> None:
    model = EchoStateNetwork(_config())
    model.teacher_forced_warmup(_synthetic_sequence(4).inputs[:3])

    assert np.any(model.reservoir_state != 0.0)
    model.reset_reservoir()
    np.testing.assert_array_equal(
        model.reservoir_state, np.zeros(model.config.reservoir_size)
    )


def test_teacher_forced_warmup_uses_supplied_rows_in_order() -> None:
    model = EchoStateNetwork(_config(input_dimension=3))
    inputs = _synthetic_sequence(3).inputs[:2]
    expected = np.zeros(model.config.reservoir_size)
    for input_value in inputs:
        expected = (
            (1.0 - model.config.leak_rate) * expected
            + model.config.leak_rate
            * np.tanh(
                model.input_weights @ input_value
                + model.reservoir_weights @ expected
                + model.reservoir_bias
            )
        )

    actual = model.teacher_forced_warmup(inputs)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-15)


def test_independent_training_sequences_are_reset_not_false_connected() -> None:
    config = _config(input_dimension=4, seed=2026)
    sequences = (
        _synthetic_sequence(4, rows=9, offset=0.0),
        _synthetic_sequence(4, rows=8, offset=50.0),
    )
    model = EchoStateNetwork(config)
    statistics = model.accumulate_ridge_statistics(sequences)
    reset_model = EchoStateNetwork(config)
    reset_gram, reset_cross, _ = _manual_statistics(
        reset_model, sequences, reset_each=True
    )
    connected_model = EchoStateNetwork(config)
    connected_gram, _, _ = _manual_statistics(
        connected_model, sequences, reset_each=False
    )

    np.testing.assert_allclose(statistics.gram, reset_gram)
    np.testing.assert_allclose(statistics.cross, reset_cross)
    assert not np.allclose(statistics.gram, connected_gram)
    np.testing.assert_array_equal(
        model.reservoir_state, np.zeros(model.config.reservoir_size)
    )


def test_training_rows_have_exact_input_to_same_row_target_alignment() -> None:
    model = EchoStateNetwork(_config(input_dimension=3, seed=123))
    sequence = _synthetic_sequence(3, rows=7)
    statistics = model.accumulate_ridge_statistics((sequence,))
    manual = EchoStateNetwork(model.config)
    gram, cross, features = _manual_statistics(
        manual, (sequence,), reset_each=True
    )

    np.testing.assert_allclose(statistics.gram, gram)
    np.testing.assert_allclose(statistics.cross, cross)
    np.testing.assert_allclose(
        statistics.cross,
        sum(
            np.outer(sequence.targets[index], features[index])
            for index in range(len(sequence.inputs))
        ),
    )


def test_bias_is_explicitly_unregularised_by_default() -> None:
    model = EchoStateNetwork(_config())

    assert model.ridge_penalty_matrix[0, 0] == 0.0
    np.testing.assert_array_equal(
        np.diag(model.ridge_penalty_matrix)[1:],
        np.ones(model.feature_dimension - 1),
    )

    regularised = ESNModelConfig(
        **{
            **model.config.__dict__,
            "regularise_bias": True,
        }
    )
    assert EchoStateNetwork(regularised).ridge_penalty_matrix[0, 0] == 1.0


def test_streaming_ridge_matches_small_direct_batch_solution() -> None:
    config = _config(input_dimension=4, seed=42, ridge=2.0e-5)
    sequences = (
        _synthetic_sequence(4, rows=10),
        _synthetic_sequence(4, rows=11, offset=30.0),
    )
    model = EchoStateNetwork(config)
    statistics = model.accumulate_ridge_statistics(sequences, washout=1)
    direct_model = EchoStateNetwork(config)
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for sequence in sequences:
        direct_model.reset_reservoir()
        for index, (input_value, target) in enumerate(
            zip(sequence.inputs, sequence.targets)
        ):
            state = direct_model._advance(input_value)
            if index >= 1:
                feature_rows.append(
                    direct_model._readout_feature(input_value, state)
                )
                target_rows.append(target)
    features = np.asarray(feature_rows)
    targets = np.asarray(target_rows)
    direct = np.linalg.solve(
        features.T @ features
        + config.ridge_regularisation * model.ridge_penalty_matrix,
        features.T @ targets,
    ).T

    model.fit(sequences, washout=1)

    np.testing.assert_allclose(statistics.gram, features.T @ features)
    np.testing.assert_allclose(statistics.cross, targets.T @ features)
    np.testing.assert_allclose(model.output_weights, direct, rtol=1.0e-10, atol=1.0e-11)


def test_autonomous_rollout_shape_first_step_and_recursive_feedback() -> None:
    model = EchoStateNetwork(_config(input_dimension=3))
    _set_direct_readout(model, state_multiplier=2.0)

    predictions = model.autonomous_rollout(
        np.array([1.0, 2.0, 3.0]), steps=3
    )

    assert predictions.shape == (3, 3)
    np.testing.assert_array_equal(predictions[0], [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(predictions[1], [4.0, 8.0, 12.0])
    np.testing.assert_array_equal(predictions[2], [8.0, 16.0, 24.0])


def test_parameter_aware_rollout_uses_each_supplied_current() -> None:
    model = EchoStateNetwork(_config(input_dimension=4))
    weights = np.zeros((3, model.feature_dimension))
    current_feature_index = 1 + 3
    weights[0, current_feature_index] = 1.0
    model.output_weights = weights
    currents = np.array([1.1, -0.4, 2.3])

    predictions = model.autonomous_rollout(
        np.zeros(3), current_values=currents
    )

    np.testing.assert_array_equal(predictions[:, 0], currents)
    np.testing.assert_array_equal(predictions[:, 1:], np.zeros((3, 2)))
    assert predictions.shape[1] == 3


def test_continuous_current_switch_does_not_reset_reservoir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = EchoStateNetwork(_config(input_dimension=4))
    _set_direct_readout(model)
    reset_calls = 0
    original_reset = model.reset_reservoir

    def counted_reset() -> None:
        nonlocal reset_calls
        reset_calls += 1
        original_reset()

    monkeypatch.setattr(model, "reset_reservoir", counted_reset)
    model.autonomous_rollout(
        np.array([0.2, -0.1, 0.3]),
        current_values=np.array([1.67, 1.67, 3.29, 3.29]),
    )

    assert reset_calls == 1


def test_independent_rollouts_reset_by_default() -> None:
    model = EchoStateNetwork(_config(input_dimension=4))
    model.fit((_synthetic_sequence(4),))
    initial = np.array([0.1, 0.2, 0.3])
    currents = np.full(4, 1.67)

    first = model.autonomous_rollout(initial, current_values=currents)
    model.teacher_forced_warmup(_synthetic_sequence(4).inputs[:5], reset=False)
    second = model.autonomous_rollout(initial, current_values=currents)

    np.testing.assert_array_equal(second, first)


def test_fitting_does_not_mutate_input_or_target_arrays() -> None:
    sequence = _synthetic_sequence(4)
    inputs = sequence.inputs.copy()
    targets = sequence.targets.copy()
    before_inputs = inputs.copy()
    before_targets = targets.copy()
    model = EchoStateNetwork(_config(input_dimension=4))

    model.fit(((inputs, targets),))

    np.testing.assert_array_equal(inputs, before_inputs)
    np.testing.assert_array_equal(targets, before_targets)


@pytest.mark.parametrize("bad_value", (np.nan, np.inf, -np.inf))
def test_nonfinite_training_values_are_rejected(bad_value: float) -> None:
    sequence = _synthetic_sequence(4)
    inputs = sequence.inputs.copy()
    inputs[2, 1] = bad_value

    with pytest.raises(ModelValidationError, match="finite"):
        EchoStateNetwork(_config()).fit(((inputs, sequence.targets),))


def test_empty_and_insufficient_training_sequences_are_rejected() -> None:
    model = EchoStateNetwork(_config())

    with pytest.raises(ModelValidationError, match="at least one"):
        model.fit(())
    with pytest.raises(ModelValidationError, match="at least two"):
        TrainingSequence(np.zeros((1, 4)), np.zeros((1, 3)))
    with pytest.raises(ModelValidationError, match="equal lengths"):
        TrainingSequence(np.zeros((3, 4)), np.zeros((2, 3)))


def test_invalid_rollout_inputs_are_rejected() -> None:
    model = EchoStateNetwork(_config(input_dimension=4))
    _set_direct_readout(model)

    with pytest.raises(ModelValidationError, match="current_values"):
        model.autonomous_rollout(np.zeros(3), current_values=np.array([]))
    with pytest.raises(ModelValidationError, match="finite"):
        model.autonomous_rollout(
            np.zeros(3), current_values=np.array([1.0, np.nan])
        )
    with pytest.raises(ModelValidationError, match="initial_state"):
        model.autonomous_rollout(np.zeros(4), current_values=np.array([1.0]))
    with pytest.raises(ModelValidationError, match="steps"):
        model.autonomous_rollout(
            np.zeros(3), steps=2, current_values=np.array([1.0])
        )


def test_prediction_before_fitting_raises_clear_error() -> None:
    model = EchoStateNetwork(_config())

    with pytest.raises(ModelNotFittedError, match="not fitted"):
        model.predict_one_step(np.zeros(4))
    with pytest.raises(ModelNotFittedError, match="not fitted"):
        model.autonomous_rollout(
            np.zeros(3), current_values=np.ones(2)
        )


def test_save_load_preserves_configuration_weights_state_and_predictions(
    tmp_path,
) -> None:
    model = EchoStateNetwork(_config(input_dimension=4, seed=123))
    sequence = _synthetic_sequence(4)
    model.fit((sequence,))
    model.teacher_forced_warmup(sequence.inputs[:3])
    saved_state = model.reservoir_state.copy()
    path = tmp_path / "model.npz"

    model.save(path)
    with np.load(path, allow_pickle=False) as bundle:
        assert all(bundle[name].dtype.kind != "O" for name in bundle.files)
    loaded = EchoStateNetwork.load(path)

    assert loaded.config == model.config
    np.testing.assert_array_equal(loaded.input_weights, model.input_weights)
    np.testing.assert_array_equal(
        loaded.reservoir_weights, model.reservoir_weights
    )
    np.testing.assert_array_equal(loaded.reservoir_bias, model.reservoir_bias)
    np.testing.assert_array_equal(loaded.output_weights, model.output_weights)
    np.testing.assert_array_equal(loaded.reservoir_state, saved_state)

    initial = np.array([0.1, -0.2, 0.3])
    currents = np.array([1.67, 3.29, 3.50])
    expected = model.autonomous_rollout(initial, current_values=currents)
    actual = loaded.autonomous_rollout(initial, current_values=currents)
    np.testing.assert_array_equal(actual, expected)
