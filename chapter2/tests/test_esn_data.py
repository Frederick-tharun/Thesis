"""Data-safety and leakage tests for the frozen Chapter 2 ESN protocol."""

from __future__ import annotations

from dataclasses import asdict, fields
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from chapter2 import esn_data as esn_data_module
from chapter2.esn_config import (
    CONTINUOUS_DATASET,
    CONTINUOUS_SEQUENCE,
    CONTINUOUS_SWITCH_INDICES,
    FINAL_SEEDS,
    FITTING_TRANSITIONS,
    FIXED_DATASETS,
    FIXED_STATE_COUNT,
    LOCKED_DATASETS,
    PARAMETER_AWARE_INPUT_DIMENSION,
    SCORED_ROLLOUT_TRANSITIONS,
    STATE_DIMENSION,
    TRAIN_CURRENTS,
    UNSEEN_CURRENTS,
    VALIDATION_TRANSITIONS,
    VALIDATION_WINDOWS,
    WARMUP_TRANSITIONS,
    TransitionRange,
    transition_split,
    validate_non_overlapping,
)
from chapter2.esn_data import (
    ContinuousCurrentTrajectory,
    DataValidationError,
    DatasetIntegrityError,
    FixedCurrentTrajectory,
    concatenate_one_step_pairs,
    create_one_step_pairs,
    file_sha256,
    fit_training_scalers,
    load_continuous_benchmark,
    load_fixed_trajectory,
    load_fixed_trajectory_file,
    load_optimisation_data,
    load_optimization_data,
    load_seen_current_held_out,
    load_unseen_benchmarks,
    locked_dataset_hashes,
    prepare_fixed_trajectory,
    scale_one_step_pairs,
)


def _synthetic_fixed(current: float, state_count: int = 8) -> FixedCurrentTrajectory:
    time = np.arange(state_count, dtype=float) * 0.01
    index = np.arange(state_count, dtype=float)
    states = np.column_stack((index, index + 100.0, -2.0 * index))
    currents = np.full(state_count, current, dtype=float)
    return FixedCurrentTrajectory(current, time, states, currents)


def _valid_npz_arrays(state_count: int = 8, current: float = 1.67) -> dict[str, np.ndarray]:
    index = np.arange(state_count, dtype=float)
    return {
        "t": index * 0.01,
        "x": index,
        "y": index + 10.0,
        "z": -index,
        "I": np.full(state_count, current, dtype=float),
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    np.savez(path, **arrays)
    return sha256(path.read_bytes()).hexdigest()


def test_frozen_protocol_constants_and_project_relative_paths() -> None:
    assert TRAIN_CURRENTS == (1.67, 3.20, 3.50)
    assert UNSEEN_CURRENTS == (3.29, 3.34)
    assert CONTINUOUS_SEQUENCE == (1.67, 3.29, 3.50, 3.34, 3.20)
    assert FINAL_SEEDS == (42, 123, 456, 789, 2026)
    assert all(record.path.is_absolute() for record in LOCKED_DATASETS)
    assert all("chapter2/outputs/data" in record.path.as_posix() for record in LOCKED_DATASETS)


def test_main_and_ordinary_input_shapes_with_three_state_targets() -> None:
    trajectory = _synthetic_fixed(1.67)
    main = create_one_step_pairs(trajectory)
    ordinary = create_one_step_pairs(trajectory, include_current=False)

    assert main.inputs.shape == (7, PARAMETER_AWARE_INPUT_DIMENSION)
    assert ordinary.inputs.shape == (7, STATE_DIMENSION)
    assert main.targets.shape == ordinary.targets.shape == (7, STATE_DIMENSION)


def test_exact_t_to_t_plus_one_alignment_and_supplied_current_column() -> None:
    trajectory = _synthetic_fixed(3.20)
    pairs = create_one_step_pairs(trajectory, TransitionRange(2, 6))

    np.testing.assert_array_equal(pairs.transition_indices, np.arange(2, 6))
    np.testing.assert_array_equal(pairs.inputs[:, :3], trajectory.states[2:6])
    np.testing.assert_array_equal(pairs.targets, trajectory.states[3:7])
    np.testing.assert_array_equal(pairs.inputs[:, 3], trajectory.current_values[2:6])
    np.testing.assert_array_equal(pairs.inputs[:, 3], np.full(4, 3.20))


def test_pairs_are_created_before_concatenation_without_false_transition() -> None:
    first = _synthetic_fixed(1.67, 5)
    second = _synthetic_fixed(3.50, 6)
    first_pairs = create_one_step_pairs(first)
    second_pairs = create_one_step_pairs(second)
    combined = concatenate_one_step_pairs((first_pairs, second_pairs))

    assert len(combined) == (first.state_count - 1) + (second.state_count - 1)
    np.testing.assert_array_equal(combined.targets[3], first.states[-1])
    np.testing.assert_array_equal(combined.inputs[4, :3], second.states[0])
    assert not np.array_equal(combined.targets[3], second.states[0])


def test_exact_split_counts_and_no_transition_overlap() -> None:
    split = transition_split(FIXED_STATE_COUNT)

    assert len(split.fitting) == 40_000
    assert len(split.validation) == 30_000
    assert len(split.held_out) == 29_999
    assert split.held_out == TransitionRange(70_000, 99_999)
    fitting = set(range(split.fitting.start, split.fitting.stop))
    validation = set(range(split.validation.start, split.validation.stop))
    held_out = set(range(split.held_out.start, split.held_out.stop))
    assert fitting.isdisjoint(validation)
    assert fitting.isdisjoint(held_out)
    assert validation.isdisjoint(held_out)
    assert len(fitting | validation | held_out) == FIXED_STATE_COUNT - 1


def test_frozen_validation_window_boundaries_lengths_and_non_overlap() -> None:
    assert len(VALIDATION_WINDOWS) == 3
    assert tuple(window.transitions for window in VALIDATION_WINDOWS) == (
        TransitionRange(40_000, 50_000),
        TransitionRange(50_000, 60_000),
        TransitionRange(60_000, 70_000),
    )
    validate_non_overlapping(window.transitions for window in VALIDATION_WINDOWS)
    assert all(len(window.warmup) == WARMUP_TRANSITIONS for window in VALIDATION_WINDOWS)
    assert all(
        len(window.scored) == SCORED_ROLLOUT_TRANSITIONS
        for window in VALIDATION_WINDOWS
    )
    transition_indices = [
        set(range(window.transitions.start, window.transitions.stop))
        for window in VALIDATION_WINDOWS
    ]
    assert transition_indices[0].isdisjoint(transition_indices[1])
    assert transition_indices[0].isdisjoint(transition_indices[2])
    assert transition_indices[1].isdisjoint(transition_indices[2])


def test_invalid_or_overlapping_protocol_ranges_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_non_overlapping(
            (TransitionRange(0, 10), TransitionRange(9, 20)), label="test ranges"
        )
    with pytest.raises(ValueError, match="greater than start"):
        TransitionRange(5, 5)
    with pytest.raises(ValueError, match="at least 70002 states"):
        transition_split(70_001)


def test_optimisation_data_contains_three_windows_for_each_training_current() -> None:
    prepared = load_optimisation_data()

    assert tuple(item.current for item in prepared) == TRAIN_CURRENTS
    assert sum(len(item.validation_windows) for item in prepared) == 9
    for item in prepared:
        assert len(item.fitting) == 40_000
        assert len(item.validation_windows) == 3
        for view in item.validation_windows:
            assert len(view.warmup) == 2_000
            assert len(view.scored) == 8_000


def test_optimisation_result_has_no_held_out_field_property_or_arrays() -> None:
    prepared = load_optimisation_data()

    for item in prepared:
        assert not hasattr(item, "held_out")
        assert {field.name for field in fields(item)} == {
            "current",
            "fitting",
            "validation_windows",
        }
        serialized = asdict(item)
        assert "held_out" not in serialized
        assert set(serialized) == {"current", "fitting", "validation_windows"}


def test_optimisation_api_contains_only_training_current_metadata() -> None:
    prepared = load_optimisation_data()

    assert tuple(item.current for item in prepared) == TRAIN_CURRENTS
    for item in prepared:
        assert item.fitting.source_current in TRAIN_CURRENTS
        assert all(
            view.warmup.source_current in TRAIN_CURRENTS
            and view.scored.source_current in TRAIN_CURRENTS
            for view in item.validation_windows
        )


def test_seen_current_held_out_requires_explicit_benchmark_loader() -> None:
    optimisation = load_optimisation_data()
    held_out = load_seen_current_held_out()

    assert all(not hasattr(item, "held_out") for item in optimisation)
    assert tuple(item.current for item in held_out) == TRAIN_CURRENTS
    assert all(len(item.pairs) == 29_999 for item in held_out)
    assert all(
        np.all(item.pairs.transition_indices >= 70_000) for item in held_out
    )


def test_optimisation_loader_never_calls_any_benchmark_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_loader(*args: object, **kwargs: object) -> None:
        raise AssertionError("optimisation attempted to open benchmark data")

    monkeypatch.setattr(
        esn_data_module, "load_seen_current_held_out", forbidden_loader
    )
    monkeypatch.setattr(esn_data_module, "load_unseen_benchmarks", forbidden_loader)
    monkeypatch.setattr(esn_data_module, "load_continuous_benchmark", forbidden_loader)

    prepared = esn_data_module.load_optimisation_data((1.67,))

    assert len(prepared) == 1
    assert prepared[0].current == 1.67
    assert not hasattr(prepared[0], "held_out")
    assert not hasattr(prepared[0], "unseen")
    assert not hasattr(prepared[0], "continuous")


@pytest.mark.parametrize("forbidden_current", UNSEEN_CURRENTS)
def test_unseen_current_is_rejected_by_optimisation_api(forbidden_current: float) -> None:
    with pytest.raises(DataValidationError, match="only training currents"):
        load_optimisation_data((1.67, forbidden_current))


def test_optimisation_api_rejects_continuous_and_duplicate_requests() -> None:
    with pytest.raises(DataValidationError, match="only training currents"):
        load_optimization_data(CONTINUOUS_SEQUENCE)
    with pytest.raises(DataValidationError, match="duplicates"):
        load_optimisation_data((1.67, 1.67))


def test_unseen_and_continuous_benchmark_loaders_are_separate() -> None:
    unseen = load_unseen_benchmarks()
    continuous = load_continuous_benchmark()

    assert tuple(item.current for item in unseen) == UNSEEN_CURRENTS
    assert continuous.current_sequence == CONTINUOUS_SEQUENCE
    assert continuous.switch_indices == CONTINUOUS_SWITCH_INDICES
    detected = tuple(
        (np.flatnonzero(np.diff(continuous.current_values) != 0.0) + 1).tolist()
    )
    assert detected == CONTINUOUS_SWITCH_INDICES


def test_scalers_fit_exactly_the_permitted_fitting_transitions() -> None:
    trajectories = tuple(load_fixed_trajectory(current) for current in TRAIN_CURRENTS)
    scalers = fit_training_scalers(trajectories)
    fitting_states = np.concatenate(
        [
            item.states[FITTING_TRANSITIONS.start : FITTING_TRANSITIONS.stop]
            for item in trajectories
        ]
    )
    fitting_currents = np.concatenate(
        [
            item.current_values[
                FITTING_TRANSITIONS.start : FITTING_TRANSITIONS.stop
            ]
            for item in trajectories
        ]
    )

    np.testing.assert_allclose(scalers.state.mean, np.mean(fitting_states, axis=0))
    np.testing.assert_allclose(scalers.state.scale, np.std(fitting_states, axis=0))
    np.testing.assert_allclose(scalers.current.mean, [np.mean(fitting_currents)])
    np.testing.assert_allclose(scalers.current.scale, [np.std(fitting_currents)])


def test_scaler_fitting_rejects_unseen_or_continuous_data() -> None:
    unseen = load_unseen_benchmarks()
    with pytest.raises(DataValidationError, match="exactly the training currents"):
        fit_training_scalers(unseen)

    continuous = load_continuous_benchmark()
    with pytest.raises(DataValidationError, match="continuous data is forbidden"):
        fit_training_scalers((continuous,))  # type: ignore[arg-type]


def test_shared_state_scaling_and_inverse_transform_recover_original_values() -> None:
    trajectories = tuple(load_fixed_trajectory(current) for current in TRAIN_CURRENTS)
    scalers = fit_training_scalers(trajectories)
    pairs = create_one_step_pairs(trajectories[0], TransitionRange(123, 456))
    scaled = scale_one_step_pairs(pairs, scalers)

    np.testing.assert_allclose(
        scalers.inverse_states(scaled.inputs[:, :3]), pairs.inputs[:, :3], rtol=1e-13
    )
    np.testing.assert_allclose(
        scalers.current.inverse_transform(scaled.inputs[:, 3:]),
        pairs.inputs[:, 3:],
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        scalers.inverse_states(scaled.targets), pairs.targets, rtol=1e-13
    )


def test_same_inputs_produce_deterministic_prepared_arrays() -> None:
    trajectory = load_fixed_trajectory(3.20)
    first = prepare_fixed_trajectory(trajectory)
    second = prepare_fixed_trajectory(trajectory)

    for first_view, second_view in ((first.fitting, second.fitting),):
        np.testing.assert_array_equal(first_view.inputs, second_view.inputs)
        np.testing.assert_array_equal(first_view.targets, second_view.targets)
        np.testing.assert_array_equal(
            first_view.transition_indices, second_view.transition_indices
        )
    for first_window, second_window in zip(
        first.validation_windows, second.validation_windows
    ):
        for first_view, second_view in (
            (first_window.warmup, second_window.warmup),
            (first_window.scored, second_window.scored),
        ):
            np.testing.assert_array_equal(first_view.inputs, second_view.inputs)
            np.testing.assert_array_equal(first_view.targets, second_view.targets)
            np.testing.assert_array_equal(
                first_view.transition_indices, second_view.transition_indices
            )


def test_real_locked_file_keys_shapes_dtypes_currents_and_hashes() -> None:
    assert locked_dataset_hashes() == {
        record.path: record.sha256 for record in LOCKED_DATASETS
    }
    for record in FIXED_DATASETS:
        with np.load(record.path, allow_pickle=False) as saved:
            assert tuple(saved.files) == ("t", "x", "y", "z", "I")
            assert all(saved[name].shape == (100_000,) for name in saved.files)
            assert all(saved[name].dtype == np.dtype("float64") for name in saved.files)
            assert np.all(saved["I"] == record.current)
    with np.load(CONTINUOUS_DATASET.path, allow_pickle=False) as saved:
        assert tuple(saved.files) == ("t", "x", "y", "z", "I")
        assert all(saved[name].shape == (500_000,) for name in saved.files)
        assert all(saved[name].dtype == np.dtype("float64") for name in saved.files)


def test_hash_mismatch_is_rejected_before_loading(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.npz"
    _write_npz(path, _valid_npz_arrays())

    with pytest.raises(DatasetIntegrityError, match="SHA-256 mismatch"):
        load_fixed_trajectory_file(
            path,
            expected_current=1.67,
            expected_sha256="0" * 64,
            expected_state_count=8,
        )


@pytest.mark.parametrize(
    ("malformation", "message"),
    (
        ("missing_key", "keys must be"),
        ("wrong_shape", "must be 1-dimensional"),
        ("nan", "finite values"),
        ("wrong_current", "must contain only I=1.67"),
        ("nonincreasing_time", "strictly increasing"),
        ("inconsistent_length", "must contain 8 states"),
        ("nonnumeric", "numeric dtype"),
    ),
)
def test_malformed_locked_files_raise_clear_errors(
    tmp_path: Path,
    malformation: str,
    message: str,
) -> None:
    arrays = _valid_npz_arrays()
    if malformation == "missing_key":
        arrays.pop("z")
    elif malformation == "wrong_shape":
        arrays["x"] = arrays["x"].reshape(-1, 1)
    elif malformation == "nan":
        arrays["y"][3] = np.nan
    elif malformation == "wrong_current":
        arrays["I"][4] = 3.29
    elif malformation == "nonincreasing_time":
        arrays["t"][4] = arrays["t"][3]
    elif malformation == "inconsistent_length":
        arrays["z"] = arrays["z"][:-1]
    elif malformation == "nonnumeric":
        arrays["x"] = np.full(8, "invalid")

    path = tmp_path / f"{malformation}.npz"
    expected_hash = _write_npz(path, arrays)
    with pytest.raises(DataValidationError, match=message):
        load_fixed_trajectory_file(
            path,
            expected_current=1.67,
            expected_sha256=expected_hash,
            expected_state_count=8,
        )


def test_insufficient_trajectory_length_raises_before_split_preparation() -> None:
    with pytest.raises(ValueError, match="at least 70002 states"):
        prepare_fixed_trajectory(_synthetic_fixed(1.67, 100))


def test_incorrect_continuous_order_or_switch_information_raises() -> None:
    time = np.arange(8, dtype=float)
    states = np.column_stack((time, time, time))
    currents = np.repeat([1.67, 3.29], 4)

    with pytest.raises(DataValidationError, match="switch indices"):
        ContinuousCurrentTrajectory(
            (1.67, 3.29), (5,), time, states, currents
        )
    with pytest.raises(DataValidationError, match="current order"):
        ContinuousCurrentTrajectory(
            (1.67, 3.34), (4,), time, states, currents
        )


def test_all_data_preparation_leaves_every_locked_npz_unchanged() -> None:
    before = {record.path: file_sha256(record.path) for record in LOCKED_DATASETS}

    training = tuple(load_fixed_trajectory(current) for current in TRAIN_CURRENTS)
    load_unseen_benchmarks()
    continuous = load_continuous_benchmark()
    prepared = load_optimisation_data()
    load_optimization_data((1.67,), include_current=False)
    create_one_step_pairs(continuous, TransitionRange(99_998, 100_002))
    concatenated = concatenate_one_step_pairs(
        tuple(item.fitting for item in prepared)
    )
    scalers = fit_training_scalers(training)
    scale_one_step_pairs(concatenated, scalers)
    locked_dataset_hashes()

    after = {record.path: file_sha256(record.path) for record in LOCKED_DATASETS}
    assert after == before == {
        record.path: record.sha256 for record in LOCKED_DATASETS
    }


def test_protocol_split_constants_are_exactly_the_documented_ranges() -> None:
    assert FITTING_TRANSITIONS == TransitionRange(0, 40_000)
    assert VALIDATION_TRANSITIONS == TransitionRange(40_000, 70_000)
