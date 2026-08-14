"""Synthetic tests for Chapter 2 rollout evaluation metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from chapter2.esn_metrics import (
    MetricValidationError,
    evaluate_rollout,
    pointwise_normalised_error,
)


def _evaluate(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    scale: np.ndarray | None = None,
    valid_threshold: float = 0.4,
    divergence_threshold: float = 5.0,
):
    return evaluate_rollout(
        predictions,
        targets,
        normalisation_scale=(
            np.ones(3) if scale is None else np.asarray(scale, dtype=float)
        ),
        dt=0.01,
        valid_prediction_threshold=valid_threshold,
        divergence_threshold=divergence_threshold,
    )


def test_perfect_rollout_has_zero_error_full_vpt_and_no_divergence() -> None:
    targets = np.arange(15, dtype=float).reshape(5, 3)

    metrics = _evaluate(targets.copy(), targets)

    assert metrics.rmse_per_state == (0.0, 0.0, 0.0)
    assert metrics.rmse_state == 0.0
    assert metrics.nrmse_per_state == (0.0, 0.0, 0.0)
    assert metrics.nrmse_state == 0.0
    assert metrics.valid_prediction_steps == 5
    assert metrics.valid_prediction_time == pytest.approx(0.05)
    assert not metrics.diverged
    assert metrics.divergence_index is None
    assert metrics.divergence_time is None
    assert metrics.divergence_reason is None


def test_rmse_and_nrmse_formulas_match_hand_calculation() -> None:
    targets = np.zeros((2, 3))
    predictions = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 3.0]])
    scale = np.array([1.0, 2.0, 3.0])

    metrics = _evaluate(
        predictions,
        targets,
        scale=scale,
        valid_threshold=2.0,
        divergence_threshold=10.0,
    )

    expected_rmse_per_state = (
        1.0,
        np.sqrt(2.0),
        3.0,
    )
    expected_nrmse_per_state = (
        1.0,
        np.sqrt(0.5),
        1.0,
    )
    np.testing.assert_allclose(metrics.rmse_per_state, expected_rmse_per_state)
    np.testing.assert_allclose(
        metrics.nrmse_per_state, expected_nrmse_per_state
    )
    assert metrics.rmse_state == pytest.approx(np.sqrt(4.0))
    assert metrics.nrmse_state == pytest.approx(np.sqrt(5.0 / 6.0))


def test_caller_supplied_scale_controls_normalisation_without_refitting() -> None:
    targets = np.full((4, 3), 7.0)
    predictions = targets + np.array([2.0, 4.0, 8.0])

    first = _evaluate(
        predictions,
        targets,
        scale=np.array([1.0, 2.0, 4.0]),
        valid_threshold=3.0,
        divergence_threshold=10.0,
    )
    second = _evaluate(
        predictions,
        targets,
        scale=np.array([2.0, 4.0, 8.0]),
        valid_threshold=3.0,
        divergence_threshold=10.0,
    )

    assert first.nrmse_per_state == pytest.approx((2.0, 2.0, 2.0))
    assert second.nrmse_per_state == pytest.approx((1.0, 1.0, 1.0))
    assert second.nrmse_state == pytest.approx(first.nrmse_state / 2.0)


def test_pointwise_normalised_error_and_vpt_use_exact_step_alignment() -> None:
    targets = np.zeros((4, 3))
    predictions = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.4, 0.4, 0.4],
            [0.3, 0.3, 0.3],
        ]
    )

    pointwise = pointwise_normalised_error(
        predictions, targets, normalisation_scale=np.ones(3)
    )
    metrics = _evaluate(predictions, targets, valid_threshold=0.4)

    np.testing.assert_allclose(pointwise, [0.1, 0.2, 0.4, 0.3])
    assert metrics.valid_prediction_steps == 2
    assert metrics.valid_prediction_time == pytest.approx(0.02)


def test_failure_on_first_prediction_has_zero_valid_prediction_time() -> None:
    targets = np.zeros((2, 3))
    predictions = np.full((2, 3), 0.4)

    metrics = _evaluate(predictions, targets, valid_threshold=0.4)

    assert metrics.valid_prediction_steps == 0
    assert metrics.valid_prediction_time == 0.0


def test_finite_error_threshold_divergence_records_index_time_and_reason() -> None:
    targets = np.zeros((4, 3))
    predictions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [0.0, 0.0, 0.0],
        ]
    )

    metrics = _evaluate(predictions, targets)

    assert metrics.diverged
    assert metrics.divergence_index == 2
    assert metrics.divergence_time == pytest.approx(0.03)
    assert metrics.divergence_reason == "normalised_error_threshold_reached"


@pytest.mark.parametrize("bad_value", (np.nan, np.inf, -np.inf))
def test_nonfinite_prediction_is_reported_and_json_safe(bad_value: float) -> None:
    targets = np.zeros((3, 3))
    predictions = targets.copy()
    predictions[1, 0] = bad_value

    metrics = _evaluate(predictions, targets)
    serialized = metrics.to_dict()

    assert metrics.diverged
    assert metrics.divergence_index == 1
    assert metrics.divergence_time == pytest.approx(0.02)
    assert metrics.divergence_reason == "non_finite_prediction"
    assert serialized["rmse_x"] is None
    assert serialized["nrmse_state"] is None
    json.dumps(serialized, allow_nan=False)


def test_metric_inputs_are_not_mutated() -> None:
    predictions = np.arange(12, dtype=float).reshape(4, 3)
    targets = predictions + 0.1
    scale = np.array([1.0, 2.0, 3.0])
    before_predictions = predictions.copy()
    before_targets = targets.copy()
    before_scale = scale.copy()

    _evaluate(predictions, targets, scale=scale)

    np.testing.assert_array_equal(predictions, before_predictions)
    np.testing.assert_array_equal(targets, before_targets)
    np.testing.assert_array_equal(scale, before_scale)


@pytest.mark.parametrize(
    ("predictions", "targets", "message"),
    (
        (np.empty((0, 3)), np.empty((0, 3)), "empty"),
        (np.zeros((2, 2)), np.zeros((2, 2)), "shape"),
        (np.zeros((2, 3)), np.zeros((3, 3)), "equal shape"),
        (np.zeros((2, 3)), np.full((2, 3), np.nan), "targets"),
    ),
)
def test_invalid_rollout_arrays_raise_clear_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(MetricValidationError, match=message):
        _evaluate(predictions, targets)


@pytest.mark.parametrize(
    "scale",
    (
        np.ones(2),
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, np.nan, 1.0]),
    ),
)
def test_invalid_normalisation_scale_is_rejected(scale: np.ndarray) -> None:
    with pytest.raises(MetricValidationError, match="normalisation_scale"):
        _evaluate(np.zeros((2, 3)), np.zeros((2, 3)), scale=scale)


@pytest.mark.parametrize(
    ("dt", "valid_threshold", "divergence_threshold", "message"),
    (
        (0.0, 0.4, 5.0, "dt"),
        (0.01, 0.0, 5.0, "valid_prediction_threshold"),
        (0.01, 0.4, 0.4, "greater"),
        (0.01, 1.0, 0.5, "greater"),
    ),
)
def test_invalid_time_or_threshold_configuration_is_rejected(
    dt: float,
    valid_threshold: float,
    divergence_threshold: float,
    message: str,
) -> None:
    with pytest.raises(MetricValidationError, match=message):
        evaluate_rollout(
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            normalisation_scale=np.ones(3),
            dt=dt,
            valid_prediction_threshold=valid_threshold,
            divergence_threshold=divergence_threshold,
        )


def test_dictionary_uses_three_state_outputs_only() -> None:
    result = _evaluate(np.zeros((2, 3)), np.zeros((2, 3))).to_dict()

    assert {"rmse_x", "rmse_y", "rmse_z"} <= set(result)
    assert {"nrmse_x", "nrmse_y", "nrmse_z"} <= set(result)
    assert not any(key.endswith("_I") for key in result)


def test_reporting_diagnostics_have_exact_per_state_and_macro_values() -> None:
    target_axis = np.array([-1.0, 0.0, 1.0])
    targets = np.column_stack((target_axis, target_axis, target_axis))
    predictions = np.column_stack(
        (target_axis, -target_axis, 2.0 * target_axis)
    )

    metrics = _evaluate(
        predictions,
        targets,
        valid_threshold=10.0,
        divergence_threshold=20.0,
    )

    assert metrics.r2_per_state == pytest.approx((1.0, -3.0, 0.0))
    assert metrics.r2_defined_per_state == (True, True, True)
    assert metrics.r2_macro == pytest.approx(-2.0 / 3.0)
    assert metrics.r2_macro_defined
    assert metrics.r2_macro_state_count == 3
    assert metrics.correlation_per_state == pytest.approx((1.0, -1.0, 1.0))
    assert metrics.correlation_macro == pytest.approx(1.0 / 3.0)
    assert metrics.correlation_macro_defined
    assert metrics.correlation_macro_state_count == 3
    expected_std = float(np.std(target_axis))
    assert metrics.prediction_std_per_state == pytest.approx(
        (expected_std, expected_std, 2.0 * expected_std)
    )
    assert metrics.target_std_per_state == pytest.approx(
        (expected_std, expected_std, expected_std)
    )
    assert metrics.std_ratio_per_state == pytest.approx((1.0, 1.0, 2.0))
    assert metrics.std_ratio_defined_per_state == (True, True, True)
    assert not metrics.prediction_collapse_any


def test_constant_target_diagnostics_are_undefined_and_json_safe() -> None:
    axis = np.array([-1.0, 0.0, 1.0])
    targets = np.column_stack((np.ones(3), axis, np.full(3, 4.0)))
    predictions = np.column_stack((axis, axis, np.ones(3)))

    metrics = _evaluate(
        predictions,
        targets,
        valid_threshold=10.0,
        divergence_threshold=20.0,
    )
    result = metrics.to_dict()

    assert metrics.r2_per_state[0] is None
    assert metrics.r2_per_state[2] is None
    assert metrics.r2_defined_per_state == (False, True, False)
    assert metrics.r2_macro == pytest.approx(1.0)
    assert metrics.r2_macro_defined
    assert metrics.r2_macro_state_count == 1
    assert metrics.correlation_defined_per_state == (False, True, False)
    assert metrics.correlation_macro == pytest.approx(1.0)
    assert metrics.correlation_macro_state_count == 1
    assert metrics.std_ratio_defined_per_state == (False, True, False)
    assert metrics.prediction_collapse_defined_per_state == (
        False,
        True,
        False,
    )
    assert result["r2_x"] is None
    assert result["correlation_z"] is None
    assert result["prediction_target_std_ratio_x"] is None
    json.dumps(result, allow_nan=False)


def test_all_constant_targets_produce_undefined_macros_with_zero_counts() -> None:
    targets = np.ones((4, 3))
    predictions = np.arange(12, dtype=float).reshape(4, 3)

    metrics = _evaluate(
        predictions,
        targets,
        valid_threshold=10.0,
        divergence_threshold=20.0,
    )

    assert metrics.r2_macro is None
    assert not metrics.r2_macro_defined
    assert metrics.r2_macro_state_count == 0
    assert metrics.correlation_macro is None
    assert not metrics.correlation_macro_defined
    assert metrics.correlation_macro_state_count == 0


def test_constant_prediction_is_collapse_and_correlation_is_undefined() -> None:
    axis = np.array([-1.0, 0.0, 1.0])
    targets = np.column_stack((axis, axis, axis))
    predictions = np.zeros((3, 3))

    metrics = _evaluate(
        predictions,
        targets,
        valid_threshold=10.0,
        divergence_threshold=20.0,
    )

    assert metrics.r2_defined_per_state == (True, True, True)
    assert metrics.r2_per_state == pytest.approx((0.0, 0.0, 0.0))
    assert metrics.correlation_defined_per_state == (False, False, False)
    assert metrics.correlation_macro is None
    assert metrics.std_ratio_per_state == pytest.approx((0.0, 0.0, 0.0))
    assert metrics.prediction_collapsed_per_state == (True, True, True)
    assert metrics.prediction_collapse_any
    assert metrics.prediction_collapse_state_count == 3
    assert metrics.prediction_collapse_defined_state_count == 3


def test_collapse_threshold_is_strictly_less_than_frozen_ratio() -> None:
    axis = np.array([-1.0, 0.0, 1.0])
    targets = np.column_stack((axis, axis, axis))
    predictions = targets * np.array([0.049, 0.05, 0.051])

    metrics = _evaluate(
        predictions,
        targets,
        valid_threshold=10.0,
        divergence_threshold=20.0,
    )

    assert metrics.collapse_std_ratio_threshold == 0.05
    assert metrics.prediction_collapsed_per_state == (True, False, False)
    assert metrics.prediction_collapse_state_count == 1


def test_nonfinite_reporting_diagnostics_are_explicitly_undefined() -> None:
    axis = np.array([-1.0, 0.0, 1.0])
    targets = np.column_stack((axis, axis, axis))
    predictions = targets.copy()
    predictions[1, 1] = np.inf

    metrics = _evaluate(predictions, targets)
    result = metrics.to_dict()

    assert metrics.r2_defined_per_state == (True, False, True)
    assert metrics.correlation_defined_per_state == (True, False, True)
    assert metrics.prediction_std_per_state[1] is None
    assert not metrics.std_ratio_defined_per_state[1]
    assert not metrics.prediction_collapse_defined_per_state[1]
    assert result["prediction_std_y"] is None
    json.dumps(result, allow_nan=False)


def test_reporting_metrics_are_deterministic_and_do_not_mutate_inputs() -> None:
    rng = np.random.default_rng(42)
    targets = rng.normal(size=(20, 3))
    predictions = targets + rng.normal(scale=0.1, size=(20, 3))
    original_predictions = predictions.copy()
    original_targets = targets.copy()

    first = _evaluate(predictions, targets).to_dict()
    second = _evaluate(predictions, targets).to_dict()

    assert first == second
    np.testing.assert_array_equal(predictions, original_predictions)
    np.testing.assert_array_equal(targets, original_targets)


def test_invalid_collapse_threshold_is_rejected() -> None:
    with pytest.raises(MetricValidationError, match="collapse_std_ratio_threshold"):
        evaluate_rollout(
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            normalisation_scale=np.ones(3),
            dt=0.01,
            valid_prediction_threshold=0.4,
            divergence_threshold=5.0,
            collapse_std_ratio_threshold=0.0,
        )
