"""Synthetic tests for shared post-forecast classification, without models."""

import json

import numpy as np
import pytest

from chapter2.cross_regime_numerics import (
    NUMERICAL_POLICY,
    evaluate_predictions,
    metric_prediction_view,
)
from chapter2.esn_metrics import evaluate_rollout, pointwise_normalised_error
from chapter2.esn_optimisation import NONFINITE_FAILURE_SCORE


def test_ordinary_finite_metrics_and_pointwise_remain_exactly_unchanged():
    rng = np.random.default_rng(19)
    targets = rng.normal(size=(40, 3))
    predictions = targets + rng.normal(scale=0.2, size=targets.shape)
    scale = np.array([0.5, 1.0, 3.0])
    expected_metrics = evaluate_rollout(
        predictions, targets, normalisation_scale=scale, dt=0.01,
        valid_prediction_threshold=0.4, divergence_threshold=5.0,
        collapse_std_ratio_threshold=0.05,
    ).to_dict()
    result, physical, pointwise = evaluate_predictions(predictions, targets, scale)
    assert result["metrics"] == expected_metrics
    assert result["failure_step"] is None
    assert not result["numerical_failure"]
    assert result["aggregate_nrmse_value"] == expected_metrics["nrmse_state"]
    np.testing.assert_array_equal(physical, predictions)
    np.testing.assert_array_equal(pointwise, pointwise_normalised_error(
        predictions, targets, normalisation_scale=scale,
    ))


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_physical_failure_preserves_prefix_and_canonicalises_copy_only(bad):
    predictions = np.array([[0.1] * 3, [0.2] * 3, [bad, 1.0, 2.0], [3.0] * 3])
    original = predictions.copy()
    with np.errstate(over="raise", invalid="raise"):
        result, physical, pointwise = evaluate_predictions(
            predictions, np.zeros_like(predictions), np.ones(3)
        )
    np.testing.assert_array_equal(predictions, original)
    np.testing.assert_array_equal(physical[:2], original[:2])
    assert np.isnan(physical[2:]).all()
    assert not np.isinf(physical).any()
    assert result["failure_step"] == result["physical_failure_step"] == 2
    assert result["failure_reason"] == "non_finite_physical_prediction"
    assert result["valid_prefix_steps"] == 2
    assert result["metrics"]["valid_prediction_steps"] == 2
    assert result["metrics"]["divergence_index"] == 2
    assert result["aggregate_nrmse_value"] == NONFINITE_FAILURE_SCORE
    np.testing.assert_array_equal(pointwise[:2], [0.1, 0.2])
    assert np.isinf(pointwise[2:]).all()
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "value,target,scale,reason",
    [
        (np.finfo(float).max, -np.finfo(float).max, 1.0,
         "metric_unsafe_residual_subtraction"),
        (1e200, 0.0, 1.0, "metric_unsafe_residual_square"),
        (1e150, 0.0, 1e-20, "metric_unsafe_normalised_residual_square"),
        (1.0, 0.0, np.nextafter(0.0, 1.0), "metric_unsafe_normalisation"),
    ],
)
def test_metric_unsafe_finite_values_fail_closed_deterministically(value, target, scale, reason):
    predictions = np.zeros((4, 3))
    targets = np.zeros_like(predictions)
    predictions[2, 0], targets[2, 0] = value, target
    scales = np.array([scale, 1.0, 1.0])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result, physical, pointwise = evaluate_predictions(predictions, targets, scales)
        repeated, _, _ = evaluate_predictions(predictions, targets, scales)
    assert result == repeated
    assert result["numerical_policy"] == NUMERICAL_POLICY
    assert result["failure_step"] == 2
    assert result["physical_failure_step"] is None
    assert result["failure_reason"] == reason
    assert result["numerical_failure"]
    assert result["aggregate_nrmse_value"] == NONFINITE_FAILURE_SCORE
    assert result["metrics"]["nrmse_state"] is None
    assert result["metrics"]["valid_prediction_steps"] == 2
    assert result["metrics"]["divergence_index"] == 2
    np.testing.assert_array_equal(physical, predictions)
    assert np.isinf(pointwise[2:]).all()


@pytest.mark.parametrize("scale,reason", [(1.0, "metric_unsafe_residual_sum"),
    (0.1, "metric_unsafe_normalised_residual_sum")])
def test_square_sum_overflow_detected_even_when_individual_squares_are_safe(scale, reason):
    # One nonzero state per row: each square is representable; the third row
    # exhausts the accumulated float64 budget. No accidental overflow is needed.
    predictions = np.zeros((5, 3))
    predictions[1:, 0] = 1e154 * scale
    with np.errstate(over="raise", invalid="raise"):
        result, physical, _ = evaluate_predictions(
            predictions, np.zeros_like(predictions), np.full(3, scale)
        )
    assert result["failure_step"] == 2
    assert result["failure_reason"] == reason
    assert result["aggregate_nrmse_value"] == NONFINITE_FAILURE_SCORE
    np.testing.assert_array_equal(physical, predictions)


def test_metric_failure_precedes_later_physical_failure_without_destroying_evidence():
    predictions = np.array([[0.0] * 3, [1e200] * 3, [4.0] * 3, [np.inf] * 3])
    result, physical, _ = evaluate_predictions(predictions, np.zeros_like(predictions), np.ones(3))
    assert result["failure_step"] == 1
    assert result["physical_failure_step"] == 3
    assert result["failure_reason"] == "metric_unsafe_residual_square"
    np.testing.assert_array_equal(physical[:3], predictions[:3])
    assert np.isnan(physical[3:]).all()
    view = metric_prediction_view(physical, result["failure_step"])
    assert np.isnan(view[1:]).all()
    assert np.isfinite(physical[:3]).all()


def test_threshold_divergence_before_numerical_failure_is_retained():
    predictions = np.array([[0.0] * 3, [6.0] * 3, [1e200] * 3])
    result, _, _ = evaluate_predictions(predictions, np.zeros_like(predictions), np.ones(3))
    assert result["failure_step"] == 2
    assert result["metrics"]["valid_prediction_steps"] == 1
    assert result["metrics"]["divergence_index"] == 1
    assert result["metrics"]["divergence_reason"] == "normalised_error_threshold_reached"


@pytest.mark.parametrize("count", [1, 2, 3, 7, 20])
def test_square_sum_representability_boundary_never_yields_unclassified_infinity(count):
    boundary = np.sqrt(np.finfo(float).max / (count * 3))
    for value in (np.nextafter(boundary, 0.0), boundary, np.nextafter(boundary, np.inf)):
        predictions = np.full((count, 3), value)
        result, _, _ = evaluate_predictions(predictions, np.zeros_like(predictions), np.ones(3))
        assert result["numerical_failure"] or np.isfinite(result["metrics"]["nrmse_state"])
