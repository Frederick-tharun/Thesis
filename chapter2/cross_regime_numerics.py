"""Canonical post-forecast numerical policy for cross-regime derived results.

This module never generates predictions. Target-dependent checks belong here,
after autonomous feedback has finished. A finite residual is metric-unsafe when
float64 subtraction, normalisation, squaring, or the nonnegative square sum is
not representable. These are numerical limits, not scientific thresholds.

Historical prediction arrays are immutable evidence: their finite metric-unsafe
values are retained in the returned physical copy. Only physical nonfiniteness
canonicalises that copy to a NaN suffix. Metrics use a separate NaN suffix from
the earlier of physical failure and metric unsafety, preserving prefix horizons
and the established failure penalty instead of inventing a finite huge RMS.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chapter2.cross_regime_config import (
    COLLAPSE_THRESHOLD,
    DIVERGENCE_THRESHOLD,
    VALID_PREDICTION_THRESHOLD,
)
from chapter2.esn_metrics import (
    _normalisation_scale,
    _rollout_arrays,
    evaluate_rollout,
    pointwise_normalised_error,
)
from chapter2.esn_optimisation import NONFINITE_FAILURE_SCORE


NUMERICAL_POLICY = "chapter2_cross_regime_float64_fail_closed_v1"
_FLOAT_MAX = np.finfo(float).max
_SQUARE_LIMIT = np.sqrt(_FLOAT_MAX)


def first_nonfinite_prediction_step(predictions: np.ndarray) -> int | None:
    """Return the first physical row containing NaN or either infinity."""
    indices = np.flatnonzero(~np.all(np.isfinite(predictions), axis=1))
    return int(indices[0]) if len(indices) else None


def metric_prediction_view(
    predictions: np.ndarray, failure_step: int | None
) -> np.ndarray:
    """Return a copy with a NaN suffix; never mutate original evidence."""
    result = np.array(predictions, dtype=float, copy=True)
    if failure_step is not None:
        result[failure_step:] = np.nan
    return result


def _first_row(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.any(mask, axis=1))
    return int(indices[0]) if len(indices) else None


def _metric_failure(
    predictions: np.ndarray, targets: np.ndarray, scale: np.ndarray
) -> tuple[int | None, str | None]:
    """Check safe prefixes before each potentially overflowing operation.

    A square-sum budget includes all states and preceding rows, so it also
    bounds every per-state and pointwise reduction. Exact power-of-two scaling
    avoids overflowing the check itself. The standard n*eps/(1-n*eps) bound
    covers roundoff in both this accumulation and ordinary metric reductions.
    Only values already proved square-safe are squared.
    """
    stop = len(predictions)
    failure_step = None
    failure_reason = None

    def restrict(mask: np.ndarray, reason: str) -> None:
        nonlocal stop, failure_step, failure_reason
        candidate = _first_row(mask)
        if candidate is not None and candidate < stop:
            stop = candidate
            failure_step, failure_reason = candidate, reason

    opposite_sign = np.signbit(predictions) != np.signbit(targets)
    restrict(
        opposite_sign & (np.abs(predictions) > _FLOAT_MAX - np.abs(targets)),
        "metric_unsafe_residual_subtraction",
    )
    residual = predictions[:stop] - targets[:stop]

    # max * scale is safe only where scale < 1; division cannot overflow for
    # positive scale >= 1. Do not evaluate an overflowing branch of np.where.
    division_limit = np.full(3, _FLOAT_MAX)
    small_scale = scale < 1.0
    division_limit[small_scale] = _FLOAT_MAX * scale[small_scale]
    restrict(np.abs(residual) > division_limit, "metric_unsafe_normalisation")
    residual = residual[:stop]
    normalised = residual / scale

    restrict(np.abs(residual) > _SQUARE_LIMIT, "metric_unsafe_residual_square")
    restrict(
        np.abs(normalised[:stop]) > _SQUARE_LIMIT,
        "metric_unsafe_normalised_residual_square",
    )
    residual, normalised = residual[:stop], normalised[:stop]
    for values, reason in (
        (residual, "metric_unsafe_residual_sum"),
        (normalised, "metric_unsafe_normalised_residual_sum"),
    ):
        with np.errstate(under="ignore"):
            fractions = np.ldexp(np.square(values[:stop]), -1023)
        # Scaling by 2**-1023 is exact except for negligible underflow. Each
        # term is below 2, so the check itself cannot overflow. Account for
        # possible underestimation here AND overestimation in np.sum/mean;
        # equality at the representability boundary must not hide overflow.
        cumulative = np.cumsum(fractions.reshape(-1)).reshape(-1, 3)
        roundoff = (fractions.size + 2) * np.finfo(float).eps
        if roundoff >= 1.0:
            raise ValueError("array too large to certify float64 metric reductions")
        gamma = roundoff / (1.0 - roundoff)
        budget = np.ldexp(_FLOAT_MAX, -1023) * (1.0 - gamma) / (1.0 + gamma)
        restrict(cumulative > budget, reason)
    return failure_step, failure_reason


def evaluate_predictions(
    predictions: np.ndarray, targets: np.ndarray, scale: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Recompute canonical classification and frozen metrics from predictions.

    Evaluator, post-hoc correction, and auditor must all use this entry point.
    The returned tuple is metadata, physical prediction copy, pointwise error.
    Physical failure takes precedence if both definitions fail on the same row.
    The input arrays are not modified and no model API is accessed.
    """
    predicted, expected = _rollout_arrays(predictions, targets)
    normalisation_scale = _normalisation_scale(scale)
    physical_step = first_nonfinite_prediction_step(predicted)
    physical = metric_prediction_view(predicted, physical_step)
    physical_prefix = len(physical) if physical_step is None else physical_step
    metric_step, metric_reason = _metric_failure(
        physical[:physical_prefix], expected[:physical_prefix], normalisation_scale
    )
    failure_step = physical_step
    failure_reason = (
        "non_finite_physical_prediction" if physical_step is not None else None
    )
    if metric_step is not None:
        failure_step, failure_reason = metric_step, metric_reason
    metric_view = metric_prediction_view(physical, failure_step)
    metrics = evaluate_rollout(
        metric_view,
        expected,
        normalisation_scale=normalisation_scale,
        dt=0.01,
        valid_prediction_threshold=VALID_PREDICTION_THRESHOLD,
        divergence_threshold=DIVERGENCE_THRESHOLD,
        collapse_std_ratio_threshold=COLLAPSE_THRESHOLD,
    ).to_dict()
    pointwise = pointwise_normalised_error(
        metric_view, expected, normalisation_scale=normalisation_scale
    )
    numerical_failure = failure_step is not None
    # The safety policy must fail closed; a future metric implementation change
    # cannot silently turn an undefined score into an unclassified record.
    if not numerical_failure and (
        metrics["nrmse_state"] is None or metrics["rmse_state"] is None
    ):
        raise ValueError("unclassified nonfinite metric under " + NUMERICAL_POLICY)
    metadata = {
        "numerical_policy": NUMERICAL_POLICY,
        "physical_failure_step": physical_step,
        "failure_step": failure_step,
        "numerical_failure": numerical_failure,
        "failure_reason": failure_reason,
        "valid_prefix_steps": failure_step if numerical_failure else len(predicted),
        "metrics": metrics,
        "aggregate_nrmse_value": (
            NONFINITE_FAILURE_SCORE if numerical_failure else metrics["nrmse_state"]
        ),
    }
    return metadata, physical, pointwise
