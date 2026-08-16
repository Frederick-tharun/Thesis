"""Leakage-safe evaluation metrics for Chapter 2 ESN rollouts.

Metrics compare predictions s_hat_(t+1) with aligned targets s_(t+1).
Normalization scales must be supplied by the caller so later experiments can
use statistics locked from fitting data rather than fitting metric scales on an
evaluation segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from .esn_config import OUTPUT_DIMENSION
except ImportError:  # Support direct imports from the chapter2 directory.
    from esn_config import OUTPUT_DIMENSION


FloatArray = NDArray[np.float64]
STATE_NAMES = ("x", "y", "z")
COLLAPSE_STD_RATIO_THRESHOLD = 0.05


class MetricValidationError(ValueError):
    """Raised when rollout metric inputs violate the evaluation contract."""


OptionalStateValues = tuple[float | None, float | None, float | None]
StateFlags = tuple[bool, bool, bool]
StateValues = tuple[float, float, float]


@dataclass(frozen=True)
class RolloutMetrics:
    """State-error, reporting-diagnostic, VPT, and divergence results."""

    sample_count: int
    dt: float
    rmse_per_state: StateValues
    rmse_state: float
    nrmse_per_state: StateValues
    nrmse_state: float
    r2_per_state: OptionalStateValues
    r2_defined_per_state: StateFlags
    r2_macro: float | None
    r2_macro_defined: bool
    r2_macro_state_count: int
    correlation_per_state: OptionalStateValues
    correlation_defined_per_state: StateFlags
    correlation_macro: float | None
    correlation_macro_defined: bool
    correlation_macro_state_count: int
    prediction_std_per_state: OptionalStateValues
    target_std_per_state: StateValues
    std_ratio_per_state: OptionalStateValues
    std_ratio_defined_per_state: StateFlags
    collapse_std_ratio_threshold: float
    prediction_collapsed_per_state: StateFlags
    prediction_collapse_defined_per_state: StateFlags
    prediction_collapse_any: bool
    prediction_collapse_state_count: int
    prediction_collapse_defined_state_count: int
    valid_prediction_threshold: float
    valid_prediction_steps: int
    valid_prediction_time: float
    divergence_threshold: float
    diverged: bool
    divergence_index: int | None
    divergence_time: float | None
    divergence_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe flat representation."""

        def finite_or_none(value: float | None) -> float | None:
            if value is None:
                return None
            return float(value) if np.isfinite(value) else None

        result: dict[str, Any] = {
            "sample_count": self.sample_count,
            "dt": self.dt,
            "rmse_state": finite_or_none(self.rmse_state),
            "nrmse_state": finite_or_none(self.nrmse_state),
            "r2_macro": finite_or_none(self.r2_macro),
            "r2_macro_defined": self.r2_macro_defined,
            "r2_macro_state_count": self.r2_macro_state_count,
            "correlation_macro": finite_or_none(self.correlation_macro),
            "correlation_macro_defined": self.correlation_macro_defined,
            "correlation_macro_state_count": self.correlation_macro_state_count,
            "collapse_std_ratio_threshold": self.collapse_std_ratio_threshold,
            "prediction_collapse_any": self.prediction_collapse_any,
            "prediction_collapse_state_count": self.prediction_collapse_state_count,
            "prediction_collapse_defined_state_count": (
                self.prediction_collapse_defined_state_count
            ),
            "valid_prediction_threshold": self.valid_prediction_threshold,
            "valid_prediction_steps": self.valid_prediction_steps,
            "valid_prediction_time": self.valid_prediction_time,
            "divergence_threshold": self.divergence_threshold,
            "diverged": self.diverged,
            "divergence_index": self.divergence_index,
            "divergence_time": self.divergence_time,
            "divergence_reason": self.divergence_reason,
        }
        for index, state_name in enumerate(STATE_NAMES):
            result[f"rmse_{state_name}"] = finite_or_none(
                self.rmse_per_state[index]
            )
            result[f"nrmse_{state_name}"] = finite_or_none(
                self.nrmse_per_state[index]
            )
            result[f"r2_{state_name}"] = finite_or_none(
                self.r2_per_state[index]
            )
            result[f"r2_{state_name}_defined"] = self.r2_defined_per_state[index]
            result[f"correlation_{state_name}"] = finite_or_none(
                self.correlation_per_state[index]
            )
            result[f"correlation_{state_name}_defined"] = (
                self.correlation_defined_per_state[index]
            )
            result[f"prediction_std_{state_name}"] = finite_or_none(
                self.prediction_std_per_state[index]
            )
            result[f"target_std_{state_name}"] = self.target_std_per_state[index]
            result[f"prediction_target_std_ratio_{state_name}"] = finite_or_none(
                self.std_ratio_per_state[index]
            )
            result[f"prediction_target_std_ratio_{state_name}_defined"] = (
                self.std_ratio_defined_per_state[index]
            )
            result[f"prediction_collapsed_{state_name}"] = (
                self.prediction_collapsed_per_state[index]
            )
            result[f"prediction_collapse_{state_name}_defined"] = (
                self.prediction_collapse_defined_per_state[index]
            )
        return result


def _real_scalar(name: str, value: float, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if not np.isfinite(resolved):
        raise MetricValidationError(f"{name} must be finite")
    if positive and resolved <= 0.0:
        raise MetricValidationError(f"{name} must be positive")
    return resolved


def _rollout_arrays(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[FloatArray, FloatArray]:
    predicted = np.asarray(predictions)
    expected = np.asarray(targets)
    for name, array in (("predictions", predicted), ("targets", expected)):
        if array.ndim != 2 or array.shape[1:] != (OUTPUT_DIMENSION,):
            raise MetricValidationError(
                f"{name} must have shape (n, {OUTPUT_DIMENSION})"
            )
        if not np.issubdtype(array.dtype, np.number):
            raise MetricValidationError(f"{name} must have a numeric dtype")
    if predicted.shape != expected.shape:
        raise MetricValidationError("predictions and targets must have equal shape")
    if len(predicted) == 0:
        raise MetricValidationError("rollout arrays must not be empty")
    if not np.all(np.isfinite(expected)):
        raise MetricValidationError("targets must contain only finite values")
    return np.asarray(predicted, dtype=float), np.asarray(expected, dtype=float)


def _normalisation_scale(values: np.ndarray) -> FloatArray:
    scale = np.asarray(values)
    if scale.ndim != 1 or scale.shape != (OUTPUT_DIMENSION,):
        raise MetricValidationError(
            f"normalisation_scale must have shape ({OUTPUT_DIMENSION},)"
        )
    if not np.issubdtype(scale.dtype, np.number):
        raise MetricValidationError("normalisation_scale must have a numeric dtype")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise MetricValidationError(
            "normalisation_scale must contain finite positive values"
        )
    return np.asarray(scale, dtype=float)


def _macro(values: list[float]) -> tuple[float | None, bool, int]:
    if not values:
        return None, False, 0
    # Divide before summing so a representable mean is not lost to overflow.
    count = len(values)
    return float(sum(value / count for value in values)), True, count


def _reporting_diagnostics(
    predicted: FloatArray,
    expected: FloatArray,
    collapse_threshold: float,
) -> dict[str, Any]:
    r2_values: list[float | None] = []
    r2_defined: list[bool] = []
    correlation_values: list[float | None] = []
    correlation_defined: list[bool] = []
    prediction_stds: list[float | None] = []
    target_stds: list[float] = []
    ratios: list[float | None] = []
    ratio_defined: list[bool] = []
    collapsed: list[bool] = []

    for state_index in range(OUTPUT_DIMENSION):
        prediction = predicted[:, state_index]
        target = expected[:, state_index]
        prediction_finite = bool(np.all(np.isfinite(prediction)))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            target_centered = target - np.mean(target)
            target_sum_squares = float(target_centered @ target_centered)
            target_std = float(np.std(target, ddof=0))
        target_stds.append(target_std)

        prediction_std = None
        if prediction_finite:
            with np.errstate(over="ignore", invalid="ignore"):
                candidate_std = float(np.std(prediction, ddof=0))
            if np.isfinite(candidate_std):
                prediction_std = candidate_std
        prediction_stds.append(prediction_std)

        r2_value = None
        if (
            prediction_finite
            and np.isfinite(target_sum_squares)
            and target_sum_squares > 0.0
        ):
            residual = prediction - target
            with np.errstate(over="ignore", invalid="ignore"):
                candidate_r2 = float(
                    1.0 - (residual @ residual) / target_sum_squares
                )
            if np.isfinite(candidate_r2):
                r2_value = candidate_r2
        r2_values.append(r2_value)
        r2_defined.append(r2_value is not None)

        correlation_value = None
        if (
            prediction_finite
            and prediction_std is not None
            and prediction_std > 0.0
            and target_std > 0.0
        ):
            with np.errstate(over="ignore", invalid="ignore"):
                prediction_centered = prediction - np.mean(prediction)
                denominator = float(
                    np.sqrt(
                        (prediction_centered @ prediction_centered)
                        * target_sum_squares
                    )
                )
                candidate_correlation = float(
                    (prediction_centered @ target_centered) / denominator
                )
            if np.isfinite(candidate_correlation):
                correlation_value = candidate_correlation
        correlation_values.append(correlation_value)
        correlation_defined.append(correlation_value is not None)

        ratio = None
        if prediction_std is not None and target_std > 0.0:
            with np.errstate(over="ignore", invalid="ignore"):
                candidate_ratio = float(prediction_std / target_std)
            if np.isfinite(candidate_ratio):
                ratio = candidate_ratio
        ratios.append(ratio)
        ratio_defined.append(ratio is not None)
        collapsed.append(
            ratio is not None and ratio < collapse_threshold
        )

    r2_macro, r2_macro_defined, r2_count = _macro(
        [value for value in r2_values if value is not None]
    )
    correlation_macro, correlation_macro_defined, correlation_count = _macro(
        [value for value in correlation_values if value is not None]
    )
    collapse_count = sum(collapsed)
    collapse_defined_count = sum(ratio_defined)

    return {
        "r2_per_state": tuple(r2_values),
        "r2_defined_per_state": tuple(r2_defined),
        "r2_macro": r2_macro,
        "r2_macro_defined": r2_macro_defined,
        "r2_macro_state_count": r2_count,
        "correlation_per_state": tuple(correlation_values),
        "correlation_defined_per_state": tuple(correlation_defined),
        "correlation_macro": correlation_macro,
        "correlation_macro_defined": correlation_macro_defined,
        "correlation_macro_state_count": correlation_count,
        "prediction_std_per_state": tuple(prediction_stds),
        "target_std_per_state": tuple(target_stds),
        "std_ratio_per_state": tuple(ratios),
        "std_ratio_defined_per_state": tuple(ratio_defined),
        "prediction_collapsed_per_state": tuple(collapsed),
        "prediction_collapse_defined_per_state": tuple(ratio_defined),
        "prediction_collapse_any": collapse_count > 0,
        "prediction_collapse_state_count": collapse_count,
        "prediction_collapse_defined_state_count": collapse_defined_count,
    }


def pointwise_normalised_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    normalisation_scale: np.ndarray,
) -> FloatArray:
    """Return e_t = sqrt(mean_j(((prediction-target)/scale_j)^2))."""
    predicted, expected = _rollout_arrays(predictions, targets)
    scale = _normalisation_scale(normalisation_scale)
    difference = predicted - expected
    safe_difference = np.where(np.isfinite(difference), difference, np.inf)
    with np.errstate(over="ignore", invalid="ignore"):
        error = np.sqrt(np.mean(np.square(safe_difference / scale), axis=1))
    return np.asarray(error, dtype=float)


def evaluate_rollout(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    normalisation_scale: np.ndarray,
    dt: float,
    valid_prediction_threshold: float,
    divergence_threshold: float,
    collapse_std_ratio_threshold: float = COLLAPSE_STD_RATIO_THRESHOLD,
) -> RolloutMetrics:
    """Evaluate one aligned autonomous rollout.

    The valid prediction horizon is the consecutive prefix whose pointwise
    normalized error is strictly below valid_prediction_threshold. Divergence
    is the first non-finite prediction row or the first row whose pointwise
    normalized error reaches divergence_threshold.
    """
    predicted, expected = _rollout_arrays(predictions, targets)
    scale = _normalisation_scale(normalisation_scale)
    resolved_dt = _real_scalar("dt", dt)
    valid_threshold = _real_scalar(
        "valid_prediction_threshold", valid_prediction_threshold
    )
    resolved_divergence_threshold = _real_scalar(
        "divergence_threshold", divergence_threshold
    )
    collapse_threshold = _real_scalar(
        "collapse_std_ratio_threshold", collapse_std_ratio_threshold
    )
    if resolved_divergence_threshold <= valid_threshold:
        raise MetricValidationError(
            "divergence_threshold must be greater than "
            "valid_prediction_threshold"
        )

    difference = predicted - expected
    safe_difference = np.where(np.isfinite(difference), difference, np.inf)
    with np.errstate(over="ignore", invalid="ignore"):
        squared = np.square(safe_difference)
        rmse_per_state_array = np.sqrt(np.mean(squared, axis=0))
        rmse_state = float(np.sqrt(np.mean(squared)))
        normalised_squared = np.square(safe_difference / scale)
        nrmse_per_state_array = np.sqrt(np.mean(normalised_squared, axis=0))
        nrmse_state = float(np.sqrt(np.mean(normalised_squared)))
        pointwise = np.sqrt(np.mean(normalised_squared, axis=1))

    diagnostics = _reporting_diagnostics(
        predicted, expected, collapse_threshold
    )
    invalid_for_vpt = np.flatnonzero(pointwise >= valid_threshold)
    valid_steps = (
        int(invalid_for_vpt[0]) if len(invalid_for_vpt) else len(predicted)
    )

    nonfinite_rows = np.flatnonzero(~np.all(np.isfinite(predicted), axis=1))
    threshold_rows = np.flatnonzero(pointwise >= resolved_divergence_threshold)
    first_nonfinite = int(nonfinite_rows[0]) if len(nonfinite_rows) else None
    first_threshold = int(threshold_rows[0]) if len(threshold_rows) else None
    candidates = [
        index for index in (first_nonfinite, first_threshold) if index is not None
    ]
    divergence_index = min(candidates) if candidates else None
    if divergence_index is None:
        divergence_reason = None
        divergence_time = None
    elif first_nonfinite == divergence_index:
        divergence_reason = "non_finite_prediction"
        divergence_time = (divergence_index + 1) * resolved_dt
    else:
        divergence_reason = "normalised_error_threshold_reached"
        divergence_time = (divergence_index + 1) * resolved_dt

    return RolloutMetrics(
        sample_count=len(predicted),
        dt=resolved_dt,
        rmse_per_state=tuple(float(value) for value in rmse_per_state_array),
        rmse_state=rmse_state,
        nrmse_per_state=tuple(
            float(value) for value in nrmse_per_state_array
        ),
        nrmse_state=nrmse_state,
        **diagnostics,
        collapse_std_ratio_threshold=collapse_threshold,
        valid_prediction_threshold=valid_threshold,
        valid_prediction_steps=valid_steps,
        valid_prediction_time=valid_steps * resolved_dt,
        divergence_threshold=resolved_divergence_threshold,
        diverged=divergence_index is not None,
        divergence_index=divergence_index,
        divergence_time=divergence_time,
        divergence_reason=divergence_reason,
    )
