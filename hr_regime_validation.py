"""Scientific validation gates for configured Hindmarsh--Rose trajectories.

The firing-regime name is a hypothesis, not evidence.  This module checks the
retained trajectory before it can enter model selection.  A positive largest
Lyapunov exponent is required before the chaotic label is accepted; irregular
inter-spike intervals alone are not treated as proof of chaos.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.signal import find_peaks


SUPPORTED_REGIMES = {
    "periodic_spiking",
    "periodic_bursting",
    "chaotic_bursting",
}


def _finite_float(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("diagnostic value is not finite")
    return value


def extract_inter_spike_intervals(
    membrane_potential: np.ndarray,
    dt: float,
    *,
    prominence: float = 0.5,
    minimum_peak_distance: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return spike indices and inter-spike intervals in model-time units."""
    signal = np.asarray(membrane_potential, dtype=float).reshape(-1)
    if signal.size < 3:
        raise ValueError("membrane_potential must contain at least 3 samples")
    if not np.all(np.isfinite(signal)):
        raise ValueError("membrane_potential contains non-finite values")
    if dt <= 0:
        raise ValueError("dt must be positive")

    peaks, _ = find_peaks(
        signal,
        prominence=float(prominence),
        distance=int(minimum_peak_distance),
    )
    return peaks, np.diff(peaks).astype(float) * float(dt)


def analyze_inter_spike_intervals(
    intervals: np.ndarray,
    *,
    expected_cycle_length: int | None,
    maximum_lag: int = 12,
) -> dict[str, float | int | None]:
    """Calculate periodicity and variability diagnostics for an ISI sequence."""
    isi = np.asarray(intervals, dtype=float).reshape(-1)
    if isi.size == 0:
        return {
            "interval_count": 0,
            "isi_mean": None,
            "isi_std": None,
            "isi_cv": None,
            "isi_min": None,
            "isi_max": None,
            "isi_max_min_ratio": None,
            "expected_cycle_length": expected_cycle_length,
            "cycle_max_abs_error": None,
            "cycle_normalized_max_error": None,
            "best_short_lag": None,
            "best_short_lag_normalized_mae": None,
        }
    if not np.all(np.isfinite(isi)) or np.any(isi <= 0):
        raise ValueError("inter-spike intervals must be finite and positive")

    mean = _finite_float(np.mean(isi))
    std = _finite_float(np.std(isi))
    result: dict[str, float | int | None] = {
        "interval_count": int(isi.size),
        "isi_mean": mean,
        "isi_std": std,
        "isi_cv": _finite_float(std / mean),
        "isi_min": _finite_float(np.min(isi)),
        "isi_max": _finite_float(np.max(isi)),
        "isi_max_min_ratio": _finite_float(np.max(isi) / np.min(isi)),
        "expected_cycle_length": expected_cycle_length,
        "cycle_max_abs_error": None,
        "cycle_normalized_max_error": None,
        "best_short_lag": None,
        "best_short_lag_normalized_mae": None,
    }

    if expected_cycle_length is not None:
        lag = int(expected_cycle_length)
        if lag <= 0:
            raise ValueError("expected_cycle_length must be positive or None")
        if isi.size > lag:
            error = np.abs(isi[lag:] - isi[:-lag])
            result["cycle_max_abs_error"] = _finite_float(np.max(error))
            result["cycle_normalized_max_error"] = _finite_float(
                np.max(error) / mean
            )

    lag_errors: list[tuple[float, int]] = []
    for lag in range(1, min(int(maximum_lag), isi.size - 1) + 1):
        normalized_mae = float(
            np.mean(np.abs(isi[lag:] - isi[:-lag])) / mean
        )
        lag_errors.append((normalized_mae, lag))
    if lag_errors:
        best_error, best_lag = min(lag_errors)
        result["best_short_lag"] = int(best_lag)
        result["best_short_lag_normalized_mae"] = _finite_float(best_error)

    return result


def _stationarity_diagnostics(trajectory: np.ndarray) -> dict[str, Any]:
    """Compare the first and final retained quarters without phase alignment."""
    values = np.asarray(trajectory, dtype=float)
    quarter = values.shape[0] // 4
    first = values[:quarter]
    last = values[-quarter:]
    scale = np.std(values, axis=0) + np.finfo(float).eps
    mean_shift = np.abs(np.mean(first, axis=0) - np.mean(last, axis=0)) / scale
    std_shift = np.abs(np.std(first, axis=0) - np.std(last, axis=0)) / scale
    return {
        "quarter_length_steps": int(quarter),
        "normalized_mean_shift_by_state": mean_shift.tolist(),
        "normalized_std_shift_by_state": std_shift.tolist(),
        "maximum_normalized_mean_shift": _finite_float(np.max(mean_shift)),
        "maximum_normalized_std_shift": _finite_float(np.max(std_shift)),
    }


def validate_hr_trajectory(
    trajectory: np.ndarray,
    *,
    dt: float,
    expected_regime: str,
    expected_cycle_length: int | None,
    transient_steps: int,
    largest_lyapunov_exponent: float | None = None,
    lyapunov_tail_std: float | None = None,
    lyapunov_separation_multiplier: float = 3.0,
    minimum_chaotic_lyapunov_exponent: float = 1e-3,
) -> dict[str, Any]:
    """Validate a retained HR trajectory against an explicit regime hypothesis."""
    values = np.asarray(trajectory, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("trajectory must have shape (n_steps, 3)")
    if values.shape[0] < 100:
        raise ValueError("trajectory is too short for regime validation")
    if not np.all(np.isfinite(values)):
        raise ValueError("trajectory contains non-finite values")
    if expected_regime not in SUPPORTED_REGIMES:
        raise ValueError(f"unsupported HR regime: {expected_regime}")

    peaks, intervals = extract_inter_spike_intervals(values[:, 0], dt)
    isi = analyze_inter_spike_intervals(
        intervals,
        expected_cycle_length=expected_cycle_length,
    )
    stationarity = _stationarity_diagnostics(values)

    conditions: dict[str, bool] = {
        "at_least_20_detected_spikes": bool(peaks.size >= 20),
        "retained_state_statistics_stable": bool(
            # Chaotic finite-time windows fluctuate more than phase-locked
            # periodic records; 0.35 SD is a conservative distributional gate.
            stationarity["maximum_normalized_mean_shift"] <= 0.35
            and stationarity["maximum_normalized_std_shift"] <= 0.25
        ),
    }
    interpretation: str

    if expected_regime == "periodic_spiking":
        lle_near_zero = bool(
            largest_lyapunov_exponent is not None
            and lyapunov_tail_std is not None
            and abs(largest_lyapunov_exponent)
            <= lyapunov_separation_multiplier * lyapunov_tail_std
        )
        conditions.update(
            {
                "regular_inter_spike_intervals": bool(
                    isi["isi_cv"] is not None and isi["isi_cv"] <= 0.02
                ),
                "single_isi_cycle_repeats": bool(
                    expected_cycle_length == 1
                    and isi["cycle_normalized_max_error"] is not None
                    and isi["cycle_normalized_max_error"] <= 0.005
                ),
                "no_burst_timescale_separation": bool(
                    isi["isi_max_min_ratio"] is not None
                    and isi["isi_max_min_ratio"] <= 1.10
                ),
                "largest_lyapunov_exponent_consistent_with_zero": lle_near_zero,
            }
        )
        interpretation = "regular tonic periodic spiking"
    elif expected_regime == "periodic_bursting":
        lle_near_zero = bool(
            largest_lyapunov_exponent is not None
            and lyapunov_tail_std is not None
            and abs(largest_lyapunov_exponent)
            <= lyapunov_separation_multiplier * lyapunov_tail_std
        )
        conditions.update(
            {
                "burst_timescale_separation": bool(
                    isi["isi_max_min_ratio"] is not None
                    and isi["isi_max_min_ratio"] >= 3.0
                ),
                "multi_isi_cycle_repeats": bool(
                    expected_cycle_length is not None
                    and expected_cycle_length >= 2
                    and isi["cycle_normalized_max_error"] is not None
                    and isi["cycle_normalized_max_error"] <= 0.005
                ),
                "largest_lyapunov_exponent_consistent_with_zero": lle_near_zero,
            }
        )
        interpretation = "periodic bursting with a repeating ISI cycle"
    else:
        lle_is_resolved_positive = bool(
            largest_lyapunov_exponent is not None
            and lyapunov_tail_std is not None
            and np.isfinite(largest_lyapunov_exponent)
            and np.isfinite(lyapunov_tail_std)
            and largest_lyapunov_exponent
            > max(
                minimum_chaotic_lyapunov_exponent,
                lyapunov_separation_multiplier * lyapunov_tail_std,
            )
        )
        conditions.update(
            {
                "burst_timescale_separation": bool(
                    isi["isi_max_min_ratio"] is not None
                    and isi["isi_max_min_ratio"] >= 2.0
                ),
                "irregular_inter_spike_intervals": bool(
                    isi["isi_cv"] is not None and isi["isi_cv"] >= 0.15
                ),
                "no_short_repeating_isi_cycle": bool(
                    isi["best_short_lag_normalized_mae"] is not None
                    and isi["best_short_lag_normalized_mae"] >= 0.08
                ),
                "resolved_positive_largest_lyapunov_exponent": (
                    lle_is_resolved_positive
                ),
            }
        )
        interpretation = (
            "chaotic bursting supported by irregular event timing and a "
            "positive largest Lyapunov exponent"
        )

    return {
        "expected_regime": expected_regime,
        "interpretation": interpretation,
        "passed": bool(all(conditions.values())),
        "conditions": conditions,
        "dt": float(dt),
        "retained_steps": int(values.shape[0]),
        "retained_time": float(values.shape[0] * dt),
        "transient_steps": int(transient_steps),
        "transient_time": float(transient_steps * dt),
        "detected_spike_count": int(peaks.size),
        "largest_lyapunov_exponent": (
            None
            if largest_lyapunov_exponent is None
            else _finite_float(largest_lyapunov_exponent)
        ),
        "lyapunov_tail_std": (
            None if lyapunov_tail_std is None else _finite_float(lyapunov_tail_std)
        ),
        "lyapunov_separation_multiplier": float(
            lyapunov_separation_multiplier
        ),
        "minimum_chaotic_lyapunov_exponent": float(
            minimum_chaotic_lyapunov_exponent
        ),
        "isi_diagnostics": isi,
        "stationarity_diagnostics": stationarity,
    }


def regime_metadata(parameter_set: Mapping[str, Any]) -> dict[str, Any]:
    """Extract non-dynamical validation metadata from a parameter set."""
    return {
        "expected_regime": str(parameter_set["expected_regime"]),
        "expected_cycle_length": parameter_set.get("expected_isi_cycle_length"),
        "transient_steps": int(parameter_set["transient_steps"]),
    }
