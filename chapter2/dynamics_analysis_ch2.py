"""Reproducible waveform, consistency, and Lyapunov diagnostics for Chapter 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import find_peaks

from scripts.analysis.estimate_hr_lyapunov import estimate_largest_lyapunov

try:
    from .config_ch2 import (
        BURST_MIN_GAP_PROMINENCE,
        BURST_MIN_INTERVALS_PER_TIMESCALE,
        BURST_MIN_LOG_ISI_GAP,
        BURST_MIN_SPIKES,
        DT,
        HALF_WINDOW_CONSISTENCY_TOLERANCE,
        HR_PARAMETERS,
        INITIAL_STATE,
        INITIAL_TRANSIENT_STEPS,
        LYAPUNOV_CHECKPOINT_STEPS,
        LYAPUNOV_CONVERGENCE_ABSOLUTE_TOLERANCE,
        LYAPUNOV_CONVERGENCE_RELATIVE_TOLERANCE,
        LYAPUNOV_ESTIMATION_STEPS,
        LYAPUNOV_POSITIVE_THRESHOLD,
        LYAPUNOV_RENORMALIZATION_STEPS,
        REGULAR_INTERBURST_INTERVAL_CV_MAX,
        REGULAR_ISI_CV_MAX,
        REGULAR_SPIKES_PER_BURST_CV_MAX,
        REGULAR_WITHIN_BURST_ISI_CV_MAX,
        SPIKE_HEIGHT,
        SPIKE_MIN_DISTANCE_STEPS,
        SPIKE_PROMINENCE,
    )
    from .hr_data_ch2 import HRTrajectory
except ImportError:  # Support direct imports from the chapter2 directory.
    from config_ch2 import (
        BURST_MIN_GAP_PROMINENCE,
        BURST_MIN_INTERVALS_PER_TIMESCALE,
        BURST_MIN_LOG_ISI_GAP,
        BURST_MIN_SPIKES,
        DT,
        HALF_WINDOW_CONSISTENCY_TOLERANCE,
        HR_PARAMETERS,
        INITIAL_STATE,
        INITIAL_TRANSIENT_STEPS,
        LYAPUNOV_CHECKPOINT_STEPS,
        LYAPUNOV_CONVERGENCE_ABSOLUTE_TOLERANCE,
        LYAPUNOV_CONVERGENCE_RELATIVE_TOLERANCE,
        LYAPUNOV_ESTIMATION_STEPS,
        LYAPUNOV_POSITIVE_THRESHOLD,
        LYAPUNOV_RENORMALIZATION_STEPS,
        REGULAR_INTERBURST_INTERVAL_CV_MAX,
        REGULAR_ISI_CV_MAX,
        REGULAR_SPIKES_PER_BURST_CV_MAX,
        REGULAR_WITHIN_BURST_ISI_CV_MAX,
        SPIKE_HEIGHT,
        SPIKE_MIN_DISTANCE_STEPS,
        SPIKE_PROMINENCE,
    )
    from hr_data_ch2 import HRTrajectory


@dataclass(frozen=True)
class SpikeBurstAnalysis:
    spike_indices: np.ndarray
    interspike_intervals: np.ndarray
    mean_isi: float
    std_isi: float
    isi_cv: float
    burst_structure: str
    burst_count: int | None
    mean_spikes_per_burst: float
    std_spikes_per_burst: float
    mean_within_burst_isi: float
    within_burst_isi_cv: float
    mean_interburst_interval: float
    interburst_interval_cv: float
    burst_gap_threshold: float | None
    notes: str


@dataclass(frozen=True)
class HalfWindowConsistency:
    result: str
    mean_shifts: np.ndarray
    std_shifts: np.ndarray
    early_mean_isi: float
    late_mean_isi: float
    isi_relative_shift: float
    notes: str


@dataclass(frozen=True)
class LyapunovAnalysis:
    exponent: float
    classification: str
    convergence: str
    checkpoint_steps: np.ndarray
    checkpoint_lle: np.ndarray
    convergence_tolerance: float


def detect_spikes(
    x: Sequence[float] | np.ndarray,
    *,
    height: float = SPIKE_HEIGHT,
    prominence: float = SPIKE_PROMINENCE,
    minimum_distance_steps: int = SPIKE_MIN_DISTANCE_STEPS,
) -> np.ndarray:
    """Detect upward HR spikes with fixed, documented peak criteria."""
    values = np.asarray(x, dtype=float)
    peaks, _ = find_peaks(
        values,
        height=height,
        prominence=prominence,
        distance=minimum_distance_steps,
    )
    return peaks.astype(int, copy=False)


def _coefficient_of_variation(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    mean = float(np.mean(values))
    return float(np.std(values, ddof=0) / mean) if mean > 0.0 else float("nan")


def analyze_spikes_and_bursts(
    trajectory: HRTrajectory,
    *,
    dt: float = DT,
) -> SpikeBurstAnalysis:
    """Measure spikes and separate within- and between-burst ISI scales.

    The candidate threshold is the geometric midpoint across the largest gap
    in sorted log-ISI values.  The gap must be large in absolute terms and
    prominent relative to the distribution's other log gaps.  This avoids
    manufacturing bursts from a regular tonic train.
    """
    peaks = detect_spikes(trajectory.x)
    isi = np.diff(peaks) * dt
    nan = float("nan")
    if len(isi) == 0:
        return SpikeBurstAnalysis(
            peaks, isi, nan, nan, nan, "uncertain", None,
            nan, nan, nan, nan, nan, nan, None,
            "fewer than two detected spikes",
        )

    mean_isi = float(np.mean(isi))
    std_isi = float(np.std(isi, ddof=0))
    isi_cv = _coefficient_of_variation(isi)
    threshold: float | None = None
    split_note = "no clear two-timescale ISI separation"

    if len(isi) >= 2 * BURST_MIN_INTERVALS_PER_TIMESCALE:
        ordered = np.sort(isi)
        log_gaps = np.diff(np.log(ordered))
        split_index = int(np.argmax(log_gaps))
        largest_gap = float(log_gaps[split_index])
        background_gaps = np.delete(log_gaps, split_index)
        positive_gaps = background_gaps[background_gaps > np.finfo(float).eps]
        typical_gap = (
            float(np.median(positive_gaps))
            if len(positive_gaps)
            else np.finfo(float).eps
        )
        prominence = largest_gap / typical_gap
        short_count = split_index + 1
        long_count = len(isi) - short_count
        if (
            largest_gap >= BURST_MIN_LOG_ISI_GAP
            and prominence >= BURST_MIN_GAP_PROMINENCE
            and short_count >= BURST_MIN_INTERVALS_PER_TIMESCALE
            and long_count >= BURST_MIN_INTERVALS_PER_TIMESCALE
        ):
            threshold = float(
                np.sqrt(ordered[split_index] * ordered[split_index + 1])
            )
            split_note = (
                f"adaptive log-ISI split gap={largest_gap:.4g}, "
                f"prominence={prominence:.4g}, threshold={threshold:.6g}"
            )

    if threshold is None:
        structure = "tonic" if isi_cv <= REGULAR_ISI_CV_MAX else "uncertain"
        return SpikeBurstAnalysis(
            peaks, isi, mean_isi, std_isi, isi_cv, structure,
            0 if structure == "tonic" else None,
            nan, nan, nan, nan, nan, nan, None, split_note,
        )

    boundaries = np.flatnonzero(isi > threshold) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(peaks)]))
    valid_groups = [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start >= BURST_MIN_SPIKES
    ]
    if len(valid_groups) < 2:
        return SpikeBurstAnalysis(
            peaks, isi, mean_isi, std_isi, isi_cv, "uncertain", None,
            nan, nan, nan, nan, nan, nan, threshold,
            f"{split_note}; fewer than two valid bursts",
        )

    spike_counts = np.asarray(
        [end - start for start, end in valid_groups], dtype=float
    )
    within_isi = np.concatenate(
        [isi[start : end - 1] for start, end in valid_groups]
    )
    burst_start_times = np.asarray(
        [trajectory.t[peaks[start]] for start, _ in valid_groups], dtype=float
    )
    interburst_intervals = np.diff(burst_start_times)
    ignored_groups = len(starts) - len(valid_groups)
    notes = split_note
    if ignored_groups:
        notes += f"; ignored {ignored_groups} singleton spike group(s)"

    return SpikeBurstAnalysis(
        peaks,
        isi,
        mean_isi,
        std_isi,
        isi_cv,
        "bursting",
        len(valid_groups),
        float(np.mean(spike_counts)),
        float(np.std(spike_counts, ddof=0)),
        float(np.mean(within_isi)),
        _coefficient_of_variation(within_isi),
        float(np.mean(interburst_intervals)),
        _coefficient_of_variation(interburst_intervals),
        threshold,
        notes,
    )


def evaluate_half_window_consistency(
    trajectory: HRTrajectory,
    *,
    tolerance: float = HALF_WINDOW_CONSISTENCY_TOLERANCE,
) -> HalfWindowConsistency:
    """Check whether measurements agree between the two retained halves."""
    midpoint = len(trajectory.t) // 2
    early = trajectory.state[:midpoint]
    late = trajectory.state[midpoint:]
    full_std = np.std(trajectory.state, axis=0, ddof=0)
    scale = np.maximum(full_std, np.finfo(float).eps)
    mean_shifts = np.abs(np.mean(early, axis=0) - np.mean(late, axis=0)) / scale
    std_shifts = np.abs(
        np.std(early, axis=0, ddof=0) - np.std(late, axis=0, ddof=0)
    ) / scale

    early_peaks = detect_spikes(trajectory.x[:midpoint])
    late_peaks = detect_spikes(trajectory.x[midpoint:])
    early_isi = np.diff(early_peaks) * DT
    late_isi = np.diff(late_peaks) * DT
    if len(early_isi) >= 2 and len(late_isi) >= 2:
        early_mean_isi = float(np.mean(early_isi))
        late_mean_isi = float(np.mean(late_isi))
        isi_relative_shift = abs(early_mean_isi - late_mean_isi) / max(
            0.5 * (early_mean_isi + late_mean_isi),
            np.finfo(float).eps,
        )
        isi_available = True
    else:
        early_mean_isi = float("nan")
        late_mean_isi = float("nan")
        isi_relative_shift = float("nan")
        isi_available = False

    state_consistent = bool(
        np.all(mean_shifts <= tolerance) and np.all(std_shifts <= tolerance)
    )
    isi_consistent = bool(isi_available and isi_relative_shift <= tolerance)
    if not isi_available:
        result = "uncertain"
    elif state_consistent and isi_consistent:
        result = "consistent"
    else:
        result = "inconsistent"

    notes = (
        f"max normalized state-mean shift={np.max(mean_shifts):.4g}; "
        f"max relative state-std shift={np.max(std_shifts):.4g}; "
        + (
            f"mean-ISI relative shift={isi_relative_shift:.4g}"
            if isi_available
            else "insufficient spikes for half-to-half mean-ISI comparison"
        )
    )
    return HalfWindowConsistency(
        result,
        mean_shifts,
        std_shifts,
        early_mean_isi,
        late_mean_isi,
        isi_relative_shift,
        notes,
    )


def estimate_lyapunov(current: float) -> LyapunovAnalysis:
    """Run the validated Benettin estimator and retain five checkpoints."""
    parameters = {
        "a": HR_PARAMETERS.a,
        "b": HR_PARAMETERS.b,
        "c": HR_PARAMETERS.c,
        "d": HR_PARAMETERS.d,
        "r": HR_PARAMETERS.r,
        "s": HR_PARAMETERS.s,
        "xr": HR_PARAMETERS.x_r,
        "I": float(current),
    }
    result = estimate_largest_lyapunov(
        parameters,
        INITIAL_STATE,
        dt=DT,
        transient_steps=INITIAL_TRANSIENT_STEPS,
        estimation_steps=LYAPUNOV_ESTIMATION_STEPS,
        renormalization_steps=LYAPUNOV_RENORMALIZATION_STEPS,
    )

    checkpoint_steps = np.asarray(LYAPUNOV_CHECKPOINT_STEPS, dtype=int)
    checkpoint_indices = (
        checkpoint_steps // LYAPUNOV_RENORMALIZATION_STEPS - 1
    )
    checkpoint_lle = result.convergence_exponents[checkpoint_indices]
    tolerance = max(
        LYAPUNOV_CONVERGENCE_ABSOLUTE_TOLERANCE,
        LYAPUNOV_CONVERGENCE_RELATIVE_TOLERANCE * abs(result.exponent),
    )
    final_changes = np.abs(np.diff(checkpoint_lle[-3:]))
    convergence = (
        "converged" if np.all(final_changes <= tolerance) else "not_converged"
    )

    if convergence == "not_converged":
        classification = (
            "weak positive"
            if result.exponent > LYAPUNOV_POSITIVE_THRESHOLD
            else "uncertain"
        )
    elif result.exponent > LYAPUNOV_POSITIVE_THRESHOLD:
        classification = "positive"
    elif abs(result.exponent) <= LYAPUNOV_POSITIVE_THRESHOLD:
        classification = "near zero"
    else:
        classification = "negative"

    return LyapunovAnalysis(
        float(result.exponent),
        classification,
        convergence,
        checkpoint_steps,
        checkpoint_lle,
        tolerance,
    )


def _burst_regularity(spikes: SpikeBurstAnalysis) -> tuple[bool, bool]:
    """Return whether detected bursts are regular or measurably irregular."""
    if spikes.burst_structure != "bursting":
        return False, False
    spikes_per_burst_cv = (
        spikes.std_spikes_per_burst / spikes.mean_spikes_per_burst
    )
    measures = (
        (spikes_per_burst_cv, REGULAR_SPIKES_PER_BURST_CV_MAX),
        (spikes.within_burst_isi_cv, REGULAR_WITHIN_BURST_ISI_CV_MAX),
        (spikes.interburst_interval_cv, REGULAR_INTERBURST_INTERVAL_CV_MAX),
    )
    if not all(np.isfinite(value) for value, _ in measures):
        return False, False
    regular = all(value <= limit for value, limit in measures)
    irregular = any(value > limit for value, limit in measures)
    return regular, irregular


def preliminary_regime(
    spikes: SpikeBurstAnalysis,
    lyapunov: LyapunovAnalysis,
) -> str:
    """Combine spike, burst, and Lyapunov evidence cautiously."""
    has_bursts = spikes.burst_structure == "bursting"
    tonic_regular = (
        spikes.burst_structure == "tonic"
        and np.isfinite(spikes.isi_cv)
        and spikes.isi_cv <= REGULAR_ISI_CV_MAX
    )
    burst_regular, burst_irregular = _burst_regularity(spikes)
    reliable_positive = (
        lyapunov.convergence == "converged"
        and lyapunov.classification == "positive"
    )
    irregular_dynamics = burst_irregular or (
        spikes.burst_structure == "uncertain"
        and np.isfinite(spikes.isi_cv)
        and spikes.isi_cv > REGULAR_ISI_CV_MAX
    )

    if reliable_positive:
        return "chaotic bursting" if has_bursts else "transition/weakly chaotic"
    if burst_regular and lyapunov.classification != "weak positive":
        return "periodic bursting"
    if tonic_regular and lyapunov.classification != "weak positive":
        return "periodic spiking"
    if irregular_dynamics and lyapunov.convergence == "not_converged":
        return "transition/weakly chaotic"
    return "uncertain"
