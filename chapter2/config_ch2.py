"""Locked configuration for Chapter 2 Hindmarsh--Rose diagnostics.

This module is intentionally independent of the Chapter 1 ``config.py``.
No ESN settings are defined here because this stage is simulation-only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HRParameters:
    """Parameters of the three-state Hindmarsh--Rose system."""

    a: float = 1.0
    b: float = 3.0
    c: float = 1.0
    d: float = 5.0
    r: float = 0.006
    s: float = 4.0
    x_r: float = -1.6


HR_PARAMETERS = HRParameters()
INITIAL_STATE = (-1.0, -3.0, 3.0)
DT = 0.01
INITIAL_TRANSIENT_STEPS = 100_000
RETAINED_SAMPLES_PER_CURRENT = 100_000

FIXED_CURRENTS = (1.67, 3.20, 3.29, 3.34, 3.50)
CONTINUOUS_CURRENT_SEQUENCE = (1.67, 3.29, 3.50, 3.34, 3.20)

# Diagnostic and analysis settings.
FIXED_COMPARISON_DURATION = 300.0
SWITCH_PRE_DURATION = 50.0
SWITCH_POST_DURATION = 200.0

# Spike detection uses x peaks above 0, prominence at least 0.5, and a
# refractory separation of 20 integration steps (0.2 model-time units).
SPIKE_HEIGHT = 0.0
SPIKE_PROMINENCE = 0.5
SPIKE_MIN_DISTANCE_STEPS = 20

# A candidate two-timescale split must be both substantial in absolute terms
# and prominent relative to the other adjacent gaps in sorted log-ISI space.
BURST_MIN_LOG_ISI_GAP = 0.15
BURST_MIN_GAP_PROMINENCE = 4.0
BURST_MIN_INTERVALS_PER_TIMESCALE = 2
BURST_MIN_SPIKES = 2

# CV limits used to call tonic spiking or burst recurrence periodic.
REGULAR_ISI_CV_MAX = 0.15
REGULAR_WITHIN_BURST_ISI_CV_MAX = 0.15
REGULAR_INTERBURST_INTERVAL_CV_MAX = 0.15
REGULAR_SPIKES_PER_BURST_CV_MAX = 0.15

# The half-window check compares the two retained halves; 10% means
# "consistent", not proof that the discarded transient was sufficient.
HALF_WINDOW_CONSISTENCY_TOLERANCE = 0.10

# Validated Benettin tangent-linear estimator settings inherited from the
# Chapter 1 analysis methodology.
LYAPUNOV_ESTIMATION_STEPS = 500_000
LYAPUNOV_RENORMALIZATION_STEPS = 10
LYAPUNOV_CHECKPOINT_STEPS = (100_000, 200_000, 300_000, 400_000, 500_000)
LYAPUNOV_POSITIVE_THRESHOLD = 1.0e-3
# The last three checkpoints converge when each consecutive change is within
# the larger of this absolute tolerance and 20% of the final estimate.
LYAPUNOV_CONVERGENCE_ABSOLUTE_TOLERANCE = 5.0e-4
LYAPUNOV_CONVERGENCE_RELATIVE_TOLERANCE = 0.20
