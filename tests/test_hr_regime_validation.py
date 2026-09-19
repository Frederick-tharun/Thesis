import numpy as np
import pytest

from hr_regime_validation import (
    analyze_inter_spike_intervals,
    validate_hr_trajectory,
)


def test_periodic_single_isi_cycle_is_detected():
    diagnostics = analyze_inter_spike_intervals(
        np.array([20.12, 20.13] * 20),
        expected_cycle_length=1,
    )

    assert diagnostics["isi_cv"] < 0.001
    assert diagnostics["isi_max_min_ratio"] < 1.01
    assert diagnostics["cycle_normalized_max_error"] < 0.001


def test_repeating_burst_cycle_is_not_tonic_spiking():
    cycle = np.array([10.36, 11.24, 12.36, 13.89, 16.14, 20.13, 32.46, 114.85])
    diagnostics = analyze_inter_spike_intervals(
        np.tile(cycle, 5),
        expected_cycle_length=8,
    )

    assert diagnostics["isi_cv"] > 1.0
    assert diagnostics["isi_max_min_ratio"] > 10.0
    assert diagnostics["cycle_normalized_max_error"] == pytest.approx(0.0)


def test_chaotic_label_requires_positive_lyapunov_evidence():
    t = np.linspace(0.0, 100.0, 10_001)
    trajectory = np.column_stack(
        [np.sin(t), np.cos(t), 0.5 * np.sin(t / 3.0)]
    )

    result = validate_hr_trajectory(
        trajectory,
        dt=0.01,
        expected_regime="chaotic_bursting",
        expected_cycle_length=None,
        transient_steps=200_000,
        largest_lyapunov_exponent=None,
    )

    assert not result["passed"]
    assert not result["conditions"][
        "resolved_positive_largest_lyapunov_exponent"
    ]


def test_invalid_cycle_length_is_rejected():
    with pytest.raises(ValueError, match="positive or None"):
        analyze_inter_spike_intervals(
            np.array([1.0, 1.1, 1.2]),
            expected_cycle_length=0,
        )
