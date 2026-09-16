"""Lightweight safety tests for the Part 2 Step 1 regime map.

These tests are deliberately cheap (short transients, few currents) and
never write into ``results/``. They check simulation mechanics and, most
importantly, that no protected Chapter 1 / Chapter 2 Part 1 file has been
modified by this package.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from chapter2.hr_data_ch2 import simulate_fixed_current
from chapter2.dynamics_analysis_ch2 import detect_spikes

from chapter2.part2_regime_map import config
from chapter2.part2_regime_map.analyse_regimes import (
    QUIESCENT,
    TRANSITION_UNCERTAIN,
    final_classification,
    select_lyapunov_currents,
)
from chapter2.part2_regime_map.generate_regime_map import simulate_and_measure

STARTING_COMMIT = "6d40954fe6dd9df377b098c14f0c662226c3917a"

PROTECTED_PATHS = (
    "chapter2/hr_data_ch2.py",
    "chapter2/config_ch2.py",
    "chapter2/esn_model.py",
    "chapter2/esn_data.py",
    "chapter2/esn_metrics.py",
    "chapter2/esn_optimisation.py",
    "chapter2/run_bayesian_optimisation.py",
    "chapter2/esn_config.py",
    "chapter2/esn_step8.py",
    "chapter2/run_step8.py",
    "chapter2/audit_step8.py",
    "chapter2/correct_step8_event_validity.py",
    "chapter2/dynamics_analysis_ch2.py",
    "chapter2/generate_diagnostics.py",
    "chapter2/run_esn_pilot.py",
    "chapter2/plot_thesis_figures.py",
    "chapter2/verify_release.py",
    "chapter2/pilot_results",
    "chapter2/release",
    "chapter2/optimisation_results",
    "chapter2/final_models",
    "chapter2/final_results",
    "chapter2/outputs",
    "chapter2/provenance",
    "chapter2/README.md",
    "chapter2/EXPERIMENT_PROTOCOL.md",
    "chapter2/PROVENANCE.md",
    "chapter2/REPRODUCIBILITY.md",
    "chapter2/ARTIFACT_DISTRIBUTION.md",
    "chapter2/tests/test_esn_data.py",
    "chapter2/tests/test_esn_metrics.py",
    "chapter2/tests/test_esn_model.py",
    "chapter2/tests/test_esn_optimisation.py",
    "chapter2/tests/test_esn_step8.py",
    "chapter2/tests/test_hr_data_ch2.py",
    "chapter2/tests/test_release_verifier.py",
    "chapter2/tests/test_step8_audit.py",
    "chapter2/tests/test_step8_resume_integrity.py",
    "chapter2/tests/test_thesis_figures.py",
    "FINAL_THESIS_RUN",
    "main.py",
    "model.py",
    "config.py",
    "control_experiment.py",
    "neuron_controllers.py",
    "optimize_model.py",
    "data_loader.py",
)


def test_protected_paths_unchanged() -> None:
    result = subprocess.run(
        ["git", "diff", "--quiet", STARTING_COMMIT, "--", *PROTECTED_PATHS],
        cwd=config.PROJECT_ROOT,
        check=False,
    )
    assert result.returncode == 0, "a protected Part-1/Chapter-1 path changed since the starting commit"


def test_simulation_is_deterministic() -> None:
    spikes_a, peaks_a, xmin_a, xmax_a = simulate_and_measure(
        3.20, transient_steps=500, retained_samples=1000
    )
    spikes_b, peaks_b, xmin_b, xmax_b = simulate_and_measure(
        3.20, transient_steps=500, retained_samples=1000
    )
    np.testing.assert_array_equal(peaks_a, peaks_b)
    assert xmin_a == xmin_b
    assert xmax_a == xmax_b
    assert spikes_a.mean_isi == spikes_b.mean_isi or (
        np.isnan(spikes_a.mean_isi) and np.isnan(spikes_b.mean_isi)
    )


def test_sweep_grid_is_sorted_bounded_and_contains_known_currents() -> None:
    grid = config.build_sweep_grid()
    assert np.all(np.diff(grid) > 0), "sweep grid must be strictly increasing"
    assert grid[0] >= config.SWEEP_MIN_CURRENT - 1e-9
    assert grid[-1] <= config.SWEEP_MAX_CURRENT + 1e-9
    for known in config.KNOWN_CURRENTS:
        assert np.any(np.abs(grid - known) < 1e-9), f"known current {known} missing from grid"


def test_transient_is_actually_excluded() -> None:
    no_transient = simulate_fixed_current(3.20, retained_samples=5, transient_steps=0)
    with_transient = simulate_fixed_current(3.20, retained_samples=5, transient_steps=5000)
    assert not np.allclose(no_transient.x, with_transient.x)


def test_local_maxima_are_within_bounds_and_are_true_peaks() -> None:
    trajectory = simulate_fixed_current(3.20, retained_samples=5000, transient_steps=2000)
    peaks = detect_spikes(trajectory.x)
    assert peaks.size == 0 or (peaks.min() >= 0 and peaks.max() < len(trajectory.x))
    for index in peaks:
        if 0 < index < len(trajectory.x) - 1:
            assert trajectory.x[index] >= trajectory.x[index - 1]
            assert trajectory.x[index] >= trajectory.x[index + 1]


def test_result_shapes_are_consistent() -> None:
    spikes, peaks, xmin, xmax = simulate_and_measure(
        1.67, transient_steps=1000, retained_samples=3000
    )
    assert peaks.ndim == 1
    assert len(spikes.spike_indices) == len(peaks)
    assert xmin <= xmax


def test_final_classification_labels_quiescent_current_as_quiescent() -> None:
    record = {
        "spike_count": 0,
        "burst_structure": "tonic",
        "isi_cv": None,
        "mean_spikes_per_burst": None,
        "std_spikes_per_burst": None,
        "within_burst_isi_cv": None,
        "interburst_interval_cv": None,
    }
    assert final_classification(record, None) == QUIESCENT


def test_final_classification_defaults_to_uncertain_without_lyapunov() -> None:
    record = {
        "spike_count": 10,
        "burst_structure": "uncertain",
        "isi_cv": 0.9,
        "mean_spikes_per_burst": None,
        "std_spikes_per_burst": None,
        "within_burst_isi_cv": None,
        "interburst_interval_cv": None,
    }
    assert final_classification(record, None) == TRANSITION_UNCERTAIN


def test_select_lyapunov_currents_always_includes_known_currents() -> None:
    records = [
        {"current": c, "burst_structure": "tonic", "isi_cv": 0.05, "spike_count": 5,
         "mean_spikes_per_burst": None, "std_spikes_per_burst": None,
         "within_burst_isi_cv": None, "interburst_interval_cv": None}
        for c in (1.0, 1.67, 2.0, 3.20, 3.0, 3.29, 3.34, 3.5, 4.0)
    ]
    labels = {r["current"]: final_classification(r, None) for r in records}
    selected = select_lyapunov_currents(
        records, known_currents=config.KNOWN_CURRENTS, max_extra=2, provisional_labels=labels
    )
    for known in config.KNOWN_CURRENTS:
        assert any(abs(known - s) < 1e-9 for s in selected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
