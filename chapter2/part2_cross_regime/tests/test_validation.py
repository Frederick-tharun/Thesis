"""Validation-logic tests: LOCO folds, gate function, generalized objective."""
from __future__ import annotations

from chapter2.part2_cross_regime import validation


def test_loco_folds_leave_exactly_one_out():
    currents = [1.0, 2.0, 3.0, 4.0]
    folds = validation.build_loco_folds(currents)
    assert len(folds) == 4
    for fold in folds:
        assert fold["validate_on"] not in fold["fit_on"]
        assert len(fold["fit_on"]) == 3
        assert set(fold["fit_on"] + [fold["validate_on"]]) == set(currents)


def _fake_rollout(current, window, vpt_steps, horizon, diverged=False, collapse=False, nonfinite=False):
    return {
        "current": current,
        "window": window,
        "objective_nrmse": 0.01,
        "nonfinite_failure": nonfinite,
        "metrics": {
            "valid_prediction_steps": vpt_steps,
            "diverged": diverged,
            "prediction_collapse_any": collapse,
        },
    }


def test_source_quality_gate_passes_clean_rollouts():
    rollouts = [_fake_rollout(1.0, w, 8000, 8000) for w in (1, 2, 3)]
    assert validation.passes_source_quality_gate(rollouts, horizon=8000) is True


def test_source_quality_gate_fails_on_low_vpt():
    rollouts = [_fake_rollout(1.0, w, 1000, 8000) for w in (1, 2, 3)]
    assert validation.passes_source_quality_gate(rollouts, horizon=8000) is False


def test_source_quality_gate_fails_on_divergence():
    rollouts = [_fake_rollout(1.0, 1, 8000, 8000, diverged=True)]
    assert validation.passes_source_quality_gate(rollouts, horizon=8000) is False


def test_source_objective_requires_exact_pair_set():
    rollouts = [_fake_rollout(1.0, 1, 8000, 8000), _fake_rollout(2.0, 1, 8000, 8000)]
    expected = {(1.0, 1), (2.0, 1)}
    assert validation.source_objective(rollouts, expected) == 0.01
    try:
        validation.source_objective(rollouts, {(1.0, 1)})
        assert False, "expected ValueError for mismatched pair set"
    except ValueError:
        pass
