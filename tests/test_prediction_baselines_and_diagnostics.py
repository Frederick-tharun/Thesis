import numpy as np

from prediction_baselines import autonomous_baselines, fit_linear_ar1
from prediction_diagnostics import chaotic_valid_prediction_horizon


def test_linear_ar1_learns_training_only_linear_map():
    values = np.empty((80, 2), dtype=float)
    values[0] = [1.0, -0.5]
    transition_true = np.array([[0.8, 0.1], [-0.2, 0.7]])
    intercept_true = np.array([0.05, -0.02])
    for index in range(1, len(values)):
        values[index] = values[index - 1] @ transition_true + intercept_true

    transition, intercept = fit_linear_ar1(values, regularization=0.0)

    assert np.allclose(transition, transition_true, atol=1e-9)
    assert np.allclose(intercept, intercept_true, atol=1e-9)


def test_baseline_rollouts_have_requested_shape():
    training = np.arange(60, dtype=float).reshape(20, 3)
    baselines = autonomous_baselines(training, 12)

    assert set(baselines) == {"persistence", "linear_ar1"}
    assert baselines["persistence"]["prediction"].shape == (12, 3)
    assert baselines["linear_ar1"]["prediction"].shape == (12, 3)
    assert not baselines["linear_ar1"]["uses_heldout_observations"]


def test_valid_prediction_horizon_reports_lyapunov_units():
    reference = np.zeros((30, 3))
    predicted = reference.copy()
    predicted[10:] = 1.0
    summary, curve = chaotic_valid_prediction_horizon(
        predicted,
        reference,
        dt=0.1,
        largest_lyapunov_exponent=0.5,
        error_threshold=0.4,
        smoothing_steps=1,
        persistence_steps=3,
    )

    assert summary["horizon_steps"] == 10
    assert summary["horizon_time"] == 1.0
    assert summary["horizon_lyapunov_units"] == 0.5
    assert not summary["right_censored_at_rollout_end"]
    assert len(curve) == 30
