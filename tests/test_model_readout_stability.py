from unittest import mock

import numpy as np

import model as model_module
from model import EchoStateNetwork


def test_readout_solver_records_successful_diagnostics():
    rng = np.random.default_rng(7)
    training = rng.normal(size=(100, 3))
    model = EchoStateNetwork(N_res=12, input_size=3, seed=7)

    model.train(training, washout=5)

    assert model.readout_diagnostics["solution_finite"]
    assert np.isfinite(model.readout_diagnostics["relative_training_residual"])
    assert model.readout_diagnostics["regularization"] >= 1e-12


def test_failed_normal_equation_solve_uses_augmented_ridge_fallback():
    if model_module.scipy_linalg is None:
        return
    rng = np.random.default_rng(8)
    training = rng.normal(size=(80, 2))
    model = EchoStateNetwork(N_res=10, input_size=2, seed=8)

    with mock.patch.object(
        model_module.scipy_linalg,
        "solve",
        side_effect=np.linalg.LinAlgError("synthetic ill conditioning"),
    ):
        model.train(training, washout=5)

    assert model.is_fitted
    assert model.readout_diagnostics["fallback_used"]
    assert model.readout_diagnostics["solver"] == (
        "augmented_ridge_least_squares_fallback"
    )
