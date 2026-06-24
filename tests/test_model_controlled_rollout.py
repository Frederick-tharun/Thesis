import sys
import types
import unittest

import numpy as np


try:
    import scipy  # noqa: F401
except ModuleNotFoundError:
    scipy_stub = types.ModuleType("scipy")
    scipy_stub.sparse = types.ModuleType("scipy.sparse")
    scipy_stub.linalg = types.ModuleType("scipy.linalg")
    sys.modules["scipy"] = scipy_stub
    sys.modules["scipy.sparse"] = scipy_stub.sparse
    sys.modules["scipy.linalg"] = scipy_stub.linalg

from model import EchoStateNetwork


def deterministic_esn(readout):
    esn = object.__new__(EchoStateNetwork)
    esn.is_fitted = True
    esn.input_size = 1
    esn.N_res = 1
    esn.normalize_input = False
    esn._normalize_apply = lambda values: np.asarray(values, dtype=float)
    esn._warmup_until_index = lambda values, n_warmup: (
        np.zeros(1),
        np.asarray(values[n_warmup], dtype=float).copy(),
    )
    esn._update_state = lambda state, current_input: state
    esn._readout = readout
    return esn


class ControlledRolloutTests(unittest.TestCase):
    def test_linear_feedback_preserves_original_update_rule(self):
        esn = deterministic_esn(lambda current_input, state: current_input.copy())
        result = esn.predict_controlled(
            train_sequence=np.array([[1.0]]),
            horizon_steps=3,
            target=np.array([0.0]),
            K=0.5,
            controller="linear_feedback",
        )
        np.testing.assert_allclose(
            result["controlled_output_norm"].reshape(-1), [0.5, 0.25, 0.125]
        )
        self.assertTrue(result["stable"])

    def test_finite_time_controller_is_dispatched(self):
        esn = deterministic_esn(lambda current_input, state: current_input.copy())
        result = esn.predict_controlled(
            train_sequence=np.array([[4.0]]),
            horizon_steps=1,
            target=np.array([0.0]),
            K=0.5,
            controller="finite_time",
            finite_s=0.5,
        )
        self.assertAlmostEqual(result["controlled_output_norm"][0, 0], 3.0)

    def test_pyragas_uses_controlled_history(self):
        esn = deterministic_esn(
            lambda current_input, state: current_input.copy() + 1.0
        )
        result = esn.predict_controlled(
            train_sequence=np.array([[0.0]]),
            horizon_steps=3,
            target=np.array([0.0]),
            K=0.5,
            controller="pyragas",
            pyragas_delay=1,
            pyragas_sign=-1,
        )
        np.testing.assert_allclose(
            result["controlled_output_norm"].reshape(-1), [1.0, 1.5, 2.0]
        )
        np.testing.assert_allclose(
            result["control_signal_norm"].reshape(-1), [0.0, 0.5, 0.5]
        )

    def test_invalid_controller_fails_and_zero_horizon_is_empty(self):
        esn = deterministic_esn(lambda current_input, state: current_input.copy())
        common = {
            "train_sequence": np.array([[1.0]]),
            "target": np.array([0.0]),
            "K": 0.5,
        }
        with self.assertRaises(ValueError):
            esn.predict_controlled(horizon_steps=1, controller="bad", **common)
        empty = esn.predict_controlled(
            horizon_steps=0, controller="linear_feedback", **common
        )
        self.assertTrue(empty["stable"])
        self.assertEqual(empty["controlled_output_norm"].shape, (0, 1))


if __name__ == "__main__":
    unittest.main()
