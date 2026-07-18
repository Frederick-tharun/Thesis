import sys
import types
import unittest
from unittest import mock

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
    def test_linear_feedback_returns_distinct_canonical_signals_and_aliases(self):
        esn = deterministic_esn(lambda current_input, state: current_input.copy())
        result = esn.predict_controlled(
            train_sequence=np.array([[1.0]]),
            horizon_steps=3,
            target=np.array([0.0]),
            K=0.5,
            controller="linear_feedback",
        )

        raw = result["raw_readout_norm"]
        corrected = result["corrected_feedback_input_norm"]
        control = result["control_signal_norm"]

        np.testing.assert_allclose(raw.reshape(-1), [1.0, 0.5, 0.25])
        np.testing.assert_allclose(corrected.reshape(-1), [0.5, 0.25, 0.125])
        np.testing.assert_allclose(control.reshape(-1), [0.5, 0.25, 0.125])
        np.testing.assert_allclose(corrected, raw - control)

        self.assertIs(result["raw_prediction_norm"], raw)
        self.assertIs(result["controlled_output_norm"], corrected)
        self.assertIs(result["feedback_input_norm"], corrected)
        self.assertIs(result["error_signal_norm"], result["raw_readout_error_norm"])
        self.assertTrue(result["stable"])
        self.assertFalse(result["divergence_detected"])
        self.assertEqual(result["steps_completed"], 3)

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
        self.assertAlmostEqual(
            result["corrected_feedback_input_norm"][0, 0], 2.0
        )
        self.assertAlmostEqual(result["raw_readout_norm"][0, 0], 4.0)

    def test_pyragas_defaults_to_raw_readout_history(self):
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
            result["raw_readout_norm"].reshape(-1), [1.0, 2.0, 2.5]
        )
        np.testing.assert_allclose(
            result["corrected_feedback_input_norm"].reshape(-1),
            [1.0, 1.5, 2.25],
        )
        np.testing.assert_allclose(
            result["control_signal_norm"].reshape(-1), [0.0, 0.5, 0.25]
        )
        self.assertEqual(result["pyragas_history_signal"], "raw_readout")

    def test_pyragas_can_use_legacy_corrected_feedback_history(self):
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
            pyragas_history_signal="corrected_feedback_input",
        )

        np.testing.assert_allclose(
            result["corrected_feedback_input_norm"].reshape(-1),
            [1.0, 1.5, 2.0],
        )
        np.testing.assert_allclose(
            result["control_signal_norm"].reshape(-1), [0.0, 0.5, 0.5]
        )

    def test_feedback_clip_reports_requested_and_applied_control(self):
        esn = deterministic_esn(lambda current_input, state: current_input.copy())
        result = esn.predict_controlled(
            train_sequence=np.array([[4.0]]),
            horizon_steps=1,
            target=np.array([0.0]),
            K=0.5,
            controller="linear_feedback",
            control_input_clip=1.0,
        )

        np.testing.assert_allclose(result["raw_readout_norm"], [[4.0]])
        np.testing.assert_allclose(result["requested_control_signal_norm"], [[2.0]])
        np.testing.assert_allclose(result["corrected_feedback_input_norm"], [[1.0]])
        np.testing.assert_allclose(result["control_signal_norm"], [[3.0]])
        np.testing.assert_allclose(
            result["corrected_feedback_input_norm"],
            result["raw_readout_norm"] - result["control_signal_norm"],
        )
        self.assertEqual(result["control_input_clip"], (-1.0, 1.0))

    def test_divergence_is_marked_with_reason_and_index(self):
        esn = deterministic_esn(lambda current_input, state: current_input.copy())
        result = esn.predict_controlled(
            train_sequence=np.array([[3.0]]),
            horizon_steps=4,
            target=np.array([0.0]),
            K=0.5,
            controller="linear_feedback",
            divergence_abs_limit=2.0,
        )

        self.assertFalse(result["stable"])
        self.assertTrue(result["divergence_detected"])
        self.assertEqual(result["divergence_reason"], "raw_readout_abs_limit_exceeded")
        self.assertEqual(result["divergence_index"], 0)
        self.assertEqual(result["steps_completed"], 1)
        self.assertTrue(np.isnan(result["corrected_feedback_input_norm"][0, 0]))

    def test_requested_control_limit_has_a_distinct_reason(self):
        esn = deterministic_esn(lambda current_input, state: current_input.copy())
        result = esn.predict_controlled(
            train_sequence=np.array([[1.0]]),
            horizon_steps=1,
            target=np.array([0.0]),
            K=3.0,
            controller="linear_feedback",
            divergence_abs_limit=2.0,
        )

        self.assertFalse(result["stable"])
        self.assertEqual(
            result["divergence_reason"],
            "requested_control_signal_abs_limit_exceeded",
        )
        np.testing.assert_allclose(result["requested_control_signal_norm"], [[3.0]])
        np.testing.assert_allclose(
            result["corrected_feedback_input_norm"],
            result["raw_readout_norm"] - result["control_signal_norm"],
        )

    def test_applied_control_limit_has_a_distinct_reason(self):
        esn = deterministic_esn(lambda current_input, state: current_input.copy())
        result = esn.predict_controlled(
            train_sequence=np.array([[1.0]]),
            horizon_steps=1,
            target=np.array([0.0]),
            K=0.0,
            controller="linear_feedback",
            control_input_clip=(-1.0, -1.0),
            divergence_abs_limit=1.5,
        )

        self.assertFalse(result["stable"])
        self.assertEqual(
            result["divergence_reason"],
            "applied_control_signal_abs_limit_exceeded",
        )
        np.testing.assert_allclose(result["requested_control_signal_norm"], [[0.0]])
        np.testing.assert_allclose(result["control_signal_norm"], [[2.0]])
        np.testing.assert_allclose(
            result["corrected_feedback_input_norm"],
            result["raw_readout_norm"] - result["control_signal_norm"],
        )

    def test_nonfinite_requested_and_applied_controls_have_distinct_reasons(self):
        esn = deterministic_esn(lambda current_input, state: current_input.copy())
        with np.errstate(over="ignore", invalid="ignore"):
            requested_result = esn.predict_controlled(
                train_sequence=np.array([[1.0]]),
                horizon_steps=1,
                target=np.array([0.0]),
                K=np.inf,
                controller="linear_feedback",
                divergence_abs_limit=np.inf,
            )
        self.assertEqual(
            requested_result["divergence_reason"],
            "nonfinite_requested_control_signal",
        )

        max_float = np.finfo(float).max
        esn = deterministic_esn(lambda current_input, state: current_input.copy())
        with mock.patch(
            "model.compute_control_signal",
            return_value=np.array([-max_float]),
        ):
            with np.errstate(over="ignore", invalid="ignore"):
                applied_result = esn.predict_controlled(
                    train_sequence=np.array([[max_float]]),
                    horizon_steps=1,
                    target=np.array([0.0]),
                    K=0.0,
                    controller="linear_feedback",
                    divergence_abs_limit=np.inf,
                )
        self.assertEqual(
            applied_result["divergence_reason"],
            "nonfinite_applied_control_signal",
        )

    def test_invalid_options_fail_and_zero_horizon_has_canonical_keys(self):
        esn = deterministic_esn(lambda current_input, state: current_input.copy())
        common = {
            "train_sequence": np.array([[1.0]]),
            "target": np.array([0.0]),
            "K": 0.5,
        }
        with self.assertRaises(ValueError):
            esn.predict_controlled(horizon_steps=1, controller="bad", **common)
        with self.assertRaises(ValueError):
            esn.predict_controlled(
                horizon_steps=1,
                controller="pyragas",
                pyragas_history_signal="ambiguous",
                **common,
            )
        with self.assertRaises(ValueError):
            esn.predict_controlled(
                horizon_steps=1,
                controller="linear_feedback",
                control_input_clip=(2.0, -2.0),
                **common,
            )
        with self.assertRaises(ValueError):
            esn.predict_controlled(
                horizon_steps=1,
                controller="linear_feedback",
                divergence_abs_limit=0.0,
                **common,
            )

        empty = esn.predict_controlled(
            horizon_steps=0, controller="linear_feedback", **common
        )
        self.assertTrue(empty["stable"])
        self.assertFalse(empty["divergence_detected"])
        self.assertEqual(empty["steps_completed"], 0)
        self.assertEqual(empty["raw_readout_norm"].shape, (0, 1))
        self.assertEqual(empty["corrected_feedback_input_norm"].shape, (0, 1))
        self.assertIs(
            empty["controlled_output_norm"],
            empty["corrected_feedback_input_norm"],
        )


if __name__ == "__main__":
    unittest.main()
