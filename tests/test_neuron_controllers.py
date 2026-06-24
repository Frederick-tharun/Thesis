import unittest

import numpy as np

from neuron_controllers import (
    available_controllers,
    compute_control_signal,
    finite_time_control,
    linear_feedback_control,
    pyragas_control,
)


class ControllerLawTests(unittest.TestCase):
    def test_all_three_controllers_are_registered(self):
        self.assertEqual(
            available_controllers(),
            ["linear_feedback", "finite_time", "pyragas"],
        )

    def test_linear_feedback_law(self):
        actual = linear_feedback_control(
            y_pred=[2.0, -1.0], target=[1.0, 1.0], K=0.5
        )
        np.testing.assert_allclose(actual, [0.5, -1.0])

    def test_finite_time_law_and_exponent_validation(self):
        actual = finite_time_control(
            y_pred=[4.0, -4.0], target=[0.0, 0.0], K=0.5, finite_s=0.5
        )
        np.testing.assert_allclose(actual, [1.0, -1.0], rtol=0, atol=1e-8)

        with self.assertRaises(ValueError):
            finite_time_control(
                y_pred=[1.0], target=[0.0], K=1.0, finite_s=1.0
            )

    def test_pyragas_waits_for_history_and_respects_sign(self):
        no_history = pyragas_control(
            y_pred=[3.0], target=[0.0], K=0.5, history=[], pyragas_delay=2
        )
        np.testing.assert_allclose(no_history, [0.0])

        history = [np.array([1.0]), np.array([2.0])]
        toward_delayed = pyragas_control(
            y_pred=[3.0],
            target=[0.0],
            K=0.5,
            history=history,
            pyragas_delay=2,
            pyragas_sign=-1,
        )
        away_from_delayed = pyragas_control(
            y_pred=[3.0],
            target=[0.0],
            K=0.5,
            history=history,
            pyragas_delay=2,
            pyragas_sign=1,
        )
        np.testing.assert_allclose(toward_delayed, [1.0])
        np.testing.assert_allclose(away_from_delayed, [-1.0])

    def test_dispatcher_rejects_unknown_controller(self):
        with self.assertRaises(ValueError):
            compute_control_signal(
                controller="unknown", y_pred=[1.0], target=[0.0], K=1.0
            )


if __name__ == "__main__":
    unittest.main()
