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

    def test_finite_time_uses_linear_branch_outside_unit_sphere(self):
        actual = finite_time_control(
            y_pred=[4.0, -4.0], target=[0.0, 0.0], K=0.5, finite_s=0.5
        )
        np.testing.assert_allclose(actual, [2.0, -2.0])

        # The branch is selected by the vector norm, even though each
        # individual component is smaller than one.
        vector_norm_outside = finite_time_control(
            y_pred=[0.8, 0.8], target=[0.0, 0.0], K=0.5, finite_s=0.5
        )
        np.testing.assert_allclose(vector_norm_outside, [0.4, 0.4])

    def test_finite_time_uses_fractional_branch_inside_and_on_unit_sphere(self):
        inside = finite_time_control(
            y_pred=[0.25, -0.25], target=[0.0, 0.0], K=0.5, finite_s=0.5
        )
        np.testing.assert_allclose(inside, [0.25, -0.25])

        boundary = finite_time_control(
            y_pred=[0.6, 0.8], target=[0.0, 0.0], K=0.5, finite_s=0.5
        )
        np.testing.assert_allclose(
            boundary,
            0.5 * np.sqrt([0.6, 0.8]),
        )

        zero = finite_time_control(
            y_pred=[0.0, 0.0], target=[0.0, 0.0], K=0.5, finite_s=0.5
        )
        np.testing.assert_array_equal(zero, [0.0, 0.0])

    def test_finite_time_exponent_validation(self):

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
