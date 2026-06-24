import sys
import types
import unittest

import numpy as np


plotting_stub = types.ModuleType("plotting")
for function_name in (
    "plot_controlled_vs_uncontrolled_x",
    "plot_controlled_all_states",
    "plot_control_signal",
    "plot_control_error",
    "plot_k_sweep_summary",
):
    setattr(plotting_stub, function_name, lambda *args, **kwargs: None)
sys.modules.setdefault("plotting", plotting_stub)

from control_experiment import _pyragas_dynamics_metrics, _settling_time


class PyragasMetricTests(unittest.TestCase):
    def test_settling_time_requires_a_complete_window_and_checks_last_window(self):
        times = np.arange(6, dtype=float)
        error = np.array([2.0, 2.0, 2.0, 0.1, 0.1, 0.1])
        self.assertEqual(_settling_time(times, error, 0, 0.2, 3), 3.0)
        self.assertTrue(np.isnan(_settling_time(times[:2], error[:2], 0, 0.2, 3)))

    def test_large_delay_keeps_a_valid_fixed_delay_comparison(self):
        n = 600
        times = np.arange(n, dtype=float)
        period = 40
        x = 3.0 * np.sin(2.0 * np.pi * times / period)
        states = np.column_stack(
            (
                x,
                0.5 * np.cos(2.0 * np.pi * times / period),
                0.2 * np.sin(2.0 * np.pi * times / period),
            )
        )
        metrics = _pyragas_dynamics_metrics(
            controlled=states,
            uncontrolled=states,
            control_signal=np.zeros_like(states),
            times=times,
            control_start_idx=100,
            pyragas_delay=320,
            spike_threshold=1.0,
        )

        self.assertTrue(
            np.isfinite(metrics["pyragas_periodicity_rmse_state_norm"])
        )
        self.assertGreaterEqual(metrics["pyragas_detected_cycle_count"], 3)


if __name__ == "__main__":
    unittest.main()
