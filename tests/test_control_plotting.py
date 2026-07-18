import os
import tempfile
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np

import plotting


class ControlPlottingTests(unittest.TestCase):
    def test_metric_box_prefers_canonical_signal_and_effort_keys(self):
        text = plotting._metric_box_text(
            {
                "controller": "linear_feedback",
                "K": 0.5,
                "corrected_feedback_input_target_rmse_state": 0.125,
                "raw_readout_target_rmse_state": 0.25,
                "control_effort_mean_sq": 0.375,
                "control_energy": 99.0,
                "settling_time": 1.5,
            }
        )

        self.assertIn("Corrected-feedback target RMSE = 0.1250", text)
        self.assertIn("Raw-readout target RMSE = 0.2500", text)
        self.assertIn("Mean-squared control effort = 0.375", text)
        self.assertNotIn("99", text)
        self.assertIn(
            "Controller-test time to tolerance = 1.5000", text
        )

    def test_metric_box_keeps_legacy_effort_fallback(self):
        text = plotting._metric_box_text(
            {
                "controller": "linear_feedback",
                "target_rmse_state": 0.5,
                "control_energy": 0.75,
            }
        )
        self.assertIn("Corrected-feedback target RMSE = 0.5000", text)
        self.assertIn("Mean-squared control effort = 0.75", text)

    def test_raw_vs_corrected_plot_uses_required_filename_and_labels(self):
        times = np.arange(5, dtype=float)
        raw = np.column_stack([times, times, times])
        corrected = raw * 0.5

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(plotting, "_savefig_to_path") as save_mock:
                plotting.plot_raw_readout_vs_corrected_feedback_input_x(
                    times,
                    raw,
                    corrected,
                    control_start_idx=1,
                    output_dir=temp_dir,
                    controller_name="linear_feedback",
                )

            path = save_mock.call_args.args[0]
            fig = save_mock.call_args.kwargs["fig"]
            labels = fig.axes[0].get_legend_handles_labels()[1]
            self.assertEqual(
                os.path.basename(path),
                "raw_readout_vs_corrected_feedback_input_x.png",
            )
            self.assertIn("Raw ESN readout x (closed loop)", labels)
            self.assertIn("Corrected feedback input x", labels)
            plt.close(fig)


    def test_control_signal_marks_validation_and_heldout_test(self):
        times = np.arange(6, dtype=float)
        control = np.zeros((6, 3), dtype=float)

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(plotting, "_savefig_to_path") as save_mock:
                plotting.plot_control_signal(
                    times,
                    control,
                    control_start_idx=1,
                    output_dir=temp_dir,
                    controller_name="linear_feedback",
                    metrics={"controller_test_start": 3},
                )

            fig = save_mock.call_args.kwargs["fig"]
            axis = fig.axes[0]
            labels = axis.get_legend_handles_labels()[1]
            self.assertIn(
                "Controller validation (not used for held-out metrics)", labels
            )
            self.assertIn("Held-out controller-test start", labels)
            self.assertEqual(axis.get_xlabel(), "Time")
            plt.close(fig)

    def test_final_comparison_plot_uses_heldout_metric_and_canonical_timing(self):
        rows = [{
            "Regime": "chaotic_bursting",
            "Optimizer": "dummy",
            "Pred_NRMSE_x": 0.2,
            "Pred_NRMSE_all": 0.3,
            "Best_K": 0.1,
            "Final_test_metric_name": "pyragas_empirical_recurrence_error_norm",
            "Final_test_metric_value": 0.12,
            "Control_target_RMSE_state": 0.4,
            "Spike_reduction_percent": 5.0,
            "Control_effort_mean_sq": 0.02,
            "Controller_test_time_to_tolerance": 1.5,
            "Control_stable": True,
        }]

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(plotting, "_savefig_to_path") as save_mock:
                plotting.plot_final_comparison_table(
                    os.path.join(temp_dir, "comparison.png"), rows
                )

            fig = save_mock.call_args.kwargs["fig"]
            table = fig.axes[0].tables[0]
            headers = [
                cell.get_text().get_text()
                for (row, _column), cell in table.get_celld().items()
                if row == 0
            ]
            self.assertIn("Final_test_metric_name", headers)
            self.assertIn("Final_test_metric_value", headers)
            self.assertIn("Controller_test_time_to_tolerance", headers)
            self.assertIn("Control_effort_mean_sq", headers)
            self.assertNotIn("Settling_time", headers)
            self.assertNotIn("Control_energy", headers)
            plt.close(fig)

    def test_k_sweep_title_and_axes_are_explicitly_validation_only(self):
        rows = [
            {
                "controller": "linear_feedback",
                "stable": True,
                "K": 0.1,
                "corrected_feedback_input_target_rmse_state": 0.4,
                "spike_reduction_percent": 10.0,
                "control_effort_mean_sq": 0.02,
            },
            {
                "controller": "linear_feedback",
                "stable": True,
                "K": 0.2,
                "corrected_feedback_input_target_rmse_state": 0.2,
                "spike_reduction_percent": 20.0,
                "control_effort_mean_sq": 0.04,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(plotting, "_savefig_to_path") as save_mock:
                plotting.plot_k_sweep_summary(
                    rows,
                    temp_dir,
                    controller_name="linear_feedback",
                )

            fig = save_mock.call_args.kwargs["fig"]
            self.assertIn("controller-validation", fig._suptitle.get_text())
            self.assertIn("controller-test segment not used", fig._suptitle.get_text())
            self.assertTrue(
                all("Validation" in ax.get_ylabel() for ax in fig.axes)
            )
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
