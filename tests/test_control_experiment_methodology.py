import csv
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np

import control_experiment as ce


class MetricConfig:
    SPIKE_THRESHOLD = 1.0
    HR_MODE = "chaotic_bursting"
    HR_PARAMETER_SETS = {
        "chaotic_bursting": {
            "a": 1.0,
            "b": 3.0,
            "c": 1.0,
            "d": 5.0,
            "r": 0.006,
            "s": 4.0,
            "xr": -1.6,
            "I": 3.25,
            "x0": [-1.0, -3.0, 3.0],
        }
    }


class LeakageConfig:
    SPIKE_THRESHOLD = 1.0
    CONTROL_SETTLING_TOLERANCE = 0.1
    CONTROL_SETTLING_CONSECUTIVE = 1
    CONTROL_LINEAR_K_SWEEP = [0.1, 0.2]
    CONTROL_VALIDATION_FRAC = 0.5
    CONTROL_TEST_FRAC = 0.5
    CONTROL_INPUT_CLIP = None
    CONTROL_DIVERGENCE_ABS_LIMIT = 100.0
    CONTROL_SCORE_ENERGY_WEIGHT = 0.01
    CONTROL_SCORE_SETTLING_WEIGHT = 0.001
    CONTROL_SCORE_SPIKE_WEIGHT = 0.0


class FakeControlledESN:
    def __init__(self):
        self.controlled_horizons = []

    def predict(self, _sequence, n_warmup):
        del n_warmup
        return np.zeros((10, 3)), np.zeros((10, 1))

    def predict_controlled(
        self,
        train_sequence,
        horizon_steps,
        target,
        K,
        control_start_idx=0,
        controller="linear_feedback",
        finite_s=0.8,
        pyragas_delay=20,
        pyragas_sign=-1,
        pyragas_history_signal="raw_readout",
        control_input_clip=None,
        divergence_abs_limit=None,
    ):
        del (
            train_sequence,
            target,
            control_start_idx,
            controller,
            finite_s,
            pyragas_delay,
            pyragas_sign,
            control_input_clip,
        )
        horizon_steps = int(horizon_steps)
        self.controlled_horizons.append((float(K), horizon_steps))
        corrected = np.zeros((horizon_steps, 3), dtype=float)
        if np.isclose(K, 0.1):
            if horizon_steps > 6:
                corrected[6:] = 10.0
        else:
            corrected[: min(6, horizon_steps)] = 1.0
        control = np.zeros_like(corrected)
        raw = corrected + control
        return {
            "stable": True,
            "divergence_detected": False,
            "divergence_reason": None,
            "divergence_index": None,
            "steps_completed": horizon_steps,
            "pyragas_history_signal": pyragas_history_signal,
            "control_input_clip": None,
            "divergence_abs_limit": divergence_abs_limit,
            "raw_readout_norm": raw,
            "corrected_feedback_input_norm": corrected,
            "control_signal_norm": control,
        }


class DivergentFinalESN(FakeControlledESN):
    def predict_controlled(self, *args, **kwargs):
        result = super().predict_controlled(*args, **kwargs)
        horizon_steps = int(kwargs["horizon_steps"])
        if horizon_steps > 6:
            result["stable"] = False
            result["divergence_detected"] = True
            result["divergence_reason"] = "synthetic_final_test_divergence"
            result["divergence_index"] = 7
        return result

class LegacyUnstableESN(FakeControlledESN):
    def predict_controlled(self, *args, **kwargs):
        result = super().predict_controlled(*args, **kwargs)
        result["stable"] = False
        result["divergence_detected"] = False
        result["divergence_reason"] = None
        return result


class ControlExperimentMethodologyTests(unittest.TestCase):
    def test_flat_above_threshold_episode_counts_as_one_spike(self):
        signal = np.array([0.0, 2.0, 2.0, 2.0, 0.0])
        self.assertEqual(ce._count_spikes(signal, threshold=1.0), 1)

    def test_separate_signal_metrics_and_true_dt_energy(self):
        times = np.array([0.0, 0.5, 1.0])
        target = np.zeros(3)
        uncontrolled = np.zeros((3, 3))
        raw = np.ones((3, 3))
        corrected = np.zeros((3, 3))
        control = np.array(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        metrics = ce._summarize_control_metrics(
            times=times,
            uncontrolled=uncontrolled,
            raw_readout=raw,
            controlled=corrected,
            target_state=target,
            control_signal=control,
            control_start_idx=0,
            eval_start_idx=0,
            eval_end_idx=3,
            x_normalization_scale=2.0,
            spike_threshold=1.0,
            settling_tolerance=0.1,
            settling_consecutive=1,
        )
        self.assertAlmostEqual(metrics["raw_readout_target_rmse_state"], 1.0)
        self.assertAlmostEqual(metrics["raw_readout_target_nrmse_x"], 0.5)
        self.assertAlmostEqual(
            metrics["corrected_feedback_input_target_rmse_state"], 0.0
        )
        self.assertAlmostEqual(metrics["control_effort_mean_sq"], 5.0 / 3.0)
        self.assertAlmostEqual(metrics["control_energy_dt_sum"], 2.5)
        self.assertEqual(
            metrics["control_energy"], metrics["control_effort_mean_sq"]
        )
        self.assertEqual(
            metrics["evaluation_time_to_tolerance"], metrics["settling_time"]
        )

    def test_empirical_quiet_reference_records_rhs_residual(self):
        train = np.tile(np.array([-1.0, -3.0, 3.0]), (100, 1))
        target_raw, _, metadata = ce._choose_target_state(
            train,
            np.zeros((1, 3)),
            np.ones((1, 3)),
            "rest_state_from_quiet_training_data",
            MetricConfig,
            hr_mode="chaotic_bursting",
            return_metadata=True,
        )
        self.assertEqual(
            metadata["reference_type"], "empirical_quiet_state_reference"
        )
        self.assertFalse(metadata["is_exact_equilibrium"])
        self.assertTrue(np.isfinite(metadata["hr_rhs_residual_norm"]))
        self.assertIn(
            "empirical quiet-state reference",
            metadata["regulation_objective"],
        )
        self.assertEqual(target_raw.shape, (3,))

    def test_gain_selection_cannot_inspect_controller_test(self):
        esn = FakeControlledESN()
        train = np.zeros((5, 3))
        test = np.zeros((10, 3))
        times = np.arange(15, dtype=float) * 0.25
        plot_names = (
            "plot_controlled_vs_uncontrolled_x",
            "plot_raw_readout_vs_corrected_feedback_input_x",
            "plot_controlled_all_states",
            "plot_control_signal",
            "plot_control_error",
            "plot_k_sweep_summary",
        )
        with tempfile.TemporaryDirectory() as temp_dir, ExitStack() as stack:
            for name in plot_names:
                stack.enter_context(mock.patch.object(ce, name, return_value=None))
            summary = ce.run_control_experiment(
                esn=esn,
                loader=None,
                config=LeakageConfig,
                train=train,
                test=test,
                train_norm=train,
                test_norm=test,
                mean=np.zeros((1, 3)),
                std=np.ones((1, 3)),
                times=times,
                base_output_dir=str(Path(temp_dir) / "chaotic_bursting"),
                hr_mode="chaotic_bursting",
                optimizer_name="test",
                control_start_frac=0.2,
                control_target_mode="zero",
                controller="linear_feedback",
            )

            # Both candidates stop at validation_end=6. Only the selected K is
            # rerun through the full horizon, so K=0.2's superior test tail
            # cannot affect selection.
            self.assertEqual(
                esn.controlled_horizons,
                [(0.1, 6), (0.2, 6), (0.1, 10)],
            )
            self.assertEqual(summary["best_k"], 0.1)
            self.assertEqual(summary["controller_validation_start"], 2)
            self.assertEqual(summary["controller_validation_end"], 6)
            self.assertEqual(summary["controller_test_start"], 6)
            self.assertEqual(summary["controller_test_end"], 10)
            self.assertEqual(
                summary["validation_metrics"][
                    "corrected_feedback_input_target_rmse_state"
                ],
                0.0,
            )
            self.assertEqual(
                summary["corrected_feedback_input_metrics"]["target_rmse_state"],
                10.0,
            )
            self.assertEqual(summary["final_test_metric_value"], 10.0)
            self.assertEqual(
                summary["final_test_metric_segment"], "controller_test"
            )
            self.assertEqual(
                summary["controller_law_coordinate_system"],
                "normalized_esn_coordinates",
            )

            rollout_path = (
                Path(temp_dir)
                / "chaotic_bursting"
                / "control"
                / "linear_feedback"
                / "best_rollout"
                / "rollout.csv"
            )
            with rollout_path.open(newline="") as handle:
                columns = csv.DictReader(handle).fieldnames
            for required in (
                "time_index",
                "raw_readout_x",
                "corrected_feedback_input_x",
                "control_signal_x",
                "controlled_x",
                "u_x",
            ):
                self.assertIn(required, columns)


    def test_validation_only_outer_candidate_never_runs_controller_test(self):
        esn = FakeControlledESN()
        train = np.zeros((5, 3))
        test = np.zeros((10, 3))
        plot_names = (
            "plot_controlled_vs_uncontrolled_x",
            "plot_raw_readout_vs_corrected_feedback_input_x",
            "plot_controlled_all_states",
            "plot_control_signal",
            "plot_control_error",
            "plot_k_sweep_summary",
        )
        with tempfile.TemporaryDirectory() as temp_dir, ExitStack() as stack:
            for name in plot_names:
                stack.enter_context(mock.patch.object(ce, name, return_value=None))
            summary = ce.run_control_experiment(
                esn=esn,
                loader=None,
                config=LeakageConfig,
                train=train,
                test=test,
                train_norm=train,
                test_norm=test,
                mean=np.zeros((1, 3)),
                std=np.ones((1, 3)),
                times=np.arange(15, dtype=float) * 0.25,
                base_output_dir=str(Path(temp_dir) / "chaotic_bursting"),
                hr_mode="chaotic_bursting",
                optimizer_name="test",
                control_start_frac=0.2,
                control_target_mode="zero",
                controller="linear_feedback",
                validation_only=True,
            )
            self.assertEqual(esn.controlled_horizons, [(0.1, 6), (0.2, 6)])
            self.assertTrue(summary["validation_only"])
            self.assertFalse(summary["controller_test_evaluated"])
            self.assertIsNone(summary["final_test_metric_value"])
            self.assertIsNone(summary["test_metrics"])
            self.assertEqual(
                summary["validation_metrics"]["metric_segment"],
                "controller_validation",
            )
            self.assertEqual(
                summary["controller_law_coordinate_system"],
                "normalized_esn_coordinates",
            )
            self.assertFalse(
                (
                    Path(temp_dir)
                    / "chaotic_bursting"
                    / "control"
                    / "linear_feedback"
                    / "best_rollout"
                ).exists()
            )

    def test_validation_only_records_all_unstable_candidate_as_rejected(self):
        esn = LegacyUnstableESN()
        train = np.zeros((5, 3))
        test = np.zeros((10, 3))
        plot_names = (
            "plot_controlled_vs_uncontrolled_x",
            "plot_raw_readout_vs_corrected_feedback_input_x",
            "plot_controlled_all_states",
            "plot_control_signal",
            "plot_control_error",
            "plot_k_sweep_summary",
        )
        with tempfile.TemporaryDirectory() as temp_dir, ExitStack() as stack:
            for name in plot_names:
                stack.enter_context(mock.patch.object(ce, name, return_value=None))
            output = Path(temp_dir) / "chaotic_bursting"
            summary = ce.run_control_experiment(
                esn=esn,
                loader=None,
                config=LeakageConfig,
                train=train,
                test=test,
                train_norm=train,
                test_norm=test,
                mean=np.zeros((1, 3)),
                std=np.ones((1, 3)),
                times=np.arange(15, dtype=float) * 0.25,
                base_output_dir=str(output),
                hr_mode="chaotic_bursting",
                optimizer_name="test",
                control_k=0.1,
                control_start_frac=0.2,
                control_target_mode="zero",
                controller="linear_feedback",
                validation_only=True,
            )

            self.assertTrue(summary["validation_only"])
            self.assertTrue(summary["candidate_rejected"])
            self.assertEqual(
                summary["candidate_status"],
                "rejected_no_stable_validation_gain",
            )
            self.assertFalse(summary["stable"])
            self.assertFalse(summary["controller_test_evaluated"])
            self.assertEqual(summary["validation_metrics"], {})
            self.assertIsNone(summary["test_metrics"])
            self.assertEqual(esn.controlled_horizons, [(0.1, 6)])
            control_dir = output / "control" / "linear_feedback"
            self.assertTrue((control_dir / "control_summary.json").is_file())
            self.assertTrue((control_dir / "k_sweep.csv").is_file())
            self.assertFalse((control_dir / "best_rollout").exists())

    def test_nonsettling_candidates_remain_rankable(self):
        common = {
            "controller": "linear_feedback",
            "stable": True,
            "divergence_detected": False,
            "control_effort_mean_sq": 0.0,
            "settling_time": float("nan"),
            "control_sample_dt": 0.25,
            "evaluation_sample_count": 4,
            "spike_reduction_percent": 0.0,
        }
        rows = [
            {
                **common,
                "K": 0.1,
                "corrected_feedback_input_target_rmse_state": 2.0,
            },
            {
                **common,
                "K": 0.2,
                "corrected_feedback_input_target_rmse_state": 1.0,
            },
        ]
        selected = ce._best_row(rows, LeakageConfig)
        self.assertEqual(selected["K"], 0.2)
        self.assertTrue(np.isfinite(selected["selection_score"]))

    def test_pyragas_score_uses_configured_weights(self):
        class Weighted:
            PYRAGAS_SCORE_FEW_SPIKES_WEIGHT = 30.0

        class Unweighted:
            PYRAGAS_SCORE_FEW_SPIKES_WEIGHT = 0.0

        row = {
            "controller": "pyragas",
            "stable": True,
            "K": 0.0,
            "control_effort_mean_sq": 0.0,
            "max_control_norm": 0.0,
            "pyragas_detected_peak_count": 0,
        }
        weighted = ce._selection_score(dict(row), Weighted)
        unweighted = ce._selection_score(dict(row), Unweighted)
        self.assertAlmostEqual(weighted - unweighted, 30.0)


    def test_all_unstable_candidates_raise(self):
        rows = [
            {
                "controller": "linear_feedback",
                "K": 0.1,
                "stable": False,
                "divergence_detected": True,
                "divergence_reason": "synthetic_validation_divergence",
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "No stable controller candidate"):
            ce._best_row(rows, LeakageConfig)

    def test_legacy_unstable_result_cannot_report_not_diverged(self):
        row, _, _, _ = ce._evaluate_k(
            esn=LegacyUnstableESN(), train_norm=np.zeros((5, 3)),
            n_base=6, K=0.1,
            target_norm=np.zeros(3), target_raw=np.zeros(3),
            mean=np.zeros((1, 3)), std=np.ones((1, 3)),
            uncontrolled=np.zeros((10, 3)),
            test_aligned=np.zeros((10, 3)),
            test_times_aligned=np.arange(10, dtype=float),
            control_start_idx=2, control_start_time=2.0,
            control_start_frac=0.2,
            control_target_mode="zero", hr_mode="chaotic_bursting",
            optimizer_name="test",
            spike_threshold=1.0, settling_tolerance=0.1,
            settling_consecutive=1,
            controller="linear_feedback", finite_s=0.8,
            pyragas_delay=20, pyragas_sign=-1,
            pyragas_history_signal="raw_readout",
            control_input_clip=None, divergence_abs_limit=100.0,
            metric_start_idx=2, metric_end_idx=6,
            metric_segment="controller_validation",
        )
        self.assertFalse(row["stable"])
        self.assertTrue(row["divergence_detected"])
        self.assertEqual(row["divergence_reason"], "rollout_reported_unstable")
        self.assertEqual(row["controller_law_coordinate_system"], "normalized_esn_coordinates")

    def test_final_controller_divergence_is_fatal_and_recorded(self):
        esn = DivergentFinalESN()
        train = np.zeros((5, 3))
        test = np.zeros((10, 3))
        plot_names = (
            "plot_controlled_vs_uncontrolled_x",
            "plot_raw_readout_vs_corrected_feedback_input_x",
            "plot_controlled_all_states",
            "plot_control_signal",
            "plot_control_error",
            "plot_k_sweep_summary",
        )
        with tempfile.TemporaryDirectory() as temp_dir, ExitStack() as stack:
            for name in plot_names:
                stack.enter_context(mock.patch.object(ce, name, return_value=None))
            output = Path(temp_dir) / "chaotic_bursting"
            with self.assertRaisesRegex(
                RuntimeError, "failed on controller test"
            ):
                ce.run_control_experiment(
                    esn=esn,
                    loader=None,
                    config=LeakageConfig,
                    train=train,
                    test=test,
                    train_norm=train,
                    test_norm=test,
                    mean=np.zeros((1, 3)),
                    std=np.ones((1, 3)),
                    times=np.arange(15, dtype=float) * 0.25,
                    base_output_dir=str(output),
                    hr_mode="chaotic_bursting",
                    optimizer_name="test",
                    control_k=0.1,
                    control_start_frac=0.2,
                    control_target_mode="zero",
                    controller="linear_feedback",
                )
            summary_path = (
                output
                / "control"
                / "linear_feedback"
                / "control_summary.json"
            )
            self.assertTrue(summary_path.exists())
            summary = __import__("json").loads(summary_path.read_text())
            self.assertFalse(summary["stable"])
            self.assertEqual(
                summary["divergence_reason"],
                "synthetic_final_test_divergence",
            )

if __name__ == "__main__":
    unittest.main()
