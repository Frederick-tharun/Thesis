import unittest
import json
import tempfile
from pathlib import Path
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np

import config
import optimize_model
import main


PARAMS = {
    "N_res": 10,
    "p": 0.10,
    "spectral_radius": 0.80,
    "leaky_coefficient": 0.25,
    "input_scaling": 0.20,
    "regularization": 1e-6,
    "washout": 5,
}


def seed_metrics(score, reservoir_seed, stable=True):
    reason = "ok" if stable else "failed"
    return float(score), {
        "reservoir_seed": int(reservoir_seed),
        "score": float(score),
        "validation_score": float(score),
        "validation_nrmse": float(score) + 0.1,
        "validation_nrmse_x": float(score) + 0.2,
        "validation_std_ratio": 1.0,
        "validation_mean_gap": 0.0,
        "validation_penalty": 0.0,
        "stable": bool(stable),
        "reason": reason,
    }


class OptimizerConfigurationTests(unittest.TestCase):
    def test_official_search_space_is_mapped_to_internal_parameter_names(self):
        search_space = {
            "reservoir_size": (10, 20, "int", False),
            "spectral_radius": (0.6, 1.1, "float", False),
            "leak_rate": (0.1, 0.7, "float", False),
            "input_scaling": (0.05, 0.9, "float", False),
            "regularization": (1e-9, 1e-4, "float", True),
            "sparsity": (0.02, 0.2, "float", False),
            "washout": (5, 25, "int", False),
        }

        with mock.patch.object(config, "BO_SEARCH_SPACE", search_space):
            dimensions = optimize_model._get_search_space(input_size=3)

        self.assertEqual(
            [dimension.name for dimension in dimensions],
            [
                "N_res",
                "p",
                "spectral_radius",
                "leaky_coefficient",
                "input_scaling",
                "regularization",
                "washout",
            ],
        )
        self.assertEqual((dimensions[0].low, dimensions[0].high), (10, 20))
        self.assertEqual((dimensions[2].low, dimensions[2].high), (0.6, 1.1))
        self.assertEqual(dimensions[5].prior, "log-uniform")

    def test_deprecated_config_name_is_a_warned_fallback(self):
        legacy_name = "_TEST_LEGACY_BO_CALLS"
        setattr(config, legacy_name, 7)
        try:
            with self.assertWarns(FutureWarning):
                value = optimize_model._get_official_config(
                    "_TEST_OFFICIAL_BO_N_CALLS",
                    (legacy_name,),
                    30,
                )
        finally:
            delattr(config, legacy_name)

        self.assertEqual(value, 7)

    def test_bo_n_calls_controls_iterations_and_default_seed_is_fixed(self):
        loader = SimpleNamespace(
            data_raw=np.arange(2700, dtype=float).reshape(900, 3)
        )
        used_seeds = []

        def fake_seed_evaluation(**kwargs):
            seed = kwargs["reservoir_seed"]
            used_seeds.append(seed)
            return seed_metrics(0.25, seed)

        with (
            mock.patch.object(config, "DATASET_MODE", "hr"),
            mock.patch.object(config, "TRAIN_RATIO", 0.70),
            mock.patch.object(config, "BO_N_CALLS", 3),
            mock.patch.object(config, "BO_N_RANDOM_STARTS", 1),
            mock.patch.object(config, "BO_RESERVOIR_SEED", 42),
            mock.patch.object(config, "BO_EVALUATION_SEEDS", [42]),
            mock.patch.object(config, "PREDICTION_VALIDATION_NUM_WINDOWS", 3),
            mock.patch.object(config, "PREDICTION_VALIDATION_WINDOW_LENGTH", 100),
            mock.patch.object(config, "BO_CALLS", 99, create=True),
            mock.patch.object(
                optimize_model,
                "_evaluate_params_for_seed",
                side_effect=fake_seed_evaluation,
            ),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            result = optimize_model.optimize_hyperparameters(
                loader,
                optimizer="dummy",
            )

        self.assertEqual(len(result.history), 3)
        self.assertEqual(used_seeds, [42, 42, 42])
        self.assertEqual(result.best_params["reservoir_seed"], 42)
        self.assertEqual(result.best_params["evaluation_seeds"], [42])
        self.assertEqual(result.best_params["evaluation_seed_count"], 1)
        self.assertEqual(result.best_params["score_aggregation"], "mean")

    def test_multiple_seed_scores_are_averaged_and_saved_per_seed(self):
        scores = {42: 1.0, 43: 3.0}

        def fake_seed_evaluation(**kwargs):
            seed = kwargs["reservoir_seed"]
            return seed_metrics(scores[seed], seed)

        train = np.arange(36, dtype=float).reshape(12, 3)
        val = np.arange(24, dtype=float).reshape(8, 3)

        with mock.patch.object(
            optimize_model,
            "_evaluate_params_for_seed",
            side_effect=fake_seed_evaluation,
        ):
            score, row = optimize_model._evaluate_params(
                params=PARAMS,
                train=train,
                val=val,
                input_size=3,
                iteration=1,
                optimizer="dummy",
                best_score=float("inf"),
                evaluation_seeds=[42, 43],
                reservoir_seed=42,
                optimizer_seed=142,
            )

        self.assertAlmostEqual(score, 2.0)
        self.assertAlmostEqual(row["validation_score"], 2.0)
        self.assertAlmostEqual(row["validation_score_std"], 1.0)
        self.assertAlmostEqual(row["validation_score_min"], 1.0)
        self.assertAlmostEqual(row["validation_score_max"], 3.0)
        self.assertEqual(row["validation_score_seed_42"], 1.0)
        self.assertEqual(row["validation_score_seed_43"], 3.0)
        self.assertEqual(row["evaluation_seeds"], [42, 43])
        self.assertEqual(row["evaluation_seed_count"], 2)
        self.assertEqual(row["optimizer_random_seed"], 142)
        self.assertTrue(row["stable"])

    def test_primary_seed_must_be_one_of_evaluation_seeds(self):
        with (
            mock.patch.object(config, "BO_RESERVOIR_SEED", 42),
            mock.patch.object(config, "BO_EVALUATION_SEEDS", [43, 44]),
        ):
            with self.assertRaisesRegex(ValueError, "must be included"):
                optimize_model._get_reservoir_seed_config()

    def test_model_helper_uses_explicit_reservoir_seed(self):
        model = optimize_model._make_model(
            PARAMS,
            input_size=3,
            reservoir_seed=77,
        )
        self.assertEqual(model.seed, 77)

    def test_final_model_helper_uses_selected_reservoir_seed(self):
        params = {**PARAMS, "reservoir_seed": 77}
        model = main.make_model(params, input_size=3)
        self.assertEqual(model.seed, 77)

    def test_selected_parameter_file_is_validated_and_marked_reused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "selected.json"
            path.write_text(
                json.dumps({**PARAMS, "reservoir_seed": 77}),
                encoding="utf-8",
            )
            loaded = main.load_selected_params(path)

        self.assertTrue(loaded["optimization_reused"])
        self.assertEqual(loaded["parameter_source_file"], str(path))
        self.assertEqual(loaded["reservoir_seed"], 77)

    def test_requested_control_failure_is_reraised(self):
        args = SimpleNamespace(
            control=True, controller="linear_feedback",
            control_target_mode="zero", control_k=0.1,
            control_start_frac=0.2, auto_control_k=False,
            k_min=None, k_max=None, k_num=None, k_refine_num=None,
            finite_s=0.8, pyragas_delay=20, pyragas_sign=-1,
            pyragas_history_signal="raw_readout", control_validation_only=False,
        )
        sample = np.zeros((3, 3))
        with (
            mock.patch.object(config, "DATASET_MODE", "hr"),
            mock.patch.object(
                main,
                "run_control_experiment",
                side_effect=RuntimeError("synthetic controller failure"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic controller failure"):
                main.run_control_if_requested(
                    args, object(), None,
                    sample, sample, sample, sample,
                    np.zeros((1, 3)), np.ones((1, 3)),
                    np.arange(6, dtype=float), "/tmp",
                    "chaotic_bursting", PARAMS, "dummy", 3,
                )

    def test_params_file_skips_optimizer_in_run_single_experiment(self):
        t = np.arange(30, dtype=float)
        series = np.column_stack((np.sin(t), np.cos(t), 0.1 * t))
        loader = SimpleNamespace(
            time=t,
            load=mock.Mock(),
            preprocess=mock.Mock(),
            detect_spikes=mock.Mock(),
            summary=mock.Mock(),
            list_neurons=mock.Mock(),
        )
        esn = SimpleNamespace(
            train=mock.Mock(),
            predict=mock.Mock(
                return_value=(np.zeros((9, 3)), np.zeros((9, 1)))
            ),
        )
        args = SimpleNamespace(
            dataset="hr", hr_mode="periodic_spiking", neuron=0,
            params_file=None, no_opt=False, optimizer="dummy",
            clean_output=True, control=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "selected.json"
            path.write_text(
                json.dumps({**PARAMS, "reservoir_seed": 77}),
                encoding="utf-8",
            )
            args.params_file = str(path)

            def fake_output_folder(*_args):
                config.OUTPUT_DIR = temp_dir
                path.unlink()
                return temp_dir

            with (
                mock.patch.object(config, "HR_MODE", config.HR_MODE),
                mock.patch.object(main, "DataLoader", return_value=loader),
                mock.patch.object(main, "get_model_series", return_value=(series, "hr_full_state")),
                mock.patch.object(main, "make_output_folder", side_effect=fake_output_folder),
                mock.patch.object(main, "make_model", return_value=esn) as make_model_mock,
                mock.patch.object(main, "optimize_hyperparameters", side_effect=AssertionError("optimizer called")),
                mock.patch.object(main, "run_all_optimizers", side_effect=AssertionError("auto optimizer called")),
                mock.patch.object(main, "save_json"),
                mock.patch.object(main, "plot_results"),
                mock.patch.object(main, "plot_all_states"),
                mock.patch.object(main, "call_experiment_report"),
                mock.patch.object(main, "run_control_if_requested", return_value=None),
            ):
                result = main.run_single_experiment(args)

        selected = make_model_mock.call_args.args[0]
        self.assertEqual(selected["reservoir_seed"], 77)
        self.assertTrue(selected["optimization_reused"])
        self.assertEqual(result["selected_params"]["reservoir_seed"], 77)


    def test_final_comparison_row_labels_legacy_effort_as_an_alias(self):
        row = main._make_final_comparison_row({
            "mode": "chaotic_bursting",
            "optimizer": "dummy",
            "selected_params": {},
            "control_result": {
                "controller": "linear_feedback",
                "control_effort_mean_sq": 0.25,
                "control_energy_dt_sum": 1.75,
                "best_control_energy": 0.25,
                "controller_test_time_to_tolerance": 2.0,
            },
        })
        self.assertEqual(row["Control_effort_mean_sq"], 0.25)
        self.assertEqual(row["Control_energy_dt_sum"], 1.75)
        self.assertEqual(row["Control_energy_legacy_alias"], 0.25)
        self.assertNotIn("Control_energy", row)

    def test_validation_only_main_skips_final_comparison_table(self):
        args = SimpleNamespace(
            output_root=None,
            run_all_regimes=False,
            control_validation_only=True,
        )
        result = {"mode": "chaotic_bursting"}
        with (
            mock.patch.object(main, "parse_args", return_value=args),
            mock.patch.object(main, "run_single_experiment", return_value=result),
            mock.patch.object(main, "save_final_comparison_table") as save_mock,
        ):
            main.main()

        save_mock.assert_not_called()

if __name__ == "__main__":
    unittest.main()
