import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import config
import final_pipeline
from final_package import (
    REQUIRED_PACKAGE_DIRECTORIES,
    validate_final_package,
)
from model import EchoStateNetwork
from optimize_model import (
    _validation_window_metrics,
    prediction_validation_spec,
)


ROOT = Path(__file__).resolve().parents[1]


class FinalizationAcceptanceTests(unittest.TestCase):
    def test_three_training_only_recursive_windows_and_heldout_boundaries(self):
        training = np.zeros((105000, 3), dtype=float)
        spec = prediction_validation_spec(
            training,
            series_is_training_portion=True,
            heldout_length=45000,
        )
        self.assertEqual(len(spec["windows"]), 3)
        self.assertEqual(
            [(row["start"], row["end"]) for row in spec["windows"]],
            [(81000, 89000), (89000, 97000), (97000, 105000)],
        )
        self.assertEqual(spec["heldout_test_start"], 105000)
        self.assertEqual(spec["heldout_test_end"], 150000)
        self.assertFalse(spec["test_data_used_for_selection"])

    def test_spike_frequency_error_changes_validation_score(self):
        true_x = np.full(120, -1.0)
        for start in (10, 50, 90):
            true_x[start : start + 4] = 1.0
        true = np.column_stack([true_x, true_x, true_x])
        wrong = np.full_like(true, -1.0)
        with (
            mock.patch.object(config, "PREDICTION_STATE_X_WEIGHT", 0.0),
            mock.patch.object(config, "PREDICTION_MULTISTATE_WEIGHT", 0.0),
            mock.patch.object(
                config, "PREDICTION_SPIKE_FREQUENCY_WEIGHT", 1.0
            ),
            mock.patch.object(
                config, "PREDICTION_SPIKE_INTERVAL_WEIGHT", 0.0
            ),
        ):
            correct_metrics = _validation_window_metrics(true, true, 0.0)
            wrong_metrics = _validation_window_metrics(wrong, true, 0.0)
        self.assertEqual(
            correct_metrics["spike_frequency_rel_error"], 0.0
        )
        self.assertGreater(
            wrong_metrics["spike_frequency_rel_error"], 0.0
        )
        self.assertGreater(wrong_metrics["score"], correct_metrics["score"])

    def test_saved_esn_round_trip_preserves_identity_and_prediction(self):
        rng = np.random.default_rng(42)
        training = rng.normal(size=(90, 3))
        evaluation = rng.normal(size=(120, 3))
        model = EchoStateNetwork(
            N_res=12,
            p=0.25,
            spectral_radius=0.7,
            leaky_coefficient=0.3,
            regularization=1e-5,
            input_size=3,
            input_scaling=0.2,
            seed=42,
        )
        model.train(training, washout=10)
        expected_prediction = model.predict(evaluation, n_warmup=20)[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model_bundle.npz"
            metadata = model.save_bundle(
                path,
                metadata={"source_regime": "chaotic_bursting"},
                external_mean=np.zeros((1, 3)),
                external_std=np.ones((1, 3)),
            )
            loaded, loaded_metadata = EchoStateNetwork.load_bundle(path)
        actual_prediction = loaded.predict(evaluation, n_warmup=20)[0]
        np.testing.assert_allclose(actual_prediction, expected_prediction)
        self.assertEqual(
            loaded.model_identity_hash(), model.model_identity_hash()
        )
        self.assertEqual(
            loaded_metadata["model_identity_hash"],
            metadata["model_identity_hash"],
        )
        self.assertTrue(loaded_metadata["loaded_from_cache"])

    def _make_minimal_valid_package(self, root: Path):
        for name in REQUIRED_PACKAGE_DIRECTORIES:
            (root / name).mkdir(parents=True)
        (root / "00_manifest" / "stage_timings.csv").write_text(
            "stage,seconds\ntotal_runtime,1.0\n",
            encoding="utf-8",
        )
        (root / "00_manifest" / "run_manifest.json").write_text(
            json.dumps(
                {
                    "git": {"commit": "abc"},
                    "machine_specific": {
                        "external_root": "/home/example/evidence"
                    },
                }
            ),
            encoding="utf-8",
        )
        for regime in final_pipeline.HR_REGIMES:
            prediction = root / "01_prediction_all_regimes" / regime
            prediction.mkdir()
            (prediction / "selected_model.json").write_text(
                json.dumps(
                    {
                        "selection_status": (
                            "locked_before_heldout_evaluation"
                        ),
                        "model_identity_hash": f"prediction-{regime}",
                    }
                ),
                encoding="utf-8",
            )
            (prediction / "heldout_test_metrics.json").write_text(
                json.dumps({"quality_gate": {"passed": True}}),
                encoding="utf-8",
            )
            (prediction / "model_bundle.npz").write_bytes(b"smoke-bundle")
            bo = root / "02_bo_optimization" / regime
            bo.mkdir()
            (bo / "best_params.json").write_text(
                json.dumps({"validation_score": 0.1}),
                encoding="utf-8",
            )
            (bo / "validation_windows.json").write_text(
                json.dumps({"windows": [1, 2, 3]}),
                encoding="utf-8",
            )
            (bo / "optimizer_validation_summary.csv").write_text(
                "optimizer,score\ndummy,0.1\n",
                encoding="utf-8",
            )
        for section, controller in zip(
            (
                "03_linear_feedback",
                "04_finite_time",
                "05_pyragas",
            ),
            ("linear_feedback", "finite_time", "pyragas"),
        ):
            (root / section / "control_summary.json").write_text(
                json.dumps(
                    {
                        "controller": controller,
                        "stable": True,
                        "model_identity_hash": "same-chaotic-model",
                        "control_model_source": "validation_selected",
                        "reference_type": (
                            "empirical_quiet_state_reference"
                        ),
                    }
                ),
                encoding="utf-8",
            )
        figure = (
            root
            / "01_prediction_all_regimes"
            / "periodic_spiking"
            / "results_all_states.png"
        )
        Image.new("RGB", (4, 4), color="white").save(figure)

    def test_package_validator_accepts_relative_curated_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_minimal_valid_package(root)
            report = validate_final_package(
                root,
                expected_commit="abc",
                clean_repository_at_start=True,
                quality_gates_passed=True,
            )
        self.assertTrue(report["valid"], report["errors"])
        self.assertTrue(report["same_control_model_identity"])
        self.assertTrue(report["no_unexpected_absolute_paths"])
        self.assertEqual(report["unexpected_directories"], [])

    def test_duplicate_prediction_figure_in_controller_is_fatal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_minimal_valid_package(root)
            source = (
                root
                / "01_prediction_all_regimes"
                / "periodic_spiking"
                / "results_all_states.png"
            )
            destination = (
                root / "03_linear_feedback" / "results_all_states.png"
            )
            destination.write_bytes(source.read_bytes())
            report = validate_final_package(
                root,
                expected_commit="abc",
                clean_repository_at_start=True,
                quality_gates_passed=True,
            )
        self.assertFalse(report["valid"])
        self.assertFalse(
            report["no_unnecessary_duplicate_prediction_figures"]
        )
        self.assertIn(
            "03_linear_feedback/results_all_states.png",
            report["misplaced_prediction_figures"],
        )

    def test_one_pass_and_no_controller_bo_source_contract(self):
        source = (ROOT / "final_pipeline.py").read_text(encoding="utf-8")
        controller_start = source.index("def _run_controllers(")
        controller_end = source.index("\ndef _comparison_tables(", controller_start)
        controller_source = source[controller_start:controller_end]
        self.assertNotIn("run_all_optimizers(", controller_source)
        self.assertIn(
            "if len(bo_invocations) != bo_invocation_count_before",
            controller_source,
        )
        self.assertIn("EchoStateNetwork.load_bundle(", controller_source)
        self.assertNotIn("09_working_outputs", source)
        self.assertIn("generate_plots=False", controller_source)

    def test_controller_orchestrator_reuses_one_loaded_model(self):
        class CachedModel:
            seed = 42

            def model_identity_hash(self):
                return "shared-chaotic-hash"

        cached_model = CachedModel()
        observed_models = []
        artifact = {
            "loader": None,
            "train": np.zeros((10, 3)),
            "test": np.zeros((6, 3)),
            "train_norm": np.zeros((10, 3)),
            "test_norm": np.zeros((6, 3)),
            "mean": np.zeros((1, 3)),
            "std": np.ones((1, 3)),
            "times": np.arange(16, dtype=float),
            "pred_norm": np.zeros((6, 3)),
            "best_params": {},
            "selected_optimizer": "dummy",
            "model_bundle_path": Path("/unused/model.npz"),
            "model_bundle_relative": (
                "01_prediction_all_regimes/chaotic_bursting/"
                "model_bundle.npz"
            ),
            "model_identity_hash": "shared-chaotic-hash",
        }

        def fake_control(**kwargs):
            observed_models.append(kwargs["esn"])
            controller = kwargs["controller"]
            provenance = kwargs["model_provenance"]
            if kwargs["validation_only"]:
                validation = {
                    "K": 0.1,
                    "stable": True,
                    "divergence_detected": False,
                    "selection_score": 0.1,
                }
                return {
                    "controller": controller,
                    "stable": True,
                    "best_k": 0.1,
                    "selection_metric_value": 0.1,
                    "validation_metrics": validation,
                    "selection_runtime_seconds": 0.01,
                    "finite_s": kwargs.get("finite_s"),
                    "pyragas_delay": kwargs.get("pyragas_delay"),
                }
            return {
                "controller": controller,
                "stable": True,
                "model_identity_hash": provenance[
                    "model_identity_hash"
                ],
                "selection_runtime_seconds": 0.01,
                "final_test_runtime_seconds": 0.02,
            }

        bo_invocations = [
            {"regime": regime} for regime in final_pipeline.HR_REGIMES
        ]
        timings = []
        with (
            mock.patch.object(
                final_pipeline.EchoStateNetwork,
                "load_bundle",
                return_value=(cached_model, {"loaded_from_cache": True}),
            ),
            mock.patch.object(
                final_pipeline,
                "run_control_experiment",
                side_effect=fake_control,
            ),
            mock.patch.object(
                final_pipeline, "FINITE_TIME_EXPONENTS", (0.8,)
            ),
            mock.patch.object(
                final_pipeline, "PYRAGAS_DELAYS", (20,)
            ),
        ):
            results = final_pipeline._run_controllers(
                Path("/tmp/package"),
                artifact,
                "commit",
                "config-hash",
                timings,
                len(bo_invocations),
                bo_invocations,
            )

        self.assertEqual(
            set(results),
            {"linear_feedback", "finite_time", "pyragas"},
        )
        self.assertTrue(observed_models)
        self.assertTrue(
            all(model is cached_model for model in observed_models)
        )
        self.assertEqual(len(bo_invocations), 3)
        self.assertEqual(
            {
                result["model_identity_hash"]
                for result in results.values()
            },
            {"shared-chaotic-hash"},
        )

    def test_selection_lock_is_written_before_heldout_array_use(self):
        source = (ROOT / "final_pipeline.py").read_text(encoding="utf-8")
        function_start = source.index("def _train_and_evaluate_regime(")
        function_end = source.index("\ndef _control_kwargs(", function_start)
        function_source = source[function_start:function_end]
        self.assertLess(
            function_source.index("_write_json(selected_model_path"),
            function_source.index("test_norm = (test - mean) / std"),
        )
        self.assertIn("selection_series=train", function_source)
        self.assertIn(
            '"heldout_array_passed_to_selection": False',
            function_source,
        )

    def test_empirical_reference_and_strict_controller_contract(self):
        source = (ROOT / "control_experiment.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("empirical_quiet_state_reference", source)
        self.assertIn(
            "regulation toward an empirical quiet-state reference", source
        )
        self.assertIn('"rejected": True', source)
        self.assertIn(
            "Selected {controller} controller failed on controller test",
            source,
        )
        self.assertIn("if generate_plots:", source)


if __name__ == "__main__":
    unittest.main()
