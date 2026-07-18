import ast
import unittest
from pathlib import Path

import config


ROOT = Path(__file__).resolve().parents[1]


def main_cli_options():
    tree = ast.parse((ROOT / "main.py").read_text())
    options = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                options.add(arg.value)
    return options


class CliContractTests(unittest.TestCase):
    def test_safe_and_chaotic_defaults(self):
        self.assertEqual(config.HR_MODE, "chaotic_bursting")
        self.assertFalse(config.CLEAR_OUTPUT_FOLDER_EACH_RUN)

    def test_three_controller_options_exist(self):
        options = main_cli_options()
        required = {
            "--control",
            "--params-file",
            "--controller",
            "--finite-s",
            "--pyragas-delay",
            "--pyragas-sign",
        }
        self.assertTrue(required.issubset(options), required - options)

    def test_slurm_controller_options_are_supported_by_main(self):
        options = main_cli_options()
        for path in ROOT.glob("*.slurm"):
            text = path.read_text()
            for option in ("--controller", "--pyragas-delay", "--pyragas-sign"):
                if option in text:
                    self.assertIn(option, options, f"{option} used by {path.name}")

    def test_official_slurm_is_one_pass_clean_and_strict(self):
        slurm = (ROOT / "run_final_thesis_pipeline.slurm").read_text()
        pipeline = (ROOT / "final_pipeline.py").read_text()
        for marker in (
            "git diff --quiet",
            "git diff --cached --quiet",
            "git ls-files --others --exclude-standard",
            "final_pipeline.py",
            "final_package_validation.json",
            "--clean-repository-at-start",
        ):
            self.assertIn(marker, slurm)
        self.assertNotIn("09_working_outputs", slurm)
        self.assertNotIn("best_params_dummy.json", slurm)
        self.assertNotIn("|| true", slurm)
        self.assertIn("for regime in HR_REGIMES", pipeline)
        self.assertIn("run_all_optimizers(", pipeline)
        self.assertIn("EchoStateNetwork.load_bundle(", pipeline)
        self.assertIn("if list(config.PYRAGAS_SIGNS) != [-1]", pipeline)
        self.assertIn("locked_validation_selection", pipeline)
        self.assertNotIn("subprocess.run([\"sbatch\"", pipeline)

    def test_final_validation_scripts_lock_selected_parameters(self):
        pyragas = (ROOT / "run_pyragas_final_validation.slurm").read_text()
        self.assertIn("--optimizer dummy", pyragas)
        self.assertIn("--control-k 0.8", pyragas)
        self.assertIn("--pyragas-delay 2400", pyragas)
        self.assertIn("--pyragas-sign -1", pyragas)

        fixed = (ROOT / "run_final_linear_finite_validation.slurm").read_text()
        self.assertIn("--controller linear_feedback", fixed)
        self.assertIn("--control-k 1.0", fixed)
        self.assertIn("--controller finite_time", fixed)
        self.assertIn("--control-k 0.4582142857142857", fixed)
        self.assertIn("--finite-s 0.8", fixed)


if __name__ == "__main__":
    unittest.main()
