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


if __name__ == "__main__":
    unittest.main()
