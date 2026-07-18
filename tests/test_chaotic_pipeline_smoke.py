import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


plotting_stub = types.ModuleType("plotting")
for function_name in (
    "plot_controlled_vs_uncontrolled_x",
    "plot_controlled_all_states",
    "plot_raw_readout_vs_corrected_feedback_input_x",
    "plot_control_signal",
    "plot_control_error",
    "plot_k_sweep_summary",
):
    setattr(plotting_stub, function_name, lambda *args, **kwargs: None)
previous_plotting_module = sys.modules.get("plotting")
sys.modules["plotting"] = plotting_stub
from control_experiment import run_control_experiment
if previous_plotting_module is None:
    sys.modules.pop("plotting", None)
else:
    sys.modules["plotting"] = previous_plotting_module
from model import EchoStateNetwork


def hr_rhs(state):
    x, y, z = state
    return np.array(
        [
            y - x**3 + 3.0 * x**2 - z + 3.25,
            1.0 - 5.0 * x**2 - y,
            0.006 * (4.0 * (x + 1.6) - z),
        ]
    )


def chaotic_hr_trajectory(n_steps=900, dt=0.01):
    trajectory = np.zeros((n_steps, 3), dtype=float)
    state = np.array([-1.0, -3.0, 3.0], dtype=float)
    for index in range(n_steps):
        trajectory[index] = state
        k1 = hr_rhs(state)
        k2 = hr_rhs(state + 0.5 * dt * k1)
        k3 = hr_rhs(state + 0.5 * dt * k2)
        k4 = hr_rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return trajectory


class SmokeConfig:
    SPIKE_THRESHOLD = 1.0
    CONTROL_SETTLING_TOLERANCE = 0.25
    CONTROL_SETTLING_CONSECUTIVE = 10
    CONTROL_AUTO_K_MIN = 0.05
    CONTROL_AUTO_K_MAX = 0.20
    CONTROL_AUTO_K_NUM = 3
    CONTROL_AUTO_K_REFINE_NUM = 3
    CONTROL_AUTO_K_REFINE_WIDTH_FRAC = 0.15
    CONTROL_SCORE_ENERGY_WEIGHT = 0.01
    CONTROL_SCORE_SETTLING_WEIGHT = 0.001
    CONTROL_SCORE_SPIKE_WEIGHT = 0.0


class ChaoticPipelineSmokeTests(unittest.TestCase):
    def test_all_three_controllers_complete_and_save_results(self):
        series = chaotic_hr_trajectory()
        split = int(0.70 * len(series))
        train, test = series[:split], series[split:]
        mean = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True)
        train_norm = (train - mean) / std
        test_norm = (test - mean) / std

        esn = EchoStateNetwork(
            N_res=20,
            p=0.20,
            spectral_radius=0.70,
            leaky_coefficient=0.25,
            regularization=1e-5,
            input_size=3,
            normalize_input=False,
            input_scaling=0.15,
            seed=42,
        )
        esn.train(train_norm, washout=20)

        with tempfile.TemporaryDirectory() as temp_dir:
            base_output = Path(temp_dir) / "chaotic_bursting"
            for controller in ("linear_feedback", "finite_time", "pyragas"):
                result = run_control_experiment(
                    esn=esn,
                    loader=None,
                    config=SmokeConfig,
                    train=train,
                    test=test,
                    train_norm=train_norm,
                    test_norm=test_norm,
                    mean=mean,
                    std=std,
                    times=np.arange(len(series), dtype=float) * 0.01,
                    base_output_dir=str(base_output),
                    hr_mode="chaotic_bursting",
                    optimizer_name="smoke",
                    control_k=0.05,
                    control_start_frac=0.20,
                    control_target_mode="rest_state",
                    controller=controller,
                    finite_s=0.8,
                    pyragas_delay=20,
                    pyragas_sign=-1,
                )

                self.assertEqual(result["controller"], controller)
                summary_path = (
                    base_output
                    / "control"
                    / controller
                    / "control_summary.json"
                )
                self.assertTrue(summary_path.exists())
                summary = json.loads(summary_path.read_text())
                self.assertEqual(summary["best"]["controller"], controller)
                self.assertIn("stable", summary["best"])


if __name__ == "__main__":
    unittest.main()
