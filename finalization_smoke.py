from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import config
from control_experiment import run_control_experiment
from data_loader import DataLoader
from final_package import (
    REQUIRED_PACKAGE_DIRECTORIES,
    validate_final_package,
)
from main import run_all_optimizers, set_hr_mode
from model import EchoStateNetwork


SMOKE_SEARCH_SPACE = {
    "reservoir_size": (79, 81, "int", False),
    "sparsity": (0.085, 0.095, "float", False),
    "spectral_radius": (0.625, 0.635, "float", False),
    "leak_rate": (0.625, 0.635, "float", False),
    "input_scaling": (0.445, 0.455, "float", False),
    "regularization": (2e-9, 4e-9, "float", True),
    "washout": (90, 110, "int", False),
}


def periodic_selection_smoke(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (
        mock.patch.object(config, "DATASET_MODE", "hr"),
        mock.patch.object(config, "HR_TOTAL_STEPS", 9000),
        mock.patch.object(config, "BO_N_CALLS", 2),
        mock.patch.object(config, "BO_N_RANDOM_STARTS", 1),
        mock.patch.object(config, "BO_SEARCH_SPACE", SMOKE_SEARCH_SPACE),
        mock.patch.object(config, "OPTIMIZERS_TO_COMPARE", ["dummy"]),
        mock.patch.object(config, "PREDICTION_VALIDATION_NUM_WINDOWS", 3),
        mock.patch.object(
            config, "PREDICTION_VALIDATION_WINDOW_LENGTH", 1200
        ),
        mock.patch.object(config, "PREDICTION_VALIDATION_WINDOW_STARTS", None),
        mock.patch.object(config, "OUTPUT_DIR", str(output_dir)),
    ):
        set_hr_mode("periodic_spiking")
        loader = DataLoader(csv_path=config.DATA_PATH)
        loader.load()
        loader.preprocess()
        series = np.asarray(loader.data_raw, dtype=float)[:, :3]
        split = int(len(series) * float(config.TRAIN_RATIO))
        training = series[:split]
        heldout_length = len(series) - split
        selected_optimizer, best_params, summary, history = run_all_optimizers(
            loader,
            0,
            selection_series=training,
            heldout_length=heldout_length,
        )

    window_metrics = best_params.get("validation_window_metrics", [])
    result = {
        "passed": bool(
            selected_optimizer == "dummy"
            and best_params.get("test_data_used_for_selection") is False
            and len(best_params.get("validation_windows", [])) == 3
            and len(window_metrics) == 3
            and all(
                "spike_frequency_rel_error" in row
                and "isi_rel_error" in row
                for row in window_metrics
            )
            and len(history) == 2
        ),
        "selected_optimizer": selected_optimizer,
        "validation_window_count": len(
            best_params.get("validation_windows", [])
        ),
        "validation_window_metrics": window_metrics,
        "heldout_array_passed_to_selection": False,
        "bo_calls": len(history),
        "optimizer_summary": summary,
    }
    if not result["passed"]:
        raise RuntimeError("Reduced periodic-spiking selection smoke failed.")
    return result


def _chaotic_trajectory(n_steps: int = 700, dt: float = 0.01) -> np.ndarray:
    params = config.HR_PARAMETER_SETS["chaotic_bursting"]
    state = np.asarray(params["x0"], dtype=float)
    trajectory = np.zeros((n_steps, 3), dtype=float)

    def rhs(value):
        x, y, z = value
        return np.asarray(
            [
                y
                - params["a"] * x**3
                + params["b"] * x**2
                - z
                + params["I"],
                params["c"] - params["d"] * x**2 - y,
                params["r"] * (params["s"] * (x - params["xr"]) - z),
            ]
        )

    for index in range(n_steps):
        trajectory[index] = state
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return trajectory


class _AlwaysDivergentModel:
    def predict_controlled(
        self,
        *,
        train_sequence,
        horizon_steps,
        target,
        K,
        control_start_idx,
        **kwargs,
    ):
        del train_sequence, target, K, control_start_idx, kwargs
        zeros = np.zeros((horizon_steps, 3), dtype=float)
        return {
            "stable": False,
            "divergence_detected": True,
            "divergence_reason": "smoke_forced_final_divergence",
            "divergence_index": 0,
            "steps_completed": 1,
            "raw_readout_norm": zeros,
            "corrected_feedback_input_norm": zeros,
            "control_signal_norm": zeros,
        }


def controller_cache_and_rejection_smoke(output_dir: Path) -> dict:
    series = _chaotic_trajectory()
    split = int(0.70 * len(series))
    train, test = series[:split], series[split:]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    train_norm = (train - mean) / std
    test_norm = (test - mean) / std
    eval_norm = np.vstack([train_norm, test_norm])
    model = EchoStateNetwork(
        N_res=18,
        p=0.25,
        spectral_radius=0.7,
        leaky_coefficient=0.3,
        regularization=1e-5,
        input_size=3,
        input_scaling=0.2,
        seed=42,
    )
    model.train(train_norm, washout=20)
    bundle_path = output_dir / "chaotic_model_bundle.npz"
    model.save_bundle(
        bundle_path,
        metadata={
            "source_regime": "chaotic_bursting",
            "selected_optimizer": "smoke",
        },
        external_mean=mean,
        external_std=std,
    )
    cached_model, metadata = EchoStateNetwork.load_bundle(bundle_path)
    pred_norm = cached_model.predict(
        eval_norm, n_warmup=len(train_norm) - 1
    )[0]
    times = np.arange(len(series), dtype=float) * 0.01
    common = {
        "loader": None,
        "config": config,
        "train": train,
        "test": test,
        "train_norm": train_norm,
        "test_norm": test_norm,
        "mean": mean,
        "std": std,
        "times": times,
        "base_output_dir": str(output_dir),
        "hr_mode": "chaotic_bursting",
        "optimizer_name": "smoke",
        "control_start_frac": 0.20,
        "control_target_mode": "rest_state_from_quiet_training_data",
        "controller": "linear_feedback",
        "generate_plots": False,
        "uncontrolled_prediction_norm": pred_norm,
        "append_global_comparison": False,
        "model_provenance": {
            "model_identity_hash": cached_model.model_identity_hash(),
            "model_loaded_from_cache": True,
            "control_model_source": "validation_selected",
        },
    }

    with mock.patch.object(
        config, "CONTROL_DIVERGENCE_ABS_LIMIT", 1e-12
    ):
        rejected = run_control_experiment(
            esn=cached_model,
            controller_output_dir=str(output_dir / "rejected"),
            artifact_relative_path="rejected",
            validation_only=True,
            control_k=0.05,
            **common,
        )
    if not (rejected.get("rejected") and not rejected.get("stable")):
        raise RuntimeError("Unstable validation candidate was not rejected.")

    with mock.patch.object(config, "CONTROL_DIVERGENCE_ABS_LIMIT", 20.0):
        selected = run_control_experiment(
            esn=cached_model,
            controller_output_dir=str(output_dir / "selected"),
            artifact_relative_path="selected",
            validation_only=True,
            control_k=0.05,
            **common,
        )
        if not selected.get("stable"):
            raise RuntimeError("Reduced controller smoke found no stable candidate.")
        final = run_control_experiment(
            esn=cached_model,
            controller_output_dir=str(output_dir / "final"),
            artifact_relative_path="final",
            validation_only=False,
            control_k=float(selected["best_k"]),
            locked_validation_selection=selected["validation_metrics"],
            **common,
        )

    strict_failure_observed = False
    try:
        run_control_experiment(
            esn=_AlwaysDivergentModel(),
            controller_output_dir=str(output_dir / "strict_failure"),
            artifact_relative_path="strict_failure",
            validation_only=False,
            control_k=0.05,
            locked_validation_selection={
                "K": 0.05,
                "stable": True,
                "divergence_detected": False,
                "selection_score": 0.1,
            },
            **common,
        )
    except RuntimeError as exc:
        strict_failure_observed = (
            "failed on controller test" in str(exc)
        )

    result = {
        "passed": bool(
            metadata["loaded_from_cache"]
            and rejected["rejected"]
            and final["stable"]
            and strict_failure_observed
            and final["model_identity_hash"]
            == cached_model.model_identity_hash()
        ),
        "cached_model_identity": cached_model.model_identity_hash(),
        "cached_model_loaded": metadata["loaded_from_cache"],
        "bo_invocations_during_control": 0,
        "unstable_candidate_rejected": rejected["rejected"],
        "final_controller_stable": final["stable"],
        "strict_final_divergence_is_fatal": strict_failure_observed,
    }
    if not result["passed"]:
        raise RuntimeError("Reduced chaotic controller smoke failed.")
    return result


def package_validator_smoke(output_dir: Path) -> dict:
    root = output_dir / "tiny_package"
    for directory in REQUIRED_PACKAGE_DIRECTORIES:
        (root / directory).mkdir(parents=True)
    (root / "00_manifest" / "run_manifest.json").write_text(
        json.dumps(
            {
                "git": {"commit": "smoke-commit"},
                "package_references": {
                    "predictions": "01_prediction_all_regimes"
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "00_manifest" / "stage_timings.csv").write_text(
        "stage,seconds\ntotal_runtime,0.1\n", encoding="utf-8"
    )
    for regime in (
        "periodic_spiking",
        "periodic_bursting",
        "chaotic_bursting",
    ):
        prediction = root / "01_prediction_all_regimes" / regime
        prediction.mkdir()
        (prediction / "selected_model.json").write_text(
            json.dumps({"locked": True}), encoding="utf-8"
        )
        (prediction / "heldout_test_metrics.json").write_text(
            json.dumps({"quality_gate": {"passed": True}}),
            encoding="utf-8",
        )
        (prediction / "model_bundle.npz").write_bytes(b"smoke-bundle")
        bo = root / "02_bo_optimization" / regime
        bo.mkdir()
        (bo / "best_params.json").write_text(
            json.dumps({"validation_score": 0.1}), encoding="utf-8"
        )
        (bo / "validation_windows.json").write_text(
            json.dumps({"windows": [1, 2, 3]}), encoding="utf-8"
        )
        (bo / "optimizer_validation_summary.csv").write_text(
            "optimizer,score\ndummy,0.1\n", encoding="utf-8"
        )
    for directory, controller in (
        ("03_linear_feedback", "linear_feedback"),
        ("04_finite_time", "finite_time"),
        ("05_pyragas", "pyragas"),
    ):
        (root / directory / "control_summary.json").write_text(
            json.dumps(
                {
                    "controller": controller,
                    "stable": True,
                    "model_identity_hash": "smoke-shared-model",
                    "control_model_source": "validation_selected",
                    "reference_type": "empirical_quiet_state_reference",
                }
            ),
            encoding="utf-8",
        )
    Image.new("RGB", (4, 4), "white").save(
        root
        / "01_prediction_all_regimes"
        / "periodic_spiking"
        / "results_all_states.png"
    )
    report = validate_final_package(
        root,
        expected_commit="smoke-commit",
        clean_repository_at_start=True,
        quality_gates_passed=True,
    )
    if not report["valid"]:
        raise RuntimeError(
            "Tiny package validator smoke failed: "
            + "; ".join(report["errors"])
        )
    return {
        "passed": True,
        "relative_paths": report["no_unexpected_absolute_paths"],
        "duplicate_prediction_figures": len(
            report["duplicate_prediction_violations"]
        ),
        "same_control_model_identity": report[
            "same_control_model_identity"
        ],
    }


def run_all_smokes(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "periodic_selection": periodic_selection_smoke(
            output_dir / "periodic_selection"
        ),
        "controller_cache_and_rejection": (
            controller_cache_and_rejection_smoke(
                output_dir / "controller_cache"
            )
        ),
        "package_validator": package_validator_smoke(
            output_dir / "package_validator"
        ),
    }
    result["passed"] = all(section["passed"] for section in result.values())
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_all_smokes(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
