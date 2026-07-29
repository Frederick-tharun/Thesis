#!/usr/bin/env python3
"""
Evaluate the locked Chapter 1 ESN/controller configuration across one reservoir seed.

Run this file once per seed. It deliberately does NOT repeat:
- Bayesian optimization,
- controller K sweeps,
- finite-time exponent search,
- Pyragas delay search.

It loads the locked configuration from a validated FINAL_THESIS_RUN folder,
changes only the ESN reservoir seed, retrains the readout, evaluates the held-out
chaotic-bursting prediction, and applies the same three locked controllers.

Designed for the repository:
    Frederick-tharun/Thesis
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

import config
from control_experiment import run_control_experiment
from data_loader import DataLoader
from main import make_model, normalize_from_train, set_hr_mode, split_train_test
from optimize_model import nrmse, resolve_washout, rmse


CONTROLLER_FILES = {
    "linear_feedback": Path("03_linear_feedback/control_summary.json"),
    "finite_time": Path("04_finite_time/control_summary.json"),
    "pyragas": Path("05_pyragas/control_summary.json"),
}

CHAOTIC_PARAMS_FILE = Path(
    "02_bo_optimization/chaotic_bursting/best_params.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one locked-configuration multiseed evaluation for the "
            "chaotic-bursting ESN and all three controllers."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Reservoir seed for this independent ESN realization.",
    )
    parser.add_argument(
        "--reference-run",
        type=Path,
        default=Path("FINAL_THESIS_RUN"),
        help=(
            "Validated FINAL_THESIS_RUN directory containing the locked "
            "best_params.json and controller control_summary.json files."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("MULTISEED_EVAL"),
        help="Shared output root. This script writes seed_<seed>/ below it.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Generate controller plots. For the first five-seed pass, leave "
            "this off; rerun only the representative seed with --plots."
        ),
    )
    parser.add_argument(
        "--keep-rollouts",
        action="store_true",
        help=(
            "Keep the large per-controller rollout.csv files. By default they "
            "are removed after metrics are safely written."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate an existing seed_<seed> output directory.",
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in: {path}")
    return value


def git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(repo), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def load_reference_configuration(
    reference_run: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    params = read_json(reference_run / CHAOTIC_PARAMS_FILE)
    controller_summaries = {
        name: read_json(reference_run / relative_path)
        for name, relative_path in CONTROLLER_FILES.items()
    }

    required_esn = {
        "N_res",
        "p",
        "spectral_radius",
        "leaky_coefficient",
        "input_scaling",
        "regularization",
        "washout",
    }
    missing = sorted(required_esn - set(params))
    if missing:
        raise ValueError(
            "Locked chaotic ESN parameter file is missing: " + ", ".join(missing)
        )

    for controller, summary in controller_summaries.items():
        if summary.get("controller") != controller:
            raise ValueError(
                f"Controller summary mismatch for {controller}: "
                f"{summary.get('controller')!r}"
            )
        if summary.get("best_k") is None:
            raise ValueError(f"Locked best_k is missing for {controller}.")

    return params, controller_summaries


def load_chaotic_data() -> tuple[
    DataLoader,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    config.DATASET_MODE = "hr"
    set_hr_mode("chaotic_bursting")

    loader = DataLoader(csv_path=config.DATA_PATH)
    loader.load()
    loader.preprocess()
    loader.detect_spikes()

    series = np.asarray(loader.data_raw, dtype=float)[:, :3]
    times = np.asarray(loader.time, dtype=float)
    train, test = split_train_test(series)

    if len(train) + len(test) != len(series):
        raise AssertionError("The train/test split does not cover the full series.")
    if train.shape[1] != 3 or test.shape[1] != 3:
        raise AssertionError("Full-state HR input must contain x, y, and z.")

    return loader, series, times, train, test


def locked_controller_parameters(
    controller: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "control_k": float(summary["best_k"]),
        "finite_s": float(getattr(config, "CONTROL_FINITE_S", 0.8)),
        "pyragas_delay": int(getattr(config, "PYRAGAS_DELAY", 20)),
        "pyragas_sign": int(getattr(config, "PYRAGAS_SIGN", -1)),
        "pyragas_history_signal": str(
            getattr(config, "PYRAGAS_HISTORY_SIGNAL", "raw_readout")
        ),
    }

    if controller == "finite_time":
        parameters["finite_s"] = float(summary["finite_s"])

    if controller == "pyragas":
        parameters["pyragas_delay"] = int(summary["pyragas_delay"])
        parameters["pyragas_sign"] = int(summary["pyragas_sign"])
        parameters["pyragas_history_signal"] = str(
            summary["pyragas_history_signal"]
        )

    return parameters


def compact_controller_result(
    controller: str,
    summary: dict[str, Any],
    locked: dict[str, Any],
    relative_summary_path: str,
) -> dict[str, Any]:
    test_metrics = dict(summary.get("test_metrics") or {})
    stable = bool(summary.get("stable", test_metrics.get("stable", False)))
    divergence = bool(
        summary.get(
            "divergence_detected",
            test_metrics.get("divergence_detected", False),
        )
    )

    quality_pass = test_metrics.get("pyragas_quality_pass")
    success = stable and not divergence
    if controller == "pyragas" and quality_pass is not None:
        success = success and bool(quality_pass)

    return {
        "status": "ok",
        "controller": controller,
        "locked_parameters": locked,
        "control_summary_file": relative_summary_path,
        "stable": stable,
        "divergence_detected": divergence,
        "divergence_reason": summary.get(
            "divergence_reason",
            test_metrics.get("divergence_reason"),
        ),
        "success": bool(success),
        "final_test_metric_name": summary.get("final_test_metric_name"),
        "final_test_metric_value": summary.get("final_test_metric_value"),
        "selection_runtime_seconds": summary.get(
            "selection_runtime_seconds"
        ),
        "final_test_runtime_seconds": summary.get(
            "final_test_runtime_seconds"
        ),
        "test_metrics": test_metrics,
    }


def remove_large_rollout(controller_dir: Path) -> None:
    rollout = controller_dir / "rollout.csv"
    if rollout.is_file():
        rollout.unlink()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    repo = Path(__file__).resolve().parent
    reference_run = args.reference_run.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    seed_dir = output_root / f"seed_{args.seed}"

    if not reference_run.is_dir():
        raise FileNotFoundError(
            f"Reference run directory does not exist: {reference_run}"
        )

    if seed_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {seed_dir}\n"
                "Use --overwrite only when you intentionally want to replace it."
            )
        shutil.rmtree(seed_dir)

    seed_dir.mkdir(parents=True)
    controller_root = seed_dir / "controllers"
    controller_root.mkdir()

    locked_params, reference_controllers = load_reference_configuration(
        reference_run
    )
    reference_seed = int(locked_params.get("reservoir_seed", 42))

    # Change only the reservoir seed. All selected hyperparameters stay locked.
    params = dict(locked_params)
    params["reference_reservoir_seed"] = reference_seed
    params["reservoir_seed"] = int(args.seed)
    params["evaluation_seeds"] = [int(args.seed)]
    params["evaluation_seed_count"] = 1
    params["multiseed_evaluation"] = True
    params["optimization_reused"] = True
    params["parameter_source_file"] = os.fspath(
        reference_run / CHAOTIC_PARAMS_FILE
    )

    write_json(seed_dir / "locked_esn_parameters.json", params)

    data_started = time.perf_counter()
    loader, series, times, train, test = load_chaotic_data()
    data_seconds = time.perf_counter() - data_started

    train_norm, test_norm, mean, std = normalize_from_train(train, test)

    model_started = time.perf_counter()
    esn = make_model(params, input_size=3)
    washout = resolve_washout(params.get("washout", 200), len(train_norm))
    esn.train(train_norm, washout=washout)
    model_seconds = time.perf_counter() - model_started

    prediction_started = time.perf_counter()
    evaluation_input = np.vstack([train_norm, test_norm])
    pred_result = esn.predict(
        evaluation_input,
        n_warmup=len(train_norm) - 1,
    )
    pred_norm = pred_result[0] if isinstance(pred_result, tuple) else pred_result
    pred_norm = np.asarray(pred_norm, dtype=float)[: len(test)]
    pred_raw = pred_norm * std + mean
    prediction_seconds = time.perf_counter() - prediction_started

    prediction_metrics = {
        "seed": int(args.seed),
        "reference_seed": reference_seed,
        "regime": "chaotic_bursting",
        "training_steps": int(len(train)),
        "heldout_steps": int(len(test)),
        "washout": int(washout),
        "rmse_recursive_x": rmse(pred_raw[:, 0], test[:, 0]),
        "nrmse_recursive_x": nrmse(pred_norm[:, 0], test_norm[:, 0]),
        "rmse_recursive_all_states": rmse(pred_raw, test),
        "nrmse_recursive_all_states": nrmse(pred_norm, test_norm),
        "stable": bool(np.all(np.isfinite(pred_norm))),
        "model_identity_hash": esn.model_identity_hash(),
        "runtime_seconds": prediction_seconds,
    }
    write_json(seed_dir / "prediction_metrics.json", prediction_metrics)
    np.savez_compressed(
        seed_dir / "prediction_rollout.npz",
        seed=np.asarray([args.seed], dtype=int),
        time=times[len(train) : len(train) + len(test)],
        truth=test,
        prediction=pred_raw,
        prediction_normalized=pred_norm,
        train_mean=mean,
        train_std=std,
    )

    provenance = {
        "source_regime": "chaotic_bursting",
        "selected_optimizer": reference_controllers[
            "linear_feedback"
        ].get("selected_optimizer", "unknown"),
        "best_parameter_file": os.fspath(
            reference_run / CHAOTIC_PARAMS_FILE
        ),
        "reference_final_run": os.fspath(reference_run),
        "reference_model_seed": reference_seed,
        "model_seed": int(args.seed),
        "model_identity_hash": esn.model_identity_hash(),
        "model_loaded_from_cache": False,
        "controller_parameters_locked": True,
        "controller_parameter_search_repeated": False,
        "multiseed_evaluation": True,
        "git_commit": git_commit(repo),
        "control_model_source": "locked_configuration_multiseed_evaluation",
    }

    controller_results: dict[str, dict[str, Any]] = {}

    for controller in (
        "linear_feedback",
        "finite_time",
        "pyragas",
    ):
        controller_started = time.perf_counter()
        controller_dir = controller_root / controller
        controller_dir.mkdir(parents=True, exist_ok=True)

        reference_summary = reference_controllers[controller]
        locked = locked_controller_parameters(controller, reference_summary)

        write_json(
            controller_dir / "locked_controller_parameters.json",
            {
                "controller": controller,
                "reference_model_seed": reference_seed,
                "evaluation_seed": int(args.seed),
                **locked,
            },
        )

        try:
            # control_k is fixed, auto_control_k=False. Therefore there is no
            # K sweep or parameter search. The single locked K is checked on
            # this seed's controller-validation window and then evaluated on
            # its held-out controller-test window.
            summary = run_control_experiment(
                esn=esn,
                loader=loader,
                config=config,
                train=train,
                test=test,
                train_norm=train_norm,
                test_norm=test_norm,
                mean=mean,
                std=std,
                times=times,
                base_output_dir=os.fspath(seed_dir),
                hr_mode="chaotic_bursting",
                best_params=params,
                optimizer_name=str(
                    reference_summary.get("selected_optimizer", "locked")
                ),
                control_k=locked["control_k"],
                control_start_frac=float(config.CONTROL_START_FRAC),
                control_target_mode=(
                    "rest_state_from_quiet_training_data"
                ),
                auto_control_k=False,
                controller=controller,
                finite_s=locked["finite_s"],
                pyragas_delay=locked["pyragas_delay"],
                pyragas_sign=locked["pyragas_sign"],
                pyragas_history_signal=locked[
                    "pyragas_history_signal"
                ],
                validation_only=False,
                generate_plots=bool(args.plots),
                uncontrolled_prediction_norm=pred_norm,
                model_provenance=provenance,
                controller_output_dir=os.fspath(controller_dir),
                artifact_relative_path=(
                    f"seed_{args.seed}/controllers/{controller}"
                ),
                append_global_comparison=False,
            )

            relative_summary = (
                Path("controllers")
                / controller
                / "control_summary.json"
            ).as_posix()
            compact = compact_controller_result(
                controller,
                summary,
                locked,
                relative_summary,
            )
            compact["total_runtime_seconds"] = (
                time.perf_counter() - controller_started
            )
            controller_results[controller] = compact

            if not args.keep_rollouts:
                remove_large_rollout(controller_dir)

        except Exception as exc:
            failure = {
                "status": "failed",
                "controller": controller,
                "locked_parameters": locked,
                "stable": False,
                "divergence_detected": True,
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "total_runtime_seconds": (
                    time.perf_counter() - controller_started
                ),
            }
            controller_results[controller] = failure
            write_json(controller_dir / "failure.json", failure)

    total_seconds = time.perf_counter() - started

    seed_summary = {
        "schema_version": "chapter1_multiseed_evaluation_v1",
        "seed": int(args.seed),
        "reference_seed": reference_seed,
        "reference_run": os.fspath(reference_run),
        "regime": "chaotic_bursting",
        "only_reservoir_seed_changed": True,
        "bayesian_optimization_repeated": False,
        "controller_parameter_search_repeated": False,
        "plots_generated": bool(args.plots),
        "large_rollouts_kept": bool(args.keep_rollouts),
        "prediction": prediction_metrics,
        "controllers": controller_results,
        "runtime": {
            "data_loading_seconds": data_seconds,
            "model_training_seconds": model_seconds,
            "prediction_seconds": prediction_seconds,
            "total_seconds": total_seconds,
        },
        "git_commit": git_commit(repo),
    }
    write_json(seed_dir / "seed_summary.json", seed_summary)

    print("=" * 72)
    print("MULTISEED EVALUATION COMPLETE")
    print("=" * 72)
    print(f"Seed          : {args.seed}")
    print(f"Output        : {seed_dir}")
    print(
        "Prediction    : "
        f"NRMSE x={prediction_metrics['nrmse_recursive_x']:.6g}"
    )
    for controller, result in controller_results.items():
        print(
            f"{controller:16s}: "
            f"status={result.get('status')} "
            f"success={result.get('success')} "
            f"stable={result.get('stable')}"
        )
    print(f"Runtime       : {total_seconds:.2f} seconds")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
