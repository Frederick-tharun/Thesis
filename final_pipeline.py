from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import config
from control_experiment import run_control_experiment
from data_loader import DataLoader
from final_package import assert_valid_final_package
from main import (
    HR_REGIMES,
    make_model,
    normalize_from_train,
    plot_all_states,
    plot_results,
    run_all_optimizers,
    set_hr_mode,
    split_train_test,
)
from model import EchoStateNetwork
from optimize_model import (
    _validation_window_metrics,
    nrmse,
    prediction_validation_spec,
    resolve_washout,
    rmse,
)


PACKAGE_DIRECTORIES = (
    "00_manifest",
    "01_prediction_all_regimes",
    "02_bo_optimization",
    "03_linear_feedback",
    "04_finite_time",
    "05_pyragas",
    "06_comparison_tables",
    "07_report_figures",
    "08_logs",
)
FINITE_TIME_EXPONENTS = (0.3, 0.5, 0.7, 0.8, 0.9)
PYRAGAS_DELAYS = (20, 80, 320, 640, 1280, 1600, 2400, 3200)
PREDICTION_FIGURE_NAMES = (
    "results_all_states.png",
    "results_full_zoom.png",
    "results_zoom_comparison.png",
    "spike_event_comparison.png",
)


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty required CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_json_safe(value), sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else _json_safe(value)
                    )
                    for key, value in row.items()
                }
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def assert_clean_repository(repo: Path) -> None:
    """Refuse final execution when reproducibility-relevant source is dirty."""
    for args, label in (
        (("diff", "--quiet"), "tracked working tree"),
        (("diff", "--cached", "--quiet"), "staging area"),
    ):
        result = subprocess.run(
            ["git", "-C", os.fspath(repo), *args],
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Final pipeline requires a clean {label}.")
    relevant = _git(
        repo,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "*.py",
        "*.slurm",
        "README.md",
        "CHAPTER1_METHOD_FIXES.md",
        "requirements.txt",
        "requirements*.txt",
    )
    if relevant:
        raise RuntimeError(
            "Final pipeline found relevant untracked source files: "
            + ", ".join(relevant.splitlines())
        )


def _source_hashes(repo: Path) -> dict[str, str]:
    paths = sorted(
        {
            *repo.glob("*.py"),
            *repo.glob("*.slurm"),
            repo / "README.md",
            repo / "CHAPTER1_METHOD_FIXES.md",
        }
    )
    return {
        path.relative_to(repo).as_posix(): _sha256(path)
        for path in paths
        if path.is_file()
    }


def _configuration_hash(repo: Path) -> str:
    return _sha256(repo / "config.py")


def _load_regime(regime: str):
    config.DATASET_MODE = "hr"
    set_hr_mode(regime)
    loader = DataLoader(csv_path=config.DATA_PATH)
    loader.load()
    loader.preprocess()
    loader.detect_spikes()
    series = np.asarray(loader.data_raw, dtype=float)[:, :3]
    times = np.asarray(loader.time, dtype=float)
    train, test = split_train_test(series)
    if len(test) != len(series) - len(train):
        raise AssertionError("Prediction split is inconsistent.")
    return loader, series, times, train, test


def _heldout_metrics(
    pred_norm: np.ndarray,
    test_norm: np.ndarray,
    pred_raw: np.ndarray,
    test_raw: np.ndarray,
    threshold_norm: float,
) -> dict:
    dynamics = _validation_window_metrics(
        pred_norm,
        test_norm,
        spike_threshold_norm=threshold_norm,
    )
    return {
        "metric_segment": "untouched_heldout_test",
        "selection_locked_before_evaluation": True,
        "test_data_used_for_selection": False,
        "heldout_steps": int(len(test_raw)),
        "rmse_recursive_x": rmse(pred_raw[:, 0], test_raw[:, 0]),
        "nrmse_recursive_x": nrmse(pred_norm[:, 0], test_norm[:, 0]),
        "rmse_recursive_all_states": rmse(pred_raw, test_raw),
        "nrmse_recursive_all_states": nrmse(pred_norm, test_norm),
        "spike_count_true": dynamics["spike_count_true"],
        "spike_count_pred": dynamics["spike_count_pred"],
        "spike_frequency_true": dynamics["spike_frequency_true"],
        "spike_frequency_pred": dynamics["spike_frequency_pred"],
        "spike_frequency_rel_error": dynamics[
            "spike_frequency_rel_error"
        ],
        "spike_frequency_units": dynamics["spike_frequency_units"],
        "mean_inter_spike_interval_true": dynamics["mean_isi_true"],
        "mean_inter_spike_interval_pred": dynamics["mean_isi_pred"],
        "inter_spike_interval_rel_error": dynamics["isi_rel_error"],
        "inter_spike_interval_units": dynamics[
            "inter_spike_interval_units"
        ],
        "divergence_detected": not bool(dynamics["stable"]),
    }


def _quality_gate(regime: str, metrics: dict) -> dict:
    if regime != "periodic_spiking":
        return {
            "required": False,
            "passed": True,
            "reason": "periodic-spiking-specific gate not applicable",
        }
    limits = {
        "max_test_nrmse_x": float(
            config.PERIODIC_SPIKING_MAX_TEST_NRMSE_X
        ),
        "max_spike_frequency_rel_error": float(
            config.PERIODIC_SPIKING_MAX_SPIKE_FREQUENCY_REL_ERROR
        ),
    }
    checks = {
        "test_nrmse_x": (
            float(metrics["nrmse_recursive_x"])
            <= limits["max_test_nrmse_x"]
        ),
        "spike_frequency_rel_error": (
            float(metrics["spike_frequency_rel_error"])
            <= limits["max_spike_frequency_rel_error"]
        ),
        "nondivergent": not bool(metrics["divergence_detected"]),
    }
    return {
        "required": True,
        "passed": all(checks.values()),
        "limits": limits,
        "checks": checks,
    }


def _plot_prediction(
    prediction_dir: Path,
    *,
    optimizer_name: str,
    series: np.ndarray,
    times: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    pred_raw: np.ndarray,
) -> None:
    config.OUTPUT_DIR = os.fspath(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    pred_times = times[len(train) : len(train) + len(pred_raw)]
    split = {
        "neuron_name": "hr_x",
        "neuron_index": 0,
        "full_time": times,
        "full_signal": series[:, 0],
        "train_time": times[: len(train)],
        "val_time": np.array([]),
        "test_time": times[len(train) :],
        "train_signal": train[:, 0],
        "val_signal": np.array([]),
        "test_signal": test[:, 0],
        "y_test": test[:, 0].reshape(-1, 1),
        "t_test_y": pred_times,
    }
    plot_results(split, pred_raw[:, 0], tag=f"ESN ({optimizer_name})")
    plot_all_states(
        t=pred_times,
        truth=test,
        pred=pred_raw,
        tag=f"ESN ({optimizer_name})",
    )


def _train_and_evaluate_regime(
    root: Path,
    repo: Path,
    regime: str,
    git_commit: str,
    config_hash: str,
    timings: list[dict],
    bo_invocations: list[dict],
) -> dict:
    data_started = time.perf_counter()
    loader, series, times, train, test = _load_regime(regime)
    timings.append(
        {
            "stage": f"data_generation_{regime}",
            "seconds": time.perf_counter() - data_started,
        }
    )

    bo_dir = root / "02_bo_optimization" / regime
    config.OUTPUT_DIR = os.fspath(bo_dir)
    bo_started = time.perf_counter()
    optimizer_name, best_params, optimizer_summary, optimizer_history = (
        run_all_optimizers(
            loader,
            0,
            selection_series=train,
            heldout_length=len(test),
        )
    )
    bo_seconds = time.perf_counter() - bo_started
    timings.append({"stage": f"bo_{regime}", "seconds": bo_seconds})
    bo_invocations.append(
        {
            "regime": regime,
            "overall_bo_stage_invocations": 1,
            "optimizers_compared": list(config.OPTIMIZERS_TO_COMPARE),
            "heldout_array_passed_to_selection": False,
            "training_selection_samples": len(train),
            "heldout_samples": len(test),
        }
    )

    validation_spec = prediction_validation_spec(
        train,
        series_is_training_portion=True,
        heldout_length=len(test),
    )
    _write_json(
        bo_dir / "validation_windows.json",
        {
            key: value
            for key, value in validation_spec.items()
            if key not in {"train", "validation"}
        },
    )
    _write_csv(
        bo_dir / "optimizer_validation_summary.csv",
        [dict(row) for row in optimizer_summary],
    )
    _write_json(bo_dir / "best_params.json", best_params)

    prediction_dir = root / "01_prediction_all_regimes" / regime
    prediction_dir.mkdir(parents=True, exist_ok=True)
    train_norm, _, mean, std = normalize_from_train(
        train, np.empty((0, train.shape[1]))
    )
    model_started = time.perf_counter()
    esn = make_model(best_params, input_size=train.shape[1])
    washout = resolve_washout(best_params.get("washout", 200), len(train_norm))
    esn.train(train_norm, washout=washout)
    model_seconds = time.perf_counter() - model_started
    timings.append(
        {
            "stage": f"final_model_training_{regime}",
            "seconds": model_seconds,
        }
    )

    model_identity = esn.model_identity_hash()
    bundle_relative = (
        Path("01_prediction_all_regimes") / regime / "model_bundle.npz"
    )
    bundle_metadata = esn.save_bundle(
        root / bundle_relative,
        metadata={
            "source_regime": regime,
            "selected_optimizer": optimizer_name,
            "best_parameter_file": (
                Path("02_bo_optimization") / regime / "best_params.json"
            ).as_posix(),
            "git_commit": git_commit,
            "configuration_hash": config_hash,
            "reservoir_seed": int(esn.seed),
            "normalization": "external_training_statistics",
        },
        external_mean=mean,
        external_std=std,
    )
    if bundle_metadata["model_identity_hash"] != model_identity:
        raise AssertionError("Model identity changed during serialization.")

    selected_model = {
        "schema_version": "chapter1_selected_model_v1",
        "selection_status": "locked_before_heldout_evaluation",
        "source_regime": regime,
        "selected_optimizer": optimizer_name,
        "validation_score": best_params["validation_score"],
        "validation_only_selection": True,
        "test_data_used_for_selection": False,
        "best_parameter_file": (
            Path("02_bo_optimization") / regime / "best_params.json"
        ).as_posix(),
        "validation_windows_file": (
            Path("02_bo_optimization") / regime / "validation_windows.json"
        ).as_posix(),
        "model_bundle": bundle_relative.as_posix(),
        "model_identity_hash": model_identity,
        "model_configuration_hash": config_hash,
        "model_seed": int(esn.seed),
        "git_commit": git_commit,
        "training_start": 0,
        "training_end": len(train),
        "heldout_test_start": len(train),
        "heldout_test_end": len(series),
    }
    selected_model_path = prediction_dir / "selected_model.json"
    _write_json(selected_model_path, selected_model)
    if not selected_model_path.is_file():
        raise AssertionError("Selected model was not locked before held-out test.")

    prediction_started = time.perf_counter()
    test_norm = (test - mean) / std
    evaluation_input = np.vstack([train_norm, test_norm])
    pred_norm, _ = esn.predict(
        evaluation_input,
        n_warmup=len(train_norm) - 1,
    )
    pred_norm = np.asarray(pred_norm, dtype=float)[: len(test)]
    pred_raw = pred_norm * std + mean
    prediction_seconds = time.perf_counter() - prediction_started
    timings.append(
        {
            "stage": f"prediction_rollout_{regime}",
            "seconds": prediction_seconds,
        }
    )

    threshold_norm = (float(config.SPIKE_THRESHOLD) - mean[0, 0]) / std[0, 0]
    heldout = _heldout_metrics(
        pred_norm,
        test_norm,
        pred_raw,
        test,
        threshold_norm,
    )
    heldout.update(
        {
            "source_regime": regime,
            "selected_optimizer": optimizer_name,
            "model_identity_hash": model_identity,
            "heldout_test_start": len(train),
            "heldout_test_end": len(series),
        }
    )
    gate = _quality_gate(regime, heldout)
    heldout["quality_gate"] = gate
    _write_json(prediction_dir / "heldout_test_metrics.json", heldout)
    _write_json(prediction_dir / "metrics.json", heldout)
    _plot_prediction(
        prediction_dir,
        optimizer_name=optimizer_name,
        series=series,
        times=times,
        train=train,
        test=test,
        pred_raw=pred_raw,
    )
    if not gate["passed"]:
        raise RuntimeError(
            "Periodic-spiking held-out quality gate failed after selection "
            "was locked; the pipeline is scientifically failed."
        )

    return {
        "regime": regime,
        "loader": loader,
        "series": series,
        "times": times,
        "train": train,
        "test": test,
        "train_norm": train_norm,
        "test_norm": test_norm,
        "mean": mean,
        "std": std,
        "pred_norm": pred_norm,
        "pred_raw": pred_raw,
        "model_bundle_path": root / bundle_relative,
        "model_bundle_relative": bundle_relative.as_posix(),
        "model_identity_hash": model_identity,
        "selected_optimizer": optimizer_name,
        "best_params": best_params,
        "heldout_metrics": heldout,
        "quality_gate": gate,
    }


def _control_kwargs(
    *,
    root: Path,
    artifact: dict,
    model: EchoStateNetwork,
    provenance: dict,
    output_dir: Path,
    artifact_relative_path: str,
    controller: str,
    generate_plots: bool,
    validation_only: bool,
    **specific,
) -> dict:
    return {
        "esn": model,
        "loader": artifact["loader"],
        "config": config,
        "train": artifact["train"],
        "test": artifact["test"],
        "train_norm": artifact["train_norm"],
        "test_norm": artifact["test_norm"],
        "mean": artifact["mean"],
        "std": artifact["std"],
        "times": artifact["times"],
        "base_output_dir": os.fspath(root),
        "hr_mode": "chaotic_bursting",
        "best_params": artifact["best_params"],
        "optimizer_name": artifact["selected_optimizer"],
        "control_start_frac": float(config.CONTROL_START_FRAC),
        "control_target_mode": "rest_state_from_quiet_training_data",
        "controller": controller,
        "validation_only": validation_only,
        "generate_plots": generate_plots,
        "uncontrolled_prediction_norm": artifact["pred_norm"],
        "model_provenance": provenance,
        "controller_output_dir": os.fspath(output_dir),
        "artifact_relative_path": artifact_relative_path,
        "append_global_comparison": False,
        **specific,
    }


def _stable_candidate(summary: dict) -> bool:
    validation = summary.get("validation_metrics") or {}
    score = summary.get("selection_metric_value")
    try:
        score = float(score)
    except (TypeError, ValueError):
        return False
    return bool(
        summary.get("stable")
        and validation.get("stable")
        and not validation.get("divergence_detected")
        and np.isfinite(score)
    )


def _select_candidate(
    candidates: list[dict],
    *,
    pyragas: bool = False,
) -> dict:
    stable = [candidate for candidate in candidates if _stable_candidate(candidate)]
    if not stable:
        raise RuntimeError("No stable validation-only controller candidate exists.")
    if pyragas:
        return min(
            stable,
            key=lambda summary: (
                0
                if (summary.get("validation_metrics") or {}).get(
                    "pyragas_quality_pass"
                )
                is True
                else 1,
                float(summary["selection_metric_value"]),
            ),
        )
    return min(stable, key=lambda summary: float(summary["selection_metric_value"]))


def _run_controllers(
    root: Path,
    artifact: dict,
    git_commit: str,
    config_hash: str,
    timings: list[dict],
    bo_invocation_count_before: int,
    bo_invocations: list[dict],
) -> dict[str, dict]:
    model, bundle_metadata = EchoStateNetwork.load_bundle(
        artifact["model_bundle_path"]
    )
    if model.model_identity_hash() != artifact["model_identity_hash"]:
        raise AssertionError("Cached chaotic model identity is inconsistent.")

    provenance = {
        "source_regime": "chaotic_bursting",
        "selected_optimizer": artifact["selected_optimizer"],
        "best_parameter_file": (
            "02_bo_optimization/chaotic_bursting/best_params.json"
        ),
        "git_commit": git_commit,
        "model_seed": int(model.seed),
        "model_identity_hash": model.model_identity_hash(),
        "model_configuration_hash": config_hash,
        "model_loaded_from_cache": True,
        "model_bundle_path": artifact["model_bundle_relative"],
        "canonical_prediction_result": (
            "01_prediction_all_regimes/chaotic_bursting/"
            "heldout_test_metrics.json"
        ),
    }

    linear = run_control_experiment(
        **_control_kwargs(
            root=root,
            artifact=artifact,
            model=model,
            provenance=provenance,
            output_dir=root / "03_linear_feedback",
            artifact_relative_path="03_linear_feedback",
            controller="linear_feedback",
            generate_plots=True,
            validation_only=False,
            auto_control_k=True,
            k_min=0.01,
            k_max=2.0,
            k_num=25,
            k_refine_num=20,
        )
    )
    timings.extend(
        [
            {
                "stage": "linear_selection",
                "seconds": linear["selection_runtime_seconds"],
            },
            {
                "stage": "linear_final_test",
                "seconds": linear["final_test_runtime_seconds"],
            },
        ]
    )

    finite_candidates = []
    finite_selection_seconds = 0.0
    for exponent in FINITE_TIME_EXPONENTS:
        label = f"s_{str(exponent).replace('.', 'p')}"
        summary = run_control_experiment(
            **_control_kwargs(
                root=root,
                artifact=artifact,
                model=model,
                provenance=provenance,
                output_dir=(
                    root / "04_finite_time" / "candidates" / label
                ),
                artifact_relative_path=(
                    f"04_finite_time/candidates/{label}"
                ),
                controller="finite_time",
                generate_plots=False,
                validation_only=True,
                finite_s=exponent,
                auto_control_k=True,
                k_min=0.01,
                k_max=1.5,
                k_num=25,
                k_refine_num=20,
            )
        )
        finite_selection_seconds += float(
            summary["selection_runtime_seconds"]
        )
        finite_candidates.append(summary)
    finite_selected = _select_candidate(finite_candidates)
    finite_validation = finite_selected["validation_metrics"]
    finite_final = run_control_experiment(
        **_control_kwargs(
            root=root,
            artifact=artifact,
            model=model,
            provenance=provenance,
            output_dir=root / "04_finite_time",
            artifact_relative_path="04_finite_time",
            controller="finite_time",
            generate_plots=True,
            validation_only=False,
            finite_s=float(finite_selected["finite_s"]),
            control_k=float(finite_selected["best_k"]),
            locked_validation_selection=finite_validation,
        )
    )
    timings.extend(
        [
            {
                "stage": "finite_time_selection",
                "seconds": finite_selection_seconds,
            },
            {
                "stage": "finite_time_final_test",
                "seconds": finite_final["final_test_runtime_seconds"],
            },
        ]
    )

    if list(config.PYRAGAS_SIGNS) != [-1]:
        raise ValueError("Official final PYRAGAS_SIGNS must be exactly [-1].")
    pyragas_candidates = []
    pyragas_selection_seconds = 0.0
    for delay in PYRAGAS_DELAYS:
        label = f"delay_{delay}_sign_minus1"
        summary = run_control_experiment(
            **_control_kwargs(
                root=root,
                artifact=artifact,
                model=model,
                provenance=provenance,
                output_dir=root / "05_pyragas" / "candidates" / label,
                artifact_relative_path=f"05_pyragas/candidates/{label}",
                controller="pyragas",
                generate_plots=False,
                validation_only=True,
                pyragas_delay=delay,
                pyragas_sign=-1,
                pyragas_history_signal="raw_readout",
                auto_control_k=True,
                k_min=0.02,
                k_max=0.8,
                k_num=13,
                k_refine_num=12,
            )
        )
        pyragas_selection_seconds += float(
            summary["selection_runtime_seconds"]
        )
        pyragas_candidates.append(summary)
    pyragas_selected = _select_candidate(pyragas_candidates, pyragas=True)
    pyragas_validation = pyragas_selected["validation_metrics"]
    pyragas_final = run_control_experiment(
        **_control_kwargs(
            root=root,
            artifact=artifact,
            model=model,
            provenance=provenance,
            output_dir=root / "05_pyragas",
            artifact_relative_path="05_pyragas",
            controller="pyragas",
            generate_plots=True,
            validation_only=False,
            pyragas_delay=int(pyragas_selected["pyragas_delay"]),
            pyragas_sign=-1,
            pyragas_history_signal="raw_readout",
            control_k=float(pyragas_selected["best_k"]),
            locked_validation_selection=pyragas_validation,
        )
    )
    timings.extend(
        [
            {
                "stage": "pyragas_selection",
                "seconds": pyragas_selection_seconds,
            },
            {
                "stage": "pyragas_final_test",
                "seconds": pyragas_final["final_test_runtime_seconds"],
            },
        ]
    )

    if len(bo_invocations) != bo_invocation_count_before:
        raise AssertionError("A controller stage invoked Bayesian optimization.")
    identities = {
        summary.get("model_identity_hash")
        for summary in (linear, finite_final, pyragas_final)
    }
    if identities != {artifact["model_identity_hash"]}:
        raise AssertionError("Controllers did not reuse one cached chaotic ESN.")

    return {
        "linear_feedback": linear,
        "finite_time": finite_final,
        "pyragas": pyragas_final,
    }


def _comparison_tables(
    root: Path,
    predictions: dict[str, dict],
    controllers: dict[str, dict],
) -> None:
    prediction_rows = []
    for regime in HR_REGIMES:
        artifact = predictions[regime]
        metrics = artifact["heldout_metrics"]
        params = artifact["best_params"]
        prediction_rows.append(
            {
                "regime": regime,
                "selected_optimizer": artifact["selected_optimizer"],
                "validation_score": params["validation_score"],
                "heldout_nrmse_x": metrics["nrmse_recursive_x"],
                "heldout_nrmse_all_states": metrics[
                    "nrmse_recursive_all_states"
                ],
                "heldout_spike_frequency_rel_error": metrics[
                    "spike_frequency_rel_error"
                ],
                "training_start": 0,
                "training_end": len(artifact["train"]),
                "heldout_test_start": len(artifact["train"]),
                "heldout_test_end": len(artifact["series"]),
                "model_identity_hash": artifact["model_identity_hash"],
                "quality_gate_passed": artifact["quality_gate"]["passed"],
            }
        )
    control_rows = []
    for name, summary in controllers.items():
        control_rows.append(
            {
                "controller": name,
                "final_test_metric_name": summary[
                    "final_test_metric_name"
                ],
                "final_test_metric_value": summary[
                    "final_test_metric_value"
                ],
                "model_identity_hash": summary["model_identity_hash"],
                "control_model_source": summary["control_model_source"],
                "reference_type": "empirical_quiet_state_reference",
                "regulation_objective": (
                    "regulation toward an empirical quiet-state reference"
                ),
                "controller_validation_start": summary[
                    "controller_validation_start"
                ],
                "controller_validation_end": summary[
                    "controller_validation_end"
                ],
                "controller_test_start": summary["controller_test_start"],
                "controller_test_end": summary["controller_test_end"],
                "stable": summary["stable"],
            }
        )
    comparison_dir = root / "06_comparison_tables"
    _write_csv(comparison_dir / "prediction_metrics_all_regimes.csv", prediction_rows)
    _write_csv(comparison_dir / "controller_final_test_metrics.csv", control_rows)
    _write_json(
        comparison_dir / "chapter1_final_comparison.json",
        {
            "prediction_metrics": prediction_rows,
            "controller_metrics": control_rows,
            "target_statement": (
                "regulation toward an empirical quiet-state reference"
            ),
        },
    )


def _report_figure_links(root: Path) -> None:
    report_dir = root / "07_report_figures"
    sources = []
    for regime in HR_REGIMES:
        prediction_dir = root / "01_prediction_all_regimes" / regime
        available = sorted(prediction_dir.glob("*.png"))
        if available:
            sources.append(
                (
                    available[0],
                    f"prediction_{regime}_{available[0].name}",
                )
            )
    for section, label in (
        ("03_linear_feedback", "linear_feedback"),
        ("04_finite_time", "finite_time"),
        ("05_pyragas", "pyragas"),
    ):
        source = root / section / "controlled_vs_uncontrolled_x.png"
        if source.is_file():
            sources.append((source, f"{label}_{source.name}"))

    links = []
    for source, name in sources:
        destination = report_dir / name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        relative_target = os.path.relpath(source, start=report_dir)
        destination.symlink_to(relative_target)
        links.append(
            {
                "report_link": destination.relative_to(root).as_posix(),
                "canonical_source": source.relative_to(root).as_posix(),
                "storage": "relative_symlink_no_physical_duplicate",
            }
        )
    _write_json(report_dir / "figure_index.json", links)


def _package_manifest(
    root: Path,
    repo: Path,
    *,
    git_commit: str,
    config_hash: str,
    repo_clean_at_start: bool,
    bo_invocations: list[dict],
    run_started_utc: str,
) -> None:
    status = _git(repo, "status", "--porcelain")
    package_references = {
        "prediction_root": "01_prediction_all_regimes",
        "bo_root": "02_bo_optimization",
        "linear_controller": "03_linear_feedback/control_summary.json",
        "finite_time_controller": "04_finite_time/control_summary.json",
        "pyragas_controller": "05_pyragas/control_summary.json",
        "comparisons": "06_comparison_tables",
        "report_figures": "07_report_figures/figure_index.json",
    }
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "scikit-optimize", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "schema_version": "chapter1_run_manifest_v1",
        "timestamps": {
            "run_started_utc": run_started_utc,
            "manifest_created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "git": {
            "commit": git_commit,
            "branch": _git(repo, "branch", "--show-current"),
            "clean_repository_at_start": repo_clean_at_start,
            "status_at_manifest_time": status.splitlines(),
        },
        "reproducibility_files": {
            "slurm_script": {
                "path": "run_final_thesis_pipeline.slurm",
                "sha256": _sha256(
                    repo / "run_final_thesis_pipeline.slurm"
                ),
            },
            "configuration": {
                "path": "config.py",
                "sha256": config_hash,
            },
        },
        "configuration": {
            "config_sha256": config_hash,
            "random_seed": int(config.RANDOM_SEED),
            "bo_reservoir_seed": int(config.BO_RESERVOIR_SEED),
            "bo_evaluation_seeds": list(config.BO_EVALUATION_SEEDS),
            "validation_windows": {
                "count": int(config.PREDICTION_VALIDATION_NUM_WINDOWS),
                "length": int(config.PREDICTION_VALIDATION_WINDOW_LENGTH),
                "starts": config.PREDICTION_VALIDATION_WINDOW_STARTS,
                "aggregation": config.PREDICTION_VALIDATION_AGGREGATION,
            },
            "control_model_source": config.CONTROL_MODEL_SOURCE,
            "pyragas_signs": list(config.PYRAGAS_SIGNS),
        },
        "bo_invocations": bo_invocations,
        "package_references": package_references,
        "source_sha256": _source_hashes(repo),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "packages": packages,
        },
        "commands": [
            (
                "python final_pipeline.py --final-root "
                "<external-evidence>/FINAL_THESIS_RUN_<JOBID> "
                "--clean-repository-at-start"
            )
        ],
        "machine_specific": {
            "repository_root": os.fspath(repo),
            "external_output_root": os.fspath(root),
        },
    }
    _write_json(root / "00_manifest" / "run_manifest.json", manifest)
    _write_json(
        root / "00_manifest" / "source_hashes.json",
        manifest["source_sha256"],
    )
    (root / "00_manifest" / "commands.txt").write_text(
        "\n".join(manifest["commands"]) + "\n",
        encoding="utf-8",
    )


def _aggregate_timings(
    timings: list[dict],
    *,
    total_runtime: float,
    packaging_seconds: float,
) -> list[dict]:
    by_name = {
        row["stage"]: float(row["seconds"])
        for row in timings
    }
    rows = [
        {
            "stage": "data_generation",
            "seconds": sum(
                value
                for key, value in by_name.items()
                if key.startswith("data_generation_")
            ),
        },
        *[
            {
                "stage": f"bo_{regime}",
                "seconds": by_name[f"bo_{regime}"],
            }
            for regime in HR_REGIMES
        ],
        {
            "stage": "final_model_training",
            "seconds": sum(
                value
                for key, value in by_name.items()
                if key.startswith("final_model_training_")
            ),
        },
        {
            "stage": "prediction_rollouts",
            "seconds": sum(
                value
                for key, value in by_name.items()
                if key.startswith("prediction_rollout_")
            ),
        },
    ]
    for stage in (
        "linear_selection",
        "linear_final_test",
        "finite_time_selection",
        "finite_time_final_test",
        "pyragas_selection",
        "pyragas_final_test",
    ):
        rows.append({"stage": stage, "seconds": by_name[stage]})
    rows.extend(
        [
            {"stage": "packaging", "seconds": float(packaging_seconds)},
            {"stage": "total_runtime", "seconds": float(total_runtime)},
        ]
    )
    return rows


def run_final_pipeline(
    *,
    repo: Path,
    final_root: Path,
    clean_repository_at_start: bool,
) -> dict:
    started = time.perf_counter()
    run_started_utc = datetime.now(timezone.utc).isoformat()
    repo = repo.resolve()
    final_root = final_root.resolve()
    if not clean_repository_at_start:
        raise RuntimeError("A successful clean-repository preflight is required.")
    assert_clean_repository(repo)
    if final_root.exists() and any(final_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty final package: {final_root}"
        )
    final_root.mkdir(parents=True, exist_ok=True)
    for name in PACKAGE_DIRECTORIES:
        (final_root / name).mkdir()

    git_commit = _git(repo, "rev-parse", "HEAD")
    config_hash = _configuration_hash(repo)
    timings: list[dict] = []
    bo_invocations: list[dict] = []
    predictions = {}
    for regime in HR_REGIMES:
        predictions[regime] = _train_and_evaluate_regime(
            final_root,
            repo,
            regime,
            git_commit,
            config_hash,
            timings,
            bo_invocations,
        )
    if [row["regime"] for row in bo_invocations] != list(HR_REGIMES):
        raise AssertionError("BO must execute exactly once per HR regime.")

    controller_bo_count = len(bo_invocations)
    controllers = _run_controllers(
        final_root,
        predictions["chaotic_bursting"],
        git_commit,
        config_hash,
        timings,
        controller_bo_count,
        bo_invocations,
    )

    packaging_started = time.perf_counter()
    _comparison_tables(final_root, predictions, controllers)
    _report_figure_links(final_root)
    _write_csv(
        final_root / "00_manifest" / "bo_invocations.csv",
        bo_invocations,
    )
    _package_manifest(
        final_root,
        repo,
        git_commit=git_commit,
        config_hash=config_hash,
        repo_clean_at_start=clean_repository_at_start,
        bo_invocations=bo_invocations,
        run_started_utc=run_started_utc,
    )
    preliminary_rows = _aggregate_timings(
        timings,
        total_runtime=time.perf_counter() - started,
        packaging_seconds=time.perf_counter() - packaging_started,
    )
    _write_csv(
        final_root / "00_manifest" / "stage_timings.csv",
        preliminary_rows,
    )
    quality_passed = all(
        artifact["quality_gate"]["passed"]
        for artifact in predictions.values()
    )
    assert_valid_final_package(
        final_root,
        expected_commit=git_commit,
        clean_repository_at_start=clean_repository_at_start,
        quality_gates_passed=quality_passed,
    )
    stage_rows = _aggregate_timings(
        timings,
        total_runtime=time.perf_counter() - started,
        packaging_seconds=time.perf_counter() - packaging_started,
    )
    _write_csv(
        final_root / "00_manifest" / "stage_timings.csv",
        stage_rows,
    )
    validation = assert_valid_final_package(
        final_root,
        expected_commit=git_commit,
        clean_repository_at_start=clean_repository_at_start,
        quality_gates_passed=quality_passed,
    )
    return {
        "git_commit": git_commit,
        "final_root": os.fspath(final_root),
        "bo_invocations": bo_invocations,
        "controller_model_identity": predictions[
            "chaotic_bursting"
        ]["model_identity_hash"],
        "package_validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the one-pass curated Chapter 1 final pipeline."
    )
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--clean-repository-at-start",
        action="store_true",
        help="Asserted only after the Slurm preflight cleanliness checks pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.clean_repository_at_start:
        raise RuntimeError(
            "Final pipeline requires a successful clean-repository preflight."
        )
    result = run_final_pipeline(
        repo=args.repo,
        final_root=args.final_root,
        clean_repository_at_start=True,
    )
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
