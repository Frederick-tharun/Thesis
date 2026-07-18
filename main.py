from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import numpy as np

import config
from data_loader import DataLoader
from model import EchoStateNetwork
from optimize_model import (
    optimize_hyperparameters,
    nrmse,
    rmse,
    resolve_washout,
    as_2d,
    get_model_series,
)

from plotting import (
    plot_results,
    plot_all_states,
    plot_optimizer_convergence,
    plot_bo_objective_landscape,
    plot_final_comparison_table,
)


try:
    from experiment_report import setup_run_output_folder, save_experiment_summary
except Exception:
    setup_run_output_folder = None
    save_experiment_summary = None


try:
    from control_experiment import run_control_experiment
except Exception as e:
    run_control_experiment = None
    CONTROL_IMPORT_ERROR = e


ROOT_OUTPUT_DIR = getattr(config, "OUTPUT_ROOT", getattr(config, "OUTPUT_DIR", "outputs"))

HR_REGIMES = [
    "periodic_spiking",
    "periodic_bursting",
    "chaotic_bursting",
]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="hr")
    p.add_argument("--neuron", type=int, default=0)

    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Override the configured output root. This is primarily used by "
            "the reproducibility Slurm job so repository outputs are preserved."
        ),
    )

    p.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        choices=["auto", "gp", "dummy", "forest", "gbrt"],
        help="auto = compare all optimizers and choose the best one",
    )

    p.add_argument(
        "--no-opt",
        action="store_true",
        help="Skip optimizer search and use default ESN parameters",
    )
    p.add_argument(
        "--params-file",
        type=str,
        default=None,
        help=(
            "Reuse selected ESN parameters from a JSON file and skip optimization."
        ),
    )

    p.add_argument(
        "--hr-mode",
        type=str,
        default=None,
        choices=HR_REGIMES,
        help="Choose one Hindmarsh-Rose regime",
    )

    p.add_argument(
        "--run-all-regimes",
        action="store_true",
        help="Run periodic_spiking, periodic_bursting, and chaotic_bursting automatically",
    )

    p.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the output folder for the selected regime before running",
    )

    p.add_argument(
        "--control",
        action="store_true",
        help="Run ESN control experiment after prediction",
    )

    p.add_argument(
        "--controller",
        type=str,
        default="linear_feedback",
        choices=["linear_feedback", "finite_time", "pyragas"],
        help="Controller to use for ESN control experiment",
    )

    p.add_argument(
        "--control-k",
        type=float,
        default=None,
        help="Single K value for control. If omitted, K sweep/auto-K is used.",
    )

    p.add_argument(
        "--auto-control-k",
        action="store_true",
        help="Automatically search K values and select the best K.",
    )

    p.add_argument(
        "--k-min",
        type=float,
        default=None,
        help="Minimum K for automatic K search.",
    )

    p.add_argument(
        "--k-max",
        type=float,
        default=None,
        help="Maximum K for automatic K search.",
    )

    p.add_argument(
        "--k-num",
        type=int,
        default=None,
        help="Number of coarse K values for automatic K search.",
    )

    p.add_argument(
        "--k-refine-num",
        type=int,
        default=None,
        help="Number of refined K values around the best coarse K.",
    )

    p.add_argument(
        "--control-start-frac",
        type=float,
        default=getattr(config, "CONTROL_START_FRAC", 0.20),
        help="Fraction of test horizon after which control starts",
    )

    p.add_argument(
        "--control-target-mode",
        type=str,
        default=getattr(config, "CONTROL_TARGET_MODE", "rest_state_from_quiet_training_data"),
        choices=["rest_state_from_quiet_training_data", "rest_state", "zero", "mean"],
        help="Target definition; 'rest_state' is a deprecated alias for the quiet-training-data target.",
    )

    p.add_argument(
        "--finite-s",
        type=float,
        default=getattr(config, "CONTROL_FINITE_S", 0.8),
        help="Exponent s for finite-time controller. Must be between 0 and 1.",
    )

    p.add_argument(
        "--pyragas-delay",
        type=int,
        default=getattr(config, "PYRAGAS_DELAY", 20),
        help="Delay steps for Pyragas time-delay feedback controller.",
    )

    p.add_argument(
        "--pyragas-sign",
        type=int,
        default=getattr(config, "PYRAGAS_SIGN", -1),
        choices=[-1, 1],
        help=(
            "Pyragas feedback sign. Use -1 for u = K * (y_pred - y_delayed), "
            "which works with next_input = y_pred - u_control to pull toward the delayed state. "
            "Use 1 to test the opposite sign."
        ),
    )

    p.add_argument(
        "--pyragas-history-signal",
        type=str,
        default=getattr(config, "PYRAGAS_HISTORY_SIGNAL", "raw_readout"),
        choices=["raw_readout", "corrected_feedback_input"],
        help=(
            "Signal stored in the Pyragas delay history; raw_readout is the "
            "paper-consistent default."
        ),
    )

    p.add_argument(
        "--control-validation-only",
        action="store_true",
        help=(
            "Select controller parameters on controller validation and stop "
            "without evaluating controller test. Used by outer Slurm sweeps."
        ),
    )

    return p.parse_args()


def json_safe(x):
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_safe(v) for v in x]
    if isinstance(x, tuple):
        return [json_safe(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def save_json(obj, filename):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, filename)

    with open(path, "w") as f:
        json.dump(json_safe(obj), f, indent=2)

    print(f"[Save] -> {path}")


def save_csv(rows, filename):
    if not rows:
        return

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, filename)

    keys = sorted(set().union(*(r.keys() for r in rows)))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Save] -> {path}")


def _format_table_value(value):
    if value == "" or value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    try:
        v = float(value)
        if not np.isfinite(v):
            return ""
        if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 1e-4):
            return f"{v:.2e}"
        if abs(v) >= 10:
            return f"{v:.3f}"
        return f"{v:.6f}"
    except Exception:
        return str(value)


def _make_final_comparison_row(result):
    control = result.get("control_result") or {}
    legacy_best = control.get("best", {})
    if isinstance(legacy_best, dict):
        control = {**legacy_best, **control}
    params = result.get("selected_params") or {}

    raw_metrics = control.get("raw_readout_metrics", {})
    corrected_metrics = control.get("corrected_feedback_input_metrics", {})

    controller_name = control.get("controller", "")
    if controller_name == "linear_feedback":
        control_method = "Linear feedback"
    elif controller_name == "finite_time":
        control_method = "Finite-time"
    elif controller_name == "pyragas":
        control_method = "Pyragas time-delay"
    else:
        control_method = ""

    return {
        "Regime": result.get("mode", ""),
        "Optimizer": result.get("optimizer", ""),
        "N_res": params.get("N_res", ""),
        "density_p": params.get("p", ""),
        "rho": params.get("spectral_radius", ""),
        "leak": params.get("leaky_coefficient", ""),
        "input_scale": params.get("input_scaling", ""),
        "ridge": params.get("regularization", ""),
        "washout": params.get("washout", ""),
        "Pred_RMSE_x": result.get("rmse_x", ""),
        "Pred_NRMSE_x": result.get("nrmse_x", ""),
        "Pred_RMSE_all": result.get("rmse_all", ""),
        "Pred_NRMSE_all": result.get("nrmse_all", ""),
        "Control_method": control_method,
        "Best_K": control.get("best_k", control.get("best_K", "")),
        "Final_test_metric_name": control.get("final_test_metric_name", ""),
        "Final_test_metric_value": control.get("final_test_metric_value", ""),
        "Selection_metric_name": control.get("selection_metric_name", ""),
        "Selection_metric_value": control.get("selection_metric_value", ""),
        "Raw_readout_target_RMSE_state": raw_metrics.get(
            "target_rmse_state", control.get("raw_readout_target_rmse_state", "")
        ),
        "Raw_readout_target_RMSE_x": raw_metrics.get(
            "target_rmse_x", control.get("raw_readout_target_rmse_x", "")
        ),
        "Raw_readout_target_NRMSE_x": raw_metrics.get(
            "target_nrmse_x", control.get("raw_readout_target_nrmse_x", "")
        ),
        "Control_target_RMSE_state": corrected_metrics.get(
            "target_rmse_state",
            control.get(
                "corrected_feedback_input_target_rmse_state",
                control.get("best_target_rmse_state", ""),
            ),
        ),
        "Control_target_RMSE_x": corrected_metrics.get(
            "target_rmse_x",
            control.get(
                "corrected_feedback_input_target_rmse_x",
                control.get("best_target_rmse_x", ""),
            ),
        ),
        "Corrected_feedback_target_NRMSE_x": corrected_metrics.get(
            "target_nrmse_x",
            control.get("corrected_feedback_input_target_nrmse_x", ""),
        ),
        "Spike_reduction_percent": control.get("best_spike_reduction_percent", ""),
        "Control_effort_mean_sq": control.get("control_effort_mean_sq", ""),
        "Control_energy_dt_sum": control.get("control_energy_dt_sum", ""),
        "Control_energy_legacy_alias": control.get("best_control_energy", ""),
        "Controller_test_time_to_tolerance": control.get(
            "controller_test_time_to_tolerance", control.get("best_settling_time", "")
        ),
        "Settling_time_legacy_alias": control.get("best_settling_time", ""),
        "Control_stable": control.get("stable", control.get("best_stable", "")),
        "Divergence_detected": control.get("divergence_detected", ""),
        "Divergence_reason": control.get("divergence_reason", ""),
        "Target_mode": control.get("target_mode", ""),
        "Controller_validation_start": control.get("controller_validation_start", ""),
        "Controller_validation_end": control.get("controller_validation_end", ""),
        "Controller_test_start": control.get("controller_test_start", ""),
        "Controller_test_end": control.get("controller_test_end", ""),
        "Pyragas_history_signal": control.get("pyragas_history_signal", ""),
        "Pyragas_quality_pass": control.get("best_pyragas_quality_pass", ""),
        "Pyragas_rhythm_CV": control.get("best_pyragas_rhythm_interval_cv", ""),
        "Pyragas_recurrence_error": control.get(
            "best_pyragas_empirical_recurrence_error_norm", ""
        ),
        "Output_folder": result.get("output_dir", ""),
    }


def _write_rows_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[Report] Saved final comparison CSV -> {path}")


def _write_rows_markdown(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    lines = []
    lines.append("# Final prediction and control comparison")
    lines.append("")
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format_table_value(row.get(k, "")) for k in fieldnames) + " |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Report] Saved final comparison Markdown -> {path}")


def _write_rows_table_png(path, rows):
    if not rows:
        return
    try:
        plot_final_comparison_table(path, rows, formatter=_format_table_value)
    except Exception as e:
        print(f"[Report] Final comparison PNG skipped: {e}")


def save_final_comparison_table(results):
    if not results:
        return

    root = getattr(config, "OUTPUT_ROOT", ROOT_OUTPUT_DIR)
    os.makedirs(root, exist_ok=True)

    rows = [_make_final_comparison_row(r) for r in results]

    csv_path = os.path.join(root, "final_prediction_control_comparison.csv")
    md_path = os.path.join(root, "final_prediction_control_comparison.md")
    png_path = os.path.join(root, "final_prediction_control_comparison.png")

    _write_rows_csv(csv_path, rows)
    _write_rows_markdown(md_path, rows)
    _write_rows_table_png(png_path, rows)

    print("\n" + "=" * 72)
    print("FINAL PREDICTION + CONTROL COMPARISON TABLE")
    print("=" * 72)

    for row in rows:
        print(
            f"{row.get('Regime', ''):>20} | "
            f"opt={row.get('Optimizer', ''):>8} | "
            f"control={row.get('Control_method', ''):>18} | "
            f"pred NRMSE x={_format_table_value(row.get('Pred_NRMSE_x')):>10} | "
            f"K={_format_table_value(row.get('Best_K')):>10} | "
            f"control RMSE={_format_table_value(row.get('Control_target_RMSE_state')):>10} | "
            f"stable={row.get('Control_stable', '')}"
        )

    print("=" * 72)


def set_hr_mode(mode: str | None):
    if mode is None:
        mode = getattr(config, "HR_MODE", "periodic_bursting")

    config.HR_MODE = mode
    config.HR_REGIME = mode
    config.HR_DYNAMICS_MODE = mode
    config.HINDMARSH_ROSE_MODE = mode

    return mode


def make_output_folder(args, active_hr_mode):
    config.OUTPUT_DIR = ROOT_OUTPUT_DIR

    if setup_run_output_folder is not None:
        try:
            folder = setup_run_output_folder(config)
        except Exception:
            folder = None
    else:
        folder = None

    if folder is None:
        if args.dataset.lower() == "hr":
            folder = os.path.join(ROOT_OUTPUT_DIR, active_hr_mode or "hr")
        else:
            folder = os.path.join(ROOT_OUTPUT_DIR, args.dataset.lower())

    folder = os.path.normpath(folder)

    if args.clean_output and os.path.isdir(folder):
        print(f"[Clean] Removing old output folder -> {folder}")
        shutil.rmtree(folder)

    os.makedirs(folder, exist_ok=True)
    config.OUTPUT_DIR = folder

    return folder


def split_train_test(series):
    series = as_2d(series)
    n_train = int(len(series) * config.TRAIN_RATIO)
    return series[:n_train], series[n_train:]


def normalize_from_train(train, test):
    train = as_2d(train)
    test = as_2d(test)

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0

    train_norm = (train - mean) / std
    test_norm = (test - mean) / std

    return train_norm, test_norm, mean, std


def default_params(input_size):
    if input_size == 3:
        return {
            "N_res": 600,
            "p": 0.10,
            "spectral_radius": 0.70,
            "leaky_coefficient": 0.25,
            "input_scaling": 0.15,
            "regularization": 1e-6,
            "washout": 250,
        }

    return {
        "N_res": 300,
        "p": 0.10,
        "spectral_radius": 0.85,
        "leaky_coefficient": 0.50,
        "input_scaling": 0.50,
        "regularization": 1e-6,
        "washout": 200,
    }
_REQUIRED_ESN_PARAMS = (
    "N_res",
    "p",
    "spectral_radius",
    "leaky_coefficient",
    "input_scaling",
    "regularization",
)


def load_selected_params(path):
    source_path = os.path.abspath(os.path.expanduser(path))
    with open(source_path, "r", encoding="utf-8") as handle:
        params = json.load(handle)

    if not isinstance(params, dict):
        raise ValueError("Selected parameter file must contain a JSON object.")

    missing = [key for key in _REQUIRED_ESN_PARAMS if key not in params]
    if missing:
        raise ValueError(
            "Selected parameter file is missing: " + ", ".join(missing)
        )

    params = dict(params)
    params["parameter_source_file"] = source_path
    params["optimization_reused"] = True
    return params


def _selected_reservoir_seed(params):
    seed = params.get("reservoir_seed")
    if seed is None:
        seeds = params.get("evaluation_seeds", params.get("bo_evaluation_seeds", []))
        if isinstance(seeds, (list, tuple)) and seeds:
            seed = seeds[0]
    if seed is None:
        seed = getattr(
            config,
            "BO_RESERVOIR_SEED",
            getattr(config, "RANDOM_SEED", 42),
        )
    return int(seed)


def make_model(params, input_size):
    return EchoStateNetwork(
        N_res=int(params["N_res"]),
        p=float(params["p"]),
        spectral_radius=float(params["spectral_radius"]),
        leaky_coefficient=float(params["leaky_coefficient"]),
        regularization=float(params["regularization"]),
        input_scaling=float(params.get("input_scaling", 0.5)),
        input_size=int(input_size),
        normalize_input=False,
        seed=_selected_reservoir_seed(params),
    )


def run_all_optimizers(
    loader,
    neuron_id,
    *,
    selection_series=None,
    heldout_length=0,
):
    optimizers = getattr(config, "OPTIMIZERS_TO_COMPARE", ["gp", "dummy", "forest", "gbrt"])

    all_history = []
    summary = []

    print("\n" + "=" * 72)
    print("AUTO OPTIMIZER COMPARISON")
    print("=" * 72)

    for opt in optimizers:
        result = optimize_hyperparameters(
            loader,
            neuron_id,
            optimizer=opt,
            selection_series=selection_series,
            heldout_length=heldout_length,
        )

        all_history.extend(result.history)

        save_json(result.best_params, f"best_params_{opt}.json")

        summary.append(
            {
                "optimizer": opt,
                "best_score": float(result.best_score),
                **result.best_params,
            }
        )

    summary = sorted(summary, key=lambda r: r["best_score"])
    best = summary[0]

    save_json(summary, "optimizer_summary.json")
    save_csv(all_history, "optimizer_history.csv")
    save_csv(all_history, "optimizer_results.csv")

    plot_optimizer_convergence(all_history)

    plot_bo_objective_landscape(
        all_history,
        optimizer=str(best.get("optimizer", "forest")),
        params=("input_scaling", "spectral_radius", "leaky_coefficient"),
        filename="bo_objective_landscape_best_optimizer.png",
    )

    print("\n" + "=" * 72)
    print("OPTIMIZER RANKING")
    print("=" * 72)

    ranking_rows = []

    for rank, row in enumerate(summary, start=1):
        ranking_row = {
            "rank": rank,
            "optimizer": row.get("optimizer"),
            "best_score": row.get("best_score"),
            "validation_nrmse_x": row.get("validation_nrmse_x", row.get("x_nrmse", 0.0)),
            "validation_nrmse_all": row.get("validation_nrmse", row.get("all_nrmse", 0.0)),
            "N_res": row.get("N_res"),
            "p": row.get("p"),
            "spectral_radius": row.get("spectral_radius"),
            "leaky_coefficient": row.get("leaky_coefficient"),
            "input_scaling": row.get("input_scaling"),
            "regularization": row.get("regularization"),
            "washout": row.get("washout"),
        }

        ranking_rows.append(ranking_row)

        print(
            f"{row['optimizer']:>8} | "
            f"score={float(row['best_score']):.6f} | "
            f"x_nrmse={float(row.get('validation_nrmse_x', row.get('x_nrmse', 0.0))):.6f} | "
            f"N={int(row['N_res'])} | "
            f"rho={float(row['spectral_radius']):.3f} | "
            f"leak={float(row['leaky_coefficient']):.3f} | "
            f"scale={float(row['input_scaling']):.3f} | "
            f"washout={int(row.get('washout', 200))}"
        )

    save_csv(ranking_rows, "optimizer_ranking_table.csv")

    best_params = {
        "N_res": int(best["N_res"]),
        "p": float(best["p"]),
        "spectral_radius": float(best["spectral_radius"]),
        "leaky_coefficient": float(best["leaky_coefficient"]),
        "input_scaling": float(best["input_scaling"]),
        "regularization": float(best["regularization"]),
        "washout": int(best.get("washout", 200)),
        "validation_score": float(best.get("validation_score", best["best_score"])),
        "validation_nrmse": float(best.get("validation_nrmse", best.get("all_nrmse", 0.0))),
        "validation_nrmse_x": float(best.get("validation_nrmse_x", best.get("x_nrmse", 0.0))),
        "validation_std_ratio": float(best.get("validation_std_ratio", 0.0)),
        "validation_mean_gap": float(best.get("validation_mean_gap", 0.0)),
    }

    # Preserve optimizer/reservoir seed and aggregation provenance when auto
    # chooses a method.
    provenance_keys = {
        "score_aggregation",
        "test_data_used_for_selection",
        "final_training_end",
        "heldout_test_start",
        "heldout_test_end",
    }
    for key, value in best.items():
        if (
            "seed" in str(key).lower()
            or str(key).startswith("validation_")
            or key in provenance_keys
        ) and key not in best_params:
            best_params[key] = value

    save_json(best_params, "best_params_auto.json")

    return best["optimizer"], best_params, summary, all_history


def call_experiment_report(
    loader,
    optimizer_name,
    params,
    metrics,
    base_output_dir,
    optimizer_summary,
):
    if save_experiment_summary is None:
        print("[Report] experiment_report.py not available, report skipped.")
        return

    try:
        save_experiment_summary(
            loader=loader,
            config=config,
            optimizer_name=optimizer_name,
            params=params,
            metrics=metrics,
            base_output_dir=base_output_dir,
            optimizer_summary=optimizer_summary,
        )
    except TypeError:
        try:
            save_experiment_summary(
                loader=loader,
                config=config,
                optimizer_name=optimizer_name,
                params=params,
                metrics=metrics,
                base_output_dir=base_output_dir,
            )
        except Exception as e:
            print(f"[Report] Summary skipped: {e}")
    except Exception as e:
        print(f"[Report] Summary skipped: {e}")


def run_control_if_requested(
    args,
    esn,
    loader,
    train,
    test,
    train_norm,
    test_norm,
    mean,
    std,
    times,
    base_output_dir,
    active_hr_mode,
    best_params,
    best_name,
    input_size,
):
    if not args.control:
        return None

    print("\n" + "=" * 72)
    print(f"{args.controller.upper()} CONTROL EXPERIMENT")
    print("=" * 72)

    if run_control_experiment is None:
        message = "control_experiment.py is missing or could not be imported"
        import_error = globals().get("CONTROL_IMPORT_ERROR")
        raise RuntimeError(message) from import_error

    if config.DATASET_MODE != "hr":
        raise ValueError("Control was requested, but control currently requires HR data")

    if input_size != 3:
        raise ValueError("Control was requested, but full-state HR input_size=3 is required")

    target_mode = args.control_target_mode
    if target_mode == "rest_state":
        print("[Control] Deprecated target 'rest_state'; using quiet-training-data target.")

    try:
        control_result = run_control_experiment(
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
            base_output_dir=base_output_dir,
            hr_mode=active_hr_mode,
            best_params=best_params,
            optimizer_name=best_name,
            control_k=args.control_k,
            control_start_frac=args.control_start_frac,
            control_target_mode=target_mode,
            auto_control_k=args.auto_control_k,
            k_min=args.k_min,
            k_max=args.k_max,
            k_num=args.k_num,
            k_refine_num=args.k_refine_num,
            controller=args.controller,
            finite_s=args.finite_s,
            pyragas_delay=args.pyragas_delay,
            pyragas_sign=args.pyragas_sign,
            pyragas_history_signal=args.pyragas_history_signal,
            validation_only=args.control_validation_only,
        )
        return control_result

    except Exception as e:
        print(f"[Control] Failed: {e}")
        raise


def run_single_experiment(args, hr_mode: str | None = None):
    config.DATASET_MODE = args.dataset.lower()
    reused_params = None
    if args.params_file:
        if args.no_opt:
            raise ValueError("--params-file and --no-opt cannot be used together.")
        reused_params = load_selected_params(args.params_file)


    if config.DATASET_MODE == "hr":
        active_hr_mode = set_hr_mode(hr_mode or args.hr_mode)
    else:
        active_hr_mode = None

    base_output_dir = make_output_folder(args, active_hr_mode)

    print("\n" + "#" * 80)
    print("STARTING EXPERIMENT")
    print("#" * 80)
    print(f"[Main] dataset mode  = {config.DATASET_MODE}")

    if active_hr_mode is not None:
        print(f"[Main] HR mode       = {active_hr_mode}")

    print(f"[Main] output folder = {config.OUTPUT_DIR}")

    print("\n" + "=" * 72)
    print("LOADING DATA")
    print("=" * 72)

    loader = DataLoader(csv_path=config.DATA_PATH)
    loader.load()
    loader.preprocess()
    loader.detect_spikes()
    loader.summary()

    if config.DATASET_MODE == "hr":
        loader.list_neurons(3)
    else:
        loader.list_neurons(1)

    series, series_name = get_model_series(loader, args.neuron)
    times = np.asarray(loader.time, dtype=float)

    input_size = series.shape[1]
    train, test = split_train_test(series)

    print("\n" + "=" * 72)
    print("MODEL SELECTION")
    print("=" * 72)
    print(f"Training series : {series_name}")
    print(f"Input size      : {input_size}")

    optimizer_summary = []
    optimizer_history = []

    if reused_params is not None:
        best_name = (
            args.optimizer if args.optimizer != "auto" else "reused_parameters"
        )
        best_params = reused_params
        print(f"Reusing parameters: {best_params['parameter_source_file']}")

    elif args.no_opt:
        best_name = "defaults"
        best_params = default_params(input_size)

    elif args.optimizer == "auto":
        best_name, best_params, optimizer_summary, optimizer_history = run_all_optimizers(
            loader, args.neuron
        )

    else:
        result = optimize_hyperparameters(loader, args.neuron, optimizer=args.optimizer)
        best_name = args.optimizer
        best_params = result.best_params
        optimizer_history = result.history
        optimizer_summary = [
            {
                "optimizer": best_name,
                "best_score": float(result.best_score),
                **best_params,
            }
        ]
        save_json(best_params, f"best_params_{best_name}.json")
        save_csv(optimizer_history, "optimizer_history.csv")
        plot_optimizer_convergence(optimizer_history)
        plot_bo_objective_landscape(
            optimizer_history,
            optimizer=best_name,
            params=("input_scaling", "spectral_radius", "leaky_coefficient"),
            filename="bo_objective_landscape_best_optimizer.png",
        )

    best_params = dict(best_params)
    best_params.setdefault("reservoir_seed", _selected_reservoir_seed(best_params))

    save_json(best_params, "best_params.json")

    print("\n" + "=" * 72)
    print("FINAL TRAINING WITH SELECTED PARAMETERS")
    print("=" * 72)
    print(f"Selected optimizer : {best_name}")

    train_norm, test_norm, mean, std = normalize_from_train(train, test)
    eval_norm = np.vstack([train_norm, test_norm])

    warmup_steps = len(train_norm) - 1
    washout = resolve_washout(best_params.get("washout", 200), len(train_norm))

    esn = make_model(best_params, input_size)
    esn.train(train_norm, washout=washout)

    pred_norm, _ = esn.predict(eval_norm, n_warmup=warmup_steps)

    truth_norm = test_norm
    pred = pred_norm * std + mean
    truth = test

    pred_x = pred[:, 0]
    truth_x = truth[:, 0]

    pred_times = times[warmup_steps + 1 : warmup_steps + 1 + len(pred_x)]

    rmse_x = rmse(pred_x, truth_x)
    nrmse_x = nrmse(pred_norm[:, 0], truth_norm[:, 0])
    rmse_all = rmse(pred, truth)
    nrmse_all = nrmse(pred_norm, truth_norm)

    print(f"Warmup steps              : {warmup_steps}")
    print(f"Training washout          : {washout}")
    print(f"Held-out prediction steps : {len(pred_x)}")
    print(f"NRMSE recursive x         : {nrmse_x:.6f}")
    print(f"RMSE recursive x          : {rmse_x:.6f}")
    print(f"NRMSE recursive all states: {nrmse_all:.6f}")
    print(f"RMSE recursive all states : {rmse_all:.6f}")

    split = {
        "neuron_name": "hr_x" if input_size == 3 else series_name,
        "neuron_index": args.neuron,
        "full_time": times,
        "full_signal": series[:, 0],
        "train_time": times[: len(train)],
        "val_time": np.array([]),
        "test_time": times[len(train) :],
        "train_signal": train[:, 0],
        "val_signal": np.array([]),
        "test_signal": test[:, 0],
        "y_test": truth_x.reshape(-1, 1),
        "t_test_y": pred_times,
    }

    print("\n" + "=" * 72)
    print("PLOTTING")
    print("=" * 72)

    plot_results(split, pred_x, tag=f"ESN ({best_name})")

    if input_size == 3:
        plot_all_states(
            t=pred_times,
            truth=truth,
            pred=pred,
            tag=f"ESN ({best_name})",
        )

    metrics = {
        "dataset_mode": config.DATASET_MODE,
        "hr_mode": active_hr_mode,
        "series": series_name,
        "optimizer": best_name,
        "input_size": input_size,
        "rmse_recursive_x": rmse_x,
        "nrmse_recursive_x": nrmse_x,
        "rmse_recursive_all_states": rmse_all,
        "nrmse_recursive_all_states": nrmse_all,
        "selected_params": best_params,
    }

    save_json(metrics, "metrics.json")

    call_experiment_report(
        loader=loader,
        optimizer_name=best_name,
        params=best_params,
        metrics={
            "rmse_x": rmse_x,
            "nrmse_x": nrmse_x,
            "rmse_all_states": rmse_all,
            "nrmse_all_states": nrmse_all,
        },
        base_output_dir=base_output_dir,
        optimizer_summary=optimizer_summary,
    )

    control_result = run_control_if_requested(
        args=args,
        esn=esn,
        loader=loader,
        train=train,
        test=test,
        train_norm=train_norm,
        test_norm=test_norm,
        mean=mean,
        std=std,
        times=times,
        base_output_dir=base_output_dir,
        active_hr_mode=active_hr_mode,
        best_params=best_params,
        best_name=best_name,
        input_size=input_size,
    )

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(f"Series    : {series_name}")
    print(f"Optimizer : {best_name}")
    print(f"RMSE x    : {rmse_x:.6f}")
    print(f"NRMSE x   : {nrmse_x:.6f}")

    if input_size == 3:
        print("Full-state prediction: x, y, z")

    if control_result is not None:
        legacy_best = control_result.get("best", {})
        if isinstance(legacy_best, dict):
            control_result = {**legacy_best, **control_result}
        print(f"Controller                : {control_result.get('controller')}")
        print(
            "Control best K            : "
            f"{control_result.get('best_k', control_result.get('best_K'))}"
        )
        print(
            "Final test metric         : "
            f"{control_result.get('final_test_metric_name')} = {control_result.get('final_test_metric_value')}"
        )
        if control_result.get("controller") == "pyragas":
            print(
                "Pyragas quality pass      : "
                f"{control_result.get('best_pyragas_quality_pass')}"
            )
            print(
                "Pyragas rhythm CV         : "
                f"{control_result.get('best_pyragas_rhythm_interval_cv')}"
            )
        else:
            print(
                "Control best target RMSE  : "
                f"{control_result.get('corrected_feedback_input_target_rmse_state', control_result.get('best_target_rmse_state'))}"
            )
        print(
            "Raw-readout target RMSE   : "
            f"{control_result.get('raw_readout_target_rmse_state')}"
        )
        print(f"Control outputs saved in  : {control_result.get('output_dir')}")

    print(f"[Done] Files saved inside: {config.OUTPUT_DIR}")

    return {
        "mode": active_hr_mode or config.DATASET_MODE,
        "optimizer": best_name,
        "rmse_x": rmse_x,
        "nrmse_x": nrmse_x,
        "rmse_all": rmse_all,
        "nrmse_all": nrmse_all,
        "output_dir": config.OUTPUT_DIR,
        "control_result": control_result,
        "selected_params": best_params,
        "rmse_all_states": rmse_all,
        "nrmse_all_states": nrmse_all,
    }


def main():
    global ROOT_OUTPUT_DIR
    args = parse_args()

    if args.output_root:
        ROOT_OUTPUT_DIR = os.path.abspath(os.path.expanduser(args.output_root))
        config.OUTPUT_ROOT = ROOT_OUTPUT_DIR
        config.OUTPUT_DIR = ROOT_OUTPUT_DIR

    if args.run_all_regimes:
        print("\n" + "#" * 80)
        print("RUNNING ALL HINDMARSH-ROSE REGIMES")
        print("#" * 80)

        results = []

        for mode in HR_REGIMES:
            results.append(run_single_experiment(args, hr_mode=mode))

        print("\n" + "=" * 72)
        print("ALL REGIMES FINISHED")
        print("=" * 72)

        for r in results:
            print(
                f"{r['mode']:>20} | "
                f"optimizer={r['optimizer']:>8} | "
                f"NRMSE x={r['nrmse_x']:.6f} | "
                f"folder={r['output_dir']}"
            )

        if args.control_validation_only:
            print(
                "[Control] Validation-only run: final comparison tables were not "
                "written because no held-out controller test was evaluated."
            )
        else:
            save_final_comparison_table(results)
            print("[Done] Global comparison files are inside the outputs folder.")

    else:
        result = run_single_experiment(args)
        if args.control_validation_only:
            print(
                "[Control] Validation-only run: final comparison tables were not "
                "written because no held-out controller test was evaluated."
            )
        else:
            save_final_comparison_table([result])


if __name__ == "__main__":
    main()
