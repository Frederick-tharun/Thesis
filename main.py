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
    from control_experiment import run_linear_feedback_control_experiment
except Exception as e:
    run_linear_feedback_control_experiment = None
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
        help="Run linear-feedback control experiment after ESN prediction",
    )

    p.add_argument(
        "--control-k",
        type=float,
        default=None,
        help="Single K value for linear feedback. If omitted, config.CONTROL_LINEAR_K_SWEEP is used.",
    )

    p.add_argument(
    "--auto-control-k",
    action="store_true",
    help="Automatically search K values and select the best K. ESN is trained once, then many K values are tested quickly.",
    )

    p.add_argument(
    "--k-min",
    type=float,
    default=None,
    help="Minimum K for automatic K search. If omitted, config.CONTROL_AUTO_K_MIN is used.",
    )

    p.add_argument(
    "--k-max",
    type=float,
    default=None,
    help="Maximum K for automatic K search. If omitted, config.CONTROL_AUTO_K_MAX is used.",
    )

    p.add_argument(
    "--k-num",
    type=int,
    default=None,
    help="Number of coarse K values for automatic K search. If omitted, config.CONTROL_AUTO_K_NUM is used.",
    )

    p.add_argument(
    "--k-refine-num",
    type=int,
    default=None,
    help="Number of refined K values around the best coarse K. If omitted, config.CONTROL_AUTO_K_REFINE_NUM is used.",
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
        default=getattr(config, "CONTROL_TARGET_MODE", "rest_state"),
        choices=["rest_state", "zero", "mean"],
        help="How to choose the control target state",
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


def _safe_float_for_table(value, default=""):
    try:
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


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
    """
    Builds one thesis/professor-facing row that combines:
    - ESN prediction result
    - selected BO hyperparameters
    - control result
    """
    control = result.get("control_result") or {}
    params = result.get("selected_params") or {}

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
        "Control_method": "Linear feedback" if control else "",
        "Best_K": control.get("best_K", ""),
        "Control_target_RMSE_state": control.get("best_target_rmse_state", ""),
        "Control_target_RMSE_x": control.get("best_target_rmse_x", ""),
        "Spike_reduction_percent": control.get("best_spike_reduction_percent", ""),
        "Control_energy": control.get("best_control_energy", ""),
        "Settling_time": control.get("best_settling_time", ""),
        "Control_stable": control.get("best_stable", ""),
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
    """
    Saves the final table your professor can look at.

    Files created in outputs/:
      - final_prediction_control_comparison.csv
      - final_prediction_control_comparison.md
      - final_prediction_control_comparison.png
    """
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
        seed=config.RANDOM_SEED,
    )


def run_all_optimizers(loader, neuron_id):
    optimizers = getattr(config, "OPTIMIZERS_TO_COMPARE", ["gp", "dummy", "forest", "gbrt"])

    all_history = []
    summary = []

    print("\n" + "=" * 72)
    print("AUTO OPTIMIZER COMPARISON")
    print("=" * 72)

    for opt in optimizers:
        result = optimize_hyperparameters(loader, neuron_id, optimizer=opt)

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

    # Paper-style BO objective/partial-dependence plot for the best optimizer.
    # This is the thesis-friendly replacement for the simple optimizer heatmap.
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
    print("LINEAR FEEDBACK CONTROL EXPERIMENT")
    print("=" * 72)

    if run_linear_feedback_control_experiment is None:
        print("[Control] Skipped: control_experiment.py is missing or could not be imported.")
        try:
            print(f"[Control] Import error: {CONTROL_IMPORT_ERROR}")
        except Exception:
            pass
        return None

    if config.DATASET_MODE != "hr":
        print("[Control] Skipped: control experiment currently expects HR data.")
        return None

    if input_size != 3:
        print("[Control] Skipped: control experiment expects full-state HR input size = 3.")
        return None

    try:
        control_result = run_linear_feedback_control_experiment(
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
    control_target_mode=args.control_target_mode,
    auto_control_k=args.auto_control_k,
    k_min=args.k_min,
    k_max=args.k_max,
    k_num=args.k_num,
    k_refine_num=args.k_refine_num,
)
        return control_result

    except Exception as e:
        print(f"[Control] Failed: {e}")
        return None


def run_single_experiment(args, hr_mode: str | None = None):
    config.DATASET_MODE = args.dataset.lower()

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

    if args.no_opt:
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
        print(f"Control best K            : {control_result.get('best_K')}")
        print(f"Control best target RMSE  : {control_result.get('best_target_rmse_state'):.6f}")
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
    args = parse_args()

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

        save_final_comparison_table(results)

        print("[Done] Global comparison files are inside the outputs folder.")

    else:
        result = run_single_experiment(args)
        save_final_comparison_table([result])


if __name__ == "__main__":
    main()