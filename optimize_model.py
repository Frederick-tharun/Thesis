from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

import config
from model import EchoStateNetwork

try:
    from skopt import Optimizer
    from skopt.space import Integer, Real
except ImportError as e:
    raise ImportError(
        "scikit-optimize is missing. Install it with: pip install scikit-optimize"
    ) from e


# ============================================================
# Basic metrics
# ============================================================

def as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if y_pred.shape != y_true.shape:
        n = min(len(y_pred), len(y_true))
        y_pred = y_pred[:n]
        y_true = y_true[:n]

    if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(y_true)):
        return float("inf")

    value = np.sqrt(np.mean((y_pred - y_true) ** 2))

    if not np.isfinite(value):
        return float("inf")

    return float(value)


def nrmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if y_pred.shape != y_true.shape:
        n = min(len(y_pred), len(y_true))
        y_pred = y_pred[:n]
        y_true = y_true[:n]

    if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(y_true)):
        return float("inf")

    denom = float(np.std(y_true))
    if denom < 1e-12:
        denom = 1.0

    value = rmse(y_pred, y_true) / denom

    if not np.isfinite(value):
        return float("inf")

    return float(value)


def _safe_float(x: Any, fallback: float = 1_000_000.0) -> float:
    try:
        x = float(x)
        if not np.isfinite(x):
            return float(fallback)
        return x
    except Exception:
        return float(fallback)


def _safe_metric(x: Any) -> float:
    return _safe_float(x, fallback=1_000_000.0)


# ============================================================
# Config helpers
# ============================================================

def _get_config(names: list[str], default: Any) -> Any:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _get_int_config(names: list[str], default: int) -> int:
    return int(_get_config(names, default))


def _get_float_config(names: list[str], default: float) -> float:
    return float(_get_config(names, default))


# ============================================================
# Data extraction
# ============================================================

def _matrix_from_object(obj: Any) -> np.ndarray | None:
    if obj is None:
        return None

    # pandas DataFrame support without importing pandas directly
    if hasattr(obj, "select_dtypes"):
        try:
            numeric = obj.select_dtypes(include=["number"])
            cols = list(numeric.columns)

            # remove time-like columns
            use_cols = [
                c for c in cols
                if str(c).lower() not in ["time", "t", "timestamp", "frame"]
            ]

            if len(use_cols) > 0:
                arr = numeric[use_cols].to_numpy(dtype=float)
            else:
                arr = numeric.to_numpy(dtype=float)

            return as_2d(arr)
        except Exception:
            return None

    try:
        arr = np.asarray(obj, dtype=float)
        if arr.ndim == 0:
            return None
        arr = as_2d(arr)
        return arr
    except Exception:
        return None


def _get_loader_matrix(loader) -> np.ndarray:
    """
    Robustly finds the signal matrix inside your DataLoader.

    Your DataLoader has used names like:
    - data_raw
    - data_norm
    - dff
    - raw_df

    For HR mode, we prefer raw x/y/z states.
    """

    preferred_names = [
        "data_raw",
        "data",
        "dff",
        "signals",
        "signal",
        "values",
        "data_norm",
        "raw_df",
    ]

    for name in preferred_names:
        if hasattr(loader, name):
            arr = _matrix_from_object(getattr(loader, name))
            if arr is not None and arr.ndim == 2 and arr.shape[0] > 1:
                return arr

    available = [
        name for name in dir(loader)
        if not name.startswith("_") and not callable(getattr(loader, name))
    ]

    raise AttributeError(
        "Could not find signal matrix inside DataLoader.\n"
        f"Available attributes: {available}\n"
        "Expected one of: data_raw, data, dff, data_norm, raw_df"
    )


def get_model_series(loader, neuron_id: int = 0):
    """
    Returns the series used by the ESN.

    For HR:
        returns full state [x, y, z]

    For real neuron CSV:
        returns one selected neuron column
    """

    data = _get_loader_matrix(loader)
    data = as_2d(data)

    dataset_mode = str(getattr(config, "DATASET_MODE", "hr")).lower()

    if dataset_mode == "hr":
        if data.shape[1] >= 3:
            return data[:, :3], "hr_full_state"
        return data, "hr_full_state"

    neuron_id = int(neuron_id)
    neuron_id = max(0, min(neuron_id, data.shape[1] - 1))

    name = f"neuron_{neuron_id}"
    if hasattr(loader, "neuron_names"):
        try:
            name = str(loader.neuron_names[neuron_id])
        except Exception:
            pass

    return data[:, neuron_id:neuron_id + 1], name


# ============================================================
# Washout
# ============================================================

def resolve_washout(washout: int | float, train_len: int) -> int:
    train_len = int(train_len)
    washout = int(washout)

    if train_len <= 5:
        return 0

    max_washout = max(1, train_len // 3)
    washout = max(0, min(washout, max_washout))

    return int(washout)


# ============================================================
# Optimization result object
# ============================================================

@dataclass
class OptimizationResult:
    best_params: dict
    best_score: float
    history: list[dict]


# ============================================================
# BO space
# ============================================================

def _get_search_space(input_size: int):
    """
    Safe ESN search space.

    Important:
    - spectral_radius is limited to reduce recursive explosion
    - input_scaling is limited to reduce numerical blow-up
    - regularization is log-uniform
    """

    n_min = _get_int_config(["BO_N_RES_MIN", "N_RES_MIN"], 250)
    n_max = _get_int_config(["BO_N_RES_MAX", "N_RES_MAX"], 800)

    p_min = _get_float_config(["BO_P_MIN", "P_MIN"], 0.02)
    p_max = _get_float_config(["BO_P_MAX", "P_MAX"], 0.20)

    rho_min = _get_float_config(["BO_RHO_MIN", "SPECTRAL_RADIUS_MIN"], 0.45)
    rho_max = _get_float_config(["BO_RHO_MAX", "SPECTRAL_RADIUS_MAX"], 1.05)

    leak_min = _get_float_config(["BO_LEAK_MIN", "LEAK_MIN"], 0.08)
    leak_max = _get_float_config(["BO_LEAK_MAX", "LEAK_MAX"], 0.80)

    scale_min = _get_float_config(["BO_SCALE_MIN", "INPUT_SCALING_MIN"], 0.04)
    scale_max = _get_float_config(["BO_SCALE_MAX", "INPUT_SCALING_MAX"], 0.60)

    ridge_min = _get_float_config(["BO_RIDGE_MIN", "RIDGE_MIN"], 1e-8)
    ridge_max = _get_float_config(["BO_RIDGE_MAX", "RIDGE_MAX"], 1e-3)

    washout_min = _get_int_config(["BO_WASHOUT_MIN", "WASHOUT_MIN"], 50)
    washout_max = _get_int_config(["BO_WASHOUT_MAX", "WASHOUT_MAX"], 500)

    dimensions = [
        Integer(n_min, n_max, name="N_res"),
        Real(p_min, p_max, name="p"),
        Real(rho_min, rho_max, name="spectral_radius"),
        Real(leak_min, leak_max, name="leaky_coefficient"),
        Real(scale_min, scale_max, name="input_scaling"),
        Real(ridge_min, ridge_max, prior="log-uniform", name="regularization"),
        Integer(washout_min, washout_max, name="washout"),
    ]

    return dimensions


def _x_to_params(x: list[Any]) -> dict:
    return {
        "N_res": int(x[0]),
        "p": float(x[1]),
        "spectral_radius": float(x[2]),
        "leaky_coefficient": float(x[3]),
        "input_scaling": float(x[4]),
        "regularization": float(x[5]),
        "washout": int(x[6]),
    }


# ============================================================
# Segment preparation
# ============================================================

def _normalize_from_train(train: np.ndarray, val: np.ndarray):
    train = as_2d(train)
    val = as_2d(val)

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)

    std[std < 1e-12] = 1.0

    train_norm = (train - mean) / std
    val_norm = (val - mean) / std

    return train_norm, val_norm, mean, std


def _prepare_optimizer_segments(series: np.ndarray):
    """
    Uses only the training part for BO.

    Final test remains unseen.
    """

    series = as_2d(series)
    n_total = len(series)

    train_ratio = float(getattr(config, "TRAIN_RATIO", 0.85))
    n_final_train = int(n_total * train_ratio)

    n_final_train = max(10, min(n_final_train, n_total - 1))
    trainval = series[:n_final_train]

    max_train_steps = _get_int_config(
        ["OPT_TRAIN_MAX_STEPS", "BO_TRAIN_MAX_STEPS"],
        50000,
    )

    val_steps_default = min(10000, max(1000, int(len(trainval) * 0.20)))
    val_steps = _get_int_config(
        ["OPT_VAL_STEPS", "BO_VAL_STEPS"],
        val_steps_default,
    )

    val_steps = max(200, min(val_steps, len(trainval) // 3))

    train_part = trainval[:-val_steps]
    val_part = trainval[-val_steps:]

    if len(train_part) > max_train_steps:
        train_part = train_part[-max_train_steps:]

    if len(train_part) < 100:
        raise ValueError(
            f"Not enough training data for optimization. train={len(train_part)}, val={len(val_part)}"
        )

    return train_part, val_part


# ============================================================
# Model helper
# ============================================================

def _make_model(params: dict, input_size: int, seed_offset: int = 0) -> EchoStateNetwork:
    seed = int(getattr(config, "RANDOM_SEED", 42)) + int(seed_offset)

    return EchoStateNetwork(
        N_res=int(params["N_res"]),
        p=float(params["p"]),
        spectral_radius=float(params["spectral_radius"]),
        leaky_coefficient=float(params["leaky_coefficient"]),
        regularization=float(params["regularization"]),
        input_scaling=float(params.get("input_scaling", 0.5)),
        input_size=int(input_size),
        normalize_input=False,
        seed=seed,
    )


# ============================================================
# Objective
# ============================================================

def _bad_result(params: dict, reason: str, iteration: int, optimizer: str, best_score: float):
    score = 1_000_000.0

    row = {
        "iteration": int(iteration),
        "optimizer": str(optimizer),
        "score": score,
        "best_score": float(min(best_score, score)),
        "validation_score": score,
        "validation_nrmse": score,
        "validation_nrmse_x": score,
        "validation_std_ratio": score,
        "validation_mean_gap": score,
        "validation_penalty": score,
        "stable": False,
        "reason": str(reason),
        **params,
    }

    return score, row


def _evaluate_params(
    params: dict,
    train: np.ndarray,
    val: np.ndarray,
    input_size: int,
    iteration: int,
    optimizer: str,
    best_score: float,
):
    """
    Long recursive stability objective.

    This function NEVER returns NaN.
    If the model explodes, it returns score = 1e6.
    """

    max_score = 1_000_000.0
    max_abs_prediction = float(getattr(config, "BO_MAX_ABS_PREDICTION", 1e6))

    try:
        train_norm, val_norm, _, _ = _normalize_from_train(train, val)

        washout = resolve_washout(params.get("washout", 200), len(train_norm))

        esn = _make_model(params, input_size=input_size, seed_offset=iteration)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)

            esn.train(train_norm, washout=washout)

            eval_norm = np.vstack([train_norm, val_norm])
            warmup_steps = len(train_norm) - 1

            pred_norm, _ = esn.predict(eval_norm, n_warmup=warmup_steps)

        pred_norm = as_2d(pred_norm)

        n = min(len(pred_norm), len(val_norm))
        pred_norm = pred_norm[:n]
        val_norm = val_norm[:n]

        if n <= 5:
            return _bad_result(params, "too_few_prediction_steps", iteration, optimizer, best_score)

        if not np.all(np.isfinite(pred_norm)):
            return _bad_result(params, "non_finite_prediction", iteration, optimizer, best_score)

        if np.max(np.abs(pred_norm)) > max_abs_prediction:
            return _bad_result(params, "prediction_exploded", iteration, optimizer, best_score)

        pred_x = pred_norm[:, 0]
        true_x = val_norm[:, 0]

        x_nrmse = nrmse(pred_x, true_x)
        all_nrmse = nrmse(pred_norm, val_norm)

        true_std = float(np.std(true_x))
        if true_std < 1e-12:
            true_std = 1.0

        pred_std = float(np.std(pred_x))
        std_ratio = pred_std / true_std

        mean_gap = abs(float(np.mean(pred_x) - np.mean(true_x))) / true_std

        penalty = 0.0

        # Penalize collapsed recursive prediction
        if std_ratio < 0.10:
            penalty += (0.10 - std_ratio) * 5.0

        # Penalize overly unstable high-variance prediction
        if std_ratio > 10.0:
            penalty += min(1000.0, std_ratio - 10.0)

        # Penalize mean drift
        if mean_gap > 1.0:
            penalty += mean_gap

        # Extra punishment if prediction becomes too large but not yet infinite
        max_abs = float(np.max(np.abs(pred_norm)))
        if max_abs > 50.0:
            penalty += min(1000.0, max_abs - 50.0)

        score = 0.70 * x_nrmse + 0.30 * all_nrmse + penalty

        if not np.isfinite(score):
            return _bad_result(params, "nan_score", iteration, optimizer, best_score)

        score = float(min(score, max_score))

        row = {
            "iteration": int(iteration),
            "optimizer": str(optimizer),
            "score": _safe_metric(score),
            "best_score": _safe_metric(min(best_score, score)),
            "validation_score": _safe_metric(score),
            "validation_nrmse": _safe_metric(all_nrmse),
            "validation_nrmse_x": _safe_metric(x_nrmse),
            "validation_std_ratio": _safe_metric(std_ratio),
            "validation_mean_gap": _safe_metric(mean_gap),
            "validation_penalty": _safe_metric(penalty),
            "stable": True,
            "reason": "ok",
            **params,
        }

        return score, row

    except Exception as e:
        return _bad_result(params, f"exception: {type(e).__name__}", iteration, optimizer, best_score)


# ============================================================
# Main optimizer
# ============================================================

def optimize_hyperparameters(loader, neuron_id: int = 0, optimizer: str = "gp") -> OptimizationResult:
    optimizer = str(optimizer).lower()

    series, series_name = get_model_series(loader, neuron_id)
    series = as_2d(series)

    input_size = int(series.shape[1])

    train_seg, val_seg = _prepare_optimizer_segments(series)

    n_calls = _get_int_config(["BO_CALLS", "N_CALLS", "N_BO_CALLS"], 30)
    random_starts = _get_int_config(["BO_RANDOM_STARTS", "N_RANDOM_STARTS"], 8)

    n_calls = max(1, int(n_calls))
    random_starts = max(1, min(int(random_starts), n_calls))

    dimensions = _get_search_space(input_size)

    base_map = {
        "gp": "GP",
        "forest": "RF",
        "gbrt": "GBRT",
        "dummy": "DUMMY",
    }

    if optimizer not in base_map:
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. Use one of: gp, dummy, forest, gbrt"
        )

    base_estimator = base_map[optimizer]

    random_seed = int(getattr(config, "RANDOM_SEED", 42))

    # Different optimizers get slightly different random streams
    seed_offset = {
        "gp": 0,
        "dummy": 100,
        "forest": 200,
        "gbrt": 300,
    }.get(optimizer, 0)

    skopt_opt = Optimizer(
        dimensions=dimensions,
        base_estimator=base_estimator,
        n_initial_points=random_starts,
        random_state=random_seed + seed_offset,
        acq_func="EI",
    )

    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Optimizer     : {optimizer}")
    print(f"Series        : {series_name}")
    print(f"Input size    : {input_size}")
    print(f"BO calls      : {n_calls}")
    print(f"Random starts : {random_starts}")
    print("Validation    : long recursive stability objective")
    print("=" * 70)

    history: list[dict] = []

    best_score = float("inf")
    best_params: dict | None = None
    best_row: dict | None = None

    for i in range(1, n_calls + 1):
        x = skopt_opt.ask()
        params = _x_to_params(x)

        score, row = _evaluate_params(
            params=params,
            train=train_seg,
            val=val_seg,
            input_size=input_size,
            iteration=i,
            optimizer=optimizer,
            best_score=best_score,
        )

        # Critical fix:
        # skopt must NEVER receive NaN or inf.
        score = _safe_float(score, fallback=1_000_000.0)

        if not np.isfinite(score):
            score = 1_000_000.0

        row["score"] = score

        if score < best_score:
            best_score = score
            best_params = params.copy()
            best_row = row.copy()

        row["best_score"] = _safe_metric(best_score)
        history.append(row)

        # Critical fix:
        # tell() must only receive finite float score.
        try:
            skopt_opt.tell(x, score)
        except ValueError:
            safe_score = 1_000_000.0
            row["score"] = safe_score
            row["reason"] = "skopt_tell_recovered_from_bad_score"
            skopt_opt.tell(x, safe_score)

        print(
            f"[{optimizer.upper()}] iter {i:>3}/{n_calls} "
            f"score={row['score']:.6f} "
            f"best={best_score:.6f} "
            f"x_nrmse={row.get('validation_nrmse_x', 1_000_000.0):.6f} "
            f"all_nrmse={row.get('validation_nrmse', 1_000_000.0):.6f} "
            f"penalty={row.get('validation_penalty', 1_000_000.0):.3f} "
            f"N={params['N_res']} "
            f"p={params['p']:.3f} "
            f"rho={params['spectral_radius']:.3f} "
            f"leak={params['leaky_coefficient']:.3f} "
            f"scale={params['input_scaling']:.3f} "
            f"ridge={params['regularization']:.1e} "
            f"washout={params['washout']}"
        )

    if best_params is None:
        best_params = {
            "N_res": 500,
            "p": 0.10,
            "spectral_radius": 0.70,
            "leaky_coefficient": 0.25,
            "input_scaling": 0.15,
            "regularization": 1e-6,
            "washout": 250,
        }
        best_score = 1_000_000.0
        best_row = {}

    # Add validation metrics into best_params because main.py expects them
    best_params["validation_score"] = _safe_metric(best_score)

    if best_row is not None:
        best_params["validation_nrmse"] = _safe_metric(
            best_row.get("validation_nrmse", 1_000_000.0)
        )
        best_params["validation_nrmse_x"] = _safe_metric(
            best_row.get("validation_nrmse_x", 1_000_000.0)
        )
        best_params["validation_std_ratio"] = _safe_metric(
            best_row.get("validation_std_ratio", 1_000_000.0)
        )
        best_params["validation_mean_gap"] = _safe_metric(
            best_row.get("validation_mean_gap", 1_000_000.0)
        )
        best_params["validation_penalty"] = _safe_metric(
            best_row.get("validation_penalty", 1_000_000.0)
        )
    else:
        best_params["validation_nrmse"] = 1_000_000.0
        best_params["validation_nrmse_x"] = 1_000_000.0
        best_params["validation_std_ratio"] = 1_000_000.0
        best_params["validation_mean_gap"] = 1_000_000.0
        best_params["validation_penalty"] = 1_000_000.0

    return OptimizationResult(
        best_params=best_params,
        best_score=float(best_score),
        history=history,
    )