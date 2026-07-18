from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

import config
from model import EchoStateNetwork

try:
    from skopt import Optimizer
    from skopt.space import Integer, Real, Space
except ImportError as e:
    raise ImportError(
        "scikit-optimize is missing. Install it with: pip install scikit-optimize"
    ) from e


class _RandomSearchOptimizer:
    """Seeded uniform random search with the Optimizer ask/tell interface."""

    def __init__(self, dimensions, random_state):
        self.space = Space(dimensions)
        self.rng = np.random.RandomState(random_state)

    def ask(self):
        return self.space.rvs(n_samples=1, random_state=self.rng)[0]

    def tell(self, _x, _score):
        return None


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


def _get_official_config(
    official_name: str,
    legacy_names: Sequence[str],
    default: Any,
) -> Any:
    """Read an official config name while retaining warned legacy fallbacks."""

    present_legacy = [name for name in legacy_names if hasattr(config, name)]

    if hasattr(config, official_name):
        if present_legacy:
            warnings.warn(
                f"Deprecated optimizer setting(s) {', '.join(present_legacy)} "
                f"are ignored because {official_name} is set.",
                FutureWarning,
                stacklevel=2,
            )
        return getattr(config, official_name)

    if present_legacy:
        legacy_name = present_legacy[0]
        warnings.warn(
            f"Optimizer setting {legacy_name} is deprecated; use "
            f"{official_name} instead.",
            FutureWarning,
            stacklevel=2,
        )
        return getattr(config, legacy_name)

    return default


def _coerce_seed(value: Any, setting_name: str) -> int:
    """Return a non-negative integer seed without silently truncating floats."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{setting_name} must be a non-negative integer, not bool")

    try:
        seed = int(value)
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{setting_name} must be a non-negative integer") from exc

    if not np.isfinite(numeric_value) or numeric_value != float(seed):
        raise ValueError(f"{setting_name} must be an integer")
    if seed < 0:
        raise ValueError(f"{setting_name} must be non-negative")

    return seed


def _validate_evaluation_seeds(values: Any) -> list[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("BO_EVALUATION_SEEDS must be a non-empty sequence of integers")

    seeds = [
        _coerce_seed(value, f"BO_EVALUATION_SEEDS[{index}]")
        for index, value in enumerate(values)
    ]

    if not seeds:
        raise ValueError("BO_EVALUATION_SEEDS must contain at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("BO_EVALUATION_SEEDS must not contain duplicate seeds")

    return seeds


def _get_reservoir_seed_config() -> tuple[int, list[int]]:
    """Resolve the final seed and the seed set used for every BO candidate."""

    default_seed = _coerce_seed(getattr(config, "RANDOM_SEED", 42), "RANDOM_SEED")
    reservoir_seed = _coerce_seed(
        getattr(config, "BO_RESERVOIR_SEED", default_seed),
        "BO_RESERVOIR_SEED",
    )
    evaluation_seeds = _validate_evaluation_seeds(
        getattr(config, "BO_EVALUATION_SEEDS", [reservoir_seed])
    )

    if reservoir_seed not in evaluation_seeds:
        raise ValueError(
            "BO_RESERVOIR_SEED must be included in BO_EVALUATION_SEEDS so the "
            "final reservoir is one that was evaluated during optimization"
        )

    return reservoir_seed, evaluation_seeds


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

_SEARCH_SPACE_FIELDS = (
    ("reservoir_size", ("N_res",), "N_res"),
    ("sparsity", ("p",), "p"),
    ("spectral_radius", (), "spectral_radius"),
    ("leak_rate", ("leaky_coefficient",), "leaky_coefficient"),
    ("input_scaling", (), "input_scaling"),
    ("regularization", (), "regularization"),
    ("washout", (), "washout"),
)


def _dimension_from_spec(config_key: str, internal_name: str, spec: Any):
    if not isinstance(spec, (tuple, list)) or len(spec) != 4:
        raise ValueError(
            f"BO_SEARCH_SPACE['{config_key}'] must be "
            "(lower, upper, 'int'|'float', log_scale)"
        )

    lower, upper, value_type, log_scale = spec
    value_type = str(value_type).strip().lower()

    if not isinstance(log_scale, (bool, np.bool_)):
        raise ValueError(
            f"BO_SEARCH_SPACE['{config_key}'] log_scale must be True or False"
        )

    try:
        lower_float = float(lower)
        upper_float = float(upper)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"BO_SEARCH_SPACE['{config_key}'] bounds must be numeric"
        ) from exc

    if not np.isfinite(lower_float) or not np.isfinite(upper_float):
        raise ValueError(
            f"BO_SEARCH_SPACE['{config_key}'] bounds must be finite"
        )
    if lower_float >= upper_float:
        raise ValueError(
            f"BO_SEARCH_SPACE['{config_key}'] lower bound must be below upper bound"
        )
    if log_scale and lower_float <= 0:
        raise ValueError(
            f"BO_SEARCH_SPACE['{config_key}'] log-scaled bounds must be positive"
        )

    prior = "log-uniform" if log_scale else "uniform"

    if value_type == "int":
        if not lower_float.is_integer() or not upper_float.is_integer():
            raise ValueError(
                f"BO_SEARCH_SPACE['{config_key}'] integer bounds must be integers"
            )
        return Integer(
            int(lower_float),
            int(upper_float),
            prior=prior,
            name=internal_name,
        )

    if value_type == "float":
        return Real(
            lower_float,
            upper_float,
            prior=prior,
            name=internal_name,
        )

    raise ValueError(
        f"BO_SEARCH_SPACE['{config_key}'] type must be 'int' or 'float'"
    )


def _get_configured_search_space(search_space: Mapping[str, Any]):
    dimensions = []

    for official_key, legacy_keys, internal_name in _SEARCH_SPACE_FIELDS:
        if official_key in search_space:
            spec = search_space[official_key]
        else:
            legacy_key = next(
                (name for name in legacy_keys if name in search_space),
                None,
            )
            if legacy_key is None:
                raise ValueError(
                    f"BO_SEARCH_SPACE is missing required key '{official_key}'"
                )
            warnings.warn(
                f"BO_SEARCH_SPACE key '{legacy_key}' is deprecated; use "
                f"'{official_key}' instead.",
                FutureWarning,
                stacklevel=2,
            )
            spec = search_space[legacy_key]

        dimensions.append(
            _dimension_from_spec(official_key, internal_name, spec)
        )

    return dimensions


def _get_legacy_search_space():
    warnings.warn(
        "Individual BO min/max settings are deprecated; define BO_SEARCH_SPACE.",
        FutureWarning,
        stacklevel=2,
    )

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

    return [
        Integer(n_min, n_max, name="N_res"),
        Real(p_min, p_max, name="p"),
        Real(rho_min, rho_max, name="spectral_radius"),
        Real(leak_min, leak_max, name="leaky_coefficient"),
        Real(scale_min, scale_max, name="input_scaling"),
        Real(ridge_min, ridge_max, prior="log-uniform", name="regularization"),
        Integer(washout_min, washout_max, name="washout"),
    ]


def _get_search_space(input_size: int):
    """
    Build the BO dimensions from the official BO_SEARCH_SPACE setting.

    input_size is retained for compatibility with existing callers.
    """
    del input_size

    if hasattr(config, "BO_SEARCH_SPACE"):
        search_space = getattr(config, "BO_SEARCH_SPACE")
        if not isinstance(search_space, Mapping):
            raise ValueError("BO_SEARCH_SPACE must be a mapping")
        return _get_configured_search_space(search_space)

    return _get_legacy_search_space()


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


def prediction_validation_spec(
    series: np.ndarray,
    *,
    series_is_training_portion: bool = False,
    heldout_length: int = 0,
) -> dict:
    """Lock non-overlapping recursive validation windows inside training data."""
    series = as_2d(series)
    n_total = len(series)
    if series_is_training_portion:
        n_final_train = n_total
        heldout_length = int(max(0, heldout_length))
    else:
        n_final_train = int(
            n_total * float(getattr(config, "TRAIN_RATIO", 0.70))
        )
        n_final_train = max(10, min(n_final_train, n_total - 1))
        heldout_length = n_total - n_final_train

    num_windows = int(
        getattr(config, "PREDICTION_VALIDATION_NUM_WINDOWS", 3)
    )
    window_length = int(
        getattr(config, "PREDICTION_VALIDATION_WINDOW_LENGTH", 8000)
    )
    if num_windows < 3:
        raise ValueError("PREDICTION_VALIDATION_NUM_WINDOWS must be at least 3")
    if window_length < 100:
        raise ValueError("PREDICTION_VALIDATION_WINDOW_LENGTH is too short")

    total_validation = num_windows * window_length
    if total_validation >= n_final_train - 100:
        raise ValueError(
            "Configured prediction-validation windows leave too little BO training data"
        )

    configured_starts = getattr(
        config, "PREDICTION_VALIDATION_WINDOW_STARTS", None
    )
    if configured_starts is None:
        first_start = n_final_train - total_validation
        starts = [
            first_start + index * window_length
            for index in range(num_windows)
        ]
    else:
        starts = [int(value) for value in configured_starts]
        if len(starts) != num_windows:
            raise ValueError(
                "PREDICTION_VALIDATION_WINDOW_STARTS length must equal "
                "PREDICTION_VALIDATION_NUM_WINDOWS"
            )
        expected = [
            starts[0] + index * window_length
            for index in range(num_windows)
        ]
        if starts != expected:
            raise ValueError(
                "Prediction validation windows must be ordered, contiguous and "
                "non-overlapping for one uninterrupted recursive rollout"
            )

    windows = []
    for index, start in enumerate(starts):
        end = start + window_length
        if start < 100 or end > n_final_train:
            raise ValueError(
                f"Prediction validation window {index} [{start}, {end}) is "
                "outside the training portion"
            )
        windows.append(
            {
                "window_index": index,
                "start": start,
                "end": end,
                "length": window_length,
                "segment": "prediction_validation",
            }
        )

    max_train_steps = _get_int_config(
        ["OPT_TRAIN_MAX_STEPS", "BO_TRAIN_MAX_STEPS"],
        50000,
    )
    training_end = starts[0]
    training_start = max(0, training_end - max_train_steps)
    train_part = series[training_start:training_end]
    validation_block = series[starts[0] : windows[-1]["end"]]

    if len(train_part) < 100:
        raise ValueError(
            f"Not enough BO training data before validation windows: {len(train_part)}"
        )

    return {
        "train": train_part,
        "validation": validation_block,
        "windows": windows,
        "training_start": training_start,
        "training_end": training_end,
        "final_training_end": n_final_train,
        "heldout_test_start": n_final_train,
        "heldout_test_end": n_final_train + heldout_length,
        "aggregation": str(
            getattr(
                config,
                "PREDICTION_VALIDATION_AGGREGATION",
                "mean_plus_max",
            )
        ),
        "test_data_used_for_selection": False,
        "index_semantics": "zero_based_half_open_[start,end)",
    }


def _prepare_optimizer_segments(series: np.ndarray):
    """Compatibility wrapper returning locked BO train/validation arrays."""
    spec = prediction_validation_spec(series)
    return spec["train"], spec["validation"]


def _validation_window_slices(validation_length: int) -> list[slice]:
    num_windows = int(
        getattr(config, "PREDICTION_VALIDATION_NUM_WINDOWS", 3)
    )
    if validation_length % num_windows != 0:
        raise ValueError(
            "Validation block length must be divisible by the number of windows"
        )
    length = validation_length // num_windows
    return [
        slice(index * length, (index + 1) * length)
        for index in range(num_windows)
    ]


# ============================================================
# Model helper
# ============================================================

def _make_model(
    params: dict,
    input_size: int,
    reservoir_seed: int | None = None,
) -> EchoStateNetwork:
    if reservoir_seed is None:
        reservoir_seed = getattr(
            config,
            "BO_RESERVOIR_SEED",
            getattr(config, "RANDOM_SEED", 42),
        )
    reservoir_seed = _coerce_seed(reservoir_seed, "reservoir_seed")

    return EchoStateNetwork(
        N_res=int(params["N_res"]),
        p=float(params["p"]),
        spectral_radius=float(params["spectral_radius"]),
        leaky_coefficient=float(params["leaky_coefficient"]),
        regularization=float(params["regularization"]),
        input_scaling=float(params.get("input_scaling", 0.5)),
        input_size=int(input_size),
        normalize_input=False,
        seed=reservoir_seed,
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


def _bad_seed_result(reason: str, reservoir_seed: int):
    score = 1_000_000.0
    return score, {
        "reservoir_seed": int(reservoir_seed),
        "score": score,
        "validation_score": score,
        "validation_nrmse": score,
        "validation_nrmse_x": score,
        "validation_std_ratio": score,
        "validation_mean_gap": score,
        "validation_penalty": score,
        "stable": False,
        "reason": str(reason),
    }


def _validation_peak_indices(x: np.ndarray, threshold: float) -> np.ndarray:
    """Return one peak index for each contiguous above-threshold episode."""
    x = np.asarray(x, dtype=float).reshape(-1)
    above = np.isfinite(x) & (x >= float(threshold))
    starts = np.flatnonzero(above & np.r_[True, ~above[:-1]])
    ends = np.flatnonzero(above & np.r_[~above[1:], True]) + 1
    if len(starts) != len(ends):
        return np.empty(0, dtype=int)
    peaks = [
        int(start + np.argmax(x[start:end]))
        for start, end in zip(starts, ends)
        if end > start
    ]
    return np.asarray(peaks, dtype=int)


def _relative_error(predicted: float, reference: float) -> float:
    if abs(reference) < 1e-12:
        return 0.0 if abs(predicted) < 1e-12 else 1.0
    return float(abs(predicted - reference) / abs(reference))


def _validation_window_metrics(
    pred_norm: np.ndarray,
    true_norm: np.ndarray,
    spike_threshold_norm: float,
) -> dict:
    pred_norm = as_2d(pred_norm)
    true_norm = as_2d(true_norm)
    pred_x = pred_norm[:, 0]
    true_x = true_norm[:, 0]

    finite = bool(
        np.all(np.isfinite(pred_norm)) and np.all(np.isfinite(true_norm))
    )
    if not finite:
        return {
            "stable": False,
            "nrmse": 1_000_000.0,
            "nrmse_x": 1_000_000.0,
            "spike_count_true": 0,
            "spike_count_pred": 0,
            "spike_frequency_true": 0.0,
            "spike_frequency_pred": 0.0,
            "spike_frequency_rel_error": 1_000_000.0,
            "mean_isi_true": 0.0,
            "mean_isi_pred": 0.0,
            "isi_rel_error": 1_000_000.0,
            "std_ratio": 1_000_000.0,
            "mean_gap": 1_000_000.0,
            "penalty": 1_000_000.0,
            "score": 1_000_000.0,
        }

    true_std = max(float(np.std(true_x)), 1e-12)
    pred_std = float(np.std(pred_x))
    std_ratio = pred_std / true_std
    mean_gap = abs(float(np.mean(pred_x) - np.mean(true_x))) / true_std

    true_peaks = _validation_peak_indices(true_x, spike_threshold_norm)
    pred_peaks = _validation_peak_indices(pred_x, spike_threshold_norm)
    length = max(1, len(true_x))
    sample_dt = (
        float(getattr(config, "HR_DT", 1.0))
        if str(getattr(config, "DATASET_MODE", "hr")).lower() == "hr"
        else 1.0
    )
    sample_dt = sample_dt if sample_dt > 0.0 else 1.0
    duration = max(sample_dt, length * sample_dt)
    true_frequency = float(len(true_peaks) / duration)
    pred_frequency = float(len(pred_peaks) / duration)
    frequency_error = _relative_error(pred_frequency, true_frequency)

    true_isi = (
        float(np.mean(np.diff(true_peaks)) * sample_dt)
        if len(true_peaks) >= 2
        else 0.0
    )
    pred_isi = (
        float(np.mean(np.diff(pred_peaks)) * sample_dt)
        if len(pred_peaks) >= 2
        else 0.0
    )
    if len(true_peaks) >= 2 and len(pred_peaks) >= 2:
        isi_error = _relative_error(pred_isi, true_isi)
    elif len(true_peaks) < 2 and len(pred_peaks) < 2:
        isi_error = 0.0
    else:
        isi_error = 1.0

    penalty = 0.0
    if std_ratio < 0.10:
        penalty += (0.10 - std_ratio) * 5.0
    if std_ratio > 10.0:
        penalty += min(1000.0, std_ratio - 10.0)
    if mean_gap > 1.0:
        penalty += mean_gap

    max_abs = float(np.max(np.abs(pred_norm)))
    if max_abs > 50.0:
        penalty += min(1000.0, max_abs - 50.0)

    x_error = nrmse(pred_x, true_x)
    all_error = nrmse(pred_norm, true_norm)
    score = (
        float(getattr(config, "PREDICTION_STATE_X_WEIGHT", 0.55)) * x_error
        + float(getattr(config, "PREDICTION_MULTISTATE_WEIGHT", 0.25))
        * all_error
        + float(
            getattr(config, "PREDICTION_SPIKE_FREQUENCY_WEIGHT", 1.0)
        )
        * frequency_error
        + float(getattr(config, "PREDICTION_SPIKE_INTERVAL_WEIGHT", 0.50))
        * isi_error
        + penalty
    )

    return {
        "stable": True,
        "nrmse": _safe_metric(all_error),
        "nrmse_x": _safe_metric(x_error),
        "spike_count_true": int(len(true_peaks)),
        "spike_count_pred": int(len(pred_peaks)),
        "spike_frequency_true": _safe_metric(true_frequency),
        "spike_frequency_pred": _safe_metric(pred_frequency),
        "spike_frequency_rel_error": _safe_metric(frequency_error),
        "spike_frequency_units": "inverse_time_unit",
        "inter_spike_interval_units": "time_unit",
        "mean_isi_true": _safe_metric(true_isi),
        "mean_isi_pred": _safe_metric(pred_isi),
        "isi_rel_error": _safe_metric(isi_error),
        "std_ratio": _safe_metric(std_ratio),
        "mean_gap": _safe_metric(mean_gap),
        "penalty": _safe_metric(penalty),
        "score": _safe_metric(score),
    }


def _aggregate_window_scores(scores: Sequence[float]) -> float:
    values = np.asarray([_safe_metric(value) for value in scores], dtype=float)
    aggregation = str(
        getattr(
            config,
            "PREDICTION_VALIDATION_AGGREGATION",
            "mean_plus_max",
        )
    ).strip().lower()
    if aggregation == "mean":
        return _safe_metric(np.mean(values))
    if aggregation == "max":
        return _safe_metric(np.max(values))
    if aggregation == "mean_plus_max":
        max_weight = float(
            getattr(config, "PREDICTION_VALIDATION_MAX_WEIGHT", 0.25)
        )
        return _safe_metric(np.mean(values) + max_weight * np.max(values))
    raise ValueError(
        "PREDICTION_VALIDATION_AGGREGATION must be mean, max or mean_plus_max"
    )


def _evaluate_params_for_seed(
    params: dict,
    train_norm: np.ndarray,
    val_norm: np.ndarray,
    input_size: int,
    reservoir_seed: int,
    spike_threshold_norm: float,
):
    """Evaluate one candidate recursively on all locked validation windows."""

    max_score = float(
        getattr(config, "PREDICTION_DIVERGENCE_PENALTY", 1_000_000.0)
    )
    max_abs_prediction = float(getattr(config, "BO_MAX_ABS_PREDICTION", 1e6))

    try:
        washout = resolve_washout(params.get("washout", 200), len(train_norm))
        esn = _make_model(
            params,
            input_size=input_size,
            reservoir_seed=reservoir_seed,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            esn.train(train_norm, washout=washout)
            eval_norm = np.vstack([train_norm, val_norm])
            pred_norm, _ = esn.predict(
                eval_norm,
                n_warmup=len(train_norm) - 1,
            )

        pred_norm = as_2d(pred_norm)
        n = min(len(pred_norm), len(val_norm))
        pred_norm = pred_norm[:n]
        val_eval = val_norm[:n]

        if n != len(val_norm):
            return _bad_seed_result(
                "incomplete_recursive_validation", reservoir_seed
            )
        if n <= 5:
            return _bad_seed_result("too_few_prediction_steps", reservoir_seed)
        if not np.all(np.isfinite(pred_norm)):
            return _bad_seed_result("non_finite_prediction", reservoir_seed)
        if np.max(np.abs(pred_norm)) > max_abs_prediction:
            return _bad_seed_result("prediction_exploded", reservoir_seed)

        window_metrics = []
        for window_index, window_slice in enumerate(
            _validation_window_slices(n)
        ):
            metrics = _validation_window_metrics(
                pred_norm[window_slice],
                val_eval[window_slice],
                spike_threshold_norm=spike_threshold_norm,
            )
            metrics["window_index"] = int(window_index)
            metrics["start_in_validation_block"] = int(window_slice.start)
            metrics["end_in_validation_block"] = int(window_slice.stop)
            window_metrics.append(metrics)

        divergent_count = sum(
            not bool(metrics["stable"]) for metrics in window_metrics
        )
        if divergent_count:
            return _bad_seed_result(
                f"divergent_validation_windows={divergent_count}",
                reservoir_seed,
            )

        score = min(
            _aggregate_window_scores(
                [metrics["score"] for metrics in window_metrics]
            ),
            max_score,
        )
        metric_mapping = {
            "validation_nrmse": "nrmse",
            "validation_nrmse_x": "nrmse_x",
            "validation_std_ratio": "std_ratio",
            "validation_mean_gap": "mean_gap",
            "validation_penalty": "penalty",
            "validation_spike_count_true": "spike_count_true",
            "validation_spike_count_pred": "spike_count_pred",
            "validation_spike_frequency_true": "spike_frequency_true",
            "validation_spike_frequency_pred": "spike_frequency_pred",
            "validation_spike_frequency_rel_error": (
                "spike_frequency_rel_error"
            ),
            "validation_mean_isi_true": "mean_isi_true",
            "validation_mean_isi_pred": "mean_isi_pred",
            "validation_isi_rel_error": "isi_rel_error",
        }
        metrics = {
            "reservoir_seed": int(reservoir_seed),
            "score": _safe_metric(score),
            "validation_score": _safe_metric(score),
            "validation_num_windows": len(window_metrics),
            "validation_window_length": int(
                len(val_eval) // len(window_metrics)
            ),
            "validation_divergent_window_count": int(divergent_count),
            "validation_spike_frequency_units": "inverse_time_unit",
            "validation_inter_spike_interval_units": "time_unit",
            "validation_aggregation": str(
                getattr(
                    config,
                    "PREDICTION_VALIDATION_AGGREGATION",
                    "mean_plus_max",
                )
            ),
            "validation_window_metrics_json": json.dumps(
                window_metrics, sort_keys=True
            ),
            "stable": True,
            "reason": "ok",
        }
        for output_name, source_name in metric_mapping.items():
            metrics[output_name] = _safe_metric(
                np.mean(
                    [
                        window[source_name]
                        for window in window_metrics
                    ]
                )
            )
        for window in window_metrics:
            index = window["window_index"]
            for name, value in window.items():
                if name == "window_index":
                    continue
                metrics[f"window_{index}_{name}"] = value

        return score, metrics

    except Exception as exc:
        return _bad_seed_result(
            f"exception: {type(exc).__name__}",
            reservoir_seed,
        )


_MEAN_VALIDATION_METRICS = (
    "validation_nrmse",
    "validation_nrmse_x",
    "validation_std_ratio",
    "validation_mean_gap",
    "validation_penalty",
    "validation_spike_count_true",
    "validation_spike_count_pred",
    "validation_spike_frequency_true",
    "validation_spike_frequency_pred",
    "validation_spike_frequency_rel_error",
    "validation_mean_isi_true",
    "validation_mean_isi_pred",
    "validation_isi_rel_error",
    "validation_divergent_window_count",
)


def _evaluate_params(
    params: dict,
    train: np.ndarray,
    val: np.ndarray,
    input_size: int,
    iteration: int,
    optimizer: str,
    best_score: float,
    evaluation_seeds: Sequence[int] | None = None,
    reservoir_seed: int | None = None,
    optimizer_seed: int | None = None,
):
    """
    Evaluate one BO candidate using the same configured reservoir seed set.

    The score sent to the optimizer is the arithmetic mean across seeds. This
    function never returns NaN or infinity; failed seed evaluations contribute
    the existing 1e6 stability penalty.
    """

    if evaluation_seeds is None:
        configured_seed, seeds = _get_reservoir_seed_config()
        primary_seed = configured_seed
    else:
        seeds = _validate_evaluation_seeds(evaluation_seeds)
        primary_seed = seeds[0] if reservoir_seed is None else _coerce_seed(
            reservoir_seed,
            "reservoir_seed",
        )
        if primary_seed not in seeds:
            raise ValueError("reservoir_seed must be included in evaluation_seeds")

    try:
        train_norm, val_norm, train_mean, train_std = _normalize_from_train(
            train, val
        )
        spike_threshold_norm = (
            float(getattr(config, "SPIKE_THRESHOLD", 0.0))
            - float(train_mean[0, 0])
        ) / float(train_std[0, 0])
        seed_results = [
            _evaluate_params_for_seed(
                params=params,
                train_norm=train_norm,
                val_norm=val_norm,
                input_size=input_size,
                reservoir_seed=seed,
                spike_threshold_norm=spike_threshold_norm,
            )[1]
            for seed in seeds
        ]
    except Exception as exc:
        seed_results = [
            _bad_seed_result(
                f"exception: {type(exc).__name__}",
                seed,
            )[1]
            for seed in seeds
        ]

    seed_scores = np.asarray(
        [_safe_metric(result.get("score")) for result in seed_results],
        dtype=float,
    )
    score = _safe_metric(np.mean(seed_scores))

    row = {
        **params,
        "iteration": int(iteration),
        "optimizer": str(optimizer),
        "score": score,
        "best_score": _safe_metric(min(best_score, score)),
        "validation_score": score,
        "validation_score_std": _safe_metric(np.std(seed_scores)),
        "validation_score_min": _safe_metric(np.min(seed_scores)),
        "validation_score_max": _safe_metric(np.max(seed_scores)),
        "stable": bool(all(bool(result.get("stable")) for result in seed_results)),
        "reason": "ok",
        "reservoir_seed": int(primary_seed),
        "evaluation_seeds": list(seeds),
        "evaluation_seed_count": len(seeds),
        "score_aggregation": "mean",
        "validation_aggregation": str(
            getattr(
                config,
                "PREDICTION_VALIDATION_AGGREGATION",
                "mean_plus_max",
            )
        ),
        "validation_num_windows": int(
            getattr(config, "PREDICTION_VALIDATION_NUM_WINDOWS", 3)
        ),
        "validation_spike_frequency_units": "inverse_time_unit",
        "validation_inter_spike_interval_units": "time_unit",
        "validation_window_length": int(
            len(val)
            // max(
                1,
                int(getattr(config, "PREDICTION_VALIDATION_NUM_WINDOWS", 3)),
            )
        ),
    }

    if optimizer_seed is not None:
        row["optimizer_random_seed"] = _coerce_seed(
            optimizer_seed,
            "optimizer_seed",
        )

    for metric_name in _MEAN_VALIDATION_METRICS:
        row[metric_name] = _safe_metric(
            np.mean(
                [
                    _safe_metric(result.get(metric_name))
                    for result in seed_results
                ]
            )
        )

    primary_result = seed_results[seeds.index(primary_seed)]
    if "validation_window_metrics_json" in primary_result:
        row["validation_window_metrics_json"] = primary_result[
            "validation_window_metrics_json"
        ]
    for key, value in primary_result.items():
        if key.startswith("window_"):
            row[key] = value

    failures = [
        f"seed_{seed}={result.get('reason', 'unknown')}"
        for seed, result in zip(seeds, seed_results)
        if not bool(result.get("stable"))
    ]
    if failures:
        row["reason"] = ";".join(failures)

    for seed, result in zip(seeds, seed_results):
        suffix = str(seed)
        row[f"validation_score_seed_{suffix}"] = _safe_metric(
            result.get("score")
        )
        row[f"stable_seed_{suffix}"] = bool(result.get("stable"))
        row[f"reason_seed_{suffix}"] = str(result.get("reason", "unknown"))

    return score, row


# ============================================================
# Main optimizer
# ============================================================

def optimize_hyperparameters(
    loader,
    neuron_id: int = 0,
    optimizer: str = "gp",
    *,
    selection_series: np.ndarray | None = None,
    heldout_length: int = 0,
) -> OptimizationResult:
    optimizer = str(optimizer).lower()

    if selection_series is None:
        series, series_name = get_model_series(loader, neuron_id)
        series = as_2d(series)
        validation_spec = prediction_validation_spec(series)
    else:
        series = as_2d(selection_series)
        series_name = "explicit_training_portion_only"
        validation_spec = prediction_validation_spec(
            series,
            series_is_training_portion=True,
            heldout_length=heldout_length,
        )

    input_size = int(series.shape[1])
    train_seg = validation_spec["train"]
    val_seg = validation_spec["validation"]

    n_calls = int(
        _get_official_config(
            "BO_N_CALLS",
            ("BO_CALLS", "N_CALLS", "N_BO_CALLS"),
            30,
        )
    )
    random_starts = int(
        _get_official_config(
            "BO_N_RANDOM_STARTS",
            ("BO_RANDOM_STARTS", "N_RANDOM_STARTS"),
            8,
        )
    )

    n_calls = max(1, int(n_calls))
    random_starts = max(1, min(int(random_starts), n_calls))

    dimensions = _get_search_space(input_size)
    reservoir_seed, evaluation_seeds = _get_reservoir_seed_config()

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

    optimizer_seed = random_seed + seed_offset
    if optimizer == "dummy":
        # scikit-optimize's DUMMY estimator resolves to None. Newer
        # scikit-learn releases attempt to inspect estimator tags before their
        # None check, so use the equivalent seeded Space.rvs random search.
        skopt_opt = _RandomSearchOptimizer(dimensions, optimizer_seed)
    else:
        skopt_opt = Optimizer(
            dimensions=dimensions,
            base_estimator=base_estimator,
            n_initial_points=random_starts,
            random_state=optimizer_seed,
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
    print(f"Reservoir seed: {reservoir_seed}")
    print(f"Evaluation seeds: {evaluation_seeds}")
    print(
        "Validation    : "
        f"{len(validation_spec['windows'])} locked recursive windows, "
        f"aggregation={validation_spec['aggregation']}"
    )
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
            evaluation_seeds=evaluation_seeds,
            reservoir_seed=reservoir_seed,
            optimizer_seed=optimizer_seed,
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
    best_params["reservoir_seed"] = int(reservoir_seed)
    best_params["evaluation_seeds"] = list(evaluation_seeds)
    best_params["evaluation_seed_count"] = len(evaluation_seeds)
    best_params["optimizer_random_seed"] = int(optimizer_seed)
    best_params["score_aggregation"] = "mean"
    best_params["validation_aggregation"] = validation_spec["aggregation"]
    best_params["validation_windows"] = validation_spec["windows"]
    best_params["validation_training_start"] = validation_spec["training_start"]
    best_params["validation_training_end"] = validation_spec["training_end"]
    best_params["final_training_end"] = validation_spec["final_training_end"]
    best_params["heldout_test_start"] = validation_spec["heldout_test_start"]
    best_params["heldout_test_end"] = validation_spec["heldout_test_end"]
    best_params["test_data_used_for_selection"] = False

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
        best_params["validation_score_std"] = _safe_metric(
            best_row.get("validation_score_std", 0.0)
        )
        best_params["validation_score_min"] = _safe_metric(
            best_row.get("validation_score_min", best_score)
        )
        best_params["validation_score_max"] = _safe_metric(
            best_row.get("validation_score_max", best_score)
        )
        for metric_name in _MEAN_VALIDATION_METRICS:
            best_params[metric_name] = _safe_metric(
                best_row.get(metric_name, 1_000_000.0)
            )
        best_params["validation_num_windows"] = int(
            best_row.get(
                "validation_num_windows",
                len(validation_spec["windows"]),
            )
        )
        best_params["validation_window_length"] = int(
            best_row.get(
                "validation_window_length",
                len(val_seg) // len(validation_spec["windows"]),
            )
        )
        if "validation_window_metrics_json" in best_row:
            best_params["validation_window_metrics"] = json.loads(
                best_row["validation_window_metrics_json"]
            )
    else:
        best_params["validation_nrmse"] = 1_000_000.0
        best_params["validation_nrmse_x"] = 1_000_000.0
        best_params["validation_std_ratio"] = 1_000_000.0
        best_params["validation_mean_gap"] = 1_000_000.0
        best_params["validation_penalty"] = 1_000_000.0
        best_params["validation_score_std"] = 0.0
        best_params["validation_score_min"] = _safe_metric(best_score)
        best_params["validation_score_max"] = _safe_metric(best_score)

    return OptimizationResult(
        best_params=best_params,
        best_score=float(best_score),
        history=history,
    )
