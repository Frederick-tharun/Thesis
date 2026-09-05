"""Frozen configuration for the parameter-aware cross-regime experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chapter2.esn_config import ESNModelConfig


BASE_COMMIT = "4dbafbafa552be6439b43a011dc60b9ee1afa9f5"
BRANCH = "chapter2-cross-regime-parameter-aware"
SEEDS = (42, 123, 456, 789, 2026)

REGULAR_CURRENTS = (1.67, 3.29, 3.50)
CHAOTIC_CURRENTS = (3.20, 3.34)
ALL_CURRENTS = (1.67, 3.20, 3.29, 3.34, 3.50)

SCENARIO_TRAINING_CURRENTS = {
    "regular_to_chaotic": REGULAR_CURRENTS,
    "chaotic_to_regular": CHAOTIC_CURRENTS,
    "mixed_shuffled": ALL_CURRENTS,
}
PRIMARY_CROSS_REGIME_CURRENTS = {
    "regular_to_chaotic": CHAOTIC_CURRENTS,
    "chaotic_to_regular": REGULAR_CURRENTS,
    "mixed_shuffled": (),
}

TRAINING_WASHOUT = 2_000
EFFECTIVE_TRAINING_BUDGET = 130_000
RAW_TRAINING_TRANSITIONS = {
    "regular_to_chaotic": {1.67: 45_334, 3.29: 45_333, 3.50: 45_333},
    "chaotic_to_regular": {3.20: 67_000, 3.34: 67_000},
    "mixed_shuffled": {current: 28_000 for current in ALL_CURRENTS},
}

MIXED_BLOCK_ORDER_RULE = (
    "np.random.default_rng(100000 + seed).permutation("
    "[1.67, 3.20, 3.29, 3.34, 3.50])"
)
MIXED_BLOCK_ORDERS = {
    42: (3.29, 3.34, 3.50, 1.67, 3.20),
    123: (3.34, 3.20, 3.29, 1.67, 3.50),
    456: (3.20, 3.50, 3.29, 1.67, 3.34),
    789: (1.67, 3.29, 3.34, 3.50, 3.20),
    2026: (3.50, 3.34, 3.29, 1.67, 3.20),
}

CONTINUOUS_SCHEDULES = {
    "regular_then_chaotic": (1.67, 3.29, 3.50, 3.20, 3.34),
    "chaotic_then_regular": (3.20, 3.34, 1.67, 3.29, 3.50),
    "alternating_mixed": (3.50, 3.34, 1.67, 3.20, 3.29),
}
CONTINUOUS_SAMPLES_PER_SEGMENT = 100_000
CONTINUOUS_SWITCH_INDICES = (100_000, 200_000, 300_000, 400_000)
CONTINUOUS_STATE_COUNT = 500_000
CONTINUOUS_WARMUP_TRANSITIONS = 2_000

SHORT_WINDOW_STARTS = (70_000, 80_000, 89_999)
WARMUP_TRANSITIONS = 2_000
SHORT_FORECAST_TRANSITIONS = 8_000
LONG_WARMUP_RANGE = (70_000, 72_000)
LONG_FORECAST_RANGE = (72_000, 99_999)

VALID_PREDICTION_THRESHOLD = 0.4
DIVERGENCE_THRESHOLD = 5.0
COLLAPSE_THRESHOLD = 0.05
REPRESENTATIVE_SEED = 42

EXPECTED_SHORT_RECORDS = 225
EXPECTED_LONG_RECORDS = 75
EXPECTED_CONTINUOUS_RECORDS = 45
EXPECTED_RECORDS = 345
EXPECTED_MODELS = 15
EXPECTED_SCHEDULE_DATASETS = 3
EXPECTED_BINARY_ARTIFACTS = 363

CHAPTER2_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = CHAPTER2_ROOT / "cross_regime_models"
RESULT_ROOT = CHAPTER2_ROOT / "cross_regime_results"
RAW_ARRAY_ROOT = RESULT_ROOT / "raw_arrays"
DATASET_ROOT = RESULT_ROOT / "datasets"
PILOT_ROOT = CHAPTER2_ROOT / "cross_regime_pilot"

FROZEN_HYPERPARAMETERS = {
    "reservoir_size": 100,
    "reservoir_connectivity": 0.08881598524963213,
    "input_scaling": 0.06402022818477646,
    "spectral_radius": 0.4118313967689876,
    "ridge_regularisation": 3.968208883661854e-10,
    "leak_rate": 0.9375840772954693,
    "bias_scaling": 0.1,
    "regularise_bias": False,
    "input_dimension": 4,
    "output_dimension": 3,
}


def model_config(seed: int) -> ESNModelConfig:
    """Return the frozen parameter-aware architecture for one paired seed."""
    return ESNModelConfig(seed=seed, **FROZEN_HYPERPARAMETERS)


def block_order(scenario: str, seed: int) -> tuple[float, ...]:
    """Return the frozen complete-trajectory order for a scenario and seed."""
    if scenario == "mixed_shuffled":
        return MIXED_BLOCK_ORDERS[seed]
    return tuple(sorted(SCENARIO_TRAINING_CURRENTS[scenario]))


def validate_configuration() -> None:
    """Fail closed if any prespecified experimental constant drifts."""
    if set(REGULAR_CURRENTS) & set(CHAOTIC_CURRENTS):
        raise ValueError("regular and chaotic regimes overlap")
    if set(REGULAR_CURRENTS + CHAOTIC_CURRENTS) != set(ALL_CURRENTS):
        raise ValueError("regime membership does not cover all fixed currents")
    for scenario, allocation in RAW_TRAINING_TRANSITIONS.items():
        effective = sum(length - TRAINING_WASHOUT for length in allocation.values())
        if effective != EFFECTIVE_TRAINING_BUDGET:
            raise ValueError(f"effective training budget mismatch for {scenario}")
        if max(allocation.values()) > 70_000:
            raise ValueError(f"training overlaps held-out transitions for {scenario}")
    for seed, expected in MIXED_BLOCK_ORDERS.items():
        generated = tuple(
            float(value)
            for value in np.random.default_rng(100_000 + seed).permutation(
                list(ALL_CURRENTS)
            )
        )
        if generated != expected:
            raise ValueError(f"mixed block order mismatch for seed {seed}")
    if EXPECTED_SHORT_RECORDS + EXPECTED_LONG_RECORDS + EXPECTED_CONTINUOUS_RECORDS != EXPECTED_RECORDS:
        raise ValueError("evaluation-record total mismatch")
    if EXPECTED_MODELS + EXPECTED_RECORDS + EXPECTED_SCHEDULE_DATASETS != EXPECTED_BINARY_ARTIFACTS:
        raise ValueError("binary-artifact total mismatch")


validate_configuration()
