"""Locked configuration for the Chapter 2 cross-regime improvement extension."""

from __future__ import annotations

from pathlib import Path

from chapter2.cross_regime_config import (
    ALL_CURRENTS,
    CHAOTIC_CURRENTS,
    COLLAPSE_THRESHOLD,
    CONTINUOUS_SCHEDULES,
    CONTINUOUS_SWITCH_INDICES,
    DIVERGENCE_THRESHOLD,
    MIXED_BLOCK_ORDERS,
    REGULAR_CURRENTS,
    SCENARIO_TRAINING_CURRENTS,
    SEEDS,
    VALID_PREDICTION_THRESHOLD,
    block_order,
)
from chapter2.esn_optimisation import (
    NONFINITE_FAILURE_SCORE,
    SEARCH_PARAMETER_ORDER,
    SEARCH_SPACE_DEFINITION,
)


STARTING_COMMIT = "cbad3fac7ffe6f8b0831a34814aed8193ba181e3"
EXPECTED_BRANCH = "chapter2-cross-regime-improvement"
ESN_SOURCE_SHA256 = "130c0f0f1753c7429a37bf14dbfd49de5bc8a0e04741893ef0b360126d8edcd3"

INPUT_DIMENSION = 4
OUTPUT_DIMENSION = 3
BIAS_SCALING = 0.1
REGULARISE_BIAS = False

FITTING_RANGE = (0, 40_000)
VALIDATION_RANGE = (40_000, 70_000)
VALIDATION_WINDOWS = (
    (1, (40_000, 42_000), (42_000, 50_000)),
    (2, (50_000, 52_000), (52_000, 60_000)),
    (3, (60_000, 62_000), (62_000, 70_000)),
)
TRAINING_WASHOUT = 2_000

BAYESIAN_CALLS = 40
INITIAL_RANDOM_CALLS = 10
TOP_CANDIDATE_COUNT = 5
CANDIDATE_MODEL_SEED = 42
OPTIMISER_SEEDS = {
    "regular_to_chaotic": 31_101,
    "chaotic_to_regular": 31_102,
    "mixed_shuffled": 31_103,
}

MATRIX_CODES = {
    ("regular_to_chaotic", "regular"): "RR",
    ("regular_to_chaotic", "chaotic"): "RC",
    ("chaotic_to_regular", "chaotic"): "CC",
    ("chaotic_to_regular", "regular"): "CR",
    ("mixed_shuffled", "regular"): "MR",
    ("mixed_shuffled", "chaotic"): "MC",
}
SCENARIO_LABELS = {
    "regular_to_chaotic": "Regular-trained",
    "chaotic_to_regular": "Chaotic-trained",
    "mixed_shuffled": "Mixed-shuffled-trained",
}

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
RESULT_ROOT = PACKAGE_ROOT / "results"
OPTIMISATION_ROOT = RESULT_ROOT / "optimisation"
MODEL_ROOT = RESULT_ROOT / "models"
RAW_ARRAY_ROOT = RESULT_ROOT / "raw_arrays"
FIGURE_ROOT = RESULT_ROOT / "figures"

BASELINE_ROOT = PROJECT_ROOT / "chapter2" / "cross_regime_results"
BASELINE_MODEL_ROOT = PROJECT_ROOT / "chapter2" / "cross_regime_models"
CORRECTION_ROOT = BASELINE_ROOT / "post_hoc_numerical_correction"


def regime_for_current(current: float) -> str:
    value = float(current)
    if value in REGULAR_CURRENTS:
        return "regular"
    if value in CHAOTIC_CURRENTS:
        return "chaotic"
    raise ValueError(f"current is outside the frozen five-current design: {current}")


def validate_config() -> None:
    if REGULAR_CURRENTS != (1.67, 3.29, 3.50):
        raise RuntimeError("regular regime definition drifted")
    if CHAOTIC_CURRENTS != (3.20, 3.34):
        raise RuntimeError("chaotic regime definition drifted")
    if set(REGULAR_CURRENTS).intersection(CHAOTIC_CURRENTS):
        raise RuntimeError("regime definitions overlap")
    if set(REGULAR_CURRENTS + CHAOTIC_CURRENTS) != set(ALL_CURRENTS):
        raise RuntimeError("regime definitions do not cover the five currents")
    if tuple(SEEDS) != (42, 123, 456, 789, 2026):
        raise RuntimeError("five-seed protocol drifted")
    if NONFINITE_FAILURE_SCORE != 1_000_000.0:
        raise RuntimeError("numerical-failure score drifted")
    if (VALID_PREDICTION_THRESHOLD, DIVERGENCE_THRESHOLD, COLLAPSE_THRESHOLD) != (
        0.4,
        5.0,
        0.05,
    ):
        raise RuntimeError("evaluation thresholds drifted")
    for seed, expected in MIXED_BLOCK_ORDERS.items():
        if block_order("mixed_shuffled", seed) != expected:
            raise RuntimeError(f"mixed block-order drift for seed {seed}")


validate_config()
