"""Shared, frozen configuration for the Part 2 rebuild.

Every Hindmarsh-Rose and ESN constant here is imported unchanged from the
frozen Part-1 modules. This module defines only new, Part-2-specific paths
and orchestration constants (data locations, candidate budget, gate values).
Nothing here redefines a Part-1 value; see ``part1_protocol.snapshot()`` for
the recorded evidence behind each reused constant.
"""
from __future__ import annotations

from pathlib import Path

from chapter2.config_ch2 import (
    DT,
    HR_PARAMETERS,
    INITIAL_STATE,
    INITIAL_TRANSIENT_STEPS,
    RETAINED_SAMPLES_PER_CURRENT,
)
from chapter2.esn_config import (
    FINAL_SEEDS,
    VALIDATION_WINDOWS,
)
from chapter2.esn_optimisation import (
    ACQUISITION_FUNCTION,
    BAYESIAN_CALLS_PER_MODEL,
    CANDIDATE_MODEL_SEED,
    COLLAPSE_STD_RATIO_THRESHOLD,
    DIVERGENCE_THRESHOLD,
    INITIAL_RANDOM_CALLS,
    NONFINITE_FAILURE_SCORE,
    PARAMETER_AWARE,
    SEARCH_PARAMETER_ORDER,
    TRAINING_WASHOUT,
    TOP_CANDIDATE_COUNT,
    VALID_PREDICTION_THRESHOLD,
    search_dimensions,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
RESULTS_DIR = PACKAGE_ROOT / "results"
DATA_DIR = PACKAGE_ROOT / "data"
DATA_SOURCE_REGULAR = DATA_DIR / "source_regular"
DATA_SOURCE_CHAOTIC = DATA_DIR / "source_chaotic"
DATA_FINAL_TEST_LOCKED = DATA_DIR / "final_test_locked"

DESIGN_DIR = RESULTS_DIR / "design"
DATA_MANIFEST_DIR = RESULTS_DIR / "data_manifest"
BASELINE_DIR = RESULTS_DIR / "baseline"
OPTIMISATION_DIR = RESULTS_DIR / "optimisation"
VALIDATION_DIR = RESULTS_DIR / "validation"

FROZEN_DESIGN_PATH = DESIGN_DIR / "frozen_design.json"
SOURCE_VALIDATION_PROTOCOL_PATH = DESIGN_DIR / "source_validation_protocol.json"
DATA_MANIFEST_PATH = DATA_MANIFEST_DIR / "data_manifest.json"

# Fitting range reused unchanged from Part-1 Step 7 (esn_config.FITTING_STOP=40000).
FITTING_TRANSITIONS_STOP = 40_000

MODEL_TYPE = PARAMETER_AWARE  # Part 2 only ever uses the parameter-aware ESN.

# Step 4D candidate budget, reused from the Part-1-style search: 1 anchor +
# 15 deterministic space-filling (LHS) + 32 GP-EI proposals = 48 unique
# candidates per family.
N_ANCHOR_CANDIDATES = 1
N_LHS_CANDIDATES = 15
N_GP_CANDIDATES = 32
N_CANDIDATES_PER_FAMILY = N_ANCHOR_CANDIDATES + N_LHS_CANDIDATES + N_GP_CANDIDATES
assert N_CANDIDATES_PER_FAMILY == 48

# Deterministic family-specific seeds for the LHS design and the GP-EI search,
# distinct from Part-1's own (2026, 2027) search seeds and from the reservoir
# seeds below, so Part-2 search streams never collide with Part-1 evidence.
FAMILY_SEARCH_SEEDS = {
    "regular": {"lhs": 51101, "gp": 61101},
    "chaotic": {"lhs": 51102, "gp": 61102},
}

CONFIRMATION_SEEDS = FINAL_SEEDS  # reuse Part-1's registered seed set unchanged
TOP_K_FOR_CONFIRMATION = 10

__all__ = [
    "DT", "HR_PARAMETERS", "INITIAL_STATE", "INITIAL_TRANSIENT_STEPS",
    "RETAINED_SAMPLES_PER_CURRENT", "FINAL_SEEDS", "VALIDATION_WINDOWS",
    "ACQUISITION_FUNCTION", "BAYESIAN_CALLS_PER_MODEL", "CANDIDATE_MODEL_SEED",
    "COLLAPSE_STD_RATIO_THRESHOLD", "DIVERGENCE_THRESHOLD",
    "INITIAL_RANDOM_CALLS", "NONFINITE_FAILURE_SCORE", "PARAMETER_AWARE",
    "SEARCH_PARAMETER_ORDER", "TRAINING_WASHOUT", "TOP_CANDIDATE_COUNT",
    "VALID_PREDICTION_THRESHOLD", "search_dimensions",
    "PACKAGE_ROOT", "PROJECT_ROOT", "RESULTS_DIR", "DATA_DIR",
    "DATA_SOURCE_REGULAR", "DATA_SOURCE_CHAOTIC", "DATA_FINAL_TEST_LOCKED",
    "DESIGN_DIR", "DATA_MANIFEST_DIR", "BASELINE_DIR", "OPTIMISATION_DIR",
    "VALIDATION_DIR", "FROZEN_DESIGN_PATH", "SOURCE_VALIDATION_PROTOCOL_PATH",
    "DATA_MANIFEST_PATH", "FITTING_TRANSITIONS_STOP", "MODEL_TYPE",
    "N_ANCHOR_CANDIDATES", "N_LHS_CANDIDATES", "N_GP_CANDIDATES",
    "N_CANDIDATES_PER_FAMILY", "FAMILY_SEARCH_SEEDS", "CONFIRMATION_SEEDS",
    "TOP_K_FOR_CONFIRMATION",
]
