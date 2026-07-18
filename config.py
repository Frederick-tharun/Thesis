import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "DRG3_MdFoF.csv")
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs")
OUTPUT_DIR = OUTPUT_ROOT
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------
# data
# -------------------------------------------------------------------

SAMPLING_INTERVAL = 0.25

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BASELINE_PERCENTILE = 10
SPIKE_THRESHOLD_STD = 1.0
MIN_SPIKE_DISTANCE = 20
NORMALIZATION_METHOD = "zscore"
EPS = 1e-8

DATASET_MODE = "real"   # changed automatically by main.py: "real" or "hr"

# -------------------------------------------------------------------
# ESN defaults
# -------------------------------------------------------------------

DEFAULT_ESN_PARAMS = {
    "reservoir_size": 600,
    "spectral_radius": 0.90,
    "leak_rate": 0.35,
    "input_scaling": 0.50,
    "regularization": 1e-6,
    "sparsity": 0.10,
    "washout": 200,
    "noise_std": 0.0,
}

RESERVOIR_BIAS_SCALE = 0.1
WINDOW_SIZE = 20
RANDOM_SEED = 42

# -------------------------------------------------------------------
# optimizer settings
# -------------------------------------------------------------------

OPTIMIZERS_TO_COMPARE = ["gp", "dummy", "forest", "gbrt"]

BO_N_CALLS = 30
BO_N_RANDOM_STARTS = 8
BO_RESERVOIR_SEED = 42
BO_EVALUATION_SEEDS = [42]

# Validation-only recursive model selection. The three non-overlapping windows
# are drawn from the 70% training portion and are never allowed to touch the
# final held-out 45,000-step test trajectory.
PREDICTION_VALIDATION_NUM_WINDOWS = 3
PREDICTION_VALIDATION_WINDOW_LENGTH = 8000
PREDICTION_VALIDATION_WINDOW_STARTS = None
PREDICTION_VALIDATION_AGGREGATION = "mean_plus_max"
PREDICTION_VALIDATION_MAX_WEIGHT = 0.25
PREDICTION_STATE_X_WEIGHT = 0.55
PREDICTION_MULTISTATE_WEIGHT = 0.25
PREDICTION_SPIKE_FREQUENCY_WEIGHT = 1.0
PREDICTION_SPIKE_INTERVAL_WEIGHT = 0.50
PREDICTION_DIVERGENCE_PENALTY = 1_000_000.0

# Locked final-test quality gates. These are broad scientific acceptability
# limits, not optimizer targets, and are evaluated only after selected_model.json
# has been written.
PERIODIC_SPIKING_MAX_TEST_NRMSE_X = 0.20
PERIODIC_SPIKING_MAX_SPIKE_FREQUENCY_REL_ERROR = 0.10

# Final Chapter 1 model/control provenance.
CONTROL_MODEL_SOURCE = "validation_selected"
PYRAGAS_SIGNS = [-1]
FINAL_HR_REGIMES = [
    "periodic_spiking",
    "periodic_bursting",
    "chaotic_bursting",
]

BO_SEARCH_SPACE = {
    "reservoir_size":  (250, 800, "int", False),
    "spectral_radius": (0.50, 1.20, "float", False),
    "leak_rate":       (0.10, 0.80, "float", False),
    "input_scaling":   (0.05, 1.00, "float", False),
    "regularization":  (1e-10, 1e-3, "float", True),
    "sparsity":        (0.02, 0.20, "float", False),
    "washout":         (50, 500, "int", False),
}
# Spectral radii above one are intentional empirical candidates. Autonomous
# stability is checked explicitly by the BO objective; rho < 1 is sufficient
# but not necessary for every leaky ESN.

TS_FOLDS = 3
TS_VAL_LEN = 80

# -------------------------------------------------------------------
# Hindmarsh-Rose settings
# -------------------------------------------------------------------

HR_DT = 0.01
HR_TOTAL_STEPS = 150000
HR_TRANSIENT = 5000

# Choose one:
# "periodic_spiking"
# "periodic_bursting"
# "chaotic_bursting"
HR_MODE = "chaotic_bursting"

HR_PARAMETER_SETS = {
    "periodic_spiking": {
        "a": 1.0,
        "b": 3.0,
        "c": 1.0,
        "d": 5.0,
        "r": 0.006,
        "s": 4.0,
        "xr": -1.6,
        "I": 2.5,
        "x0": [0.1, 0.0, 0.0],
    },

    "periodic_bursting": {
        "a": 1.0,
        "b": 3.0,
        "c": 1.0,
        "d": 5.0,
        "r": 0.003,
        "s": 4.0,
        "xr": -1.6,
        "I": 3.0,
        "x0": [0.1, 0.0, 0.0],
    },

    "chaotic_bursting": {
        "a": 1.0,
        "b": 3.0,
        "c": 1.0,
        "d": 5.0,
        "r": 0.006,
        "s": 4.0,
        "xr": -1.6,
        "I": 3.25,
        "x0": [-1.0, -3.0, 3.0],
    },
}


_hr = HR_PARAMETER_SETS[HR_MODE]
HR_A = _hr["a"]
HR_B = _hr["b"]
HR_C = _hr["c"]
HR_D = _hr["d"]
HR_R = _hr["r"]
HR_S = _hr["s"]
HR_XR = _hr["xr"]
HR_I = _hr["I"]

# ============================================================
# Clean output organization
# ============================================================

# True = each regime folder is cleaned before a new run.
# This prevents old confusing files from mixing with new files.
CLEAR_OUTPUT_FOLDER_EACH_RUN = False

# Optional aliases for compatibility with different loader versions.
HR_REGIME = HR_MODE
HR_DYNAMICS_MODE = HR_MODE
HINDMARSH_ROSE_MODE = HR_MODE

# ============================================================
# Plot settings
# ============================================================

SPIKE_THRESHOLD = 1.0
SPIKE_TOLERANCE_STEPS = 5

# For recursive zoom plot.
# 1200 steps is clearer than only 120 for long HR data.
ZOOM_STEPS = 1200

# ============================================================
# Linear-feedback control settings
# ============================================================

# Main control intensity sweep for the ESN digital twin.
CONTROL_LINEAR_K_SWEEP = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50]

# Fraction of the test horizon after which linear feedback starts.
# Example: 0.20 means first 20% of the test segment is free-run,
# then control is switched on.
CONTROL_START_FRAC = 0.20

# Final Chapter 1 control target: the median data-derived
# empirical quiet-state reference from training. "zero" and "mean" remain
# compatibility-only exploratory modes and are not used by the final pipeline.
CONTROL_TARGET_MODE = "rest_state_from_quiet_training_data"

# Split the post-control horizon: tune on validation, report once on test.
CONTROL_VALIDATION_FRAC = 0.50
CONTROL_TEST_FRAC = 0.50

# Settling-time rule:
# error norm must stay below CONTROL_SETTLING_TOL for
# CONTROL_SETTLING_HOLD_STEPS consecutive samples.
CONTROL_SETTLING_TOL = 0.15
CONTROL_SETTLING_HOLD_STEPS = 100

# Names consumed by control_experiment.py.
CONTROL_SETTLING_TOLERANCE = CONTROL_SETTLING_TOL
CONTROL_SETTLING_CONSECUTIVE = CONTROL_SETTLING_HOLD_STEPS

CONTROL_FINITE_S = 0.8
PYRAGAS_DELAY = 20
PYRAGAS_SIGN = -1
PYRAGAS_HISTORY_SIGNAL = "raw_readout"

# Optional clamp on corrected ESN input after control is applied.
# Set to None to disable.
# Examples:
#   None
#   5.0
#   (-5.0, 5.0)
CONTROL_INPUT_CLIP = None

# Flag a controlled rollout as diverged if any state magnitude exceeds this.
CONTROL_DIVERGENCE_ABS_LIMIT = 20.0

# ============================================================
# Automatic K selection settings
# ============================================================

# Used when running:
# python main.py --dataset hr --hr-mode periodic_bursting --control --auto-control-k

CONTROL_AUTO_K_MIN = 0.05
CONTROL_AUTO_K_MAX = 2.00
CONTROL_AUTO_K_NUM = 25
CONTROL_AUTO_K_REFINE_NUM = 15
CONTROL_AUTO_K_REFINE_WIDTH_FRAC = 0.15

# K-selection score:
# lower score = better K
#
# score = corrected_feedback_input_target_rmse_state
#       + CONTROL_SCORE_ENERGY_WEIGHT * control_effort_mean_sq
#       + CONTROL_SCORE_SETTLING_WEIGHT * evaluation_time_to_tolerance
#       - CONTROL_SCORE_SPIKE_WEIGHT * spike_reduction_percent / 100

CONTROL_SCORE_ENERGY_WEIGHT = 0.01
CONTROL_SCORE_SETTLING_WEIGHT = 0.001
CONTROL_SCORE_SPIKE_WEIGHT = 0.0

# ============================================================
# Pyragas periodic-orbit selection settings
# ============================================================


# Validation metrics use the explicit post-control validation window. Held-out
# test metrics use the controller-test window without an extra discarded transient.
PYRAGAS_MIN_EVALUATION_PEAKS = 6

# Reject flat or strongly amplified trajectories.
PYRAGAS_MIN_AMPLITUDE_RATIO = 0.20
PYRAGAS_MIN_STD_RATIO = 0.15

PYRAGAS_PREFERRED_AMPLITUDE_MIN = 0.80
PYRAGAS_PREFERRED_AMPLITUDE_MAX = 1.30
PYRAGAS_PREFERRED_STD_MIN = 0.50
PYRAGAS_PREFERRED_STD_MAX = 1.50

# Periodicity requirements.
PYRAGAS_TARGET_INTERVAL_CV = 0.05
PYRAGAS_TARGET_PERIODICITY_NORM = 0.15
PYRAGAS_TARGET_DELAY_MISMATCH = 0.20

# Current metric names used by control_experiment.py. Keep the aliases above
# for compatibility with older result-analysis scripts.
PYRAGAS_MIN_EVALUATION_CYCLES = 3
PYRAGAS_TARGET_RHYTHM_CV = PYRAGAS_TARGET_INTERVAL_CV
PYRAGAS_TARGET_CYCLE_WINDOW_COVERAGE = 0.50
PYRAGAS_MAX_EMPIRICAL_RECURRENCE_ERROR_NORM = PYRAGAS_TARGET_PERIODICITY_NORM
PYRAGAS_MIN_EMPIRICAL_RECURRENCE_CORRELATION = 0.65

# Require activity throughout the evaluation interval.
PYRAGAS_TARGET_PEAK_WINDOW_COVERAGE = 0.75
PYRAGAS_MAX_PEAK_AMPLITUDE_CV = 0.25
PYRAGAS_MAX_WINDOW_AMPLITUDE_CV = 0.30
PYRAGAS_MAX_DRIFT_RATIO = 0.20
PYRAGAS_MIN_TAIL_ACTIVITY_RATIO = 0.50

# The final two delayed trajectory segments should form the same orbit.
PYRAGAS_MAX_TAIL_CLOSURE_ERROR_NORM = 0.20
PYRAGAS_MAX_EMPIRICAL_TAIL_CLOSURE_ERROR_NORM = PYRAGAS_MAX_TAIL_CLOSURE_ERROR_NORM

# Classical Pyragas feedback should become small after stabilization.
PYRAGAS_TARGET_NONINVASIVENESS = 0.10
PYRAGAS_TARGET_CONTROL_DECAY = 0.50

PYRAGAS_MISSING_INTERVAL_CV_PENALTY = 2.0

# Selection-score weights consumed by control_experiment._selection_score.
PYRAGAS_SCORE_FEW_SPIKES_WEIGHT = 30.0
PYRAGAS_SCORE_FEW_CYCLES_WEIGHT = 30.0
PYRAGAS_SCORE_FLAT_AMPLITUDE_WEIGHT = 25.0
PYRAGAS_SCORE_FLAT_STD_WEIGHT = 10.0
PYRAGAS_SCORE_AMPLITUDE_RANGE_WEIGHT = 8.0
PYRAGAS_SCORE_STD_RANGE_WEIGHT = 3.0

PYRAGAS_SCORE_INTERVAL_CV_WEIGHT = 8.0
PYRAGAS_SCORE_INTERVAL_CV_EXCESS_WEIGHT = 15.0
PYRAGAS_SCORE_PERIODICITY_WEIGHT = 10.0
PYRAGAS_SCORE_PERIODICITY_EXCESS_WEIGHT = 20.0
PYRAGAS_SCORE_EMPIRICAL_CORRELATION_WEIGHT = 10.0
PYRAGAS_SCORE_PEAK_COVERAGE_WEIGHT = 15.0
PYRAGAS_SCORE_CYCLE_COVERAGE_WEIGHT = 15.0
PYRAGAS_SCORE_PEAK_AMPLITUDE_WEIGHT = 6.0
PYRAGAS_SCORE_WINDOW_AMPLITUDE_WEIGHT = 4.0
PYRAGAS_SCORE_DRIFT_WEIGHT = 20.0
PYRAGAS_SCORE_TAIL_ACTIVITY_WEIGHT = 20.0
PYRAGAS_SCORE_TAIL_CLOSURE_WEIGHT = 15.0
PYRAGAS_SCORE_QUALITY_ISSUE_WEIGHT = 25.0
PYRAGAS_SCORE_TOO_MANY_SPIKES_WEIGHT = 2.0
PYRAGAS_SCORE_ENERGY_WEIGHT = 0.05
PYRAGAS_SCORE_MAX_CONTROL_WEIGHT = 0.01
PYRAGAS_SCORE_K_WEIGHT = 0.02

# Legacy diagnostic weights retained for old notebooks. Delay mismatch,
# noninvasiveness, and control decay are reported diagnostics, not selection terms.
PYRAGAS_SCORE_DELAY_MISMATCH_WEIGHT = 1.0
PYRAGAS_SCORE_NONINVASIVENESS_WEIGHT = 1.0
PYRAGAS_SCORE_CONTROL_DECAY_WEIGHT = 0.5

# Suggested coarse search only; CLI arguments can override these.
PYRAGAS_RECOMMENDED_K_MIN = 0.01
PYRAGAS_RECOMMENDED_K_MAX = 0.15
PYRAGAS_RECOMMENDED_K_NUM = 30
PYRAGAS_RECOMMENDED_DELAYS = [10, 20, 40, 80, 160, 320]
