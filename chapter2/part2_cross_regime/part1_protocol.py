"""Step 3A: record the exact Part-1 successful protocol as evidence (read-only).

Every value below is read directly from the frozen Part-1 source/results; none
is guessed. Writes only results/baseline/part1_protocol_snapshot.json.
"""
from __future__ import annotations

import json

from chapter2 import esn_config, esn_optimisation

from . import config


def snapshot() -> dict:
    selection = json.loads(
        (config.PROJECT_ROOT / "chapter2/optimisation_results/step7_selection.json").read_text()
    )
    pa = selection["models"]["parameter_aware"]
    return {
        "kind": "part1_protocol_snapshot",
        "source_files": [
            "chapter2/esn_config.py",
            "chapter2/esn_optimisation.py",
            "chapter2/esn_step8.py",
            "chapter2/optimisation_results/step7_selection.json",
        ],
        "search_space": {
            "reservoir_size": {"type": "categorical", "values": [100, 200, 300]},
            "reservoir_connectivity": {"type": "uniform", "low": 0.01, "high": 1.0},
            "input_scaling": {"type": "uniform", "low": 0.01, "high": 3.0},
            "spectral_radius": {"type": "uniform", "low": 0.01, "high": 3.0},
            "ridge_regularisation": {"type": "log-uniform", "low": 1.0e-10, "high": 1.0e-2},
            "leak_rate": {"type": "uniform", "low": 0.01, "high": 1.0},
        },
        "bias_scaling": 0.1,
        "regularise_bias": False,
        "search_settings": {
            "bayesian_calls_per_model": esn_optimisation.BAYESIAN_CALLS_PER_MODEL,
            "initial_random_calls": esn_optimisation.INITIAL_RANDOM_CALLS,
            "acquisition_function": esn_optimisation.ACQUISITION_FUNCTION,
            "search_seed_parameter_aware": esn_optimisation.SEARCH_SEEDS[esn_optimisation.PARAMETER_AWARE],
            "candidate_model_seed": esn_optimisation.CANDIDATE_MODEL_SEED,
            "base_estimator": esn_optimisation.OPTIMIZER_BASE_ESTIMATOR,
            "initial_point_generator": esn_optimisation.OPTIMIZER_INITIAL_POINT_GENERATOR,
        },
        "training_washout": esn_optimisation.TRAINING_WASHOUT,
        "fitting_transition_range": [0, 40_000],
        "validation_windows": [
            {
                "number": w.number,
                "warmup_range": [w.warmup.start, w.warmup.stop],
                "scored_range": [w.scored.start, w.scored.stop],
            }
            for w in esn_config.VALIDATION_WINDOWS
        ],
        "thresholds": {
            "valid_prediction_threshold": esn_optimisation.VALID_PREDICTION_THRESHOLD,
            "divergence_threshold": esn_optimisation.DIVERGENCE_THRESHOLD,
            "collapse_std_ratio_threshold": esn_optimisation.COLLAPSE_STD_RATIO_THRESHOLD,
            "nonfinite_failure_score": esn_optimisation.NONFINITE_FAILURE_SCORE,
        },
        "final_seeds": list(esn_config.FINAL_SEEDS),
        "top_candidate_count_for_confirmation": esn_optimisation.TOP_CANDIDATE_COUNT,
        "step7_selected_parameter_aware_hyperparameters": pa["best_configuration"],
        "step7_selected_parameter_aware_robust_aggregate": pa["best_robust_aggregate"],
        "original_training_currents": list(esn_config.TRAIN_CURRENTS),
        "original_unseen_currents": list(esn_config.UNSEEN_CURRENTS),
        "notes": (
            "The original Part-1 Step 7 trained one JOINT parameter-aware ESN "
            "on all three currents (1.67, 3.20, 3.50) together, objective = "
            "mean NRMSE over 9 rollouts (3 currents x 3 windows). Part 2 reuses "
            "this exact architecture, search space, washout, and validation-"
            "window shape, but trains SEPARATE regular-only and chaotic-only "
            "models, each over its own (larger) training-current list, with "
            "the objective generalised accordingly (see validation.source_objective)."
        ),
    }


def write() -> None:
    config.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    (config.BASELINE_DIR / "part1_protocol_snapshot.json").write_text(
        json.dumps(snapshot(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    write()
    print(f"wrote {config.BASELINE_DIR / 'part1_protocol_snapshot.json'}")
