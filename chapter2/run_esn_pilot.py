"""Run the single small real-data ESN pilot authorized for Chapter 2.

This is a mechanics and plumbing check, not hyperparameter selection. It uses
only fitting data and one shortened scored prefix of a predefined validation
window from the permitted training currents.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .esn_config import (
        CHAPTER2_ROOT,
        TRAIN_CURRENTS,
        ESNModelConfig,
    )
    from .esn_data import (
        fit_training_scalers,
        load_fixed_trajectory,
        load_optimisation_data,
        scale_one_step_pairs,
    )
    from .esn_metrics import evaluate_rollout
    from .esn_model import EchoStateNetwork, TrainingSequence
except ImportError:  # Support direct execution from the chapter2 directory.
    from esn_config import CHAPTER2_ROOT, TRAIN_CURRENTS, ESNModelConfig
    from esn_data import (
        fit_training_scalers,
        load_fixed_trajectory,
        load_optimisation_data,
        scale_one_step_pairs,
    )
    from esn_metrics import evaluate_rollout
    from esn_model import EchoStateNetwork, TrainingSequence


PILOT_SCHEMA = "chapter2_step5_real_data_pilot_v1"
PILOT_FIT_TRANSITIONS_PER_CURRENT = 5_000
PILOT_TRAINING_WASHOUT = 100
PILOT_VALIDATION_CURRENT = 3.20
PILOT_VALIDATION_WINDOW = 1
PILOT_SCORED_TRANSITIONS = 1_000
PILOT_VALID_PREDICTION_THRESHOLD = 0.4
PILOT_DIVERGENCE_THRESHOLD = 5.0
DEFAULT_OUTPUT_PATH = (
    CHAPTER2_ROOT / "pilot_results" / "step5_real_data_pilot.json"
)
PILOT_MODEL_CONFIG = ESNModelConfig(
    reservoir_size=30,
    spectral_radius=0.8,
    leak_rate=0.5,
    input_scaling=0.3,
    bias_scaling=0.1,
    reservoir_connectivity=0.2,
    ridge_regularisation=1.0e-6,
    seed=42,
    input_dimension=4,
    output_dimension=3,
)


def run_pilot() -> dict[str, Any]:
    """Train and evaluate exactly one permitted small real-data pilot."""
    trajectories = tuple(
        load_fixed_trajectory(current) for current in TRAIN_CURRENTS
    )
    scalers = fit_training_scalers(trajectories)
    optimisation_data = load_optimisation_data()

    training_sequences: list[TrainingSequence] = []
    for prepared in optimisation_data:
        scaled = scale_one_step_pairs(prepared.fitting, scalers)
        training_sequences.append(
            TrainingSequence(
                scaled.inputs[:PILOT_FIT_TRANSITIONS_PER_CURRENT],
                scaled.targets[:PILOT_FIT_TRANSITIONS_PER_CURRENT],
            )
        )

    model = EchoStateNetwork(PILOT_MODEL_CONFIG)
    model.fit(tuple(training_sequences), washout=PILOT_TRAINING_WASHOUT)

    selected = next(
        item
        for item in optimisation_data
        if item.current == PILOT_VALIDATION_CURRENT
    )
    window = next(
        view
        for view in selected.validation_windows
        if view.definition.number == PILOT_VALIDATION_WINDOW
    )
    warmup = scale_one_step_pairs(window.warmup, scalers)
    scored = scale_one_step_pairs(window.scored, scalers)
    if not np.array_equal(warmup.targets[-1], scored.inputs[0, :3]):
        raise RuntimeError(
            "pilot warm-up target and first autonomous input state are misaligned"
        )

    predictions_scaled = model.autonomous_rollout(
        warmup.targets[-1],
        current_values=scored.inputs[:PILOT_SCORED_TRANSITIONS, 3],
        warmup_inputs=warmup.inputs,
        reset=True,
    )
    targets_scaled = scored.targets[:PILOT_SCORED_TRANSITIONS]
    predictions = scalers.inverse_states(predictions_scaled)
    targets = scalers.inverse_states(targets_scaled)
    metrics = evaluate_rollout(
        predictions,
        targets,
        normalisation_scale=scalers.state.scale,
        dt=0.01,
        valid_prediction_threshold=PILOT_VALID_PREDICTION_THRESHOLD,
        divergence_threshold=PILOT_DIVERGENCE_THRESHOLD,
    )

    return {
        "schema": PILOT_SCHEMA,
        "purpose": (
            "Small real-data mechanics pilot only; no hyperparameter selection "
            "or benchmark evaluation."
        ),
        "scientific_claim": (
            "This pilot checks pipeline operation only and does not establish "
            "biological prediction accuracy or generalisation."
        ),
        "model_config": asdict(PILOT_MODEL_CONFIG),
        "model_config_is_selected_hyperparameters": False,
        "data_scope": {
            "training_currents": list(TRAIN_CURRENTS),
            "fitting_transition_range_used_per_current": [
                0,
                PILOT_FIT_TRANSITIONS_PER_CURRENT,
            ],
            "fitting_transitions_per_current": (
                PILOT_FIT_TRANSITIONS_PER_CURRENT
            ),
            "total_fitting_transitions": (
                len(TRAIN_CURRENTS) * PILOT_FIT_TRANSITIONS_PER_CURRENT
            ),
            "training_washout_per_current": PILOT_TRAINING_WASHOUT,
            "validation_current": PILOT_VALIDATION_CURRENT,
            "validation_window": PILOT_VALIDATION_WINDOW,
            "warmup_transition_range": [
                window.warmup.transition_indices[0].item(),
                window.warmup.transition_indices[-1].item() + 1,
            ],
            "warmup_transitions": len(window.warmup),
            "scored_transition_range": [
                window.scored.transition_indices[0].item(),
                (
                    window.scored.transition_indices[0].item()
                    + PILOT_SCORED_TRANSITIONS
                ),
            ],
            "scored_transitions": PILOT_SCORED_TRANSITIONS,
            "held_out_loaded": False,
            "unseen_current_loaded": False,
            "continuous_benchmark_loaded": False,
        },
        "scaling": {
            "fitted_from_all_training_current_fitting_inputs": True,
            "state_mean": scalers.state.mean.tolist(),
            "state_scale": scalers.state.scale.tolist(),
            "current_mean": scalers.current.mean.tolist(),
            "current_scale": scalers.current.scale.tolist(),
            "metric_normalisation": (
                "training-fitted state standard deviations; validation "
                "statistics were not fitted"
            ),
        },
        "alignment": {
            "prediction_row_zero": "state at transition target 42001",
            "target_row_zero": "state at transition target 42001",
            "supplied_current_is_predicted": False,
        },
        "metric_thresholds_are_final_selected_values": False,
        "metrics": metrics.to_dict(),
    }


def write_result(result: dict[str, Any], path: str | Path) -> Path:
    """Write one deterministic, strict-JSON pilot record."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the one authorized small Chapter 2 real-data ESN pilot."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Pilot result JSON path.",
    )
    args = parser.parse_args()
    result = run_pilot()
    destination = write_result(result, args.output)
    print(f"Wrote {destination}")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
