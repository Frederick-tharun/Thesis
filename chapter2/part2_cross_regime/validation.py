"""Temporal source validation, LOCO, and the absolute source-quality gate.

Generalises three Part-1 (chapter2.esn_optimisation) helpers -- which are
hard-locked to exactly Part-1's three training currents and nine validation
cases -- to Part 2's larger, per-family training-current lists, while
importing every other Part-1 primitive (EchoStateNetwork, model_config,
_recursive_case_rollout, aggregate_rollouts, evaluate_rollout,
_fit_scalers_from_permitted_views, TrainingSequence, ValidationCase,
ModelOptimisationData, SourceCandidateEvaluator) is generic and reused.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Mapping, Sequence

import numpy as np

from chapter2.esn_data import (
    PreparedOptimisationTrajectory,
    StateCurrentScalers,
    scale_one_step_pairs,
)
from chapter2.esn_model import EchoStateNetwork, TrainingSequence
from chapter2.esn_metrics import evaluate_rollout
# _fit_scalers_from_permitted_views is a private but fully generic helper
# (no hard-coded current list); ValidationCase/ModelOptimisationData,
# model_config, aggregate_rollouts and _recursive_case_rollout are likewise
# generic (no dependency on a fixed current count). Reused unmodified.
# esn_optimisation.RealCandidateEvaluator and .rollout_objective are NOT
# reused: RealCandidateEvaluator's __call__ hard-calls the locked
# rollout_objective(), which raises unless there are exactly Part-1's three
# currents by three windows. SourceCandidateEvaluator below is the smallest
# isolated Part-2 wrapper reproducing the same fit+score steps, calling
# validation.source_objective() (generalised to any current/window set)
# in its place.
from chapter2.esn_optimisation import (  # noqa: F401  (re-exported for callers)
    ModelOptimisationData,
    TRAINING_WASHOUT,
    ValidationCase,
    VALID_PREDICTION_THRESHOLD,
    DIVERGENCE_THRESHOLD,
    COLLAPSE_STD_RATIO_THRESHOLD,
    NONFINITE_FAILURE_SCORE,
    _fit_scalers_from_permitted_views,
    _recursive_case_rollout,
    aggregate_rollouts,
    model_config,
)

from . import config


def prepare_model_data_for_currents(
    prepared: Sequence[PreparedOptimisationTrajectory],
    *,
    model_type: str = config.MODEL_TYPE,
) -> ModelOptimisationData:
    """Part-2 analogue of esn_optimisation.prepare_model_data, for ANY current list.

    Same scaling/windowing logic as Part 1's function body; only the
    hard-coded ``TRAIN_CURRENTS`` / "exactly nine cases" assertions are
    removed, since Part 2 uses a different number of training currents.
    """
    items = tuple(prepared)
    if not items:
        raise ValueError("at least one prepared trajectory is required")
    if any(item.fitting.inputs.shape[1] != 4 for item in items):
        raise ValueError("Part 2 uses the parameter-aware (4-column) inputs only")

    scalers = _fit_scalers_from_permitted_views(items)
    training_sequences: list[TrainingSequence] = []
    validation_cases: list[ValidationCase] = []

    for item in items:
        scaled_fitting = scale_one_step_pairs(item.fitting, scalers)
        training_sequences.append(TrainingSequence(scaled_fitting.inputs, scaled_fitting.targets))

        for view in item.validation_windows:
            scaled_warmup = scale_one_step_pairs(view.warmup, scalers)
            scaled_scored = scale_one_step_pairs(view.scored, scalers)
            if not np.array_equal(scaled_warmup.targets[-1], scaled_scored.inputs[0, :3]):
                raise ValueError("validation warm-up and scored states are misaligned")
            validation_cases.append(
                ValidationCase(
                    current=item.current,
                    window=view.definition.number,
                    warmup_inputs=np.asarray(scaled_warmup.inputs, dtype=float).copy(),
                    initial_state=scaled_warmup.targets[-1].copy(),
                    current_values=scaled_scored.inputs[:, 3].copy(),
                    targets_physical=scalers.inverse_states(scaled_scored.targets),
                    warmup_range=(
                        int(view.warmup.transition_indices[0]),
                        int(view.warmup.transition_indices[-1]) + 1,
                    ),
                    scored_range=(
                        int(view.scored.transition_indices[0]),
                        int(view.scored.transition_indices[-1]) + 1,
                    ),
                )
            )

    return ModelOptimisationData(
        model_type=model_type,
        input_dimension=4,
        training_sequences=tuple(training_sequences),
        validation_cases=tuple(validation_cases),
        scalers=scalers,
    )


class SourceCandidateEvaluator:
    """Fit one candidate and score every validation rollout in ``data``.

    Reproduces esn_optimisation.RealCandidateEvaluator's fit+score body
    exactly (same EchoStateNetwork, same model_config, same
    _recursive_case_rollout, same evaluate_rollout/aggregate_rollouts), but
    calls validation.source_objective() with THIS data's own (current,
    window) set instead of Part-1's hard-coded three-by-three set. This is
    the "smallest isolated Part-2 wrapper" the master task authorizes.
    """

    def __init__(self, data: ModelOptimisationData) -> None:
        self.data = data
        self._expected_pairs = {
            (case.current, case.window) for case in data.validation_cases
        }

    def __call__(self, parameters: dict[str, Any], model_seed: int) -> dict[str, Any]:
        model = EchoStateNetwork(model_config(parameters, self.data.model_type, model_seed))
        model.fit(self.data.training_sequences, washout=TRAINING_WASHOUT)
        rollouts: list[dict[str, Any]] = []

        for case in self.data.validation_cases:
            predicted_scaled, failure_step, failure_reason = _recursive_case_rollout(
                model, case, self.data.model_type
            )
            with np.errstate(over="ignore", invalid="ignore"):
                predictions_physical = (
                    predicted_scaled * self.data.scalers.state.scale
                    + self.data.scalers.state.mean
                )
                normalised_difference = (
                    predictions_physical - case.targets_physical
                ) / self.data.scalers.state.scale
                pointwise_error = np.sqrt(np.mean(np.square(normalised_difference), axis=1))
            numerical_failure_rows = np.flatnonzero(
                ~np.all(np.isfinite(predictions_physical), axis=1) | ~np.isfinite(pointwise_error)
            )
            if failure_step is None and len(numerical_failure_rows):
                failure_step = int(numerical_failure_rows[0])
                failure_reason = (
                    "non_finite_physical_prediction"
                    if not np.all(np.isfinite(predictions_physical[failure_step]))
                    else "non_finite_normalised_error"
                )
            metrics = evaluate_rollout(
                predictions_physical,
                case.targets_physical,
                normalisation_scale=self.data.scalers.state.scale,
                dt=0.01,
                valid_prediction_threshold=VALID_PREDICTION_THRESHOLD,
                divergence_threshold=DIVERGENCE_THRESHOLD,
                collapse_std_ratio_threshold=COLLAPSE_STD_RATIO_THRESHOLD,
            ).to_dict()
            if metrics["nrmse_state"] is None and failure_step is None:
                failure_step = 0
                failure_reason = "non_finite_rollout_metric"
            nonfinite_failure = failure_step is not None or metrics["nrmse_state"] is None
            objective_nrmse = (
                NONFINITE_FAILURE_SCORE if nonfinite_failure else float(metrics["nrmse_state"])
            )
            rollouts.append(
                {
                    "current": case.current,
                    "window": case.window,
                    "warmup_range": list(case.warmup_range),
                    "scored_range": list(case.scored_range),
                    "model_seed": int(model_seed),
                    "objective_nrmse": objective_nrmse,
                    "nonfinite_failure": nonfinite_failure,
                    "failure_step": failure_step,
                    "failure_reason": failure_reason,
                    "metrics": metrics,
                }
            )

        objective = source_objective(rollouts, self._expected_pairs)
        return {"objective": objective, "rollouts": rollouts, "aggregate": aggregate_rollouts(rollouts)}


def source_objective(rollouts: Sequence[Mapping[str, Any]], expected_pairs: set[tuple[float, int]]) -> float:
    """Equally weighted mean NRMSE over the expected (current, window) set.

    Generalises esn_optimisation.rollout_objective (which hard-codes Part-1's
    three-currents-by-three-windows set) to Part 2's larger per-family set.
    """
    items = tuple(rollouts)
    actual = {(float(item["current"]), int(item["window"])) for item in items}
    if len(items) != len(expected_pairs) or actual != expected_pairs:
        raise ValueError("objective requires exactly the expected (current, window) set")
    values = np.asarray([float(item["objective_nrmse"]) for item in items])
    if not np.all(np.isfinite(values)):
        raise ValueError("rollout objective values must be finite")
    return float(np.mean(values))


def source_quality_tuple(aggregate: Mapping[str, Any], n_rollouts: int) -> tuple:
    """Lexicographic search-phase ranking key (Step 4E). Lower is better."""
    return (
        int(aggregate.get("numerical_failure_count", 0)),
        int(aggregate["divergence_rollout_count"]),
        int(aggregate["collapse_rollout_count"]),
        -min_vpt_fraction(aggregate),
        -median_vpt_fraction(aggregate, n_rollouts),
        float(aggregate["worst_objective_nrmse"]),
        float(aggregate["median_objective_nrmse"]),
        float(aggregate["mean_objective_nrmse"]),
    )


def min_vpt_fraction(aggregate: Mapping[str, Any]) -> float:
    # aggregate_rollouts does not store per-rollout VPT list; caller passes it in.
    return float(aggregate["min_vpt_fraction"])


def median_vpt_fraction(aggregate: Mapping[str, Any], n_rollouts: int) -> float:
    return float(aggregate["median_vpt_fraction"])


def augment_aggregate_with_vpt_fraction(
    aggregate: dict[str, Any], rollouts: Sequence[Mapping[str, Any]], horizon: int
) -> dict[str, Any]:
    """Add min/median VPT-fraction and numerical-failure count to aggregate_rollouts' output."""
    vpt_fractions = [
        float(item["metrics"]["valid_prediction_steps"]) / horizon for item in rollouts
    ]
    aggregate = dict(aggregate)
    aggregate["min_vpt_fraction"] = float(np.min(vpt_fractions))
    aggregate["median_vpt_fraction"] = float(np.median(vpt_fractions))
    aggregate["numerical_failure_count"] = int(
        sum(bool(item["nonfinite_failure"]) for item in rollouts)
    )
    return aggregate


def build_loco_folds(training_currents: Sequence[float]) -> list[dict[str, Any]]:
    """Leave-one-training-current-out fold definitions (Step 4C)."""
    currents = list(training_currents)
    folds = []
    for held_out in currents:
        fit_on = [c for c in currents if c != held_out]
        folds.append({"fit_on": fit_on, "validate_on": held_out})
    return folds


# --- Absolute source-quality gate (Step 4F) -------------------------------
# Derived BEFORE any Part-2 optimisation from Part-1's own achieved robust
# parameter-aware performance (chapter2/optimisation_results/step7_selection.json,
# "best_robust_aggregate": mean_valid_prediction_steps=7638.18/8000=0.9548,
# zero divergence/collapse across all 45 five-seed rollouts). Part 2 trains on
# more currents in a harder multi-regime setting, so the bar is set at a
# demonstrably strong but slightly more permissive level, fixed here and never
# adjusted after seeing Part-2 results.
SOURCE_QUALITY_GATE = {
    "min_vpt_fraction_per_rollout": 0.80,
    "max_numerical_failures": 0,
    "max_divergence_count": 0,
    "max_collapse_count": 0,
    "derivation": (
        "Part-1 step7_selection.json best_robust_aggregate achieved "
        "mean_valid_prediction_steps=7638.18/8000=0.9548 with zero divergence "
        "and zero collapse across 45 five-seed rollouts (3 currents x 3 "
        "windows x 5 seeds). This gate requires >=0.80 VPT fraction on EVERY "
        "individual rollout (not just the mean) and zero failures/divergence/"
        "collapse, fixed before Part-2 production optimisation."
    ),
}


def passes_source_quality_gate(rollouts: Sequence[Mapping[str, Any]], horizon: int) -> bool:
    if any(bool(item["nonfinite_failure"]) for item in rollouts):
        return False
    if any(bool(item["metrics"]["diverged"]) for item in rollouts):
        return False
    if any(bool(item["metrics"]["prediction_collapse_any"]) for item in rollouts):
        return False
    vpt_fractions = [float(item["metrics"]["valid_prediction_steps"]) / horizon for item in rollouts]
    return min(vpt_fractions) >= SOURCE_QUALITY_GATE["min_vpt_fraction_per_rollout"]


def write_source_validation_protocol(regular_currents, chaotic_currents) -> dict:
    """Freeze the temporal source-validation protocol BEFORE optimisation (Step 4B)."""
    protocol = {
        "kind": "source_validation_protocol",
        "reused_from_part1": (
            "chapter2.esn_config.VALIDATION_WINDOWS (three windows per current: "
            "reset, 2000-transition teacher-forced warm-up, 8000-transition "
            "autonomous scored rollout, at transitions "
            "[40000,42000)/[42000,50000), [50000,52000)/[52000,60000), "
            "[60000,62000)/[62000,70000)); chapter2.esn_optimisation."
            "TRAINING_WASHOUT=2000 within a [0,40000) fitting range."
        ),
        "fitting_transition_range": [0, config.FITTING_TRANSITIONS_STOP],
        "training_washout": config.TRAINING_WASHOUT,
        "validation_windows": [
            {
                "number": w.number,
                "warmup_range": [w.warmup.start, w.warmup.stop],
                "scored_range": [w.scored.start, w.scored.stop],
            }
            for w in config.VALIDATION_WINDOWS
        ],
        "regular_training_currents": list(regular_currents),
        "chaotic_training_currents": list(chaotic_currents),
        "source_quality_gate": SOURCE_QUALITY_GATE,
        "loco_folds": {
            "regular": build_loco_folds(regular_currents),
            "chaotic": build_loco_folds(chaotic_currents),
        },
        "note": (
            "Frozen before any Part-2 optimisation call. Not changed after "
            "observing optimisation results."
        ),
    }
    config.DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    config.SOURCE_VALIDATION_PROTOCOL_PATH.write_text(
        json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return protocol
