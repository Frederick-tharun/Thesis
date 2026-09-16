"""Median-ensemble recovery for the chaotic source model.

Diagnosis found a clean, uniform split: at the Part-1-anchor hyperparameters,
seeds {42,123,2026} are excellent for the full 80-time-unit horizon and seeds
{456,789} diverge catastrophically on every chaotic current/window. A per-step
per-dimension MEDIAN across several independently-seeded EchoStateNetwork
instances is a standard, well-founded robust-statistics technique for exactly
this failure pattern: as long as a strict majority of ensemble members are
"good" at a given step, the median is mathematically guaranteed to select a
good member's value component-wise and ignore outlier (diverging) members.

This is still the unmodified, frozen chapter2.esn_model.EchoStateNetwork
architecture -- no new layer type, no learned combiner. The "model" becomes
an odd-sized ensemble of independently-seeded instances of the SAME sanctioned
class, combined by a fixed (non-learned) robust statistic, run in a coupled
recursive loop (all members see the same shared consensus state each step).
"""
from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from chapter2.esn_model import EchoStateNetwork
from chapter2.esn_metrics import evaluate_rollout

from . import config, data, validation
from .validation import (
    COLLAPSE_STD_RATIO_THRESHOLD,
    DIVERGENCE_THRESHOLD,
    NONFINITE_FAILURE_SCORE,
    TRAINING_WASHOUT,
    VALID_PREDICTION_THRESHOLD,
    model_config,
)

RESULTS_DIR = config.PACKAGE_ROOT / "results" / "chaotic_ensemble"
ROLE = "chaotic_training"


def fit_ensemble(model_data, hyperparameters: dict, seeds: Sequence[int]) -> list[EchoStateNetwork]:
    models = []
    for seed in seeds:
        model = EchoStateNetwork(model_config(hyperparameters, config.MODEL_TYPE, seed))
        model.fit(model_data.training_sequences, washout=TRAINING_WASHOUT)
        models.append(model)
    return models


def median_ensemble_rollout(models: Sequence[EchoStateNetwork], case, model_type: str):
    for model in models:
        model.reset_reservoir()
        model.teacher_forced_warmup(case.warmup_inputs, reset=False)
    state = case.initial_state.copy()
    horizon = len(case.targets_physical)
    predictions_scaled = np.full((horizon, 3), np.nan, dtype=float)
    member_predictions_log = np.full((horizon, len(models), 3), np.nan, dtype=float)

    failure_step = None
    for step in range(horizon):
        input_value = np.concatenate((state, [case.current_values[step]]))
        member_preds = np.stack([m.predict_one_step(input_value) for m in models], axis=0)
        member_predictions_log[step] = member_preds
        if not np.all(np.isfinite(member_preds)):
            # A member has already blown up to non-finite; median over the
            # finite survivors if a majority are still finite, else fail.
            finite_rows = member_preds[np.all(np.isfinite(member_preds), axis=1)]
            if len(finite_rows) <= len(models) // 2:
                failure_step = step
                break
            consensus = np.median(finite_rows, axis=0)
        else:
            consensus = np.median(member_preds, axis=0)
        predictions_scaled[step] = consensus
        state = consensus
    return predictions_scaled, member_predictions_log, failure_step


def evaluate_ensemble(model_data, models: Sequence[EchoStateNetwork]) -> dict:
    rollouts = []
    for case in model_data.validation_cases:
        predictions_scaled, member_log, failure_step = median_ensemble_rollout(models, case, config.MODEL_TYPE)
        predictions_physical = (
            predictions_scaled * model_data.scalers.state.scale + model_data.scalers.state.mean
        )
        metrics = evaluate_rollout(
            predictions_physical, case.targets_physical,
            normalisation_scale=model_data.scalers.state.scale, dt=config.DT,
            valid_prediction_threshold=VALID_PREDICTION_THRESHOLD,
            divergence_threshold=DIVERGENCE_THRESHOLD,
            collapse_std_ratio_threshold=COLLAPSE_STD_RATIO_THRESHOLD,
        ).to_dict()
        nonfinite = failure_step is not None or metrics["nrmse_state"] is None
        objective_nrmse = NONFINITE_FAILURE_SCORE if nonfinite else float(metrics["nrmse_state"])
        rollouts.append(
            {
                "current": case.current, "window": case.window,
                "objective_nrmse": objective_nrmse, "nonfinite_failure": nonfinite,
                "failure_step": failure_step, "metrics": metrics,
            }
        )
    horizon = len(model_data.validation_cases[0].targets_physical)
    vpt_fractions = [r["metrics"]["valid_prediction_steps"] / horizon for r in rollouts]
    return {
        "rollouts": rollouts,
        "min_vpt_fraction": float(np.min(vpt_fractions)),
        "median_vpt_fraction": float(np.median(vpt_fractions)),
        "mean_nrmse": float(np.mean([r["objective_nrmse"] for r in rollouts])),
        "median_nrmse": float(np.median([r["objective_nrmse"] for r in rollouts])),
        "worst_nrmse": float(np.max([r["objective_nrmse"] for r in rollouts])),
        "divergence_count": int(sum(bool(r["metrics"]["diverged"]) for r in rollouts)),
        "collapse_count": int(sum(bool(r["metrics"]["prediction_collapse_any"]) for r in rollouts)),
        "numerical_failure_count": int(sum(bool(r["nonfinite_failure"]) for r in rollouts)),
        "n_rollouts": len(rollouts),
    }


def main() -> dict:
    design = data.load_frozen_design()
    anchor_hp = json.loads(
        (config.BASELINE_DIR / "part1_protocol_snapshot.json").read_text()
    )["step7_selected_parameter_aware_hyperparameters"]
    currents = [e["current"] for e in design["models"]["chaotic_trained"]["training_currents"]]
    prepared = [
        data.prepare_optimisation_trajectory(data.load_source_trajectory(ROLE, c))
        for c in currents
    ]
    model_data = validation.prepare_model_data_for_currents(prepared)

    import os
    seeds = json.loads(os.environ.get("ENSEMBLE_SEEDS", "[42, 123, 456, 789, 2026]"))
    print(f"Fitting ensemble of {len(seeds)} members (seeds={seeds}) at anchor hyperparameters...", flush=True)
    models = fit_ensemble(model_data, anchor_hp, seeds)

    print("Evaluating median-ensemble rollout on all 5 chaotic source currents x 3 windows...", flush=True)
    result = evaluate_ensemble(model_data, models)
    print(
        f"min_vpt_fraction={result['min_vpt_fraction']:.4f} median_vpt_fraction={result['median_vpt_fraction']:.4f} "
        f"divergence={result['divergence_count']} collapse={result['collapse_count']} "
        f"numerical_failure={result['numerical_failure_count']} mean_nrmse={result['mean_nrmse']:.4g}",
        flush=True,
    )
    for r in result["rollouts"]:
        print(
            f"  I={r['current']:.4f} win={r['window']} vpt={r['metrics']['valid_prediction_steps']:.1f} "
            f"nrmse={r['objective_nrmse']:.4g} diverged={r['metrics']['diverged']}",
            flush=True,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "ensemble_seeds": seeds, "hyperparameters": anchor_hp,
        "training_currents": currents, "result": result,
        "gate": validation.SOURCE_QUALITY_GATE,
        "passes_gate": result["min_vpt_fraction"] >= validation.SOURCE_QUALITY_GATE["min_vpt_fraction_per_rollout"]
        and result["divergence_count"] == 0 and result["collapse_count"] == 0 and result["numerical_failure_count"] == 0,
    }
    out_name = f"median_ensemble_result_n{len(seeds)}.json"
    (RESULTS_DIR / out_name).write_text(
        json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {RESULTS_DIR / out_name}")
    print(f"PASSES FROZEN GATE: {out['passes_gate']}")
    return out


if __name__ == "__main__":
    main()
