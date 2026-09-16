"""Step 3B: Part-1-anchor baseline for the regular-source and chaotic-source models.

Trains the UNCHANGED chapter2.esn_model.EchoStateNetwork, configured with
Part-1's exact selected parameter-aware hyperparameters, on each family's
source training currents only, and reports the frozen temporal source
validation (Step 4B windows) plus LOCO. No final-test current is opened here.
"""
from __future__ import annotations

import json
from typing import Any

from . import config, data, part1_protocol, validation


def _family_prepared(design: dict, family: str):
    key = "regular_trained" if family == "regular" else "chaotic_trained"
    role = "regular_training" if family == "regular" else "chaotic_training"
    currents = [e["current"] for e in design["models"][key]["training_currents"]]
    prepared = [
        data.prepare_optimisation_trajectory(data.load_source_trajectory(role, c))
        for c in currents
    ]
    return currents, prepared


def evaluate_anchor(design: dict, family: str, anchor_hyperparameters: dict) -> dict:
    currents, prepared = _family_prepared(design, family)
    model_data = validation.prepare_model_data_for_currents(prepared)
    evaluator = validation.SourceCandidateEvaluator(model_data)
    result = evaluator(anchor_hyperparameters, config.CANDIDATE_MODEL_SEED)

    horizon = len(model_data.validation_cases[0].targets_physical)
    aggregate = validation.augment_aggregate_with_vpt_fraction(
        result["aggregate"], result["rollouts"], horizon
    )
    gate_passed = validation.passes_source_quality_gate(result["rollouts"], horizon)
    return {
        "family": family,
        "training_currents": currents,
        "anchor_hyperparameters": anchor_hyperparameters,
        "objective": result["objective"],
        "aggregate": aggregate,
        "rollouts": result["rollouts"],
        "source_quality_gate_passed": gate_passed,
    }


def run() -> dict:
    part1_protocol.write()
    snapshot = part1_protocol.snapshot()
    anchor = snapshot["step7_selected_parameter_aware_hyperparameters"]

    design = data.load_frozen_design()
    report: dict[str, Any] = {"kind": "step3b_anchor_baseline", "anchor_hyperparameters": anchor}
    for family in ("regular", "chaotic"):
        report[family] = evaluate_anchor(design, family, anchor)
        print(
            f"[{family}] objective={report[family]['objective']:.6g} "
            f"min_vpt_fraction={report[family]['aggregate']['min_vpt_fraction']:.4f} "
            f"divergences={report[family]['aggregate']['divergence_rollout_count']} "
            f"collapses={report[family]['aggregate']['collapse_rollout_count']} "
            f"gate_passed={report[family]['source_quality_gate_passed']}",
            flush=True,
        )

    config.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.BASELINE_DIR / "anchor_baseline_report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return report


if __name__ == "__main__":
    run()
