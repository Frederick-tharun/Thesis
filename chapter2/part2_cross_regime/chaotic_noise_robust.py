"""Noise-regularised chaotic source training (targeted fix from Phase B diagnosis).

Diagnosis (chaotic_diagnostic.py) found that seed-dependent divergence is
caused by the trained readout never having seen anything but the exact,
noise-free teacher-forced trajectory: once autonomous recursion drifts even
slightly off that trajectory (as it eventually must, for a chaotic system
with a positive Lyapunov exponent), the readout has no learned "restoring"
behaviour and some reservoir realisations run away into saturation.

The classical, well-established remedy (Jaeger 2001; standard practice for
generative/autonomous ESNs) is to add small Gaussian noise to the STATE
columns of the training inputs (never to the supplied current, and never to
the targets) so the readout is trained on a small neighbourhood of the true
trajectory rather than a single noise-free curve. This is purely a training-
data augmentation implemented in Part-2 wrapper code: it does not touch
chapter2/esn_model.py, does not change the ESN architecture, does not use
teacher forcing during scored recursion, and does not alter the HR equations.
Validation/scoring remains on the exact same deterministic, noise-free
protocol as every other Part-2 stage.
"""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import numpy as np

from chapter2.esn_model import TrainingSequence

from . import config, data, validation
from .validation import ModelOptimisationData, SourceCandidateEvaluator

RESULTS_DIR = config.PACKAGE_ROOT / "results" / "chaotic_noise_robust"
ROLE = "chaotic_training"


def _noisy_training_sequences(
    model_data: ModelOptimisationData, noise_std: float, noise_seed: int
) -> tuple[TrainingSequence, ...]:
    """Add iid Gaussian noise (in already-standardised units) to the state
    columns only of each training sequence's inputs. Targets are untouched.
    noise_std=0 reproduces the original (unregularised) training data exactly.
    """
    if noise_std == 0.0:
        return model_data.training_sequences
    rng = np.random.default_rng(noise_seed)
    out = []
    for seq in model_data.training_sequences:
        inputs = seq.inputs.copy()
        inputs[:, :3] += rng.normal(0.0, noise_std, size=inputs[:, :3].shape)
        out.append(TrainingSequence(inputs, seq.targets))
    return tuple(out)


def build_noisy_model_data(model_data: ModelOptimisationData, noise_std: float, noise_seed: int) -> ModelOptimisationData:
    noisy = _noisy_training_sequences(model_data, noise_std, noise_seed)
    return replace(model_data, training_sequences=noisy)


def evaluate_noise_level(model_data: ModelOptimisationData, hyperparameters: dict, noise_std: float, seed: int) -> dict:
    """Fit with noisy training data (noise seed tied to model seed for
    reproducibility), score on the exact, noise-free validation protocol.
    """
    noisy_data = build_noisy_model_data(model_data, noise_std, noise_seed=10_000_000 + seed)
    evaluator = SourceCandidateEvaluator(noisy_data)
    result = evaluator(hyperparameters, seed)
    horizon = len(model_data.validation_cases[0].targets_physical)
    aggregate = validation.augment_aggregate_with_vpt_fraction(result["aggregate"], result["rollouts"], horizon)
    return {"objective": result["objective"], "aggregate": aggregate, "rollouts": result["rollouts"]}


def load_anchor_and_data():
    design = data.load_frozen_design()
    part1_hp = json.loads(
        (config.BASELINE_DIR / "part1_protocol_snapshot.json").read_text()
    )["step7_selected_parameter_aware_hyperparameters"]
    currents = [e["current"] for e in design["models"]["chaotic_trained"]["training_currents"]]
    prepared = [
        data.prepare_optimisation_trajectory(data.load_source_trajectory(ROLE, c))
        for c in currents
    ]
    model_data = validation.prepare_model_data_for_currents(prepared)
    return design, part1_hp, model_data, currents


def proof_of_concept() -> dict:
    """Quick sweep: fixed (anchor) hyperparameters, varying noise_std, all 5 seeds."""
    design, anchor_hp, model_data, currents = load_anchor_and_data()
    seeds = [42, 123, 456, 789, 2026]
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]

    results = {}
    for noise_std in noise_levels:
        per_seed = {}
        for seed in seeds:
            r = evaluate_noise_level(model_data, anchor_hp, noise_std, seed)
            per_seed[str(seed)] = r["aggregate"]
            print(
                f"noise={noise_std:.3f} seed={seed:5d} "
                f"min_vpt={r['aggregate']['min_vpt_fraction']:.4f} "
                f"div={r['aggregate']['divergence_rollout_count']} "
                f"coll={r['aggregate']['collapse_rollout_count']} "
                f"mean_nrmse={r['aggregate']['mean_objective_nrmse']:.4g}",
                flush=True,
            )
        results[str(noise_std)] = per_seed

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "proof_of_concept.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return results


if __name__ == "__main__":
    proof_of_concept()
