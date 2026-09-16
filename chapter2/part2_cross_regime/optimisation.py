"""Step 4: candidate generation, search-phase ranking, confirmation, LOCO, locking.

Reuses chapter2.esn_optimisation.search_dimensions() (Part-1's exact frozen
search space) and skopt's GP-EI optimizer (the same library/settings Part-1
uses) via a small Part-2-only orchestration wrapper. No Part-1 optimisation
code is modified.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
from skopt import Optimizer
from skopt.utils import use_named_args

from . import config, data, validation


def _lhs_candidates(seed: int, n_per_size: int = 5) -> list[dict[str, Any]]:
    """15 deterministic space-filling candidates, 5 per reservoir size {100,200,300}."""
    rng = np.random.default_rng(seed)
    sizes = [100, 200, 300]
    candidates = []
    continuous_specs = [
        ("reservoir_connectivity", 0.01, 1.0, "uniform"),
        ("input_scaling", 0.01, 3.0, "uniform"),
        ("spectral_radius", 0.01, 3.0, "uniform"),
        ("ridge_regularisation", 1.0e-10, 1.0e-2, "log-uniform"),
        ("leak_rate", 0.01, 1.0, "uniform"),
    ]
    for size in sizes:
        # Stratified (Latin-hypercube-style) samples within each dimension.
        strata = (np.arange(n_per_size) + rng.uniform(size=n_per_size)) / n_per_size
        columns = {}
        for name, low, high, prior in continuous_specs:
            permuted = rng.permutation(strata)
            if prior == "log-uniform":
                log_low, log_high = np.log10(low), np.log10(high)
                columns[name] = 10 ** (log_low + permuted * (log_high - log_low))
            else:
                columns[name] = low + permuted * (high - low)
        for i in range(n_per_size):
            candidates.append(
                {
                    "reservoir_size": size,
                    "reservoir_connectivity": float(columns["reservoir_connectivity"][i]),
                    "input_scaling": float(columns["input_scaling"][i]),
                    "spectral_radius": float(columns["spectral_radius"][i]),
                    "ridge_regularisation": float(columns["ridge_regularisation"][i]),
                    "leak_rate": float(columns["leak_rate"][i]),
                }
            )
    return candidates


def _dedup_key(hp: dict[str, Any]) -> str:
    return json.dumps({k: hp[k] for k in sorted(hp)}, sort_keys=True)


def build_candidate_list(family: str, anchor_hyperparameters: dict) -> list[dict[str, Any]]:
    """1 anchor + 15 LHS + (GP-EI filled in during the search loop)."""
    seeds = config.FAMILY_SEARCH_SEEDS[family]
    lhs = _lhs_candidates(seeds["lhs"])
    seen = {_dedup_key(anchor_hyperparameters)}
    unique_lhs = []
    for hp in lhs:
        key = _dedup_key(hp)
        if key not in seen:
            unique_lhs.append(hp)
            seen.add(key)
    return [dict(anchor_hyperparameters, _source="anchor")] + [
        dict(hp, _source="lhs") for hp in unique_lhs
    ]


def evaluate_candidate(model_data, hyperparameters: dict, seed: int) -> dict:
    hp = {k: v for k, v in hyperparameters.items() if not k.startswith("_")}
    evaluator = validation.SourceCandidateEvaluator(model_data)
    result = evaluator(hp, seed)
    horizon = len(model_data.validation_cases[0].targets_physical)
    aggregate = validation.augment_aggregate_with_vpt_fraction(
        result["aggregate"], result["rollouts"], horizon
    )
    return {
        "hyperparameters": hp,
        "source": hyperparameters.get("_source", "gp"),
        "objective": result["objective"],
        "aggregate": aggregate,
        "rank_key": list(validation.source_quality_tuple(aggregate, len(result["rollouts"]))),
    }


def run_family_search(family: str, design: dict, anchor_hyperparameters: dict) -> dict:
    key = "regular_trained" if family == "regular" else "chaotic_trained"
    role = "regular_training" if family == "regular" else "chaotic_training"
    currents = [e["current"] for e in design["models"][key]["training_currents"]]
    prepared = [
        data.prepare_optimisation_trajectory(data.load_source_trajectory(role, c))
        for c in currents
    ]
    model_data = validation.prepare_model_data_for_currents(prepared)

    history: list[dict[str, Any]] = []
    candidates = build_candidate_list(family, anchor_hyperparameters)
    for hp in candidates:
        record = evaluate_candidate(model_data, hp, config.CANDIDATE_MODEL_SEED)
        history.append(record)
        print(f"[{family}] {record['source']:6s} objective={record['objective']:.6g}", flush=True)

    dimensions = config.search_dimensions()

    @use_named_args(dimensions)
    def _unused(**kwargs):  # only for parameter-name validation, not called
        return 0.0

    seed = config.FAMILY_SEARCH_SEEDS[family]["gp"]
    optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator="GP",
        acq_func="EI",
        n_initial_points=0,
        initial_point_generator="random",
        random_state=seed,
    )
    seen_keys = {_dedup_key(r["hyperparameters"]) for r in history}
    # Tell the optimizer every anchor+LHS observation before asking for proposals.
    for record in history:
        point = [record["hyperparameters"][name] for name in config.SEARCH_PARAMETER_ORDER]
        optimizer.tell(point, record["objective"])

    n_gp_needed = config.N_GP_CANDIDATES
    attempts = 0
    while sum(1 for r in history if r["source"] == "gp") < n_gp_needed and attempts < n_gp_needed * 3:
        attempts += 1
        point = optimizer.ask()
        hp = dict(zip(config.SEARCH_PARAMETER_ORDER, point))
        hp["reservoir_size"] = int(hp["reservoir_size"])
        key = _dedup_key(hp)
        if key in seen_keys:
            # Duplicate proposal: tell it its existing objective and continue.
            existing = next(r for r in history if _dedup_key(r["hyperparameters"]) == key)
            optimizer.tell(point, existing["objective"])
            continue
        record = evaluate_candidate(model_data, dict(hp, _source="gp"), config.CANDIDATE_MODEL_SEED)
        history.append(record)
        seen_keys.add(key)
        optimizer.tell(point, record["objective"])
        print(f"[{family}] gp     objective={record['objective']:.6g} ({len(history)}/48)", flush=True)

    ranked = sorted(history, key=lambda r: r["rank_key"] + [_dedup_key(r["hyperparameters"])])
    return {
        "family": family,
        "training_currents": currents,
        "candidate_count": len(history),
        "history": history,
        "ranked_hyperparameter_keys": [_dedup_key(r["hyperparameters"]) for r in ranked],
    }


def top_k(search_result: dict, k: int = config.TOP_K_FOR_CONFIRMATION) -> list[dict]:
    by_key = {_dedup_key(r["hyperparameters"]): r for r in search_result["history"]}
    keys = search_result["ranked_hyperparameter_keys"][:k]
    return [by_key[key] for key in keys]


# --- Confirmation (Step 4G) ------------------------------------------------
# Robust-seed rule, frozen before confirmation runs: a candidate is "eligible"
# only if it has ZERO numerical failures/divergence/collapse on EVERY one of
# the five confirmation seeds, AND its per-rollout VPT fraction gate
# (validation.SOURCE_QUALITY_GATE, >=0.80) passes on at least 4 of the 5
# seeds. This tolerates at most one weak reservoir initialization but never
# tolerates a genuine numerical failure.
ROBUST_SEED_RULE = {
    "seeds": list(config.CONFIRMATION_SEEDS),
    "max_failing_seeds_for_vpt_gate": 1,
    "zero_tolerance_for": ["numerical_failure", "divergence", "collapse"],
}


def confirm_candidate(model_data, candidate: dict) -> dict:
    per_seed = {}
    for seed in config.CONFIRMATION_SEEDS:
        if seed == config.CANDIDATE_MODEL_SEED:
            # Reuse the seed-42 search evidence bit-for-bit.
            per_seed[seed] = {
                "objective": candidate["objective"],
                "aggregate": candidate["aggregate"],
                "reused_from_search": True,
            }
            continue
        record = evaluate_candidate(model_data, dict(candidate["hyperparameters"], _source="confirm"), seed)
        per_seed[seed] = {
            "objective": record["objective"],
            "aggregate": record["aggregate"],
            "reused_from_search": False,
        }

    zero_tolerance_ok = all(
        info["aggregate"]["numerical_failure_count"] == 0
        and info["aggregate"]["divergence_rollout_count"] == 0
        and info["aggregate"]["collapse_rollout_count"] == 0
        for info in per_seed.values()
    )
    gate_pass_count = sum(
        1
        for info in per_seed.values()
        if info["aggregate"]["min_vpt_fraction"] >= validation.SOURCE_QUALITY_GATE["min_vpt_fraction_per_rollout"]
    )
    failing_seeds = len(per_seed) - gate_pass_count
    eligible = zero_tolerance_ok and failing_seeds <= ROBUST_SEED_RULE["max_failing_seeds_for_vpt_gate"]

    return {
        "hyperparameters": candidate["hyperparameters"],
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "zero_tolerance_ok": zero_tolerance_ok,
        "gate_pass_count": gate_pass_count,
        "failing_seed_count": failing_seeds,
        "eligible": eligible,
    }


# --- LOCO (Step 4C / 4H) ---------------------------------------------------

def run_loco_for_candidate(design: dict, family: str, candidate: dict) -> dict:
    key = "regular_trained" if family == "regular" else "chaotic_trained"
    role = "regular_training" if family == "regular" else "chaotic_training"
    currents = [e["current"] for e in design["models"][key]["training_currents"]]
    folds = validation.build_loco_folds(currents)

    fold_rollouts: list[dict] = []
    for seed in config.CONFIRMATION_SEEDS:
        for fold in folds:
            fit_prepared = [
                data.prepare_optimisation_trajectory(data.load_source_trajectory(role, c))
                for c in fold["fit_on"]
            ]
            val_prepared = data.prepare_optimisation_trajectory(
                data.load_source_trajectory(role, fold["validate_on"])
            )
            fit_data = validation.prepare_model_data_for_currents(fit_prepared)
            model_cfg = validation.model_config(candidate["hyperparameters"], config.MODEL_TYPE, seed)
            from chapter2.esn_model import EchoStateNetwork

            esn = EchoStateNetwork(model_cfg)
            esn.fit(fit_data.training_sequences, washout=config.TRAINING_WASHOUT)

            val_model_data = validation.prepare_model_data_for_currents([val_prepared])
            for case in val_model_data.validation_cases:
                from chapter2.esn_optimisation import _recursive_case_rollout

                predicted_scaled, failure_step, failure_reason = _recursive_case_rollout(
                    esn, case, config.MODEL_TYPE
                )
                predictions_physical = (
                    predicted_scaled * val_model_data.scalers.state.scale
                    + val_model_data.scalers.state.mean
                )
                from chapter2.esn_metrics import evaluate_rollout

                metrics = evaluate_rollout(
                    predictions_physical,
                    case.targets_physical,
                    normalisation_scale=val_model_data.scalers.state.scale,
                    dt=config.DT,
                    valid_prediction_threshold=config.VALID_PREDICTION_THRESHOLD,
                    divergence_threshold=config.DIVERGENCE_THRESHOLD,
                    collapse_std_ratio_threshold=config.COLLAPSE_STD_RATIO_THRESHOLD,
                ).to_dict()
                nonfinite = failure_step is not None or metrics["nrmse_state"] is None
                objective_nrmse = config.NONFINITE_FAILURE_SCORE if nonfinite else float(metrics["nrmse_state"])
                fold_rollouts.append(
                    {
                        "held_out_current": fold["validate_on"],
                        "seed": seed,
                        "window": case.window,
                        "objective_nrmse": objective_nrmse,
                        "nonfinite_failure": nonfinite,
                        "metrics": metrics,
                    }
                )

    vpts = [r["metrics"]["valid_prediction_steps"] for r in fold_rollouts]
    nrmses = [r["objective_nrmse"] for r in fold_rollouts]
    loco_rank_key = [
        sum(1 for r in fold_rollouts if r["nonfinite_failure"]),
        sum(1 for r in fold_rollouts if r["metrics"]["diverged"]),
        sum(1 for r in fold_rollouts if r["metrics"]["prediction_collapse_any"]),
        float(np.max(nrmses)),
        float(np.median(nrmses)),
        float(np.mean(nrmses)),
        -float(np.median(vpts)),
    ]
    return {
        "hyperparameters": candidate["hyperparameters"],
        "n_folds": len(folds),
        "n_seeds": len(config.CONFIRMATION_SEEDS),
        "rollouts": fold_rollouts,
        "median_nrmse": float(np.median(nrmses)),
        "mean_nrmse": float(np.mean(nrmses)),
        "worst_nrmse": float(np.max(nrmses)),
        "median_vpt": float(np.median(vpts)),
        "loco_rank_key": loco_rank_key,
    }


class NoEligibleCandidateError(RuntimeError):
    """Raised when zero candidates survive the robust source-quality gate.

    This is a controlled failure (Step 4/testing requirement), not a bug: a
    family whose entire top-10 fails robust confirmation must stop, not fall
    back to a weaker pick.
    """


def _write_json(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_and_lock(family: str) -> dict:
    design = data.load_frozen_design()
    design_hash = design["design_hash_sha256"]
    key = "regular_trained" if family == "regular" else "chaotic_trained"
    role = "regular_training" if family == "regular" else "chaotic_training"
    currents = [e["current"] for e in design["models"][key]["training_currents"]]

    part1_hp = json.loads(
        (config.BASELINE_DIR / "part1_protocol_snapshot.json").read_text()
    )["step7_selected_parameter_aware_hyperparameters"]

    print(f"=== [{family}] Step 4D-4E: 48-candidate search ===", flush=True)
    search_result = run_family_search(family, design, part1_hp)
    _write_json(
        config.OPTIMISATION_DIR / f"{family}_history.json",
        {"kind": "search_history", "design_hash": design_hash, **search_result},
    )

    print(f"=== [{family}] Step 4G: top-{config.TOP_K_FOR_CONFIRMATION} multi-seed confirmation ===", flush=True)
    top10 = top_k(search_result)
    prepared = [
        data.prepare_optimisation_trajectory(data.load_source_trajectory(role, c))
        for c in currents
    ]
    model_data = validation.prepare_model_data_for_currents(prepared)
    confirmations = []
    for candidate in top10:
        conf = confirm_candidate(model_data, candidate)
        confirmations.append(conf)
        print(
            f"[{family}] confirm gate_pass_count={conf['gate_pass_count']}/5 "
            f"zero_tolerance_ok={conf['zero_tolerance_ok']} eligible={conf['eligible']}",
            flush=True,
        )
    _write_json(
        config.OPTIMISATION_DIR / f"{family}_confirmation.json",
        {
            "kind": "confirmation",
            "design_hash": design_hash,
            "robust_seed_rule": ROBUST_SEED_RULE,
            "confirmations": confirmations,
        },
    )

    eligible = [c for c in confirmations if c["eligible"]]
    if not eligible:
        _write_json(
            config.OPTIMISATION_DIR / f"{family}_selection.json",
            {
                "kind": "selection_failure",
                "design_hash": design_hash,
                "reason": "zero candidates passed the robust source-quality gate",
                "top10_confirmations": confirmations,
            },
        )
        raise NoEligibleCandidateError(
            f"[{family}] zero of the top {len(top10)} candidates passed the robust "
            "source-quality gate; stopping (controlled failure), not locking a winner."
        )

    print(f"=== [{family}] Step 4H: LOCO over {len(eligible)} eligible candidate(s) ===", flush=True)
    loco_results = []
    for conf in eligible:
        loco = run_loco_for_candidate(design, family, conf)
        loco_results.append(loco)
        print(f"[{family}] LOCO median_nrmse={loco['median_nrmse']:.6g} median_vpt={loco['median_vpt']:.1f}", flush=True)
    _write_json(
        config.OPTIMISATION_DIR / f"{family}_loco.json",
        {"kind": "loco", "design_hash": design_hash, "results": loco_results},
    )

    winner = min(loco_results, key=lambda r: r["loco_rank_key"] + [_dedup_key(r["hyperparameters"])])
    selection = {
        "kind": "selection",
        "design_hash": design_hash,
        "family": family,
        "training_currents": currents,
        "locked_hyperparameters": winner["hyperparameters"],
        "loco_summary": {
            "median_nrmse": winner["median_nrmse"],
            "mean_nrmse": winner["mean_nrmse"],
            "worst_nrmse": winner["worst_nrmse"],
            "median_vpt": winner["median_vpt"],
        },
        "eligible_candidate_count": len(eligible),
        "top_k_confirmed_count": len(top10),
    }
    _write_json(config.OPTIMISATION_DIR / f"{family}_selection.json", selection)
    print(f"=== [{family}] LOCKED: {winner['hyperparameters']} ===", flush=True)
    return selection


if __name__ == "__main__":
    import sys

    run_and_lock(sys.argv[1])
