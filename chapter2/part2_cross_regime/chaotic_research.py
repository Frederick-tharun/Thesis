"""Phase C: chaotic-only seed-robust recovery search.

Only run after chaotic_diagnostic.py classifies the failure as
NO_BUG_STABILITY_PROBLEM. Seed robustness enters the SEARCH phase itself
(search seeds 42, 123, 456; confirmation adds 789, 2026), per the frozen
recovery protocol. Reuses the unmodified Part-1 ESN and the existing Part-2
validation/optimisation primitives; does not touch final-test data.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
from skopt import Optimizer

from . import config, data, validation
from .optimisation import (
    ROBUST_SEED_RULE,
    _dedup_key,
    _lhs_candidates,
    run_loco_for_candidate,
)

RESULTS_DIR = config.PACKAGE_ROOT / "results" / "chaotic_research"
FAMILY = "chaotic"
ROLE = "chaotic_training"

SEARCH_SEEDS = [42, 123, 456]
CONFIRMATION_ONLY_SEEDS = [789, 2026]
ALL_CONFIRMATION_SEEDS = [42, 123, 456, 789, 2026]

N_ANCHOR = 1
N_PREV_BEST = 1
N_LHS = 30
N_GP = 64
N_TOTAL = N_ANCHOR + N_PREV_BEST + N_LHS + N_GP
assert N_TOTAL == 96

LHS_SEED = 71201  # distinct from the original Step-4 chaotic LHS seed (51102)
GP_SEED = 81201  # distinct from the original Step-4 chaotic GP seed (61102)
TOP_K_FOR_CONFIRMATION = 12


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_previous_best_non_anchor() -> dict:
    hist = json.loads((config.OPTIMISATION_DIR / "chaotic_history.json").read_text())
    by_key = {_dedup_key(r["hyperparameters"]): r for r in hist["history"]}
    for key in hist["ranked_hyperparameter_keys"]:
        record = by_key[key]
        if record["source"] != "anchor":
            return record["hyperparameters"]
    raise RuntimeError("no non-anchor candidate found in the old chaotic history")


def build_candidate_list(anchor_hp: dict) -> list[dict[str, Any]]:
    prev_best = load_previous_best_non_anchor()
    lhs = _lhs_candidates(seed=LHS_SEED, n_per_size=10)  # 10 per size x 3 sizes = 30
    seen = {_dedup_key(anchor_hp), _dedup_key(prev_best)}
    unique_lhs = []
    for hp in lhs:
        key = _dedup_key(hp)
        if key not in seen:
            unique_lhs.append(hp)
            seen.add(key)
    candidates = [dict(anchor_hp, _source="anchor"), dict(prev_best, _source="prev_best")]
    candidates += [dict(hp, _source="lhs") for hp in unique_lhs]
    return candidates


def evaluate_candidate_multiseed(model_data, hyperparameters: dict, seeds: list[int]) -> dict:
    hp = {k: v for k, v in hyperparameters.items() if not k.startswith("_")}
    horizon = len(model_data.validation_cases[0].targets_physical)
    per_seed = {}
    per_seed_aggregate_augmented = {}
    all_rollouts = []
    for seed in seeds:
        evaluator = validation.SourceCandidateEvaluator(model_data)
        result = evaluator(hp, seed)
        per_seed[seed] = result
        per_seed_aggregate_augmented[str(seed)] = validation.augment_aggregate_with_vpt_fraction(
            result["aggregate"], result["rollouts"], horizon
        )
        all_rollouts.extend(result["rollouts"])
    vpt_fractions = np.array([r["metrics"]["valid_prediction_steps"] / horizon for r in all_rollouts])
    nrmses = np.array([r["objective_nrmse"] for r in all_rollouts])
    numerical_failures = sum(bool(r["nonfinite_failure"]) for r in all_rollouts)
    divergences = sum(bool(r["metrics"]["diverged"]) for r in all_rollouts)
    collapses = sum(bool(r["metrics"]["prediction_collapse_any"]) for r in all_rollouts)

    rank_key = [
        int(numerical_failures),
        int(divergences),
        int(collapses),
        -float(np.min(vpt_fractions)),
        -float(np.percentile(vpt_fractions, 10)),
        -float(np.median(vpt_fractions)),
        float(np.max(nrmses)),
        float(np.median(nrmses)),
        float(np.mean(nrmses)),
        0.0,  # reservoir-stability penalty: not pre-registered by Phase B, fixed at 0
    ]
    survives_hard_filter = (numerical_failures == 0 and divergences == 0 and collapses == 0)
    return {
        "hyperparameters": hp,
        "source": hyperparameters.get("_source", "gp"),
        "mean_objective_over_search_seeds": float(np.mean(nrmses)),
        "rank_key": rank_key,
        "survives_hard_filter": survives_hard_filter,
        "per_seed_aggregate": per_seed_aggregate_augmented,
        "n_rollouts": len(all_rollouts),
        "min_vpt_fraction": float(np.min(vpt_fractions)),
        "p10_vpt_fraction": float(np.percentile(vpt_fractions, 10)),
        "median_vpt_fraction": float(np.median(vpt_fractions)),
    }


def run_search() -> dict:
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

    protocol = {
        "kind": "chaotic_recovery_search_protocol",
        "design_hash": design["design_hash_sha256"],
        "search_seeds": SEARCH_SEEDS,
        "confirmation_only_seeds": CONFIRMATION_ONLY_SEEDS,
        "all_confirmation_seeds": ALL_CONFIRMATION_SEEDS,
        "candidate_composition": {
            "anchor": N_ANCHOR, "previous_best_non_anchor": N_PREV_BEST,
            "lhs": N_LHS, "gp_ei": N_GP, "total": N_TOTAL,
        },
        "lhs_seed": LHS_SEED,
        "gp_seed": GP_SEED,
        "search_space": "unchanged Part-1 space (no restriction: Phase B found no region-specific pathology, only a seed-dependent one)",
        "search_space_restriction_applied": False,
        "hard_filter": "any numerical failure / divergence / collapse on ANY of the 45 search-seed rollouts excludes the candidate from confirmation",
        "top_k_for_confirmation": TOP_K_FOR_CONFIRMATION,
        "source_quality_gate": validation.SOURCE_QUALITY_GATE,
        "robust_seed_rule": ROBUST_SEED_RULE,
    }
    _write_json(RESULTS_DIR / "search_protocol.json", protocol)

    history: list[dict[str, Any]] = []
    candidates = build_candidate_list(part1_hp)
    for hp in candidates:
        record = evaluate_candidate_multiseed(model_data, hp, SEARCH_SEEDS)
        history.append(record)
        print(
            f"[{FAMILY}] {record['source']:10s} mean_obj={record['mean_objective_over_search_seeds']:.6g} "
            f"min_vpt={record['min_vpt_fraction']:.4f} survives_filter={record['survives_hard_filter']}",
            flush=True,
        )

    dimensions = config.search_dimensions()
    optimizer = Optimizer(
        dimensions=dimensions, base_estimator="GP", acq_func="EI",
        n_initial_points=0, initial_point_generator="random", random_state=GP_SEED,
    )
    seen_keys = {_dedup_key(r["hyperparameters"]) for r in history}
    for record in history:
        point = [record["hyperparameters"][name] for name in config.SEARCH_PARAMETER_ORDER]
        optimizer.tell(point, record["mean_objective_over_search_seeds"])

    attempts = 0
    while sum(1 for r in history if r["source"] == "gp") < N_GP and attempts < N_GP * 3:
        attempts += 1
        point = optimizer.ask()
        hp = dict(zip(config.SEARCH_PARAMETER_ORDER, point))
        hp["reservoir_size"] = int(hp["reservoir_size"])
        key = _dedup_key(hp)
        if key in seen_keys:
            existing = next(r for r in history if _dedup_key(r["hyperparameters"]) == key)
            optimizer.tell(point, existing["mean_objective_over_search_seeds"])
            continue
        record = evaluate_candidate_multiseed(model_data, dict(hp, _source="gp"), SEARCH_SEEDS)
        history.append(record)
        seen_keys.add(key)
        optimizer.tell(point, record["mean_objective_over_search_seeds"])
        print(
            f"[{FAMILY}] gp         mean_obj={record['mean_objective_over_search_seeds']:.6g} "
            f"min_vpt={record['min_vpt_fraction']:.4f} survives_filter={record['survives_hard_filter']} "
            f"({len(history)}/96)",
            flush=True,
        )

    survivors = [r for r in history if r["survives_hard_filter"]]
    ranked_survivors = sorted(survivors, key=lambda r: r["rank_key"] + [_dedup_key(r["hyperparameters"])])
    ranked_all = sorted(history, key=lambda r: r["rank_key"] + [_dedup_key(r["hyperparameters"])])

    _write_json(
        RESULTS_DIR / "search_history.json",
        {
            "kind": "chaotic_recovery_search_history",
            "design_hash": design["design_hash_sha256"],
            "candidate_count": len(history),
            "survivor_count": len(survivors),
            "history": history,
            "ranked_all_keys": [_dedup_key(r["hyperparameters"]) for r in ranked_all],
            "ranked_survivor_keys": [_dedup_key(r["hyperparameters"]) for r in ranked_survivors],
        },
    )
    print(f"=== search done: {len(history)} candidates, {len(survivors)} survive the hard stability filter ===", flush=True)
    return {
        "design": design, "model_data": model_data, "history": history,
        "ranked_survivors": ranked_survivors, "currents": currents,
    }


class NoEligibleCandidateError(RuntimeError):
    pass


def confirm_and_lock(search_result: dict) -> dict:
    design = search_result["design"]
    model_data = search_result["model_data"]
    ranked_survivors = search_result["ranked_survivors"]
    top12 = ranked_survivors[:TOP_K_FOR_CONFIRMATION]

    confirmations = []
    for candidate in top12:
        per_seed_aggregate = dict(candidate["per_seed_aggregate"])
        for seed in CONFIRMATION_ONLY_SEEDS:
            evaluator = validation.SourceCandidateEvaluator(model_data)
            result = evaluator(candidate["hyperparameters"], seed)
            horizon = len(model_data.validation_cases[0].targets_physical)
            per_seed_aggregate[str(seed)] = validation.augment_aggregate_with_vpt_fraction(
                result["aggregate"], result["rollouts"], horizon
            )
        zero_tol_ok = True
        gate_pass_count = 0
        for seed in ALL_CONFIRMATION_SEEDS:
            agg = per_seed_aggregate[str(seed)]
            nf = agg.get("numerical_failure_count", 0)
            div = agg["divergence_rollout_count"]
            coll = agg["collapse_rollout_count"]
            if nf or div or coll:
                zero_tol_ok = False
            min_vpt = agg.get("min_vpt_fraction")
            if min_vpt is not None and min_vpt >= validation.SOURCE_QUALITY_GATE["min_vpt_fraction_per_rollout"]:
                gate_pass_count += 1

        failing = len(ALL_CONFIRMATION_SEEDS) - gate_pass_count
        eligible = zero_tol_ok and failing <= ROBUST_SEED_RULE["max_failing_seeds_for_vpt_gate"]
        confirmations.append(
            {
                "hyperparameters": candidate["hyperparameters"],
                "per_seed": per_seed_aggregate,
                "zero_tolerance_ok": zero_tol_ok,
                "gate_pass_count": gate_pass_count,
                "failing_seed_count": failing,
                "eligible": eligible,
            }
        )
        print(f"[confirm] gate_pass_count={gate_pass_count}/5 zero_tolerance_ok={zero_tol_ok} eligible={eligible}", flush=True)

    _write_json(
        RESULTS_DIR / "confirmation_results.json",
        {"kind": "confirmation", "design_hash": design["design_hash_sha256"],
         "robust_seed_rule": ROBUST_SEED_RULE, "confirmations": confirmations},
    )

    eligible = [c for c in confirmations if c["eligible"]]
    if not eligible:
        _write_json(
            RESULTS_DIR / "chaotic_selection.json",
            {
                "kind": "selection_failure",
                "design_hash": design["design_hash_sha256"],
                "reason": "zero candidates passed the robust source-quality gate in the 96-candidate recovery search",
                "top12_confirmations": confirmations,
            },
        )
        raise NoEligibleCandidateError(
            f"zero of the top {len(top12)} recovery-search candidates passed the robust gate"
        )

    loco_results = []
    for conf in eligible:
        loco = run_loco_for_candidate(design, FAMILY, conf)
        loco_results.append(loco)
        print(f"[LOCO] median_nrmse={loco['median_nrmse']:.6g} median_vpt={loco['median_vpt']:.1f}", flush=True)
    _write_json(RESULTS_DIR / "loco_results.json", {"kind": "loco", "design_hash": design["design_hash_sha256"], "results": loco_results})

    winner = min(loco_results, key=lambda r: r["loco_rank_key"] + [_dedup_key(r["hyperparameters"])])
    selection = {
        "kind": "selection",
        "design_hash": design["design_hash_sha256"],
        "family": FAMILY,
        "training_currents": search_result["currents"],
        "locked_hyperparameters": winner["hyperparameters"],
        "loco_summary": {
            "median_nrmse": winner["median_nrmse"], "mean_nrmse": winner["mean_nrmse"],
            "worst_nrmse": winner["worst_nrmse"], "median_vpt": winner["median_vpt"],
        },
        "eligible_candidate_count": len(eligible),
        "top_k_confirmed_count": len(top12),
        "recovery_search": True,
    }
    _write_json(RESULTS_DIR / "chaotic_selection.json", selection)

    import hashlib

    lock_hash = hashlib.sha256(
        json.dumps(selection, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(RESULTS_DIR / "chaotic_lock.json", {"selection_sha256": lock_hash, "design_hash": design["design_hash_sha256"]})
    print(f"=== LOCKED chaotic winner: {winner['hyperparameters']} ===", flush=True)
    return selection


def main() -> None:
    search_result = run_search()
    if not search_result["ranked_survivors"]:
        _write_json(
            RESULTS_DIR / "chaotic_selection.json",
            {
                "kind": "selection_failure",
                "design_hash": search_result["design"]["design_hash_sha256"],
                "reason": "zero of 96 candidates survived the search-phase hard stability filter (numerical failure/divergence/collapse on at least one of 3 search seeds)",
            },
        )
        print("CONTROLLED FAILURE: zero candidates survived the search-phase hard stability filter.")
        return
    try:
        confirm_and_lock(search_result)
    except NoEligibleCandidateError as exc:
        print(f"CONTROLLED FAILURE: {exc}")


if __name__ == "__main__":
    main()
