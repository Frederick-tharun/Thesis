"""Phase C tests: candidate budget, seed registration, hard filter logic."""
from __future__ import annotations

from chapter2.part2_cross_regime import chaotic_research as cr


def test_seed_split_matches_frozen_protocol():
    assert cr.SEARCH_SEEDS == [42, 123, 456]
    assert cr.CONFIRMATION_ONLY_SEEDS == [789, 2026]
    assert sorted(cr.ALL_CONFIRMATION_SEEDS) == [42, 123, 456, 789, 2026]


def test_candidate_budget_is_exactly_96():
    assert cr.N_TOTAL == 96
    assert cr.N_ANCHOR + cr.N_PREV_BEST + cr.N_LHS + cr.N_GP == 96


def test_lhs_seed_distinct_from_original_step4_search():
    # Step-4 chaotic family used lhs=51102, gp=61102 (config.FAMILY_SEARCH_SEEDS)
    assert cr.LHS_SEED != 51102
    assert cr.GP_SEED != 61102


def _fake_record(nf=0, div=0, coll=0, min_vpt=0.9):
    return {
        "survives_hard_filter": (nf == 0 and div == 0 and coll == 0),
        "rank_key": [nf, div, coll, -min_vpt, -min_vpt, -min_vpt, 0.01, 0.01, 0.01, 0.0],
    }


def test_hard_filter_excludes_any_failure_divergence_or_collapse():
    assert _fake_record(nf=1)["survives_hard_filter"] is False
    assert _fake_record(div=1)["survives_hard_filter"] is False
    assert _fake_record(coll=1)["survives_hard_filter"] is False
    assert _fake_record()["survives_hard_filter"] is True


def test_build_candidate_list_includes_anchor_and_prev_best_first():
    anchor_hp = {
        "reservoir_size": 100, "reservoir_connectivity": 0.05, "input_scaling": 0.1,
        "spectral_radius": 0.5, "ridge_regularisation": 1e-8, "leak_rate": 0.9,
    }
    candidates = cr.build_candidate_list(anchor_hp)
    assert candidates[0]["_source"] == "anchor"
    assert candidates[1]["_source"] == "prev_best"
    assert len(candidates) == 1 + 1 + 30  # anchor + prev_best + up-to-30 unique LHS
    keys = [cr._dedup_key({k: v for k, v in c.items() if not k.startswith("_")}) for c in candidates]
    assert len(keys) == len(set(keys)), "candidates must be unique"
