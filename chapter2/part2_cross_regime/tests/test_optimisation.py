"""Optimisation-logic tests: candidate budget, determinism, anchor inclusion."""
from __future__ import annotations

from chapter2.part2_cross_regime import config, optimisation


ANCHOR = {
    "reservoir_size": 100,
    "reservoir_connectivity": 0.08881598524963213,
    "input_scaling": 0.06402022818477646,
    "spectral_radius": 0.4118313967689876,
    "ridge_regularisation": 3.968208883661854e-10,
    "leak_rate": 0.9375840772954693,
}


def test_lhs_candidates_are_deterministic():
    a = optimisation._lhs_candidates(seed=51101)
    b = optimisation._lhs_candidates(seed=51101)
    assert a == b


def test_lhs_candidates_balanced_across_reservoir_sizes():
    candidates = optimisation._lhs_candidates(seed=51101, n_per_size=5)
    sizes = [c["reservoir_size"] for c in candidates]
    assert sizes.count(100) == 5
    assert sizes.count(200) == 5
    assert sizes.count(300) == 5


def test_lhs_candidates_within_bounds():
    for c in optimisation._lhs_candidates(seed=51101):
        assert 0.01 <= c["reservoir_connectivity"] <= 1.0
        assert 0.01 <= c["input_scaling"] <= 3.0
        assert 0.01 <= c["spectral_radius"] <= 3.0
        assert 1.0e-10 <= c["ridge_regularisation"] <= 1.0e-2
        assert 0.01 <= c["leak_rate"] <= 1.0


def test_candidate_list_includes_anchor_first():
    candidates = optimisation.build_candidate_list("regular", ANCHOR)
    assert candidates[0]["_source"] == "anchor"
    for key in ANCHOR:
        assert candidates[0][key] == ANCHOR[key]


def test_candidate_list_deduplicates_against_anchor():
    candidates = optimisation.build_candidate_list("regular", ANCHOR)
    keys = [optimisation._dedup_key({k: v for k, v in c.items() if not k.startswith("_")}) for c in candidates]
    assert len(keys) == len(set(keys))


def test_family_search_seeds_are_distinct_from_part1_seeds():
    part1_seeds = {2026, 2027}
    for family, seeds in config.FAMILY_SEARCH_SEEDS.items():
        assert seeds["lhs"] not in part1_seeds
        assert seeds["gp"] not in part1_seeds


def test_robust_seed_rule_uses_registered_confirmation_seeds():
    assert optimisation.ROBUST_SEED_RULE["seeds"] == list(config.CONFIRMATION_SEEDS)
    assert list(config.CONFIRMATION_SEEDS) == [42, 123, 456, 789, 2026]
