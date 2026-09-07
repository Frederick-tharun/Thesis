"""Contracts for the isolated scenario-optimised cross-regime extension."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from chapter2.cross_regime_config import MIXED_BLOCK_ORDERS, block_order
from chapter2.esn_data import FixedCurrentTrajectory
from chapter2.esn_optimisation import atomic_write_json
from chapter2.cross_regime_improvement import config
from chapter2.cross_regime_improvement import experiment
from chapter2.cross_regime_improvement import figures
from chapter2.cross_regime_improvement import optimisation as opt
from chapter2.cross_regime_improvement import protection


def trajectory(current: float, transitions: int = 70, offset: float = 0.0):
    time = np.arange(transitions + 1, dtype=float) * 0.01
    states = np.column_stack((time + current + offset, 2 * time, -time))
    return FixedCurrentTrajectory(
        current, time, states, np.full(transitions + 1, current)
    )


def scenario_data(scenario: str):
    return opt.prepare_scenario_data(
        scenario,
        {current: trajectory(current) for current in config.SCENARIO_TRAINING_CURRENTS[scenario]},
        fitting_range=(0, 40),
        validation_windows=((1, (40, 50), (50, 65)),),
    )


def metric(nrmse=1.0, vpt=0.5, diverged=False, collapse=False):
    return {
        "nrmse_state": nrmse,
        "rmse_state": nrmse,
        "valid_prediction_time": vpt,
        "diverged": diverged,
        "prediction_collapse_any": collapse,
        "r2_macro": 0.5,
        "correlation_macro": 0.8,
    }


def rollout(nrmse=1.0, vpt=0.5, diverged=False, failed=False, current=1.67):
    return {
        "current": current,
        "aggregate_nrmse_value": 1_000_000.0 if failed else nrmse,
        "numerical_failure": failed,
        "metrics": metric(None if failed else nrmse, vpt, diverged or failed),
    }


def ranked(rollouts, value="a"):
    return {
        "aggregate": opt.aggregate_rollouts(rollouts),
        "hyperparameters": {
            "reservoir_size": 100,
            "reservoir_connectivity": 0.1,
            "input_scaling": 0.2,
            "spectral_radius": 0.3,
            "ridge_regularisation": 1e-6,
            "leak_rate": 0.4 if value == "a" else 0.5,
        },
    }


def test_regime_definitions_are_exact_disjoint_and_complete():
    assert config.REGULAR_CURRENTS == (1.67, 3.29, 3.50)
    assert config.CHAOTIC_CURRENTS == (3.20, 3.34)
    assert not set(config.REGULAR_CURRENTS) & set(config.CHAOTIC_CURRENTS)
    assert set(config.REGULAR_CURRENTS + config.CHAOTIC_CURRENTS) == set(config.ALL_CURRENTS)


@pytest.mark.parametrize(
    ("scenario", "forbidden"),
    (("regular_to_chaotic", 3.20), ("chaotic_to_regular", 1.67)),
)
def test_scenario_loader_rejects_opposite_regime_before_io(scenario, forbidden):
    calls = []
    with pytest.raises(PermissionError):
        opt._authorised_load(scenario, forbidden, 70_000, lambda *x: calls.append(x))
    assert calls == []


def test_mixed_order_is_the_existing_deterministic_complete_block_order():
    assert {seed: block_order("mixed_shuffled", seed) for seed in config.SEEDS} == MIXED_BLOCK_ORDERS


def test_samples_inside_blocks_are_chronological_and_no_cross_block_pair_exists():
    data = scenario_data("mixed_shuffled")
    sequences = opt.training_sequences(data, 42)
    assert len(sequences) == 5
    for current, sequence in zip(block_order("mixed_shuffled", 42), sequences):
        source = data.trajectories[current]
        expected_targets = data.scalers.transform_targets(source.states[1:41])
        np.testing.assert_array_equal(sequence.targets, expected_targets)
        assert np.all(np.diff(source.time[:41]) > 0)
    # Five separate objects prove no last-state/next-block-first-state pair exists.
    assert sum(len(item.inputs) for item in sequences) == 5 * 40


def test_model_dimensions_and_esn_source_are_locked():
    parameters = {
        "reservoir_size": 100, "reservoir_connectivity": .1,
        "input_scaling": .2, "spectral_radius": .3,
        "ridge_regularisation": 1e-6, "leak_rate": .4,
    }
    model_config = opt.esn_config(parameters, 42)
    assert (model_config.input_dimension, model_config.output_dimension) == (4, 3)
    protection.assert_esn_source_unchanged()


def test_scaler_fits_only_scenario_fitting_inputs_and_validation_is_separate():
    currents = config.REGULAR_CURRENTS
    blocks = {current: trajectory(current) for current in currents}
    for block in blocks.values():
        block.states[40:] += 1e8
    data = opt.prepare_scenario_data(
        "regular_to_chaotic", blocks, fitting_range=(0, 40),
        validation_windows=((1, (40, 50), (50, 65)),),
    )
    expected = np.concatenate([blocks[current].states[:40] for current in currents]).mean(axis=0)
    np.testing.assert_allclose(data.scalers.state.mean, expected)
    assert data.fitting_range[1] <= data.validation_windows[0][1][0]


def test_numerical_failure_score_and_thresholds_are_unchanged():
    assert config.NONFINITE_FAILURE_SCORE == 1_000_000.0
    assert (config.VALID_PREDICTION_THRESHOLD, config.DIVERGENCE_THRESHOLD, config.COLLAPSE_THRESHOLD) == (0.4, 5.0, 0.05)


def test_divergence_precedes_nrmse_in_authoritative_ranking():
    stable = ranked([rollout(nrmse=100.0, diverged=False)])
    divergent = ranked([rollout(nrmse=0.001, diverged=True)], value="b")
    assert opt.candidate_rank_key(stable) < opt.candidate_rank_key(divergent)


def test_numerical_failure_precedes_divergence_in_authoritative_ranking():
    divergent = ranked([rollout(nrmse=100.0, diverged=True)])
    failed = ranked([rollout(nrmse=0.001, failed=True)], value="b")
    assert opt.candidate_rank_key(divergent) < opt.candidate_rank_key(failed)


def test_candidate_ranking_has_deterministic_hyperparameter_tie_break():
    left = ranked([rollout()], "a")
    right = ranked([rollout()], "b")
    assert opt.candidate_rank_key(left) != opt.candidate_rank_key(right)
    assert sorted([right, left], key=opt.candidate_rank_key) == [left, right]


def test_five_seed_confirmation_aggregate_records_every_seed_and_worst_seed():
    results = []
    for index, seed in enumerate(config.SEEDS):
        rolls = [rollout(nrmse=index + 1.0, vpt=5 - index)]
        results.append({"model_seed": seed, "rollouts": rolls, "aggregate": opt.aggregate_rollouts(rolls)})
    aggregate = opt._confirmation_aggregate(results)
    assert aggregate["seed_count"] == 5
    assert set(aggregate["per_seed"]) == {str(x) for x in config.SEEDS}
    assert aggregate["worst_seed"] == 2026


def test_historical_results_cannot_supply_optimisation_metrics():
    source = Path(opt.__file__).read_text(encoding="utf-8")
    assert "corrected_results.json" not in source
    assert "cross_regime_raw_results.json" not in source
    metadata = opt.describe_design()
    assert metadata["model_dimensions"] == [4, 3]


def test_recursive_rollout_uses_prediction_and_supplied_current_not_future_truth():
    unit = experiment.StateCurrentScalers(
        experiment.NumpyStandardScaler(np.zeros(3), np.ones(3)),
        experiment.NumpyStandardScaler(np.zeros(1), np.ones(1)),
    )
    states = np.arange(30, dtype=float).reshape(10, 3)
    currents = np.linspace(1, 2, 10)
    spy = experiment._FeedbackSpy()
    first, _, _ = experiment.recursive_forecast(spy, unit, states, currents, warmup_range=(0, 2), forecast_range=(2, 9))
    states[3:] = -999
    second, _, _ = experiment.recursive_forecast(experiment._FeedbackSpy(), unit, states, currents, warmup_range=(0, 2), forecast_range=(2, 9))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.asarray(spy.inputs)[:, 3], currents[2:9])


def test_numerical_failure_handling_retains_prefix_and_aligns_output():
    unit = experiment.StateCurrentScalers(
        experiment.NumpyStandardScaler(np.zeros(3), np.ones(3)),
        experiment.NumpyStandardScaler(np.zeros(1), np.ones(1)),
    )
    predicted, step, reason = experiment.recursive_forecast(
        experiment._FeedbackSpy(fail_at=3), unit, np.zeros((10, 3)), np.ones(10),
        warmup_range=(0, 2), forecast_range=(2, 9),
    )
    assert predicted.shape == (7, 3)
    assert step == 2 and reason == "non_finite_prediction"
    assert np.isfinite(predicted[:2]).all() and np.isnan(predicted[2:]).all()


def test_strict_json_serialisation_rejects_nonfinite(tmp_path):
    with pytest.raises(ValueError):
        atomic_write_json(tmp_path / "bad.json", {"value": np.nan})


def test_output_overwrite_protection(tmp_path):
    path = tmp_path / "locked.txt"
    path.write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        protection.refuse_existing(path)
    protection.refuse_existing(path, resume=True)


def test_final_evaluation_is_blocked_without_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "SELECTION_PATH", tmp_path / "missing.json")
    with pytest.raises(Exception, match="locked"):
        experiment._selection()


def test_selection_artifact_explicitly_records_no_baseline_tuning():
    assert "historical_baseline_used_for_tuning" in Path(opt.__file__).read_text(encoding="utf-8")


def test_chapter1_and_completed_chapter2_tracked_paths_are_unchanged():
    protection.assert_tracked_protected_unchanged()


def test_existing_cross_regime_baseline_hash_inventory_is_unchanged():
    result = protection.assert_baseline_artifacts_unchanged()
    assert result["baseline_hashes_checked"] >= 370


def _figure_records():
    records = []
    for scenario in config.SCENARIO_TRAINING_CURRENTS:
        for seed in config.SEEDS:
            for current in config.ALL_CURRENTS:
                regime = config.regime_for_current(current)
                records.append(
                    {
                        "family": "fixed_short", "scenario": scenario,
                        "seed": seed, "current": current, "window": 1,
                        "matrix_code": config.MATRIX_CODES[(scenario, regime)],
                        "numerical_failure": False, "metrics": metric(.2, 1.0),
                        "raw_arrays_path": "unused.npz",
                    }
                )
            for schedule in config.CONTINUOUS_SCHEDULES:
                records.append(
                    {
                        "family": "continuous", "scenario": scenario,
                        "seed": seed, "current": None, "window": None,
                        "schedule": schedule, "matrix_code": None,
                        "numerical_failure": False, "metrics": metric(.3, 1.0),
                        "raw_arrays_path": "unused.npz",
                    }
                )
    return records


def test_all_figure_generators_smoke(monkeypatch, tmp_path):
    records = _figure_records()
    time = np.arange(200, dtype=float) * .01
    arrays = {
        "time": time,
        "targets": np.column_stack((np.sin(time), np.cos(time), np.sin(2*time))),
        "predictions": np.column_stack((np.sin(time), np.cos(time), np.sin(2*time))),
        "current": np.repeat([1.67, 3.20], 100),
        "pointwise_normalised_error": np.zeros(200),
    }
    monkeypatch.setattr(figures, "_arrays", lambda record: arrays)
    monkeypatch.setattr(figures, "load_fixed_trajectory", lambda current: trajectory(current, 72_100))
    monkeypatch.setattr(figures, "load_strict_json", lambda path: {"records": records})
    def save(fig, stem, root):
        plt.close(fig)
        return [str(root / f"{stem}.png"), str(root / f"{stem}.pdf")]
    monkeypatch.setattr(figures, "_save", save)
    paths = []
    paths += figures.regime_overview(tmp_path)
    paths += figures.prediction_figure(records, "regular_to_chaotic", 1.67, "rr", "RR", tmp_path)
    paths += figures.mixed_prediction_figure(records, tmp_path)
    paths += figures.matrix_heatmaps(records, tmp_path)
    paths += figures.vpt_comparison(records, tmp_path)
    paths += figures.seed_robustness(records, tmp_path)
    paths += figures.phase_portraits(records, tmp_path)
    paths += figures.continuous_switch_figure(records, tmp_path)
    paths += figures.baseline_vs_improved(records, tmp_path)
    assert len(paths) == 22


def test_summary_matrix_contains_all_six_direction_codes():
    assert set(config.MATRIX_CODES.values()) == {"RR", "RC", "CC", "CR", "MR", "MC"}


def test_continuous_schedule_design_is_exact():
    assert set(config.CONTINUOUS_SCHEDULES) == {
        "regular_then_chaotic", "chaotic_then_regular", "alternating_mixed"
    }


def test_pilot_contract_small_subsets(tmp_path):
    report = experiment.run_pilot(output_path=tmp_path / "pilot.json")
    assert report["passed"] is True
    assert report["synthetic_development_subsets_only"] is True
    assert len(report["scenario_checks"]) == 3
    loaded = json.loads((tmp_path / "pilot.json").read_text(encoding="utf-8"))
    assert loaded["passed"] is True
