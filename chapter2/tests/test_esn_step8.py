"""Synthetic-only tests for the locked Chapter 2 Step 8 workflow."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import chapter2.esn_step8 as step8
from chapter2.esn_config import (
    FINAL_SEEDS,
    STEP8_FINAL_TRAINING_STOP,
    STEP8_FORECAST_TRANSITIONS,
    STEP8_TRAINING_WASHOUT,
    STEP8_WARMUP_TRANSITIONS,
    STEP8_WINDOW_STARTS,
    TRAIN_CURRENTS,
    ESNModelConfig,
    LockedDataset,
)
from chapter2.esn_data import (
    FixedCurrentTrajectory,
    NumpyStandardScaler,
    StateCurrentScalers,
)
from chapter2.esn_model import EchoStateNetwork, TrainingSequence
from chapter2.esn_optimisation import ORDINARY_BASELINE, PARAMETER_AWARE


def _selection() -> dict:
    models = {}
    for model_type in step8.MODEL_TYPES:
        search = {
            key: value
            for key, value in step8.FULL_CONFIGURATIONS[model_type].items()
            if key not in {"bias_scaling", "regularise_bias", "output_dimension"}
        }
        robust = dict(step8.ROBUST_VALIDATION[model_type])
        models[model_type] = {
            "selected_source_trial_index": step8.SOURCE_TRIALS[model_type],
            "best_configuration": search,
            "best_robust_aggregate": robust,
        }
    return {
        "label": "VALIDATION-SELECTED — BENCHMARKS NOT OPENED",
        "step7_complete": True,
        "data_access": {
            "held_out_opened": False,
            "unseen_current_opened": False,
            "continuous_benchmark_opened": False,
            "benchmark_results_present": False,
        },
        "models": models,
    }


@pytest.fixture(scope="module")
def training_prefixes() -> tuple[FixedCurrentTrajectory, ...]:
    count = STEP8_FINAL_TRAINING_STOP + 1
    t = np.arange(count, dtype=float) * 0.01
    base = np.column_stack(
        (
            np.sin(t),
            np.cos(0.5 * t),
            0.001 * np.arange(count, dtype=float),
        )
    )
    return tuple(
        FixedCurrentTrajectory(
            current,
            t,
            base + index,
            np.full(count, current),
        )
        for index, current in enumerate(TRAIN_CURRENTS)
    )


def _metrics(*, diverged: bool = False, collapsed: bool = False) -> dict:
    return {
        "valid_prediction_steps": 4,
        "valid_prediction_time": 0.04,
        "diverged": diverged,
        "prediction_collapse_any": collapsed,
    }


def _record(
    model_type: str,
    value: float,
    *,
    seed: int = 42,
    current: float = 1.67,
    window: int = 1,
    diverged: bool = False,
    collapsed: bool = False,
) -> dict:
    return {
        "record_id": f"{model_type}-{seed}-{current}-{window}-{value}",
        "model_type": model_type,
        "seed": seed,
        "current": current,
        "window": window,
        "aggregate_nrmse_value": value,
        "metrics": _metrics(diverged=diverged, collapsed=collapsed),
    }


def test_exact_selection_validation_and_lock_constants() -> None:
    selection = _selection()
    step8.validate_selected_models(selection)
    for model_type in step8.MODEL_TYPES:
        assert step8.SOURCE_TRIALS[model_type] == selection["models"][model_type]["selected_source_trial_index"]
        assert step8.FULL_CONFIGURATIONS[model_type]["bias_scaling"] == 0.1
        assert step8.FULL_CONFIGURATIONS[model_type]["regularise_bias"] is False
        assert step8.FULL_CONFIGURATIONS[model_type]["output_dimension"] == 3


def test_selection_mismatch_fails_closed() -> None:
    selection = _selection()
    selection["models"][PARAMETER_AWARE]["best_configuration"]["leak_rate"] = 0.5
    with pytest.raises(step8.PreflightError, match="configuration mismatch"):
        step8.validate_selected_models(selection)


def test_dataset_hash_mismatch_fails_before_load(tmp_path: Path) -> None:
    path = tmp_path / "locked.npz"
    path.write_bytes(b"not-the-frozen-file")
    record = LockedDataset(path, "0" * 64, 10, current=1.67)
    with pytest.raises(step8.PreflightError, match="hash mismatch"):
        step8.validate_dataset_hashes((record,))


def test_final_training_uses_only_zero_to_70000(training_prefixes) -> None:
    scalers, sequences = step8.prepare_final_training(
        training_prefixes, PARAMETER_AWARE
    )
    assert [len(sequence.inputs) for sequence in sequences] == [70_000] * 3
    expected = np.concatenate(
        [item.states[:70_000] for item in training_prefixes], axis=0
    )
    assert np.array_equal(scalers.state.mean, np.mean(expected, axis=0))
    assert all(sequence.targets.shape == (70_000, 3) for sequence in sequences)


def test_training_preparation_never_calls_benchmark_loaders(
    training_prefixes, monkeypatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("benchmark loader called")

    monkeypatch.setattr(step8, "load_fixed_trajectory", forbidden)
    monkeypatch.setattr(step8, "load_continuous_benchmark", forbidden)
    _, sequences = step8.prepare_final_training(
        training_prefixes, ORDINARY_BASELINE
    )
    assert len(sequences) == 3


def test_exact_washout_and_five_seeds_are_frozen() -> None:
    assert STEP8_TRAINING_WASHOUT == 2_000
    assert FINAL_SEEDS == (42, 123, 456, 789, 2026)
    assert len(set(FINAL_SEEDS)) == 5


def test_final_model_input_and_output_dimensions() -> None:
    aware = step8.final_model_config(PARAMETER_AWARE, 42)
    baseline = step8.final_model_config(ORDINARY_BASELINE, 42)
    assert (aware.input_dimension, aware.output_dimension) == (4, 3)
    assert (baseline.input_dimension, baseline.output_dimension) == (3, 3)


def test_safe_model_bundle_round_trip_inference(tmp_path: Path) -> None:
    config = ESNModelConfig(
        reservoir_size=8,
        reservoir_connectivity=0.5,
        input_dimension=4,
        seed=42,
    )
    rng = np.random.default_rng(4)
    inputs = rng.normal(size=(30, 4))
    targets = rng.normal(size=(30, 3))
    model = EchoStateNetwork(config).fit(
        (TrainingSequence(inputs, targets),), washout=2
    )
    scalers = StateCurrentScalers(
        NumpyStandardScaler(np.zeros(3), np.ones(3)),
        NumpyStandardScaler(np.zeros(1), np.ones(1)),
    )
    path = tmp_path / "model.npz"
    step8.save_final_model(path, model, scalers, {"model_type": PARAMETER_AWARE, "seed": 42})
    expected = model.predict_one_step(inputs[0])
    model.reset_reservoir()
    loaded, loaded_scalers, metadata = step8.load_final_model(path)
    actual = loaded.predict_one_step(inputs[0])
    assert np.array_equal(actual, expected)
    assert np.array_equal(loaded_scalers.state.scale, np.ones(3))
    assert metadata == {"model_type": PARAMETER_AWARE, "seed": 42}
    with np.load(path, allow_pickle=False) as saved:
        assert "metadata_json" in saved.files


def test_benchmark_access_is_blocked_without_locks_and_models(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(step8, "SELECTION_LOCK", tmp_path / "selected.json")
    monkeypatch.setattr(step8, "EVALUATION_MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(step8, "MODEL_MANIFEST_PATH", tmp_path / "models.json")
    with pytest.raises(step8.BenchmarkAccessError, match="missing"):
        step8.validate_benchmark_gate({})


def test_approved_heldout_window_boundaries_are_exact() -> None:
    windows = step8.fixed_windows()
    assert STEP8_WINDOW_STARTS == (70_000, 80_000, 89_999)
    assert [item["warmup_range"] for item in windows] == [
        [70_000, 72_000],
        [80_000, 82_000],
        [89_999, 91_999],
    ]
    assert [item["forecast_range"] for item in windows] == [
        [72_000, 80_000],
        [82_000, 90_000],
        [91_999, 99_999],
    ]
    assert all(
        item["warmup_range"][1] - item["warmup_range"][0] == STEP8_WARMUP_TRANSITIONS
        and item["forecast_range"][1] - item["forecast_range"][0] == STEP8_FORECAST_TRANSITIONS
        for item in windows
    )


def test_scored_intervals_do_not_overlap_and_shared_transition_is_unscored() -> None:
    windows = step8.fixed_windows()
    scored = [set(range(*item["forecast_range"])) for item in windows]
    assert not scored[0] & scored[1]
    assert not scored[1] & scored[2]
    assert 89_999 in scored[1]
    assert 89_999 in range(*windows[2]["warmup_range"])
    assert 89_999 not in scored[2]


def test_unseen_current_is_rejected_by_training_prefix_loader() -> None:
    record = next(item for item in step8.FIXED_DATASETS if item.current == 3.29)
    with pytest.raises(step8.PreflightError, match="rejects"):
        step8.load_training_prefix(record)


class RecordingModel:
    def __init__(self) -> None:
        self.warmups: list[tuple[np.ndarray, bool]] = []
        self.inputs: list[np.ndarray] = []
        self.reset_calls = 0

    def teacher_forced_warmup(self, inputs, *, reset=True):
        self.warmups.append((np.asarray(inputs).copy(), reset))
        if reset:
            self.reset_calls += 1

    def predict_one_step(self, value):
        row = np.asarray(value, dtype=float).copy()
        self.inputs.append(row)
        return row[:3] + 1.0

    def reset_reservoir(self):
        self.reset_calls += 1


def _identity_scalers() -> StateCurrentScalers:
    return StateCurrentScalers(
        NumpyStandardScaler(np.zeros(3), np.ones(3)),
        NumpyStandardScaler(np.zeros(1), np.ones(1)),
    )


def test_recursive_feedback_uses_previous_prediction() -> None:
    model = RecordingModel()
    states = np.zeros((8, 3))
    currents = np.ones(8)
    predictions, failure, _ = step8.recursive_forecast(
        model,
        _identity_scalers(),
        ORDINARY_BASELINE,
        states,
        currents,
        warmup_range=(0, 2),
        forecast_range=(2, 6),
    )
    assert failure is None
    assert np.array_equal(predictions[:, 0], np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.array_equal(model.inputs[1][:3], predictions[0])


def test_aware_current_is_transition_aligned() -> None:
    model = RecordingModel()
    states = np.zeros((8, 3))
    currents = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
    step8.recursive_forecast(
        model,
        _identity_scalers(),
        PARAMETER_AWARE,
        states,
        currents,
        warmup_range=(0, 2),
        forecast_range=(2, 6),
    )
    assert [row[3] for row in model.inputs] == [12.0, 13.0, 14.0, 15.0]


def test_continuous_boundaries_do_not_reset_or_rewarm() -> None:
    model = RecordingModel()
    states = np.zeros((12, 3))
    currents = np.array([1.0] * 4 + [2.0] * 4 + [3.0] * 4)
    step8.recursive_forecast(
        model,
        _identity_scalers(),
        PARAMETER_AWARE,
        states,
        currents,
        warmup_range=(0, 2),
        forecast_range=(2, 11),
    )
    assert len(model.warmups) == 1
    assert model.warmups[0][1] is True
    assert model.reset_calls == 2
    assert len(model.inputs) == 9


def test_equal_weight_aggregation_is_hand_calculated() -> None:
    records = [
        _record(PARAMETER_AWARE, 1.0, window=1),
        _record(PARAMETER_AWARE, 3.0, window=2),
        _record(ORDINARY_BASELINE, 2.0, window=1),
        _record(ORDINARY_BASELINE, 4.0, window=2),
    ]
    result = step8.aggregate_family(records)
    assert result["models"][PARAMETER_AWARE]["overall"]["mean_nrmse"] == 2.0
    assert result["models"][ORDINARY_BASELINE]["overall"]["mean_nrmse"] == 3.0
    assert result["paired_baseline_minus_aware_nrmse"]["mean"] == 1.0


def test_divergent_and_collapsed_rollouts_remain_in_aggregates() -> None:
    records = [
        _record(PARAMETER_AWARE, 1_000_000.0, diverged=True, collapsed=True),
        _record(ORDINARY_BASELINE, 2.0, diverged=True, collapsed=True),
    ]
    result = step8.aggregate_family(records)
    aware = result["models"][PARAMETER_AWARE]["overall"]
    assert aware["rollout_count"] == 1
    assert aware["mean_nrmse"] == 1_000_000.0
    assert aware["divergence_count"] == 1
    assert aware["collapse_count"] == 1


def test_strict_json_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        step8.strict_json_text({"bad": float("nan")})
    assert json.loads(step8.strict_json_text({"undefined": None})) == {"undefined": None}


def test_figure_selection_rules_are_deterministic() -> None:
    manifest = step8.build_evaluation_manifest(
        {"inspection_basis": ["hashes", "headers", "shapes", "transition counts"]}
    )
    assert manifest["plot_selection"] == {
        "representative_seed": 42,
        "representative_window": 1,
        "cherry_picking_allowed": False,
        "formats": ["png", "pdf"],
    }


def test_synthetic_spike_and_burst_detection() -> None:
    states = np.zeros((360, 3))
    peaks = [10, 30, 50, 130, 150, 170, 270, 290, 310]
    states[peaks, 0] = 2.0
    result = step8.event_metrics(states.copy(), states.copy())
    assert result["defined"] is True
    assert result["target"]["spike_count"] == 9
    assert result["target"]["burst_count"] == 3
    assert result["errors"]["spike_count_absolute_error"] == 0.0
    assert result["errors"]["spike_time_mean_absolute_error"] == 0.0


def test_nonfinite_event_metrics_are_explicitly_undefined() -> None:
    target = np.zeros((50, 3))
    prediction = target.copy()
    prediction[20, 0] = np.nan
    result = step8.event_metrics(prediction, target)
    assert result["defined"] is False
    assert result["errors"] is None
    assert result["prediction"]["reason"] == "non_finite_or_invalid_trajectory"




def test_divergent_interval_event_metrics_are_invalidated() -> None:
    states = np.zeros((100, 3))
    states[[20, 40, 60, 80], 0] = 2.0
    event = step8.event_metrics(states, states)
    corrected = step8.invalidate_divergent_event_metrics(
        event, {"diverged": True}
    )
    assert corrected["defined"] is False
    assert corrected["errors"] is None
    assert corrected["target"] == event["target"]
    assert corrected["reason"] == "event metrics invalidated by interval divergence"
    unchanged = step8.invalidate_divergent_event_metrics(
        event, {"diverged": False}
    )
    assert unchanged is event
def test_step8_helpers_do_not_write_chapter1_files() -> None:
    before = step8.tracked_non_chapter2_tree_hash()
    step8.fixed_windows()
    step8.validate_selected_models(_selection())
    after = step8.tracked_non_chapter2_tree_hash()
    assert after == before


def test_chapter1_guard_detects_protected_changes_and_ignores_unrelated_files(
    tmp_path: Path,
) -> None:
    chapter1_path = tmp_path / "config.py"
    unrelated_path = tmp_path / "scripts/analysis/estimate_hr_lyapunov.py"
    chapter1_path.write_text("protected = 1\n", encoding="utf-8")
    unrelated_path.parent.mkdir(parents=True)
    unrelated_path.write_text("approved = 1\n", encoding="utf-8")
    tracked_names = ("config.py", "scripts/analysis/estimate_hr_lyapunov.py")

    baseline = step8._chapter1_tree_hash(tmp_path, tracked_names)
    unrelated_path.write_text("approved = 2\n", encoding="utf-8")
    assert step8._chapter1_tree_hash(tmp_path, tracked_names) == baseline

    chapter1_path.write_text("protected = 2\n", encoding="utf-8")
    assert step8._chapter1_tree_hash(tmp_path, tracked_names) != baseline
