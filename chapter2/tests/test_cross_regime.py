"""Synthetic checks for cross-regime isolation, feedback, persistence and auditing."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from chapter2 import cross_regime as core
from chapter2 import run_cross_regime as runner
from chapter2 import audit_cross_regime as audit
from chapter2.cross_regime_config import model_config
from chapter2.esn_data import NumpyStandardScaler, StateCurrentScalers, file_sha256
from chapter2.esn_model import EchoStateNetwork, TrainingSequence


def unit_scalers():
    return StateCurrentScalers(NumpyStandardScaler(np.zeros(3), np.ones(3)),
                               NumpyStandardScaler(np.zeros(1), np.ones(1)))


def test_training_uses_only_allocated_blocks_and_excludes_last_target(monkeypatch):
    monkeypatch.setattr(core, "RAW_TRAINING_TRANSITIONS", {"chaotic_to_regular": {3.20: 6, 3.34: 6}})
    trajectories = {c: runner.synthetic_prefix(c, 6) for c in (3.20, 3.34)}
    trajectories[3.20].states[-1] = 1e9
    scalers, sequences, provenance = core.prepare_training("chaotic_to_regular", 42,
                                                         trajectories=trajectories, washout=2)
    inputs = np.concatenate([t.states[:-1] for t in trajectories.values()])
    np.testing.assert_array_equal(scalers.state.mean, inputs.mean(axis=0))
    assert [len(s.inputs) for s in sequences] == [6, 6]
    assert provenance["effective_samples"] == 8
    assert scalers.current.mean[0] == pytest.approx(3.27)
    with pytest.raises(core.CrossRegimeError, match="membership"):
        core.prepare_training("chaotic_to_regular", 42, trajectories={3.20: trajectories[3.20]}, washout=2)


def test_training_prefix_never_exposes_heldout_suffix(tmp_path, monkeypatch):
    path = tmp_path / "fixed.npz"
    data = np.arange(2200, dtype=np.float64)
    np.savez(path, t=data * .01, x=data, y=data + 1, z=data + 2, I=np.full(2200, 3.20))
    record = SimpleNamespace(path=path, sha256=file_sha256(path), state_count=2200)
    monkeypatch.setattr(core, "fixed_dataset", lambda current: record)
    result = core.load_training_prefix(3.20, 2001)
    assert result.states.shape == (2002, 3)
    assert result.states[-1, 0] == 2001
    path.write_bytes(b"corrupt")
    with pytest.raises(core.CrossRegimeError, match="hash mismatch"):
        core.load_training_prefix(3.20, 2001)


def test_independent_reservoir_blocks_add_statistics():
    model = EchoStateNetwork(replace(model_config(42), reservoir_size=8))
    rng = np.random.default_rng(12)
    blocks = [TrainingSequence(rng.normal(size=(12, 4)), rng.normal(size=(12, 3))) for _ in range(2)]
    separate = [model.accumulate_ridge_statistics([s], washout=2) for s in blocks]
    combined = model.accumulate_ridge_statistics(blocks, washout=2)
    np.testing.assert_allclose(combined.gram, separate[0].gram + separate[1].gram)
    np.testing.assert_allclose(combined.cross, separate[0].cross + separate[1].cross)
    assert combined.sample_count == 20


class FeedbackSpy:
    def __init__(self, fail_at=None):
        self.inputs = []
        self.warmups = []
        self.resets = 0
        self.fail_at = fail_at

    def teacher_forced_warmup(self, inputs, reset=True):
        self.warmups.append((inputs.copy(), reset))

    def predict_one_step(self, value):
        self.inputs.append(value.copy())
        if len(self.inputs) == self.fail_at:
            return np.full(3, np.nan)
        return value[:3] + 1

    def reset_reservoir(self):
        self.resets += 1


def test_recursive_feedback_ignores_future_truth_and_never_rewarms_at_switch():
    model = FeedbackSpy()
    states = np.arange(30, dtype=float).reshape(10, 3)
    currents = np.array([1., 1., 1., 3., 3., 2., 2., 2., 2., 2.])
    predicted, step, reason = core.recursive_forecast(model, unit_scalers(), states, currents,
                                                     warmup_range=(0, 2), forecast_range=(2, 9))
    np.testing.assert_array_equal(predicted, states[2] + np.arange(1, 8)[:, None])
    np.testing.assert_array_equal(np.array(model.inputs)[:, 3], currents[2:9])
    assert len(model.warmups) == 1 and model.resets == 1
    assert step is None and reason is None
    states[3:] = -1e8
    again, _, _ = core.recursive_forecast(FeedbackSpy(), unit_scalers(), states, currents,
                                         warmup_range=(0, 2), forecast_range=(2, 9))
    np.testing.assert_array_equal(predicted, again)


def test_nonfinite_forecast_retains_prefix_and_marks_suffix():
    predictions, step, reason = core.recursive_forecast(FeedbackSpy(fail_at=3), unit_scalers(),
        np.zeros((10, 3)), np.ones(10), warmup_range=(0, 2), forecast_range=(2, 9))
    assert step == 2 and reason == "non_finite_prediction"
    assert np.isfinite(predictions[:2]).all() and np.isnan(predictions[2:]).all()


def test_model_training_serialization_and_resume_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "MODEL_MANIFEST_PATH", tmp_path / "models.json")
    monkeypatch.setattr(runner, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "EXPECTED_MODELS", 1)
    monkeypatch.setattr(runner, "expected_model_keys", lambda: {("mixed_shuffled", 42)})
    config = replace(model_config(42), reservoir_size=8, ridge_regularisation=1e-3)
    monkeypatch.setattr(runner, "model_config", lambda seed: config)
    rng = np.random.default_rng(10)
    model = EchoStateNetwork(config).fit([TrainingSequence(rng.normal(size=(30,4)), rng.normal(size=(30,3)))], washout=2)
    monkeypatch.setattr(runner, "train_scenario_model", lambda *args: (model, unit_scalers(), {"effective_samples": 130000}))
    first = runner.train_models("protocol-a")
    before = file_sha256(tmp_path / "models.json")
    monkeypatch.setattr(runner, "train_scenario_model", lambda *args: pytest.fail("resume retrained a model"))
    assert runner.train_models("protocol-a") == first
    assert file_sha256(tmp_path / "models.json") == before
    with pytest.raises(core.CrossRegimeError, match="identity mismatch"):
        runner.train_models("protocol-b")


def test_freeze_requires_current_passing_pilot(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(runner, "PILOT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "source_hashes", lambda: {"source": "current"})
    core.atomic_write_json(tmp_path / "pilot_report.json", {"passed": True, "source_hashes": {"source": "old"}})
    with pytest.raises(core.CrossRegimeError, match="passing pilot"):
        runner.freeze_protocol()
    assert not (tmp_path / "manifest.json").exists()


def test_record_matrix_rejects_missing_and_duplicate_entries():
    rows = [{"record_id": key} for key in core.expected_record_ids()]
    assert len(rows) == 345
    core.validate_record_matrix(rows)
    with pytest.raises(core.CrossRegimeError, match="matrix mismatch"):
        core.validate_record_matrix(rows[:-1])
    with pytest.raises(core.CrossRegimeError, match="duplicate"):
        core.validate_record_matrix(rows + rows[:1])


def test_audit_checks_source_targets_and_failure_penalty(tmp_path, monkeypatch):
    trajectory = runner.synthetic_prefix(3.20, 100000)
    begin, stop = 72000, 80000
    item = core.build_evaluation_record(identifier=core.record_id("fixed_short", "regular_to_chaotic", 42, current=3.20, window=1),
        family="fixed_short", scenario="regular_to_chaotic", seed=42, scalers=unit_scalers(),
        predictions=trajectory.states[begin+1:stop+1] + .001, targets=trajectory.states[begin+1:stop+1],
        times=trajectory.time[begin+1:stop+1], currents=trajectory.current_values[begin:stop], raw_path=tmp_path / "raw.npz",
        warmup_range=(70000,72000), forecast_range=(begin,stop), failure_step=None, failure_reason=None, current=3.20, window=1)
    monkeypatch.setattr(audit, "validate_record_matrix", lambda records: None)
    monkeypatch.setattr(audit, "load_fixed_trajectory", lambda current: trajectory)
    monkeypatch.setattr(audit, "load_schedules", lambda manifest: {})
    monkeypatch.setattr(audit, "strict_load_json", lambda path: {})
    monkeypatch.setattr(audit, "load_model_bundle", lambda path: (None, unit_scalers(), {}))
    models = {"models": [{"scenario":"regular_to_chaotic", "seed":42, "path":"unused"}]}
    assert audit.audit_records({"records":[item]}, models)["record_count"] == 1
    item["aggregate_nrmse_value"] = 999
    with pytest.raises(audit.CrossRegimeAuditError, match="penalty"):
        audit.audit_records({"records":[item]}, models)
    item["aggregate_nrmse_value"] = item["metrics"]["nrmse_state"]
    trajectory.states[begin+1, 0] += 1
    with pytest.raises(audit.CrossRegimeAuditError, match="source trajectory"):
        audit.audit_records({"records":[item]}, models)


def test_strict_json_rejects_nonfinite_and_preserves_previous_file(tmp_path):
    path = tmp_path / "value.json"
    core.atomic_write_json(path, {"value": 1})
    with pytest.raises(ValueError):
        core.atomic_write_json(path, {"value": float("nan")})
    assert core.strict_load_json(path) == {"value": 1}
