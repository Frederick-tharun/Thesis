"""Focused regression tests for the post-Step-8 integrity audit."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chapter2 import audit_step8 as audit
from chapter2.esn_data import NumpyStandardScaler, StateCurrentScalers, file_sha256
from chapter2.esn_optimisation import ORDINARY_BASELINE, PARAMETER_AWARE


def _metrics(*, diverged: bool = False) -> dict:
    return {
        "valid_prediction_steps": 4,
        "valid_prediction_time": 0.04,
        "diverged": diverged,
        "prediction_collapse_any": False,
    }


def _record(model_type: str, value: float, *, diverged: bool = False) -> dict:
    return {
        "record_id": f"{model_type}-{value}",
        "model_type": model_type,
        "seed": 42,
        "current": 1.67,
        "window": 1,
        "aggregate_nrmse_value": value,
        "metrics": _metrics(diverged=diverged),
    }


def test_audit_strict_json_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        audit.strict_json_text({"invalid": float("nan")})
    assert json.loads(audit.strict_json_text({"defined": False, "value": None})) == {
        "defined": False,
        "value": None,
    }


def test_expected_record_matrix_contains_every_combination_once() -> None:
    keys = audit.expected_record_keys()
    assert len(keys) == 210
    assert sum(key[0] == "known_short" for key in keys) == 90
    assert sum(key[0] == "unseen_short" for key in keys) == 60
    assert sum(key[0] == "known_long" for key in keys) == 30
    assert sum(key[0] == "unseen_long" for key in keys) == 20
    assert sum(key[0] == "continuous" for key in keys) == 10


def test_real_final_model_metadata_matches_seed_model_and_lock() -> None:
    manifest = json.loads(audit.MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    for item in manifest["models"]:
        path = audit.CHAPTER2_ROOT.parent / item["path"]
        with np.load(path, allow_pickle=False) as saved:
            assert all(saved[name].dtype.kind != "O" for name in saved.files)
            metadata = json.loads(str(saved["metadata_json"].item()))
        audit.validate_model_identity(
            item["model_type"], int(item["seed"]), item["configuration"], metadata
        )


def test_real_raw_fixed_record_has_exact_current_and_target_alignment() -> None:
    raw = json.loads(audit.RAW_RESULTS_PATH.read_text(encoding="utf-8"))
    item = next(
        record
        for record in raw["records"]
        if record["record_id"]
        == "known_short__parameter_aware__seed_42__I_3p20__window_1"
    )
    trajectory = audit.load_fixed_trajectory(3.20)
    arrays = audit._load_raw_arrays(item)
    start, stop = item["forecast_range"]
    np.testing.assert_array_equal(arrays["current"], trajectory.current_values[start:stop])
    np.testing.assert_array_equal(arrays["targets"], trajectory.states[start + 1 : stop + 1])


class RecordingModel:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []
        self.warmups: list[np.ndarray] = []
        self.reset_count = 0

    def teacher_forced_warmup(self, inputs: np.ndarray, *, reset: bool) -> None:
        self.warmups.append(np.asarray(inputs).copy())
        if reset:
            self.reset_count += 1

    def predict_one_step(self, value: np.ndarray) -> np.ndarray:
        row = np.asarray(value).copy()
        self.inputs.append(row)
        return row[:3] + 1.0

    def reset_reservoir(self) -> None:
        self.reset_count += 1


def test_replay_uses_aligned_current_recursive_feedback_and_no_boundary_reset(
    tmp_path: Path,
) -> None:
    states = np.zeros((12, 3))
    currents = np.asarray([1.0] * 4 + [2.0] * 4 + [3.0] * 4)
    trajectory = SimpleNamespace(
        states=states,
        current_values=currents,
        time=np.arange(12, dtype=float) * 0.01,
    )
    predictions = np.repeat(np.arange(1.0, 10.0)[:, None], 3, axis=1)
    path = tmp_path / "raw.npz"
    np.savez(
        path,
        predictions=predictions,
        targets=np.zeros((9, 3)),
        pointwise_normalised_error=np.zeros(9),
        time=np.arange(3, 12, dtype=float) * 0.01,
        current=currents[2:11],
    )
    item = {
        "record_id": "synthetic-continuous",
        "family": "continuous",
        "model_type": PARAMETER_AWARE,
        "seed": 42,
        "current": None,
        "window": None,
        "warmup_range": [0, 2],
        "forecast_range": [2, 11],
        "failure_step": None,
        "failure_reason": None,
        "raw_arrays_path": str(path),
        "raw_arrays_sha256": file_sha256(path),
    }
    scalers = StateCurrentScalers(
        NumpyStandardScaler(np.zeros(3), np.ones(3)),
        NumpyStandardScaler(np.zeros(1), np.ones(1)),
    )
    model = RecordingModel()

    result = audit.replay_record(item, model, scalers, trajectory)

    assert result["exact_array_match"]
    assert model.reset_count == 2
    assert len(model.warmups) == 1
    assert [row[3] for row in model.inputs] == currents[2:11].tolist()
    np.testing.assert_array_equal(model.inputs[1][:3], predictions[0])


def test_independent_aggregate_retains_divergent_records() -> None:
    records = [
        _record(PARAMETER_AWARE, 9.0, diverged=True),
        _record(ORDINARY_BASELINE, 1.0),
    ]
    result = audit.independent_family_aggregate(records)
    aware = result["models"][PARAMETER_AWARE]["overall"]
    assert aware["rollout_count"] == 1
    assert aware["mean_nrmse"] == 9.0
    assert aware["divergence_count"] == 1


def test_event_contributor_count_excludes_undefined_divergent_record() -> None:
    defined = {
        "family": "known_long",
        "model_type": PARAMETER_AWARE,
        "seed": 42,
        "current": 1.67,
        "event_metrics": {
            "defined": True,
            "errors": {
                "spike_count_absolute_error": 2.0,
                "spike_count_error_defined": True,
                "burst_count_absolute_error": None,
                "burst_count_error_defined": False,
            },
        },
    }
    undefined = {
        **defined,
        "seed": 456,
        "event_metrics": {"defined": False, "errors": None},
    }
    baseline = {
        **defined,
        "model_type": ORDINARY_BASELINE,
        "event_metrics": {
            "defined": True,
            "errors": {
                "spike_count_absolute_error": 1.0,
                "spike_count_error_defined": True,
                "burst_count_absolute_error": 0.0,
                "burst_count_error_defined": True,
            },
        },
    }
    rows = audit.event_error_rows([defined, undefined, baseline])
    aware = next(
        row
        for row in rows
        if row["current"] == 1.67 and row["model_type"] == PARAMETER_AWARE
    )
    assert aware["rollout_count"] == 2
    assert aware["event_record_contributor_count"] == 1
    assert aware["spike_count_error_contributor_count"] == 1
    assert aware["burst_count_error_contributor_count"] == 0


def test_representative_selection_is_seed42_window1_only() -> None:
    records = [
        {"family": "known_short", "seed": 42, "window": 1, "current": 3.20, "model_type": PARAMETER_AWARE},
        {"family": "known_short", "seed": 42, "window": 1, "current": 3.20, "model_type": ORDINARY_BASELINE},
        {"family": "known_short", "seed": 456, "window": 1, "current": 3.20, "model_type": PARAMETER_AWARE},
        {"family": "known_short", "seed": 42, "window": 2, "current": 3.20, "model_type": PARAMETER_AWARE},
    ]
    selected = audit.representative_records(records, "known_short")
    assert len(selected) == 2
    assert all(item["seed"] == 42 and item["window"] == 1 for item in selected)


def test_tracked_chapter1_hash_still_matches_prebenchmark_record() -> None:
    status = json.loads(audit.STATUS_PATH.read_text(encoding="utf-8"))
    assert status["preflight"]["chapter1_tracked_tree_hash"] == (
        "7f68c47e235ff64ceaf853e37255f851aace45d62167a4ccf1c85b4b077db7f6"
    )
    assert (
        audit.tracked_non_chapter2_tree_hash()
        == audit.reference_chapter1_tree_hash()
    )


def test_atomic_audit_json_and_csv_writes(tmp_path: Path) -> None:
    json_path = tmp_path / "audit.json"
    csv_path = tmp_path / "table.csv"
    audit.write_strict_json(json_path, {"status": "ok", "value": None})
    audit._write_csv(csv_path, [{"seed": 42, "status": "ok"}])
    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "status": "ok",
        "value": None,
    }
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "seed,status",
        "42,ok",
    ]
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("writer_name", ["json", "csv"])
def test_atomic_audit_write_failure_preserves_target_and_cleans_temp(
    tmp_path: Path, monkeypatch, writer_name: str
) -> None:
    target = tmp_path / ("audit.json" if writer_name == "json" else "table.csv")
    target.write_text("historical bytes\n", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(audit.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        if writer_name == "json":
            audit.write_strict_json(target, {"new": True})
        else:
            audit._write_csv(target, [{"new": True}])

    assert target.read_text(encoding="utf-8") == "historical bytes\n"
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))



def test_future_audit_path_is_versioned_new_and_not_historical(
    tmp_path: Path,
) -> None:
    with pytest.raises(audit.Step8AuditError, match="historical"):
        audit.validate_future_audit_path(audit.AUDIT_PATH)
    with pytest.raises(audit.Step8AuditError, match="versioned"):
        audit.validate_future_audit_path(tmp_path / "audit.json")
    existing = tmp_path / "audit_v2.json"
    existing.write_text("{}\n", encoding="utf-8")
    with pytest.raises(audit.Step8AuditError, match="overwrite"):
        audit.validate_future_audit_path(existing)
    destination = tmp_path / "audit_v3.json"
    assert audit.validate_future_audit_path(destination) == destination
