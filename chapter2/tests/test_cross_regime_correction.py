"""Synthetic tests of immutable-artifact post-hoc numerical correction."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chapter2 import audit_cross_regime as audit
from chapter2 import correct_cross_regime_numerics as correction
from chapter2 import cross_regime as core
from chapter2 import run_cross_regime as runner
from chapter2.cross_regime_config import ALL_CURRENTS, CONTINUOUS_SCHEDULES, SCENARIO_TRAINING_CURRENTS, SEEDS
from chapter2.esn_data import NumpyStandardScaler, StateCurrentScalers, file_sha256
from chapter2.esn_model import EchoStateNetwork


def _scalers():
    return StateCurrentScalers(NumpyStandardScaler(np.zeros(3), np.ones(3)),
                               NumpyStandardScaler(np.zeros(1), np.ones(1)))


def _historical_record(tmp_path, *, unsafe=False):
    trajectory = runner.synthetic_prefix(3.20, 100000)
    begin, stop = 72000, 80000
    targets = trajectory.states[begin + 1:stop + 1].copy()
    predictions = targets.copy()
    predictions[2, 0] = 1e200 if unsafe else np.inf
    path = tmp_path / "historical.npz"
    arrays = {"predictions": predictions, "targets": targets,
              "pointwise_normalised_error": np.zeros(len(targets)),
              "time": trajectory.time[begin + 1:stop + 1].copy(),
              "current": trajectory.current_values[begin:stop].copy()}
    np.savez(path, **arrays)
    record = {
        "record_id": core.record_id("fixed_short", "regular_to_chaotic", 42,
                                    current=3.20, window=1),
        "family": "fixed_short", "scenario": "regular_to_chaotic", "seed": 42,
        "current": 3.20, "window": 1, "schedule": None,
        "evaluation_class": "cross-regime generalization",
        "warmup_range": [70000, 72000], "forecast_range": [begin, stop],
        "failure_step": None, "failure_reason": None, "numerical_failure": False,
        "valid_prefix_steps": len(predictions), "metrics": {}, "event_metrics": {},
        "aggregate_nrmse_value": 0.0,
        "raw_arrays_path": str(path), "raw_arrays_sha256": file_sha256(path),
    }
    return record, arrays, trajectory


def _forbid_forecasts(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("post-hoc correction invoked forecast generation or training")
    for name in ("recursive_forecast", "train_scenario_model"):
        monkeypatch.setattr(core, name, forbidden)
    for name in ("recursive_forecast", "train_scenario_model", "execute_benchmarks", "train_models", "run_full"):
        monkeypatch.setattr(runner, name, forbidden)
    monkeypatch.setattr(EchoStateNetwork, "predict_one_step", forbidden)
    monkeypatch.setattr(EchoStateNetwork, "fit", forbidden)


@pytest.mark.parametrize("unsafe", [False, True], ids=["physical-inf", "finite-metric-overflow"])
def test_correct_existing_artifact_preserves_bytes_and_never_forecasts(tmp_path, monkeypatch, unsafe):
    record, arrays, trajectory = _historical_record(tmp_path, unsafe=unsafe)
    old_record = deepcopy(record)
    original_bytes = Path(record["raw_arrays_path"]).read_bytes()
    _forbid_forecasts(monkeypatch)
    with np.errstate(over="raise", invalid="raise"):
        corrected, change = correction.correct_record(
            record, normalisation_scale=np.ones(3), trajectory=trajectory
        )
    assert corrected["failure_step"] == 2
    assert corrected["valid_prefix_steps"] == 2
    assert corrected["numerical_failure"] is True
    assert corrected["aggregate_nrmse_value"] == core.NONFINITE_FAILURE_SCORE
    assert corrected["failure_reason"] == (
        "metric_unsafe_residual_square" if unsafe else "non_finite_physical_prediction"
    )
    assert record == old_record
    path = Path(record["raw_arrays_path"])
    assert path.read_bytes() == original_bytes
    assert file_sha256(path) == record["raw_arrays_sha256"] == corrected["raw_arrays_sha256"]
    with np.load(path, allow_pickle=False) as original:
        np.testing.assert_array_equal(original["predictions"], arrays["predictions"])
    assert change["old"]["failure_step"] is None
    assert change["new"]["failure_step"] == 2
    assert change["old"]["aggregate_nrmse_value"] == 0.0
    assert change["new"]["aggregate_nrmse_value"] == core.NONFINITE_FAILURE_SCORE
    assert {item["field"] for item in change["changes"]} >= {
        "failure_step", "numerical_failure", "aggregate_nrmse_value"
    }


def test_corrected_metadata_passes_shared_auditor_while_raw_pointwise_is_historical(tmp_path, monkeypatch):
    record, _, trajectory = _historical_record(tmp_path)
    corrected, _ = correction.correct_record(record, normalisation_scale=np.ones(3), trajectory=trajectory)
    monkeypatch.setattr(audit, "validate_record_matrix", lambda records: None)
    monkeypatch.setattr(audit, "load_fixed_trajectory", lambda current: trajectory)
    monkeypatch.setattr(audit, "load_schedules", lambda manifest: {})
    monkeypatch.setattr(audit, "strict_load_json", lambda path: {})
    monkeypatch.setattr(audit, "load_model_bundle", lambda path: (None, _scalers(), {}))
    models = {"models": [{"scenario": "regular_to_chaotic", "seed": 42, "path": "unused"}]}
    assert audit.audit_records({"records": [corrected]}, models, derived=True)["failure_count"] == 1
    corrected["failure_step"] = 4
    with pytest.raises(core.CrossRegimeError):
        audit.audit_records({"records": [corrected]}, models, derived=True)


@pytest.mark.parametrize("key", ["targets", "time", "current"])
def test_correct_record_checks_original_source_even_when_artifact_hash_matches(tmp_path, key):
    record, arrays, trajectory = _historical_record(tmp_path)
    arrays[key][0] += 1
    np.savez(record["raw_arrays_path"], **arrays)
    record["raw_arrays_sha256"] = file_sha256(Path(record["raw_arrays_path"]))
    with pytest.raises(core.CrossRegimeError, match="source trajectory"):
        correction.correct_record(record, normalisation_scale=np.ones(3), trajectory=trajectory)


def test_tampered_artifact_fails_before_correction(tmp_path, monkeypatch):
    record, _, trajectory = _historical_record(tmp_path)
    path = Path(record["raw_arrays_path"])
    with path.open("ab") as stream:
        stream.write(b"tampering")
    monkeypatch.setattr(core, "derive_record_fields", lambda *args: pytest.fail("unverified artifact used"))
    with pytest.raises(core.CrossRegimeError, match="hash mismatch"):
        correction.correct_record(record, normalisation_scale=np.ones(3), trajectory=trajectory)


def _matrix_records():
    records = []
    for scenario in SCENARIO_TRAINING_CURRENTS:
        for seed in SEEDS:
            for current in ALL_CURRENTS:
                for family, window in [("fixed_short", 1), ("fixed_short", 2), ("fixed_short", 3), ("fixed_long", None)]:
                    records.append({"record_id": core.record_id(family, scenario, seed, current=current, window=window),
                                    "family": family, "scenario": scenario, "seed": seed,
                                    "current": current, "window": window, "schedule": None})
            for schedule in CONTINUOUS_SCHEDULES:
                records.append({"record_id": core.record_id("continuous", scenario, seed, schedule=schedule),
                                "family": "continuous", "scenario": scenario, "seed": seed,
                                "current": None, "window": None, "schedule": schedule})
    return records


def test_correction_preserves_exact_345_record_matrix_and_order(tmp_path, monkeypatch):
    records = _matrix_records()
    core.validate_record_matrix(records)
    models = {"models": [{"scenario": scenario, "seed": seed, "path": "unused"}
                         for scenario in SCENARIO_TRAINING_CURRENTS for seed in SEEDS]}
    path = tmp_path / "raw.json"
    core.atomic_write_json(path, {"records": records})
    monkeypatch.setattr(runner, "RAW_RESULTS_PATH", path)
    monkeypatch.setattr(core, "load_model_bundle", lambda path: (None, _scalers(), {}))
    monkeypatch.setattr(correction, "load_fixed_trajectory", lambda current: None)
    monkeypatch.setattr(runner, "load_schedules", lambda manifest: dict.fromkeys(CONTINUOUS_SCHEDULES))
    visited = []
    def derive(item, **kwargs):
        visited.append(item["record_id"])
        return dict(item), None
    monkeypatch.setattr(correction, "correct_record", derive)
    _forbid_forecasts(monkeypatch)
    corrected, _ = correction._derive_records({"raw": {"records": records, "lock_hashes": {}},
                                               "models": models, "datasets": {}})
    expected = [item["record_id"] for item in records]
    assert visited == expected == [item["record_id"] for item in corrected["records"]]
    assert len(visited) == len(set(visited)) == 345
    assert set(visited) == core.expected_record_ids()


def test_existing_output_directory_is_refused_before_inputs_are_used(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "git_output", lambda *args: correction.BRANCH)
    monkeypatch.setattr(correction, "_load_originals", lambda: pytest.fail("inputs should not be used"))
    root = tmp_path / "post_hoc_numerical_correction"
    root.mkdir()
    marker = root / "keep.json"
    marker.write_text("historical", encoding="utf-8")
    with pytest.raises(core.CrossRegimeError, match="refusing to overwrite"):
        correction.run_correction()
    assert marker.read_text(encoding="utf-8") == "historical"


def test_output_path_must_be_new_child_of_result_root(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULT_ROOT", tmp_path / "results")
    with pytest.raises(core.CrossRegimeError, match="new subdirectory"):
        correction._output_root(tmp_path)
    with pytest.raises(core.CrossRegimeError, match="new subdirectory"):
        correction._output_root(tmp_path / "results")


@pytest.fixture
def correction_roundtrip(tmp_path, monkeypatch):
    """Use a real synthetic archive/derived audit; stub the HPC-only inventory."""
    record, _, trajectory = _historical_record(tmp_path, unsafe=True)
    _forbid_forecasts(monkeypatch)
    raw_path, frozen_path = tmp_path / "original_raw.json", tmp_path / "frozen.json"
    raw = {"records": [record], "lock_hashes": {}}
    core.atomic_write_json(raw_path, raw)
    core.atomic_write_json(frozen_path, {"frozen": True})
    source_path = tmp_path / "source.py"
    source_path.write_text("# synthetic correction source\n", encoding="utf-8")
    original_hashes = core.file_hash_inventory([
        raw_path, frozen_path, Path(record["raw_arrays_path"])
    ])
    inputs = {
        "raw": raw, "manifest": {}, "status": {}, "datasets": {},
        "models": {"models": [{"scenario": "regular_to_chaotic", "seed": 42, "path": "unused"}]},
        "input_hashes": original_hashes,
        "provenance": {"original_benchmark_commit": "b" * 40},
        "model_audit": {}, "dataset_audit": {}, "original_binaries": {"valid": True},
        "historical_aggregate": {"scenarios": {}, "created_at": "old"},
    }
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "RESULT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "RAW_RESULTS_PATH", raw_path)
    monkeypatch.setattr(runner, "MANIFEST_PATH", frozen_path)
    monkeypatch.setattr(runner, "SOURCE_PATHS", (source_path,))
    monkeypatch.setattr(runner, "git_output", lambda *args: correction.BRANCH)
    monkeypatch.setattr(correction, "EXPECTED_RECORDS", 1)
    monkeypatch.setattr(correction, "_load_originals", lambda: inputs)
    monkeypatch.setattr(correction, "correction_source_provenance", lambda **kwargs: {
        "source_hashes": {"source.py": file_sha256(source_path)}, "dirty": True,
        "commit": "c" * 40,
    })
    monkeypatch.setattr(core, "validate_record_matrix", lambda records: None)
    monkeypatch.setattr(audit, "validate_record_matrix", lambda records: None)
    monkeypatch.setattr(core, "load_model_bundle", lambda path: (None, _scalers(), {}))
    monkeypatch.setattr(audit, "load_model_bundle", lambda path: (None, _scalers(), {}))
    monkeypatch.setattr(correction, "load_fixed_trajectory", lambda current: trajectory)
    monkeypatch.setattr(audit, "load_fixed_trajectory", lambda current: trajectory)
    monkeypatch.setattr(runner, "load_schedules", lambda manifest: {})
    monkeypatch.setattr(audit, "load_schedules", lambda manifest: {})
    monkeypatch.setattr(audit, "strict_load_json", lambda path: {})
    monkeypatch.setattr(runner, "aggregate_results", lambda raw, write=False: {
        "scenarios": {"penalty": raw["records"][0]["aggregate_nrmse_value"]},
        "created_at": "new",
    })
    def tables(records, aggregate, *, output_root):
        for name in correction.OUTPUT_NAMES:
            if name.endswith(".csv"):
                core.atomic_write_csv(output_root / name, [{"record_id": records[0]["record_id"]}])
    monkeypatch.setattr(runner, "write_tables", tables)
    return inputs, tmp_path / "post_hoc_numerical_correction", source_path


def test_correction_output_manifest_roundtrip_preserves_originals(correction_roundtrip):
    inputs, root, _ = correction_roundtrip
    manifest = correction.run_correction()
    assert manifest["no_model_retraining"] and manifest["no_forecast_rerun"]
    assert manifest["original_prediction_artifacts_preserved"]
    assert manifest["classification_or_penalty_changed_count"] == 1
    assert set(manifest["output_hashes"]) == correction.OUTPUT_NAMES
    assert correction.audit_correction()["verdict"] == "DERIVED CORRECTION AUDIT PASSED"
    for path, digest in inputs["input_hashes"].items():
        assert file_sha256(Path(path)) == digest
    derived = core.strict_load_json(root / "corrected_results.json")["records"][0]
    assert derived["raw_arrays_sha256"] == inputs["raw"]["records"][0]["raw_arrays_sha256"]


@pytest.mark.parametrize("tamper", ["claim", "source", "log", "table"])
def test_correction_reaudit_rejects_tampered_derived_provenance(correction_roundtrip, tamper):
    _, root, source = correction_roundtrip
    correction.run_correction()
    manifest_path = root / "correction_manifest.json"
    manifest = core.strict_load_json(manifest_path)
    if tamper == "claim":
        manifest["no_forecast_rerun"] = False
    elif tamper == "source":
        source.write_text("changed source", encoding="utf-8")
    elif tamper == "log":
        path = root / "correction_changes.json"
        log = core.strict_load_json(path)
        log["classification_or_penalty_changed_count"] = 0
        core.atomic_write_json(path, log)
        manifest["output_hashes"][path.name] = file_sha256(path)
    else:
        manifest["output_hashes"].pop("fixed_short_results.csv")
    core.atomic_write_json(manifest_path, manifest)
    with pytest.raises(core.CrossRegimeError):
        correction.audit_correction()


def test_correction_rejects_sources_changing_during_derivation(correction_roundtrip, monkeypatch):
    _, root, source = correction_roundtrip
    derive = correction._derive_records
    def changed_source(inputs):
        result = derive(inputs)
        source.write_text("changed during derivation", encoding="utf-8")
        return result
    monkeypatch.setattr(correction, "_derive_records", changed_source)
    with pytest.raises(core.CrossRegimeError, match="correction source hash"):
        correction.run_correction()
    assert not root.exists()
