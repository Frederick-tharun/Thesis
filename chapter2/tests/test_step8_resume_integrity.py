"""Synthetic corruption tests for fail-closed Step 8 resume handling."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import chapter2.esn_step8 as step8
from chapter2.esn_data import NumpyStandardScaler, StateCurrentScalers
from chapter2.esn_model import EchoStateNetwork
from chapter2.esn_optimisation import PARAMETER_AWARE


@pytest.fixture
def model_resume_fixture(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    chapter_root = tmp_path / "chapter2"
    model_root = chapter_root / "final_models"
    result_root = chapter_root / "final_results"
    model_root.mkdir(parents=True)
    result_root.mkdir()
    monkeypatch.setattr(step8, "CHAPTER2_ROOT", chapter_root)
    monkeypatch.setattr(step8, "FINAL_MODELS", model_root)
    monkeypatch.setattr(step8, "MODEL_MANIFEST_PATH", model_root / "model_manifest.json")

    lock_hashes = {
        "chapter2/final_results/selected_model.json": "1" * 64,
        "chapter2/final_results/step8_evaluation_manifest.json": "2" * 64,
    }
    preflight = {
        "dataset_hashes": {
            "chapter2/outputs/data/fixed_I_1p67.npz": "3" * 64,
            "chapter2/outputs/data/fixed_I_3p20.npz": "4" * 64,
            "chapter2/outputs/data/fixed_I_3p50.npz": "5" * 64,
        }
    }
    model_type = PARAMETER_AWARE
    seed = 42
    config = step8.final_model_config(model_type, seed)
    model = EchoStateNetwork(config)
    model.output_weights = np.zeros((config.output_dimension, model.feature_dimension))
    scalers = StateCurrentScalers(
        NumpyStandardScaler(np.zeros(3), np.ones(3)),
        NumpyStandardScaler(np.zeros(1), np.ones(1)),
    )
    path = step8.model_path(model_type, seed)
    metadata = step8._model_metadata(model_type, seed, preflight, lock_hashes)
    step8.save_final_model(path, model, scalers, metadata)
    entry = {
        "model_type": model_type,
        "seed": seed,
        "path": step8.project_relative(path),
        "sha256": step8.file_sha256(path),
        "configuration": asdict(config),
        "round_trip_inference_exact": True,
        "training_transition_range": [0, step8.STEP8_FINAL_TRAINING_STOP],
        "washout_per_trajectory": step8.STEP8_TRAINING_WASHOUT,
    }
    manifest = {
        "schema": step8.MODEL_MANIFEST_SCHEMA,
        "status": "in_progress",
        "lock_hashes": lock_hashes,
        "models": [entry],
    }
    return SimpleNamespace(
        preflight=preflight,
        lock_hashes=lock_hashes,
        manifest=manifest,
        entry=entry,
        path=path,
    )


def _validate_model_fixture(fixture: SimpleNamespace, manifest: dict | None = None) -> None:
    step8.validate_model_manifest(
        manifest or fixture.manifest,
        preflight=fixture.preflight,
        lock_hashes=fixture.lock_hashes,
        require_complete=False,
    )


def _rewrite_bundle(path: Path, **updates: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: np.asarray(saved[key]).copy() for key in saved.files}
    arrays.update(updates)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def test_valid_partial_model_manifest_is_accepted(model_resume_fixture) -> None:
    _validate_model_fixture(model_resume_fixture)


def test_wrong_model_seed_is_rejected(model_resume_fixture) -> None:
    manifest = deepcopy(model_resume_fixture.manifest)
    manifest["models"][0]["seed"] = 123
    with pytest.raises(step8.PreflightError, match="mismatch"):
        _validate_model_fixture(model_resume_fixture, manifest)


def test_wrong_model_hyperparameters_are_rejected(model_resume_fixture) -> None:
    manifest = deepcopy(model_resume_fixture.manifest)
    manifest["models"][0]["configuration"]["leak_rate"] = 0.5
    with pytest.raises(step8.PreflightError, match="configuration"):
        _validate_model_fixture(model_resume_fixture, manifest)


def test_wrong_scaler_shape_is_rejected(model_resume_fixture) -> None:
    _rewrite_bundle(model_resume_fixture.path, state_mean=np.zeros(2))
    model_resume_fixture.entry["sha256"] = step8.file_sha256(model_resume_fixture.path)
    with pytest.raises(step8.PreflightError, match="invalid final model bundle"):
        _validate_model_fixture(model_resume_fixture)


def test_wrong_model_manifest_lock_hash_is_rejected(model_resume_fixture) -> None:
    manifest = deepcopy(model_resume_fixture.manifest)
    manifest["lock_hashes"]["chapter2/final_results/selected_model.json"] = "9" * 64
    with pytest.raises(step8.PreflightError, match="lock hashes"):
        _validate_model_fixture(model_resume_fixture, manifest)


def test_wrong_embedded_model_lock_hash_is_rejected(
    model_resume_fixture,
) -> None:
    with np.load(model_resume_fixture.path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"].item()))
    metadata["lock_hashes"][
        "chapter2/final_results/selected_model.json"
    ] = "9" * 64
    _rewrite_bundle(
        model_resume_fixture.path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    model_resume_fixture.entry["sha256"] = step8.file_sha256(
        model_resume_fixture.path
    )
    with pytest.raises(step8.PreflightError, match="scientific content"):
        _validate_model_fixture(model_resume_fixture)


def test_wrong_model_hash_is_rejected(model_resume_fixture) -> None:
    manifest = deepcopy(model_resume_fixture.manifest)
    manifest["models"][0]["sha256"] = "0" * 64
    with pytest.raises(step8.PreflightError, match="model hash"):
        _validate_model_fixture(model_resume_fixture, manifest)


def test_wrong_realised_spectral_radius_is_rejected(model_resume_fixture) -> None:
    with np.load(model_resume_fixture.path, allow_pickle=False) as saved:
        size = np.asarray(saved["reservoir_weights"]).shape[0]
    _rewrite_bundle(
        model_resume_fixture.path,
        reservoir_weights=np.zeros((size, size)),
    )
    model_resume_fixture.entry["sha256"] = step8.file_sha256(model_resume_fixture.path)
    with pytest.raises(step8.PreflightError, match="spectral radius"):
        _validate_model_fixture(model_resume_fixture)


def test_scientific_lock_content_is_compared_but_named_fields_are_excluded() -> None:
    expected = {
        "created_at": "old",
        "runtime": {"host": "old"},
        "models": {"aware": {"seed": 42}},
    }
    harmless = {
        "created_at": "new",
        "runtime": {"host": "new"},
        "models": {"aware": {"seed": 42}},
    }
    step8._require_scientific_equality(
        label="selection lock",
        actual=harmless,
        expected=expected,
        excluded_fields=("created_at", "runtime"),
    )
    altered = deepcopy(harmless)
    altered["models"]["aware"]["seed"] = 456
    with pytest.raises(step8.PreflightError, match="scientific content"):
        step8._require_scientific_equality(
            label="selection lock",
            actual=altered,
            expected=expected,
            excluded_fields=("created_at", "runtime"),
        )


def test_altered_evaluation_manifest_content_is_rejected() -> None:
    expected = {
        "created_at": "old",
        "runtime": {"host": "old"},
        "fixed_short_windows": [{"window": 1, "forecast_range": [2, 10]}],
    }
    actual = deepcopy(expected)
    actual["created_at"] = "new"
    actual["fixed_short_windows"][0]["forecast_range"] = [2, 11]
    with pytest.raises(step8.PreflightError, match="scientific content"):
        step8._require_scientific_equality(
            label="evaluation manifest",
            actual=actual,
            expected=expected,
            excluded_fields=("created_at", "runtime"),
        )


@pytest.fixture
def raw_resume_fixture(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    chapter_root = tmp_path / "chapter2"
    raw_root = chapter_root / "final_results" / "raw_arrays"
    raw_root.mkdir(parents=True)
    monkeypatch.setattr(step8, "CHAPTER2_ROOT", chapter_root)
    monkeypatch.setattr(step8, "FINAL_RESULTS", raw_root.parent)
    monkeypatch.setattr(step8, "RAW_ARRAYS", raw_root)
    monkeypatch.setattr(step8, "RAW_RESULTS_PATH", raw_root.parent / "step8_raw_results.json")
    monkeypatch.setattr(step8, "STATUS_PATH", raw_root.parent / "step8_status.json")

    identifier = step8._record_id(
        "known_short", PARAMETER_AWARE, 42, 1.67, 1
    )
    horizon = step8.STEP8_FORECAST_TRANSITIONS
    path = raw_root / f"{identifier}.npz"
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            predictions=np.zeros((horizon, 3)),
            targets=np.zeros((horizon, 3)),
            pointwise_normalised_error=np.zeros(horizon),
            time=np.arange(horizon, dtype=float) * step8.DT,
            current=np.full(horizon, 1.67),
        )
    record = {
        "record_id": identifier,
        "family": "known_short",
        "model_type": PARAMETER_AWARE,
        "seed": 42,
        "current": 1.67,
        "window": 1,
        "warmup_range": [70_000, 72_000],
        "forecast_range": [72_000, 80_000],
        "metrics": {"sample_count": horizon},
        "numerical_failure": False,
        "failure_step": None,
        "failure_reason": None,
        "raw_arrays_path": step8.project_relative(path),
        "raw_arrays_sha256": step8.file_sha256(path),
    }
    lock_hashes = {"selection": "1" * 64, "evaluation": "2" * 64}
    raw = {
        "schema": step8.STEP8_SCHEMA,
        "lock_hashes": lock_hashes,
        "records": [record],
    }
    status = {
        "lock_hashes": lock_hashes,
        "completed_record_ids": [identifier],
        "completed_record_count": 1,
    }
    manifest = {"models": [{"model_type": PARAMETER_AWARE, "seed": 42}]}
    return SimpleNamespace(
        identifier=identifier,
        path=path,
        record=record,
        raw=raw,
        status=status,
        manifest=manifest,
        lock_hashes=lock_hashes,
    )


def _validate_raw_fixture(fixture: SimpleNamespace) -> None:
    step8.validate_resumed_records(
        fixture.raw,
        status=fixture.status,
        manifest=fixture.manifest,
        lock_hashes=fixture.lock_hashes,
    )


def test_valid_partial_raw_checkpoint_is_accepted(raw_resume_fixture) -> None:
    _validate_raw_fixture(raw_resume_fixture)


def test_missing_raw_npz_is_rejected(raw_resume_fixture) -> None:
    raw_resume_fixture.path.unlink()
    with pytest.raises(step8.BenchmarkAccessError, match="missing raw NPZ"):
        _validate_raw_fixture(raw_resume_fixture)


def test_corrupted_raw_npz_is_rejected(raw_resume_fixture) -> None:
    raw_resume_fixture.path.write_bytes(b"not-an-npz")
    raw_resume_fixture.record["raw_arrays_sha256"] = step8.file_sha256(
        raw_resume_fixture.path
    )
    with pytest.raises(step8.BenchmarkAccessError, match="corrupted raw NPZ"):
        _validate_raw_fixture(raw_resume_fixture)


def test_incorrect_raw_array_hash_is_rejected(raw_resume_fixture) -> None:
    raw_resume_fixture.record["raw_arrays_sha256"] = "0" * 64
    with pytest.raises(step8.BenchmarkAccessError, match="hash mismatch"):
        _validate_raw_fixture(raw_resume_fixture)


def test_mismatched_record_metadata_is_rejected(raw_resume_fixture) -> None:
    raw_resume_fixture.record["current"] = 3.2
    with pytest.raises(step8.BenchmarkAccessError, match="metadata mismatch"):
        _validate_raw_fixture(raw_resume_fixture)


def test_duplicate_resumed_record_ids_are_rejected(raw_resume_fixture) -> None:
    duplicate = deepcopy(raw_resume_fixture.record)
    raw_resume_fixture.raw["records"].append(duplicate)
    raw_resume_fixture.status["completed_record_ids"].append(
        raw_resume_fixture.identifier
    )
    raw_resume_fixture.status["completed_record_count"] = 2
    with pytest.raises(step8.BenchmarkAccessError, match="duplicate"):
        _validate_raw_fixture(raw_resume_fixture)


def test_incomplete_status_record_ids_are_rejected(raw_resume_fixture) -> None:
    raw_resume_fixture.status["completed_record_ids"] = []
    raw_resume_fixture.status["completed_record_count"] = 0
    with pytest.raises(step8.BenchmarkAccessError, match="incomplete"):
        _validate_raw_fixture(raw_resume_fixture)


@pytest.mark.parametrize("damage", ["missing_key", "wrong_shape", "nonfinite"])
def test_raw_npz_schema_shape_and_finiteness_are_checked(
    raw_resume_fixture, damage: str
) -> None:
    horizon = step8.STEP8_FORECAST_TRANSITIONS
    arrays = {
        "predictions": np.zeros((horizon, 3)),
        "targets": np.zeros((horizon, 3)),
        "pointwise_normalised_error": np.zeros(horizon),
        "time": np.arange(horizon, dtype=float) * step8.DT,
        "current": np.full(horizon, 1.67),
    }
    if damage == "missing_key":
        arrays.pop("current")
    elif damage == "wrong_shape":
        arrays["targets"] = np.zeros((horizon - 1, 3))
    else:
        arrays["predictions"][0, 0] = np.nan
    with raw_resume_fixture.path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    raw_resume_fixture.record["raw_arrays_sha256"] = step8.file_sha256(
        raw_resume_fixture.path
    )
    with pytest.raises(step8.BenchmarkAccessError):
        _validate_raw_fixture(raw_resume_fixture)


def test_corrupt_resume_aborts_before_models_or_benchmark_loaders(
    raw_resume_fixture, monkeypatch
) -> None:
    raw_resume_fixture.record["raw_arrays_sha256"] = "0" * 64
    raw_resume_fixture.path.parent.parent.mkdir(parents=True, exist_ok=True)
    raw_resume_fixture.raw["records"] = [raw_resume_fixture.record]
    raw_resume_fixture.status["completed_record_ids"] = [
        raw_resume_fixture.identifier
    ]
    raw_resume_fixture.status["completed_record_count"] = 1
    raw_resume_fixture.path.parent.parent.joinpath(
        "step8_raw_results.json"
    ).write_text(json.dumps(raw_resume_fixture.raw), encoding="utf-8")
    raw_resume_fixture.path.parent.parent.joinpath(
        "step8_status.json"
    ).write_text(json.dumps(raw_resume_fixture.status), encoding="utf-8")

    monkeypatch.setattr(
        step8, "validate_benchmark_gate", lambda *args, **kwargs: raw_resume_fixture.manifest
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("prediction model or benchmark loader was reached")

    monkeypatch.setattr(step8, "_load_models", forbidden)
    monkeypatch.setattr(step8, "load_fixed_trajectory", forbidden)
    monkeypatch.setattr(step8, "load_continuous_benchmark", forbidden)
    with pytest.raises(step8.BenchmarkAccessError, match="hash mismatch"):
        step8.execute_benchmarks(
            {},
            raw_resume_fixture.lock_hashes,
            raw_resume_fixture.manifest,
        )
