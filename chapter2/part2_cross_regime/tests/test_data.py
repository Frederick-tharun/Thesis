"""Data tests: deterministic simulation, lengths, hashes, firewall, no NaN/Inf."""
from __future__ import annotations

import json

import numpy as np
import pytest

from chapter2.part2_cross_regime import config, data


def test_simulation_is_deterministic():
    a = data.generate_trajectory(1.6700, "regular_training")
    b = data.generate_trajectory(1.6700, "regular_training")
    assert data.file_sha256(a) == data.file_sha256(b)


def test_manifest_lengths_and_finiteness():
    manifest = json.loads(config.DATA_MANIFEST_PATH.read_text())
    for row in manifest["trajectories"]:
        assert row["n_samples"] == config.RETAINED_SAMPLES_PER_CURRENT
        assert row["all_finite"] is True
        assert len(row["sha256"]) == 64


def test_manifest_hashes_match_files_on_disk():
    manifest = json.loads(config.DATA_MANIFEST_PATH.read_text())
    for row in manifest["trajectories"]:
        path = config.PROJECT_ROOT / row["path"]
        assert data.file_sha256(path) == row["sha256"]


def test_final_test_access_guard_blocks_plain_load():
    manifest = json.loads(config.DATA_MANIFEST_PATH.read_text())
    final_rows = [r for r in manifest["trajectories"] if r["role"] == "final_test"]
    assert final_rows, "expected at least one final_test trajectory"
    current = final_rows[0]["current"]
    with pytest.raises(data.FinalTestAccessError):
        data.load_source_trajectory("final_test", current)


def test_final_test_access_guard_blocks_wrong_token():
    manifest = json.loads(config.DATA_MANIFEST_PATH.read_text())
    final_rows = [r for r in manifest["trajectories"] if r["role"] == "final_test"]
    current = final_rows[0]["current"]
    with pytest.raises(data.FinalTestAccessError):
        data.load_final_test_trajectory(current, caller_token="not-the-real-token")


def test_final_test_access_guard_allows_correct_token():
    manifest = json.loads(config.DATA_MANIFEST_PATH.read_text())
    final_rows = [r for r in manifest["trajectories"] if r["role"] == "final_test"]
    current = final_rows[0]["current"]
    traj = data.load_final_test_trajectory(current, caller_token=data.FINAL_EVALUATION_ENTRY_POINT_TOKEN)
    assert traj.current == current


def test_source_trajectories_load_and_are_not_locked():
    manifest = json.loads(config.DATA_MANIFEST_PATH.read_text())
    reg_rows = [r for r in manifest["trajectories"] if r["role"] == "regular_training"]
    assert reg_rows
    traj = data.load_source_trajectory("regular_training", reg_rows[0]["current"])
    assert np.all(np.isfinite(traj.states))
    assert np.all(np.isfinite(traj.current_values))
