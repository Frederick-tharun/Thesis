"""Audit tests: tampering is detected."""
from __future__ import annotations

import hashlib
import json

from chapter2.part2_cross_regime import audit, config


def test_audit_design_and_data_checks_pass():
    # Chapter 2 Part 2 is finalised: both models are locked and the final
    # RR/RC/CC/CR benchmark has legitimately run once, deliberately, through
    # the sanctioned token entry point. check_no_final_test_result_exists is
    # therefore EXPECTED to report the result directory now -- that was only
    # ever a pre-benchmark guard, not a permanent invariant -- so it is
    # intentionally excluded here rather than asserted as a failure.
    result = audit.AuditResult()
    audit.check_part1_and_chapter1_unchanged(result)
    audit.check_design_hash(result)
    audit.check_no_train_test_overlap(result)
    audit.check_final_test_firewall(result)
    audit.check_trajectory_hashes(result)
    audit.check_search_space_freeze(result)
    failed = [c for c in result.checks if not c["passed"]]
    assert not failed, f"unexpected audit failures: {failed}"


def test_final_benchmark_result_exists_exactly_once():
    # Confirms the ONE current, correct final-benchmark result directory
    # exists, and that the old superseded (80-time-unit, unfair single-seed
    # vs ensemble) result directory was removed rather than left to confuse
    # future readers.
    results_dir = config.PACKAGE_ROOT / "results"
    assert (results_dir / "final_evaluation_long" / "final_long_horizon_results.json").exists()
    assert not (results_dir / "final_evaluation").exists()


def test_audit_detects_design_hash_tampering(tmp_path, monkeypatch):
    original = config.FROZEN_DESIGN_PATH.read_text()
    tampered = json.loads(original)
    tampered["models"]["chaotic_trained"]["training_currents"][0]["current"] += 0.001
    tamper_path = tmp_path / "frozen_design.json"
    tamper_path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(config, "FROZEN_DESIGN_PATH", tamper_path)

    result = audit.AuditResult()
    audit.check_design_hash(result)
    assert result.checks[0]["passed"] is False


def test_audit_detects_missing_manifest_file(tmp_path, monkeypatch):
    manifest = json.loads(config.DATA_MANIFEST_PATH.read_text())
    manifest["trajectories"][0]["sha256"] = "0" * 64  # wrong hash
    tamper_path = tmp_path / "data_manifest.json"
    tamper_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(config, "DATA_MANIFEST_PATH", tamper_path)

    result = audit.AuditResult()
    audit.check_trajectory_hashes(result)
    assert result.checks[0]["passed"] is False


def test_final_lock_hash_is_self_consistent():
    lock_path = config.PACKAGE_ROOT / "results" / "CHAPTER2_PART2_FINAL_LOCK.json"
    lock = json.loads(lock_path.read_text())
    stored = lock.pop("lock_sha256")
    recomputed = hashlib.sha256(
        json.dumps(lock, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    assert stored == recomputed


def test_final_lock_references_match_the_live_files():
    lock = json.loads((config.PACKAGE_ROOT / "results" / "CHAPTER2_PART2_FINAL_LOCK.json").read_text())
    from chapter2.part2_cross_regime import data as data_module

    for entry in (lock["locked_models"]["regular"], lock["locked_models"]["chaotic"], lock["final_benchmark"]):
        path = config.PROJECT_ROOT / entry["source_file"]
        assert data_module.file_sha256(path) == entry["file_sha256"], entry["source_file"]
