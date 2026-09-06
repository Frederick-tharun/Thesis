"""Source-lock tests use immutable synthetic Git responses, never new commits."""

import hashlib

import pytest

from chapter2 import cross_regime_provenance as provenance


FREEZE_COMMIT = "a" * 40
EXECUTION_COMMIT = "b" * 40
REPAIR_COMMIT = "c" * 40


@pytest.fixture
def locked_source(monkeypatch):
    sources = {
        "chapter2/CROSS_REGIME_PROTOCOL.md": b"Frozen protocol\n",
        "chapter2/optimisation_results/step7_selection.json": b'{"fixed": true}\n',
        "run_chapter2_cross_regime.slurm": b"#!/bin/bash\n",
        "chapter2/cross_regime.py": b"original_forecast = True\n",
        "chapter2/cross_regime_config.py": b"original_configuration = True\n",
        "chapter2/run_cross_regime.py": b"original_runner = True\n",
        "chapter2/audit_cross_regime.py": b"original_auditor = True\n",
        "chapter2/tests/test_cross_regime.py": b"original_test = True\n",
    }
    calls = []

    def git(root, *args):
        calls.append(args)
        if args[:2] == ("rev-parse", "--verify"):
            return (args[2].split("^")[0] + "\n").encode()
        if args[:2] == ("merge-base", "--is-ancestor"):
            assert args[2:] == (FREEZE_COMMIT, EXECUTION_COMMIT)
            return b""
        if args[:4] == ("ls-tree", "-r", "--name-only", "-z"):
            assert args[4] == EXECUTION_COMMIT
            return "\0".join([*sources, "chapter2/tests/test_unrelated.py", "README.md", ""]).encode()
        if args[0] == "show":
            commit, name = args[1].split(":", 1)
            assert commit == EXECUTION_COMMIT, "must never inspect repaired HEAD or pre-implementation freeze HEAD"
            return sources[name]
        if args == ("rev-parse", "HEAD"):
            return (REPAIR_COMMIT + "\n").encode()
        if args == ("branch", "--show-current"):
            return b"chapter2-cross-regime-parameter-aware\n"
        if args[:2] == ("status", "--short"):
            return b" M chapter2/cross_regime.py\n?? chapter2/correct_cross_regime_numerics.py\n"
        raise AssertionError(f"unexpected Git access: {args}")

    monkeypatch.setattr(provenance, "_git", git)
    manifest = {
        "preflight": {"head": FREEZE_COMMIT, "tracked_worktree_clean": False},
        "source_hashes": {name: hashlib.sha256(value).hexdigest() for name, value in sources.items()},
    }
    return manifest, {"implementation_commit": EXECUTION_COMMIT}, sources, calls


def test_original_execution_blob_hashes_ignore_repaired_working_tree(tmp_path, locked_source):
    manifest, status, _, calls = locked_source
    source = tmp_path / "chapter2/cross_regime.py"
    source.parent.mkdir()
    source.write_bytes(b"completely different uncommitted correction\r\n")
    result = provenance.verify_original_source_lock(manifest, project_root=tmp_path, execution_info=status)
    assert result["original_benchmark_commit"] == EXECUTION_COMMIT
    assert result["frozen_preflight_commit"] == FREEZE_COMMIT
    assert result["source_lock_verified"] is True
    assert result["source_hashes"] == manifest["source_hashes"]
    assert result["execution_commit_origin"] == ["execution_info.implementation_commit"]
    assert ("rev-parse", "HEAD") not in calls


def test_dirty_frozen_preflight_without_execution_status_fails_closed(tmp_path, locked_source):
    manifest, _, _, _ = locked_source
    with pytest.raises(provenance.CrossRegimeProvenanceError, match="original status implementation_commit"):
        provenance.verify_original_source_lock(manifest, project_root=tmp_path)


def test_clean_frozen_preflight_is_usable_only_as_recorded_commit(tmp_path, locked_source, monkeypatch):
    manifest, _, _, _ = locked_source
    manifest["preflight"] = {"head": EXECUTION_COMMIT, "tracked_worktree_clean": True}
    original_git = provenance._git

    def git(root, *args):
        if args[:2] == ("merge-base", "--is-ancestor"):
            assert args[2:] == (EXECUTION_COMMIT, EXECUTION_COMMIT)
            return b""
        return original_git(root, *args)

    monkeypatch.setattr(provenance, "_git", git)
    result = provenance.verify_original_source_lock(manifest, project_root=tmp_path)
    assert result["original_benchmark_commit"] == EXECUTION_COMMIT
    assert result["execution_commit_origin"] == ["manifest.preflight.head"]


def test_original_source_tampering_fails_even_if_working_tree_matches(tmp_path, locked_source):
    manifest, status, _, _ = locked_source
    source = tmp_path / "chapter2/cross_regime.py"
    source.parent.mkdir()
    source.write_bytes(b"tampered source\n")
    manifest["source_hashes"]["chapter2/cross_regime.py"] = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(provenance.CrossRegimeProvenanceError, match="original execution source hash mismatch"):
        provenance.verify_original_source_lock(manifest, project_root=tmp_path, execution_info=status)


def test_original_source_hashing_never_normalizes_line_endings(tmp_path, locked_source):
    manifest, status, sources, _ = locked_source
    sources["chapter2/cross_regime.py"] = sources["chapter2/cross_regime.py"].replace(b"\n", b"\r\n")
    with pytest.raises(provenance.CrossRegimeProvenanceError, match="source hash mismatch"):
        provenance.verify_original_source_lock(manifest, project_root=tmp_path, execution_info=status)


def test_missing_locked_file_does_not_weaken_inventory_check(tmp_path, locked_source):
    manifest, status, _, _ = locked_source
    manifest["source_hashes"].pop("chapter2/cross_regime.py")
    with pytest.raises(provenance.CrossRegimeProvenanceError, match="source inventory differs"):
        provenance.verify_original_source_lock(manifest, project_root=tmp_path, execution_info=status)


def test_conflicting_recorded_execution_commits_fail_closed(tmp_path, locked_source):
    manifest, status, _, _ = locked_source
    manifest["implementation_commit"] = REPAIR_COMMIT
    with pytest.raises(provenance.CrossRegimeProvenanceError, match="conflicting original execution commits"):
        provenance.verify_original_source_lock(manifest, project_root=tmp_path, execution_info=status)


@pytest.mark.parametrize("commit", ["HEAD", "main", "b" * 7, None])
def test_original_commit_must_be_full_immutable_object_id(tmp_path, locked_source, commit):
    manifest, status, _, _ = locked_source
    status["implementation_commit"] = commit
    with pytest.raises(provenance.CrossRegimeProvenanceError, match="full immutable Git object ID"):
        provenance.verify_original_source_lock(manifest, project_root=tmp_path, execution_info=status)


@pytest.mark.parametrize("name", ["../cross_regime.py", "/chapter2/cross_regime.py", "C:/cross_regime.py", "chapter2\\cross_regime.py"])
def test_noncanonical_source_paths_fail_closed(tmp_path, locked_source, name):
    manifest, status, _, _ = locked_source
    manifest["source_hashes"][name] = "d" * 64
    with pytest.raises(provenance.CrossRegimeProvenanceError, match="canonical and project-relative"):
        provenance.verify_original_source_lock(manifest, project_root=tmp_path, execution_info=status)


def test_original_commit_must_descend_from_frozen_preflight(tmp_path, locked_source, monkeypatch):
    manifest, status, _, _ = locked_source
    original_git = provenance._git

    def git(root, *args):
        if args[:2] == ("merge-base", "--is-ancestor"):
            raise provenance.CrossRegimeProvenanceError("not an ancestor")
        return original_git(root, *args)

    monkeypatch.setattr(provenance, "_git", git)
    with pytest.raises(provenance.CrossRegimeProvenanceError, match="not an ancestor"):
        provenance.verify_original_source_lock(manifest, project_root=tmp_path, execution_info=status)


def test_correction_provenance_records_actual_dirty_working_bytes(tmp_path, locked_source):
    source = tmp_path / "chapter2/cross_regime.py"
    source.parent.mkdir()
    source.write_bytes(b"corrected bytes\r\n")
    result = provenance.correction_source_provenance(project_root=tmp_path, source_paths=[source])
    assert result["commit"] == REPAIR_COMMIT
    assert result["dirty"] is True
    assert result["source_hashes"] == {"chapter2/cross_regime.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    assert "?? chapter2/correct_cross_regime_numerics.py" in result["git_status_short"]


def test_correction_source_paths_cannot_escape_root(tmp_path, locked_source):
    with pytest.raises(provenance.CrossRegimeProvenanceError, match="escapes project root"):
        provenance.correction_source_provenance(project_root=tmp_path, source_paths=["../outside.py"])
