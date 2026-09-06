"""Separate the immutable forecast source lock from post-hoc repair sources.

The protocol was frozen in a dirty preflight tree and committed afterward.
Consequently its preflight HEAD can precede the clean implementation commit
recorded by the execution status. Only a recorded execution commit (or a
demonstrably clean frozen preflight) may identify the original source tree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any


class CrossRegimeProvenanceError(ValueError):
    """A historical source lock cannot be verified without guessing."""


def _git(project_root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={project_root.as_posix()}", *arguments],
            cwd=project_root, check=True, capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise CrossRegimeProvenanceError(
            f"cannot verify original Git provenance: {' '.join(arguments)}"
        ) from error


def _commit(project_root: Path, value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) is None:
        raise CrossRegimeProvenanceError("original execution commit must be a full immutable Git object ID")
    resolved = _git(project_root, "rev-parse", "--verify", f"{value}^{{commit}}").decode("ascii").strip()
    if resolved != value:
        raise CrossRegimeProvenanceError("recorded Git object does not identify the exact execution commit")
    return resolved


def _locked_source_paths(project_root: Path, commit: str) -> set[str]:
    """Reproduce the frozen runner's SOURCE_PATHS from its original Git tree."""
    paths = _git(project_root, "ls-tree", "-r", "--name-only", "-z", commit).decode("utf-8").split("\0")
    selected = {
        "chapter2/cross_regime_config.py",
        "chapter2/cross_regime.py",
        "chapter2/run_cross_regime.py",
        "chapter2/audit_cross_regime.py",
        "chapter2/CROSS_REGIME_PROTOCOL.md",
        "chapter2/optimisation_results/step7_selection.json",
        "run_chapter2_cross_regime.slurm",
    }
    for name in paths:
        path = PurePosixPath(name)
        if path.parent == PurePosixPath("chapter2") and path.suffix == ".py":
            selected.add(name)
        elif path.parent == PurePosixPath("chapter2/tests") and path.match("test_cross_regime*.py"):
            selected.add(name)
    return selected


def verify_original_source_lock(
    manifest: Mapping[str, Any], *, project_root: Path,
    execution_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify every frozen SHA-256 against original Git blob bytes, fail closed.

    ``execution_info`` normally is the historical cross_regime_status.json,
    whose ``implementation_commit`` was captured by the clean execution gate.
    Today's HEAD and working source files never identify or validate the old
    execution. No line-ending conversion or other hash normalization is used.
    """
    root = Path(project_root).resolve()
    preflight = manifest.get("preflight")
    if not isinstance(preflight, Mapping):
        raise CrossRegimeProvenanceError("frozen manifest lacks preflight provenance")
    frozen_commit = _commit(root, preflight.get("head"))
    claims: list[tuple[str, Any]] = []
    for origin, info in (("manifest", manifest), ("execution_info", execution_info)):
        if info is None:
            continue
        if not isinstance(info, Mapping):
            raise CrossRegimeProvenanceError("execution provenance must be a mapping")
        for key in ("implementation_commit", "execution_commit"):
            if key in info:
                claims.append((f"{origin}.{key}", info[key]))
        if origin == "execution_info" and info.get("tracked_worktree_clean") is True and "head" in info:
            claims.append((f"{origin}.head", info["head"]))
    if not claims:
        if preflight.get("tracked_worktree_clean") is not True:
            raise CrossRegimeProvenanceError(
                "dirty frozen preflight cannot identify execution; original status implementation_commit is required"
            )
        claims.append(("manifest.preflight.head", frozen_commit))
    commits = {_commit(root, value) for _, value in claims}
    if len(commits) != 1:
        raise CrossRegimeProvenanceError("conflicting original execution commits in recorded provenance")
    execution_commit = commits.pop()
    _git(root, "merge-base", "--is-ancestor", frozen_commit, execution_commit)

    hashes = manifest.get("source_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        raise CrossRegimeProvenanceError("frozen manifest source_hashes must be nonempty")
    for name, digest in hashes.items():
        if (not isinstance(name, str) or "\\" in name or ":" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or PurePosixPath(name).as_posix() != name):
            raise CrossRegimeProvenanceError("frozen source path must be canonical and project-relative")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise CrossRegimeProvenanceError(f"invalid frozen source SHA-256: {name}")
    expected_paths = _locked_source_paths(root, execution_commit)
    if set(hashes) != expected_paths:
        raise CrossRegimeProvenanceError(
            "frozen source inventory differs from original execution source inventory "
            f"(missing={sorted(expected_paths - set(hashes))}, extra={sorted(set(hashes) - expected_paths)})"
        )
    verified = {}
    for name, expected in sorted(hashes.items()):
        actual = hashlib.sha256(_git(root, "show", f"{execution_commit}:{name}")).hexdigest()
        if actual != expected:
            raise CrossRegimeProvenanceError(f"original execution source hash mismatch: {name}")
        verified[name] = actual
    return {
        "original_benchmark_commit": execution_commit,
        "frozen_preflight_commit": frozen_commit,
        "frozen_preflight_worktree_clean": preflight.get("tracked_worktree_clean"),
        "execution_commit_origin": [origin for origin, _ in claims],
        "source_lock_verified": True,
        "verification_method": "SHA-256 of exact Git blob bytes at recorded original execution commit",
        "source_hashes": verified,
    }


def correction_source_provenance(
    *, project_root: Path, source_paths: Sequence[Path | str],
) -> dict[str, Any]:
    """Record repaired working bytes, including uncommitted correction code."""
    root = Path(project_root).resolve()
    hashes = {}
    for source_path in source_paths:
        path = Path(source_path)
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            name = path.relative_to(root).as_posix()
        except ValueError as error:
            raise CrossRegimeProvenanceError("correction source path escapes project root") from error
        try:
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise CrossRegimeProvenanceError(f"cannot hash correction source: {name}") from error
    if not hashes:
        raise CrossRegimeProvenanceError("correction source inventory must be nonempty")
    head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    branch = _git(root, "branch", "--show-current").decode("utf-8").strip()
    status = _git(root, "status", "--short", "--untracked-files=all").decode("utf-8").rstrip("\r\n")
    return {
        "commit": head,
        "branch": branch,
        "dirty": bool(status),
        "git_status_short": status,
        "source_hashes": dict(sorted(hashes.items())),
        "source_hash_basis": "actual correction working-tree bytes; HEAD alone does not identify uncommitted repairs",
    }
