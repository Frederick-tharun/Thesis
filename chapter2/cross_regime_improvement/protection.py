"""Fail-closed guards for the frozen thesis and baseline artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

from chapter2.esn_data import file_sha256

from .config import (
    CORRECTION_ROOT,
    ESN_SOURCE_SHA256,
    PACKAGE_ROOT,
    PROJECT_ROOT,
    STARTING_COMMIT,
)


class ProtectionError(RuntimeError):
    """A protected input or output contract was violated."""


TRACKED_PROTECTED_PATHS = (
    "FINAL_THESIS_RUN",
    "chapter2/optimisation_results",
    "chapter2/final_models",
    "chapter2/final_results",
    "chapter2/outputs/data",
    "chapter2/outputs/figures",
    "chapter2/pilot_results",
    "chapter2/cross_regime_config.py",
    "chapter2/cross_regime.py",
    "chapter2/run_cross_regime.py",
    "chapter2/audit_cross_regime.py",
    "chapter2/cross_regime_numerics.py",
    "chapter2/correct_cross_regime_numerics.py",
    "chapter2/esn_model.py",
    "chapter2/esn_data.py",
    "chapter2/esn_metrics.py",
    "chapter2/esn_optimisation.py",
    "chapter2/EXPERIMENT_PROTOCOL.md",
)


def assert_esn_source_unchanged() -> None:
    path = PROJECT_ROOT / "chapter2" / "esn_model.py"
    actual = file_sha256(path)
    if actual != ESN_SOURCE_SHA256:
        raise ProtectionError(f"EchoStateNetwork source changed: {actual}")


def assert_tracked_protected_unchanged() -> None:
    result = subprocess.run(
        ["git", "diff", "--quiet", STARTING_COMMIT, "--", *TRACKED_PROTECTED_PATHS],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise ProtectionError("a protected tracked path differs from the starting commit")


def _load_manifest() -> dict[str, Any]:
    path = CORRECTION_ROOT / "correction_manifest.json"
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("verdict") != "DERIVED CORRECTION AUDIT PASSED":
        raise ProtectionError("baseline correction manifest is not audit-passed")
    return value


def assert_baseline_artifacts_unchanged() -> dict[str, int]:
    """Verify every old model/raw input plus every corrected tabular output."""
    manifest = _load_manifest()
    checked = 0
    for recorded_path, expected in manifest["original_input_hashes"].items():
        path = Path(recorded_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        elif not path.exists():
            marker = "/chapter2/"
            if marker not in recorded_path:
                raise ProtectionError(f"cannot relocate baseline path: {recorded_path}")
            path = PROJECT_ROOT / "chapter2" / recorded_path.split(marker, 1)[1]
        if not path.is_file() or file_sha256(path) != expected:
            raise ProtectionError(f"frozen baseline input hash mismatch: {path}")
        checked += 1
    for name, expected in manifest["output_hashes"].items():
        path = CORRECTION_ROOT / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ProtectionError(f"frozen corrected output hash mismatch: {path}")
        checked += 1
    assert_esn_source_unchanged()
    assert_tracked_protected_unchanged()
    return {"baseline_hashes_checked": checked, "tracked_guards": len(TRACKED_PROTECTED_PATHS)}


def refuse_existing(path: Path, *, resume: bool = False) -> None:
    if path.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


def source_inventory() -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(PACKAGE_ROOT.glob("*.py"))
    }
