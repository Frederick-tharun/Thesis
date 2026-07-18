from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # Reported as a validation failure, not silently skipped.
    Image = None


REQUIRED_PACKAGE_DIRECTORIES = (
    "00_manifest",
    "01_prediction_all_regimes",
    "02_bo_optimization",
    "03_linear_feedback",
    "04_finite_time",
    "05_pyragas",
    "06_comparison_tables",
    "07_report_figures",
    "08_logs",
)
CONTROL_SECTIONS = (
    "03_linear_feedback",
    "04_finite_time",
    "05_pyragas",
)
STANDARD_PREDICTION_FIGURES = {
    "results_all_states.png",
    "results_full_zoom.png",
    "results_zoom_comparison.png",
    "spike_event_comparison.png",
}
_ABSOLUTE_UNIX_PATH = re.compile(r"^/(?:home|scratch|tmp|var|opt|usr)/")


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_strings(
    value: Any,
    *,
    key_path: tuple[str, ...] = (),
) -> list[dict]:
    if key_path and key_path[0] == "machine_specific":
        return []
    if isinstance(value, dict):
        findings = []
        for key, child in value.items():
            findings.extend(
                _absolute_strings(
                    child,
                    key_path=(*key_path, str(key)),
                )
            )
        return findings
    if isinstance(value, list):
        findings = []
        for index, child in enumerate(value):
            findings.extend(
                _absolute_strings(
                    child,
                    key_path=(*key_path, str(index)),
                )
            )
        return findings
    if isinstance(value, str) and _ABSOLUTE_UNIX_PATH.match(value.strip()):
        return [{"key": ".".join(key_path), "value": value}]
    return []


def validate_final_package(
    root,
    *,
    expected_commit: str,
    clean_repository_at_start: bool,
    quality_gates_passed: bool,
    write_report: bool = True,
) -> dict:
    """Validate the curated Chapter 1 package and return a portable report."""
    root = Path(root).resolve()
    manifest_dir = root / "00_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    warnings: list[str] = []
    top_level_dirs = sorted(
        path.name for path in root.iterdir() if path.is_dir()
    )
    missing_dirs = sorted(set(REQUIRED_PACKAGE_DIRECTORIES) - set(top_level_dirs))
    unexpected_dirs = sorted(set(top_level_dirs) - set(REQUIRED_PACKAGE_DIRECTORIES))
    if missing_dirs:
        errors.append("missing package directories: " + ", ".join(missing_dirs))
    if unexpected_dirs:
        errors.append("unexpected package directories: " + ", ".join(unexpected_dirs))

    required_files = [
        "00_manifest/run_manifest.json",
        "00_manifest/stage_timings.csv",
    ]
    for regime in (
        "periodic_spiking",
        "periodic_bursting",
        "chaotic_bursting",
    ):
        required_files.extend(
            [
                f"01_prediction_all_regimes/{regime}/selected_model.json",
                f"01_prediction_all_regimes/{regime}/heldout_test_metrics.json",
                f"01_prediction_all_regimes/{regime}/model_bundle.npz",
                f"02_bo_optimization/{regime}/best_params.json",
                f"02_bo_optimization/{regime}/validation_windows.json",
                (
                    f"02_bo_optimization/{regime}/"
                    "optimizer_validation_summary.csv"
                ),
            ]
        )
    required_files.extend(
        f"{section}/control_summary.json" for section in CONTROL_SECTIONS
    )
    missing_files = [
        path for path in required_files if not (root / path).is_file()
    ]
    if missing_files:
        errors.append("missing required files: " + ", ".join(missing_files))

    json_errors: list[dict] = []
    csv_errors: list[dict] = []
    png_errors: list[dict] = []
    absolute_path_findings: list[dict] = []
    hash_groups: dict[tuple[str, str], list[str]] = {}
    file_hashes: list[dict] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = _relative(root, path)
        if relative == "00_manifest/final_package_validation.json":
            continue
        suffix = path.suffix.lower()
        if suffix in {".json", ".csv", ".png"}:
            digest = _sha256(path)
            file_hashes.append(
                {
                    "file": relative,
                    "extension": suffix,
                    "sha256": digest,
                }
            )
            hash_groups.setdefault((suffix, digest), []).append(relative)

        if suffix == ".json":
            try:
                with path.open(encoding="utf-8") as handle:
                    value = json.load(handle)
                for finding in _absolute_strings(value):
                    absolute_path_findings.append(
                        {"file": relative, **finding}
                    )
            except Exception as exc:
                json_errors.append(
                    {"file": relative, "error": type(exc).__name__}
                )
        elif suffix == ".csv":
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.reader(handle))
                if not rows or not rows[0]:
                    raise ValueError("CSV has no header")
                for row_index, row in enumerate(rows):
                    for column_index, value in enumerate(row):
                        if _ABSOLUTE_UNIX_PATH.match(value.strip()):
                            absolute_path_findings.append(
                                {
                                    "file": relative,
                                    "key": f"row_{row_index}.column_{column_index}",
                                    "value": value,
                                }
                            )
            except Exception as exc:
                csv_errors.append(
                    {"file": relative, "error": type(exc).__name__}
                )
        elif suffix == ".png":
            if Image is None:
                png_errors.append(
                    {"file": relative, "error": "Pillow_not_installed"}
                )
            else:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as exc:
                    png_errors.append(
                        {"file": relative, "error": type(exc).__name__}
                    )

    duplicate_groups = [
        {
            "extension": suffix,
            "sha256": digest,
            "files": files,
        }
        for (suffix, digest), files in sorted(hash_groups.items())
        if len(files) > 1
    ]
    duplicate_prediction_violations = []
    for group in duplicate_groups:
        files = group["files"]
        if group["extension"] != ".png":
            continue
        has_prediction = any(
            path.startswith("01_prediction_all_regimes/") for path in files
        )
        has_controller = any(
            path.startswith(tuple(f"{section}/" for section in CONTROL_SECTIONS))
            for path in files
        )
        if has_prediction and has_controller:
            duplicate_prediction_violations.append(group)

    misplaced_prediction_figures = sorted(
        _relative(root, path)
        for section in CONTROL_SECTIONS
        for path in (root / section).rglob("*.png")
        if path.name in STANDARD_PREDICTION_FIGURES
    )

    if json_errors:
        errors.append("invalid JSON files detected")
    if csv_errors:
        errors.append("invalid CSV files detected")
    if png_errors:
        errors.append("unreadable PNG files detected")
    if absolute_path_findings:
        errors.append("unexpected absolute internal paths detected")
    if duplicate_prediction_violations:
        errors.append("duplicated prediction figures detected in controller sections")
    if misplaced_prediction_figures:
        errors.append("standard prediction figures found in controller sections")

    control_summaries = []
    control_identity_hashes = []
    for section in CONTROL_SECTIONS:
        path = root / section / "control_summary.json"
        if not path.is_file():
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        control_summaries.append(
            {
                "section": section,
                "controller": summary.get("controller"),
                "stable": summary.get("stable"),
                "model_identity_hash": summary.get("model_identity_hash"),
                "control_model_source": summary.get("control_model_source"),
                "reference_type": summary.get("reference_type"),
            }
        )
        control_identity_hashes.append(summary.get("model_identity_hash"))
        if summary.get("control_model_source") != "validation_selected":
            errors.append(f"{section} does not use validation_selected model source")
        if summary.get("reference_type") != "empirical_quiet_state_reference":
            errors.append(f"{section} lacks empirical quiet-state terminology")
        if summary.get("stable") is not True:
            errors.append(f"{section} final selected controller is not stable")

    same_control_model_identity = bool(control_identity_hashes) and (
        len(control_identity_hashes) == len(CONTROL_SECTIONS)
        and None not in control_identity_hashes
        and len(set(control_identity_hashes)) == 1
    )
    if not same_control_model_identity:
        errors.append("final controllers do not share one base ESN identity")

    manifest_commit = None
    manifest_path = root / "00_manifest" / "run_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_commit = manifest.get("git", {}).get("commit")
        except Exception:
            manifest = {}
    manifest_commit_matches = manifest_commit == str(expected_commit)
    if not manifest_commit_matches:
        errors.append("manifest commit does not match the running commit")
    if not clean_repository_at_start:
        errors.append("repository was not clean at run start")
    periodic_gate_passed = False
    periodic_metrics_path = (
        root
        / "01_prediction_all_regimes"
        / "periodic_spiking"
        / "heldout_test_metrics.json"
    )
    if periodic_metrics_path.is_file():
        try:
            periodic_metrics = json.loads(
                periodic_metrics_path.read_text(encoding="utf-8")
            )
            periodic_gate_passed = (
                periodic_metrics.get("quality_gate", {}).get("passed")
                is True
            )
        except Exception:
            periodic_gate_passed = False
    effective_quality_passed = bool(
        quality_gates_passed and periodic_gate_passed
    )
    if not effective_quality_passed:
        errors.append("final prediction quality gates did not pass")

    if duplicate_groups:
        warnings.append(
            "identical PNG/CSV/JSON files are listed for review; only "
            "cross-section prediction duplicates are fatal"
        )

    report = {
        "schema_version": "chapter1_package_validation_v1",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "required_directories": list(REQUIRED_PACKAGE_DIRECTORIES),
        "top_level_directories": top_level_dirs,
        "missing_directories": missing_dirs,
        "unexpected_directories": unexpected_dirs,
        "required_files_present": not missing_files,
        "missing_required_files": missing_files,
        "json_valid": not json_errors,
        "json_errors": json_errors,
        "csv_valid": not csv_errors,
        "csv_errors": csv_errors,
        "png_readable": not png_errors,
        "png_errors": png_errors,
        "no_unexpected_absolute_paths": not absolute_path_findings,
        "absolute_path_findings": absolute_path_findings,
        "file_hashes": file_hashes,
        "duplicate_hash_groups": duplicate_groups,
        "no_unnecessary_duplicate_prediction_figures": (
            not duplicate_prediction_violations
        ),
        "duplicate_prediction_violations": duplicate_prediction_violations,
        "misplaced_prediction_figures": misplaced_prediction_figures,
        "expected_controller_summaries_present": (
            len(control_summaries) == len(CONTROL_SECTIONS)
        ),
        "controller_summaries": control_summaries,
        "same_control_model_identity": same_control_model_identity,
        "manifest_commit": manifest_commit,
        "expected_commit": str(expected_commit),
        "manifest_commit_matches_current_commit": manifest_commit_matches,
        "clean_repository_at_run_start": bool(clean_repository_at_start),
        "periodic_spiking_quality_gate_passed": periodic_gate_passed,
        "final_quality_gates_passed": effective_quality_passed,
    }
    if write_report:
        destination = manifest_dir / "final_package_validation.json"
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def assert_valid_final_package(*args, **kwargs) -> dict:
    report = validate_final_package(*args, **kwargs)
    if not report["valid"]:
        raise RuntimeError(
            "Final package validation failed: " + "; ".join(report["errors"])
        )
    return report
