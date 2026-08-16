"""Read-only-by-default verifier for the Chapter 2 thesis release."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA = "chapter2_release_manifest_v1"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "release" / "release_manifest.json"


class ReleaseVerificationError(RuntimeError):
    """Raised when the release manifest itself is unsafe or malformed."""


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def load_strict_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_constant=_reject_nonfinite)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ReleaseVerificationError("artifact path must be a non-empty string")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ReleaseVerificationError(
            f"artifact path escapes repository root: {relative}"
        ) from error
    return candidate


def _artifact_entries(section: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    artifacts = section.get("artifacts")
    if not isinstance(artifacts, list):
        raise ReleaseVerificationError("artifact section must contain a list")
    if any(not isinstance(item, Mapping) for item in artifacts):
        raise ReleaseVerificationError("artifact entries must be objects")
    return artifacts


def verify_release(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Verify hashes/counts/strict JSON without loading models or writing files."""
    manifest_path = Path(manifest_path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else manifest_path.parents[2]
    )
    errors: list[str] = []
    checked: list[str] = []

    try:
        manifest = load_strict_json(manifest_path)
    except Exception as error:
        return {
            "schema": "chapter2_release_verification_v1",
            "valid": False,
            "manifest": str(manifest_path),
            "checked_file_count": 0,
            "errors": [f"strict manifest parse failed: {error}"],
        }
    if not isinstance(manifest, Mapping) or manifest.get("schema") != SCHEMA:
        errors.append("release manifest schema mismatch")

    def check_artifact(item: Mapping[str, Any], *, strict_json: bool = False) -> None:
        relative = item.get("path")
        try:
            path = _safe_path(root, relative)
        except ReleaseVerificationError as error:
            errors.append(str(error))
            return
        if not path.is_file():
            errors.append(f"missing artifact: {relative}")
            return
        expected_hash = item.get("sha256")
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            errors.append(f"hash mismatch: {relative}")
        expected_size = item.get("size_bytes")
        if expected_size is not None and path.stat().st_size != expected_size:
            errors.append(f"size mismatch: {relative}")
        if strict_json:
            try:
                load_strict_json(path)
            except Exception as error:
                errors.append(f"strict JSON parse failed for {relative}: {error}")
        checked.append(str(relative))

    try:
        figure_section = manifest["official_figures"]
        table_section = manifest["final_tables"]
        figures = _artifact_entries(figure_section)
        tables = _artifact_entries(table_section)
    except (KeyError, ReleaseVerificationError, TypeError) as error:
        errors.append(f"invalid release artifact sections: {error}")
        figures = []
        tables = []

    for item in figures:
        check_artifact(item)
    for item in tables:
        check_artifact(item)

    pdf_count = sum(str(item.get("path", "")).endswith(".pdf") for item in figures)
    png_count = sum(str(item.get("path", "")).endswith(".png") for item in figures)
    if pdf_count != 4 or png_count != 4 or len(figures) != 8:
        errors.append("official figure manifest must list exactly four PDFs and four PNGs")
    if len(tables) != 5:
        errors.append("final table manifest must list exactly five CSV files")

    for section_name, expected_names in (
        (
            "official_figures",
            {
                Path(str(item.get("path"))).name for item in figures
            }
            | {Path(str(manifest.get("figure_manifest", {}).get("path", ""))).name},
        ),
        (
            "final_tables",
            {Path(str(item.get("path"))).name for item in tables},
        ),
    ):
        section = manifest.get(section_name, {})
        try:
            directory = _safe_path(root, section.get("directory"))
        except ReleaseVerificationError as error:
            errors.append(str(error))
            continue
        actual_names = (
            {item.name for item in directory.iterdir() if item.is_file()}
            if directory.is_dir()
            else set()
        )
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        if missing:
            errors.append(f"{section_name} missing names: {missing}")
        if unexpected:
            errors.append(f"{section_name} unexpected names: {unexpected}")

    figure_manifest = manifest.get("figure_manifest")
    if isinstance(figure_manifest, Mapping):
        check_artifact(figure_manifest, strict_json=True)
    else:
        errors.append("figure_manifest entry is missing")

    references = manifest.get("references")
    if not isinstance(references, Mapping):
        errors.append("references section is missing")
    else:
        for name, expected_count_key, list_key in (
            ("raw_results", "expected_record_count", "records"),
            ("model_manifest", "expected_model_count", "models"),
        ):
            item = references.get(name)
            if not isinstance(item, Mapping):
                errors.append(f"missing reference: {name}")
                continue
            check_artifact(item, strict_json=True)
            try:
                value = load_strict_json(_safe_path(root, item["path"]))
                actual_count = len(value[list_key])
                if actual_count != item[expected_count_key]:
                    errors.append(f"{name} count mismatch")
            except Exception as error:
                errors.append(f"{name} content validation failed: {error}")

    sources = manifest.get("release_sources")
    if not isinstance(sources, Mapping):
        errors.append("release_sources section is missing")
    else:
        try:
            source_items = _artifact_entries(sources)
        except ReleaseVerificationError as error:
            errors.append(str(error))
            source_items = []
        for item in source_items:
            check_artifact(item)

    return {
        "schema": "chapter2_release_verification_v1",
        "valid": not errors,
        "manifest": str(manifest_path),
        "checked_file_count": len(checked),
        "official_pdf_count": pdf_count,
        "official_png_count": png_count,
        "final_table_count": len(tables),
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for a new verification report; omitted means no writes",
    )
    arguments = parser.parse_args(argv)
    report = verify_release(arguments.manifest)
    if arguments.output is not None:
        atomic_write_text(arguments.output, strict_json_text(report))
    if report["valid"]:
        print(
            "PASS: "
            f"{report['official_pdf_count']} PDFs, "
            f"{report['official_png_count']} PNGs, "
            f"{report['final_table_count']} tables; "
            f"{report['checked_file_count']} files checked"
        )
        return 0
    print(f"FAIL: {len(report['errors'])} release-integrity error(s)")
    for error in report["errors"]:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
