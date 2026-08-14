"""Focused tests for the non-mutating Chapter 2 release verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chapter2 import verify_release as verifier


def _entry(root: Path, path: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": verifier.file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


@pytest.fixture
def release_fixture(tmp_path: Path) -> Path:
    figure_dir = tmp_path / "chapter2" / "final_results" / "figures_thesis"
    table_dir = tmp_path / "chapter2" / "final_results" / "tables_final"
    model_dir = tmp_path / "chapter2" / "final_models"
    release_dir = tmp_path / "chapter2" / "release"
    figure_dir.mkdir(parents=True)
    table_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    release_dir.mkdir(parents=True)

    figures = []
    for index in range(1, 5):
        for suffix in ("pdf", "png"):
            path = figure_dir / f"{index:02d}_figure.{suffix}"
            path.write_bytes(f"figure-{index}-{suffix}".encode())
            figures.append(_entry(tmp_path, path))
    figure_manifest_path = figure_dir / "figure_manifest.json"
    figure_manifest_path.write_text(
        json.dumps({"schema": "fixture", "figure_count": 4}),
        encoding="utf-8",
    )

    tables = []
    for index in range(1, 6):
        path = table_dir / f"{index:02d}_table.csv"
        path.write_text("value\n1\n", encoding="utf-8")
        tables.append(_entry(tmp_path, path))

    raw_path = tmp_path / "chapter2" / "final_results" / "step8_raw_results.json"
    raw_path.write_text(
        json.dumps({"records": [{"id": 1}, {"id": 2}]}), encoding="utf-8"
    )
    model_path = model_dir / "model_manifest.json"
    model_path.write_text(json.dumps({"models": [{"id": 1}]}), encoding="utf-8")
    source_path = tmp_path / "chapter2" / "verify_release.py"
    source_path.write_text("# fixture source\n", encoding="utf-8")

    manifest = {
        "schema": verifier.SCHEMA,
        "release_version": "fixture-v1",
        "official_figures": {
            "directory": figure_dir.relative_to(tmp_path).as_posix(),
            "artifacts": figures,
        },
        "figure_manifest": _entry(tmp_path, figure_manifest_path),
        "final_tables": {
            "directory": table_dir.relative_to(tmp_path).as_posix(),
            "artifacts": tables,
        },
        "references": {
            "raw_results": {
                **_entry(tmp_path, raw_path),
                "expected_record_count": 2,
            },
            "model_manifest": {
                **_entry(tmp_path, model_path),
                "expected_model_count": 1,
            },
        },
        "release_sources": {"artifacts": [_entry(tmp_path, source_path)]},
    }
    manifest_path = release_dir / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_default_verification_is_non_mutating(release_fixture: Path) -> None:
    root = release_fixture.parents[2]
    before = {
        path: (path.stat().st_mtime_ns, verifier.file_sha256(path))
        for path in root.rglob("*")
        if path.is_file()
    }
    report = verifier.verify_release(release_fixture)
    after = {
        path: (path.stat().st_mtime_ns, verifier.file_sha256(path))
        for path in root.rglob("*")
        if path.is_file()
    }
    assert report["valid"] is True
    assert report["official_pdf_count"] == 4
    assert report["official_png_count"] == 4
    assert report["final_table_count"] == 5
    assert after == before


@pytest.mark.parametrize("failure", ["missing", "hash", "unexpected"])
def test_missing_hash_mismatch_and_unexpected_artifacts_are_detected(
    release_fixture: Path, failure: str
) -> None:
    manifest = json.loads(release_fixture.read_text(encoding="utf-8"))
    root = release_fixture.parents[2]
    figure = root / manifest["official_figures"]["artifacts"][0]["path"]
    if failure == "missing":
        figure.unlink()
    elif failure == "hash":
        figure.write_bytes(b"changed")
    else:
        figure.with_name("unexpected.png").write_bytes(b"unexpected")
    report = verifier.verify_release(release_fixture)
    assert report["valid"] is False
    assert any(failure.split("_")[0] in error for error in report["errors"])


def test_nonfinite_manifest_is_rejected_strictly(release_fixture: Path) -> None:
    release_fixture.write_text('{"schema": NaN}', encoding="utf-8")
    report = verifier.verify_release(release_fixture)
    assert report["valid"] is False
    assert "strict manifest parse failed" in report["errors"][0]


def test_output_is_written_only_when_explicitly_requested(
    release_fixture: Path, tmp_path: Path
) -> None:
    output = tmp_path / "reports" / "verification.json"
    assert verifier.main(
        ["--manifest", str(release_fixture), "--output", str(output)]
    ) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["valid"] is True
