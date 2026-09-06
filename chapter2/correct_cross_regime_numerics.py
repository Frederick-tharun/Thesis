"""Derive corrected metadata from the completed benchmark, without forecasting.

Run ``python -m chapter2.correct_cross_regime_numerics`` on the host containing
the original artifacts. Outputs go into a new directory; historical JSON and
NPZ files are only read. ``--audit-only`` verifies an existing correction.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from chapter2 import cross_regime as core
from chapter2 import run_cross_regime as runner
from chapter2.cross_regime_config import ALL_CURRENTS, BRANCH, EXPECTED_RECORDS
from chapter2.cross_regime_numerics import NUMERICAL_POLICY
from chapter2.cross_regime_provenance import (
    correction_source_provenance,
    verify_original_source_lock,
)
from chapter2.esn_data import file_sha256, load_fixed_trajectory


CORRECTION_SCHEMA = "chapter2_cross_regime_post_hoc_numerical_correction_v1"
CORRECTION_REASON = (
    "Physical inverse-scaling overflow was classified after the stored failure "
    "step or not classified; finite residuals could overflow established metric "
    "arithmetic. Reclassify and recompute derived metrics from original stored "
    "predictions using the shared floating-point representability policy."
)
PROVENANCE_STATEMENT = (
    "Original forecasts were produced under the frozen source lock; numerical "
    "classification and derived metrics were corrected afterward without "
    "rerunning the forecasts."
)
CLASSIFICATION_FIELDS = (
    "failure_step", "numerical_failure", "failure_reason", "aggregate_nrmse_value"
)
OUTPUT_NAMES = frozenset({
    "corrected_results.json", "corrected_aggregate_results.json", "correction_changes.json",
    "fixed_short_results.csv", "fixed_long_results.csv", "continuous_schedule_results.csv",
    "transfer_target_comparison.csv", "per_seed_summary.csv", "divergence_summary.csv",
    "scenario_comparison.csv",
})


def _changes(old: Any, new: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Record exact changes, including newly added and removed metadata fields."""
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        result = []
        for key in sorted(set(old) | set(new)):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in old or key not in new:
                result.append({"field": name, "old_present": key in old,
                               "new_present": key in new,
                               "old": old.get(key), "new": new.get(key)})
            else:
                result.extend(_changes(old[key], new[key], name))
        return result
    if old == new:
        return []
    return [{"field": prefix, "old_present": True, "new_present": True,
             "old": old, "new": new}]


def correct_record(
    record: Mapping[str, Any],
    *,
    normalisation_scale: np.ndarray,
    trajectory: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read and validate one historical NPZ; derive metadata without saving it."""
    arrays = core.validate_raw_array(record)  # Hash checked before array loading.
    core.validate_record_source(record, arrays, trajectory)
    fields, pointwise = core.derive_record_fields(record, arrays, normalisation_scale)
    corrected = deepcopy(dict(record))
    corrected.update(fields)
    corrected["derived_pointwise_sha256"] = core.pointwise_sha256(pointwise)
    return corrected, _record_change(record, corrected)


def _record_change(record: Mapping[str, Any], corrected: Mapping[str, Any]) -> dict[str, Any] | None:
    changes = _changes(record, corrected)
    if not changes:
        return None
    reasons = ["shared canonical numerical classification and metric recomputation"]
    if record.get("failure_step") != corrected["failure_step"]:
        reasons.append("first numerical failure step corrected")
    if record.get("failure_reason") != corrected["failure_reason"]:
        reasons.append(str(corrected["failure_reason"] or "no numerical failure"))
    return {
        "record_id": record["record_id"],
        "reasons": reasons,
        "old": {key: record.get(key) for key in CLASSIFICATION_FIELDS},
        "new": {key: corrected.get(key) for key in CLASSIFICATION_FIELDS},
        "changes": changes,
        "original_raw_arrays_path": record["raw_arrays_path"],
        "original_raw_arrays_sha256": record["raw_arrays_sha256"],
    }


def _verify_hashes(inventory: Mapping[str, str], *, label: str) -> None:
    for name, expected in inventory.items():
        path = Path(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise core.CrossRegimeError(f"{label} hash mismatch: {path}")


def _load_originals() -> dict[str, Any]:
    """Validate frozen inputs and execution provenance without invoking a runner."""
    from chapter2.audit_cross_regime import audit_datasets, audit_models

    paths = (runner.MANIFEST_PATH, runner.STATUS_PATH, runner.RAW_RESULTS_PATH,
             runner.MODEL_MANIFEST_PATH, runner.DATASET_MANIFEST_PATH,
             runner.AGGREGATE_PATH)
    if any(not path.is_file() for path in paths):
        raise core.CrossRegimeError("original completed benchmark inputs are missing")
    before = core.file_hash_inventory(paths)
    manifest, status, raw, models, datasets, historical_aggregate = (
        core.strict_load_json(path) for path in paths
    )
    if status.get("state") not in ("BENCHMARK_COMPLETE_AUDIT_REQUIRED", "COMPLETE"):
        raise core.CrossRegimeError("original benchmark is not complete")
    if status.get("record_count") != EXPECTED_RECORDS:
        raise core.CrossRegimeError("original benchmark status record count mismatch")
    if status.get("aggregate_sha256") != before[str(runner.AGGREGATE_PATH)]:
        raise core.CrossRegimeError("original aggregate status hash mismatch")
    provenance = verify_original_source_lock(
        manifest, project_root=runner.PROJECT_ROOT, execution_info=status
    )
    expected_locks = {
        "protocol_manifest": before[str(runner.MANIFEST_PATH)],
        "model_manifest": before[str(runner.MODEL_MANIFEST_PATH)],
        "dataset_manifest": before[str(runner.DATASET_MANIFEST_PATH)],
    }
    if raw.get("lock_hashes") != expected_locks:
        raise core.CrossRegimeError("original raw-result locks mismatch")
    core.validate_record_matrix(raw["records"])
    raw_paths = [Path(item["raw_arrays_path"]).resolve() for item in raw["records"]]
    if len(set(raw_paths)) != EXPECTED_RECORDS:
        raise core.CrossRegimeError("original raw artifact paths are not 345 unique files")
    fixed_hashes = runner.dataset_hashes()
    if fixed_hashes != manifest["preflight"]["fixed_dataset_hashes"]:
        raise core.CrossRegimeError("fixed datasets differ from frozen manifest")
    dataset_audit = audit_datasets(datasets)
    model_audit = audit_models(models)
    if not runner.protected_paths_unchanged():
        raise core.CrossRegimeError("protected tracked path changed")
    original_binaries = runner.original_binary_inventory()
    if not original_binaries["valid"]:
        raise core.CrossRegimeError("original 226 scientific binaries changed")
    before.update({str(runner.PROJECT_ROOT / name): digest
                   for name, digest in fixed_hashes.items()})
    for item in [*models["models"], *datasets["datasets"]]:
        before[str(runner.PROJECT_ROOT / item["path"])] = item["sha256"]
    before.update({str(path): item["raw_arrays_sha256"]
                   for path, item in zip(raw_paths, raw["records"])})
    _verify_hashes(before, label="original input")
    return {"manifest": manifest, "status": status, "raw": raw, "models": models,
            "datasets": datasets, "input_hashes": before, "provenance": provenance,
            "model_audit": model_audit, "dataset_audit": dataset_audit,
            "original_binaries": original_binaries,
            "historical_aggregate": historical_aggregate}


def _derive_records(inputs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    scalers = {}
    for item in inputs["models"]["models"]:
        _, scaler, _ = core.load_model_bundle(runner.PROJECT_ROOT / item["path"])
        scalers[(item["scenario"], int(item["seed"]))] = scaler
    fixed = {current: load_fixed_trajectory(current) for current in ALL_CURRENTS}
    schedules = runner.load_schedules(inputs["datasets"])
    corrected_records, changed = [], []
    for item in inputs["raw"]["records"]:
        trajectory = (schedules[item["schedule"]] if item["family"] == "continuous"
                      else fixed[item["current"]])
        corrected, change = correct_record(
            item,
            normalisation_scale=scalers[(item["scenario"], int(item["seed"]))].state.scale,
            trajectory=trajectory,
        )
        corrected_records.append(corrected)
        if change is not None:
            changed.append(change)
    core.validate_record_matrix(corrected_records)
    if [item["record_id"] for item in corrected_records] != [
        item["record_id"] for item in inputs["raw"]["records"]
    ]:
        raise core.CrossRegimeError("correction changed record identities or order")
    corrected_raw = {
        "schema": CORRECTION_SCHEMA,
        "created_at": runner.utc_now(),
        "lock_hashes": deepcopy(inputs["raw"]["lock_hashes"]),
        "original_raw_results_sha256": file_sha256(runner.RAW_RESULTS_PATH),
        "records": corrected_records,
        "derived_only": True,
        "raw_prediction_artifacts_rewritten": False,
    }
    return corrected_raw, _change_log(corrected_records, changed)


def _change_log(records: Sequence[Mapping[str, Any]], changed: list[dict[str, Any]]) -> dict[str, Any]:
    changed_ids = {change["record_id"] for change in changed}
    return {
        "schema": CORRECTION_SCHEMA,
        "record_count": len(records),
        "changed_record_count": len(changed),
        "classification_or_penalty_changed_count": sum(
            change["old"] != change["new"] for change in changed
        ),
        "unchanged_record_ids": [item["record_id"] for item in records
                                 if item["record_id"] not in changed_ids],
        "records": changed,
    }


def _output_root(output_dir: Path | None) -> Path:
    root = (output_dir or runner.RESULT_ROOT / "post_hoc_numerical_correction").resolve()
    if not root.is_relative_to(runner.RESULT_ROOT.resolve()) or root == runner.RESULT_ROOT.resolve():
        raise core.CrossRegimeError("correction output must be a new subdirectory of result root")
    return root


def run_correction(output_dir: Path | None = None) -> dict[str, Any]:
    """Correct all 345 existing artifacts into an exclusively new output folder."""
    from chapter2.audit_cross_regime import audit_records

    if runner.git_output("branch", "--show-current") != BRANCH:
        raise core.CrossRegimeError(f"correction requires branch {BRANCH}")
    root = _output_root(output_dir)
    if root.exists():
        raise core.CrossRegimeError(f"refusing to overwrite existing correction output: {root}")
    source = correction_source_provenance(
        project_root=runner.PROJECT_ROOT, source_paths=runner.SOURCE_PATHS
    )
    inputs = _load_originals()
    corrected, change_log = _derive_records(inputs)
    record_audit = audit_records(corrected, inputs["models"], derived=True)
    aggregate = runner.aggregate_results(corrected, write=False)
    change_log["aggregate_changes"] = _changes(
        {key: value for key, value in inputs["historical_aggregate"].items()
         if key != "created_at"},
        {key: value for key, value in aggregate.items() if key != "created_at"},
    )
    source_hashes = {str(runner.PROJECT_ROOT / name): digest
                     for name, digest in source["source_hashes"].items()}
    _verify_hashes(source_hashes, label="correction source")
    _verify_hashes(inputs["input_hashes"], label="preserved original input")
    root.mkdir(parents=True, exist_ok=False)
    core.atomic_write_json(root / "corrected_results.json", corrected)
    core.atomic_write_json(root / "corrected_aggregate_results.json", aggregate)
    core.atomic_write_json(root / "correction_changes.json", change_log)
    runner.write_tables(corrected["records"], aggregate, output_root=root)
    _verify_hashes(inputs["input_hashes"], label="preserved original input")
    output_hashes = {path.relative_to(root).as_posix(): file_sha256(path)
                     for path in sorted(root.rglob("*")) if path.is_file()}
    if set(output_hashes) != OUTPUT_NAMES:
        raise core.CrossRegimeError("correction output inventory mismatch")
    _verify_hashes(source_hashes, label="correction source")
    manifest = {
        "schema": CORRECTION_SCHEMA,
        "numerical_policy": NUMERICAL_POLICY,
        "corrected_at": runner.utc_now(),
        "reason": CORRECTION_REASON,
        "provenance_statement": PROVENANCE_STATEMENT,
        "no_model_retraining": True,
        "no_forecast_rerun": True,
        "post_hoc_numerical_classification_and_metric_repair_only": True,
        "original_prediction_artifacts_preserved": True,
        "original_prediction_artifact_count": len(inputs["raw"]["records"]),
        "original_benchmark_commit": inputs["provenance"]["original_benchmark_commit"],
        "original_source_lock": inputs["provenance"],
        "original_frozen_manifest_sha256": file_sha256(runner.MANIFEST_PATH),
        "original_raw_results_sha256": file_sha256(runner.RAW_RESULTS_PATH),
        "original_raw_artifacts": [{"record_id": item["record_id"],
                                    "path": item["raw_arrays_path"],
                                    "sha256": item["raw_arrays_sha256"]}
                                   for item in inputs["raw"]["records"]],
        "original_input_hashes": inputs["input_hashes"],
        "correction_source": source,
        "output_hashes": output_hashes,
        "changed_record_count": change_log["changed_record_count"],
        "classification_or_penalty_changed_count": change_log["classification_or_penalty_changed_count"],
        "record_audit": record_audit,
        "model_audit": inputs["model_audit"],
        "dataset_audit": inputs["dataset_audit"],
        "original_226_scientific_binaries": inputs["original_binaries"],
        "verdict": "DERIVED CORRECTION AUDIT PASSED",
    }
    core.atomic_write_json(root / "correction_manifest.json", manifest)
    return manifest


def audit_correction(output_dir: Path | None = None) -> dict[str, Any]:
    """Recheck original provenance, output hashes and all corrected records."""
    from chapter2.audit_cross_regime import audit_records, close

    root = _output_root(output_dir)
    manifest = core.strict_load_json(root / "correction_manifest.json")
    if (manifest.get("schema") != CORRECTION_SCHEMA or
            manifest.get("numerical_policy") != NUMERICAL_POLICY):
        raise core.CrossRegimeError("correction manifest schema mismatch")
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*")
                    if path.is_file() and path.name != "correction_manifest.json"}
    if set(manifest["output_hashes"]) != OUTPUT_NAMES or actual_files != OUTPUT_NAMES:
        raise core.CrossRegimeError("correction output inventory mismatch")
    _verify_hashes({str(root / name): digest for name, digest in manifest["output_hashes"].items()},
                   label="correction output")
    _verify_hashes(manifest["original_input_hashes"], label="preserved original input")
    inputs = _load_originals()
    if (inputs["input_hashes"] != manifest["original_input_hashes"] or
            inputs["provenance"] != manifest["original_source_lock"]):
        raise core.CrossRegimeError("correction original provenance mismatch")
    original_fields = {
        "original_benchmark_commit": inputs["provenance"]["original_benchmark_commit"],
        "original_frozen_manifest_sha256": file_sha256(runner.MANIFEST_PATH),
        "original_raw_results_sha256": file_sha256(runner.RAW_RESULTS_PATH),
        "original_raw_artifacts": [{"record_id": item["record_id"], "path": item["raw_arrays_path"],
                                    "sha256": item["raw_arrays_sha256"]}
                                   for item in inputs["raw"]["records"]],
        "original_prediction_artifact_count": EXPECTED_RECORDS,
        "original_prediction_artifacts_preserved": True,
        "no_model_retraining": True,
        "no_forecast_rerun": True,
        "post_hoc_numerical_classification_and_metric_repair_only": True,
        "model_audit": inputs["model_audit"],
        "dataset_audit": inputs["dataset_audit"],
        "original_226_scientific_binaries": inputs["original_binaries"],
        "reason": CORRECTION_REASON,
        "provenance_statement": PROVENANCE_STATEMENT,
        "verdict": "DERIVED CORRECTION AUDIT PASSED",
    }
    if any(manifest.get(key) != value for key, value in original_fields.items()):
        raise core.CrossRegimeError("correction manifest provenance claims mismatch")
    current_source = correction_source_provenance(
        project_root=runner.PROJECT_ROOT, source_paths=runner.SOURCE_PATHS
    )
    if current_source["source_hashes"] != manifest["correction_source"]["source_hashes"]:
        raise core.CrossRegimeError("correction source hash mismatch")
    corrected = core.strict_load_json(root / "corrected_results.json")
    if (corrected.get("schema") != CORRECTION_SCHEMA or
            corrected.get("lock_hashes") != inputs["raw"]["lock_hashes"] or
            corrected.get("original_raw_results_sha256") != manifest["original_raw_results_sha256"] or
            corrected.get("derived_only") is not True or
            corrected.get("raw_prediction_artifacts_rewritten") is not False or
            len(corrected["records"]) != len(inputs["raw"]["records"])):
        raise core.CrossRegimeError("corrected results original provenance mismatch")
    changes = []
    for original, derived in zip(inputs["raw"]["records"], corrected["records"]):
        if any(original[key] != derived[key] for key in
               ("record_id", "raw_arrays_path", "raw_arrays_sha256")):
            raise core.CrossRegimeError("derived record original artifact identity mismatch")
        change = _record_change(original, derived)
        if change is not None:
            changes.append(change)
    record_audit = audit_records(corrected, inputs["models"], derived=True)
    if record_audit != manifest["record_audit"]:
        raise core.CrossRegimeError("correction record audit mismatch")
    aggregate = runner.aggregate_results(corrected, write=False)
    saved = core.strict_load_json(root / "corrected_aggregate_results.json")
    if not close({key: value for key, value in aggregate.items() if key != "created_at"},
                 {key: value for key, value in saved.items() if key != "created_at"}):
        raise core.CrossRegimeError("corrected aggregate recomputation mismatch")
    expected_changes = _change_log(corrected["records"], changes)
    expected_changes["aggregate_changes"] = _changes(
        {key: value for key, value in inputs["historical_aggregate"].items() if key != "created_at"},
        {key: value for key, value in saved.items() if key != "created_at"},
    )
    if core.strict_load_json(root / "correction_changes.json") != expected_changes:
        raise core.CrossRegimeError("correction change log recomputation mismatch")
    if any(manifest.get(key) != expected_changes[key] for key in
           ("changed_record_count", "classification_or_penalty_changed_count")):
        raise core.CrossRegimeError("correction change counts mismatch")
    _verify_hashes(inputs["input_hashes"], label="preserved original input")
    return {"verdict": "DERIVED CORRECTION AUDIT PASSED",
            "provenance_statement": PROVENANCE_STATEMENT, "record_audit": record_audit}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    result = (audit_correction if args.audit_only else run_correction)(args.output_dir)
    print(result["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
