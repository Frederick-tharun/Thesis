"""Audited generic correction for divergent continuous-interval event metrics.

This postprocessing correction does not rerun forecasts, read source benchmark
arrays, or change models, scalers, locks, state metrics, or aggregates.
"""

from __future__ import annotations

from pathlib import Path
import shutil

from chapter2.esn_data import file_sha256
from chapter2.esn_step8 import (
    AGGREGATE_PATH,
    CHAPTER2_ROOT,
    EVALUATION_MANIFEST,
    MODEL_MANIFEST_PATH,
    RAW_RESULTS_PATH,
    SELECTION_LOCK,
    STATUS_PATH,
    VERIFICATION_PATH,
    atomic_write_json,
    invalidate_divergent_event_metrics,
    load_strict_json,
    project_relative,
    update_status,
    utc_now,
    verify_results,
)


AUDIT_ROOT = (
    CHAPTER2_ROOT
    / "final_results"
    / "post_benchmark_event_correction"
)
PRE_ROOT = AUDIT_ROOT / "pre_correction"
POST_ROOT = AUDIT_ROOT / "post_correction"
AUDIT_MANIFEST = AUDIT_ROOT / "correction_manifest.json"


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.name: file_sha256(path)
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def main() -> None:
    status = load_strict_json(STATUS_PATH)
    if status.get("first_benchmark_access_timestamp") is None:
        raise RuntimeError("correction requires recorded benchmark access")
    if status.get("state") != "FAILED":
        raise RuntimeError("correction requires the audited FAILED checkpoint")
    if not PRE_ROOT.is_dir():
        raise RuntimeError("pre-correction audit artifacts are missing")

    raw = load_strict_json(RAW_RESULTS_PATH)
    pre_raw_hash = file_sha256(RAW_RESULTS_PATH)
    locked_source_hash = load_strict_json(SELECTION_LOCK)["source_code_hashes"][
        "chapter2/esn_step8.py"
    ]
    affected: list[dict[str, object]] = []
    for record in raw["records"]:
        if record["family"] != "continuous":
            continue
        for interval in record["per_current_interval"]:
            before = interval["event_metrics"]
            corrected = invalidate_divergent_event_metrics(
                before, interval["metrics"]
            )
            if corrected != before:
                interval["event_metrics"] = corrected
                affected.append(
                    {
                        "record_id": record["record_id"],
                        "seed": record["seed"],
                        "model_type": record["model_type"],
                        "current": interval["current"],
                        "transition_range": interval["transition_range"],
                    }
                )
    if len(affected) != 6:
        raise RuntimeError(
            f"expected exactly six affected intervals, found {len(affected)}"
        )

    correction = {
        "type": "generic_postprocessing_defect",
        "decided_at": utc_now(),
        "benchmark_access_had_started": True,
        "reason": (
            "A continuous per-current interval whose state metrics diverged "
            "must have undefined spike/burst metrics. Forecasts and state "
            "metrics were already correct."
        ),
        "scientific_scope": (
            "Event-metric defined flags/errors only; no forecast, model, scaler, "
            "lock, state metric, aggregate, window, seed, or hyperparameter changed."
        ),
        "rerun_method": (
            "Deterministic postprocessing of the existing strict JSON record; "
            "source benchmark arrays were not reopened and forecasts were not rerun."
        ),
        "affected_intervals": affected,
        "affected_interval_count": len(affected),
        "pre_correction_raw_sha256": pre_raw_hash,
        "locked_esn_step8_source_sha256": locked_source_hash,
        "corrected_esn_step8_source_sha256": file_sha256(
            CHAPTER2_ROOT / "esn_step8.py"
        ),
        "regression_test": (
            "chapter2/tests/test_esn_step8.py::"
            "test_divergent_interval_event_metrics_are_invalidated"
        ),
    }
    raw.setdefault("post_benchmark_corrections", []).append(correction)
    atomic_write_json(RAW_RESULTS_PATH, raw)
    correction["post_correction_raw_sha256"] = file_sha256(RAW_RESULTS_PATH)

    aggregates = load_strict_json(AGGREGATE_PATH)
    model_manifest = load_strict_json(MODEL_MANIFEST_PATH)
    previous_verification = load_strict_json(VERIFICATION_PATH)
    verification = verify_results(
        status["preflight"],
        status["lock_hashes"],
        model_manifest,
        raw,
        aggregates,
        previous_verification["figure_artifacts"],
    )
    verification["post_benchmark_correction"] = correction
    verification["state_error_aggregates_changed_by_correction"] = False
    atomic_write_json(VERIFICATION_PATH, verification)

    update_status(
        "STEP8_COMPLETE",
        completed_record_ids=[item["record_id"] for item in raw["records"]],
        completed_record_count=len(raw["records"]),
        raw_results_hash=file_sha256(RAW_RESULTS_PATH),
        aggregate_hash=file_sha256(AGGREGATE_PATH),
        verification_hash=file_sha256(VERIFICATION_PATH),
        post_benchmark_correction=correction,
        failure=None,
    )

    POST_ROOT.mkdir(parents=True, exist_ok=True)
    for path in (RAW_RESULTS_PATH, AGGREGATE_PATH, VERIFICATION_PATH, STATUS_PATH):
        shutil.copy2(path, POST_ROOT / path.name)
    manifest = {
        "schema": "chapter2_step8_post_benchmark_correction_v1",
        "created_at": utc_now(),
        "pre_correction_artifact_hashes": _artifact_hashes(PRE_ROOT),
        "post_correction_artifact_hashes": _artifact_hashes(POST_ROOT),
        "selection_lock_sha256": file_sha256(SELECTION_LOCK),
        "evaluation_manifest_sha256": file_sha256(EVALUATION_MANIFEST),
        "model_manifest_sha256": file_sha256(MODEL_MANIFEST_PATH),
        "correction": correction,
    }
    atomic_write_json(AUDIT_MANIFEST, manifest)
    print(
        f"Corrected {len(affected)} divergent continuous interval event records.",
        flush=True,
    )


if __name__ == "__main__":
    main()
