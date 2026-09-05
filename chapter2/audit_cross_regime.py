"""Fail-closed independent audit and presentation for cross-regime results."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from chapter2.cross_regime import (
    CrossRegimeError,
    atomic_write_json,
    atomic_write_text,
    expected_model_keys,
    file_hash_inventory,
    load_model_bundle,
    prepare_training,
    strict_load_json,
    record_id,
    evaluation_classification,
    transition_boundary_metrics,
    validate_raw_array,
    validate_record_matrix,
)
from chapter2.cross_regime_config import (
    BASE_COMMIT,
    CONTINUOUS_SCHEDULES,
    CONTINUOUS_SWITCH_INDICES,
    EFFECTIVE_TRAINING_BUDGET,
    EXPECTED_BINARY_ARTIFACTS,
    EXPECTED_MODELS,
    EXPECTED_RECORDS,
    EXPECTED_SCHEDULE_DATASETS,
    MODEL_ROOT,
    REPRESENTATIVE_SEED,
    RESULT_ROOT,
    SCENARIO_TRAINING_CURRENTS,
    SEEDS,
    model_config,
)
from chapter2.esn_data import file_sha256, load_fixed_trajectory
from chapter2.esn_optimisation import NONFINITE_FAILURE_SCORE
from chapter2.esn_metrics import evaluate_rollout, pointwise_normalised_error
from chapter2.esn_step8 import event_metrics, invalidate_divergent_event_metrics
from chapter2.run_cross_regime import (
    AGGREGATE_PATH,
    DATASET_MANIFEST_PATH,
    HASH_PATH,
    MANIFEST_PATH,
    MODEL_MANIFEST_PATH,
    PROJECT_ROOT,
    PROTOCOL_DOCUMENT,
    RAW_RESULTS_PATH,
    RESULTS_DOCUMENT,
    STATUS_PATH,
    VERIFICATION_PATH,
    aggregate_results,
    original_binary_inventory,
    protected_paths_unchanged,
    source_hashes,
    load_schedules,
    utc_now,
)


FIGURE_ROOT = RESULT_ROOT / "figures"


class CrossRegimeAuditError(CrossRegimeError):
    """Raised for any missing, duplicated, corrupt, or inconsistent artifact."""


def close(left: Any, right: Any, tolerance: float = 1.0e-12) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    if left is None or right is None or isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    return left == right


def audit_models(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = [(item["scenario"], int(item["seed"])) for item in manifest["models"]]
    if len(keys) != EXPECTED_MODELS or set(keys) != expected_model_keys() or len(keys) != len(set(keys)):
        raise CrossRegimeAuditError("model matrix is incomplete or duplicated")
    records = []
    for item in manifest["models"]:
        scenario = str(item["scenario"]); seed = int(item["seed"]); path = PROJECT_ROOT / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise CrossRegimeAuditError(f"model hash mismatch: {path}")
        model, scalers, metadata = load_model_bundle(path)
        if model.config != model_config(seed) or metadata["scenario"] != scenario:
            raise CrossRegimeAuditError(f"model identity mismatch: {path}")
        if metadata.get("protocol_manifest_sha256") != file_sha256(MANIFEST_PATH):
            raise CrossRegimeAuditError("model protocol lock mismatch")
        expected_scalers, sequences, provenance = prepare_training(scenario, seed)
        if not close(metadata.get("training"), provenance) or not close(item.get("training"), provenance):
            raise CrossRegimeAuditError("model training provenance mismatch")
        if provenance["effective_samples"] != EFFECTIVE_TRAINING_BUDGET:
            raise CrossRegimeAuditError("training budget mismatch")
        scaler_pairs = (
            (scalers.state.mean, expected_scalers.state.mean),
            (scalers.state.scale, expected_scalers.state.scale),
            (scalers.current.mean, expected_scalers.current.mean),
            (scalers.current.scale, expected_scalers.current.scale),
        )
        if not all(np.array_equal(left, right) for left, right in scaler_pairs):
            raise CrossRegimeAuditError(f"scaler prefix-scope mismatch: {path}")
        if sum(len(sequence.inputs) - 2_000 for sequence in sequences) != 130_000:
            raise CrossRegimeAuditError("effective readout count mismatch")
        records.append({"scenario": scenario, "seed": seed, "path": item["path"], "sha256": item["sha256"], "configuration_match": True, "training_budget": 130_000, "scaler_prefix_recomputation_exact": True})
    return {"model_count": len(records), "records": records, "exact": True}


def audit_datasets(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if len(manifest["datasets"]) != EXPECTED_SCHEDULE_DATASETS:
        raise CrossRegimeAuditError("schedule dataset count mismatch")
    names = []
    for item in manifest["datasets"]:
        path = PROJECT_ROOT / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise CrossRegimeAuditError(f"schedule hash mismatch: {path}")
        if tuple(item["sequence"]) != CONTINUOUS_SCHEDULES[item["name"]]:
            raise CrossRegimeAuditError("schedule sequence mismatch")
        if tuple(item["switch_indices"]) != CONTINUOUS_SWITCH_INDICES or item["state_count"] != 500_000:
            raise CrossRegimeAuditError("schedule shape/switch mismatch")
        with np.load(path, allow_pickle=False) as saved:
            if tuple(saved.files) != ("t", "x", "y", "z", "I") or any(saved[key].shape != (500_000,) for key in saved.files):
                raise CrossRegimeAuditError(f"schedule array schema mismatch: {path}")
            detected = tuple((np.flatnonzero(np.diff(saved["I"]) != 0) + 1).tolist())
            if detected != CONTINUOUS_SWITCH_INDICES:
                raise CrossRegimeAuditError(f"schedule current boundary mismatch: {path}")
        names.append(item["name"])
    if set(names) != set(CONTINUOUS_SCHEDULES) or len(names) != len(set(names)):
        raise CrossRegimeAuditError("schedule names are incomplete or duplicated")
    return {"dataset_count": len(names), "exact": True, "names": names}


def audit_records(raw: Mapping[str, Any], model_manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = raw["records"]
    validate_record_matrix(records)
    scalers = {}
    for item in model_manifest["models"]:
        _, scaler, _ = load_model_bundle(PROJECT_ROOT / item["path"])
        scalers[(item["scenario"], int(item["seed"]))] = scaler
    fixed = {current: load_fixed_trajectory(current) for current in (1.67, 3.20, 3.29, 3.34, 3.50)}
    schedules = load_schedules(strict_load_json(DATASET_MANIFEST_PATH))
    recomputed = 0
    failures = 0
    divergences = 0
    collapses = 0
    for item in records:
        arrays = validate_raw_array(item)
        expected_id = record_id(item["family"], item["scenario"], int(item["seed"]),
                                current=item["current"], window=item["window"], schedule=item["schedule"])
        if expected_id != item["record_id"]:
            raise CrossRegimeAuditError("record identity mismatch")
        if item["family"] == "fixed_short":
            start = (70_000, 80_000, 89_999)[item["window"] - 1]
            warmup, forecast = [start, start + 2_000], [start + 2_000, start + 10_000]
        elif item["family"] == "fixed_long":
            warmup, forecast = [70_000, 72_000], [72_000, 99_999]
        else:
            warmup, forecast = [0, 2_000], [2_000, 499_999]
        if item["warmup_range"] != warmup or item["forecast_range"] != forecast:
            raise CrossRegimeAuditError("evaluation window mismatch")
        trajectory = schedules[item["schedule"]] if item["family"] == "continuous" else fixed[item["current"]]
        begin, stop = forecast
        expected_arrays = {"targets": trajectory.states[begin + 1:stop + 1],
                           "time": trajectory.time[begin + 1:stop + 1],
                           "current": trajectory.current_values[begin:stop]}
        if not all(np.array_equal(arrays[key], value) for key, value in expected_arrays.items()):
            raise CrossRegimeAuditError("raw targets/time/current differ from source trajectory")
        expected_class = ("continuous mixed-regime transition schedule" if item["family"] == "continuous"
                          else evaluation_classification(item["scenario"], item["current"]))
        if item["evaluation_class"] != expected_class:
            raise CrossRegimeAuditError("evaluation classification mismatch")
        scaler = scalers[(item["scenario"], int(item["seed"]))]
        metrics = evaluate_rollout(
            arrays["predictions"], arrays["targets"],
            normalisation_scale=scaler.state.scale, dt=0.01,
            valid_prediction_threshold=0.4, divergence_threshold=5.0,
            collapse_std_ratio_threshold=0.05,
        ).to_dict()
        if not close(metrics, item["metrics"]):
            raise CrossRegimeAuditError(f"metric recomputation mismatch: {item['record_id']}")
        pointwise = pointwise_normalised_error(arrays["predictions"], arrays["targets"], normalisation_scale=scaler.state.scale)
        if not np.array_equal(pointwise, arrays["pointwise_normalised_error"], equal_nan=True):
            raise CrossRegimeAuditError(f"pointwise-error mismatch: {item['record_id']}")
        events = invalidate_divergent_event_metrics(event_metrics(arrays["predictions"], arrays["targets"], 0.01), metrics)
        if not close(events, item["event_metrics"]):
            raise CrossRegimeAuditError(f"event recomputation mismatch: {item['record_id']}")
        penalty = metrics["nrmse_state"] if metrics["nrmse_state"] is not None else NONFINITE_FAILURE_SCORE
        if not close(penalty, item["aggregate_nrmse_value"]):
            raise CrossRegimeAuditError("failure penalty mismatch")
        bad = np.flatnonzero(~np.all(np.isfinite(arrays["predictions"]), axis=1))
        failure_step = int(bad[0]) if len(bad) else None
        if item["failure_step"] != failure_step or item["numerical_failure"] != bool(len(bad)):
            raise CrossRegimeAuditError("numerical failure classification mismatch")
        if item["valid_prefix_steps"] != (failure_step if failure_step is not None else len(arrays["predictions"])):
            raise CrossRegimeAuditError("valid prefix length mismatch")
        if item["family"] == "continuous":
            boundaries = transition_boundary_metrics(arrays["predictions"], arrays["targets"], arrays["current"], scaler.state.scale, forecast_start=begin)
            if not close(boundaries, item.get("transition_boundaries")):
                raise CrossRegimeAuditError("transition boundary recomputation mismatch")
            if len(item.get("transition_boundaries", [])) != 4:
                raise CrossRegimeAuditError(f"continuous boundary metrics missing: {item['record_id']}")
            if item["warmup_range"] != [0, 2_000] or item["forecast_range"] != [2_000, 499_999]:
                raise CrossRegimeAuditError(f"continuous reset/warm-up contract mismatch: {item['record_id']}")
        failures += bool(item["numerical_failure"])
        divergences += bool(metrics["diverged"])
        collapses += bool(metrics["prediction_collapse_any"])
        recomputed += 1
    return {"record_count": recomputed, "expected_record_count": EXPECTED_RECORDS, "exact_matrix": True, "metric_recomputation_exact": True, "event_recomputation_exact": True, "failure_count": failures, "divergence_count": divergences, "collapse_count": collapses, "all_failures_retained": True}


def load_arrays(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return validate_raw_array(record)


def generate_figures(records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    artifacts=[]
    def save(fig: Any, stem: str) -> None:
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path=FIGURE_ROOT/f"{stem}.{suffix}"; fig.savefig(path,dpi=300,bbox_inches="tight"); artifacts.append({"path":path.relative_to(PROJECT_ROOT).as_posix(),"sha256":file_sha256(path)})
        plt.close(fig)
    def representative(scenario: str, current: float, stem: str) -> None:
        item=next(x for x in records if x["family"]=="fixed_short" and x["scenario"]==scenario and x["seed"]==REPRESENTATIVE_SEED and x["current"]==current and x["window"]==1)
        a=load_arrays(item); fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
        for index,name in enumerate(("x","y","z")):
            axes[index].plot(a["time"],a["targets"][:,index],color="black",lw=1,label="truth"); axes[index].plot(a["time"],a["predictions"][:,index],color="tab:blue",lw=.9,label="prediction"); axes[index].axvline(a["time"][0],color="grey",ls="--",label="warm-up boundary" if index==0 else None); axes[index].set_ylabel(name)
        axes[0].legend(); axes[-1].set_xlabel("model time"); fig.suptitle(f"{scenario} | I={current:.2f} | seed=42 | forecast {item['forecast_range']}"); save(fig,stem)
    representative("regular_to_chaotic",3.20,"01_regular_trained_on_chaotic")
    representative("chaotic_to_regular",1.67,"02_chaotic_trained_on_regular")
    representative("mixed_shuffled",3.20,"03_mixed_shuffled_representative")

    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=False)
    for axis,(schedule,_) in zip(axes,CONTINUOUS_SCHEDULES.items()):
        item=next(x for x in records if x["family"]=="continuous" and x["scenario"]=="mixed_shuffled" and x["seed"]==42 and x["schedule"]==schedule); a=load_arrays(item); stride=max(1,len(a["time"])//12000); axis.plot(a["time"][::stride],a["targets"][::stride,0],color="black",lw=.7,label="truth x"); axis.plot(a["time"][::stride],a["predictions"][::stride,0],color="tab:blue",lw=.7,label="prediction x"); [axis.axvline(boundary*.01,color="grey",ls="--",lw=.6) for boundary in CONTINUOUS_SWITCH_INDICES]; axis.set_title(f"mixed_shuffled | {schedule} | seed=42 | [2000,499999)"); axis.set_ylabel("x")
    axes[0].legend(); axes[-1].set_xlabel("model time"); save(fig,"04_all_continuous_schedules")

    item=next(x for x in records if x["family"]=="continuous" and x["scenario"]=="mixed_shuffled" and x["seed"]==42 and x["schedule"]=="alternating_mixed"); a=load_arrays(item); fig,axes=plt.subplots(4,1,figsize=(11,9),sharex=True)
    for axis,boundary in zip(axes,CONTINUOUS_SWITCH_INDICES):
        local=boundary-2000; start=local-1000; stop=local+1001; relative=(np.arange(start,stop)-local)*.01; axis.plot(relative,a["targets"][start:stop,0],color="black",label="truth x"); axis.plot(relative,a["predictions"][start:stop,0],color="tab:blue",label="prediction x"); axis.axvline(0,color="red",ls="--"); axis.set_ylabel("x"); axis.set_title(f"switch {boundary}: {a['current'][local-1]:.2f}→{a['current'][local]:.2f}")
    axes[0].legend(); axes[-1].set_xlabel("time relative to current switch"); save(fig,"05_transition_boundary_tracking")

    scenarios=list(SCENARIO_TRAINING_CURRENTS); colors=["tab:blue","tab:orange","tab:green"]
    fig,ax=plt.subplots(figsize=(9,5))
    for scenario,color in zip(scenarios,colors):
        values=[statistics.fmean(float(x["aggregate_nrmse_value"]) for x in records if x["scenario"]==scenario and x["seed"]==seed) for seed in SEEDS]; ax.plot(SEEDS,values,marker="o",label=scenario,color=color)
    ax.set_yscale("symlog"); ax.set_xlabel("reservoir seed"); ax.set_ylabel("mean failure-penalized NRMSE"); ax.legend(); save(fig,"06_per_seed_nrmse")
    fig,ax=plt.subplots(figsize=(9,5))
    for scenario,color in zip(scenarios,colors):
        values=[statistics.fmean(float(x["metrics"]["valid_prediction_time"]) for x in records if x["scenario"]==scenario and x["seed"]==seed) for seed in SEEDS]; ax.plot(SEEDS,values,marker="o",label=scenario,color=color)
    ax.set_xlabel("reservoir seed"); ax.set_ylabel("mean valid prediction time"); ax.legend(); save(fig,"07_per_seed_vpt")
    fig,axes=plt.subplots(1,2,figsize=(10,4)); divergence=[sum(x["metrics"]["diverged"] for x in records if x["scenario"]==s)/115 for s in scenarios]; collapse=[sum(x["metrics"]["prediction_collapse_any"] for x in records if x["scenario"]==s)/115 for s in scenarios]; axes[0].bar(scenarios,divergence,color=colors); axes[1].bar(scenarios,collapse,color=colors); axes[0].set_title("Divergence rate"); axes[1].set_title("Collapse rate"); [axis.tick_params(axis="x",rotation=25) for axis in axes]; save(fig,"08_divergence_collapse_rates")
    fig,ax=plt.subplots(figsize=(9,5)); means=[statistics.fmean(float(x["aggregate_nrmse_value"]) for x in records if x["scenario"]==s) for s in scenarios]; ax.bar(scenarios,means,color=colors); ax.set_yscale("symlog"); ax.set_ylabel("failure-penalized mean NRMSE"); ax.tick_params(axis="x",rotation=20); save(fig,"09_aggregate_scenario_comparison")
    labels=[]; values=[]
    for scenario in scenarios:
        for kind in sorted({x["evaluation_class"] for x in records if x["scenario"]==scenario and x["family"]!="continuous"}):
            labels.append(f"{scenario}\n{kind}"); values.append(statistics.fmean(float(x["aggregate_nrmse_value"]) for x in records if x["scenario"]==scenario and x["evaluation_class"]==kind))
    fig,ax=plt.subplots(figsize=(12,5)); ax.bar(range(len(values)),values); ax.set_xticks(range(len(values)),labels,rotation=35,ha="right"); ax.set_yscale("symlog"); ax.set_ylabel("failure-penalized mean NRMSE"); save(fig,"10_within_vs_cross_regime")
    return artifacts


def write_verified_results(aggregate: Mapping[str, Any], record_audit: Mapping[str, Any]) -> None:
    lines=["# Chapter 2 cross-regime results","","All statements below refer to the independently audited 345-record experiment.","","## Method","","Fifteen parameter-aware ESNs used the frozen Step 7 architecture, five paired reservoir seeds, independent per-block reservoir resets and 2,000-step washouts, scenario-only scalers, and exactly 130,000 effective readout samples per scenario.","","## Verified scenario summary","","| Scenario | Finite mean NRMSE | Penalized mean NRMSE | Mean VPT | Divergence | Collapse |","|---|---:|---:|---:|---:|---:|"]
    for scenario,data in aggregate["scenarios"].items():
        item=data["overall"]; finite="undefined" if item["finite_nrmse_mean"] is None else f"{item['finite_nrmse_mean']:.6g}"; lines.append(f"| `{scenario}` | {finite} | {item['failure_penalized_nrmse_mean']:.6g} | {item['mean_valid_prediction_time']:.6g} | {item['divergence_count']}/{item['record_count']} | {item['collapse_count']}/{item['record_count']} |")
    lines += ["", "## Transfer versus mixed training on matching targets", "",
              "Physical state RMSE is shown below to avoid scenario-specific NRMSE scaling. Means include finite values only; divergence counts include every record.", "",
              "| Training | Family | Target I | Transfer RMSE | Mixed RMSE | Transfer divergence | Mixed divergence |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for comparison in aggregate["transfer_target_comparisons"]:
        left, right = comparison["transfer"], comparison["mixed_reference"]
        def fmt(value):
            return "undefined" if value is None else f"{value:.6g}"
        lines.append(f"| {comparison['training_scenario']} | {comparison['family']} | {comparison['target_current']:.2f} | {fmt(left['finite_rmse_mean'])} | {fmt(right['finite_rmse_mean'])} | {left['divergence_count']}/{left['record_count']} | {right['divergence_count']}/{right['record_count']} |")
    cross=[]
    for scenario in ("regular_to_chaotic","chaotic_to_regular"):
        item=aggregate["scenarios"][scenario]["by_evaluation_class"]["cross-regime generalization"]
        cross.append(f"`{scenario}` cross-regime records have finite mean NRMSE {item['finite_nrmse_mean']} and divergence {item['divergence_count']}/{item['record_count']}.")
    lines += ["","## Scientific interpretation","",*cross,"",f"Across all records the audit retained {record_audit['divergence_count']} divergences, {record_audit['collapse_count']} collapses, and {record_audit['failure_count']} numerical failures. Mixed-training performance and directional asymmetry are reported descriptively; five seeds do not justify a standalone significance claim.","","The evidence addresses cross-regime transfer only for this frozen architecture, these five currents, seeds, horizons, and schedules. Generalization is supported only to the extent that the state error, valid-prediction time, event, divergence, collapse, and transition results above agree.","","## Limitations","","NRMSE, valid-prediction time and divergence use each scenario's training-only state scale. Cross-scenario normalized differences therefore combine prediction error and normalization differences; consult physical RMSE in the CSV tables. Pooled summaries mix unequal horizons and seen/unseen currents and are descriptive, not the primary transfer comparison.","","The frozen hyperparameters were previously selected using mixed-regime validation currents. This experiment isolates the effect of the final training data and fitted readout under a fixed, previously selected ESN architecture. It does not demonstrate that every design decision was learned exclusively from one regime.","","`I=3.29` is prespecified as regular/non-chaotic using its converged near-zero LLE and consistent half-window measurements, but its original qualitative regime label remains uncertain."]
    atomic_write_text(RESULTS_DOCUMENT,"\n".join(lines)+"\n")


def run_audit() -> dict[str, Any]:
    required=(MANIFEST_PATH,STATUS_PATH,RAW_RESULTS_PATH,AGGREGATE_PATH,MODEL_MANIFEST_PATH,DATASET_MANIFEST_PATH)
    if any(not path.is_file() for path in required): raise CrossRegimeAuditError("required experiment artifact is missing")
    status=strict_load_json(STATUS_PATH)
    if status.get("state") != "BENCHMARK_COMPLETE_AUDIT_REQUIRED": raise CrossRegimeAuditError("benchmark status is not ready for audit")
    manifest=strict_load_json(MANIFEST_PATH); models=strict_load_json(MODEL_MANIFEST_PATH); datasets=strict_load_json(DATASET_MANIFEST_PATH); raw=strict_load_json(RAW_RESULTS_PATH); saved_aggregate=strict_load_json(AGGREGATE_PATH)
    if manifest["source_hashes"] != source_hashes(): raise CrossRegimeAuditError("executed source differs from lock")
    dataset_audit=audit_datasets(datasets); model_audit=audit_models(models); record_audit=audit_records(raw,models)
    expected_locks = {"protocol_manifest": file_sha256(MANIFEST_PATH), "model_manifest": file_sha256(MODEL_MANIFEST_PATH), "dataset_manifest": file_sha256(DATASET_MANIFEST_PATH)}
    if raw.get("lock_hashes") != expected_locks:
        raise CrossRegimeAuditError("raw-result locks mismatch")
    if status.get("aggregate_sha256") != file_sha256(AGGREGATE_PATH):
        raise CrossRegimeAuditError("aggregate status hash mismatch")
    recomputed=aggregate_results(raw, write=False)
    if not close(recomputed["scenarios"],saved_aggregate["scenarios"]): raise CrossRegimeAuditError("aggregate recomputation mismatch")
    if not close(recomputed["transfer_target_comparisons"], saved_aggregate["transfer_target_comparisons"]): raise CrossRegimeAuditError("transfer target comparison mismatch")
    if not close(recomputed["paired_comparisons"], saved_aggregate["paired_comparisons"]): raise CrossRegimeAuditError("paired comparison recomputation mismatch")
    if not protected_paths_unchanged(): raise CrossRegimeAuditError("protected tracked path changed")
    original=original_binary_inventory()
    if not original["valid"]: raise CrossRegimeAuditError("original 226 binaries changed")
    figures=generate_figures(raw["records"]); write_verified_results(recomputed,record_audit)
    large=[*(PROJECT_ROOT/item["path"] for item in models["models"]),*(PROJECT_ROOT/item["path"] for item in datasets["datasets"]),*(Path(item["raw_arrays_path"]) for item in raw["records"])]
    if len(large)!=EXPECTED_BINARY_ARTIFACTS or len(set(path.resolve() for path in large))!=EXPECTED_BINARY_ARTIFACTS: raise CrossRegimeAuditError("large binary artifact count mismatch")
    small=[path for path in RESULT_ROOT.rglob("*") if path.is_file() and path.suffix != ".npz" and path not in (STATUS_PATH, HASH_PATH, VERIFICATION_PATH) and not path.name.startswith(".")]+[RESULTS_DOCUMENT, PROTOCOL_DOCUMENT]
    hashes={"large_binary_artifacts":file_hash_inventory(large),"tables_figures_manifests":file_hash_inventory(small)}; atomic_write_json(HASH_PATH,hashes)
    json_paths=[path for path in RESULT_ROOT.rglob("*.json") if path != VERIFICATION_PATH]
    for path in json_paths: strict_load_json(path)
    verification={"schema":"chapter2_cross_regime_verification_v1","verified_at":utc_now(),"verdict":"AUDIT PASSED","record_audit":record_audit,"model_audit":model_audit,"dataset_audit":dataset_audit,"aggregate_recomputation_match":True,"source_and_manifest_match":True,"protected_paths_match_base_commit":BASE_COMMIT,"original_226_scientific_binaries":original,"large_binary_artifact_count":len(large),"figure_artifacts":figures,"artifact_hashes_path":HASH_PATH.relative_to(PROJECT_ROOT).as_posix(),"strict_json_count":len(json_paths),"representative_seed":42}
    atomic_write_json(VERIFICATION_PATH,verification)
    status.update({"state":"COMPLETE","audit_completed_at":utc_now(),"verification_sha256":file_sha256(VERIFICATION_PATH),"artifact_hashes_sha256":file_sha256(HASH_PATH)}); atomic_write_json(STATUS_PATH,status)
    return verification


def main(argv: Sequence[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.parse_args(argv); result=run_audit(); print(result["verdict"]); return 0


if __name__ == "__main__": raise SystemExit(main())
