"""Fail-closed audit of the Part 2 rebuild's design, data, and optimisation state.

Every check defaults to FAIL; a check only reports "ok" if it can positively
verify the condition. Missing evidence is a failure, not a skip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import subprocess

from . import config, data, design, optimisation, validation

PROTECTED_PATHS = (
    "chapter2/hr_data_ch2.py",
    "chapter2/config_ch2.py",
    "chapter2/esn_model.py",
    "chapter2/esn_data.py",
    "chapter2/esn_metrics.py",
    "chapter2/esn_optimisation.py",
    "chapter2/esn_config.py",
    "chapter2/esn_step8.py",
    "chapter2/run_step8.py",
    "chapter2/audit_step8.py",
    "chapter2/correct_step8_event_validity.py",
    "chapter2/dynamics_analysis_ch2.py",
    "chapter2/generate_diagnostics.py",
    "chapter2/run_esn_pilot.py",
    "chapter2/plot_thesis_figures.py",
    "chapter2/verify_release.py",
    "chapter2/pilot_results",
    "chapter2/release",
    "chapter2/optimisation_results",
    "chapter2/final_models",
    "chapter2/final_results",
    "chapter2/outputs",
    "chapter2/provenance",
    "FINAL_THESIS_RUN",
)

STARTING_COMMIT = "6d40954fe6dd9df377b098c14f0c662226c3917a"


@dataclass
class AuditResult:
    checks: list[dict] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    @property
    def all_passed(self) -> bool:
        return all(c["passed"] for c in self.checks)


def check_part1_and_chapter1_unchanged(result: AuditResult) -> None:
    proc = subprocess.run(
        ["git", "diff", "--quiet", STARTING_COMMIT, "--", *PROTECTED_PATHS],
        cwd=config.PROJECT_ROOT,
        check=False,
    )
    result.add("part1_and_chapter1_unchanged", proc.returncode == 0, f"git diff returncode={proc.returncode}")


def check_design_hash(result: AuditResult) -> None:
    try:
        raw = config.FROZEN_DESIGN_PATH.read_text()
        d = json.loads(raw)
        stored = d.pop("design_hash_sha256")
        recomputed = hashlib.sha256(
            json.dumps(d, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()
        result.add("design_hash_matches", stored == recomputed, f"stored={stored} recomputed={recomputed}")
    except Exception as exc:  # fail-closed
        result.add("design_hash_matches", False, f"exception: {exc}")


def check_no_train_test_overlap(result: AuditResult) -> None:
    try:
        d = json.loads(config.FROZEN_DESIGN_PATH.read_text())
        reg_train = {c["current"] for c in d["models"]["regular_trained"]["training_currents"]}
        cha_train = {c["current"] for c in d["models"]["chaotic_trained"]["training_currents"]}
        rr = {c["current"] for c in d["models"]["regular_trained"]["rr_test_currents"]}
        rc = {c["current"] for c in d["models"]["regular_trained"]["rc_test_currents"]}
        cc = {c["current"] for c in d["models"]["chaotic_trained"]["cc_test_currents"]}
        cr = {c["current"] for c in d["models"]["chaotic_trained"]["cr_test_currents"]}
        overlap = (rr & reg_train) | (rc & reg_train) | (cc & cha_train) | (cr & cha_train)
        result.add("no_train_test_overlap", not overlap, f"overlap={overlap}")
    except Exception as exc:
        result.add("no_train_test_overlap", False, f"exception: {exc}")


def check_final_test_firewall(result: AuditResult) -> None:
    try:
        manifest = json.loads(config.DATA_MANIFEST_PATH.read_text())
        final_rows = [r for r in manifest["trajectories"] if r["role"] == "final_test"]
        if not final_rows:
            result.add("final_test_firewall", False, "no final_test rows in manifest")
            return
        blocked = False
        try:
            data.load_source_trajectory("final_test", final_rows[0]["current"])
        except data.FinalTestAccessError:
            blocked = True
        result.add("final_test_firewall", blocked, "plain load correctly refused" if blocked else "plain load NOT refused")
    except Exception as exc:
        result.add("final_test_firewall", False, f"exception: {exc}")


def check_no_final_test_result_exists(result: AuditResult) -> None:
    forbidden_names = ("rr_result", "rc_result", "cc_result", "cr_result", "final_evaluation")
    hits = []
    if config.RESULTS_DIR.exists():
        for path in config.RESULTS_DIR.rglob("*"):
            if any(name in path.name.lower() for name in forbidden_names):
                hits.append(str(path))
    result.add("no_final_test_result_exists", not hits, f"hits={hits}")


def check_trajectory_hashes(result: AuditResult) -> None:
    try:
        manifest = json.loads(config.DATA_MANIFEST_PATH.read_text())
        mismatches = []
        for row in manifest["trajectories"]:
            path = config.PROJECT_ROOT / row["path"]
            if not path.exists() or data.file_sha256(path) != row["sha256"]:
                mismatches.append(row["path"])
        result.add("trajectory_hashes_match", not mismatches, f"mismatches={mismatches}")
    except Exception as exc:
        result.add("trajectory_hashes_match", False, f"exception: {exc}")


def check_search_space_freeze(result: AuditResult) -> None:
    try:
        dims = config.search_dimensions()
        expected_names = list(config.SEARCH_PARAMETER_ORDER)
        actual_names = [d.name for d in dims]
        result.add("search_space_matches_part1", actual_names == expected_names, f"{actual_names}")
    except Exception as exc:
        result.add("search_space_matches_part1", False, f"exception: {exc}")


def check_optimisation_locks(result: AuditResult) -> None:
    for family in ("regular", "chaotic"):
        sel_path = config.OPTIMISATION_DIR / f"{family}_selection.json"
        if not sel_path.exists():
            result.add(f"{family}_selection_exists", False, "missing")
            continue
        sel = json.loads(sel_path.read_text())
        result.add(f"{family}_selection_exists", True, "")
        candidate_ok = "locked_hyperparameters" in sel and "design_hash" in sel
        result.add(f"{family}_selection_well_formed", candidate_ok, str(list(sel.keys())))
        design_d = json.loads(config.FROZEN_DESIGN_PATH.read_text())
        result.add(
            f"{family}_selection_design_hash_matches",
            sel.get("design_hash") == design_d.get("design_hash_sha256"),
            f"{sel.get('design_hash')} vs {design_d.get('design_hash_sha256')}",
        )


PRE_TASK_BASELINE_PATH = config.PACKAGE_ROOT / "results" / "chaotic_diagnostic" / "pre_task_baseline_hashes.json"


def check_regular_and_frozen_artifacts_unchanged(result: AuditResult) -> None:
    if not PRE_TASK_BASELINE_PATH.exists():
        result.add("regular_and_frozen_artifacts_unchanged", False, "no pre-task baseline hash file recorded")
        return
    baseline = json.loads(PRE_TASK_BASELINE_PATH.read_text())
    mismatches = []
    for rel_path, expected in baseline.items():
        path = config.PROJECT_ROOT / rel_path
        actual = data.file_sha256(path) if path.exists() else None
        if actual != expected:
            mismatches.append({"path": rel_path, "expected": expected, "actual": actual})
    result.add("regular_and_frozen_artifacts_unchanged", not mismatches, f"mismatches={mismatches}")


def check_no_rr_rc_cc_cr_artifact(result: AuditResult) -> None:
    forbidden = ("rr_result", "rc_result", "cc_result", "cr_result", "final_evaluation")
    hits = []
    for base in (config.RESULTS_DIR,):
        if base.exists():
            for path in base.rglob("*"):
                if any(name in path.name.lower() for name in forbidden):
                    hits.append(str(path))
    result.add("no_rr_rc_cc_cr_artifact", not hits, f"hits={hits}")


def check_chaotic_recovery_search_protocol(result: AuditResult) -> None:
    research_dir = config.PACKAGE_ROOT / "results" / "chaotic_research"
    protocol_path = research_dir / "search_protocol.json"
    if not protocol_path.exists():
        result.add("chaotic_recovery_protocol_exists", False, "missing (no recovery search run yet)")
        return
    protocol = json.loads(protocol_path.read_text())
    result.add("chaotic_recovery_protocol_exists", True, "")
    result.add(
        "chaotic_recovery_search_seeds_registered",
        protocol.get("search_seeds") == [42, 123, 456],
        str(protocol.get("search_seeds")),
    )
    result.add(
        "chaotic_recovery_confirmation_seeds_registered",
        sorted(protocol.get("all_confirmation_seeds", [])) == [42, 123, 456, 789, 2026],
        str(protocol.get("all_confirmation_seeds")),
    )
    comp = protocol.get("candidate_composition", {})
    result.add("chaotic_recovery_candidate_count_is_96", comp.get("total") == 96, str(comp))

    history_path = research_dir / "search_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        keys = [json.dumps(r["hyperparameters"], sort_keys=True) for r in history["history"]]
        result.add("chaotic_recovery_candidates_unique", len(keys) == len(set(keys)), f"n={len(keys)} unique={len(set(keys))}")
        result.add("chaotic_recovery_candidate_count_matches", len(history["history"]) == 96, f"n={len(history['history'])}")

        for record in history["history"]:
            expected_survives = (
                record["rank_key"][0] == 0 and record["rank_key"][1] == 0 and record["rank_key"][2] == 0
            )
            if record["survives_hard_filter"] != expected_survives:
                result.add("chaotic_recovery_hard_filter_recomputes", False, f"mismatch on {record['hyperparameters']}")
                break
        else:
            result.add("chaotic_recovery_hard_filter_recomputes", True, "")

    confirm_path = research_dir / "confirmation_results.json"
    loco_path = research_dir / "loco_results.json"
    if loco_path.exists() and not confirm_path.exists():
        result.add("chaotic_recovery_loco_only_after_confirmation", False, "loco exists without confirmation")
    else:
        result.add("chaotic_recovery_loco_only_after_confirmation", True, "")

    selection_path = research_dir / "chaotic_selection.json"
    if selection_path.exists() and loco_path.exists():
        selection = json.loads(selection_path.read_text())
        loco = json.loads(loco_path.read_text())
        if selection.get("kind") == "selection":
            best = min(
                loco["results"],
                key=lambda r: r["loco_rank_key"] + [json.dumps(r["hyperparameters"], sort_keys=True)],
            )
            result.add(
                "chaotic_recovery_winner_matches_loco_ranking",
                json.dumps(best["hyperparameters"], sort_keys=True)
                == json.dumps(selection["locked_hyperparameters"], sort_keys=True),
                "",
            )
            lock_path = research_dir / "chaotic_lock.json"
            if lock_path.exists():
                lock = json.loads(lock_path.read_text())
                recomputed = hashlib.sha256(
                    json.dumps(selection, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
                ).hexdigest()
                result.add("chaotic_recovery_lock_hash_matches", lock.get("selection_sha256") == recomputed, "")


def run_audit() -> AuditResult:
    result = AuditResult()
    check_part1_and_chapter1_unchanged(result)
    check_design_hash(result)
    check_no_train_test_overlap(result)
    check_final_test_firewall(result)
    check_no_final_test_result_exists(result)
    check_no_rr_rc_cc_cr_artifact(result)
    check_regular_and_frozen_artifacts_unchanged(result)
    check_chaotic_recovery_search_protocol(result)
    check_trajectory_hashes(result)
    check_search_space_freeze(result)
    check_optimisation_locks(result)
    return result


def main() -> int:
    result = run_audit()
    for check in result.checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['name']}: {check['detail']}")
    print(f"AUDIT {'PASSED' if result.all_passed else 'FAILED'}")
    out_path = config.RESULTS_DIR / "audit_report.json"
    out_path.write_text(
        json.dumps({"all_passed": result.all_passed, "checks": result.checks}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
