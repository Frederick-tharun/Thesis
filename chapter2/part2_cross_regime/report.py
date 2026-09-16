"""Step 4J: source-only validation report and figures for the locked winners.

Reads only already-computed search/confirmation/LOCO JSON (no final-test
access). Produces at most two PNGs, no RR/RC/CC/CR figures.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, validation


def _load(path):
    return json.loads(path.read_text())


def build_family_report(family: str) -> dict:
    selection = _load(config.OPTIMISATION_DIR / f"{family}_selection.json")
    confirmation = _load(config.OPTIMISATION_DIR / f"{family}_confirmation.json")
    loco = _load(config.OPTIMISATION_DIR / f"{family}_loco.json")
    history = _load(config.OPTIMISATION_DIR / f"{family}_history.json")

    from .optimisation import _dedup_key

    winner_hp = selection["locked_hyperparameters"]
    winner_history_record = next(
        r for r in history["history"] if _dedup_key(r["hyperparameters"]) == _dedup_key(winner_hp)
    )
    winner_confirmation = next(
        c for c in confirmation["confirmations"] if _dedup_key(c["hyperparameters"]) == _dedup_key(winner_hp)
    )
    winner_loco = next(
        l for l in loco["results"] if _dedup_key(l["hyperparameters"]) == _dedup_key(winner_hp)
    )

    per_current_search = winner_history_record["aggregate"]["per_current_mean_nrmse"]

    return {
        "family": family,
        "training_currents": selection["training_currents"],
        "locked_hyperparameters": winner_hp,
        "search_seed_objective": winner_history_record["objective"],
        "search_seed_aggregate": winner_history_record["aggregate"],
        "per_current_search_nrmse": per_current_search,
        "confirmation_per_seed": winner_confirmation["per_seed"],
        "confirmation_gate_pass_count": winner_confirmation["gate_pass_count"],
        "loco_summary": selection["loco_summary"],
        "loco_n_folds": winner_loco["n_folds"],
        "loco_n_seeds": winner_loco["n_seeds"],
    }


def _plot_family(report: dict, output_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    currents = sorted(float(k) for k in report["per_current_search_nrmse"])
    nrmse = [report["per_current_search_nrmse"][f"{c:.2f}"] for c in currents]
    axes[0].bar([f"{c:.3g}" for c in currents], nrmse, color="#0072B2")
    axes[0].set_ylabel("mean NRMSE (search seed, 3 windows)")
    axes[0].set_title(f"{report['family']}-trained: per-current source NRMSE")
    axes[0].tick_params(axis="x", rotation=45)

    seeds = sorted(int(s) for s in report["confirmation_per_seed"])
    vpt_fracs = [
        report["confirmation_per_seed"][str(s)]["aggregate"]["min_vpt_fraction"] for s in seeds
    ]
    axes[1].bar([str(s) for s in seeds], vpt_fracs, color="#D55E00")
    axes[1].axhline(
        validation.SOURCE_QUALITY_GATE["min_vpt_fraction_per_rollout"],
        color="#222222", linestyle="--", linewidth=1, label="gate threshold",
    )
    axes[1].set_ylabel("min VPT fraction across rollouts")
    axes[1].set_title(f"{report['family']}-trained: seed robustness")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    fig.suptitle(
        f"Part 2 {report['family']}-trained source-only validation "
        f"(LOCO median NRMSE={report['loco_summary']['median_nrmse']:.4g})"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run() -> None:
    config.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    combined = {}
    for family in ("regular", "chaotic"):
        selection = _load(config.OPTIMISATION_DIR / f"{family}_selection.json")
        if selection.get("kind") != "selection":
            combined[family] = {
                "kind": "not_locked",
                "reason": selection.get("reason", "no winner locked"),
            }
            print(f"[{family}] no winner locked ({selection.get('kind')}); skipping report/figure")
            continue
        report = build_family_report(family)
        combined[family] = report
        _plot_family(report, config.VALIDATION_DIR / f"{family}_source_validation.png")
        print(f"wrote {config.VALIDATION_DIR / f'{family}_source_validation.png'}")

    out_path = config.VALIDATION_DIR / "source_validation_report.json"
    out_path.write_text(json.dumps(combined, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    run()
