"""Long-horizon RR/RC/CC/CR benchmark, matching Chapter 2 Part 1's own window.

Three corrections over the first (80-time-unit) pass:

1. HORIZON. Part 1's Step-8 long-horizon benchmark used warm-up
   [70000,72000) and autonomous forecast [72000,99999) -- 27,999 steps =
   ~280 time units. That is reused verbatim here, so these numbers are
   directly comparable to Part 1's published `unseen_long` family and show
   many oscillations (at I=1.67 the inter-burst interval is ~146 time units,
   so an 80-unit window cannot even contain one full burst cycle -- exactly
   what chapter2/professor_long_window_diagnostic concluded).

2. FAIRNESS. The first pass compared a single-seed regular model against a
   5-seed median-ensemble chaotic model. Both families now use the identical
   5-seed median ensemble, so RR/RC and CC/CR differ only in training regime.

3. METRICS. Over ~280 time units a positive-Lyapunov system cannot be tracked
   pointwise -- that is mathematically impossible, not a model defect. So
   alongside pointwise NRMSE/VPT ("weather"), this reports attractor-level
   statistical fidelity ("climate"): spike count, mean/CV of the inter-spike
   interval, and per-state standard deviation and range, predicted vs true,
   reusing the project's own frozen spike detector.

Final-test data is opened only through the sanctioned token entry point.
"""
from __future__ import annotations

import json

import numpy as np

from chapter2.dynamics_analysis_ch2 import detect_spikes
from chapter2.esn_config import TransitionRange, ValidationWindow
from chapter2.esn_data import create_one_step_pairs, scale_one_step_pairs
from chapter2.esn_metrics import evaluate_rollout

from . import config, data, validation
from .chaotic_ensemble import fit_ensemble, median_ensemble_rollout
from .validation import (
    COLLAPSE_STD_RATIO_THRESHOLD,
    DIVERGENCE_THRESHOLD,
    NONFINITE_FAILURE_SCORE,
    VALID_PREDICTION_THRESHOLD,
)

RESULTS_DIR = config.PACKAGE_ROOT / "results" / "final_evaluation_long"

# Part 1 Step-8 long-horizon window, reused verbatim.
LONG_WINDOW = ValidationWindow(
    number=1,
    transitions=TransitionRange(70_000, 99_999),
    warmup=TransitionRange(70_000, 72_000),
    scored=TransitionRange(72_000, 99_999),
)
ENSEMBLE_SEEDS = [42, 123, 456, 789, 2026]


def _prepared_long(fixed):
    """Build a PreparedOptimisationTrajectory carrying only the long window."""
    from chapter2.esn_data import PreparedOptimisationTrajectory, ValidationWindowView

    fitting = create_one_step_pairs(fixed, TransitionRange(0, config.FITTING_TRANSITIONS_STOP), include_current=True)
    warmup = create_one_step_pairs(fixed, LONG_WINDOW.warmup, include_current=True)
    scored = create_one_step_pairs(fixed, LONG_WINDOW.scored, include_current=True)
    view = ValidationWindowView(LONG_WINDOW, warmup, scored)
    return PreparedOptimisationTrajectory(fixed.current, fitting, (view,))


def _validation_cases_with_scalers(prepared, scalers):
    """Build ValidationCase objects for ``prepared`` using scalers already
    fitted on the TRAINING model's own currents, instead of fitting new ones
    on the final-test current itself (which would leak test-current
    statistics into the scaling). Same per-window construction as
    validation.prepare_model_data_for_currents, minus the internal
    _fit_scalers_from_permitted_views call.
    """
    cases = []
    for view in prepared.validation_windows:
        scaled_warmup = scale_one_step_pairs(view.warmup, scalers)
        scaled_scored = scale_one_step_pairs(view.scored, scalers)
        if not np.array_equal(scaled_warmup.targets[-1], scaled_scored.inputs[0, :3]):
            raise ValueError("validation warm-up and scored states are misaligned")
        cases.append(
            validation.ValidationCase(
                current=prepared.current,
                window=view.definition.number,
                warmup_inputs=np.asarray(scaled_warmup.inputs, dtype=float).copy(),
                initial_state=scaled_warmup.targets[-1].copy(),
                current_values=scaled_scored.inputs[:, 3].copy(),
                targets_physical=scalers.inverse_states(scaled_scored.targets),
                warmup_range=(
                    int(view.warmup.transition_indices[0]),
                    int(view.warmup.transition_indices[-1]) + 1,
                ),
                scored_range=(
                    int(view.scored.transition_indices[0]),
                    int(view.scored.transition_indices[-1]) + 1,
                ),
            )
        )
    return cases


def _climate_stats(series: np.ndarray) -> dict:
    """Attractor-level ("climate") statistics for one 3-state trajectory."""
    x = series[:, 0]
    if not np.all(np.isfinite(series)):
        return {"finite": False}
    peaks = detect_spikes(x)
    isi = np.diff(peaks) * config.DT
    return {
        "finite": True,
        "spike_count": int(len(peaks)),
        "mean_isi": float(np.mean(isi)) if len(isi) else None,
        "isi_cv": float(np.std(isi, ddof=0) / np.mean(isi)) if len(isi) and np.mean(isi) > 0 else None,
        "state_std": [float(v) for v in np.std(series, axis=0)],
        "state_min": [float(v) for v in np.min(series, axis=0)],
        "state_max": [float(v) for v in np.max(series, axis=0)],
    }


def _score_long(models, case, scalers) -> dict:
    pred_scaled, _, failure_step = median_ensemble_rollout(models, case, config.MODEL_TYPE)
    predictions_physical = pred_scaled * scalers.state.scale + scalers.state.mean
    metrics = evaluate_rollout(
        predictions_physical, case.targets_physical,
        normalisation_scale=scalers.state.scale, dt=config.DT,
        valid_prediction_threshold=VALID_PREDICTION_THRESHOLD,
        divergence_threshold=DIVERGENCE_THRESHOLD,
        collapse_std_ratio_threshold=COLLAPSE_STD_RATIO_THRESHOLD,
    ).to_dict()
    nonfinite = failure_step is not None or metrics["nrmse_state"] is None
    pred_climate = _climate_stats(predictions_physical)
    true_climate = _climate_stats(case.targets_physical)
    spike_ratio = None
    if pred_climate.get("spike_count") is not None and true_climate.get("spike_count"):
        spike_ratio = pred_climate["spike_count"] / true_climate["spike_count"]
    return {
        "current": case.current,
        "horizon": len(case.targets_physical),
        "objective_nrmse": NONFINITE_FAILURE_SCORE if nonfinite else float(metrics["nrmse_state"]),
        "nonfinite_failure": nonfinite,
        "metrics": metrics,
        "predicted_climate": pred_climate,
        "true_climate": true_climate,
        "spike_count_ratio": spike_ratio,
        "predictions_physical": predictions_physical,
        "targets_physical": case.targets_physical,
    }


def _summary(rollouts: list[dict]) -> dict:
    vpt_time = [r["metrics"]["valid_prediction_steps"] * config.DT for r in rollouts]
    nrmses = [r["objective_nrmse"] for r in rollouts]
    ratios = [r["spike_count_ratio"] for r in rollouts if r["spike_count_ratio"] is not None]
    return {
        "n_rollouts": len(rollouts),
        "horizon_time_units": rollouts[0]["horizon"] * config.DT,
        "median_nrmse": float(np.median(nrmses)),
        "mean_nrmse": float(np.mean(nrmses)),
        "median_vpt_time": float(np.median(vpt_time)),
        "mean_vpt_time": float(np.mean(vpt_time)),
        "divergence_count": int(sum(bool(r["metrics"]["diverged"]) for r in rollouts)),
        "collapse_count": int(sum(bool(r["metrics"]["prediction_collapse_any"]) for r in rollouts)),
        "numerical_failure_count": int(sum(bool(r["nonfinite_failure"]) for r in rollouts)),
        "median_spike_count_ratio": float(np.median(ratios)) if ratios else None,
        "spike_ratio_within_10pct": int(sum(1 for v in ratios if 0.9 <= v <= 1.1)),
    }


def run() -> dict:
    design = data.load_frozen_design()

    families = {}
    for family, key, role in (
        ("regular", "regular_trained", "regular_training"),
        ("chaotic", "chaotic_trained", "chaotic_training"),
    ):
        currents = [e["current"] for e in design["models"][key]["training_currents"]]
        prepared = [
            data.prepare_optimisation_trajectory(data.load_source_trajectory(role, c))
            for c in currents
        ]
        model_data = validation.prepare_model_data_for_currents(prepared)
        if family == "regular":
            hp = json.loads((config.OPTIMISATION_DIR / "regular_selection.json").read_text())["locked_hyperparameters"]
        else:
            hp = json.loads(
                (config.PACKAGE_ROOT / "results" / "chaotic_research" / "chaotic_ensemble_selection.json").read_text()
            )["ensemble_hyperparameters"]
        models = fit_ensemble(model_data, hp, ENSEMBLE_SEEDS)
        families[family] = {"models": models, "scalers": model_data.scalers, "hp": hp, "currents": currents}
        print(f"{family} 5-seed ensemble fitted (hp={hp})", flush=True)

    directions = {
        "RR": ("regular", [e["current"] for e in design["models"]["regular_trained"]["rr_test_currents"]]),
        "RC": ("regular", [e["current"] for e in design["models"]["regular_trained"]["rc_test_currents"]]),
        "CC": ("chaotic", [e["current"] for e in design["models"]["chaotic_trained"]["cc_test_currents"]]),
        "CR": ("chaotic", [e["current"] for e in design["models"]["chaotic_trained"]["cr_test_currents"]]),
    }

    out_directions = {}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, (family, test_currents) in directions.items():
        fam = families[family]
        rollouts, raw = [], []
        for current in test_currents:
            fixed = data.load_final_test_trajectory(current, caller_token=data.FINAL_EVALUATION_ENTRY_POINT_TOKEN)
            prepared = _prepared_long(fixed)
            cases = _validation_cases_with_scalers(prepared, fam["scalers"])
            for case in cases:
                scored = _score_long(fam["models"], case, fam["scalers"])
                raw.append(scored)
                rollouts.append({k: v for k, v in scored.items() if k not in ("predictions_physical", "targets_physical")})
            print(f"[{name}] I={current:.6f} done", flush=True)
        summary = _summary(rollouts)
        out_directions[name] = {"family": family, "test_currents": test_currents, "rollouts": rollouts, "summary": summary}
        print(
            f"=== {name} ({summary['horizon_time_units']:.0f} tu): median_nrmse={summary['median_nrmse']:.4g} "
            f"median_vpt_time={summary['median_vpt_time']:.1f} tu  div={summary['divergence_count']}/{summary['n_rollouts']} "
            f"spike_ratio={summary['median_spike_count_ratio']} ===",
            flush=True,
        )
        # Save every rollout's arrays for figures (3 per direction, modest size).
        for scored in raw:
            np.savez_compressed(
                RESULTS_DIR / f"long_{name}_I{scored['current']:.6f}.npz".replace(".", "p", 1),
                truth=scored["targets_physical"], prediction=scored["predictions_physical"],
                current=scored["current"], nrmse=scored["objective_nrmse"],
                vpt_time=scored["metrics"]["valid_prediction_steps"] * config.DT,
            )

    payload = {
        "kind": "final_rr_rc_cc_cr_long_horizon_benchmark",
        "design_hash": design["design_hash_sha256"],
        "window": {
            "warmup_range": [LONG_WINDOW.warmup.start, LONG_WINDOW.warmup.stop],
            "forecast_range": [LONG_WINDOW.scored.start, LONG_WINDOW.scored.stop],
            "forecast_steps": LONG_WINDOW.scored.stop - LONG_WINDOW.scored.start,
            "forecast_time_units": (LONG_WINDOW.scored.stop - LONG_WINDOW.scored.start) * config.DT,
            "source": "identical to chapter2/esn_step8.py Step-8 long_horizon window",
        },
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "both_families_use_median_ensemble": True,
        "directions": out_directions,
    }
    (RESULTS_DIR / "final_long_horizon_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {RESULTS_DIR / 'final_long_horizon_results.json'}")
    return payload


if __name__ == "__main__":
    run()
