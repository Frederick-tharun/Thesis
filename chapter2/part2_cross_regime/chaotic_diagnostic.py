"""Phase A/B: read-only reconstruction + source-only diagnostic of the chaotic
source-model failure.

Reuses the frozen Part-1 EchoStateNetwork and Part-2 validation utilities
unchanged; only calls PUBLIC methods (accumulate_ridge_statistics,
ridge_penalty_matrix, teacher_forced_warmup, predict_one_step,
reservoir_state) to get more visibility into the fit than SourceCandidateEvaluator
exposes. Never writes to any Part-1 file. Never opens final-test trajectories.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chapter2.dynamics_analysis_ch2 import analyze_spikes_and_bursts, detect_spikes
from chapter2.esn_data import TransitionRange, create_one_step_pairs, scale_one_step_pairs
from chapter2.esn_model import EchoStateNetwork

from . import config, data, validation
from .validation import (
    COLLAPSE_STD_RATIO_THRESHOLD,
    DIVERGENCE_THRESHOLD,
    NONFINITE_FAILURE_SCORE,
    TRAINING_WASHOUT,
    VALID_PREDICTION_THRESHOLD,
    model_config,
)
from chapter2.esn_metrics import evaluate_rollout

RESULTS_DIR = config.PACKAGE_ROOT / "results" / "chaotic_diagnostic"
CHAOTIC_ROLE = "chaotic_training"


def _json_default(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def _finite_or_none(v):
    v = float(v)
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------
# Phase A: reload the existing search/confirmation evidence (no recompute)
# --------------------------------------------------------------------------

def load_existing_evidence() -> dict:
    opt_dir = config.OPTIMISATION_DIR
    history = json.loads((opt_dir / "chaotic_history.json").read_text())
    confirmation = json.loads((opt_dir / "chaotic_confirmation.json").read_text())
    selection = json.loads((opt_dir / "chaotic_selection.json").read_text())
    by_key = {
        json.dumps(r["hyperparameters"], sort_keys=True): r for r in history["history"]
    }
    top10 = [by_key[k] for k in history["ranked_hyperparameter_keys"][:10]]
    return {"history": history, "confirmation": confirmation, "selection": selection, "top10": top10}


def reproduce_top_candidate(design: dict, top10: list[dict], *, seed: int) -> dict:
    """Re-fit and re-score the rank-1 (anchor) candidate at one seed; compare bit-for-bit."""
    from .optimisation import evaluate_candidate

    currents = [e["current"] for e in design["models"]["chaotic_trained"]["training_currents"]]
    prepared = [
        data.prepare_optimisation_trajectory(data.load_source_trajectory(CHAOTIC_ROLE, c))
        for c in currents
    ]
    model_data = validation.prepare_model_data_for_currents(prepared)
    hp = top10[0]["hyperparameters"]
    fresh = evaluate_candidate(model_data, dict(hp, _source="reproduction_check"), seed)
    stored = top10[0]
    matches = (
        abs(fresh["objective"] - stored["objective"]) < 1e-9
        and fresh["aggregate"]["divergence_rollout_count"] == stored["aggregate"]["divergence_rollout_count"]
        and fresh["aggregate"]["collapse_rollout_count"] == stored["aggregate"]["collapse_rollout_count"]
    )
    return {
        "seed": seed,
        "stored_objective": stored["objective"],
        "fresh_objective": fresh["objective"],
        "matches": matches,
        "model_data": model_data,
    }


# --------------------------------------------------------------------------
# Phase B: fit once per seed, run one-step + autonomous + reservoir/readout
# diagnostics for every (current, window).
# --------------------------------------------------------------------------

def _fit_with_diagnostics(model_data, hyperparameters: dict, seed: int):
    model = EchoStateNetwork(model_config(hyperparameters, config.MODEL_TYPE, seed))
    stats = model.accumulate_ridge_statistics(model_data.training_sequences, washout=TRAINING_WASHOUT)
    system = stats.gram + model.config.ridge_regularisation * model.ridge_penalty_matrix
    weights = np.linalg.solve(system, stats.cross.T).T
    model.output_weights = np.ascontiguousarray(weights, dtype=float)
    model.reset_reservoir()

    reservoir_size = model.config.reservoir_size
    w_out = model.output_weights  # shape (3, 1 + 4 + reservoir_size)
    bias_block = w_out[:, 0]
    input_block = w_out[:, 1:5]
    reservoir_block = w_out[:, 5:]
    readout_diag = {
        "frobenius_norm": float(np.linalg.norm(w_out)),
        "max_abs_coefficient": float(np.max(np.abs(w_out))),
        "bias_block_norm": float(np.linalg.norm(bias_block)),
        "input_block_norm": float(np.linalg.norm(input_block)),
        "reservoir_block_norm": float(np.linalg.norm(reservoir_block)),
        "gram_condition_number_regularised": float(np.linalg.cond(system)),
        "gram_condition_number_unregularised": float(np.linalg.cond(stats.gram)),
        "ridge_sample_count": int(stats.sample_count),
    }
    return model, readout_diag


def _recursive_rollout_with_states(model, case, model_type: str):
    """Copy of esn_optimisation._recursive_case_rollout, extended to also
    record the reservoir state and pointwise-normalised error at every step
    (needed for B2/B3), without touching esn_model.py or esn_optimisation.py.
    """
    model.reset_reservoir()
    model.teacher_forced_warmup(case.warmup_inputs, reset=False)
    state = case.initial_state.copy()
    horizon = len(case.targets_physical)
    predictions_scaled = np.full((horizon, 3), np.nan, dtype=float)
    reservoir_norms = np.full(horizon, np.nan, dtype=float)
    reservoir_max_abs = np.full(horizon, np.nan, dtype=float)
    frac_sat_90 = np.full(horizon, np.nan, dtype=float)
    frac_sat_95 = np.full(horizon, np.nan, dtype=float)
    frac_sat_99 = np.full(horizon, np.nan, dtype=float)

    failure_step = None
    failure_reason = None
    for step in range(horizon):
        if model_type == config.PARAMETER_AWARE:
            input_value = np.concatenate((state, [case.current_values[step]]))
        else:
            input_value = state
        prediction = model.predict_one_step(input_value)
        r = model.reservoir_state
        reservoir_norms[step] = float(np.linalg.norm(r))
        reservoir_max_abs[step] = float(np.max(np.abs(r)))
        frac_sat_90[step] = float(np.mean(np.abs(r) > 0.90))
        frac_sat_95[step] = float(np.mean(np.abs(r) > 0.95))
        frac_sat_99[step] = float(np.mean(np.abs(r) > 0.99))
        predictions_scaled[step] = prediction
        if not np.all(np.isfinite(prediction)):
            failure_step = step
            failure_reason = "non_finite_prediction"
            break
        state = prediction
    return {
        "predictions_scaled": predictions_scaled,
        "reservoir_norms": reservoir_norms,
        "reservoir_max_abs": reservoir_max_abs,
        "frac_sat_90": frac_sat_90,
        "frac_sat_95": frac_sat_95,
        "frac_sat_99": frac_sat_99,
        "failure_step": failure_step,
        "failure_reason": failure_reason,
    }


PHYSICAL_BOUNDS = {"x": (-2.5, 2.5), "y": (-16.0, 3.0), "z": (0.5, 4.5)}


def _classify_time_resolved(
    pointwise_error: np.ndarray, predictions_physical: np.ndarray, dt: float
) -> dict:
    crossing = np.flatnonzero(pointwise_error >= VALID_PREDICTION_THRESHOLD)
    first_cross_step = int(crossing[0]) if len(crossing) else None
    first_cross_time = first_cross_step * dt if first_cross_step is not None else None

    recovered = False
    if first_cross_step is not None and first_cross_step + 50 < len(pointwise_error):
        tail = pointwise_error[first_cross_step + 20 :]
        recovered = bool(np.any(tail < VALID_PREDICTION_THRESHOLD))

    out_of_bounds = False
    if np.all(np.isfinite(predictions_physical)):
        x, y, z = predictions_physical[:, 0], predictions_physical[:, 1], predictions_physical[:, 2]
        out_of_bounds = bool(
            np.any(x < PHYSICAL_BOUNDS["x"][0]) or np.any(x > PHYSICAL_BOUNDS["x"][1])
            or np.any(y < PHYSICAL_BOUNDS["y"][0]) or np.any(y > PHYSICAL_BOUNDS["y"][1])
            or np.any(z < PHYSICAL_BOUNDS["z"][0]) or np.any(z > PHYSICAL_BOUNDS["z"][1])
        )

    diverged = bool(np.any(pointwise_error >= DIVERGENCE_THRESHOLD)) or not np.all(np.isfinite(predictions_physical))
    explosive = (not np.all(np.isfinite(predictions_physical))) or out_of_bounds
    classification = (
        "true_divergence" if (diverged and explosive)
        else "phase_separation" if diverged
        else "within_threshold"
    )
    return {
        "first_threshold_crossing_step": first_cross_step,
        "first_threshold_crossing_time": first_cross_time,
        "error_recovers_after_crossing": recovered,
        "leaves_plausible_hr_state_range": out_of_bounds,
        "diverged_by_registered_rule": diverged,
        "classification": classification,
    }


def diagnose_candidate(
    label: str, hyperparameters: dict, seeds: list[int], model_data, dt: float = 0.01
) -> dict:
    per_seed = {}
    for seed in seeds:
        model, readout_diag = _fit_with_diagnostics(model_data, hyperparameters, seed)
        rollouts = []
        for case in model_data.validation_cases:
            result = _recursive_rollout_with_states(model, case, config.MODEL_TYPE)
            predictions_physical = (
                result["predictions_scaled"] * model_data.scalers.state.scale
                + model_data.scalers.state.mean
            )
            with np.errstate(over="ignore", invalid="ignore"):
                normalised_difference = (
                    predictions_physical - case.targets_physical
                ) / model_data.scalers.state.scale
                pointwise_error = np.sqrt(np.mean(np.square(normalised_difference), axis=1))
            metrics = evaluate_rollout(
                predictions_physical, case.targets_physical,
                normalisation_scale=model_data.scalers.state.scale, dt=dt,
                valid_prediction_threshold=VALID_PREDICTION_THRESHOLD,
                divergence_threshold=DIVERGENCE_THRESHOLD,
                collapse_std_ratio_threshold=COLLAPSE_STD_RATIO_THRESHOLD,
            ).to_dict()
            time_resolved = _classify_time_resolved(
                np.nan_to_num(pointwise_error, nan=np.inf), predictions_physical, dt
            )

            valid_mask = np.isfinite(result["reservoir_norms"])
            reservoir_summary = {
                "mean_abs_r": float(np.nanmean(np.abs(result["reservoir_max_abs"][valid_mask]))) if valid_mask.any() else None,
                "max_abs_r": float(np.nanmax(result["reservoir_max_abs"][valid_mask])) if valid_mask.any() else None,
                "mean_frac_sat_90": _finite_or_none(np.nanmean(result["frac_sat_90"][valid_mask])) if valid_mask.any() else None,
                "mean_frac_sat_95": _finite_or_none(np.nanmean(result["frac_sat_95"][valid_mask])) if valid_mask.any() else None,
                "mean_frac_sat_99": _finite_or_none(np.nanmean(result["frac_sat_99"][valid_mask])) if valid_mask.any() else None,
                "reservoir_norm_std": _finite_or_none(np.nanstd(result["reservoir_norms"][valid_mask])) if valid_mask.any() else None,
            }

            # One-step teacher-forced diagnostic on the same scored range.
            one_step = _one_step_diagnostic(model_data, case, model)

            # B7: dynamical fidelity, only meaningful for finite, non-divergent runs.
            fidelity = None
            if not metrics["diverged"] and result["failure_step"] is None:
                fidelity = _dynamical_fidelity_stats(predictions_physical, case.targets_physical)

            rollouts.append(
                {
                    "current": case.current,
                    "window": case.window,
                    "metrics": metrics,
                    "time_resolved": time_resolved,
                    "reservoir": reservoir_summary,
                    "one_step": one_step,
                    "dynamical_fidelity": fidelity,
                    "failure_step_reservoir_loop": result["failure_step"],
                }
            )
        per_seed[str(seed)] = {"readout": readout_diag, "rollouts": rollouts}
    return {"label": label, "hyperparameters": hyperparameters, "per_seed": per_seed}


def _one_step_diagnostic(model_data, case, model) -> dict:
    """Teacher-forced one-step accuracy on the same scored window (no feedback)."""
    horizon = len(case.targets_physical)
    predictions_scaled = np.full((horizon, 3), np.nan, dtype=float)
    model.reset_reservoir()
    model.teacher_forced_warmup(case.warmup_inputs, reset=False)
    for step in range(horizon):
        # Reconstruct true scaled input: state_t (true) + current_t (true).
        if step == 0:
            state_scaled = case.initial_state.copy()
        else:
            state_scaled = (case.targets_physical[step - 1] - model_data.scalers.state.mean) / model_data.scalers.state.scale
        input_value = np.concatenate((state_scaled, [case.current_values[step]]))
        predictions_scaled[step] = model.predict_one_step(input_value)
    predictions_physical = predictions_scaled * model_data.scalers.state.scale + model_data.scalers.state.mean
    diff = predictions_physical - case.targets_physical
    with np.errstate(over="ignore", invalid="ignore"):
        rmse = float(np.sqrt(np.mean(np.square(diff))))
        mae = float(np.mean(np.abs(diff)))
        nrmse = float(
            np.sqrt(np.mean(np.square(diff / model_data.scalers.state.scale)))
        )
    return {"one_step_nrmse": _finite_or_none(nrmse), "one_step_rmse": _finite_or_none(rmse), "one_step_mae": _finite_or_none(mae)}


# --------------------------------------------------------------------------
# B6/B7 aggregation helpers
# --------------------------------------------------------------------------

def current_specific_summary(diagnosis: dict) -> dict:
    by_current: dict[float, list] = {}
    for seed, info in diagnosis["per_seed"].items():
        for r in info["rollouts"]:
            by_current.setdefault(r["current"], []).append(r)
    summary = {}
    for current, rollouts in by_current.items():
        nrmse = [r["metrics"]["nrmse_state"] for r in rollouts if r["metrics"]["nrmse_state"] is not None]
        vpt = [r["metrics"]["valid_prediction_steps"] for r in rollouts]
        summary[f"{current:.6f}"] = {
            "median_nrmse": float(np.median(nrmse)) if nrmse else None,
            "median_vpt": float(np.median(vpt)),
            "min_vpt": float(np.min(vpt)),
            "divergence_count": int(sum(bool(r["metrics"]["diverged"]) for r in rollouts)),
            "collapse_count": int(sum(bool(r["metrics"]["prediction_collapse_any"]) for r in rollouts)),
            "numerical_failure_count": int(sum(r["failure_step_reservoir_loop"] is not None for r in rollouts)),
            "n_rollouts": len(rollouts),
        }
    return summary


def _dynamical_fidelity_stats(predictions_physical: np.ndarray, targets_physical: np.ndarray) -> dict:
    """B7: spike/burst/state-range comparison, reusing the project's own detector."""
    pred_x = predictions_physical[:, 0]
    true_x = targets_physical[:, 0]
    pred_spikes = detect_spikes(pred_x)
    true_spikes = detect_spikes(true_x)

    def _isi_stats(peaks):
        isi = np.diff(peaks) * 0.01
        if len(isi) == 0:
            return None, None
        return float(np.mean(isi)), float(np.std(isi, ddof=0))

    pred_mean_isi, pred_std_isi = _isi_stats(pred_spikes)
    true_mean_isi, true_std_isi = _isi_stats(true_spikes)
    return {
        "predicted_spike_count": int(len(pred_spikes)),
        "true_spike_count": int(len(true_spikes)),
        "predicted_mean_isi": pred_mean_isi,
        "true_mean_isi": true_mean_isi,
        "predicted_state_std": [float(v) for v in np.std(predictions_physical, axis=0)],
        "true_state_std": [float(v) for v in np.std(targets_physical, axis=0)],
        "predicted_state_range": [
            [float(np.min(predictions_physical[:, i])), float(np.max(predictions_physical[:, i]))]
            for i in range(3)
        ],
        "true_state_range": [
            [float(np.min(targets_physical[:, i])), float(np.max(targets_physical[:, i]))]
            for i in range(3)
        ],
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _rebuild_case_arrays(model_data, hyperparameters, seed, current, window):
    model, _ = _fit_with_diagnostics(model_data, hyperparameters, seed)
    case = next(c for c in model_data.validation_cases if c.current == current and c.window == window)
    result = _recursive_rollout_with_states(model, case, config.MODEL_TYPE)
    predictions_physical = (
        result["predictions_scaled"] * model_data.scalers.state.scale + model_data.scalers.state.mean
    )
    return case, result, predictions_physical


def plot_good_vs_bad_seed(model_data, hyperparameters, good_seed, bad_seed, current, window, out_path) -> None:
    case_g, res_g, pred_g = _rebuild_case_arrays(model_data, hyperparameters, good_seed, current, window)
    case_b, res_b, pred_b = _rebuild_case_arrays(model_data, hyperparameters, bad_seed, current, window)
    t = np.arange(len(case_g.targets_physical)) * 0.01

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, case_g.targets_physical[:, 0], color="#222222", lw=1, label="truth")
    axes[0].plot(t, pred_g[:, 0], color="#0072B2", lw=1, label=f"seed {good_seed} (good)")
    axes[0].set_ylabel("x")
    axes[0].set_title(f"Good seed {good_seed} vs truth (I={current:.3g}, window {window})")
    axes[0].legend()

    axes[1].plot(t, case_b.targets_physical[:, 0], color="#222222", lw=1, label="truth")
    axes[1].plot(t, pred_b[:, 0], color="#D55E00", lw=1, label=f"seed {bad_seed} (bad)")
    axes[1].set_ylabel("x")
    axes[1].set_xlabel("time")
    axes[1].set_title(f"Bad seed {bad_seed} vs truth (I={current:.3g}, window {window})")
    axes[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_growth(model_data, hyperparameters, seeds, current, window, out_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(seeds)))
    for seed, color in zip(seeds, colors):
        case, res, pred = _rebuild_case_arrays(model_data, hyperparameters, seed, current, window)
        with np.errstate(over="ignore", invalid="ignore"):
            err = np.sqrt(np.mean(np.square((pred - case.targets_physical) / model_data.scalers.state.scale), axis=1))
        err = np.nan_to_num(err, nan=np.inf, posinf=1e6)
        t = np.arange(len(err)) * 0.01
        ax.plot(t, np.clip(err, 0, 20), label=f"seed {seed}", color=color)
    ax.axhline(VALID_PREDICTION_THRESHOLD, color="black", linestyle="--", lw=1, label="VPT threshold")
    ax.axhline(DIVERGENCE_THRESHOLD, color="red", linestyle=":", lw=1, label="divergence threshold")
    ax.set_xlabel("time")
    ax.set_ylabel("pointwise normalised error (clipped at 20)")
    ax.set_title(f"Error growth across seeds (I={current:.3g}, window {window})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reservoir_stability(diagnosis: dict, out_path) -> None:
    seeds = list(diagnosis["per_seed"].keys())
    mean_sat = []
    max_r = []
    for seed in seeds:
        rollouts = diagnosis["per_seed"][seed]["rollouts"]
        sats = [r["reservoir"]["mean_frac_sat_95"] for r in rollouts if r["reservoir"]["mean_frac_sat_95"] is not None]
        maxs = [r["reservoir"]["max_abs_r"] for r in rollouts if r["reservoir"]["max_abs_r"] is not None]
        mean_sat.append(np.mean(sats) if sats else np.nan)
        max_r.append(np.max(maxs) if maxs else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(seeds, mean_sat, color="#0072B2")
    axes[0].set_ylabel("mean fraction |r| > 0.95")
    axes[0].set_title(f"{diagnosis['label']}: reservoir saturation by seed")
    axes[1].bar(seeds, max_r, color="#D55E00")
    axes[1].axhline(1.0, color="black", linestyle="--", lw=1)
    axes[1].set_ylabel("max |r| observed")
    axes[1].set_title(f"{diagnosis['label']}: peak reservoir activation by seed")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

ALL_SEEDS = [42, 123, 456, 789, 2026]
GOOD_SEED = 42
BAD_SEED = 456


def main() -> None:
    design = data.load_frozen_design()
    evidence = load_existing_evidence()
    top10 = evidence["top10"]

    print("=== Phase A: reproducibility check ===", flush=True)
    repro = reproduce_top_candidate(design, top10, seed=42)
    print(f"stored objective={repro['stored_objective']:.10g} fresh objective={repro['fresh_objective']:.10g} matches={repro['matches']}", flush=True)
    if not repro["matches"]:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "diagnostic_results.json").write_text(
            json.dumps({"phase_a_reproducibility": repro, "STOP": "reproduction mismatch"}, indent=2, default=_json_default) + "\n"
        )
        print("STOP: old chaotic result could not be reproduced deterministically.")
        return

    model_data = repro["model_data"]

    print("=== Phase B: diagnosing candidates ===", flush=True)
    candidates_to_diagnose = [
        ("rank1_anchor", top10[0]["hyperparameters"], ALL_SEEDS),
        ("rank2_best_non_anchor", top10[1]["hyperparameters"], ALL_SEEDS),
        ("rank3", top10[2]["hyperparameters"], [GOOD_SEED, BAD_SEED]),
        ("rank6", top10[5]["hyperparameters"], [GOOD_SEED, BAD_SEED]),
    ]

    diagnoses = {}
    for label, hp, seeds in candidates_to_diagnose:
        print(f"  diagnosing {label} over seeds {seeds}...", flush=True)
        diagnoses[label] = diagnose_candidate(label, hp, seeds, model_data)
        diagnoses[label]["current_specific_summary"] = current_specific_summary(diagnoses[label])

    print("=== Figures ===", flush=True)
    anchor_hp = top10[0]["hyperparameters"]
    plot_good_vs_bad_seed(model_data, anchor_hp, GOOD_SEED, BAD_SEED, 3.20, 1, RESULTS_DIR / "good_seed_vs_bad_seed.png")
    plot_error_growth(model_data, anchor_hp, ALL_SEEDS, 3.20, 1, RESULTS_DIR / "error_growth_comparison.png")
    plot_reservoir_stability(diagnoses["rank1_anchor"], RESULTS_DIR / "reservoir_stability_comparison.png")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "phase_a": {
            "reproducibility": {k: v for k, v in repro.items() if k != "model_data"},
            "top10_hyperparameters": [
                {"rank": i + 1, "source": r["source"], "hyperparameters": r["hyperparameters"],
                 "seed42_objective": r["objective"], "seed42_aggregate": r["aggregate"]}
                for i, r in enumerate(top10)
            ],
            "confirmation_summary": evidence["confirmation"]["confirmations"],
        },
        "phase_b_diagnoses": diagnoses,
        "chaotic_training_currents": [e["current"] for e in design["models"]["chaotic_trained"]["training_currents"]],
    }
    (RESULTS_DIR / "diagnostic_results.json").write_text(
        json.dumps(out, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    print(f"wrote {RESULTS_DIR / 'diagnostic_results.json'}")


if __name__ == "__main__":
    main()
