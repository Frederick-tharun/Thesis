"""Continuous current-switching benchmark, mirroring Chapter 2 Part 1's own
02_continuous_current_prediction figure style (supplied current / truth /
prediction, one panel each, all currents on one shared timeline).

Unlike the per-current long-horizon benchmark (final_evaluation_long.py),
this builds ONE continuous ground-truth trajectory per model where the
current steps between several test currents and the reservoir state is
NEVER reset at a switch -- reusing chapter2.hr_data_ch2.simulate_continuous_currents
unchanged, exactly as Part 1 does.

The current values used here are the already-finalised RR/RC (regular model)
and CC/CR (chaotic model) test currents -- no new final-test file is opened;
this is a fresh HR simulation from the frozen public equations at currents
that have already been spent by the final evaluation, purely for the
combined visualisation Part 1's own figure uses.
"""
from __future__ import annotations

import json

import numpy as np

from chapter2.hr_data_ch2 import simulate_continuous_currents

from . import config, data, validation
from .chaotic_ensemble import fit_ensemble

RESULTS_DIR = config.PACKAGE_ROOT / "results" / "final_evaluation_long"
SAMPLES_PER_SEGMENT = 30_000  # 300 time units/segment: settles past the ~167tu slow-variable timescale
ENSEMBLE_SEEDS = [42, 123, 456, 789, 2026]
WARMUP_STEPS = 2_000


def _ensemble_predict(models, state, current_values):
    """Autonomous rollout: teacher-forced warm-up, then never reset, never
    re-warm at a current switch -- exactly Part 1's continuous convention.
    """
    for m in models:
        m.reset_reservoir()
    warm_inputs = np.column_stack([state[:WARMUP_STEPS], current_values[:WARMUP_STEPS]])
    for m in models:
        m.teacher_forced_warmup(warm_inputs, reset=False)

    horizon = len(current_values)
    preds = np.full((horizon, 3), np.nan)
    preds[:WARMUP_STEPS] = state[:WARMUP_STEPS]
    live = state[WARMUP_STEPS - 1].copy()
    for step in range(WARMUP_STEPS, horizon):
        input_value = np.concatenate((live, [current_values[step - 1]]))
        member_preds = np.stack([m.predict_one_step(input_value) for m in models], axis=0)
        if not np.all(np.isfinite(member_preds)):
            finite = member_preds[np.all(np.isfinite(member_preds), axis=1)]
            if len(finite) <= len(models) // 2:
                preds[step:] = np.nan
                break
            live = np.median(finite, axis=0)
        else:
            live = np.median(member_preds, axis=0)
        preds[step] = live
    return preds


def build_family(family: str, key: str, role: str, schedule_currents: list[float]) -> dict:
    design = data.load_frozen_design()
    train_currents = [e["current"] for e in design["models"][key]["training_currents"]]
    prepared = [
        data.prepare_optimisation_trajectory(data.load_source_trajectory(role, c))
        for c in train_currents
    ]
    model_data = validation.prepare_model_data_for_currents(prepared)
    if family == "regular":
        hp = json.loads((config.OPTIMISATION_DIR / "regular_selection.json").read_text())["locked_hyperparameters"]
    else:
        hp = json.loads(
            (config.PACKAGE_ROOT / "results" / "chaotic_research" / "chaotic_ensemble_selection.json").read_text()
        )["ensemble_hyperparameters"]
    models = fit_ensemble(model_data, hp, ENSEMBLE_SEEDS)

    trajectory, switch_indices = simulate_continuous_currents(
        schedule_currents,
        samples_per_segment=SAMPLES_PER_SEGMENT,
        transient_steps=config.INITIAL_TRANSIENT_STEPS,
    )
    state = trajectory.state  # (N,3) physical
    current_values = trajectory.I

    # Scale using the TRAINING model's scalers -- never refit here.
    scalers = model_data.scalers
    state_scaled = scalers.state.transform(state)
    current_scaled = scalers.current.transform(current_values.reshape(-1, 1)).ravel()

    pred_scaled = _ensemble_predict(models, state_scaled, current_scaled)
    pred_physical = pred_scaled * scalers.state.scale + scalers.state.mean

    return {
        "schedule_currents": schedule_currents,
        "switch_indices": switch_indices.tolist(),
        "time": trajectory.t,
        "current": current_values,
        "truth": state,
        "prediction": pred_physical,
    }


def main() -> None:
    design = data.load_frozen_design()
    rr = [round(e["current"], 6) for e in design["models"]["regular_trained"]["rr_test_currents"]]
    rc = [round(e["current"], 6) for e in design["models"]["regular_trained"]["rc_test_currents"]]
    cc = [round(e["current"], 6) for e in design["models"]["chaotic_trained"]["cc_test_currents"]]
    cr = [round(e["current"], 6) for e in design["models"]["chaotic_trained"]["cr_test_currents"]]

    print("Building REGULAR-model continuous schedule (RR then RC currents)...", flush=True)
    regular_result = build_family("regular", "regular_trained", "regular_training", rr + rc)
    print("Building CHAOTIC-model continuous schedule (CC then CR currents)...", flush=True)
    chaotic_result = build_family("chaotic", "chaotic_trained", "chaotic_training", cc + cr)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, r in (("regular", regular_result), ("chaotic", chaotic_result)):
        np.savez_compressed(
            RESULTS_DIR / f"continuous_schedule_{name}.npz",
            time=r["time"], current=r["current"], truth=r["truth"], prediction=r["prediction"],
            switch_indices=np.array(r["switch_indices"]), schedule_currents=np.array(r["schedule_currents"]),
        )
        print(f"wrote {RESULTS_DIR / f'continuous_schedule_{name}.npz'}", flush=True)


if __name__ == "__main__":
    main()
