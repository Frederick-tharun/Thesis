"""Before/after figure: single bad-seed runaway vs. median-ensemble fix."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chapter2.esn_model import EchoStateNetwork

from . import config, data, validation
from .chaotic_ensemble import RESULTS_DIR, fit_ensemble, median_ensemble_rollout
from .validation import TRAINING_WASHOUT, model_config


def main() -> None:
    design = data.load_frozen_design()
    anchor_hp = json.loads(
        (config.BASELINE_DIR / "part1_protocol_snapshot.json").read_text()
    )["step7_selected_parameter_aware_hyperparameters"]
    currents = [e["current"] for e in design["models"]["chaotic_trained"]["training_currents"]]
    prepared = [
        data.prepare_optimisation_trajectory(data.load_source_trajectory("chaotic_training", c))
        for c in currents
    ]
    model_data = validation.prepare_model_data_for_currents(prepared)
    case = next(c for c in model_data.validation_cases if c.current == 3.20 and c.window == 1)
    t = np.arange(len(case.targets_physical)) * 0.01

    # Single bad seed (456).
    bad_model = EchoStateNetwork(model_config(anchor_hp, config.MODEL_TYPE, 456))
    bad_model.fit(model_data.training_sequences, washout=TRAINING_WASHOUT)
    from chapter2.esn_optimisation import _recursive_case_rollout

    bad_pred_scaled, _, _ = _recursive_case_rollout(bad_model, case, config.MODEL_TYPE)
    bad_pred_physical = bad_pred_scaled * model_data.scalers.state.scale + model_data.scalers.state.mean

    # 5-member median ensemble (includes the same bad seed 456 as a minority member).
    seeds = [42, 123, 456, 789, 2026]
    models = fit_ensemble(model_data, anchor_hp, seeds)
    ens_pred_scaled, _, _ = median_ensemble_rollout(models, case, config.MODEL_TYPE)
    ens_pred_physical = ens_pred_scaled * model_data.scalers.state.scale + model_data.scalers.state.mean

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, case.targets_physical[:, 0], color="#222222", lw=1, label="truth")
    axes[0].plot(t, bad_pred_physical[:, 0], color="#D55E00", lw=1, label="single model, seed 456 (bad)")
    axes[0].set_ylabel("x")
    axes[0].set_title("Before: a single bad-seed reservoir diverges (I=3.20, window 1)")
    axes[0].legend()

    axes[1].plot(t, case.targets_physical[:, 0], color="#222222", lw=1, label="truth")
    axes[1].plot(t, ens_pred_physical[:, 0], color="#0072B2", lw=1, label="5-seed median ensemble (incl. seed 456)")
    axes[1].set_ylabel("x")
    axes[1].set_xlabel("time")
    axes[1].set_title("After: median ensemble of the same 5 seeds stays bounded and accurate")
    axes[1].legend()
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "before_after_ensemble_fix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
