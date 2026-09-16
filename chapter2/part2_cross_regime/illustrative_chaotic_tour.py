"""Illustrative-only supplementary figure: the chaotic model touring the FULL
confirmed chaotic range (all training + test currents together), so the
narrow width of that range is visible in one continuous picture.

This does NOT touch, recompute, or change the official locked CC/CR
benchmark in any way -- it reuses the already-locked chaotic ensemble and
simply runs it across a wider illustrative schedule for visualisation.
"""
from __future__ import annotations

import numpy as np

from . import config, data
from .final_continuous_schedule import build_family, ENSEMBLE_SEEDS

RESULTS_DIR = config.PACKAGE_ROOT / "results" / "final_evaluation_long"


def main() -> None:
    design = data.load_frozen_design()
    train = sorted(round(e["current"], 6) for e in design["models"]["chaotic_trained"]["training_currents"])
    test = sorted(round(e["current"], 6) for e in design["models"]["chaotic_trained"]["cc_test_currents"])
    schedule = sorted(set(train) | set(test))
    print("Illustrative chaotic tour schedule (low to high):", schedule)
    print(f"Full confirmed-chaotic span: {schedule[0]:.4f} to {schedule[-1]:.4f} "
          f"({schedule[-1]-schedule[0]:.4f} wide)")

    result = build_family("chaotic", "chaotic_trained", "chaotic_training", schedule)

    np.savez_compressed(
        RESULTS_DIR / "illustrative_chaotic_tour.npz",
        time=result["time"], current=result["current"],
        truth=result["truth"], prediction=result["prediction"],
        switch_indices=np.array(result["switch_indices"]),
        schedule_currents=np.array(result["schedule_currents"]),
        is_test_current=np.array([c in test for c in schedule]),
    )
    print(f"wrote {RESULTS_DIR / 'illustrative_chaotic_tour.npz'}")


if __name__ == "__main__":
    main()
