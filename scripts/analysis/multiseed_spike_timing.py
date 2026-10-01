"""Spike timing and valid-prediction horizon for each multiseed rollout.

Reads the per-seed held-out rollouts saved by run_multiseed_evaluation.slurm
(prediction_rollout.npz, kept outside Git) and writes
MULTISEED_EVAL/chaotic_spike_timing_by_seed.csv. These values are reported
in Table 2.6 of the thesis.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from prediction_diagnostics import chaotic_valid_prediction_horizon

SEEDS = (42, 123, 456, 789, 2026)
LATE_WINDOW = 30.0  # final time units used for the late-error share


def spike_peaks(x: np.ndarray, threshold: float) -> np.ndarray:
    above = np.r_[False, x >= threshold, False].astype(int)
    starts = np.flatnonzero(np.diff(above) == 1)
    ends = np.flatnonzero(np.diff(above) == -1)
    return np.array([s + int(np.argmax(x[s:e])) for s, e in zip(starts, ends)], dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True,
                        help="Multiseed output root containing seed_<seed>/prediction_rollout.npz")
    args = parser.parse_args()

    metrics = json.loads(
        (REPO_ROOT / "FINAL_THESIS_RUN/01_prediction_all_regimes/chaotic_bursting"
         / "heldout_test_metrics.json").read_text()
    )
    lyapunov = metrics["chaotic_valid_prediction_horizon"]["largest_lyapunov_exponent"]
    dt = float(config.HR_DT)

    rows = []
    for seed in SEEDS:
        data = np.load(args.rollout_root / f"seed_{seed}" / "prediction_rollout.npz")
        reference, prediction = data["truth"], data["prediction"]
        mean = data["train_mean"].reshape(1, -1)
        std = data["train_std"].reshape(1, -1)
        time = data["time"] - data["time"][0]

        horizon, _ = chaotic_valid_prediction_horizon(
            (prediction - mean) / std, (reference - mean) / std,
            dt=dt, largest_lyapunov_exponent=lyapunov,
        )
        ref_peaks = spike_peaks(reference[:, 0], config.SPIKE_THRESHOLD)
        pred_peaks = spike_peaks(prediction[:, 0], config.SPIKE_THRESHOLD)
        if len(ref_peaks) != len(pred_peaks):
            raise RuntimeError(f"seed {seed}: spike counts differ")
        offsets = time[pred_peaks] - time[ref_peaks]
        squared_error = (prediction[:, 0] - reference[:, 0]) ** 2
        late = time >= time[-1] - LATE_WINDOW + dt

        rows.append({
            "seed": seed,
            "nrmse_x": float(np.sqrt(squared_error.mean()) / reference[:, 0].std()),
            "valid_prediction_horizon": horizon["horizon_time"],
            "spike_count": len(ref_peaks),
            "max_abs_offset_first_six": float(np.abs(offsets[:6]).max()),
            "max_abs_spike_offset": float(np.abs(offsets).max()),
            "late_squared_error_share_x": float(squared_error[late].sum() / squared_error.sum()),
            "spike_offsets": " ".join(f"{o:+.2f}" for o in offsets),
        })

    out = REPO_ROOT / "MULTISEED_EVAL" / "chaotic_spike_timing_by_seed.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
