"""Step 1 support: targeted Lyapunov diagnostics to finalize train/test currents.

No ESN is created here. This only calls the frozen, previously validated
chapter2.dynamics_analysis_ch2.estimate_lyapunov (Benettin tangent-linear
estimator) at a handful of new currents chosen to (a) span the chaotic band
more broadly than Step 1's sparse subset, and (b) test whether the narrow
I~2.56-2.58 periodic window is confidently bracketed by chaotic dynamics on
both sides. No new Lyapunov estimator is implemented.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from chapter2.dynamics_analysis_ch2 import analyze_spikes_and_bursts, detect_spikes
from chapter2.hr_data_ch2 import simulate_fixed_current
from chapter2.dynamics_analysis_ch2 import estimate_lyapunov
from chapter2.config_ch2 import INITIAL_TRANSIENT_STEPS, RETAINED_SAMPLES_PER_CURRENT

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "design"

# Candidates spanning the lower/middle chaotic band (2.1-2.9), plus the two
# points immediately bracketing the I~2.56-2.58 periodic window on the low
# side (2.45, 2.50) and the high side (2.62, 2.68, 2.75).
CANDIDATE_CURRENTS = [2.15, 2.30, 2.45, 2.50, 2.62, 2.68, 2.75, 2.90]


def _json_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def run() -> dict:
    t0 = time.time()
    records = []
    for current in CANDIDATE_CURRENTS:
        traj = simulate_fixed_current(
            current,
            retained_samples=RETAINED_SAMPLES_PER_CURRENT,
            transient_steps=INITIAL_TRANSIENT_STEPS,
        )
        spikes = analyze_spikes_and_bursts(traj)
        lyap = estimate_lyapunov(current)
        records.append(
            {
                "current": current,
                "spike_count": int(len(spikes.spike_indices)),
                "burst_structure": spikes.burst_structure,
                "isi_cv": _json_float(spikes.isi_cv),
                "mean_spikes_per_burst": _json_float(spikes.mean_spikes_per_burst),
                "within_burst_isi_cv": _json_float(spikes.within_burst_isi_cv),
                "interburst_interval_cv": _json_float(spikes.interburst_interval_cv),
                "lyapunov_exponent": _json_float(lyap.exponent),
                "lyapunov_classification": lyap.classification,
                "lyapunov_convergence": lyap.convergence,
            }
        )
        print(
            f"I={current:.4f} spikes={records[-1]['spike_count']} "
            f"burst={records[-1]['burst_structure']} "
            f"LLE={records[-1]['lyapunov_exponent']} "
            f"{records[-1]['lyapunov_classification']}/{records[-1]['lyapunov_convergence']}",
            flush=True,
        )

    payload = {
        "kind": "step1_targeted_lyapunov_diagnostics",
        "candidate_currents": CANDIDATE_CURRENTS,
        "records": records,
        "elapsed_seconds": time.time() - t0,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "targeted_lyapunov_diagnostics.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return payload


if __name__ == "__main__":
    run()
