"""Generate the Part 2 Step 1 Hindmarsh-Rose dense current sweep.

This script only ever calls the frozen, previously validated Chapter 2
Part 1 simulator (``chapter2.hr_data_ch2.simulate_fixed_current``) and
diagnostics (``chapter2.dynamics_analysis_ch2``). It writes exclusively under
``chapter2/part2_regime_map/results/`` and never touches any Part-1 path.

No ESN is created, trained, or evaluated anywhere in this module.

Usage
-----
    python -m chapter2.part2_regime_map.generate_regime_map --pilot
    python -m chapter2.part2_regime_map.generate_regime_map --run
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import subprocess
import sys
import time
from typing import Any

import numpy as np

from chapter2.dynamics_analysis_ch2 import (
    SpikeBurstAnalysis,
    analyze_spikes_and_bursts,
    detect_spikes,
    estimate_lyapunov,
)
from chapter2.hr_data_ch2 import simulate_fixed_current

from . import config
from .analyse_regimes import provisional_label_from_spikes, select_lyapunov_currents


def _json_float(value: float) -> float | None:
    """Convert NaN/inf to ``None`` so the output remains strict JSON."""
    value = float(value)
    return value if math.isfinite(value) else None


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def simulate_and_measure(
    current: float, *, transient_steps: int, retained_samples: int
) -> tuple[SpikeBurstAnalysis, np.ndarray, float, float]:
    """Simulate one fixed-current trajectory and extract bifurcation features.

    Returns the spike/burst analysis, the spike-peak x-values (the
    bifurcation-diagram y-values for this current), and the settled x range.
    """
    trajectory = simulate_fixed_current(
        current,
        retained_samples=retained_samples,
        transient_steps=transient_steps,
    )
    spikes = analyze_spikes_and_bursts(trajectory)
    peak_indices = detect_spikes(trajectory.x)
    peak_x = trajectory.x[peak_indices]
    return spikes, peak_x, float(np.min(trajectory.x)), float(np.max(trajectory.x))


def _record_from_analysis(
    current: float,
    spikes: SpikeBurstAnalysis,
    peak_x: np.ndarray,
    x_min: float,
    x_max: float,
    *,
    is_known_current: bool,
) -> dict[str, Any]:
    return {
        "current": float(current),
        "is_known_current": bool(is_known_current),
        "spike_peak_x": [float(v) for v in peak_x],
        "spike_count": int(len(spikes.spike_indices)),
        "mean_isi": _json_float(spikes.mean_isi),
        "std_isi": _json_float(spikes.std_isi),
        "isi_cv": _json_float(spikes.isi_cv),
        "burst_structure": spikes.burst_structure,
        "burst_count": spikes.burst_count,
        "mean_spikes_per_burst": _json_float(spikes.mean_spikes_per_burst),
        "std_spikes_per_burst": _json_float(spikes.std_spikes_per_burst),
        "mean_within_burst_isi": _json_float(spikes.mean_within_burst_isi),
        "within_burst_isi_cv": _json_float(spikes.within_burst_isi_cv),
        "mean_interburst_interval": _json_float(spikes.mean_interburst_interval),
        "interburst_interval_cv": _json_float(spikes.interburst_interval_cv),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "notes": spikes.notes,
    }


def run_dense_sweep(*, transient_steps: int, retained_samples: int) -> list[dict[str, Any]]:
    grid = config.build_sweep_grid()
    known = {round(float(c), 9) for c in config.KNOWN_CURRENTS}
    records: list[dict[str, Any]] = []
    for i, current in enumerate(grid):
        spikes, peak_x, x_min, x_max = simulate_and_measure(
            float(current),
            transient_steps=transient_steps,
            retained_samples=retained_samples,
        )
        records.append(
            _record_from_analysis(
                float(current),
                spikes,
                peak_x,
                x_min,
                x_max,
                is_known_current=round(float(current), 9) in known,
            )
        )
        if (i + 1) % 25 == 0 or (i + 1) == len(grid):
            print(f"dense sweep: {i + 1}/{len(grid)} currents done", flush=True)
    return records


def run_lyapunov_subset(
    dense_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    provisional = {
        rec["current"]: provisional_label_from_spikes(rec) for rec in dense_records
    }
    selected = select_lyapunov_currents(
        dense_records,
        known_currents=config.KNOWN_CURRENTS,
        max_extra=config.MAX_EXTRA_LYAPUNOV_CURRENTS,
        provisional_labels=provisional,
    )
    results: list[dict[str, Any]] = []
    for i, current in enumerate(selected):
        analysis = estimate_lyapunov(current)
        results.append(
            {
                "current": float(current),
                "largest_lyapunov_exponent": _json_float(analysis.exponent),
                "classification": analysis.classification,
                "convergence": analysis.convergence,
                "convergence_tolerance": _json_float(analysis.convergence_tolerance),
            }
        )
        print(
            f"lyapunov subset: {i + 1}/{len(selected)} currents done "
            f"(I={current:.6g}, exponent={analysis.exponent:.6g}, "
            f"{analysis.classification}, {analysis.convergence})",
            flush=True,
        )
    return results


def _write_json(path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_pilot() -> None:
    """Tiny mechanics check: a handful of currents, a short window.

    Verifies the pipeline executes end to end without error. It is not a
    scientific result and must never be mistaken for the production sweep.
    """
    pilot_currents = [1.0, 1.67, 2.5, 3.20, 4.0]
    transient_steps = 2_000
    retained_samples = 5_000
    t0 = time.time()
    records = []
    for current in pilot_currents:
        spikes, peak_x, x_min, x_max = simulate_and_measure(
            current, transient_steps=transient_steps, retained_samples=retained_samples
        )
        records.append(
            _record_from_analysis(
                current, spikes, peak_x, x_min, x_max, is_known_current=False
            )
        )
    lyapunov_probe = estimate_lyapunov(3.20)
    payload = {
        "kind": "pilot",
        "currents": pilot_currents,
        "transient_steps": transient_steps,
        "retained_samples": retained_samples,
        "records": records,
        "lyapunov_probe_current": 3.20,
        "lyapunov_probe_exponent": _json_float(lyapunov_probe.exponent),
        "elapsed_seconds": time.time() - t0,
        "git_head": _git_head(),
    }
    _write_json(config.RESULTS_DIR / "pilot_report.json", payload)
    print(f"Pilot complete in {payload['elapsed_seconds']:.1f}s. No production result written.")


def run_production() -> None:
    t0 = time.time()
    grid = config.build_sweep_grid()
    print(
        f"Dense sweep: {len(grid)} currents in "
        f"[{config.SWEEP_MIN_CURRENT}, {config.SWEEP_MAX_CURRENT}], "
        f"transient={config.TRANSIENT_STEPS}, retained={config.RETAINED_SAMPLES}",
        flush=True,
    )
    dense_records = run_dense_sweep(
        transient_steps=config.TRANSIENT_STEPS,
        retained_samples=config.RETAINED_SAMPLES,
    )
    dense_elapsed = time.time() - t0
    _write_json(
        config.DENSE_SWEEP_PATH,
        {
            "kind": "dense_sweep",
            "sweep_min_current": config.SWEEP_MIN_CURRENT,
            "sweep_max_current": config.SWEEP_MAX_CURRENT,
            "sweep_count_requested": config.SWEEP_COUNT,
            "sweep_count_actual": len(grid),
            "known_currents": list(config.KNOWN_CURRENTS),
            "transient_steps": config.TRANSIENT_STEPS,
            "retained_samples": config.RETAINED_SAMPLES,
            "dt": config.DT,
            "hr_parameters": asdict(config.HR_PARAMETERS),
            "initial_state": list(config.INITIAL_STATE),
            "records": dense_records,
            "elapsed_seconds": dense_elapsed,
            "git_head": _git_head(),
        },
    )
    print(f"Dense sweep complete in {dense_elapsed:.1f}s -> {config.DENSE_SWEEP_PATH}")

    t1 = time.time()
    lyapunov_records = run_lyapunov_subset(dense_records)
    lyapunov_elapsed = time.time() - t1
    _write_json(
        config.RESULTS_DIR / "lyapunov_results.json",
        {
            "kind": "lyapunov_subset",
            "max_extra_currents": config.MAX_EXTRA_LYAPUNOV_CURRENTS,
            "selected_count": len(lyapunov_records),
            "records": lyapunov_records,
            "elapsed_seconds": lyapunov_elapsed,
            "git_head": _git_head(),
        },
    )
    print(
        f"Lyapunov subset complete ({len(lyapunov_records)} currents) in "
        f"{lyapunov_elapsed:.1f}s -> {config.RESULTS_DIR / 'lyapunov_results.json'}"
    )
    print(f"Total elapsed: {time.time() - t0:.1f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="Tiny mechanics check only.")
    mode.add_argument("--run", action="store_true", help="Full production dense sweep.")
    args = parser.parse_args(argv)

    if args.pilot:
        run_pilot()
    else:
        run_production()
    return 0


if __name__ == "__main__":
    sys.exit(main())
