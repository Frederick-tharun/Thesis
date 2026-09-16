"""Classify the Part 2 Step 1 dense sweep and build the bifurcation figure.

This module never re-simulates the Hindmarsh-Rose system and never invents a
new Lyapunov estimator. It only combines, per current:

* the spike/burst diagnostics already computed by
  ``chapter2.dynamics_analysis_ch2.analyze_spikes_and_bursts`` (dense grid,
  every current), and
* the Lyapunov exponent already computed by the unmodified, previously
  validated ``chapter2.dynamics_analysis_ch2.estimate_lyapunov`` /
  ``scripts.analysis.estimate_hr_lyapunov`` Benettin estimator (sparse
  subset only, see ``config.MAX_EXTRA_LYAPUNOV_CURRENTS``).

The five-category decision cascade below mirrors
``chapter2.dynamics_analysis_ch2.preliminary_regime`` exactly (same evidence,
same imported thresholds). It adds two presentational refinements that
module does not need: a distinct "chaotic spiking" label for a positive,
converged Lyapunov exponent on a non-bursting (tonic) trajectory, and a
"quiescent (no spiking)" label for a settled trajectory with zero detected
spikes. Neither is a new scientific method; both only relabel the same
underlying evidence. Wherever no Lyapunov estimate exists for a current, the
function returns "transition/uncertain" for any point it cannot confidently
classify from spike/burst structure alone, rather than guessing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chapter2.config_ch2 import (
    REGULAR_INTERBURST_INTERVAL_CV_MAX,
    REGULAR_ISI_CV_MAX,
    REGULAR_SPIKES_PER_BURST_CV_MAX,
    REGULAR_WITHIN_BURST_ISI_CV_MAX,
)

from . import config

QUIESCENT = "quiescent (no spiking)"
PERIODIC_BURSTING = "periodic bursting"
PERIODIC_SPIKING = "periodic spiking"
CHAOTIC_BURSTING = "chaotic bursting"
CHAOTIC_SPIKING = "chaotic spiking"
TRANSITION_UNCERTAIN = "transition/uncertain"

# Previously reported labels, transcribed unchanged from
# chapter2/outputs/dynamics_summary.md ("preliminary_regime" column). Used
# only for the Step 8 side-by-side comparison; never modified or reused as
# ground truth.
PREVIOUS_CLASSIFICATION = {
    1.67: "periodic bursting",
    3.20: "chaotic bursting",
    3.29: "uncertain",
    3.34: "chaotic bursting",
    3.50: "periodic spiking",
}


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _burst_regularity(record: Mapping[str, Any]) -> tuple[bool, bool]:
    """Direct port of chapter2.dynamics_analysis_ch2._burst_regularity for dicts."""
    if record["burst_structure"] != "bursting":
        return False, False
    mean_spb = record["mean_spikes_per_burst"]
    std_spb = record["std_spikes_per_burst"]
    if not _finite(mean_spb) or mean_spb == 0:
        return False, False
    spikes_per_burst_cv = std_spb / mean_spb
    measures = (
        (spikes_per_burst_cv, REGULAR_SPIKES_PER_BURST_CV_MAX),
        (record["within_burst_isi_cv"], REGULAR_WITHIN_BURST_ISI_CV_MAX),
        (record["interburst_interval_cv"], REGULAR_INTERBURST_INTERVAL_CV_MAX),
    )
    if not all(_finite(value) for value, _ in measures):
        return False, False
    regular = all(value <= limit for value, limit in measures)
    irregular = any(value > limit for value, limit in measures)
    return regular, irregular


def irregularity_evidence(record: Mapping[str, Any]) -> bool:
    """Descriptive-only flag: does this current show irregular ISI/burst structure?

    Not a classification by itself (see module docstring); used only to
    propose candidate chaotic regions in ``propose_candidate_regions`` when
    no Lyapunov estimate was computed at a given grid point.
    """
    _, burst_irregular = _burst_regularity(record)
    if burst_irregular:
        return True
    if record["burst_structure"] == "uncertain" and _finite(record["isi_cv"]):
        return record["isi_cv"] > REGULAR_ISI_CV_MAX
    return False


def final_classification(
    record: Mapping[str, Any], lyapunov: Mapping[str, Any] | None
) -> str:
    """Classify one current from spike/burst evidence and an optional Lyapunov entry."""
    if record["spike_count"] == 0:
        return QUIESCENT

    burst_structure = record["burst_structure"]
    isi_cv = record["isi_cv"]
    tonic_regular = (
        burst_structure == "tonic" and _finite(isi_cv) and isi_cv <= REGULAR_ISI_CV_MAX
    )
    burst_regular, _ = _burst_regularity(record)

    reliable_positive = (
        lyapunov is not None
        and lyapunov["convergence"] == "converged"
        and lyapunov["classification"] == "positive"
    )
    weak_positive = lyapunov is not None and lyapunov["classification"] == "weak positive"

    if reliable_positive:
        return CHAOTIC_BURSTING if burst_structure == "bursting" else CHAOTIC_SPIKING
    if burst_regular and not weak_positive:
        return PERIODIC_BURSTING
    if tonic_regular and not weak_positive:
        return PERIODIC_SPIKING
    return TRANSITION_UNCERTAIN


def provisional_label_from_spikes(record: Mapping[str, Any]) -> str:
    """Classification using only spike/burst evidence (no Lyapunov available yet).

    Used exclusively to decide which currents most need a Lyapunov estimate
    (see ``select_lyapunov_currents``); never written as a final label.
    """
    return final_classification(record, lyapunov=None)


def _dedup_sorted(values: Sequence[float], *, tol: float = 1e-9) -> list[float]:
    ordered = sorted(float(v) for v in values)
    out: list[float] = []
    for value in ordered:
        if not out or value - out[-1] > tol:
            out.append(value)
    return out


def select_lyapunov_currents(
    dense_records: Sequence[Mapping[str, Any]],
    *,
    known_currents: Sequence[float],
    max_extra: int,
    provisional_labels: Mapping[float, str],
) -> list[float]:
    """Pick currents for the expensive Lyapunov estimate.

    Always includes every previously studied current. Beyond that, it adds
    (a) both currents on either side of every provisional-label change
    (boundary probes) and (b) one representative current from the middle of
    every other contiguous same-label run (excluding quiescent runs, which
    are unambiguous fixed points and do not need confirmation), bounded to
    ``max_extra`` additional currents.
    """
    ordered = sorted(dense_records, key=lambda rec: rec["current"])
    known = set(_dedup_sorted(known_currents))

    boundary_candidates: list[float] = []
    for left, right in zip(ordered, ordered[1:]):
        if provisional_labels[left["current"]] != provisional_labels[right["current"]]:
            boundary_candidates.append(left["current"])
            boundary_candidates.append(right["current"])

    representative_candidates: list[float] = []
    run_start = 0
    for i in range(1, len(ordered) + 1):
        at_end = i == len(ordered)
        changed = (
            not at_end
            and provisional_labels[ordered[i]["current"]]
            != provisional_labels[ordered[run_start]["current"]]
        )
        if at_end or changed:
            label = provisional_labels[ordered[run_start]["current"]]
            if label != QUIESCENT:
                mid = run_start + (i - run_start) // 2
                representative_candidates.append(ordered[mid]["current"])
            run_start = i

    extras: list[float] = []
    for current in boundary_candidates + representative_candidates:
        if len(extras) >= max_extra:
            break
        if any(abs(current - k) < 1e-9 for k in known):
            continue
        if any(abs(current - e) < 1e-9 for e in extras):
            continue
        extras.append(float(current))

    return _dedup_sorted(list(known) + extras)


def _coarse_category(label: str, evidence_irregular: bool) -> str:
    if label == QUIESCENT:
        return "quiescent"
    if label in (PERIODIC_BURSTING, PERIODIC_SPIKING):
        return "regular"
    if label in (CHAOTIC_BURSTING, CHAOTIC_SPIKING):
        return "chaotic_confirmed"
    if label == TRANSITION_UNCERTAIN and evidence_irregular:
        return "chaotic_candidate"
    return "uncertain"


def propose_candidate_regions(
    per_current: Sequence[Mapping[str, Any]], *, min_span: float = 0.075
) -> dict[str, Any]:
    """Segment the classified sweep into candidate regular/chaotic/transition runs.

    This is descriptive region-finding for later train/test-current
    selection; it does not choose or lock any current for training.
    """
    ordered = sorted(per_current, key=lambda rec: rec["current"])
    coarse = [
        _coarse_category(rec["classification"], rec["irregularity_evidence"])
        for rec in ordered
    ]

    segments: list[dict[str, Any]] = []
    run_start = 0
    for i in range(1, len(ordered) + 1):
        if i == len(ordered) or coarse[i] != coarse[run_start]:
            start_I = ordered[run_start]["current"]
            end_I = ordered[i - 1]["current"]
            if end_I - start_I >= min_span:
                labels = sorted({ordered[j]["classification"] for j in range(run_start, i)})
                segments.append(
                    {
                        "category": coarse[run_start],
                        "current_start": start_I,
                        "current_end": end_I,
                        "observed_labels": labels,
                        "n_grid_points": i - run_start,
                    }
                )
            run_start = i

    regular_regions = [s for s in segments if s["category"] == "regular"]
    chaotic_regions = [
        s for s in segments if s["category"] in ("chaotic_confirmed", "chaotic_candidate")
    ]
    transition_regions = [s for s in segments if s["category"] == "uncertain"]

    def _interior_candidates(region: Mapping[str, Any], n: int = 2) -> list[float]:
        start, end = region["current_start"], region["current_end"]
        margin = (end - start) * 0.15
        lo, hi = start + margin, end - margin
        if hi <= lo:
            return [round((start + end) / 2, 4)]
        return [round(float(v), 4) for v in np.linspace(lo, hi, n)]

    candidate_unseen_currents = sorted(
        {
            value
            for region in regular_regions + chaotic_regions
            for value in _interior_candidates(region)
        }
    )

    return {
        "regular_regions": regular_regions,
        "chaotic_regions": chaotic_regions,
        "transition_regions": transition_regions,
        "candidate_unseen_currents": candidate_unseen_currents,
        "min_span_current": min_span,
        "note": (
            "chaotic_candidate segments have irregular burst/ISI evidence but no "
            "converged positive Lyapunov estimate at every point in the segment; "
            "only chaotic_confirmed segments and individual per_current entries "
            "with a non-null lyapunov_exponent are Lyapunov-confirmed."
        ),
    }


def build_bifurcation_figure(dense_records: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    currents: list[float] = []
    peaks: list[float] = []
    for record in dense_records:
        for peak in record["spike_peak_x"]:
            currents.append(record["current"])
            peaks.append(peak)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(currents, peaks, s=0.6, c="#222222", alpha=0.5, linewidths=0)
    for known in config.KNOWN_CURRENTS:
        ax.axvline(known, color="#D55E00", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("External current $I$")
    ax.set_ylabel("Settled spike-peak $x$")
    ax.set_title("Hindmarsh–Rose bifurcation map (spike peaks of $x$)")
    ax.set_xlim(config.SWEEP_MIN_CURRENT, config.SWEEP_MAX_CURRENT)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def build_regime_summary(
    dense_payload: Mapping[str, Any], lyapunov_payload: Mapping[str, Any]
) -> dict[str, Any]:
    dense_records = dense_payload["records"]
    lyapunov_by_current = {
        rec["current"]: rec for rec in lyapunov_payload["records"]
    }

    per_current: list[dict[str, Any]] = []
    for record in dense_records:
        lyapunov = lyapunov_by_current.get(record["current"])
        classification = final_classification(record, lyapunov)
        per_current.append(
            {
                "current": record["current"],
                "is_known_current": record["is_known_current"],
                "classification": classification,
                "irregularity_evidence": irregularity_evidence(record),
                "spike_count": record["spike_count"],
                "isi_cv": record["isi_cv"],
                "burst_structure": record["burst_structure"],
                "x_min": record["x_min"],
                "x_max": record["x_max"],
                "lyapunov_exponent": (
                    lyapunov["largest_lyapunov_exponent"] if lyapunov else None
                ),
                "lyapunov_classification": lyapunov["classification"] if lyapunov else None,
                "lyapunov_convergence": lyapunov["convergence"] if lyapunov else None,
            }
        )

    known_comparison = []
    for known in config.KNOWN_CURRENTS:
        match = next(
            rec for rec in per_current if abs(rec["current"] - known) < 1e-9
        )
        known_comparison.append(
            {
                "current": known,
                "new_classification": match["classification"],
                "previous_classification": PREVIOUS_CLASSIFICATION.get(known),
                "lyapunov_exponent": match["lyapunov_exponent"],
                "lyapunov_classification": match["lyapunov_classification"],
            }
        )

    return {
        "metadata": {
            "sweep_min_current": dense_payload["sweep_min_current"],
            "sweep_max_current": dense_payload["sweep_max_current"],
            "sweep_count_actual": dense_payload["sweep_count_actual"],
            "transient_steps": dense_payload["transient_steps"],
            "retained_samples": dense_payload["retained_samples"],
            "dt": dense_payload["dt"],
            "hr_parameters": dense_payload["hr_parameters"],
            "initial_state": dense_payload["initial_state"],
            "known_currents": dense_payload["known_currents"],
            "lyapunov_currents_evaluated": sorted(lyapunov_by_current.keys()),
        },
        "per_current": per_current,
        "known_current_comparison": known_comparison,
        "candidate_regions": propose_candidate_regions(per_current),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    dense_payload = json.loads(config.DENSE_SWEEP_PATH.read_text(encoding="utf-8"))
    lyapunov_payload = json.loads(
        (config.RESULTS_DIR / "lyapunov_results.json").read_text(encoding="utf-8")
    )
    summary = build_regime_summary(dense_payload, lyapunov_payload)

    if args.check_only:
        print("check-only: summary computed, nothing written")
    else:
        config.REGIME_SUMMARY_PATH.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        build_bifurcation_figure(dense_payload["records"], config.FIGURE_PATH)
        print(f"Wrote {config.REGIME_SUMMARY_PATH}")
        print(f"Wrote {config.FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
