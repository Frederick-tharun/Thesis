"""Step 1F: assemble and freeze frozen_design.json.

Combines the Step-1 regime-map evidence (chapter2/part2_regime_map/results)
with the Step-1 (this package) targeted Lyapunov diagnostics to select final
training/test currents, then writes a hash-locked design file. After this
file exists, no later stage may change the selected currents.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from chapter2.part2_regime_map.analyse_regimes import final_classification

from . import config

PROTOCOL_VERSION = "part2_cross_regime_design_v1"

REGIME_MAP_RESULTS = config.PROJECT_ROOT / "chapter2/part2_regime_map/results"
EVIDENCE_FILES = [
    REGIME_MAP_RESULTS / "dense_sweep_results.json",
    REGIME_MAP_RESULTS / "lyapunov_results.json",
    REGIME_MAP_RESULTS / "regime_summary.json",
    config.DESIGN_DIR / "targeted_lyapunov_diagnostics.json",
]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def evidence_hash() -> dict:
    return {str(p.relative_to(config.PROJECT_ROOT)): _file_sha256(p) for p in EVIDENCE_FILES}


def _load_dense():
    dense = json.loads((REGIME_MAP_RESULTS / "dense_sweep_results.json").read_text())["records"]
    lyap = json.loads((REGIME_MAP_RESULTS / "lyapunov_results.json").read_text())["records"]
    targeted = json.loads(
        (config.DESIGN_DIR / "targeted_lyapunov_diagnostics.json").read_text()
    )["records"]
    lyap_by_I: dict[float, dict] = {round(r["current"], 6): r for r in lyap}
    for r in targeted:
        lyap_by_I[round(r["current"], 6)] = {
            "current": r["current"],
            "largest_lyapunov_exponent": r["lyapunov_exponent"],
            "classification": r["lyapunov_classification"],
            "convergence": r["lyapunov_convergence"],
        }
    return dense, lyap_by_I


def _nearest(dense, target):
    return min(dense, key=lambda r: abs(r["current"] - target))


def _evidence(dense, lyap_by_I, target, role, reason, interp_status):
    rec = _nearest(dense, target)
    key = round(rec["current"], 6)
    ly = lyap_by_I.get(key)
    cls = final_classification(rec, ly)
    return {
        "current": rec["current"],
        "role": role,
        "regime_label": cls,
        "is_previously_known_current": rec["is_known_current"],
        "spike_count": rec["spike_count"],
        "burst_structure": rec["burst_structure"],
        "mean_spikes_per_burst": rec["mean_spikes_per_burst"],
        "isi_cv": rec["isi_cv"],
        "within_burst_isi_cv": rec["within_burst_isi_cv"],
        "interburst_interval_cv": rec["interburst_interval_cv"],
        "lyapunov": (
            {
                "exponent": ly["largest_lyapunov_exponent"],
                "classification": ly["classification"],
                "convergence": ly["convergence"],
            }
            if ly
            else None
        ),
        "selection_reason": reason,
        "interpolation_status": interp_status,
    }


def build() -> dict:
    dense, lyap_by_I = _load_dense()
    ev = lambda *a, **k: _evidence(dense, lyap_by_I, *a, **k)

    regular_training = [
        ev(1.42, "regular_training", "Interior low-I periodic spiking (tonic), away from the quiescent boundary (~1.30) and the spiking->bursting onset (~1.56).", "n/a (training)"),
        ev(1.67, "regular_training", "Professor-anchor current; previously studied (Chapter 2 5-current study); clean periodic bursting, 2 spikes/burst.", "n/a (training)"),
        ev(1.90, "regular_training", "Second, deeper-interior low-I periodic-bursting point for within-flavor diversity.", "n/a (training)"),
        ev(3.43, "regular_training", "Interior high-I periodic bursting (2 spikes/burst), deep inside the narrow post-chaos bursting band.", "n/a (training)"),
        ev(3.50, "regular_training", "Professor-anchor current; previously studied; clean tonic periodic spiking.", "n/a (training)"),
        ev(3.80, "regular_training", "Deep-interior high-I periodic spiking, far from any boundary.", "n/a (training)"),
    ]

    chaotic_training = [
        ev(2.985, "chaotic_training", "Lowest CONFIRMED (Lyapunov positive, converged) chaotic-bursting current found anywhere in the evidence gathered (see negative_evidence_2p1_to_2p9 below for the extensive band ruled OUT below this).", "n/a (training)"),
        ev(3.20, "chaotic_training", "Professor-anchor current; confirmed chaotic bursting; brackets I=3.29 from BELOW (distance 0.09).", "n/a (training)"),
        ev(3.34, "chaotic_training", "Professor-anchor current; confirmed chaotic bursting; brackets I=3.29 from ABOVE (distance 0.05).", "n/a (training)"),
        ev(3.368, "chaotic_training", "Confirmed chaotic bursting, extends the training span toward the upper edge of the confirmed band.", "n/a (training)"),
        ev(3.391, "chaotic_training", "Confirmed CHAOTIC SPIKING (not bursting) -- a second dynamical flavor within the same confirmed band.", "n/a (training)"),
    ]

    rr_test = [
        ev(1.80, "RR_test", "Interior low-I periodic bursting, strictly between training currents 1.67 and 1.90 (true interpolation); reused as a CR-test target too (see CR notes).", "interpolation (between 1.67 and 1.90)"),
        ev(2.00, "RR_test", "Interior low-I periodic bursting near the upper edge of the low-I regular region; within the regular model's global training range.", "interpolation (within global training range [1.42,3.80])"),
        ev(3.65, "RR_test", "Interior high-I periodic spiking, strictly between training currents 3.50 and 3.80 (true interpolation).", "interpolation (between 3.50 and 3.80)"),
    ]

    shared_chaotic_test = [
        ev(3.376, "RC_and_CC_test", "LOW-within-band confirmed chaotic bursting, interpolation between chaotic-training anchors 3.368 and 3.391.", "interpolation for both models"),
        ev(3.383, "RC_and_CC_test", "MID-within-band confirmed CHAOTIC SPIKING, interpolation between 3.368 and 3.391; distinct flavor from the bursting tests.", "interpolation for both models"),
        ev(3.398, "RC_and_CC_test", "HIGH-within-band confirmed chaotic bursting, one grid step (0.007) past the chaotic-training upper anchor 3.391.", "interpolation for the regular model; mild (~1 grid step) extrapolation for the chaotic model"),
    ]

    cr_test = [
        ev(3.29, "CR_test", "STAR CANDIDATE / MAIN INTERPOLATION TEST. Converged, near-zero Lyapunov (0.000686) confirms non-chaotic; sits strictly between chaotic-training anchors 3.20 (below, 0.09) and 3.34 (above, 0.05): genuine interpolation for the chaotic-trained model.", "interpolation (strictly between 3.20 and 3.34)"),
        ev(3.44, "CR_test", "MAIN INTERPOLATION TEST (secondary). Regular (periodic bursting) current immediately above the chaotic training band's upper anchor (3.391), distance 0.05.", "mild extrapolation (nearest regular point above the chaotic training span)"),
        ev(1.80, "CR_test", "EXTRAPOLATION STRESS TEST (intentional). Deep low-I periodic bursting, far from the chaotic training band [2.985,3.391] (distance ~1.19); characterizes accuracy degradation with extrapolation distance. Reused from the RR-test pool.", "extrapolation (deep, intentional stress test)"),
    ]

    negative_evidence_2p1_to_2p9 = {
        "investigated_currents": [2.15, 2.30, 2.45, 2.50, 2.62, 2.68, 2.75, 2.90],
        "finding": (
            "All eight targeted Lyapunov diagnostics across I in [2.15,2.90] "
            "returned 'near zero' (six points, five converged) or 'uncertain, "
            "not_converged' (three points) -- NONE returned a confirmed "
            "positive, converged exponent. This directly contradicts the "
            "regime map's coarse 'chaotic_candidate' labeling for this range, "
            "which was based only on spike/burst irregularity (high ISI-CV, "
            "non-integer spikes-per-burst), not on Lyapunov confirmation. "
            "Per Step-1 instructions, irregular burst statistics alone are "
            "descriptive evidence, not proof of chaos."
        ),
        "conclusion": (
            "The confirmed-chaotic band in this HR parameter set is "
            "considerably narrower than assumed: only I in approximately "
            "[2.985,3.398] has directly Lyapunov-confirmed positive, "
            "converged chaos anywhere in the evidence gathered (Step-1 "
            "regime-map sparse subset plus these 8 new targeted points). "
            "The chaotic training/test currents below are therefore drawn "
            "entirely from this narrower, honestly-confirmed band rather "
            "than being artificially spread across the originally assumed "
            "wider [2.08,3.40] range."
        ),
        "raw_diagnostic_file": "results/design/targeted_lyapunov_diagnostics.json",
    }

    window_2_56_investigation = {
        "window_currents_confirmed_regular": [2.5564, 2.5639, 2.5714],
        "bracketing_points_tested": {"below": [2.50], "above": [2.62]},
        "bracketing_result": (
            "Both 2.50 and 2.62 (immediately below/above the window) returned "
            "non-chaotic Lyapunov results (uncertain/not_converged and near "
            "zero/converged respectively). Combined with the pre-existing "
            "evidence that the nearest confirmed-chaotic point (2.985) is "
            "0.41-0.42 away, this window has NO confirmed chaotic bracketing "
            "on either side within the evidence gathered."
        ),
        "decision": "NOT selected as a CR test (would be wide, one-sided extrapolation, not interpolation).",
    }

    design = {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "step1_evidence_file_hashes": evidence_hash(),
        "professor_anchor_currents": {
            "regular": [1.67, 3.50],
            "chaotic": [3.20, 3.34],
            "special_window": 3.29,
            "represented_as_intended": True,
        },
        "models": {
            "regular_trained": {
                "training_currents": regular_training,
                "rr_test_currents": rr_test,
                "rc_test_currents": shared_chaotic_test,
            },
            "chaotic_trained": {
                "training_currents": chaotic_training,
                "cc_test_currents": shared_chaotic_test,
                "cr_test_currents": cr_test,
            },
        },
        "negative_evidence_2p1_to_2p9_band": negative_evidence_2p1_to_2p9,
        "window_2_56_2_58_investigation": window_2_56_investigation,
        "i_3_29_withheld_from_chaotic_training": True,
        "i_3_29_withholding_rationale": (
            "I=3.29 is a deliberate test-design decision, not a silent "
            "reinterpretation: it is excluded from chaotic_training (never "
            "fitted, never used for scaling/optimisation/LOCO) specifically "
            "so it can serve as the star CR interpolation test."
        ),
        "no_esn_trained_for_this_design": True,
    }
    return design


def freeze() -> tuple[dict, str]:
    design = build()
    canonical = json.dumps(design, indent=2, sort_keys=True, allow_nan=False)
    design_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    design["design_hash_sha256"] = design_hash
    final_text = json.dumps(design, indent=2, sort_keys=True, allow_nan=False) + "\n"
    config.DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    config.FROZEN_DESIGN_PATH.write_text(final_text, encoding="utf-8")
    return design, design_hash


if __name__ == "__main__":
    design, design_hash = freeze()
    print(f"wrote {config.FROZEN_DESIGN_PATH}")
    print(f"design_hash_sha256={design_hash}")
