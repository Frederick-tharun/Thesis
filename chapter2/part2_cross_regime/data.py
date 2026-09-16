"""Step 2: fixed-current dataset generation, manifest, and final-test firewall.

Every trajectory is produced only by the frozen, unmodified
``chapter2.hr_data_ch2.simulate_fixed_current``. This module never alters the
simulator and never overwrites a Part-1 trajectory file under
``chapter2/outputs/data/``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Sequence

import numpy as np

from chapter2.esn_data import (
    FixedCurrentTrajectory,
    OneStepPairs,
    PreparedOptimisationTrajectory,
    StateCurrentScalers,
    ValidationWindowView,
    NumpyStandardScaler,
    create_one_step_pairs,
)
from chapter2.esn_config import TransitionRange
from chapter2.hr_data_ch2 import (
    HRTrajectory,
    load_trajectory_npz,
    save_trajectory_npz,
    simulate_fixed_current,
)

from . import config


class FinalTestAccessError(PermissionError):
    """Raised when non-final-evaluation code tries to open a locked final-test file."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_slug(current: float) -> str:
    return f"{current:.6f}".replace(".", "p").replace("-", "m")


def role_directory(role: str) -> Path:
    if role in ("regular_training", "chaotic_training_regular_side"):
        return config.DATA_SOURCE_REGULAR
    if role == "chaotic_training":
        return config.DATA_SOURCE_CHAOTIC
    if role.endswith("_test") or "test" in role:
        return config.DATA_FINAL_TEST_LOCKED
    raise ValueError(f"unrecognised role: {role}")


def trajectory_path(role: str, current: float) -> Path:
    return role_directory(role) / f"fixed_I_{current_slug(current)}.npz"


def hr_simulator_source_hash() -> str:
    return file_sha256(config.PACKAGE_ROOT.parent / "hr_data_ch2.py")


def generate_trajectory(current: float, role: str, *, overwrite: bool = False) -> Path:
    """Simulate and save one fixed-current trajectory using the frozen simulator."""
    path = trajectory_path(role, current)
    if path.exists() and not overwrite:
        return path
    trajectory = simulate_fixed_current(
        current,
        retained_samples=config.RETAINED_SAMPLES_PER_CURRENT,
        transient_steps=config.INITIAL_TRANSIENT_STEPS,
        initial_state=config.INITIAL_STATE,
        dt=config.DT,
        parameters=config.HR_PARAMETERS,
    )
    if not np.all(np.isfinite(trajectory.as_matrix())):
        raise ValueError(f"non-finite trajectory generated at I={current}")
    save_trajectory_npz(path, trajectory)
    return path


def _guarded_load_npz(path: Path, *, allow_final_test: bool) -> HRTrajectory:
    resolved = path.resolve()
    if config.DATA_FINAL_TEST_LOCKED.resolve() in resolved.parents and not allow_final_test:
        raise FinalTestAccessError(
            f"refused to load locked final-test trajectory outside the final-"
            f"evaluation entry point: {path}"
        )
    return load_trajectory_npz(path)


def load_source_trajectory(role: str, current: float) -> FixedCurrentTrajectory:
    """Load a regular/chaotic *source* (training) trajectory. Never final-test."""
    path = trajectory_path(role, current)
    traj = _guarded_load_npz(path, allow_final_test=False)
    return FixedCurrentTrajectory(
        current=current, time=traj.t, states=traj.state, current_values=traj.I, path=path
    )


def load_final_test_trajectory(current: float, *, caller_token: str) -> FixedCurrentTrajectory:
    """Load a locked final-test trajectory. Requires the explicit access token.

    ``caller_token`` must equal the module-level sentinel
    ``FINAL_EVALUATION_ENTRY_POINT_TOKEN``; every other caller is refused.
    This is an explicit, auditable guard, not a security sandbox.
    """
    if caller_token != FINAL_EVALUATION_ENTRY_POINT_TOKEN:
        raise FinalTestAccessError(
            "final-test trajectories may only be loaded from the dedicated "
            "future final-evaluation entry point"
        )
    path = trajectory_path("final_test", current)
    traj = _guarded_load_npz(path, allow_final_test=True)
    return FixedCurrentTrajectory(
        current=current, time=traj.t, states=traj.state, current_values=traj.I, path=path
    )


FINAL_EVALUATION_ENTRY_POINT_TOKEN = "PART2_FINAL_EVALUATION_STEP5_NOT_YET_IMPLEMENTED"


def prepare_optimisation_trajectory(
    fixed: FixedCurrentTrajectory,
    *,
    fitting_stop: int = config.FITTING_TRANSITIONS_STOP,
    validation_windows=config.VALIDATION_WINDOWS,
) -> PreparedOptimisationTrajectory:
    """Build fitting + validation-window views, reusing Part-1's exact window shape."""
    fitting = create_one_step_pairs(fixed, TransitionRange(0, fitting_stop), include_current=True)
    views = []
    for window in validation_windows:
        warmup = create_one_step_pairs(fixed, window.warmup, include_current=True)
        scored = create_one_step_pairs(fixed, window.scored, include_current=True)
        views.append(ValidationWindowView(window, warmup, scored))
    return PreparedOptimisationTrajectory(fixed.current, fitting, tuple(views))


def load_frozen_design() -> dict:
    return json.loads(config.FROZEN_DESIGN_PATH.read_text(encoding="utf-8"))


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def generate_all_datasets(design: dict) -> list[dict]:
    """Generate every trajectory named in the frozen design. Returns manifest rows."""
    manifest_rows: list[dict] = []
    simulator_hash = hr_simulator_source_hash()

    def _record(role: str, current: float, regime_label: str) -> None:
        path = generate_trajectory(current, "chaotic_training" if role == "chaotic_training" else (
            "regular_training" if role == "regular_training" else "final_test"
        ))
        digest = file_sha256(path)
        with np.load(path, allow_pickle=False) as arrays:
            n_samples = len(arrays["t"])
            finite = all(np.all(np.isfinite(arrays[k])) for k in ("t", "x", "y", "z", "I"))
        manifest_rows.append(
            {
                "current": current,
                "regime_label": regime_label,
                "role": role,
                "path": str(path.relative_to(config.PROJECT_ROOT)),
                "n_samples": n_samples,
                "dt": config.DT,
                "transient_steps": config.INITIAL_TRANSIENT_STEPS,
                "retained_samples": config.RETAINED_SAMPLES_PER_CURRENT,
                "initial_state": list(config.INITIAL_STATE),
                "hr_parameters": {
                    "a": config.HR_PARAMETERS.a, "b": config.HR_PARAMETERS.b,
                    "c": config.HR_PARAMETERS.c, "d": config.HR_PARAMETERS.d,
                    "r": config.HR_PARAMETERS.r, "s": config.HR_PARAMETERS.s,
                    "x_r": config.HR_PARAMETERS.x_r,
                },
                "sha256": digest,
                "simulator_source_sha256": simulator_hash,
                "all_finite": bool(finite),
            }
        )

    for entry in design["models"]["regular_trained"]["training_currents"]:
        _record("regular_training", entry["current"], entry["regime_label"])
    for entry in design["models"]["chaotic_trained"]["training_currents"]:
        _record("chaotic_training", entry["current"], entry["regime_label"])

    final_currents: dict[float, str] = {}
    for entry in design["models"]["regular_trained"]["rr_test_currents"]:
        final_currents[entry["current"]] = entry["regime_label"]
    for entry in design["models"]["regular_trained"]["rc_test_currents"]:
        final_currents[entry["current"]] = entry["regime_label"]
    for entry in design["models"]["chaotic_trained"]["cc_test_currents"]:
        final_currents[entry["current"]] = entry["regime_label"]
    for entry in design["models"]["chaotic_trained"]["cr_test_currents"]:
        final_currents[entry["current"]] = entry["regime_label"]
    for current, label in final_currents.items():
        _record("final_test", current, label)

    return manifest_rows


def write_manifest(rows: list[dict], design_hash: str) -> None:
    payload = {
        "kind": "part2_cross_regime_data_manifest",
        "design_hash": design_hash,
        "git_head": _git_head(),
        "trajectories": rows,
    }
    config.DATA_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    design = load_frozen_design()
    design_hash = design["design_hash_sha256"]
    print(f"generating datasets for design_hash={design_hash}")
    rows = generate_all_datasets(design)
    write_manifest(rows, design_hash)
    print(f"wrote {len(rows)} trajectory records to {config.DATA_MANIFEST_PATH}")
    for row in rows:
        print(f"  I={row['current']:.6f} role={row['role']:18s} n={row['n_samples']} finite={row['all_finite']}")


if __name__ == "__main__":
    main()
