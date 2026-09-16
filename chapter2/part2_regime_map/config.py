"""Locked configuration for the Part 2 Step 1 HR regime map.

All Hindmarsh-Rose parameters, the integration step, the initial state, the
transient/retained sample counts, and the spike/burst/Lyapunov thresholds are
imported unchanged from the frozen ``chapter2.config_ch2`` module. This
package defines only new, Part-2-specific settings: the dense sweep range and
resolution, and the policy for selecting the small subset of currents that
receive a full Lyapunov estimate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chapter2.config_ch2 import (
    DT,
    FIXED_CURRENTS as KNOWN_CURRENTS,
    HR_PARAMETERS,
    INITIAL_STATE,
    INITIAL_TRANSIENT_STEPS,
    RETAINED_SAMPLES_PER_CURRENT,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
RESULTS_DIR = PACKAGE_ROOT / "results"
DENSE_SWEEP_PATH = RESULTS_DIR / "dense_sweep_results.json"
REGIME_SUMMARY_PATH = RESULTS_DIR / "regime_summary.json"
FIGURE_PATH = RESULTS_DIR / "bifurcation_map.png"

# Reconnaissance (20,000-transient / 20,000-retained, not saved as a
# scientific artifact) showed a resting fixed point for I <~ 1.3 and
# monotonically increasing tonic-spiking frequency continuing past I = 4.5
# with no further qualitative change. The dense grid below is therefore
# bounded to [1.0, 4.0]: it fully contains the resting-to-oscillatory onset,
# all five previously studied currents (1.67-3.50), and one unit of margin
# above the highest previously studied current.
SWEEP_MIN_CURRENT = 1.0
SWEEP_MAX_CURRENT = 4.0
SWEEP_COUNT = 400

# Every previously studied current is inserted into the grid exactly (not
# approximated by its nearest neighbour) so Step 8 compares like with like.
_BASE_GRID = np.linspace(SWEEP_MIN_CURRENT, SWEEP_MAX_CURRENT, SWEEP_COUNT)
_MERGE_TOLERANCE = 1.0e-9


def build_sweep_grid() -> np.ndarray:
    """Return the sorted, deduplicated dense sweep grid."""
    merged = np.concatenate([_BASE_GRID, np.asarray(KNOWN_CURRENTS, dtype=float)])
    merged.sort()
    keep = np.ones(len(merged), dtype=bool)
    keep[1:] = np.diff(merged) > _MERGE_TOLERANCE
    return merged[keep]


# Reusing the full 100,000-step transient and 100,000-sample retained window
# already established for the five fixed currents (config_ch2). A dense
# bifurcation sweep is exactly the setting where a short trajectory is most
# likely to mislabel a genuine attractor as transient-contaminated, so no
# shorter alternative is used anywhere in this package.
TRANSIENT_STEPS = INITIAL_TRANSIENT_STEPS
RETAINED_SAMPLES = RETAINED_SAMPLES_PER_CURRENT

# The Benettin Lyapunov estimator (~600,000 tangent-linear steps per current)
# costs roughly 6x a single dense-grid trajectory. Running it at all ~400+
# grid points would take on the order of 7-8 CPU-hours for this one
# diagnostic step alone. It is therefore run only at: (1) the five previously
# studied currents, and (2) a bounded number of automatically selected
# representative/boundary currents drawn from the spike/burst-only
# classification (see analyse_regimes.select_lyapunov_currents). No new
# Lyapunov estimator is implemented anywhere in this package; every exponent
# is produced by the unmodified, previously validated
# chapter2.dynamics_analysis_ch2.estimate_lyapunov / Benettin estimator.
MAX_EXTRA_LYAPUNOV_CURRENTS = 40

__all__ = [
    "DT",
    "HR_PARAMETERS",
    "INITIAL_STATE",
    "KNOWN_CURRENTS",
    "TRANSIENT_STEPS",
    "RETAINED_SAMPLES",
    "SWEEP_MIN_CURRENT",
    "SWEEP_MAX_CURRENT",
    "SWEEP_COUNT",
    "MAX_EXTRA_LYAPUNOV_CURRENTS",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "RESULTS_DIR",
    "DENSE_SWEEP_PATH",
    "REGIME_SUMMARY_PATH",
    "FIGURE_PATH",
    "build_sweep_grid",
]
