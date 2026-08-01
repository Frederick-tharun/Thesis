"""Independent Hindmarsh--Rose simulation utilities for Chapter 2.

The equations, state-recording convention, and RK4 operation ordering match
the Chapter 1 implementation in ``data_loader.py``.  Keeping the code here
separate protects the reproducibility of the completed Chapter 1 workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Sequence
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

try:
    from .config_ch2 import DT, HR_PARAMETERS, HRParameters, INITIAL_STATE
except ImportError:  # Support direct execution from the chapter2 directory.
    from config_ch2 import DT, HR_PARAMETERS, HRParameters, INITIAL_STATE


@dataclass(frozen=True)
class HRTrajectory:
    """Named arrays for one recorded HR trajectory."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    I: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(getattr(self, name)) for name in "txyzI")
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("All trajectory arrays must be one-dimensional")
        lengths = {len(array) for array in arrays}
        if len(lengths) != 1:
            raise ValueError("All trajectory arrays must have equal lengths")
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("Trajectory arrays must contain only finite values")

    def as_matrix(self) -> np.ndarray:
        """Return columns ordered as ``(t, x, y, z, I)``."""
        return np.column_stack((self.t, self.x, self.y, self.z, self.I))

    @property
    def state(self) -> np.ndarray:
        """Return the three state variables as ``(n_samples, 3)``."""
        return np.column_stack((self.x, self.y, self.z))


def _validated_state(state: Sequence[float] | np.ndarray) -> np.ndarray:
    result = np.asarray(state, dtype=float)
    if result.shape != (3,):
        raise ValueError(f"state must have shape (3,), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("state must contain only finite values")
    return result.copy()


def _validate_simulation_inputs(current: float, dt: float) -> None:
    if not np.isfinite(current):
        raise ValueError("current must be finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite number")


def hr_rhs(
    state: Sequence[float] | np.ndarray,
    current: float,
    parameters: HRParameters = HR_PARAMETERS,
) -> np.ndarray:
    """Evaluate the Hindmarsh--Rose vector field.

    The equations are

    ``dx/dt = y - a*x^3 + b*x^2 - z + I``
    ``dy/dt = c - d*x^2 - y``
    ``dz/dt = r*(s*(x - x_r) - z)``.
    """
    x, y, z = np.asarray(state, dtype=float)

    dx = y - parameters.a * x**3 + parameters.b * x**2 - z + current
    dy = parameters.c - parameters.d * x**2 - y
    dz = parameters.r * (parameters.s * (x - parameters.x_r) - z)

    return np.array([dx, dy, dz], dtype=float)


def rk4_step(
    state: Sequence[float] | np.ndarray,
    dt: float,
    current: float,
    parameters: HRParameters = HR_PARAMETERS,
) -> np.ndarray:
    """Advance one classical fourth-order Runge--Kutta step."""
    state_array = np.asarray(state, dtype=float)
    k1 = hr_rhs(state_array, current, parameters)
    k2 = hr_rhs(state_array + 0.5 * dt * k1, current, parameters)
    k3 = hr_rhs(state_array + 0.5 * dt * k2, current, parameters)
    k4 = hr_rhs(state_array + dt * k3, current, parameters)
    return state_array + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_fixed_current(
    current: float,
    *,
    retained_samples: int,
    transient_steps: int,
    initial_state: Sequence[float] | np.ndarray = INITIAL_STATE,
    dt: float = DT,
    parameters: HRParameters = HR_PARAMETERS,
) -> HRTrajectory:
    """Simulate one fixed-current trajectory after a discarded transient.

    As in Chapter 1, the state is recorded before advancing a step.  Thus the
    first retained sample is the state after exactly ``transient_steps`` RK4
    updates, and its displayed time is zero.
    """
    _validate_simulation_inputs(float(current), float(dt))
    if retained_samples <= 0:
        raise ValueError("retained_samples must be positive")
    if transient_steps < 0:
        raise ValueError("transient_steps must be non-negative")

    state = _validated_state(initial_state)
    for _ in range(transient_steps):
        state = rk4_step(state, dt, current, parameters)

    states = np.empty((retained_samples, 3), dtype=float)
    for index in range(retained_samples):
        states[index] = state
        state = rk4_step(state, dt, current, parameters)

    time = np.arange(retained_samples, dtype=float) * dt
    currents = np.full(retained_samples, float(current), dtype=float)
    return HRTrajectory(
        t=time,
        x=states[:, 0],
        y=states[:, 1],
        z=states[:, 2],
        I=currents,
    )


def simulate_continuous_currents(
    currents: Sequence[float],
    *,
    samples_per_segment: int,
    transient_steps: int,
    initial_state: Sequence[float] | np.ndarray = INITIAL_STATE,
    dt: float = DT,
    parameters: HRParameters = HR_PARAMETERS,
) -> tuple[HRTrajectory, np.ndarray]:
    """Simulate one state-continuous trajectory with piecewise-constant I.

    The transient is applied once using the first current.  At subsequent
    switches only ``I`` changes; the state is never reset.
    """
    current_values = np.asarray(currents, dtype=float)
    if current_values.ndim != 1 or len(current_values) == 0:
        raise ValueError("currents must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(current_values)):
        raise ValueError("currents must contain only finite values")
    _validate_simulation_inputs(float(current_values[0]), float(dt))
    if samples_per_segment <= 0:
        raise ValueError("samples_per_segment must be positive")
    if transient_steps < 0:
        raise ValueError("transient_steps must be non-negative")

    state = _validated_state(initial_state)
    for _ in range(transient_steps):
        state = rk4_step(state, dt, float(current_values[0]), parameters)

    total_samples = len(current_values) * samples_per_segment
    states = np.empty((total_samples, 3), dtype=float)
    current_series = np.repeat(current_values, samples_per_segment)

    for index, current in enumerate(current_series):
        states[index] = state
        state = rk4_step(state, dt, float(current), parameters)

    time = np.arange(total_samples, dtype=float) * dt
    switch_indices = np.arange(
        samples_per_segment,
        total_samples,
        samples_per_segment,
        dtype=int,
    )
    return (
        HRTrajectory(
            t=time,
            x=states[:, 0],
            y=states[:, 1],
            z=states[:, 2],
            I=current_series,
        ),
        switch_indices,
    )


def save_trajectory_npz(path: str | Path, trajectory: HRTrajectory) -> None:
    """Write a byte-reproducible NPZ containing ``t, x, y, z, I`` arrays.

    NumPy-compatible ``.npy`` members are stored with fixed ZIP metadata and a
    stable order, avoiding timestamp-dependent archives.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    members = (
        ("t", trajectory.t),
        ("x", trajectory.x),
        ("y", trajectory.y),
        ("z", trajectory.z),
        ("I", trajectory.I),
    )

    with ZipFile(destination, mode="w", compression=ZIP_DEFLATED) as archive:
        for name, values in members:
            buffer = BytesIO()
            np.lib.format.write_array(
                buffer,
                np.asarray(values),
                version=(1, 0),
                allow_pickle=False,
            )
            info = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                buffer.getvalue(),
                compress_type=ZIP_DEFLATED,
                compresslevel=9,
            )



def load_trajectory_npz(path: str | Path) -> HRTrajectory:
    """Load a saved Chapter 2 trajectory without changing it."""
    with np.load(Path(path), allow_pickle=False) as saved:
        required = ("t", "x", "y", "z", "I")
        if tuple(saved.files) != required:
            raise ValueError(
                f"trajectory fields must be {required}, got {tuple(saved.files)}"
            )
        return HRTrajectory(
            **{name: np.asarray(saved[name]).copy() for name in required}
        )
