"""
Modified controller definitions for ESN control experiments.

This module defines three controllers: linear feedback, finite‑time and Pyragas delayed
feedback.  The Pyragas controller has been extended with a configurable feedback sign
parameter to support both sign conventions commonly used in the literature.

Changes from the original implementation:
  • Added ``pyragas_sign`` parameter to ``pyragas_control``.  This parameter can
    be set to ``1`` to apply ``u = K * (delayed_state – y_pred)`` or ``-1`` to
    apply ``u = K * (y_pred – delayed_state)``.  The default is ``1``.
  • Updated ``compute_control_signal`` to pass through ``pyragas_sign`` when
    invoking the Pyragas controller.
  • Added validation for ``pyragas_sign`` and improved docstrings.

These changes are intended to facilitate experimentation with different Pyragas
control conventions without affecting the existing linear and finite‑time
controllers.
"""

from __future__ import annotations

import numpy as np


# Supported controller names.  Extend this tuple when adding new controllers.
SUPPORTED_CONTROLLERS = (
    "linear_feedback",
    "finite_time",
    "pyragas",
)


def _as_1d(x) -> np.ndarray:
    """Return a 1‑D NumPy array of floats from any array‑like input."""
    return np.asarray(x, dtype=float).reshape(-1)


def linear_feedback_control(*, y_pred, target, K, **kwargs) -> np.ndarray:
    """
    Linear feedback control.

    Parameters
    ----------
    y_pred : array‑like
        Current predicted state of the system.
    target : array‑like
        Desired target state.
    K : float
        Feedback gain.

    Returns
    -------
    ndarray
        Control signal with the same shape as ``y_pred``.

    Notes
    -----
    The control law is ``u = K * (y_pred – target)``.  In the model this
    control signal is subtracted from the prediction to form the next input.
    """
    y_pred = _as_1d(y_pred)
    target = _as_1d(target)
    error = y_pred - target
    return float(K) * error


def finite_time_control(*, y_pred, target, K, finite_s=0.8, eps=1e-8, **kwargs) -> np.ndarray:
    """
    Finite‑time feedback control.

    Parameters
    ----------
    y_pred : array‑like
        Current predicted state of the system.
    target : array‑like
        Desired target state.
    K : float
        Feedback gain.
    finite_s : float, optional
        Nonlinear exponent for the finite‑time controller (``0 < finite_s < 1``).
    eps : float, optional
        Small constant to avoid zero division when the error is exactly zero.

    Returns
    -------
    ndarray
        Control signal with the same shape as ``y_pred``.

    Notes
    -----
    The control law is ``u = K * sign(error) * |error|^s`` where ``error = y_pred – target``
    and ``0 < s < 1``.  Smaller values of ``s`` produce stronger nonlinear
    corrections near the target.
    """
    y_pred = _as_1d(y_pred)
    target = _as_1d(target)
    finite_s = float(finite_s)
    if not (0.0 < finite_s < 1.0):
        raise ValueError(f"finite_s must be between 0 and 1.  Got finite_s={finite_s}")
    error = y_pred - target
    return float(K) * np.sign(error) * (np.abs(error) + float(eps)) ** finite_s


def pyragas_control(
    *,
    y_pred: np.ndarray,
    target: np.ndarray,
    K: float,
    history: list | np.ndarray | None = None,
    pyragas_delay: int = 20,
    pyragas_sign: int = 1,
    **kwargs,
) -> np.ndarray:
    """
    Pyragas time‑delay feedback control.

    This controller applies a delayed feedback term to stabilise oscillatory
    dynamics.  Unlike linear or finite‑time control, Pyragas control does not
    track a fixed target; instead it aims to regulate the difference between
    the current state and a state from ``pyragas_delay`` timesteps in the past.

    Parameters
    ----------
    y_pred : array‑like of shape (n,)
        Current predicted state of the system from the ESN.
    target : array‑like of shape (n,)
        Desired target state.  This argument is ignored by Pyragas control but
        included for API compatibility.
    K : float
        Feedback gain; controls the strength of the delayed feedback.
    history : list or ndarray, optional
        Sequence of previous controlled states.  The length of ``history``
        determines whether a delayed state can be extracted.  If ``history``
        is ``None`` or shorter than ``pyragas_delay``, the control signal is
        zero.
    pyragas_delay : int, optional
        Number of timesteps to look back for the delayed feedback.  Must be
        a positive integer.
    pyragas_sign : int, optional
        Sign convention for the feedback.  Use ``+1`` for
        ``u = K * (y_delayed − y_pred)`` and ``−1`` for
        ``u = K * (y_pred − y_delayed)``.  Defaults to ``+1``.

    Returns
    -------
    ndarray of shape (n,)
        The control signal ``u`` to be subtracted from the prediction.

    Raises
    ------
    ValueError
        If ``pyragas_delay`` is not a positive integer, or if ``pyragas_sign``
        is not ``+1`` or ``−1``.

    Notes
    -----
    Pyragas control is often used to stabilise unstable periodic or chaotic
    trajectories without specifying an explicit target.  The sign convention
    controls whether the feedback pushes the state toward the delayed state
    (``pyragas_sign`` = +1) or away from it (``pyragas_sign`` = −1).  If the
    history is shorter than the delay, no feedback is applied.
    """
    # Convert to 1‑D array.
    y_pred = _as_1d(y_pred)

    # If there is no history, return zero control (cannot compute delayed state).
    if history is None:
        return np.zeros_like(y_pred)

    # Validate pyragas_delay.
    try:
        delay_int: int = int(pyragas_delay)
    except (TypeError, ValueError):  # pragma: no cover - handled uniformly
        raise ValueError(f"pyragas_delay must be an integer.  Got pyragas_delay={pyragas_delay}")
    if delay_int <= 0:
        raise ValueError(f"pyragas_delay must be positive.  Got pyragas_delay={pyragas_delay}")

    # Validate pyragas_sign.
    if pyragas_sign not in (1, -1):
        raise ValueError(
            f"pyragas_sign must be +1 or -1.  Got pyragas_sign={pyragas_sign}"
        )

    # If history is too short, return zero control.
    if len(history) < delay_int:
        return np.zeros_like(y_pred)

    # Extract delayed state and ensure it matches the current state shape.
    delayed_state = _as_1d(history[-delay_int])
    if delayed_state.shape != y_pred.shape:
        raise ValueError(
            f"Pyragas delayed_state shape {delayed_state.shape} does not match y_pred shape {y_pred.shape}"
        )

    # Compute the difference according to the sign convention.
    if pyragas_sign == 1:
        diff = delayed_state - y_pred
    else:
        diff = y_pred - delayed_state

    return float(K) * diff


# Mapping from controller name to function.
CONTROLLER_FUNCTIONS = {
    "linear_feedback": linear_feedback_control,
    "finite_time": finite_time_control,
    "pyragas": pyragas_control,
}


def compute_control_signal(
    *,
    controller,
    y_pred,
    target,
    K,
    history=None,
    finite_s=0.8,
    pyragas_delay=20,
    pyragas_sign=1,
    eps=1e-8,
) -> np.ndarray:
    """
    Dispatch control signal computation based on controller name.

    Parameters
    ----------
    controller : str
        One of ``linear_feedback``, ``finite_time`` or ``pyragas``.
    y_pred : array‑like
        Current predicted state of the system.
    target : array‑like
        Desired target state.
    K : float
        Feedback gain.
    history : list or ndarray, optional
        History of previous controlled states (required for Pyragas).
    finite_s : float, optional
        Exponent for the finite‑time controller.
    pyragas_delay : int, optional
        Delay for Pyragas control.
    pyragas_sign : int, optional
        Sign convention for Pyragas control (see ``pyragas_control``).
    eps : float, optional
        Small number to avoid zero division in finite‑time control.

    Returns
    -------
    ndarray
        Control signal with the same shape as ``y_pred``.
    """
    if controller not in CONTROLLER_FUNCTIONS:
        raise ValueError(
            f"Unknown controller '{controller}'.  Available controllers: {list(CONTROLLER_FUNCTIONS.keys())}"
        )
    y_pred = _as_1d(y_pred)
    target = _as_1d(target)
    if y_pred.shape != target.shape:
        raise ValueError(
            f"y_pred shape {y_pred.shape} does not match target shape {target.shape}"
        )
    # Build keyword arguments for the controller function.  Only include
    # parameters that the specific controller will understand.
    kwargs = {
        "y_pred": y_pred,
        "target": target,
        "K": K,
        "history": history,
        "finite_s": finite_s,
        "pyragas_delay": pyragas_delay,
        "pyragas_sign": pyragas_sign,
        "eps": eps,
    }
    # Invoke the selected controller.  Extra keyword arguments are ignored by
    # controllers that do not define them.
    u_control = CONTROLLER_FUNCTIONS[controller](**kwargs)
    u_control = _as_1d(u_control)
    if u_control.shape != y_pred.shape:
        raise ValueError(
            f"u_control shape {u_control.shape} does not match y_pred shape {y_pred.shape}"
        )
    if not np.all(np.isfinite(u_control)):
        raise FloatingPointError(
            f"Non‑finite control signal generated by controller '{controller}'"
        )
    return u_control


def available_controllers() -> list[str]:
    """Return a list of available controller names."""
    return list(CONTROLLER_FUNCTIONS.keys())