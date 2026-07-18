from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from scipy import linalg as scipy_linalg
except ImportError:  # NumPy fallback for lightweight test environments.
    scipy_linalg = None

from neuron_controllers import compute_control_signal


class EchoStateNetwork:
    """
    Echo State Network for one-step and recursive prediction.

    Important:
    - train() and predict() use the same reservoir update logic.
    - predict_controlled() also uses the same logic.
    - This avoids mismatch between normal ESN prediction and control experiment.
    """

    def __init__(
        self,
        N_res: int = 300,
        p: float = 0.10,
        spectral_radius: float = 0.85,
        leaky_coefficient: float = 0.50,
        regularization: float = 1e-6,
        input_size: int = 1,
        normalize_input: bool = False,
        input_scaling: float = 0.50,
        seed: int = 42,
    ) -> None:
        self.N_res = int(N_res)
        self.p = float(p)
        self.spectral_radius = float(spectral_radius)
        self.leaky_coefficient = float(leaky_coefficient)
        self.regularization = float(regularization)
        self.input_size = int(input_size)
        self.normalize_input = bool(normalize_input)
        self.input_scaling = float(input_scaling)
        self.seed = int(seed)

        self.Win = None
        self.W = None
        self.Wres = None
        self.Wout = None

        self.input_mean = None
        self.input_std = None

        self.is_fitted = False

        self._rng = np.random.default_rng(self.seed)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.Win = rng.uniform(
            low=-self.input_scaling,
            high=self.input_scaling,
            size=(self.N_res, self.input_size + 1),
        )

        mask = rng.random((self.N_res, self.N_res)) < self.p
        W_dense = rng.uniform(-1.0, 1.0, size=(self.N_res, self.N_res)) * mask

        np.fill_diagonal(W_dense, 0.0)

        eigvals = np.linalg.eigvals(W_dense)
        radius = np.max(np.abs(eigvals))

        if radius < 1e-12:
            radius = 1.0

        W_dense = W_dense * (self.spectral_radius / radius)

        self.W = W_dense
        self.Wres = W_dense

    @staticmethod
    def _as_2d(u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if u.ndim == 1:
            return u.reshape(-1, 1)
        return u

    @staticmethod
    def _as_1d(u: np.ndarray) -> np.ndarray:
        return np.asarray(u, dtype=float).reshape(-1)

    def _normalize_fit(self, u: np.ndarray) -> np.ndarray:
        u = self._as_2d(u)

        if not self.normalize_input:
            self.input_mean = np.zeros((1, u.shape[1]))
            self.input_std = np.ones((1, u.shape[1]))
            return u

        self.input_mean = u.mean(axis=0, keepdims=True)
        self.input_std = u.std(axis=0, keepdims=True)
        self.input_std[self.input_std < 1e-12] = 1.0

        return (u - self.input_mean) / self.input_std

    def _normalize_apply(self, u: np.ndarray) -> np.ndarray:
        u = self._as_2d(u)

        if not self.normalize_input:
            return u

        return (u - self.input_mean) / self.input_std

    def _denormalize(self, u: np.ndarray) -> np.ndarray:
        u = self._as_2d(u)

        if not self.normalize_input:
            return u

        return u * self.input_std + self.input_mean

    def _input_with_bias(self, u_t: np.ndarray) -> np.ndarray:
        u_t = self._as_1d(u_t)
        return np.concatenate([[1.0], u_t])

    def _update_state(self, x_prev: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        u_bias = self._input_with_bias(u_t)

        pre_activation = self.Win @ u_bias + self.W @ x_prev
        x_new_raw = np.tanh(pre_activation)

        alpha = self.leaky_coefficient
        x_new = (1.0 - alpha) * x_prev + alpha * x_new_raw

        return x_new

    def _make_readout_feature(self, u_t: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        """
        Readout feature:
            [bias, input, reservoir_state]

        Shape:
            1 + input_size + N_res
        """
        u_t = self._as_1d(u_t)
        x_t = self._as_1d(x_t)
        return np.concatenate([[1.0], u_t, x_t])

    def _readout(self, u_t: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        if self.Wout is None:
            raise RuntimeError("ESN is not trained yet. Call train() first.")

        feature = self._make_readout_feature(u_t, x_t)
        y = self.Wout @ feature

        return self._as_1d(y)

    def train(self, u: np.ndarray, washout: int = 50) -> None:
        """
        Teacher-forced training.

        Learns:
            u(t) -> u(t+1)
        """
        u = self._as_2d(u)
        u = self._normalize_fit(u)

        if u.shape[1] != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, got {u.shape[1]}"
            )

        T = len(u)

        if T < 3:
            raise ValueError("Need at least 3 time steps to train ESN.")

        washout = int(max(0, min(washout, T - 2)))

        x = np.zeros(self.N_res, dtype=float)

        features = []
        targets = []

        for t in range(T - 1):
            u_t = u[t]
            y_target = u[t + 1]

            x = self._update_state(x, u_t)

            if t >= washout:
                feature = self._make_readout_feature(u_t, x)
                features.append(feature)
                targets.append(y_target)

        X = np.asarray(features, dtype=float)
        Y = np.asarray(targets, dtype=float)

        if X.ndim != 2 or Y.ndim != 2:
            raise RuntimeError("Invalid training matrices created.")

        ridge = max(float(self.regularization), 1e-12)

        XtX = X.T @ X
        XtY = X.T @ Y

        I = np.eye(XtX.shape[0])
        A = XtX + ridge * I

        try:
            if scipy_linalg is not None:
                solution = scipy_linalg.solve(A, XtY, assume_a="pos")
            else:
                solution = np.linalg.solve(A, XtY)
        except Exception:
            solution = np.linalg.pinv(A) @ XtY

        self.Wout = solution.T
        self.is_fitted = True

    def _warmup_until_index(self, u: np.ndarray, n_warmup: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Warm reservoir using ground-truth sequence.

        After warmup:
            current_input = u[n_warmup]
        """
        u = self._as_2d(u)

        n_warmup = int(max(0, min(n_warmup, len(u) - 1)))

        x = np.zeros(self.N_res, dtype=float)

        for t in range(n_warmup):
            x = self._update_state(x, u[t])

        current_input = u[n_warmup].copy()

        return x, current_input

    def predict(self, u: np.ndarray, n_warmup: int = 0):
        """
        Recursive autonomous prediction.
        """
        if not self.is_fitted:
            raise RuntimeError("ESN is not trained yet. Call train() first.")

        u = self._as_2d(u)
        u = self._normalize_apply(u)

        x, current_input = self._warmup_until_index(u, n_warmup)

        horizon = len(u) - int(n_warmup) - 1

        if horizon <= 0:
            return np.empty((0, self.input_size)), np.empty((0, self.N_res))

        predictions = np.zeros((horizon, self.input_size), dtype=float)
        states = np.zeros((horizon, self.N_res), dtype=float)

        for k in range(horizon):
            x = self._update_state(x, current_input)
            y_pred = self._readout(current_input, x)

            predictions[k] = y_pred
            states[k] = x

            current_input = y_pred

        predictions = self._denormalize(predictions)

        return predictions, states

    def predict_controlled(
        self,
        train_sequence: np.ndarray,
        horizon_steps: int,
        target: np.ndarray,
        K: float,
        control_start_idx: int = 0,
        max_abs_value: float = 1e6,
        controller: str = "linear_feedback",
        finite_s: float = 0.8,
        pyragas_delay: int = 20,
        pyragas_sign: int = -1,
        pyragas_history_signal: str = "raw_readout",
        control_input_clip=None,
        divergence_abs_limit: float | None = None,
    ) -> dict:
        """
        Recursive prediction with controller.

        Supported controllers:
            linear_feedback
            finite_time
            pyragas

        Canonical signals:
        - raw_readout_norm is the ESN readout before control is applied.
        - corrected_feedback_input_norm is the corrected signal fed back to the
          ESN at the next recursive step.
        - control_signal_norm is the correction actually applied, so
          corrected feedback input = raw readout - applied control signal.
        - Legacy output names are retained as aliases for existing callers.

        Pyragas note:
        - model.py applies control as: next_input = y_pred - u_control.
        - Therefore, pyragas_sign=-1 moves the next input toward the delayed
          state when the controller computes K * (y_pred - delayed_state).
        - pyragas_history_signal chooses whether delay history contains raw ESN
          readouts (the paper-aligned default) or corrected feedback inputs
          (the legacy behaviour).

        Safety note:
        - control_input_clip clips the corrected feedback input, not the raw
          controller request. If clipping changes the request, the returned
          applied control is recomputed to preserve the signal identity.
        - divergence_abs_limit supersedes max_abs_value when supplied. Limits
          and clipping operate in the coordinates passed to this method
          (normally externally normalized ESN coordinates).
        """
        if not self.is_fitted:
            raise RuntimeError("ESN is not trained yet. Call train() first.")

        history_signal = str(pyragas_history_signal).strip().lower()
        valid_history_signals = {"raw_readout", "corrected_feedback_input"}
        if history_signal not in valid_history_signals:
            raise ValueError(
                "pyragas_history_signal must be 'raw_readout' or "
                f"'corrected_feedback_input'. Got {pyragas_history_signal!r}."
            )

        if divergence_abs_limit is None:
            resolved_divergence_limit = float(max_abs_value)
        else:
            resolved_divergence_limit = float(divergence_abs_limit)
        if np.isnan(resolved_divergence_limit) or resolved_divergence_limit <= 0.0:
            raise ValueError(
                "divergence_abs_limit/max_abs_value must be positive. "
                f"Got {resolved_divergence_limit}."
            )

        clip_bounds = None
        if control_input_clip is not None:
            if np.isscalar(control_input_clip):
                clip_magnitude = float(control_input_clip)
                if not np.isfinite(clip_magnitude) or clip_magnitude < 0.0:
                    raise ValueError(
                        "Scalar control_input_clip must be finite and non-negative. "
                        f"Got {control_input_clip!r}."
                    )
                clip_bounds = (-clip_magnitude, clip_magnitude)
            else:
                clip_values = np.asarray(control_input_clip, dtype=float).reshape(-1)
                if clip_values.size != 2:
                    raise ValueError(
                        "control_input_clip must be None, a non-negative scalar, "
                        "or a (lower, upper) pair."
                    )
                lower, upper = float(clip_values[0]), float(clip_values[1])
                if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                    raise ValueError(
                        "control_input_clip bounds must be finite with lower <= upper. "
                        f"Got ({lower}, {upper})."
                    )
                clip_bounds = (lower, upper)

        train_sequence = self._as_2d(train_sequence)
        train_sequence = self._normalize_apply(train_sequence)

        target = self._as_1d(target)
        if target.size != self.input_size:
            raise ValueError(
                f"Target size must be {self.input_size}, got {target.size}"
            )

        horizon_steps = int(horizon_steps)
        if horizon_steps <= 0:
            empty_signal = np.empty((0, self.input_size))
            return {
                "stable": True,
                "divergence_detected": False,
                "divergence_reason": None,
                "divergence_index": None,
                "steps_completed": 0,
                "pyragas_history_signal": history_signal,
                "control_input_clip": clip_bounds,
                "divergence_abs_limit": resolved_divergence_limit,
                "raw_readout_norm": empty_signal,
                "corrected_feedback_input_norm": empty_signal,
                "requested_control_signal_norm": empty_signal,
                "control_signal_norm": empty_signal,
                "raw_readout_error_norm": empty_signal,
                "corrected_feedback_error_norm": empty_signal,
                "raw_prediction_norm": empty_signal,
                "controlled_output_norm": empty_signal,
                "feedback_input_norm": empty_signal,
                "error_signal_norm": empty_signal,
                "states": np.empty((0, self.N_res)),
            }

        control_start_idx = int(max(0, min(control_start_idx, horizon_steps - 1)))
        x, current_input = self._warmup_until_index(
            train_sequence,
            n_warmup=len(train_sequence) - 1,
        )

        raw_readout = np.full((horizon_steps, self.input_size), np.nan)
        corrected_feedback_input = np.full(
            (horizon_steps, self.input_size), np.nan
        )
        requested_control_signal = np.full(
            (horizon_steps, self.input_size), np.nan
        )
        applied_control_signal = np.full(
            (horizon_steps, self.input_size), np.nan
        )
        raw_readout_error = np.full((horizon_steps, self.input_size), np.nan)
        corrected_feedback_error = np.full(
            (horizon_steps, self.input_size), np.nan
        )
        states = np.full((horizon_steps, self.N_res), np.nan)

        stable = True
        divergence_reason = None
        divergence_index = None
        steps_completed = 0
        history = []

        for k in range(horizon_steps):
            x = self._update_state(x, current_input)
            y_pred = self._readout(current_input, x)
            error = y_pred - target

            raw_readout[k] = y_pred
            raw_readout_error[k] = error
            states[k] = x
            steps_completed = k + 1

            if not np.all(np.isfinite(y_pred)):
                stable = False
                divergence_reason = "nonfinite_raw_readout"
                divergence_index = k
                break
            if not np.all(np.isfinite(x)):
                stable = False
                divergence_reason = "nonfinite_reservoir_state"
                divergence_index = k
                break
            if np.max(np.abs(y_pred)) > resolved_divergence_limit:
                stable = False
                divergence_reason = "raw_readout_abs_limit_exceeded"
                divergence_index = k
                break
            if np.max(np.abs(x)) > resolved_divergence_limit:
                stable = False
                divergence_reason = "reservoir_state_abs_limit_exceeded"
                divergence_index = k
                break

            if k >= control_start_idx:
                try:
                    requested_control = compute_control_signal(
                        controller=controller,
                        y_pred=y_pred,
                        target=target,
                        K=K,
                        history=history,
                        finite_s=finite_s,
                        pyragas_delay=pyragas_delay,
                        pyragas_sign=pyragas_sign,
                    )
                except FloatingPointError:
                    stable = False
                    divergence_reason = "nonfinite_requested_control_signal"
                    divergence_index = k
                    break
                next_input = y_pred - requested_control
                if clip_bounds is not None:
                    next_input = np.clip(
                        next_input, clip_bounds[0], clip_bounds[1]
                    )
                applied_control = y_pred - next_input
            else:
                requested_control = np.zeros_like(y_pred)
                applied_control = np.zeros_like(y_pred)
                next_input = y_pred

            corrected_feedback_input[k] = next_input
            requested_control_signal[k] = requested_control
            applied_control_signal[k] = applied_control
            corrected_feedback_error[k] = next_input - target

            if not np.all(np.isfinite(requested_control)):
                stable = False
                divergence_reason = "nonfinite_requested_control_signal"
                divergence_index = k
                break
            if np.max(np.abs(requested_control)) > resolved_divergence_limit:
                stable = False
                divergence_reason = "requested_control_signal_abs_limit_exceeded"
                divergence_index = k
                break
            if not np.all(np.isfinite(applied_control)):
                stable = False
                divergence_reason = "nonfinite_applied_control_signal"
                divergence_index = k
                break
            if np.max(np.abs(applied_control)) > resolved_divergence_limit:
                stable = False
                divergence_reason = "applied_control_signal_abs_limit_exceeded"
                divergence_index = k
                break
            if not np.all(np.isfinite(next_input)):
                stable = False
                divergence_reason = "nonfinite_corrected_feedback_input"
                divergence_index = k
                break
            if np.max(np.abs(next_input)) > resolved_divergence_limit:
                stable = False
                divergence_reason = "corrected_feedback_input_abs_limit_exceeded"
                divergence_index = k
                break

            history_value = (
                y_pred if history_signal == "raw_readout" else next_input
            )
            history.append(history_value.copy())

            current_input = next_input

        return {
            "stable": stable,
            "divergence_detected": not stable,
            "divergence_reason": divergence_reason,
            "divergence_index": divergence_index,
            "steps_completed": int(steps_completed),
            "controller": controller,
            "K": float(K),
            "finite_s": float(finite_s),
            "pyragas_delay": int(pyragas_delay),
            "pyragas_sign": int(pyragas_sign),
            "pyragas_history_signal": history_signal,
            "control_input_clip": clip_bounds,
            "divergence_abs_limit": resolved_divergence_limit,
            "raw_readout_norm": raw_readout,
            "corrected_feedback_input_norm": corrected_feedback_input,
            "requested_control_signal_norm": requested_control_signal,
            "control_signal_norm": applied_control_signal,
            "raw_readout_error_norm": raw_readout_error,
            "corrected_feedback_error_norm": corrected_feedback_error,
            # Backward-compatible aliases. controlled_output_norm and
            # feedback_input_norm both mean corrected feedback input.
            "raw_prediction_norm": raw_readout,
            "controlled_output_norm": corrected_feedback_input,
            "feedback_input_norm": corrected_feedback_input,
            "error_signal_norm": raw_readout_error,
            "states": states,
        }

    def model_identity_hash(self) -> str:
        """Return a deterministic identity for architecture and trained weights."""
        if not self.is_fitted or self.Wout is None:
            raise RuntimeError("Cannot identify an unfitted ESN.")
        scalars = {
            "N_res": self.N_res,
            "p": self.p,
            "spectral_radius": self.spectral_radius,
            "leaky_coefficient": self.leaky_coefficient,
            "regularization": self.regularization,
            "input_size": self.input_size,
            "normalize_input": self.normalize_input,
            "input_scaling": self.input_scaling,
            "seed": self.seed,
        }
        digest = hashlib.sha256(
            json.dumps(
                scalars, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        for name, value in (
            ("Win", self.Win),
            ("W", self.W),
            ("Wout", self.Wout),
            ("input_mean", self.input_mean),
            ("input_std", self.input_std),
        ):
            array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
            digest.update(name.encode("utf-8"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def save_bundle(
        self,
        path,
        *,
        metadata: dict | None = None,
        external_mean: np.ndarray | None = None,
        external_std: np.ndarray | None = None,
    ) -> dict:
        """Save every deterministic component needed to reuse this fitted ESN."""
        if not self.is_fitted or self.Wout is None:
            raise RuntimeError("Cannot save an unfitted ESN.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        identity = self.model_identity_hash()
        bundle_metadata = dict(metadata or {})
        bundle_metadata.update(
            {
                "schema_version": "chapter1_esn_bundle_v1",
                "model_identity_hash": identity,
                "model_seed": int(self.seed),
            }
        )
        np.savez_compressed(
            destination,
            Win=np.asarray(self.Win, dtype=float),
            W=np.asarray(self.W, dtype=float),
            Wout=np.asarray(self.Wout, dtype=float),
            input_mean=np.asarray(self.input_mean, dtype=float),
            input_std=np.asarray(self.input_std, dtype=float),
            external_mean=np.asarray(
                np.empty((0, self.input_size))
                if external_mean is None
                else external_mean,
                dtype=float,
            ),
            external_std=np.asarray(
                np.empty((0, self.input_size))
                if external_std is None
                else external_std,
                dtype=float,
            ),
            metadata_json=np.asarray(
                json.dumps(bundle_metadata, sort_keys=True)
            ),
            N_res=np.asarray(self.N_res),
            p=np.asarray(self.p),
            spectral_radius=np.asarray(self.spectral_radius),
            leaky_coefficient=np.asarray(self.leaky_coefficient),
            regularization=np.asarray(self.regularization),
            input_size=np.asarray(self.input_size),
            normalize_input=np.asarray(self.normalize_input),
            input_scaling=np.asarray(self.input_scaling),
            seed=np.asarray(self.seed),
        )
        return bundle_metadata

    @classmethod
    def load_bundle(cls, path):
        """Load a saved bundle without training and verify its identity hash."""
        source = Path(path)
        with np.load(source, allow_pickle=False) as bundle:
            model = cls(
                N_res=int(bundle["N_res"]),
                p=float(bundle["p"]),
                spectral_radius=float(bundle["spectral_radius"]),
                leaky_coefficient=float(bundle["leaky_coefficient"]),
                regularization=float(bundle["regularization"]),
                input_size=int(bundle["input_size"]),
                normalize_input=bool(bundle["normalize_input"]),
                input_scaling=float(bundle["input_scaling"]),
                seed=int(bundle["seed"]),
            )
            model.Win = np.asarray(bundle["Win"], dtype=float)
            model.W = np.asarray(bundle["W"], dtype=float)
            model.Wres = model.W
            model.Wout = np.asarray(bundle["Wout"], dtype=float)
            model.input_mean = np.asarray(bundle["input_mean"], dtype=float)
            model.input_std = np.asarray(bundle["input_std"], dtype=float)
            model.is_fitted = True
            metadata = json.loads(str(bundle["metadata_json"]))
            external_mean = np.asarray(bundle["external_mean"], dtype=float)
            external_std = np.asarray(bundle["external_std"], dtype=float)

        expected = str(metadata.get("model_identity_hash", ""))
        actual = model.model_identity_hash()
        if expected and expected != actual:
            raise ValueError(
                "Saved ESN identity hash does not match its serialized weights."
            )
        metadata["model_identity_hash"] = actual
        metadata["loaded_from_cache"] = True
        metadata["external_mean"] = external_mean
        metadata["external_std"] = external_std
        return model, metadata
