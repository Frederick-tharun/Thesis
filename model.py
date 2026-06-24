from __future__ import annotations

import numpy as np
from scipy import linalg

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
            solution = linalg.solve(A, XtY, assume_a="pos")
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
    ) -> dict:
        """
        Recursive prediction with controller.

        Supported controllers:
            linear_feedback
            finite_time
            pyragas

        Important:
        - The ESN rollout logic is exactly the same as predict().
        - Before control starts, controlled output = normal ESN output.
        - After control starts, controlled output = controlled input fed back to ESN.
        - Controller formulas are stored in neuron_controllers.py.

        Pyragas note:
        - model.py applies control as: next_input = y_pred - u_control
        - Therefore, pyragas_sign=-1 makes the Pyragas signal move the next input
          toward the delayed state when neuron_controllers.py computes
          u_control = K * (y_pred - delayed_state).
        """
        if not self.is_fitted:
            raise RuntimeError("ESN is not trained yet. Call train() first.")

        train_sequence = self._as_2d(train_sequence)
        train_sequence = self._normalize_apply(train_sequence)

        target = self._as_1d(target)

        if target.size != self.input_size:
            raise ValueError(
                f"Target size must be {self.input_size}, got {target.size}"
            )

        horizon_steps = int(horizon_steps)

        if horizon_steps <= 0:
            return {
                "stable": True,
                "raw_prediction_norm": np.empty((0, self.input_size)),
                "controlled_output_norm": np.empty((0, self.input_size)),
                "feedback_input_norm": np.empty((0, self.input_size)),
                "control_signal_norm": np.empty((0, self.input_size)),
                "error_signal_norm": np.empty((0, self.input_size)),
                "states": np.empty((0, self.N_res)),
            }

        control_start_idx = int(max(0, min(control_start_idx, horizon_steps - 1)))

        x, current_input = self._warmup_until_index(
            train_sequence,
            n_warmup=len(train_sequence) - 1,
        )

        raw_prediction = np.full((horizon_steps, self.input_size), np.nan)
        controlled_output = np.full((horizon_steps, self.input_size), np.nan)
        feedback_input = np.full((horizon_steps, self.input_size), np.nan)
        control_signal = np.full((horizon_steps, self.input_size), np.nan)
        error_signal = np.full((horizon_steps, self.input_size), np.nan)
        states = np.full((horizon_steps, self.N_res), np.nan)

        stable = True
        history = []

        for k in range(horizon_steps):
            x = self._update_state(x, current_input)
            y_pred = self._readout(current_input, x)

            error = y_pred - target

            if k >= control_start_idx:
                u_control = compute_control_signal(
                    controller=controller,
                    y_pred=y_pred,
                    target=target,
                    K=K,
                    history=history,
                    finite_s=finite_s,
                    pyragas_delay=pyragas_delay,
                    pyragas_sign=pyragas_sign,
                )
                next_input = y_pred - u_control
                y_controlled = next_input
            else:
                u_control = np.zeros_like(y_pred)
                next_input = y_pred
                y_controlled = y_pred

            raw_prediction[k] = y_pred
            controlled_output[k] = y_controlled
            feedback_input[k] = next_input
            control_signal[k] = u_control
            error_signal[k] = error
            states[k] = x

            history.append(y_controlled.copy())

            if (
                not np.all(np.isfinite(next_input))
                or not np.all(np.isfinite(x))
                or np.max(np.abs(next_input)) > max_abs_value
                or np.max(np.abs(x)) > max_abs_value
            ):
                stable = False
                break

            current_input = next_input

        return {
            "stable": stable,
            "controller": controller,
            "K": float(K),
            "finite_s": float(finite_s),
            "pyragas_delay": int(pyragas_delay),
            "pyragas_sign": int(pyragas_sign),
            "raw_prediction_norm": raw_prediction,
            "controlled_output_norm": controlled_output,
            "feedback_input_norm": feedback_input,
            "control_signal_norm": control_signal,
            "error_signal_norm": error_signal,
            "states": states,
        }
