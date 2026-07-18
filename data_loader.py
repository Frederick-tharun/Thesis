from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import config


# -------------------------------------------------------------------
# Hindmarsh-Rose simulator
# -------------------------------------------------------------------

def _get_hr_params():
    mode = getattr(config, "HR_MODE", "periodic_spiking")

    if mode not in config.HR_PARAMETER_SETS:
        valid = ", ".join(config.HR_PARAMETER_SETS.keys())
        raise ValueError(f"Unknown HR_MODE='{mode}'. Valid modes: {valid}")

    return config.HR_PARAMETER_SETS[mode]


def _hr_rhs(state, params):
    x, y, z = state

    a = params["a"]
    b = params["b"]
    c = params["c"]
    d = params["d"]
    r = params["r"]
    s = params["s"]
    xr = params["xr"]
    I = params["I"]

    dx = y - a * x**3 + b * x**2 - z + I
    dy = c - d * x**2 - y
    dz = r * (s * (x - xr) - z)

    return np.array([dx, dy, dz], dtype=float)


def _rk4_hr(x0, n_steps, dt, params):
    out = np.zeros((n_steps, 3), dtype=float)
    state = np.asarray(x0, dtype=float).copy()

    for i in range(n_steps):
        out[i] = state

        k1 = _hr_rhs(state, params)
        k2 = _hr_rhs(state + 0.5 * dt * k1, params)
        k3 = _hr_rhs(state + 0.5 * dt * k2, params)
        k4 = _hr_rhs(state + dt * k3, params)

        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return out


# -------------------------------------------------------------------
# DataLoader
# -------------------------------------------------------------------

class DataLoader:
    """
    Supports:
    1. Hindmarsh-Rose synthetic data
    2. CSV file: first column = time, remaining columns = neurons
    3. Folder with .npy files
    """

    def __init__(
        self,
        csv_path: str | None = None,
        time_file: str = "time.npy",
        signal_file: str = "u_series.npy",
        single_neuron_name: str = "neuron_1",
    ):
        self.csv_path = csv_path if csv_path is not None else config.DATA_PATH
        self.time_file = time_file
        self.signal_file = signal_file
        self.single_neuron_name = single_neuron_name

        self.raw_df = None

        self.time = None
        self.neuron_names = None
        self.data_raw = None

        self.n_samples = 0
        self.n_neurons = 0

        self.dff = None
        self.data_norm = None
        self.spike_indices = None

    # -------------------------------------------------------------------
    # public methods
    # -------------------------------------------------------------------

    def load(self) -> None:
        mode = getattr(config, "DATASET_MODE", "real").lower()

        if mode == "hr":
            self._load_hr()
            return

        if os.path.isdir(self.csv_path):
            self._load_npy_folder()
            return

        self._load_csv()

    def preprocess(self) -> None:
        mode = getattr(config, "DATASET_MODE", "real").lower()

        if mode == "hr":
            # HR is already a model state, not fluorescence.
            self.dff = self.data_raw.copy()
        else:
            self._compute_dff()

        self._normalise()
        print("[DataLoader] Preprocessing done")

    def detect_spikes(self) -> None:
        mode = getattr(config, "DATASET_MODE", "real").lower()
        self.spike_indices = {}

        if mode == "hr":
            # For HR, spikes should be detected only from x.
            x = self.data_norm[:, 0]
            peaks = self._detect_hr_spikes(x)

            for name in self.neuron_names:
                self.spike_indices[name] = np.array([], dtype=int)

            self.spike_indices["hr_x"] = peaks

            print(
                f"[DataLoader] Spike detection done — "
                f"{len(peaks)} HR x-spikes detected"
            )
            return

        total = 0
        for i, name in enumerate(self.neuron_names):
            peaks = self._detect_single_neuron(self.dff[:, i])
            self.spike_indices[name] = peaks
            total += len(peaks)

        avg = total / max(self.n_neurons, 1)
        print(
            f"[DataLoader] Spike detection done — {total} spikes across "
            f"{self.n_neurons} neurons (avg {avg:.1f} / neuron)"
        )

    def get_neuron(self, neuron_id):
        idx, name = self._resolve(neuron_id)

        spk = (
            self.spike_indices[name]
            if self.spike_indices is not None and name in self.spike_indices
            else np.array([], dtype=int)
        )

        return (
            self.time,
            self.data_raw[:, idx],
            self.data_norm[:, idx],
            spk,
            name,
        )

    def summary(self) -> None:
        dataset_mode = getattr(config, "DATASET_MODE", "real").lower()
        is_hr = dataset_mode == "hr"
        total_s = self.time[-1] - self.time[0] if len(self.time) > 1 else 0.0

        print("\n" + "=" * 56)
        print("DATASET SUMMARY")
        print("=" * 56)
        if is_hr:
            print("Source            : Synthetic Hindmarsh-Rose generated trajectory (RK4)")
        else:
            print(f"File              : {self.csv_path}")
        print(f"Total samples     : {self.n_samples}")
        print(f"Total neurons     : {self.n_neurons}")
        if is_hr:
            print(f"Duration          : {total_s:.2f} simulation time units")
        else:
            print(f"Duration          : {total_s/60:.2f} min ({total_s:.0f} s)")
        print(f"First neuron      : {self.neuron_names[0]}")
        print(f"Last neuron       : {self.neuron_names[-1]}")
        if is_hr:
            print(
                f"Prediction split  : train={config.TRAIN_RATIO:.2f}, "
                f"held-out test={1.0 - config.TRAIN_RATIO:.2f}"
            )
            print("BO validation     : selected only from the training portion")
        else:
            print(
                f"Split ratios      : train={config.TRAIN_RATIO:.2f}, "
                f"val={config.VAL_RATIO:.2f}, test={config.TEST_RATIO:.2f}"
            )

        if self.spike_indices:
            counts = [len(v) for v in self.spike_indices.values()]
            print(f"Spike range       : {min(counts)}–{max(counts)} per neuron")
            print(f"Spike mean        : {np.mean(counts):.1f} per neuron")

        print("=" * 56 + "\n")

    def list_neurons(self, n=10) -> None:
        n = min(n, self.n_neurons)

        print(f"\nFirst {n} neurons:")
        for i in range(n):
            name = self.neuron_names[i]
            count = 0
            if self.spike_indices is not None and name in self.spike_indices:
                count = len(self.spike_indices[name])
            print(f"  [{i:4d}]  {name:<20} spikes: {count}")

        remaining = self.n_neurons - n
        print(f"… and {remaining} more.\n")

    # -------------------------------------------------------------------
    # supervised window helpers, kept for compatibility
    # -------------------------------------------------------------------

    def make_split(
        self,
        neuron_id,
        window_size: int,
        horizon: int = 1,
        debug: bool = False,
    ):
        time, raw, norm, spk, name = self.get_neuron(neuron_id)
        signal = np.asarray(norm, dtype=float).reshape(-1)

        n_total = len(signal)
        n_train = int(config.TRAIN_RATIO * n_total)
        n_val = int(config.VAL_RATIO * n_total)

        train_sig = signal[:n_train]
        val_sig = signal[n_train:n_train + n_val]
        test_sig = signal[n_train + n_val:]

        train_time = time[:n_train]
        val_time = time[n_train:n_train + n_val]
        test_time = time[n_train + n_val:]

        X_train, y_train, t_train_y = self._make_windows(
            train_sig, train_time, window_size, horizon
        )
        X_val, y_val, t_val_y = self._make_windows(
            val_sig, val_time, window_size, horizon
        )
        X_test, y_test, t_test_y = self._make_windows(
            test_sig, test_time, window_size, horizon
        )

        if debug:
            print(f"[Split Debug] neuron={name} window={window_size} horizon={horizon}")
            print(
                f"[Split Debug] train/val/test raw lengths = "
                f"{len(train_sig)} / {len(val_sig)} / {len(test_sig)}"
            )

        return {
            "neuron_name": name,
            "neuron_index": self._resolve(neuron_id)[0],

            "full_time": time,
            "full_signal": signal,

            "train_time": train_time,
            "val_time": val_time,
            "test_time": test_time,

            "train_signal": train_sig,
            "val_signal": val_sig,
            "test_signal": test_sig,

            "X_train": X_train,
            "y_train": y_train,
            "t_train_y": t_train_y,

            "X_val": X_val,
            "y_val": y_val,
            "t_val_y": t_val_y,

            "X_test": X_test,
            "y_test": y_test,
            "t_test_y": t_test_y,

            "spike_idx": spk,
            "raw_signal": raw,
        }

    def make_bo_windows(
        self,
        neuron_id,
        window_size: int,
        horizon: int = 1,
        n_folds: int = 3,
        val_len: int = 40,
        debug: bool = False,
    ):
        _, _, norm, _, name = self.get_neuron(neuron_id)
        signal = np.asarray(norm, dtype=float).reshape(-1)

        folds = []
        total = len(signal)
        min_train = max(120, window_size + 40)

        for k in range(n_folds):
            val_end = total - (n_folds - 1 - k) * val_len
            val_start = val_end - val_len
            train_end = val_start

            if train_end < min_train:
                continue

            train_sig = signal[:train_end]
            val_sig = signal[val_start - window_size:val_end]

            X_train, y_train, _ = self._make_windows(
                train_sig,
                np.arange(len(train_sig), dtype=float),
                window_size,
                horizon,
            )
            X_val, y_val, _ = self._make_windows(
                val_sig,
                np.arange(len(val_sig), dtype=float),
                window_size,
                horizon,
            )

            if len(X_train) == 0 or len(X_val) == 0:
                continue

            folds.append({
                "neuron_name": name,
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
            })

        if debug:
            print(f"[BO Debug] built {len(folds)} folds for neuron={name}")

        return folds

    # -------------------------------------------------------------------
    # loaders
    # -------------------------------------------------------------------

    def _load_hr(self) -> None:
        print("\n[DataLoader] Generating Hindmarsh–Rose data")

        total = int(config.HR_TOTAL_STEPS)
        burn = int(config.HR_TRANSIENT)
        dt = float(config.HR_DT)

        params = _get_hr_params()
        mode = getattr(config, "HR_MODE", "periodic_spiking")

        traj = _rk4_hr(
            x0=params["x0"],
            n_steps=total + burn,
            dt=dt,
            params=params,
        )

        traj = traj[burn:]

        self.time = np.arange(len(traj)) * dt
        self.neuron_names = ["hr_x", "hr_y", "hr_z"]
        self.data_raw = traj

        self.n_samples = self.data_raw.shape[0]
        self.n_neurons = self.data_raw.shape[1]

        print(f"[DataLoader] HR mode: {mode}")
        print(f"[DataLoader] HR input current I: {params['I']}")
        print(f"[DataLoader] Loaded HR: {self.n_samples} steps x {self.n_neurons} states")
        print(f"[DataLoader] Time: {self.time[0]:.2f}s to {self.time[-1]:.2f}s")

    def _load_npy_folder(self) -> None:
        time_path = os.path.join(self.csv_path, self.time_file)
        signal_path = os.path.join(self.csv_path, self.signal_file)

        if not os.path.exists(time_path):
            raise FileNotFoundError(f"Missing time file: {time_path}")
        if not os.path.exists(signal_path):
            raise FileNotFoundError(f"Missing signal file: {signal_path}")

        print(f"\n[DataLoader] Loading npy folder: {self.csv_path}")
        print(f"[DataLoader] Time file   : {os.path.basename(time_path)}")
        print(f"[DataLoader] Signal file : {os.path.basename(signal_path)}")

        self.time = np.load(time_path).astype(float).reshape(-1)
        signal = np.load(signal_path).astype(float)

        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
            names = [self.single_neuron_name]
        elif signal.ndim == 2:
            if signal.shape[0] != len(self.time) and signal.shape[1] == len(self.time):
                signal = signal.T
            names = [f"neuron_{i+1}" for i in range(signal.shape[1])]
        else:
            raise ValueError(f"Unsupported signal shape: {signal.shape}")

        if signal.shape[0] != len(self.time):
            raise ValueError(
                f"Time length {len(self.time)} does not match signal length {signal.shape[0]}"
            )

        self.data_raw = signal
        self.neuron_names = names
        self.n_samples = signal.shape[0]
        self.n_neurons = signal.shape[1]

        duration_s = self.time[-1] - self.time[0] if len(self.time) > 1 else 0.0
        print(f"[DataLoader] Loaded: {self.n_samples} time-steps x {self.n_neurons} neuron(s)")
        print(f"[DataLoader] Time: {self.time[0]:.2f}s to {self.time[-1]:.2f}s ({duration_s/60:.2f} min)")

    def _load_csv(self) -> None:
        print(f"\n[DataLoader] Loading: {self.csv_path}")
        self.raw_df = pd.read_csv(self.csv_path, header=0)

        self.time = self.raw_df.iloc[:, 0].values.astype(float)
        self.neuron_names = list(self.raw_df.columns[1:])
        self.data_raw = self.raw_df.iloc[:, 1:].values.astype(float)

        self.n_samples = self.data_raw.shape[0]
        self.n_neurons = self.data_raw.shape[1]

        duration_s = self.time[-1] - self.time[0] if len(self.time) > 1 else 0.0
        print(f"[DataLoader] Loaded: {self.n_samples} time-steps x {self.n_neurons} neurons")
        print(f"[DataLoader] Time: {self.time[0]:.2f}s to {self.time[-1]:.2f}s ({duration_s/60:.2f} min)")

    # -------------------------------------------------------------------
    # preprocessing
    # -------------------------------------------------------------------

    def _compute_dff(self) -> None:
        data = np.asarray(self.data_raw, dtype=float)

        baseline = np.percentile(data, config.BASELINE_PERCENTILE, axis=0)
        baseline = np.asarray(baseline, dtype=float)
        baseline[np.abs(baseline) < config.EPS] = config.EPS

        self.dff = (data - baseline) / np.abs(baseline)

    def _normalise(self) -> None:
        data = np.asarray(self.dff, dtype=float)

        method = getattr(config, "NORMALIZATION_METHOD", "zscore").lower()

        if method == "zscore":
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
            std[std < config.EPS] = 1.0
            self.data_norm = (data - mean) / std
            return

        if method == "minmax":
            lo = data.min(axis=0, keepdims=True)
            hi = data.max(axis=0, keepdims=True)
            scale = hi - lo
            scale[scale < config.EPS] = 1.0
            self.data_norm = 2.0 * (data - lo) / scale - 1.0
            return

        self.data_norm = data.copy()

    # -------------------------------------------------------------------
    # spike helpers
    # -------------------------------------------------------------------

    def _detect_hr_spikes(self, x_norm):
        x_norm = np.asarray(x_norm, dtype=float).reshape(-1)

        peaks, _ = find_peaks(
            x_norm,
            prominence=0.5,
            distance=config.MIN_SPIKE_DISTANCE,
        )

        return peaks.astype(int)

    def _detect_single_neuron(self, signal):
        signal = np.asarray(signal, dtype=float).reshape(-1)

        threshold = np.mean(signal) + config.SPIKE_THRESHOLD_STD * np.std(signal)

        peaks, _ = find_peaks(
            signal,
            height=threshold,
            distance=config.MIN_SPIKE_DISTANCE,
        )

        return peaks.astype(int)

    # -------------------------------------------------------------------
    # small helpers
    # -------------------------------------------------------------------

    def _resolve(self, neuron_id):
        if isinstance(neuron_id, str):
            if neuron_id not in self.neuron_names:
                raise ValueError(f"Unknown neuron name: {neuron_id}")
            idx = self.neuron_names.index(neuron_id)
            return idx, neuron_id

        idx = int(neuron_id)
        if idx < 0 or idx >= self.n_neurons:
            raise IndexError(f"Neuron index out of range: {idx}")

        return idx, self.neuron_names[idx]

    def _make_windows(self, signal, time, window_size, horizon):
        signal = np.asarray(signal, dtype=float).reshape(-1)
        time = np.asarray(time, dtype=float).reshape(-1)

        X = []
        y = []
        ty = []

        last = len(signal) - window_size - horizon + 1

        for i in range(max(0, last)):
            X.append(signal[i:i + window_size])
            target_idx = i + window_size + horizon - 1
            y.append(signal[target_idx])
            ty.append(time[target_idx])

        return np.asarray(X), np.asarray(y).reshape(-1, 1), np.asarray(ty)