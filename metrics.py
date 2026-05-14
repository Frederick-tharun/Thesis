from __future__ import annotations
import numpy as np

LARGE_PENALTY = 1e6
EPS = 1e-12


# ── small helpers ─────────────────────────────────────────────────────

def _v(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = _v(a)
    b = _v(b)
    n = min(len(a), len(b))
    return a[:n], b[:n]


# ── basic regression metrics ──────────────────────────────────────────

def mse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _align(a, b)
    return float(np.mean((a - b) ** 2))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mse(a, b)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _align(a, b)
    return float(np.mean(np.abs(a - b)))


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _align(a, b)
    return rmse(a, b) / (float(np.std(a)) + EPS)


def r2(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _align(a, b)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2)) + EPS
    return 1.0 - ss_res / ss_tot


def mape(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _align(a, b)
    mask = np.abs(a) > 1e-8
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100.0)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _align(a, b)
    if len(a) < 2:
        return float("nan")
    sa = np.std(a)
    sb = np.std(b)
    if sa < EPS or sb < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ── spike metrics ─────────────────────────────────────────────────────

def spike_threshold(signal: np.ndarray, n_sigma: float = 2.0) -> float:
    signal = _v(signal)
    return float(np.mean(signal) + n_sigma * np.std(signal))


def spike_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_sigma: float = 2.0,
) -> dict:
    y_true, y_pred = _align(y_true, y_pred)

    thr = spike_threshold(y_true, n_sigma)

    true_mask = y_true > thr
    pred_mask = y_pred > thr

    tp = int((true_mask & pred_mask).sum())
    tn = int((~true_mask & ~pred_mask).sum())
    fp = int((~true_mask & pred_mask).sum())
    fn = int((true_mask & ~pred_mask).sum())

    accuracy = (tp + tn) / (tp + tn + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)

    true_times = np.where(true_mask)[0]
    pred_times = np.where(pred_mask)[0]

    if len(true_times) > 0 and len(pred_times) > 0:
        timing_err = float(np.mean([np.min(np.abs(pred_times - t)) for t in true_times]))
    else:
        timing_err = float("nan")

    amp_err = (
        float(np.mean(np.abs(y_true[true_mask] - y_pred[true_mask])))
        if true_mask.any()
        else float("nan")
    )

    return {
        "threshold": float(thr),
        "n_true": int(true_mask.sum()),
        "n_pred": int(pred_mask.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "timing_err": timing_err,
        "amp_err": amp_err,
    }


# ── recursive-only helpers for BO ─────────────────────────────────────

def tail_nrmse(y_true: np.ndarray, y_pred: np.ndarray, tail_frac: float = 0.5) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    n = len(y_true)
    k = max(1, int(n * tail_frac))
    return nrmse(y_true[-k:], y_pred[-k:])


def variance_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.std(y_pred) / (np.std(y_true) + EPS))


def mean_shift(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    return float(abs(np.mean(y_pred) - np.mean(y_true)) / (np.std(y_true) + EPS))


def range_violation_penalty(
    y_pred: np.ndarray,
    ref_signal: np.ndarray,
    margin: float = 0.25,
) -> float:
    y_pred = _v(y_pred)
    ref_signal = _v(ref_signal)

    lo = float(np.min(ref_signal)) - margin
    hi = float(np.max(ref_signal)) + margin

    below = np.maximum(lo - y_pred, 0.0)
    above = np.maximum(y_pred - hi, 0.0)
    return float(np.mean(below + above))


def flatline_penalty(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vr = variance_ratio(y_true, y_pred)

    # good if variance ratio is around 1
    # punish very low variance strongly (collapse)
    if vr < 0.10:
        return 2.0
    if vr < 0.25:
        return 1.0
    return float(abs(np.log(vr + EPS)))


def spike_count_penalty(y_true: np.ndarray, y_pred: np.ndarray, n_sigma: float = 2.0) -> float:
    sm = spike_metrics(y_true, y_pred, n_sigma=n_sigma)
    return float(abs(sm["n_pred"] - sm["n_true"]) / (sm["n_true"] + 1.0))


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray, n_sigma: float = 2.0) -> float:
    sm = spike_metrics(y_true, y_pred, n_sigma=n_sigma)
    return float(sm["fp"] / (sm["fp"] + sm["tn"] + EPS))


# ── evaluation bundle ─────────────────────────────────────────────────

def evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_sigma: float = 2.0,
) -> dict:
    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "pearson": pearson_corr(y_true, y_pred),
        **spike_metrics(y_true, y_pred, n_sigma=n_sigma),
    }


# ── BO score: recursive-first ─────────────────────────────────────────

def bo_score(
    y_true: np.ndarray,
    one_pred: np.ndarray,
    rec_pred: np.ndarray,
    ref_signal: np.ndarray | None = None,
) -> float:
    """
    Lower is better.

    Stronger focus on recursive behavior:
    - recursive NRMSE
    - late-horizon recursive NRMSE
    - flatline / collapse
    - wrong mean level
    - false positives
    - spike-count mismatch
    """
    try:
        y_true, one_pred = _align(y_true, one_pred)
        _, rec_pred = _align(y_true, rec_pred)

        n = min(len(y_true), len(one_pred), len(rec_pred))
        y_true = y_true[:n]
        one_pred = one_pred[:n]
        rec_pred = rec_pred[:n]

        s_one = nrmse(y_true, one_pred)
        s_rec = nrmse(y_true, rec_pred)
        s_tail = tail_nrmse(y_true, rec_pred, tail_frac=0.5)

        s_flat = flatline_penalty(y_true, rec_pred)
        s_mean = mean_shift(y_true, rec_pred)
        s_fp = false_positive_rate(y_true, rec_pred)
        s_cnt = spike_count_penalty(y_true, rec_pred)

        s_range = 0.0
        if ref_signal is not None:
            s_range = range_violation_penalty(rec_pred, ref_signal)

        score = (
            0.10 * s_one +
            0.30 * s_rec +
            0.25 * s_tail +
            0.10 * s_flat +
            0.10 * s_mean +
            0.08 * s_fp +
            0.05 * s_cnt +
            0.02 * s_range
        )

        if not np.isfinite(score):
            return LARGE_PENALTY
        return float(score)

    except Exception:
        return LARGE_PENALTY


# ── pretty printers ───────────────────────────────────────────────────

def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float) and np.isnan(v):
        return "nan"
    return f"{float(v):.6f}"


def print_metrics(
    y_true: np.ndarray,
    one_pred: np.ndarray,
    rec_pred: np.ndarray,
    label: str = "ESN",
    n_sigma: float = 2.0,
) -> None:
    y_true, one_pred = _align(y_true, one_pred)
    _, rec_pred = _align(y_true, rec_pred)

    n = min(len(y_true), len(one_pred), len(rec_pred))
    y_true = y_true[:n]
    one_pred = one_pred[:n]
    rec_pred = rec_pred[:n]

    one = evaluate_metrics(y_true, one_pred, n_sigma=n_sigma)
    rec = evaluate_metrics(y_true, rec_pred, n_sigma=n_sigma)

    W = 24
    print(f"\n[{label}]")
    print("─" * (W + 28))
    print(f"  {'Metric':<{W}} {'One-step':>12} {'Recursive':>12}")
    print("─" * (W + 28))

    rows = [
        ("RMSE", "rmse"),
        ("MAE", "mae"),
        ("NRMSE", "nrmse"),
        ("R²", "r2"),
        ("MAPE%", "mape"),
        ("Pearson", "pearson"),
        ("Spike Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
        ("True Spikes", "n_true"),
        ("Pred Spikes", "n_pred"),
        ("Timing Err", "timing_err"),
        ("Amplitude Err", "amp_err"),
    ]

    for title, key in rows:
        print(f"  {title:<{W}} {_fmt(one[key]):>12} {_fmt(rec[key]):>12}")

    print("─" * (W + 28))


def print_train_test_metrics(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    label: str = "ESN",
    n_sigma: float = 2.0,
) -> None:
    train = evaluate_metrics(y_train_true, y_train_pred, n_sigma=n_sigma)
    test = evaluate_metrics(y_test_true, y_test_pred, n_sigma=n_sigma)

    W = 24
    print(f"\n[{label} | TRAIN vs TEST]")
    print("─" * (W + 28))
    print(f"  {'Metric':<{W}} {'Train':>12} {'Test':>12}")
    print("─" * (W + 28))

    rows = [
        ("RMSE", "rmse"),
        ("MAE", "mae"),
        ("NRMSE", "nrmse"),
        ("R²", "r2"),
        ("MAPE%", "mape"),
        ("Pearson", "pearson"),
        ("Spike Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
        ("True Spikes", "n_true"),
        ("Pred Spikes", "n_pred"),
        ("Timing Err", "timing_err"),
        ("Amplitude Err", "amp_err"),
    ]

    for title, key in rows:
        print(f"  {title:<{W}} {_fmt(train[key]):>12} {_fmt(test[key]):>12}")

    print("─" * (W + 28))