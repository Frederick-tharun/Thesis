"""Thesis-quality figures for the long-horizon (280 time-unit) RR/RC/CC/CR benchmark.

Reads only saved results; never touches final-test trajectories.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config

RESULTS_DIR = config.PACKAGE_ROOT / "results" / "final_evaluation_long"
DT = 0.01

LABELS = {
    "RR": "RR   Regular-trained -> unseen regular   (control)",
    "RC": "RC   Regular-trained -> unseen chaotic   (cross-regime result)",
    "CC": "CC   Chaotic-trained -> unseen chaotic   (control)",
    "CR": "CR   Chaotic-trained -> unseen regular   (cross-regime result)",
}
COLORS = {"RR": "#0072B2", "RC": "#009E73", "CC": "#D55E00", "CR": "#CC79A7"}

PART1_UNSEEN_SHORT = {"median_nrmse": 0.5856, "vpt_time": 17.01}
PART1_UNSEEN_LONG = {"median_nrmse": 0.7544, "vpt_time": 14.59}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def _load(direction, current):
    slug = f"{current:.6f}".replace(".", "p")
    return np.load(RESULTS_DIR / f"long_{direction}_I{slug}.npz")


def load_results():
    return json.loads((RESULTS_DIR / "final_long_horizon_results.json").read_text())


def _rollout(results, direction, current):
    rollouts = results["directions"][direction]["rollouts"]
    return min(rollouts, key=lambda r: abs(r["current"] - current))


def figure_long_horizon(results, out_path):
    picks = {"RR": 2.000000, "RC": 3.383459, "CC": 3.383459, "CR": 3.290000}
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for ax, direction in zip(axes, ["RR", "RC", "CC", "CR"]):
        npz = _load(direction, picks[direction])
        truth, pred = npz["truth"], npz["prediction"]
        t = np.arange(len(truth)) * DT
        ax.plot(t, truth[:, 0], color="#111111", lw=0.7, label="Hindmarsh-Rose truth")
        ax.plot(t, pred[:, 0], color=COLORS[direction], lw=0.7, alpha=0.85,
                label="ESN autonomous prediction")
        vpt = float(npz["vpt_time"])
        ax.axvline(vpt, color="grey", ls=":", lw=1)
        ax.text(vpt + 3, ax.get_ylim()[1] * 0.7, f"pointwise VPT {vpt:.0f} tu",
                fontsize=7.5, color="grey", va="top")
        r = _rollout(results, direction, picks[direction])
        pc, tc = r["predicted_climate"], r["true_climate"]
        ax.set_title(
            f"{LABELS[direction]}    |    I = {picks[direction]:.4g}    |    "
            f"spikes {pc['spike_count']} predicted vs {tc['spike_count']} true    |    "
            f"mean ISI {pc['mean_isi']:.1f} vs {tc['mean_isi']:.1f} tu",
            loc="left")
        ax.set_ylabel("membrane potential  x")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel(
        "time (model time units)  -  full 280-unit autonomous forecast, no re-warming, no truth fed back")
    fig.suptitle(
        "Chapter 2 Part 2 - cross-regime generalisation over a 280-time-unit autonomous horizon\n"
        "warm-up [70000,72000), then 27,999 fully autonomous steps (identical window to Chapter 2 Part 1 Step 8)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path)
    plt.close(fig)


def figure_asymmetry(results, out_path):
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.55, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    for i, d in enumerate(["RR", "RC", "CC", "CR"]):
        errs = []
        for r in results["directions"][d]["rollouts"]:
            pc, tc = r["predicted_climate"], r["true_climate"]
            if pc.get("mean_isi") and tc.get("mean_isi"):
                errs.append(abs(pc["mean_isi"] - tc["mean_isi"]) / tc["mean_isi"] * 100)
        ax.scatter([i] * len(errs), errs, color=COLORS[d], s=55, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(["RR", "RC", "CC", "CR"])
    ax.set_yscale("symlog", linthresh=1)
    ax.axhline(5, color="grey", ls="--", lw=1)
    ax.text(-0.35, 6, "5% error", fontsize=7.5, color="grey")
    ax.set_ylabel("mean-ISI error vs truth (%)")
    ax.set_title("Rhythm fidelity\nlower = correct attractor", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    for i, d in enumerate(["RR", "RC", "CC", "CR"]):
        ratios = [r["spike_count_ratio"] for r in results["directions"][d]["rollouts"] if r["spike_count_ratio"]]
        ax.scatter([i] * len(ratios), ratios, color=COLORS[d], s=55, zorder=3)
    ax.axhline(1.0, color="#111111", lw=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(["RR", "RC", "CC", "CR"])
    ax.set_yscale("log")
    ax.set_ylabel("predicted / true spike count")
    ax.set_title("Spike-count fidelity\n1.0 = exactly right", loc="left")

    ax = fig.add_subplot(gs[0, 2])
    for i, d in enumerate(["RR", "RC", "CC", "CR"]):
        vpts = [r["metrics"]["valid_prediction_steps"] * DT for r in results["directions"][d]["rollouts"]]
        ax.scatter([i] * len(vpts), vpts, color=COLORS[d], s=55, zorder=3)
    ax.axhline(PART1_UNSEEN_LONG["vpt_time"], color="#111111", ls="--", lw=1.2)
    ax.text(-0.35, PART1_UNSEEN_LONG["vpt_time"] + 3,
            "Part 1 unseen-current baseline (14.6 tu)", fontsize=7.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(["RR", "RC", "CC", "CR"])
    ax.set_ylabel("valid prediction time (tu)")
    ax.set_title("Pointwise horizon\nhigher = better", loc="left")

    contrasts = [
        (3.290000, "INTERPOLATION\ninside chaotic training span"),
        (3.443609, "NEAR EDGE\njust above training span"),
        (1.796992, "FAR EXTRAPOLATION\nfar below training span"),
    ]
    for col, (current, verdict) in enumerate(contrasts):
        ax = fig.add_subplot(gs[1, col])
        npz = _load("CR", current)
        truth, pred = npz["truth"], npz["prediction"]
        t = np.arange(len(truth)) * DT
        ax.plot(t, truth[:, 0], color="#111111", lw=0.6, label="truth")
        ax.plot(t, pred[:, 0], color=COLORS["CR"], lw=0.6, alpha=0.85, label="prediction")
        r = _rollout(results, "CR", current)
        pc, tc = r["predicted_climate"], r["true_climate"]
        ok = 0.9 <= (r["spike_count_ratio"] or 0) <= 1.1
        mark = "CORRECT RHYTHM" if ok else "WRONG ATTRACTOR"
        ax.set_title(
            f"CR,  I = {current:.4g}\n{verdict}\n"
            f"spikes {pc['spike_count']} vs {tc['spike_count']}  ->  {mark}",
            loc="left", fontsize=9, color=("#111111" if ok else "#B22222"))
        ax.set_xlabel("time (tu)")
        if col == 0:
            ax.set_ylabel("x")
            ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(
        "The asymmetry: regular -> chaotic transfers well; chaotic -> regular transfers only by interpolation",
        fontsize=12.5)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure_context(results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    names = ["Part 1\nunseen_short", "Part 1\nunseen_long", "Part 2\nRR", "Part 2\nRC", "Part 2\nCC", "Part 2\nCR"]
    nrmse = [PART1_UNSEEN_SHORT["median_nrmse"], PART1_UNSEEN_LONG["median_nrmse"],
             *[results["directions"][d]["summary"]["median_nrmse"] for d in ["RR", "RC", "CC", "CR"]]]
    vpt = [PART1_UNSEEN_SHORT["vpt_time"], PART1_UNSEEN_LONG["vpt_time"],
           *[results["directions"][d]["summary"]["median_vpt_time"] for d in ["RR", "RC", "CC", "CR"]]]
    cols = ["#999999", "#555555", COLORS["RR"], COLORS["RC"], COLORS["CC"], COLORS["CR"]]

    axes[0].bar(names, nrmse, color=cols)
    axes[0].set_ylabel("median NRMSE (lower = better)")
    axes[0].set_title("Prediction error on genuinely unseen currents", loc="left")
    axes[0].tick_params(axis="x", labelsize=8)

    axes[1].bar(names, vpt, color=cols)
    axes[1].set_ylabel("median valid prediction time (tu)")
    axes[1].set_title("Pointwise accuracy horizon (higher = better)", loc="left")
    axes[1].tick_params(axis="x", labelsize=8)

    fig.suptitle(
        "Part 2 in context: Chapter 2 Part 1's own unseen-current benchmark is the fair comparison\n"
        "(Part 1's widely quoted NRMSE = 0.005 was measured on the currents it was trained on)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path)
    plt.close(fig)


def write_table(results, out_path):
    roles = {"RR": "control", "RC": "**cross-regime result**", "CC": "control", "CR": "**cross-regime result**"}
    lines = [
        "# Chapter 2 Part 2 - long-horizon cross-regime results (280 time units)",
        "",
        "| Direction | Role | Median NRMSE | Median VPT (tu) | Diverged | Spike count within 10% |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for d in ["RR", "RC", "CC", "CR"]:
        s = results["directions"][d]["summary"]
        lines.append(
            f"| {d} | {roles[d]} | {s['median_nrmse']:.3g} | {s['median_vpt_time']:.1f} | "
            f"{s['divergence_count']}/{s['n_rollouts']} | {s['spike_ratio_within_10pct']}/{s['n_rollouts']} |")
    lines += [
        "",
        "Reference - Chapter 2 Part 1, parameter-aware ESN, genuinely unseen currents:",
        "",
        "| Part 1 family | Median NRMSE | Mean VPT (tu) | Divergence |",
        "|---|---:|---:|---:|",
        "| unseen_short (80 tu) | 0.586 | 17.0 | 6/30 |",
        "| unseen_long (280 tu) | 0.754 | 14.6 | 2/10 |",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    results = load_results()
    figure_long_horizon(results, RESULTS_DIR / "fig1_cross_regime_long_horizon.png")
    figure_asymmetry(results, RESULTS_DIR / "fig2_cross_regime_asymmetry.png")
    figure_context(results, RESULTS_DIR / "fig3_cross_regime_context.png")
    write_table(results, RESULTS_DIR / "long_horizon_summary_table.md")
    print(f"wrote 3 figures + table under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
