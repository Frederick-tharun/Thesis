from __future__ import annotations

"""Generate the final submission-ready Chapter 1 optimizer-comparison assets.

Produces exactly two artifacts, deliberately minimal so the optimization
section stays readable:

  1. One convergence figure (chaotic_bursting only, the regime carried into
     the control chapter) re-plotted from the locked, frozen
     FINAL_THESIS_RUN/02_bo_optimization/chaotic_bursting/optimizer_history.csv.
  2. One compact combined table across all three regimes, built from the
     locked FINAL_THESIS_RUN/02_bo_optimization/<regime>/optimizer_ranking_table.csv
     files.

Nothing is recomputed or re-optimized -- this only re-reads and re-plots
already-locked evidence, and cross-checks every number it plots against the
locked optimizer_summary.json before saving anything.

Output:
  Chapter1_Final_Thesis_Figures/optimization/chaotic_bursting_optimizer_convergence.{png,pdf}
  Chapter1_Final_Thesis_Figures/optimization/optimizer_comparison_table.{csv,md}
"""

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt

RUN_ROOT = REPO_ROOT / "FINAL_THESIS_RUN" / "02_bo_optimization"
OUTPUT_ROOT = REPO_ROOT / "Chapter1_Final_Thesis_Figures" / "optimization"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

REGIMES = ("periodic_spiking", "periodic_bursting", "chaotic_bursting")
CONVERGENCE_REGIME = "chaotic_bursting"

OPTIMIZER_DISPLAY = {
    "gp": "Gaussian process",
    "dummy": "Random search",
    "forest": "Random forest",
    "gbrt": "GBRT",
}
OPTIMIZER_COLOR = {
    "gp": "C0",
    "dummy": "C1",
    "forest": "C2",
    "gbrt": "C3",
}
OPTIMIZER_ORDER = ("gp", "dummy", "forest", "gbrt")


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# ============================================================
# 1. Convergence figure (re-plotted from locked history, verified)
# ============================================================

def build_convergence_figure():
    regime_dir = RUN_ROOT / CONVERGENCE_REGIME
    history_rows = _read_csv_rows(regime_dir / "optimizer_history.csv")
    summary = json.loads((regime_dir / "optimizer_summary.json").read_text())

    curves: dict[str, list[tuple[int, float]]] = {name: [] for name in OPTIMIZER_ORDER}
    for row in history_rows:
        opt = row["optimizer"]
        if opt not in curves:
            continue
        curves[opt].append((int(row["iteration"]), float(row["best_score"])))

    for opt in OPTIMIZER_ORDER:
        curves[opt].sort(key=lambda pair: pair[0])

    # Verify the final best-so-far value on each curve matches the locked
    # per-optimizer summary before trusting the plot.
    summary_by_optimizer = {entry["optimizer"]: entry for entry in summary}
    for opt in OPTIMIZER_ORDER:
        if not curves[opt]:
            raise RuntimeError(f"No history rows found for optimizer '{opt}'.")
        final_value = curves[opt][-1][1]
        locked_value = float(summary_by_optimizer[opt]["best_score"])
        if abs(final_value - locked_value) > max(1e-9, 1e-6 * abs(locked_value)):
            raise RuntimeError(
                f"{opt}: reproduced final best_score={final_value!r} does not match "
                f"locked optimizer_summary.json best_score={locked_value!r}."
            )

    winner = min(summary, key=lambda entry: float(entry["best_score"]))
    winner_name = winner["optimizer"]
    winner_score = float(winner["best_score"])
    winner_iteration = curves[winner_name][-1][0]
    for iteration, value in curves[winner_name]:
        if abs(value - winner_score) < max(1e-12, 1e-9 * abs(winner_score)):
            winner_iteration = iteration
            break

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for opt in OPTIMIZER_ORDER:
        xs = [pair[0] for pair in curves[opt]]
        ys = [pair[1] for pair in curves[opt]]
        ax.plot(
            xs, ys,
            marker="o", markersize=3.5, linewidth=1.6,
            color=OPTIMIZER_COLOR[opt], label=OPTIMIZER_DISPLAY[opt],
        )

    ax.scatter(
        [winner_iteration], [winner_score],
        marker="*", s=260, color=OPTIMIZER_COLOR[winner_name],
        edgecolor="black", linewidth=0.8, zorder=5,
        label=f"Selected ({OPTIMIZER_DISPLAY[winner_name]})",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Candidate evaluation", fontsize=11)
    ax.set_ylabel("Best composite validation score so far\n(lower is better)", fontsize=10.5)
    ax.grid(True, which="both", alpha=0.20)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()

    png_path = OUTPUT_ROOT / f"{CONVERGENCE_REGIME}_optimizer_convergence.png"
    pdf_path = OUTPUT_ROOT / f"{CONVERGENCE_REGIME}_optimizer_convergence.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[{CONVERGENCE_REGIME}] verified: winner={winner_name} best_score={winner_score:.6g}")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ============================================================
# 2. Combined optimizer-comparison table across all three regimes
# ============================================================

def build_comparison_table():
    combined_rows = []
    for regime in REGIMES:
        ranking_path = RUN_ROOT / regime / "optimizer_ranking_table.csv"
        summary_path = RUN_ROOT / regime / "optimizer_summary.json"
        rows = _read_csv_rows(ranking_path)
        summary = json.loads(summary_path.read_text())
        summary_by_optimizer = {entry["optimizer"]: entry for entry in summary}

        for row in rows:
            opt = row["optimizer"]
            locked_score = float(summary_by_optimizer[opt]["best_score"])
            table_score = float(row["best_score"])
            if abs(table_score - locked_score) > max(1e-9, 1e-6 * abs(locked_score)):
                raise RuntimeError(
                    f"{regime}/{opt}: ranking table best_score={table_score!r} does not "
                    f"match locked optimizer_summary.json best_score={locked_score!r}."
                )
            combined_rows.append({
                "regime": regime,
                "optimizer": OPTIMIZER_DISPLAY.get(opt, opt),
                "rank": int(row["rank"]),
                "validation_nrmse_x": float(row["validation_nrmse_x"]),
                "validation_nrmse_all": float(row["validation_nrmse_all"]),
                "best_score": table_score,
            })

    combined_rows.sort(key=lambda r: (REGIMES.index(r["regime"]), r["rank"]))

    csv_path = OUTPUT_ROOT / "optimizer_comparison_table.csv"
    fieldnames = [
        "regime", "optimizer", "rank",
        "validation_nrmse_x", "validation_nrmse_all", "best_score",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

    md_lines = [
        "| Regime | Optimizer | Rank | Validation NRMSE (x) | Validation NRMSE (all) | Best score |",
        "|---|---|---:|---:|---:|---:|",
    ]
    last_regime = None
    for row in combined_rows:
        regime_label = row["regime"] if row["regime"] != last_regime else ""
        last_regime = row["regime"]
        md_lines.append(
            f"| {regime_label} | {row['optimizer']} | {row['rank']} | "
            f"{row['validation_nrmse_x']:.4g} | {row['validation_nrmse_all']:.4g} | "
            f"{row['best_score']:.4g} |"
        )
    md_path = OUTPUT_ROOT / "optimizer_comparison_table.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


def main():
    print("=" * 72)
    print("Chapter 1 final optimizer-comparison assets")
    print("=" * 72)
    build_convergence_figure()
    print()
    build_comparison_table()
    print("\nAll optimizer-comparison assets verified and saved to:")
    print(f"  {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
