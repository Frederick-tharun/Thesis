import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def normalize_01(x):
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def binned_partial_curve(x, score, bins=18):
    x = np.asarray(x, dtype=float)
    score = np.asarray(score, dtype=float)

    edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.full(bins, np.nan)

    for i in range(bins):
        if i < bins - 1:
            mask = (x >= edges[i]) & (x < edges[i + 1])
        else:
            mask = (x >= edges[i]) & (x <= edges[i + 1])

        if np.any(mask):
            vals[i] = np.nanmin(score[mask])

    valid = np.isfinite(vals)
    return centers[valid], vals[valid]


def diagonal_partial(ax, df, col, score_col, best_row, label_map):
    x = df[col].to_numpy(dtype=float)
    score = df[score_col].to_numpy(dtype=float)

    xc, yc = binned_partial_curve(x, score, bins=18)

    ax.plot(xc, yc, linewidth=1.8)
    ax.scatter(x, score, s=8, alpha=0.25)
    ax.axvline(best_row[col], color="red", linestyle="--", linewidth=1.0)
    ax.scatter(best_row[col], best_row[score_col], c="red", s=30, zorder=5)

    ax.set_title(label_map.get(col, col), fontsize=11, pad=6)
    ax.set_ylabel("Partial dep.")
    ax.tick_params(labelsize=8)
    ax.locator_params(axis="x", nbins=4)
    ax.locator_params(axis="y", nbins=4)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(False)


def pair_surface(ax, df, xcol, ycol, score_col, best_row, label_map):
    pair = df[[xcol, ycol, score_col]].dropna().copy()
    pair = pair.groupby([xcol, ycol], as_index=False)[score_col].min()

    x = pair[xcol].to_numpy(dtype=float)
    y = pair[ycol].to_numpy(dtype=float)
    z = pair[score_col].to_numpy(dtype=float)

    if len(x) >= 3:
        try:
            triang = mtri.Triangulation(x, y)
            mappable = ax.tricontourf(triang, z, levels=18, cmap="viridis")
        except Exception:
            mappable = ax.scatter(x, y, c=z, cmap="viridis", s=18)
    else:
        mappable = ax.scatter(x, y, c=z, cmap="viridis", s=18)

    ax.scatter(x, y, s=8, c="black", alpha=0.85)
    ax.scatter(best_row[xcol], best_row[ycol], s=45, c="red", marker="o", zorder=5)

    ax.set_xlabel(label_map.get(xcol, xcol))
    ax.set_ylabel(label_map.get(ycol, ycol))
    ax.tick_params(labelsize=8)
    ax.locator_params(axis="x", nbins=4)
    ax.locator_params(axis="y", nbins=4)
    ax.grid(False)

    return mappable


def main():
    parser = argparse.ArgumentParser(description="Create paper-style BO objective landscape plot.")
    parser.add_argument("--csv", required=True, help="Path to optimizer_results.csv")
    parser.add_argument(
        "--out",
        required=True,
        help="Output PNG path, e.g. outputs/periodic_spiking/bo_objective_landscape_paper_style.png",
    )
    parser.add_argument(
        "--optimizer",
        default=None,
        help="Optional optimizer filter, e.g. gp",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if args.optimizer is not None and "optimizer" in df.columns:
        df = df[df["optimizer"].astype(str).str.lower() == args.optimizer.lower()].copy()

    param_cols = ["spectral_radius", "leaky_coefficient", "input_scaling"]
    score_col = "score"

    missing = [c for c in param_cols + [score_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[param_cols + [score_col]].dropna().copy()
    if df.empty:
        raise ValueError("No rows left after filtering/dropping NaNs.")

    df["score_norm"] = normalize_01(df[score_col].to_numpy(dtype=float))

    best_idx = df[score_col].idxmin()
    best_row = df.loc[best_idx].copy()
    best_row["score_norm"] = df.loc[best_idx, "score_norm"]

    label_map = {
        "spectral_radius": r"$\rho$",
        "leaky_coefficient": r"$\alpha$",
        "input_scaling": r"$\gamma$",
    }

    n = len(param_cols)
    fig, axes = plt.subplots(
        n,
        n,
        figsize=(8.5, 8.0),
        gridspec_kw={"wspace": 0.18, "hspace": 0.18},
    )

    mappable = None

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if i < j:
                ax.axis("off")
                continue

            if i == j:
                diagonal_partial(
                    ax=ax,
                    df=df,
                    col=param_cols[i],
                    score_col="score_norm",
                    best_row=best_row,
                    label_map=label_map,
                )
            else:
                mappable = pair_surface(
                    ax=ax,
                    df=df,
                    xcol=param_cols[j],
                    ycol=param_cols[i],
                    score_col="score_norm",
                    best_row=best_row,
                    label_map=label_map,
                )

    cbar = fig.colorbar(mappable, ax=axes, fraction=0.03, pad=0.04)
    cbar.set_label("Value of obj. func. (normalized)")

    title = "BO objective landscape (gp)"
    subtitle = (
        f"Best score={best_row['score']:.4e}; "
        f"ρ={best_row['spectral_radius']:.4g}, "
        f"α={best_row['leaky_coefficient']:.4g}, "
        f"γ={best_row['input_scaling']:.4g}"
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.95, 0.93])
    plt.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()