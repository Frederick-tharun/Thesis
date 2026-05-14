from __future__ import annotations

import argparse
import csv
import json
import os
import numpy as np

import config
from data_loader import DataLoader
from optimize_model import optimize_hyperparameters
from plotting import plot_optimizer_convergence, plot_optimizer_heatmap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="hr")
    p.add_argument("--neuron", type=int, default=0)
    p.add_argument(
        "--optimizers",
        nargs="+",
        default=getattr(config, "OPTIMIZERS_TO_COMPARE", ["gp", "dummy", "forest", "gbrt"]),
    )
    return p.parse_args()


def json_safe(x):
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_safe(v) for v in x]
    if isinstance(x, tuple):
        return [json_safe(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def save_json(obj, filename):
    path = os.path.join(config.OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(json_safe(obj), f, indent=2)
    print(f"[Save] -> {path}")


def save_history_csv(rows, filename="optimizer_history.csv"):
    if not rows:
        return

    path = os.path.join(config.OUTPUT_DIR, filename)
    keys = sorted(set().union(*(r.keys() for r in rows)))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[Save] -> {path}")


def main():
    args = parse_args()
    config.DATASET_MODE = args.dataset.lower()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 72)
    print("LOADING DATA")
    print("=" * 72)

    loader = DataLoader(csv_path=config.DATA_PATH)
    loader.load()
    loader.preprocess()
    loader.detect_spikes()
    loader.summary()

    all_history = []
    summary = []

    for opt in args.optimizers:
        result = optimize_hyperparameters(loader, args.neuron, opt)
        all_history.extend(result.history)

        save_json(result.best_params, f"best_params_{opt}.json")
        summary.append({
            "optimizer": opt,
            "best_score": result.best_score,
            **result.best_params,
        })

    save_json(summary, "optimizer_summary.json")
    save_history_csv(all_history)

    plot_optimizer_convergence(all_history)
    plot_optimizer_heatmap(all_history)

    print("\n" + "=" * 72)
    print("OPTIMIZER SUMMARY")
    print("=" * 72)
    for row in sorted(summary, key=lambda r: r["best_score"]):
        print(
            f"{row['optimizer']:>8}  score={row['best_score']:.6f}  "
            f"x_nrmse={row['validation_nrmse_x']:.6f}  "
            f"N={row['N_res']}  rho={row['spectral_radius']:.3f}  "
            f"leak={row['leaky_coefficient']:.3f}  scale={row['input_scaling']:.3f}"
        )
    print("[Done]")


if __name__ == "__main__":
    main()
