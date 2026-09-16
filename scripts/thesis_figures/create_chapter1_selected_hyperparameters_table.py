from __future__ import annotations

"""Build the Chapter 1 selected-ESN-hyperparameters table.

For each regime, reads FINAL_THESIS_RUN/02_bo_optimization/<regime>/best_params.json
and cross-checks every value against the actual locked, hash-verified
model_bundle.npz (EchoStateNetwork.load_bundle verifies its own identity
hash on load) -- so the table reports the literal parameters of the trained
model, not just a claim in a JSON file.

Output: Chapter1_Final_Thesis_Figures/optimization/selected_hyperparameters_table.{csv,md}
"""

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import EchoStateNetwork

RUN_ROOT = REPO_ROOT / "FINAL_THESIS_RUN"
OUTPUT_ROOT = REPO_ROOT / "Chapter1_Final_Thesis_Figures" / "optimization"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

REGIMES = ("periodic_spiking", "periodic_bursting", "chaotic_bursting")

REGIME_DISPLAY = {
    "periodic_spiking": "Periodic spiking",
    "periodic_bursting": "Periodic bursting",
    "chaotic_bursting": "Chaotic bursting",
}

FIELDS = (
    ("N_res", "Reservoir size", "{:d}"),
    ("spectral_radius", "Spectral radius", "{:.4f}"),
    ("leaky_coefficient", "Leak rate", "{:.4f}"),
    ("input_scaling", "Input scaling", "{:.4f}"),
    ("p", "Sparsity", "{:.4f}"),
    ("regularization", "Ridge regularization", "{:.3e}"),
    ("washout", "Washout", "{:d}"),
)


def build_table():
    rows = []
    for regime in REGIMES:
        opt_dir = RUN_ROOT / "02_bo_optimization" / regime
        pred_dir = RUN_ROOT / "01_prediction_all_regimes" / regime

        best_params = json.loads((opt_dir / "best_params.json").read_text())
        selected_model = json.loads((pred_dir / "selected_model.json").read_text())

        model, metadata = EchoStateNetwork.load_bundle(pred_dir / "model_bundle.npz")
        if metadata["model_identity_hash"] != selected_model["model_identity_hash"]:
            raise RuntimeError(
                f"{regime}: model_bundle.npz identity hash does not match "
                "selected_model.json -- refusing to report unverified parameters."
            )

        bundle_values = {
            "N_res": model.N_res,
            "spectral_radius": model.spectral_radius,
            "leaky_coefficient": model.leaky_coefficient,
            "input_scaling": model.input_scaling,
            "p": model.p,
            "regularization": model.regularization,
        }
        for key, value in bundle_values.items():
            json_value = float(best_params[key])
            if abs(float(value) - json_value) > max(1e-9, 1e-6 * abs(json_value)):
                raise RuntimeError(
                    f"{regime}: best_params.json {key}={json_value!r} does not match "
                    f"the locked model bundle's {key}={value!r}."
                )

        row = {"regime": regime, "optimizer": selected_model["selected_optimizer"]}
        for key, _, _ in FIELDS:
            row[key] = bundle_values.get(key, best_params.get(key))
        rows.append(row)

    csv_path = OUTPUT_ROOT / "selected_hyperparameters_table.csv"
    fieldnames = ["regime", "optimizer"] + [key for key, _, _ in FIELDS]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    header = ["Regime", "Selected optimizer"] + [label for _, label, _ in FIELDS]
    md_lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in rows:
        cells = [REGIME_DISPLAY[row["regime"]], row["optimizer"].upper()]
        for key, _, fmt in FIELDS:
            value = row[key]
            cells.append(fmt.format(int(value)) if fmt.endswith("d}") else fmt.format(float(value)))
        md_lines.append("| " + " | ".join(cells) + " |")
    md_path = OUTPUT_ROOT / "selected_hyperparameters_table.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    for line in md_lines:
        print(line)


if __name__ == "__main__":
    build_table()
