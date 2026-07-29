from __future__ import annotations

import csv
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from plotting import plot_optimizer_convergence


RUN_DIRECTORY = (
    REPO_ROOT / "FINAL_THESIS_RUN" / "02_bo_optimization"
)

REGIMES = (
    "periodic_spiking",
    "periodic_bursting",
    "chaotic_bursting",
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing optimisation history: {path}")

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def main() -> None:
    for regime in REGIMES:
        regime_directory = RUN_DIRECTORY / regime
        history_path = regime_directory / "optimizer_history.csv"

        rows = read_csv_rows(history_path)

        # The plotting utility saves into config.OUTPUT_DIR.
        config.OUTPUT_DIR = str(regime_directory)

        plot_optimizer_convergence(
            rows,
            filename="optimizer_convergence.png",
        )

        print(
            "Regenerated:",
            regime_directory / "optimizer_convergence.png",
        )


if __name__ == "__main__":
    main()