import argparse
import json
from pathlib import Path

from plotting_better import (
    plot_controlled_vs_uncontrolled_x_better,
    plot_error_and_signal_better,
    plot_k_sweep_results_better,
)


def normalize_run_dir(project_root: Path, run_dir_str: str) -> Path:
    # handles Windows-style backslashes stored in json
    normalized = Path(run_dir_str.replace("\\", "/"))
    if normalized.is_absolute():
        return normalized
    return project_root / normalized


def main():
    parser = argparse.ArgumentParser(description="Regenerate better plots from saved control results.")
    parser.add_argument(
        "--control-root",
        required=True,
        help="Example: outputs/periodic_spiking/control/linear_feedback"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    control_root = (project_root / args.control_root).resolve()

    best_json = control_root / "best_linear_feedback_result.json"
    summary_csv = control_root / "linear_feedback_metrics_summary.csv"

    if not best_json.exists():
        raise FileNotFoundError(f"Missing: {best_json}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing: {summary_csv}")

    with open(best_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    run_dir = normalize_run_dir(project_root, metrics["run_dir"])
    rollout_csv = run_dir / "controlled_rollout.csv"

    if not rollout_csv.exists():
        raise FileNotFoundError(f"Missing rollout CSV: {rollout_csv}")

    output_dir = control_root / "better_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Control root: {control_root}")
    print(f"[Info] Best run dir:  {run_dir}")
    print(f"[Info] Output dir:    {output_dir}")

    plot_controlled_vs_uncontrolled_x_better(
        rollout_csv=str(rollout_csv),
        output_dir=str(output_dir),
        metrics=metrics,
    )

    plot_error_and_signal_better(
        rollout_csv=str(rollout_csv),
        output_dir=str(output_dir),
        metrics=metrics,
    )

    plot_k_sweep_results_better(
        summary_csv=str(summary_csv),
        output_dir=str(output_dir),
    )

    print("\nDone. Better plots were created without rerunning training.")


if __name__ == "__main__":
    main()