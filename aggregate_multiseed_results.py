#!/usr/bin/env python3
"""
Aggregate seed_<seed>/seed_summary.json files produced by multiseed_evaluation.py.

Outputs:
- all_seed_prediction_results.csv
- all_seed_controller_results.csv
- aggregate_prediction_metrics.csv
- aggregate_controller_metrics.csv
- representative_seed.json
- multiseed_summary.json
- multiseed_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PREDICTION_METRICS = (
    "rmse_recursive_x",
    "nrmse_recursive_x",
    "rmse_recursive_all_states",
    "nrmse_recursive_all_states",
)

CONTROLLER_METRICS = (
    "corrected_feedback_input_target_rmse_state",
    "corrected_feedback_input_target_rmse_x",
    "corrected_feedback_input_target_nrmse_x",
    "raw_readout_target_rmse_state",
    "raw_readout_target_rmse_x",
    "raw_readout_target_nrmse_x",
    "control_effort_mean_sq",
    "control_energy_dt_sum",
    "evaluation_time_to_tolerance",
    "settling_time",
    "spike_reduction_percent",
    "pyragas_empirical_recurrence_error_norm",
    "pyragas_empirical_recurrence_correlation",
    "pyragas_empirical_tail_closure_error_norm",
    "pyragas_rhythm_interval_cv",
    "pyragas_x_amplitude_ratio",
    "pyragas_x_std_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("MULTISEED_EVAL"),
        help="Directory containing seed_<seed>/seed_summary.json.",
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def aggregate_values(values: list[Any]) -> dict[str, Any]:
    clean = [
        number
        for value in values
        if (number := finite_float(value)) is not None
    ]
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }

    arr = np.asarray(clean, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": (
            float(np.std(arr, ddof=1))
            if len(arr) > 1
            else 0.0
        ),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def flatten_prediction(summary: dict[str, Any]) -> dict[str, Any]:
    prediction = summary.get("prediction") or {}
    row = {
        "seed": summary["seed"],
        "stable": prediction.get("stable"),
        "model_identity_hash": prediction.get("model_identity_hash"),
    }
    for metric in PREDICTION_METRICS:
        row[metric] = prediction.get(metric)
    row["runtime_total_seconds"] = (
        summary.get("runtime") or {}
    ).get("total_seconds")
    return row


def flatten_controller(
    seed: int,
    controller: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    test_metrics = result.get("test_metrics") or {}
    locked = result.get("locked_parameters") or {}
    row = {
        "seed": seed,
        "controller": controller,
        "status": result.get("status"),
        "success": result.get("success"),
        "stable": result.get("stable"),
        "divergence_detected": result.get("divergence_detected"),
        "divergence_reason": result.get("divergence_reason"),
        "locked_K": locked.get("control_k"),
        "locked_finite_s": locked.get("finite_s"),
        "locked_pyragas_delay": locked.get("pyragas_delay"),
        "locked_pyragas_sign": locked.get("pyragas_sign"),
        "final_test_metric_name": result.get(
            "final_test_metric_name"
        ),
        "final_test_metric_value": result.get(
            "final_test_metric_value"
        ),
        "selection_runtime_seconds": result.get(
            "selection_runtime_seconds"
        ),
        "final_test_runtime_seconds": result.get(
            "final_test_runtime_seconds"
        ),
        "total_runtime_seconds": result.get(
            "total_runtime_seconds"
        ),
        "error_type": result.get("error_type"),
        "error": result.get("error"),
    }
    for metric in CONTROLLER_METRICS:
        row[metric] = test_metrics.get(metric)
    row["pyragas_quality_pass"] = test_metrics.get(
        "pyragas_quality_pass"
    )
    return row


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()

    paths = sorted(
        output_root.glob("seed_*/seed_summary.json"),
        key=lambda path: int(path.parent.name.split("_", 1)[1]),
    )
    if not paths:
        raise FileNotFoundError(
            f"No seed_*/seed_summary.json files found in {output_root}"
        )

    summaries = [read_json(path) for path in paths]
    prediction_rows = [flatten_prediction(summary) for summary in summaries]

    controller_rows: list[dict[str, Any]] = []
    for summary in summaries:
        for controller, result in (
            summary.get("controllers") or {}
        ).items():
            controller_rows.append(
                flatten_controller(
                    int(summary["seed"]),
                    controller,
                    result,
                )
            )

    write_csv(
        output_root / "all_seed_prediction_results.csv",
        prediction_rows,
    )
    write_csv(
        output_root / "all_seed_controller_results.csv",
        controller_rows,
    )

    prediction_aggregate_rows = []
    for metric in PREDICTION_METRICS:
        stats = aggregate_values(
            [row.get(metric) for row in prediction_rows]
        )
        prediction_aggregate_rows.append(
            {"metric": metric, **stats}
        )
    write_csv(
        output_root / "aggregate_prediction_metrics.csv",
        prediction_aggregate_rows,
    )

    controller_aggregate_rows = []
    controllers = sorted(
        {row["controller"] for row in controller_rows}
    )
    for controller in controllers:
        rows = [
            row
            for row in controller_rows
            if row["controller"] == controller
        ]
        successful = sum(bool(row.get("success")) for row in rows)
        attempted = len(rows)

        controller_aggregate_rows.append(
            {
                "controller": controller,
                "metric": "success_rate",
                "n": attempted,
                "mean": successful / attempted if attempted else None,
                "std": None,
                "median": None,
                "min": None,
                "max": None,
                "successful_seeds": successful,
                "attempted_seeds": attempted,
            }
        )

        controller_aggregate_rows.append(
            {
                "controller": controller,
                "metric": "final_test_metric_value",
                **aggregate_values(
                    [row.get("final_test_metric_value") for row in rows]
                ),
                "successful_seeds": successful,
                "attempted_seeds": attempted,
            }
        )

        for metric in CONTROLLER_METRICS:
            controller_aggregate_rows.append(
                {
                    "controller": controller,
                    "metric": metric,
                    **aggregate_values(
                        [row.get(metric) for row in rows]
                    ),
                    "successful_seeds": successful,
                    "attempted_seeds": attempted,
                }
            )

    write_csv(
        output_root / "aggregate_controller_metrics.csv",
        controller_aggregate_rows,
    )

    # Representative seed = prediction NRMSE closest to the median NRMSE.
    valid_prediction = [
        row
        for row in prediction_rows
        if finite_float(row.get("nrmse_recursive_x")) is not None
    ]
    if not valid_prediction:
        representative = {
            "seed": None,
            "reason": "No finite prediction NRMSE values were available.",
        }
    else:
        median_nrmse = float(
            np.median(
                [
                    float(row["nrmse_recursive_x"])
                    for row in valid_prediction
                ]
            )
        )
        chosen = min(
            valid_prediction,
            key=lambda row: (
                abs(
                    float(row["nrmse_recursive_x"])
                    - median_nrmse
                ),
                int(row["seed"]),
            ),
        )
        representative = {
            "seed": int(chosen["seed"]),
            "selection_metric": "nrmse_recursive_x",
            "median_metric_value": median_nrmse,
            "representative_metric_value": float(
                chosen["nrmse_recursive_x"]
            ),
            "rule": (
                "Seed whose held-out prediction NRMSE x is closest "
                "to the median across evaluated seeds."
            ),
        }

    write_json(
        output_root / "representative_seed.json",
        representative,
    )

    summary = {
        "schema_version": "chapter1_multiseed_aggregate_v1",
        "seed_count": len(summaries),
        "seeds": [int(summary["seed"]) for summary in summaries],
        "prediction_aggregates": prediction_aggregate_rows,
        "controller_aggregates": controller_aggregate_rows,
        "representative_seed": representative,
    }
    write_json(output_root / "multiseed_summary.json", summary)

    md_lines = [
        "# Chapter 1 multiseed evaluation",
        "",
        f"Seeds evaluated: {', '.join(map(str, summary['seeds']))}",
        "",
        "## Prediction",
        "",
        "| Metric | Mean | Std | Median | Min | Max | n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in prediction_aggregate_rows:
        md_lines.append(
            "| {metric} | {mean} | {std} | {median} | {min} | {max} | {n} |".format(
                **{
                    key: (
                        ""
                        if row.get(key) is None
                        else f"{row[key]:.6g}"
                        if isinstance(row[key], float)
                        else row[key]
                    )
                    for key in (
                        "metric",
                        "mean",
                        "std",
                        "median",
                        "min",
                        "max",
                        "n",
                    )
                }
            )
        )

    md_lines.extend(["", "## Controller success", ""])
    md_lines.extend(
        [
            "| Controller | Successful | Attempted | Success rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for controller in controllers:
        row = next(
            row
            for row in controller_aggregate_rows
            if row["controller"] == controller
            and row["metric"] == "success_rate"
        )
        md_lines.append(
            f"| {controller} | {row['successful_seeds']} | "
            f"{row['attempted_seeds']} | {100.0 * row['mean']:.1f}% |"
        )

    md_lines.extend(
        [
            "",
            "## Representative seed",
            "",
            (
                f"Representative seed: **{representative.get('seed')}** "
                f"using the median prediction-NRMSE rule."
            ),
            "",
        ]
    )
    (output_root / "multiseed_summary.md").write_text(
        "\n".join(md_lines),
        encoding="utf-8",
    )

    print("=" * 72)
    print("MULTISEED AGGREGATION COMPLETE")
    print("=" * 72)
    print(f"Seeds          : {summary['seeds']}")
    print(f"Output root    : {output_root}")
    print(
        "Representative: "
        f"seed {representative.get('seed')}"
    )
    for controller in controllers:
        row = next(
            row
            for row in controller_aggregate_rows
            if row["controller"] == controller
            and row["metric"] == "success_rate"
        )
        print(
            f"{controller:16s}: "
            f"{row['successful_seeds']}/{row['attempted_seeds']} successful"
        )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
