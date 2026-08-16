"""Command-line entry point for Chapter 2 Step 7 Bayesian optimisation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .esn_config import CHAPTER2_ROOT
    from .esn_optimisation import (
        BAYESIAN_CALLS_PER_MODEL,
        MODEL_TYPES,
        ORDINARY_BASELINE,
        PARAMETER_AWARE,
        RealCandidateEvaluator,
        SearchSettings,
        checkpoint_metadata,
        history_path,
        prepare_both_model_data,
        run_bayesian_search,
        run_robust_confirmation,
        selection_path,
        training_dataset_hashes,
        write_selection_artifact,
    )
except ImportError:  # Support direct execution from the chapter2 directory.
    from esn_config import CHAPTER2_ROOT
    from esn_optimisation import (
        BAYESIAN_CALLS_PER_MODEL,
        MODEL_TYPES,
        ORDINARY_BASELINE,
        PARAMETER_AWARE,
        RealCandidateEvaluator,
        SearchSettings,
        checkpoint_metadata,
        history_path,
        prepare_both_model_data,
        run_bayesian_search,
        run_robust_confirmation,
        selection_path,
        training_dataset_hashes,
        write_selection_artifact,
    )


DEFAULT_OUTPUT_DIR = CHAPTER2_ROOT / "optimisation_results"
CLI_MODEL_TYPES = {
    "parameter-aware": PARAMETER_AWARE,
    "ordinary-baseline": ORDINARY_BASELINE,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run validation-only Step 7 Bayesian optimisation. This command "
            "does not open benchmarks or train final models."
        )
    )
    parser.add_argument(
        "--model",
        choices=("both", *CLI_MODEL_TYPES),
        default="both",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=BAYESIAN_CALLS_PER_MODEL,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume compatible strict-JSON checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Build selection from two already complete histories.",
    )
    args = parser.parse_args()

    parameter_history_path = history_path(args.output_dir, PARAMETER_AWARE)
    ordinary_history_path = history_path(args.output_dir, ORDINARY_BASELINE)
    final_selection_path = selection_path(args.output_dir)

    if args.selection_only:
        selection = write_selection_artifact(
            parameter_history_path,
            ordinary_history_path,
            final_selection_path,
        )
        print(json.dumps(selection["models"], indent=2, sort_keys=True))
        return

    if args.n_calls != BAYESIAN_CALLS_PER_MODEL:
        parser.error(
            f"Step 7 requires exactly {BAYESIAN_CALLS_PER_MODEL} calls per model"
        )

    selected_models = (
        MODEL_TYPES
        if args.model == "both"
        else (CLI_MODEL_TYPES[args.model],)
    )
    model_data = prepare_both_model_data()
    dataset_hashes = training_dataset_hashes()

    for model_type in selected_models:
        settings = SearchSettings.frozen(model_type)
        data = model_data[model_type]
        metadata = checkpoint_metadata(
            model_type,
            settings,
            data,
            dataset_hashes,
        )
        evaluator = RealCandidateEvaluator(data)
        path = history_path(args.output_dir, model_type)
        run_bayesian_search(
            checkpoint_path=path,
            model_type=model_type,
            evaluator=evaluator,
            metadata=metadata,
            settings=settings,
            resume=args.resume,
        )
        run_robust_confirmation(
            checkpoint_path=path,
            evaluator=evaluator,
        )

    if args.model == "both":
        selection = write_selection_artifact(
            parameter_history_path,
            ordinary_history_path,
            final_selection_path,
        )
        print(f"Wrote {final_selection_path}")
        for model_type in MODEL_TYPES:
            model = selection["models"][model_type]
            print(
                f"{model_type}: "
                f"robust_mean="
                f"{model['best_robust_aggregate']['mean_objective_nrmse']:.9g}"
            )


if __name__ == "__main__":
    main()
