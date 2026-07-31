"""CLI entry point for Step 12 detector evaluation."""

from __future__ import annotations

from Exercise3.evaluation.configuration import (
    apply_explicit_arguments,
    load_evaluation_config,
    parse_evaluation_arguments,
)
from Exercise3.evaluation.evaluator import evaluate_checkpoint


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def main() -> None:
    arguments, overrides = parse_evaluation_arguments()
    config = load_evaluation_config(arguments.config, overrides)
    config = apply_explicit_arguments(config, arguments)

    print("\n=== Exercise 3.3 - Step 12: detector evaluation ===")
    print(f"Checkpoint: {config.checkpoint.path}")
    print(f"Split: {config.evaluation.split}")
    print(f"Device: {config.runtime.device}")
    print(f"COCO backend: {config.evaluation.backend}")
    print(f"Model score threshold: {config.evaluation.model_score_threshold}")
    print(
        "Fixed diagnostics: score >= "
        f"{config.evaluation.fixed_score_threshold}, "
        f"IoU >= {config.evaluation.fixed_iou_threshold}"
    )
    print(f"Batch limit: {config.evaluation.max_batches}")
    print(f"Test explicitly allowed: {config.evaluation.allow_test}")

    report = evaluate_checkpoint(config)
    standard = report["metrics"]["coco_style"]
    fixed = report["metrics"]["fixed_threshold"]

    print("\nCOCO-style metrics:")
    print(f"  mAP@[0.50:0.95]: {_format(standard['map'])}")
    print(f"  AP50:             {_format(standard['map_50'])}")
    print(f"  AP75:             {_format(standard['map_75'])}")
    print(f"  AP small:         {_format(standard['map_small'])}")
    print(f"  AP medium:        {_format(standard['map_medium'])}")
    print(f"  AP large:         {_format(standard['map_large'])}")
    print(f"  AR@100:           {_format(standard['mar_100'])}")

    print("\nFixed-threshold diagnostics:")
    print(f"  precision: {fixed['precision']:.6f}")
    print(f"  recall:    {fixed['recall']:.6f}")
    print(f"  F1:        {fixed['f1']:.6f}")
    print(
        "  TP / FP / FN: "
        f"{fixed['true_positives']} / "
        f"{fixed['false_positives']} / "
        f"{fixed['false_negatives']}"
    )
    print(
        "  empty-image FP rate: "
        f"{_format(fixed['empty_target_false_positive_rate'])}"
    )

    print("\nEvaluation completed.")
    print(
        "Images evaluated: "
        f"{report['dataset']['images_evaluated']} / "
        f"{report['dataset']['split_images']}"
    )
    print(f"Scientific full evaluation: {report['evaluation']['scientific_full_evaluation']}")
    print(f"Outputs: {report['artifacts']['output_directory']}")


if __name__ == "__main__":
    main()
