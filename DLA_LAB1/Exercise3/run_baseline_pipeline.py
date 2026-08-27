"""Run baseline training followed automatically by validation evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from Exercise3.training.configuration import (
    load_training_config,
    resolve_output_root,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the COCO frozen-backbone baseline and evaluate its best "
            "checkpoint immediately afterwards."
        )
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("Exercise3/configs/baseline.yaml"),
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=Path("Exercise3/configs/evaluation.yaml"),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--run-name", type=str, default="baseline-coco-frozen")
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--training-override",
        action="append",
        default=[],
        help="Additional OmegaConf training override; repeat as needed.",
    )
    parser.add_argument(
        "--evaluation-override",
        action="append",
        default=[],
        help="Additional OmegaConf evaluation override; repeat as needed.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also evaluate the test split after validation. Not recommended during model selection.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run(command: list[str], *, dry_run: bool) -> None:
    print("\n$ " + " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    arguments = parse_arguments()
    training_overrides = [
        f"experiment.device={arguments.device}",
        f"experiment.run_name={arguments.run_name}",
        f"tracking.use_wandb={'true' if arguments.wandb else 'false'}",
        *arguments.training_override,
    ]
    training_config = load_training_config(
        arguments.training_config,
        training_overrides,
    )
    output_root = resolve_output_root(training_config)
    output_root.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in output_root.iterdir() if path.is_dir()}

    training_command = [
        sys.executable,
        "-m",
        "Exercise3.train_baseline",
        "--config",
        str(arguments.training_config),
        *training_overrides,
    ]
    _run(training_command, dry_run=arguments.dry_run)
    if arguments.dry_run:
        print("\nDry run complete; evaluation checkpoint is resolved after training.")
        return

    after = {path.resolve() for path in output_root.iterdir() if path.is_dir()}
    new_directories = sorted(
        after - before,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not new_directories:
        raise RuntimeError("Training completed but no new run directory was found.")
    run_dir = new_directories[0]
    checkpoint = run_dir / "best_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")

    common_evaluation = [
        sys.executable,
        "-m",
        "Exercise3.evaluate_detector",
        "--config",
        str(arguments.evaluation_config),
        "--checkpoint",
        str(checkpoint),
        "--device",
        arguments.device,
        *arguments.evaluation_override,
    ]
    _run(
        [*common_evaluation, "--split", "validation"],
        dry_run=False,
    )
    if arguments.evaluate_test:
        _run(
            [*common_evaluation, "--split", "test", "--allow-test"],
            dry_run=False,
        )

    print("\nPipeline completed.")
    print(f"Run directory: {run_dir}")
    print(f"Best checkpoint: {checkpoint}")
    print(f"Validation evaluation: {run_dir / 'evaluation' / 'validation'}")
    if arguments.evaluate_test:
        print(f"Test evaluation: {run_dir / 'evaluation' / 'test'}")


if __name__ == "__main__":
    main()
