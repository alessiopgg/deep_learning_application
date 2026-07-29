"""Evaluate an Exercise 2 checkpoint on the official GTSRB test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import classification_report

from checkpointing import load_checkpoint
from data import create_dataloaders
from engine import evaluate, print_epoch_metrics
from models import create_input_transform, create_model
from optimization import create_loss
from runtime import describe_device, resolve_device, set_reproducibility


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the best Exercise 2 checkpoint on GTSRB."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the best_model.pt checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override: auto, cpu, cuda, cuda:0, ...",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    checkpoint_path = args.checkpoint.expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    stored_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "config" not in stored_checkpoint:
        raise KeyError(
            "The checkpoint does not contain the experiment configuration."
        )

    config = OmegaConf.create(stored_checkpoint["config"])

    set_reproducibility(
        seed=config.experiment.seed,
        deterministic=config.experiment.deterministic,
    )

    requested_device = (
        args.device
        if args.device is not None
        else config.experiment.device
    )
    device = resolve_device(requested_device)

    print("\n=== Exercise 2 test evaluation ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Selected device: {describe_device(device)}")
    print(f"Stored best epoch: {stored_checkpoint['epoch']}")
    print(
        f"Stored {stored_checkpoint['monitor']}: "
        f"{stored_checkpoint['monitored_value']:.6f}"
    )

    transform = create_input_transform(config.model.name)
    data_loaders = create_dataloaders(
        config=config,
        transform=transform,
        device=device,
    )

    model_bundle = create_model(
        config=config,
        device=device,
    )
    load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model_bundle.model,
        device=device,
    )

    criterion = create_loss(config)

    test_result = evaluate(
        model=model_bundle.model,
        dataloader=data_loaders.test,
        criterion=criterion,
        device=device,
        collect_predictions=True,
    )

    if (
        test_result.true_labels is None
        or test_result.predictions is None
    ):
        raise RuntimeError("Test predictions were not collected.")

    print_epoch_metrics(
        title="Final test results",
        metrics=test_result.metrics,
    )

    printable_report = classification_report(
        test_result.true_labels,
        test_result.predictions,
        labels=np.arange(config.model.num_classes),
        digits=4,
        zero_division=0,
    )
    report_dictionary = classification_report(
        test_result.true_labels,
        test_result.predictions,
        labels=np.arange(config.model.num_classes),
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    print("\n=== Classification report ===")
    print(printable_report)

    run_dir = checkpoint_path.parent
    metrics = {
        "best_epoch": int(stored_checkpoint["epoch"]),
        "checkpoint_monitor": str(stored_checkpoint["monitor"]),
        "best_validation_value": float(
            stored_checkpoint["monitored_value"]
        ),
        "test_loss": float(test_result.metrics.loss),
        "test_accuracy": float(test_result.metrics.accuracy),
        "test_macro_f1": float(test_result.metrics.macro_f1),
        "test_samples": int(test_result.metrics.processed_samples),
        "test_seconds": float(test_result.metrics.seconds),
    }

    with (run_dir / "test_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    with (run_dir / "classification_report.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report_dictionary, file, indent=4)

    np.savez_compressed(
        run_dir / "predictions.npz",
        true_labels=test_result.true_labels,
        predictions=test_result.predictions,
    )

    print("\nTest outputs saved in:")
    print(run_dir)
    print("- test_metrics.json")
    print("- classification_report.json")
    print("- predictions.npz")


if __name__ == "__main__":
    main()
