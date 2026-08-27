"""Evaluate an Exercise 2 checkpoint on the official GTSRB test set."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import classification_report

from data import create_dataloaders
from models import create_input_transform, create_model
from training import (
    create_loss,
    describe_device,
    evaluate,
    load_checkpoint,
    print_epoch_metrics,
    resolve_device,
    set_reproducibility,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate the best Exercise 2 checkpoint on GTSRB."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ...")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    stored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in stored:
        raise KeyError("The checkpoint does not contain the experiment configuration.")
    config = OmegaConf.create(stored["config"])

    set_reproducibility(config.experiment.seed, config.experiment.deterministic)
    device = resolve_device(args.device or config.experiment.device)

    print("\n=== Exercise 2 test evaluation ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Selected device: {describe_device(device)}")
    print(f"Stored best epoch: {stored['epoch']}")
    print(f"Stored {stored['monitor']}: {stored['monitored_value']:.6f}")

    transform = create_input_transform(config.model.name)
    loaders, _ = create_dataloaders(config, transform, device)
    model, _ = create_model(config, device)
    load_checkpoint(checkpoint_path, model, device)

    metrics, true_labels, predictions = evaluate(
        model,
        loaders["test"],
        create_loss(config),
        device,
        collect_predictions=True,
    )
    print_epoch_metrics("Final test results", metrics)

    report = classification_report(
        true_labels,
        predictions,
        labels=np.arange(config.model.num_classes),
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    print("\n=== Classification report ===")
    print(
        classification_report(
            true_labels,
            predictions,
            labels=np.arange(config.model.num_classes),
            digits=4,
            zero_division=0,
        )
    )

    run_dir = checkpoint_path.parent
    test_metrics = {
        "best_epoch": int(stored["epoch"]),
        "checkpoint_monitor": str(stored["monitor"]),
        "best_validation_value": float(stored["monitored_value"]),
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_samples": metrics["processed_samples"],
        "test_seconds": metrics["seconds"],
    }

    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=4)
    with (run_dir / "classification_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)
    np.savez_compressed(
        run_dir / "predictions.npz",
        true_labels=true_labels,
        predictions=predictions,
    )

    print(f"\nTest outputs saved in: {run_dir}")
    print("- test_metrics.json")
    print("- classification_report.json")
    print("- predictions.npz")


if __name__ == "__main__":
    main()
