"""Multi-epoch fitting and best-checkpoint selection for Exercise 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet

from checkpointing import load_checkpoint, save_checkpoint
from engine import EpochMetrics, evaluate, train_one_epoch


@dataclass(frozen=True)
class EpochRecord:
    """Training and validation metrics associated with one epoch."""

    epoch: int
    training: EpochMetrics
    validation: EpochMetrics
    monitored_value: float

    def to_dict(self) -> dict[str, Any]:
        """Return one flat, JSON-compatible epoch record."""
        return {
            "epoch": int(self.epoch),
            "training_loss": float(self.training.loss),
            "training_accuracy": float(self.training.accuracy),
            "training_macro_f1": float(self.training.macro_f1),
            "training_seconds": float(self.training.seconds),
            "validation_loss": float(self.validation.loss),
            "validation_accuracy": float(self.validation.accuracy),
            "validation_macro_f1": float(self.validation.macro_f1),
            "validation_seconds": float(self.validation.seconds),
            "monitored_value": float(self.monitored_value),
        }


@dataclass(frozen=True)
class FitResult:
    """Complete outcome of a multi-epoch fitting procedure."""

    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_monitored_value: float
    monitor: str
    mode: str
    checkpoint_path: str
    total_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return the aggregate fitting summary."""
        result = asdict(self)
        result["history"] = [
            record.to_dict() for record in self.history
        ]
        return result


def _validation_metric(metrics: EpochMetrics, monitor: str) -> float:
    """Read the configured checkpoint metric from validation results."""
    metric_mapping = {
        "validation_loss": metrics.loss,
        "validation_accuracy": metrics.accuracy,
        "validation_macro_f1": metrics.macro_f1,
    }

    try:
        return float(metric_mapping[monitor])
    except KeyError as error:
        raise ValueError(
            f"Unsupported checkpoint metric: {monitor}"
        ) from error


def _initial_best_value(mode: str) -> float:
    """Return the neutral initial value for min or max selection."""
    if mode == "min":
        return float("inf")
    if mode == "max":
        return float("-inf")
    raise ValueError("checkpoint mode must be 'min' or 'max'.")


def _is_improvement(
    candidate: float,
    current_best: float,
    mode: str,
) -> bool:
    """Return whether a validation metric improves the current best."""
    if mode == "min":
        return candidate < current_best
    if mode == "max":
        return candidate > current_best
    raise ValueError("checkpoint mode must be 'min' or 'max'.")


def _print_epoch_summary(record: EpochRecord, total_epochs: int) -> None:
    """Print training, validation and checkpoint metrics for one epoch."""
    print(f"\n=== Epoch {record.epoch}/{total_epochs} summary ===")
    print(
        "Training | "
        f"loss={record.training.loss:.4f} | "
        f"accuracy={record.training.accuracy:.4f} | "
        f"macro-F1={record.training.macro_f1:.4f}"
    )
    print(
        "Validation | "
        f"loss={record.validation.loss:.4f} | "
        f"accuracy={record.validation.accuracy:.4f} | "
        f"macro-F1={record.validation.macro_f1:.4f}"
    )
    print(f"Monitored value: {record.monitored_value:.6f}")


def fit(
    model: ResNet,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: DictConfig,
    checkpoint_path: Path,
) -> FitResult:
    """Train for all configured epochs and reload the best checkpoint."""
    monitor = str(config.checkpoint.monitor)
    mode = str(config.checkpoint.mode)
    best_value = _initial_best_value(mode)
    best_epoch = 0
    history: list[EpochRecord] = []
    training_start = perf_counter()

    for epoch in range(1, int(config.training.epochs) + 1):
        print(f"\n=== Epoch {epoch}/{config.training.epochs} ===")

        training_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            fine_tuning_strategy=(
                config.model.fine_tuning_strategy
            ),
            log_interval=config.logging.batch_interval,
        )
        validation_result = evaluate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
            collect_predictions=False,
        )

        monitored_value = _validation_metric(
            metrics=validation_result.metrics,
            monitor=monitor,
        )
        record = EpochRecord(
            epoch=epoch,
            training=training_metrics,
            validation=validation_result.metrics,
            monitored_value=monitored_value,
        )
        history.append(record)
        _print_epoch_summary(record, int(config.training.epochs))

        if _is_improvement(monitored_value, best_value, mode):
            best_value = monitored_value
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                monitor=monitor,
                monitored_value=monitored_value,
            )
            print(
                "New best checkpoint saved | "
                f"epoch={epoch} | {monitor}={monitored_value:.6f}"
            )

    total_seconds = perf_counter() - training_start

    if best_epoch == 0:
        raise RuntimeError(
            "Training completed without selecting a best checkpoint."
        )

    checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
    )

    loaded_epoch = int(checkpoint["epoch"])
    if loaded_epoch != best_epoch:
        raise RuntimeError(
            "The reloaded checkpoint does not match the selected epoch."
        )

    result = FitResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_monitored_value=float(best_value),
        monitor=monitor,
        mode=mode,
        checkpoint_path=str(checkpoint_path),
        total_seconds=float(total_seconds),
    )
    print_fit_summary(result)
    return result


def print_fit_summary(result: FitResult) -> None:
    """Print the final best-checkpoint and training-time summary."""
    print("\n=== Exercise 2 fitting completed ===")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Monitored metric: {result.monitor}")
    print(f"Selection mode: {result.mode}")
    print(f"Best value: {result.best_monitored_value:.6f}")
    print(f"Total fitting time: {result.total_seconds:.2f} seconds")
    print(f"Best checkpoint: {result.checkpoint_path}")
