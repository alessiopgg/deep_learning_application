"""Reusable training and evaluation loops for Exercise 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet

from models import set_fine_tuning_mode


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregate metrics computed over one training or evaluation pass."""

    loss: float
    accuracy: float
    macro_f1: float
    processed_samples: int
    processed_batches: int
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the metrics."""
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation metrics and, optionally, sample-level predictions."""

    metrics: EpochMetrics
    true_labels: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None

    def to_dict(self) -> dict[str, Any]:
        """Return only the aggregate, JSON-compatible evaluation metrics."""
        return self.metrics.to_dict()


def _validate_loop_arguments(
    dataloader: DataLoader,
    max_batches: Optional[int],
) -> int:
    """Validate loop limits and return the number of batches to process."""
    available_batches = len(dataloader)
    if available_batches == 0:
        raise ValueError("The DataLoader contains no batches.")

    if max_batches is None:
        return available_batches

    if max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")

    return min(max_batches, available_batches)


def _build_metrics(
    total_loss: float,
    total_correct: int,
    true_labels: list[np.ndarray],
    predictions: list[np.ndarray],
    processed_samples: int,
    processed_batches: int,
    seconds: float,
) -> tuple[EpochMetrics, np.ndarray, np.ndarray]:
    """Create aggregate metrics and concatenate collected predictions."""
    if processed_samples == 0:
        raise RuntimeError("No samples were processed by the loop.")

    labels_array = np.concatenate(true_labels, axis=0)
    predictions_array = np.concatenate(predictions, axis=0)

    metrics = EpochMetrics(
        loss=float(total_loss / processed_samples),
        accuracy=float(total_correct / processed_samples),
        macro_f1=float(
            f1_score(
                labels_array,
                predictions_array,
                average="macro",
                zero_division=0,
            )
        ),
        processed_samples=int(processed_samples),
        processed_batches=int(processed_batches),
        seconds=float(seconds),
    )
    return metrics, labels_array, predictions_array


def train_one_epoch(
    model: ResNet,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    fine_tuning_strategy: str,
    log_interval: int,
    max_batches: Optional[int] = None,
) -> EpochMetrics:
    """Train the model for one epoch, or for a limited smoke-test pass."""
    if log_interval <= 0:
        raise ValueError("log_interval must be positive.")

    batches_to_process = _validate_loop_arguments(
        dataloader=dataloader,
        max_batches=max_batches,
    )
    set_fine_tuning_mode(
        model=model,
        strategy=fine_tuning_strategy,
    )

    total_loss = 0.0
    total_correct = 0
    processed_samples = 0
    processed_batches = 0
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    interval_loss = 0.0
    interval_correct = 0
    interval_samples = 0

    start_time = perf_counter()

    for batch_number, (images, labels) in enumerate(dataloader, start=1):
        if batch_number > batches_to_process:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        correct = (predictions == labels).sum().item()

        weighted_loss = loss.item() * batch_size
        total_loss += weighted_loss
        total_correct += correct
        processed_samples += batch_size
        processed_batches += 1

        interval_loss += weighted_loss
        interval_correct += correct
        interval_samples += batch_size

        all_labels.append(labels.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())

        should_log = (
            batch_number % log_interval == 0
            or batch_number == batches_to_process
        )
        if should_log:
            recent_loss = interval_loss / interval_samples
            recent_accuracy = interval_correct / interval_samples
            print(
                f"Batch {batch_number}/{batches_to_process} | "
                f"Recent loss: {recent_loss:.4f} | "
                f"Recent accuracy: {recent_accuracy:.4f}"
            )
            interval_loss = 0.0
            interval_correct = 0
            interval_samples = 0

    elapsed_seconds = perf_counter() - start_time
    metrics, _, _ = _build_metrics(
        total_loss=total_loss,
        total_correct=total_correct,
        true_labels=all_labels,
        predictions=all_predictions,
        processed_samples=processed_samples,
        processed_batches=processed_batches,
        seconds=elapsed_seconds,
    )
    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect_predictions: bool = False,
    max_batches: Optional[int] = None,
) -> EvaluationResult:
    """Evaluate a model without gradients or parameter updates."""
    batches_to_process = _validate_loop_arguments(
        dataloader=dataloader,
        max_batches=max_batches,
    )
    model.eval()

    total_loss = 0.0
    total_correct = 0
    processed_samples = 0
    processed_batches = 0
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    start_time = perf_counter()

    with torch.inference_mode():
        for batch_number, (images, labels) in enumerate(
            dataloader,
            start=1,
        ):
            if batch_number > batches_to_process:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            processed_samples += batch_size
            processed_batches += 1

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    elapsed_seconds = perf_counter() - start_time
    metrics, labels_array, predictions_array = _build_metrics(
        total_loss=total_loss,
        total_correct=total_correct,
        true_labels=all_labels,
        predictions=all_predictions,
        processed_samples=processed_samples,
        processed_batches=processed_batches,
        seconds=elapsed_seconds,
    )

    if not collect_predictions:
        labels_array = None
        predictions_array = None

    return EvaluationResult(
        metrics=metrics,
        true_labels=labels_array,
        predictions=predictions_array,
    )


def print_epoch_metrics(title: str, metrics: EpochMetrics) -> None:
    """Print a compact summary of one training or evaluation pass."""
    print(f"\n=== {title} ===")
    print(f"Loss: {metrics.loss:.4f}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Macro F1-score: {metrics.macro_f1:.4f}")
    print(f"Processed samples: {metrics.processed_samples}")
    print(f"Processed batches: {metrics.processed_batches}")
    print(f"Elapsed time: {metrics.seconds:.2f} seconds")
