"""Runtime, optimization, training, evaluation and checkpointing for Exercise 2."""

import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score
from torch import nn

from models import set_fine_tuning_mode


def set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = bool(torch.cuda.is_available() and not deterministic)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {index} is unavailable; found {torch.cuda.device_count()} device(s)."
            )
    return device


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    index = 0 if device.index is None else device.index
    return f"CUDA:{index} - {torch.cuda.get_device_name(index)}"


def create_loss(config: DictConfig) -> nn.Module:
    if config.training.loss_function != "cross_entropy":
        raise ValueError(f"Unsupported loss: {config.training.loss_function}")
    return nn.CrossEntropyLoss()


def create_training_components(model, config: DictConfig):
    """Create CrossEntropyLoss and AdamW with differentiated learning rates."""
    criterion = create_loss(config)
    strategy = config.model.fine_tuning_strategy

    if strategy == "classifier":
        groups = [{"params": model.fc.parameters(), "lr": config.training.classifier_learning_rate}]
        group_names = ["classifier"]
    elif strategy == "last_block":
        groups = [
            {"params": model.layer4.parameters(), "lr": config.training.backbone_learning_rate},
            {"params": model.fc.parameters(), "lr": config.training.classifier_learning_rate},
        ]
        group_names = ["last_block", "classifier"]
    else:
        backbone = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
        groups = [
            {"params": backbone, "lr": config.training.backbone_learning_rate},
            {"params": model.fc.parameters(), "lr": config.training.classifier_learning_rate},
        ]
        group_names = ["backbone", "classifier"]

    if config.training.optimizer != "adamw":
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")
    optimizer = torch.optim.AdamW(groups, weight_decay=config.training.weight_decay)

    info = {
        "loss_function": config.training.loss_function,
        "optimizer": config.training.optimizer,
        "backbone_learning_rate": float(config.training.backbone_learning_rate),
        "classifier_learning_rate": float(config.training.classifier_learning_rate),
        "weight_decay": float(config.training.weight_decay),
        "parameter_groups": group_names,
    }
    return criterion, optimizer, info


def print_optimization_summary(info: dict) -> None:
    print("\n=== Exercise 2 optimization configuration ===")
    print(f"Loss function: {info['loss_function']}")
    print(f"Optimizer: {info['optimizer']}")
    print(f"Backbone learning rate: {info['backbone_learning_rate']}")
    print(f"Classifier learning rate: {info['classifier_learning_rate']}")
    print(f"Weight decay: {info['weight_decay']}")
    print(f"Parameter groups: {info['parameter_groups']}")


def _finalize_metrics(total_loss, total_correct, labels, predictions, samples, batches, seconds):
    if samples == 0:
        raise RuntimeError("No samples were processed.")
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    metrics = {
        "loss": float(total_loss / samples),
        "accuracy": float(total_correct / samples),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "processed_samples": int(samples),
        "processed_batches": int(batches),
        "seconds": float(seconds),
    }
    return metrics, y_true, y_pred


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    strategy,
    log_interval: int,
    max_batches: int | None = None,
):
    set_fine_tuning_mode(model, strategy)
    limit = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    if limit <= 0:
        raise ValueError("The training loop must process at least one batch.")

    total_loss = total_correct = samples = batches = 0
    interval_loss = interval_correct = interval_samples = 0
    all_labels, all_predictions = [], []
    start = perf_counter()

    for batch_number, (images, labels) in enumerate(dataloader, start=1):
        if batch_number > limit:
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
        samples += batch_size
        batches += 1
        interval_loss += weighted_loss
        interval_correct += correct
        interval_samples += batch_size
        all_labels.append(labels.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())

        if batch_number % log_interval == 0 or batch_number == limit:
            print(
                f"Batch {batch_number}/{limit} | "
                f"Recent loss: {interval_loss / interval_samples:.4f} | "
                f"Recent accuracy: {interval_correct / interval_samples:.4f}"
            )
            interval_loss = interval_correct = interval_samples = 0

    return _finalize_metrics(
        total_loss, total_correct, all_labels, all_predictions,
        samples, batches, perf_counter() - start,
    )[0]


def evaluate(
    model,
    dataloader,
    criterion,
    device,
    collect_predictions: bool = False,
    max_batches: int | None = None,
):
    model.eval()
    limit = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    if limit <= 0:
        raise ValueError("The evaluation loop must process at least one batch.")

    total_loss = total_correct = samples = batches = 0
    all_labels, all_predictions = [], []
    start = perf_counter()

    with torch.inference_mode():
        for batch_number, (images, labels) in enumerate(dataloader, start=1):
            if batch_number > limit:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            samples += batch_size
            batches += 1
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    metrics, y_true, y_pred = _finalize_metrics(
        total_loss, total_correct, all_labels, all_predictions,
        samples, batches, perf_counter() - start,
    )
    return metrics, (y_true if collect_predictions else None), (y_pred if collect_predictions else None)


def print_epoch_metrics(title: str, metrics: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    print(f"Processed samples: {metrics['processed_samples']}")
    print(f"Processed batches: {metrics['processed_batches']}")
    print(f"Elapsed time: {metrics['seconds']:.2f} seconds")


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def save_checkpoint(path: Path, model, optimizer, config, epoch, monitor, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": int(epoch),
            "monitor": str(monitor),
            "monitored_value": float(value),
            "model_state_dict": _to_cpu(model.state_dict()),
            "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
            "config": OmegaConf.to_container(config, resolve=True, enum_to_str=True),
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path: Path, model, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def fit(model, train_loader, validation_loader, criterion, optimizer, device, config, checkpoint_path: Path):
    """Train all epochs, select by validation metric, then reload the best model."""
    monitor = str(config.checkpoint.monitor)
    mode = str(config.checkpoint.mode)
    best = float("inf") if mode == "min" else float("-inf")
    best_epoch = 0
    history = []
    start = perf_counter()

    metric_key = {
        "validation_loss": "loss",
        "validation_accuracy": "accuracy",
        "validation_macro_f1": "macro_f1",
    }[monitor]

    for epoch in range(1, int(config.training.epochs) + 1):
        print(f"\n=== Epoch {epoch}/{config.training.epochs} ===")
        training_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            config.model.fine_tuning_strategy, config.logging.batch_interval,
        )
        validation_metrics, _, _ = evaluate(
            model, validation_loader, criterion, device
        )
        value = float(validation_metrics[metric_key])

        history.append({
            "epoch": epoch,
            **{f"training_{k}": v for k, v in training_metrics.items()},
            **{f"validation_{k}": v for k, v in validation_metrics.items()},
            "monitored_value": value,
        })

        print(
            "Training | "
            f"loss={training_metrics['loss']:.4f} | "
            f"accuracy={training_metrics['accuracy']:.4f} | "
            f"macro-F1={training_metrics['macro_f1']:.4f}"
        )
        print(
            "Validation | "
            f"loss={validation_metrics['loss']:.4f} | "
            f"accuracy={validation_metrics['accuracy']:.4f} | "
            f"macro-F1={validation_metrics['macro_f1']:.4f}"
        )

        improved = value < best if mode == "min" else value > best
        if improved:
            best, best_epoch = value, epoch
            save_checkpoint(
                checkpoint_path, model, optimizer, config,
                epoch, monitor, value,
            )
            print(f"New best checkpoint saved | epoch={epoch} | {monitor}={value:.6f}")

    if best_epoch == 0:
        raise RuntimeError("Training completed without selecting a best checkpoint.")

    checkpoint = load_checkpoint(checkpoint_path, model, device)
    if int(checkpoint["epoch"]) != best_epoch:
        raise RuntimeError("Reloaded checkpoint does not match the selected best epoch.")

    result = {
        "history": history,
        "best_epoch": best_epoch,
        "best_monitored_value": float(best),
        "monitor": monitor,
        "mode": mode,
        "checkpoint_path": str(checkpoint_path),
        "total_seconds": float(perf_counter() - start),
    }
    print("\n=== Exercise 2 fitting completed ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Monitored metric: {monitor}")
    print(f"Selection mode: {mode}")
    print(f"Best value: {best:.6f}")
    print(f"Total fitting time: {result['total_seconds']:.2f} seconds")
    print(f"Best checkpoint: {checkpoint_path}")
    return result
