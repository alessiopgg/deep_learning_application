"""Multi-epoch orchestration, local artifacts, checkpointing and resume."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader

from Exercise3.training.checkpointing import (
    atomic_save_checkpoint,
    build_checkpoint,
    restore_training_state,
)
from Exercise3.training.configuration import BaselineTrainingConfig
from Exercise3.training.engine import (
    EpochLossMetrics,
    evaluate_validation_loss,
    train_one_epoch,
)
from Exercise3.training.tracking import ExperimentTracker


HISTORY_FIELDS = (
    "epoch",
    "learning_rate",
    "next_learning_rate",
    "train_total_loss",
    "train_loss_classifier",
    "train_loss_box_reg",
    "train_loss_objectness",
    "train_loss_rpn_box_reg",
    "train_batches",
    "train_images",
    "train_objects",
    "train_empty_images",
    "train_duration_seconds",
    "train_optimizer_steps",
    "train_amp_skipped_steps",
    "train_gradient_clip_norm",
    "train_peak_allocated_bytes",
    "train_peak_reserved_bytes",
    "validation_total_loss",
    "validation_loss_classifier",
    "validation_loss_box_reg",
    "validation_loss_objectness",
    "validation_loss_rpn_box_reg",
    "validation_batches",
    "validation_images",
    "validation_objects",
    "validation_empty_images",
    "validation_duration_seconds",
    "validation_peak_allocated_bytes",
    "validation_peak_reserved_bytes",
    "is_best",
)


def sanitize_run_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    normalized = normalized.strip("-._")
    if not normalized:
        raise ValueError("experiment.run_name becomes empty after sanitization.")
    return normalized


def create_run_directory(
    output_root: Path,
    configured_name: str | None,
    resume_from: Path | None,
) -> tuple[str, Path]:
    if resume_from is not None:
        run_dir = resume_from.resolve().parent
        return run_dir.name, run_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = (
        sanitize_run_name(configured_name)
        if configured_name is not None
        else "coco-frozen-backbone"
    )
    run_name = f"{timestamp}_{suffix}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_name, run_dir


def build_optimizer(model: nn.Module, config: BaselineTrainingConfig) -> Optimizer:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("The model has no trainable parameters.")
    optimizer = SGD(
        trainable,
        lr=config.optimizer.learning_rate,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in trainable}:
        raise ValueError("Optimizer groups do not match trainable parameters.")
    return optimizer


def build_scheduler(
    optimizer: Optimizer,
    config: BaselineTrainingConfig,
) -> LRScheduler:
    return StepLR(
        optimizer,
        step_size=config.scheduler.step_size,
        gamma=config.scheduler.gamma,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def save_history(history: list[dict[str, Any]], run_dir: Path) -> None:
    _atomic_write_text(
        run_dir / "history.json",
        json.dumps(history, indent=2, ensure_ascii=False),
    )
    csv_path = run_dir / "history.csv"
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(HISTORY_FIELDS))
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, csv_path)


def _epoch_record(
    *,
    epoch: int,
    learning_rate: float,
    next_learning_rate: float,
    train: EpochLossMetrics,
    validation: EpochLossMetrics,
    is_best: bool,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "next_learning_rate": next_learning_rate,
        **{f"train_{key}": value for key, value in train.to_dict().items() if key != "split"},
        **{
            f"validation_{key}": value
            for key, value in validation.to_dict().items()
            if key not in {"split", "optimizer_steps", "amp_skipped_steps", "gradient_clip_norm"}
        },
        "is_best": is_best,
    }


def print_epoch_summary(record: dict[str, Any]) -> None:
    print(
        f"Epoch {record['epoch']}: "
        f"train_loss={record['train_total_loss']:.6f}, "
        f"validation_loss={record['validation_total_loss']:.6f}, "
        f"lr={record['learning_rate']:.6g}, "
        f"best={record['is_best']}"
    )
    print(
        "  train components: "
        f"classifier={record['train_loss_classifier']:.6f}, "
        f"box={record['train_loss_box_reg']:.6f}, "
        f"objectness={record['train_loss_objectness']:.6f}, "
        f"rpn_box={record['train_loss_rpn_box_reg']:.6f}"
    )
    print(
        "  validation components: "
        f"classifier={record['validation_loss_classifier']:.6f}, "
        f"box={record['validation_loss_box_reg']:.6f}, "
        f"objectness={record['validation_loss_objectness']:.6f}, "
        f"rpn_box={record['validation_loss_rpn_box_reg']:.6f}"
    )


def fit_detector(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: Any,
    device: torch.device,
    config: BaselineTrainingConfig,
    run_name: str,
    run_dir: Path,
    tracker: ExperimentTracker,
    resume_checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    start_epoch = 1
    global_step = 0
    best_metric = math.inf
    best_epoch = 0

    if resume_checkpoint is not None:
        restore_training_state(
            checkpoint=resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=train_loader,
        )
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        global_step = int(resume_checkpoint["global_step"])
        best_metric = float(resume_checkpoint["best_metric"])
        best_epoch = int(resume_checkpoint["best_epoch"])
        history = list(resume_checkpoint["history"])

    if start_epoch > config.training.epochs:
        raise ValueError(
            f"Checkpoint already completed epoch {start_epoch - 1}, but "
            f"training.epochs={config.training.epochs}. Increase the target."
        )

    best_path = run_dir / "best_model.pt"
    last_path = run_dir / "last_checkpoint.pt"
    fit_start = time.perf_counter()

    for epoch in range(start_epoch, config.training.epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=config.training.amp,
            freeze_backbone=config.model.freeze_backbone,
            epoch=epoch,
            global_step=global_step,
            logging_interval=config.training.logging_interval,
            gradient_clip_norm=config.training.gradient_clip_norm,
            max_batches=config.training.max_train_batches,
            batch_logger=tracker.log_batch,
        )
        validation_metrics = evaluate_validation_loss(
            model=model,
            loader=validation_loader,
            device=device,
            amp_enabled=config.training.amp,
            freeze_backbone=config.model.freeze_backbone,
            validation_seed=config.experiment.seed + 10_000,
            max_batches=config.training.max_validation_batches,
        )

        monitored = validation_metrics.total_loss
        is_best = monitored < best_metric
        if is_best:
            best_metric = monitored
            best_epoch = epoch

        scheduler.step()
        next_lr = float(optimizer.param_groups[0]["lr"])
        record = _epoch_record(
            epoch=epoch,
            learning_rate=current_lr,
            next_learning_rate=next_lr,
            train=train_metrics,
            validation=validation_metrics,
            is_best=is_best,
        )
        history.append(record)
        save_history(history, run_dir)

        checkpoint = build_checkpoint(
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            best_epoch=best_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config.to_dict(),
            history=history,
            train_loader=train_loader,
            wandb_run_id=tracker.run_id,
        )
        if is_best:
            atomic_save_checkpoint(checkpoint, best_path)
        if config.checkpoint.save_last:
            atomic_save_checkpoint(checkpoint, last_path)

        tracker.log_epoch(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                **{
                    f"train_epoch_{key}": value
                    for key, value in train_metrics.to_dict().items()
                    if key != "split"
                },
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.to_dict().items()
                    if key != "split"
                },
                "best_validation_total_loss": best_metric,
                "best_epoch": best_epoch,
            }
        )
        print_epoch_summary(record)

    duration = time.perf_counter() - fit_start
    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "completed_epochs": config.training.epochs,
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_metric,
        "global_step": global_step,
        "duration_seconds": duration,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path) if config.checkpoint.save_last else None,
        "limited_train_batches": config.training.max_train_batches,
        "limited_validation_batches": config.training.max_validation_batches,
        "scientific_run": (
            config.training.max_train_batches is None
            and config.training.max_validation_batches is None
        ),
        "test_evaluated": False,
    }
    _atomic_write_text(
        run_dir / "run_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False),
    )
    tracker.update_summary(
        {
            "best_epoch": best_epoch,
            "best_validation_total_loss": best_metric,
            "training_duration_seconds": duration,
            "scientific_run": summary["scientific_run"],
        }
    )
    tracker.log_best_checkpoint(best_path, run_name)
    return summary


def build_runtime_metadata(
    *,
    device: torch.device,
    model_metadata: dict[str, Any],
    loader_settings: dict[str, Any],
    dataset_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0 if device.index is None else device.index)
            if device.type == "cuda"
            else None
        ),
        "model": model_metadata,
        "loader_settings": loader_settings,
        "dataset": dataset_metadata,
    }
