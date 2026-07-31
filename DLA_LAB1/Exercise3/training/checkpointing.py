"""Atomic checkpoints and exact-resume state for Exercise 3 training."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


def _move_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_cpu(item) for item in value)
    return value


def capture_rng_state(train_loader: DataLoader) -> dict[str, Any]:
    generator_state = None
    if train_loader.generator is not None:
        generator_state = train_loader.generator.get_state()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "train_loader_generator": generator_state,
    }


def restore_rng_state(state: dict[str, Any], train_loader: DataLoader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator_state = state.get("train_loader_generator")
    if generator_state is not None:
        if train_loader.generator is None:
            raise ValueError(
                "Checkpoint contains a DataLoader generator state, but the "
                "current train DataLoader has no generator."
            )
        train_loader.generator.set_state(generator_state)


def build_checkpoint(
    *,
    epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: Any,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    train_loader: DataLoader,
    wandb_run_id: str | None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "model_state_dict": _move_to_cpu(model.state_dict()),
        "optimizer_state_dict": _move_to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": _move_to_cpu(scheduler.state_dict()),
        "scaler_state_dict": _move_to_cpu(scaler.state_dict()),
        "config": config,
        "history": history,
        "rng_state": capture_rng_state(train_loader),
        "wandb_run_id": wandb_run_id,
    }


def atomic_save_checkpoint(checkpoint: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def load_checkpoint(path: Path, map_location: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint root must be a dictionary.")
    required = {
        "epoch",
        "global_step",
        "best_metric",
        "best_epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "config",
        "history",
        "rng_state",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint is missing fields: {sorted(missing)}.")
    return checkpoint


def restore_training_state(
    *,
    checkpoint: dict[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: Any,
    train_loader: DataLoader,
) -> None:
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_rng_state(checkpoint["rng_state"], train_loader)
