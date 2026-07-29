"""Checkpoint persistence for the Exercise 2 training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn


def _move_to_cpu(value: Any) -> Any:
    """Recursively detach tensors and move them to CPU memory."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()

    if isinstance(value, dict):
        return {
            key: _move_to_cpu(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_move_to_cpu(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_move_to_cpu(item) for item in value)

    return value


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: DictConfig,
    epoch: int,
    monitor: str,
    monitored_value: float,
) -> None:
    """Atomically save the current best model and experiment state."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )

    checkpoint = {
        "epoch": int(epoch),
        "monitor": str(monitor),
        "monitored_value": float(monitored_value),
        "model_state_dict": _move_to_cpu(model.state_dict()),
        "optimizer_state_dict": _move_to_cpu(optimizer.state_dict()),
        "config": OmegaConf.to_container(
            config,
            resolve=True,
            enum_to_str=True,
        ),
    }

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Load a checkpoint into a model and return its stored metadata."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
