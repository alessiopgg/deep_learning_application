"""Runtime utilities for reproducible Exercise 2 experiments."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_reproducibility(seed: int, deterministic: bool) -> None:
    """Seed Python, NumPy and PyTorch and optionally request deterministic ops."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def resolve_device(requested_device: str) -> torch.device:
    """Resolve ``auto``, CPU or a requested CUDA device with clear errors."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested_device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA device was requested, but CUDA is not available. "
                "Use experiment.device=cpu or experiment.device=auto."
            )

        device_index = 0 if device.index is None else device.index
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device_index} is unavailable. "
                f"Detected CUDA devices: {torch.cuda.device_count()}."
            )

    return device


def describe_device(device: torch.device) -> str:
    """Return a readable description of the selected execution device."""
    if device.type != "cuda":
        return "CPU"

    device_index = 0 if device.index is None else device.index
    return f"CUDA:{device_index} - {torch.cuda.get_device_name(device_index)}"
