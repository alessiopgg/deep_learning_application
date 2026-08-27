"""One-epoch Faster R-CNN training and validation-loss loops."""

from __future__ import annotations

import math
import random
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Callable, ContextManager, Iterator

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from Exercise3.models.faster_rcnn import configure_model_for_training

EXPECTED_LOSS_KEYS = (
    "loss_classifier",
    "loss_box_reg",
    "loss_objectness",
    "loss_rpn_box_reg",
)
BatchLogger = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class EpochLossMetrics:
    split: str
    total_loss: float
    loss_classifier: float
    loss_box_reg: float
    loss_objectness: float
    loss_rpn_box_reg: float
    batches: int
    images: int
    objects: int
    empty_images: int
    duration_seconds: float
    optimizer_steps: int
    amp_skipped_steps: int
    gradient_clip_norm: float | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA index {index} is invalid; found "
                f"{torch.cuda.device_count()} CUDA device(s)."
            )
    return device


def set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def build_grad_scaler(
    device: torch.device,
    enabled: bool,
    initial_scale: float,
) -> Any:
    enabled = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler(
            device.type, init_scale=initial_scale, enabled=enabled
        )
    except TypeError:
        return torch.cuda.amp.GradScaler(
            init_scale=initial_scale, enabled=enabled
        )


def autocast_context(
    device: torch.device,
    enabled: bool,
) -> ContextManager[Any]:
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.float16)


def move_batch_to_device(images, targets, device):
    non_blocking = device.type == "cuda"
    images = [image.to(device, non_blocking=non_blocking) for image in images]
    targets = [
        {
            key: (
                value.to(device, non_blocking=non_blocking)
                if torch.is_tensor(value)
                else value
            )
            for key, value in target.items()
        }
        for target in targets
    ]
    return images, targets


def validate_and_sum_losses(
    loss_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    if set(loss_dict) != set(EXPECTED_LOSS_KEYS):
        raise ValueError(
            f"Expected losses {list(EXPECTED_LOSS_KEYS)}, "
            f"found {sorted(loss_dict)}."
        )
    scalar = {}
    for name in EXPECTED_LOSS_KEYS:
        value = loss_dict[name]
        if not torch.is_tensor(value) or value.numel() != 1:
            raise TypeError(f"Loss {name!r} must be a scalar tensor.")
        number = float(value.detach().float())
        if not math.isfinite(number) or number < 0:
            raise FloatingPointError(f"Invalid loss {name!r}: {number}.")
        scalar[name] = number
    total = sum(loss_dict.values())
    if not torch.isfinite(total):
        raise FloatingPointError("Total detection loss contains NaN or Inf.")
    return total, scalar


def _memory(device: torch.device, reset: bool = False):
    if device.type != "cuda":
        return (None, None)
    index = 0 if device.index is None else device.index
    torch.cuda.set_device(index)
    if reset:
        torch.cuda.reset_peak_memory_stats(index)
        return (None, None)
    torch.cuda.synchronize(index)
    return (
        int(torch.cuda.max_memory_allocated(index)),
        int(torch.cuda.max_memory_reserved(index)),
    )


def _counts(targets) -> tuple[int, int]:
    objects = sum(int(target["boxes"].shape[0]) for target in targets)
    empty = sum(int(target["boxes"].shape[0] == 0) for target in targets)
    return objects, empty


def _metrics(
    split: str,
    losses: dict[str, float],
    images: int,
    objects: int,
    empty: int,
    batches: int,
    duration: float,
    optimizer_steps: int,
    skipped_steps: int,
    clip_norm: float | None,
    device: torch.device,
) -> EpochLossMetrics:
    if images <= 0 or batches <= 0:
        raise RuntimeError(f"No samples processed for split {split!r}.")
    allocated, reserved = _memory(device)
    mean = {name: value / images for name, value in losses.items()}
    return EpochLossMetrics(
        split=split,
        total_loss=sum(mean.values()),
        **mean,
        batches=batches,
        images=images,
        objects=objects,
        empty_images=empty,
        duration_seconds=duration,
        optimizer_steps=optimizer_steps,
        amp_skipped_steps=skipped_steps,
        gradient_clip_norm=clip_norm,
        peak_allocated_bytes=allocated,
        peak_reserved_bytes=reserved,
    )


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    trainable_backbone: str,
    epoch: int,
    global_step: int,
    logging_interval: int,
    gradient_clip_norm: float | None,
    max_batches: int | None,
    batch_logger: BatchLogger | None = None,
) -> tuple[EpochLossMetrics, int]:
    configure_model_for_training(
        model, trainable_backbone=trainable_backbone
    )
    _memory(device, reset=True)
    losses = {name: 0.0 for name in EXPECTED_LOSS_KEYS}
    images_seen = objects_seen = empty_seen = batches = 0
    optimizer_steps = skipped_steps = 0
    start = time.perf_counter()

    for batch_index, (images_cpu, targets_cpu) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images, targets = move_batch_to_device(
            images_cpu, targets_cpu, device
        )
        batch_size = len(images)
        batch_objects, batch_empty = _counts(targets)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            loss_dict = model(images, targets)
            total_loss, scalar_losses = validate_and_sum_losses(loss_dict)

        scale_before = float(scaler.get_scale())
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

        if gradient_clip_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                gradient_clip_norm,
                error_if_nonfinite=True,
            )
            if not math.isfinite(float(norm)):
                raise FloatingPointError("Gradient norm is NaN or Inf.")

        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        skipped = scale_after < scale_before
        skipped_steps += int(skipped)
        optimizer_steps += int(not skipped)

        batches += 1
        images_seen += batch_size
        objects_seen += batch_objects
        empty_seen += batch_empty
        global_step += 1
        for name, value in scalar_losses.items():
            losses[name] += value * batch_size

        if batches == 1 or batches % logging_interval == 0:
            payload = {
                "global_step": global_step,
                "epoch": epoch,
                "batch_index": batch_index,
                "train_batch_total_loss": float(total_loss.detach()),
                **{
                    f"train_batch_{name}": value
                    for name, value in scalar_losses.items()
                },
                "train_batch_images": batch_size,
                "train_batch_objects": batch_objects,
                "train_batch_empty_images": batch_empty,
                "train_amp_scale": scale_after,
                "train_amp_step_skipped": int(skipped),
                "train_elapsed_seconds": time.perf_counter() - start,
            }
            print(
                f"  epoch {epoch} batch {batches}: "
                f"loss={payload['train_batch_total_loss']:.6f}, "
                f"objects={batch_objects}, amp_skipped={skipped}"
            )
            if batch_logger:
                batch_logger(payload)

    return (
        _metrics(
            "train",
            losses,
            images_seen,
            objects_seen,
            empty_seen,
            batches,
            time.perf_counter() - start,
            optimizer_steps,
            skipped_steps,
            gradient_clip_norm,
            device,
        ),
        global_step,
    )


@contextmanager
def preserve_random_state() -> Iterator[None]:
    state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])


@torch.no_grad()
def evaluate_validation_loss(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    trainable_backbone: str,
    validation_seed: int,
    max_batches: int | None,
) -> EpochLossMetrics:
    """Faster R-CNN exposes losses in train mode; gradients remain disabled."""
    _memory(device, reset=True)
    losses = {name: 0.0 for name in EXPECTED_LOSS_KEYS}
    images_seen = objects_seen = empty_seen = batches = 0
    start = time.perf_counter()

    with preserve_random_state():
        random.seed(validation_seed)
        np.random.seed(validation_seed)
        torch.manual_seed(validation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(validation_seed)

        configure_model_for_training(
            model, trainable_backbone=trainable_backbone
        )
        for batch_index, (images_cpu, targets_cpu) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images, targets = move_batch_to_device(
                images_cpu, targets_cpu, device
            )
            batch_size = len(images)
            batch_objects, batch_empty = _counts(targets)
            with autocast_context(device, amp_enabled):
                _, scalar = validate_and_sum_losses(model(images, targets))

            batches += 1
            images_seen += batch_size
            objects_seen += batch_objects
            empty_seen += batch_empty
            for name, value in scalar.items():
                losses[name] += value * batch_size

    return _metrics(
        "validation",
        losses,
        images_seen,
        objects_seen,
        empty_seen,
        batches,
        time.perf_counter() - start,
        0,
        0,
        None,
        device,
    )
