"""One-epoch training and validation-loss loops for Faster R-CNN."""

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


BatchLogger = Callable[[dict[str, Any]], None]


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
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
    resolved_enabled = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler(
            device.type,
            init_scale=initial_scale,
            enabled=resolved_enabled,
        )
    except TypeError:
        return torch.cuda.amp.GradScaler(
            init_scale=initial_scale,
            enabled=resolved_enabled,
        )


def autocast_context(
    device: torch.device,
    enabled: bool,
) -> ContextManager[Any]:
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def move_batch_to_device(
    images: list[torch.Tensor],
    targets: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    non_blocking = device.type == "cuda"
    moved_images = [
        image.to(device, non_blocking=non_blocking) for image in images
    ]
    moved_targets = [
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
    return moved_images, moved_targets


def validate_and_sum_losses(
    loss_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    if set(loss_dict) != set(EXPECTED_LOSS_KEYS):
        raise ValueError(
            f"Expected loss keys {list(EXPECTED_LOSS_KEYS)}, "
            f"found {sorted(loss_dict)}."
        )

    scalar_losses: dict[str, float] = {}
    for name in EXPECTED_LOSS_KEYS:
        loss = loss_dict[name]
        if not torch.is_tensor(loss) or loss.numel() != 1:
            raise TypeError(f"Loss {name!r} must be a scalar tensor.")
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Loss {name!r} contains NaN or Inf.")
        scalar = float(loss.detach().float().item())
        if scalar < 0:
            raise ValueError(f"Loss {name!r} is negative: {scalar}.")
        scalar_losses[name] = scalar

    total_loss = sum(loss_dict.values())
    if not torch.isfinite(total_loss).all():
        raise FloatingPointError("The total detection loss contains NaN or Inf.")
    return total_loss, scalar_losses


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        torch.cuda.set_device(index)
        torch.cuda.reset_peak_memory_stats(index)


def _capture_peak_memory(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda":
        return None, None
    index = 0 if device.index is None else device.index
    torch.cuda.synchronize(index)
    return (
        int(torch.cuda.max_memory_allocated(index)),
        int(torch.cuda.max_memory_reserved(index)),
    )


def _batch_counts(targets: list[dict[str, Any]]) -> tuple[int, int]:
    objects = sum(int(target["boxes"].shape[0]) for target in targets)
    empty = sum(int(target["boxes"].shape[0] == 0) for target in targets)
    return objects, empty


def _finalize_metrics(
    *,
    split: str,
    weighted_losses: dict[str, float],
    images: int,
    objects: int,
    empty_images: int,
    batches: int,
    duration_seconds: float,
    optimizer_steps: int,
    amp_skipped_steps: int,
    gradient_clip_norm: float | None,
    device: torch.device,
) -> EpochLossMetrics:
    if images <= 0 or batches <= 0:
        raise RuntimeError(f"No samples were processed for split {split!r}.")
    peak_allocated, peak_reserved = _capture_peak_memory(device)
    means = {name: value / images for name, value in weighted_losses.items()}
    total_loss = sum(means.values())
    return EpochLossMetrics(
        split=split,
        total_loss=total_loss,
        loss_classifier=means["loss_classifier"],
        loss_box_reg=means["loss_box_reg"],
        loss_objectness=means["loss_objectness"],
        loss_rpn_box_reg=means["loss_rpn_box_reg"],
        batches=batches,
        images=images,
        objects=objects,
        empty_images=empty_images,
        duration_seconds=duration_seconds,
        optimizer_steps=optimizer_steps,
        amp_skipped_steps=amp_skipped_steps,
        gradient_clip_norm=gradient_clip_norm,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    freeze_backbone: bool,
    epoch: int,
    global_step: int,
    logging_interval: int,
    gradient_clip_norm: float | None,
    max_batches: int | None,
    batch_logger: BatchLogger | None = None,
) -> tuple[EpochLossMetrics, int]:
    configure_model_for_training(model, freeze_backbone=freeze_backbone)
    _reset_peak_memory(device)

    weighted_losses = {name: 0.0 for name in EXPECTED_LOSS_KEYS}
    processed_images = 0
    processed_objects = 0
    empty_images = 0
    processed_batches = 0
    optimizer_steps = 0
    amp_skipped_steps = 0
    start_time = time.perf_counter()

    for batch_index, (images_cpu, targets_cpu) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, targets = move_batch_to_device(images_cpu, targets_cpu, device)
        batch_size = len(images)
        batch_objects, batch_empty = _batch_counts(targets)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            loss_dict = model(images, targets)
            total_loss, scalar_losses = validate_and_sum_losses(loss_dict)

        scale_before = float(scaler.get_scale())
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

        if gradient_clip_norm is not None:
            clipped_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=gradient_clip_norm,
                error_if_nonfinite=True,
            )
            if not math.isfinite(float(clipped_norm)):
                raise FloatingPointError("The gradient norm is NaN or Inf.")

        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        step_skipped = scale_after < scale_before
        if step_skipped:
            amp_skipped_steps += 1
        else:
            optimizer_steps += 1

        processed_batches += 1
        processed_images += batch_size
        processed_objects += batch_objects
        empty_images += batch_empty
        global_step += 1
        for name, value in scalar_losses.items():
            weighted_losses[name] += value * batch_size

        should_log = (
            processed_batches == 1
            or processed_batches % logging_interval == 0
        )
        if should_log:
            elapsed = time.perf_counter() - start_time
            batch_payload = {
                "global_step": global_step,
                "epoch": epoch,
                "batch_index": batch_index,
                "train_batch_total_loss": float(
                    total_loss.detach().float().item()
                ),
                **{
                    f"train_batch_{name}": value
                    for name, value in scalar_losses.items()
                },
                "train_batch_images": batch_size,
                "train_batch_objects": batch_objects,
                "train_batch_empty_images": batch_empty,
                "train_amp_scale": scale_after,
                "train_amp_step_skipped": int(step_skipped),
                "train_elapsed_seconds": elapsed,
            }
            print(
                f"  epoch {epoch} batch {processed_batches}: "
                f"loss={batch_payload['train_batch_total_loss']:.6f}, "
                f"objects={batch_objects}, "
                f"amp_skipped={step_skipped}"
            )
            if batch_logger is not None:
                batch_logger(batch_payload)

    duration = time.perf_counter() - start_time
    metrics = _finalize_metrics(
        split="train",
        weighted_losses=weighted_losses,
        images=processed_images,
        objects=processed_objects,
        empty_images=empty_images,
        batches=processed_batches,
        duration_seconds=duration,
        optimizer_steps=optimizer_steps,
        amp_skipped_steps=amp_skipped_steps,
        gradient_clip_norm=gradient_clip_norm,
        device=device,
    )
    return metrics, global_step


@contextmanager
def preserve_random_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@torch.no_grad()
def evaluate_validation_loss(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    freeze_backbone: bool,
    validation_seed: int,
    max_batches: int | None,
) -> EpochLossMetrics:
    """Compute validation losses without gradients or parameter updates.

    Torchvision detection models return losses only in training mode. We thus
    put the RPN and RoI heads in training mode under ``torch.no_grad()`` while
    keeping the frozen backbone in evaluation mode. The RNG state is restored
    afterwards so validation sampling cannot perturb the next training epoch.
    """
    _reset_peak_memory(device)
    weighted_losses = {name: 0.0 for name in EXPECTED_LOSS_KEYS}
    processed_images = 0
    processed_objects = 0
    empty_images = 0
    processed_batches = 0
    start_time = time.perf_counter()

    with preserve_random_state():
        random.seed(validation_seed)
        np.random.seed(validation_seed)
        torch.manual_seed(validation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(validation_seed)

        configure_model_for_training(model, freeze_backbone=freeze_backbone)
        for batch_index, (images_cpu, targets_cpu) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, targets = move_batch_to_device(images_cpu, targets_cpu, device)
            batch_size = len(images)
            batch_objects, batch_empty = _batch_counts(targets)

            with autocast_context(device, amp_enabled):
                loss_dict = model(images, targets)
                _, scalar_losses = validate_and_sum_losses(loss_dict)

            processed_batches += 1
            processed_images += batch_size
            processed_objects += batch_objects
            empty_images += batch_empty
            for name, value in scalar_losses.items():
                weighted_losses[name] += value * batch_size

    duration = time.perf_counter() - start_time
    return _finalize_metrics(
        split="validation",
        weighted_losses=weighted_losses,
        images=processed_images,
        objects=processed_objects,
        empty_images=empty_images,
        batches=processed_batches,
        duration_seconds=duration,
        optimizer_steps=0,
        amp_skipped_steps=0,
        gradient_clip_norm=None,
        device=device,
    )
