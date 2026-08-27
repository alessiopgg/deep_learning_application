"""Step 10 end-to-end smoke test for the Faster R-CNN baseline.

The script executes exactly one training step and one inference pass.  Its goal
is structural validation, not model-quality evaluation:

1. obtain one non-empty training sample from the final DataLoader pipeline;
2. run Faster R-CNN in training mode and inspect its four losses;
3. backpropagate, inspect gradients and execute one optimizer step;
4. prove that trainable detector parameters changed while the frozen backbone
   remained bitwise unchanged;
5. run inference and validate boxes, labels and scores;
6. record CUDA peak memory when available.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager

import torch
import torchvision
from torch import nn
from torch.optim import SGD, Optimizer

from Exercise3.data_pipeline.loaders import build_detection_dataloaders
from Exercise3.data_pipeline.loading import (
    DATASET_CONFIGURATION,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    DEFAULT_CACHE_DIR,
    load_detection_dataset,
)
from Exercise3.data_pipeline.taxonomy import NUM_DETECTOR_CLASSES
from Exercise3.models.faster_rcnn import (
    FasterRCNNBaselineConfig,
    build_faster_rcnn_baseline,
    configure_model_for_training,
)
from Exercise3.paths import OUTPUT_DIR


DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "step_10" / "smoke_test.json"
EXPECTED_LOSS_KEYS = {
    "loss_classifier",
    "loss_box_reg",
    "loss_objectness",
    "loss_rpn_box_reg",
}
REPRESENTATIVE_BACKBONE_PARAMETER = "backbone.body.conv1.weight"
REPRESENTATIVE_TRAINABLE_PARAMETER = (
    "roi_heads.box_predictor.cls_score.weight"
)
DEFAULT_AMP_INITIAL_SCALE = 1024.0
DEFAULT_MAX_AMP_RETRIES = 6


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Faster R-CNN training step and one inference pass on a "
            "real traffic-sign sample."
        )
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda or cuda:<index>.",
    )
    parser.add_argument(
        "--weights",
        choices=("coco", "none"),
        default="coco",
        help="The scientific baseline uses official COCO weights.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Hugging Face cache directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Zero is the safe default on Windows.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.005,
        help="SGD learning rate used only for this one-step smoke test.",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument(
        "--max-search-batches",
        type=int,
        default=100,
        help="Maximum train batches inspected to find one non-empty sample.",
    )
    parser.add_argument(
        "--inference-score-threshold",
        type=float,
        default=0.0,
        help=(
            "Temporary score threshold for structural inference validation. "
            "Zero avoids hiding outputs from the newly initialized head."
        ),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable AMP. Default: enabled automatically on CUDA.",
    )
    parser.add_argument(
        "--amp-initial-scale",
        type=float,
        default=DEFAULT_AMP_INITIAL_SCALE,
        help=(
            "Initial GradScaler scale. A conservative value avoids first-step "
            "FP16 overflow in the newly initialized detection head."
        ),
    )
    parser.add_argument(
        "--max-amp-retries",
        type=int,
        default=DEFAULT_MAX_AMP_RETRIES,
        help=(
            "Maximum number of retries after GradScaler detects non-finite "
            "gradients and skips an optimizer step."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide Torchvision weight-download progress.",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if arguments.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if arguments.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than zero.")
    if not 0 <= arguments.momentum < 1:
        raise ValueError("--momentum must satisfy 0 <= momentum < 1.")
    if arguments.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")
    if arguments.max_search_batches <= 0:
        raise ValueError("--max-search-batches must be greater than zero.")
    if not 0 <= arguments.inference_score_threshold <= 1:
        raise ValueError(
            "--inference-score-threshold must be within [0,1]."
        )
    if arguments.amp_initial_scale <= 0:
        raise ValueError("--amp-initial-scale must be greater than zero.")
    if arguments.max_amp_retries < 0:
        raise ValueError("--max-amp-retries cannot be negative.")


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
                f"CUDA device index {index} is invalid; found "
                f"{torch.cuda.device_count()} CUDA device(s)."
            )
    return device


def set_reproducibility_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def capture_cuda_memory(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "available": False,
            "device_name": None,
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }

    synchronize_cuda(device)
    index = 0 if device.index is None else device.index
    return {
        "available": True,
        "device_index": index,
        "device_name": torch.cuda.get_device_name(index),
        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "reserved_bytes": int(torch.cuda.memory_reserved(index)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
    }


def format_megabytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024 ** 2):.2f} MiB"


def resolve_amp_enabled(
    requested: bool | None,
    device: torch.device,
) -> bool:
    if requested is None:
        return device.type == "cuda"
    if requested and device.type != "cuda":
        raise ValueError(
            "This smoke test enables AMP only on CUDA. Use --no-amp on CPU."
        )
    return bool(requested)


def autocast_context(
    device: torch.device,
    enabled: bool,
) -> ContextManager[Any]:
    if not enabled:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=True,
    )


def build_grad_scaler(
    device: torch.device,
    enabled: bool,
    *,
    initial_scale: float,
) -> Any:
    """Create a GradScaler with a conservative, explicit initial scale."""
    try:
        return torch.amp.GradScaler(
            device.type,
            init_scale=initial_scale,
            enabled=enabled,
        )
    except TypeError:
        # Compatibility with older PyTorch releases where the first positional
        # device argument was not available on torch.amp.GradScaler.
        return torch.cuda.amp.GradScaler(
            init_scale=initial_scale,
            enabled=enabled,
        )


def build_smoke_test_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> Optimizer:
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("The detector has no trainable parameters.")

    optimizer = SGD(
        trainable_parameters,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    expected_parameter_ids = {id(parameter) for parameter in trainable_parameters}
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_parameter_ids != expected_parameter_ids:
        raise ValueError(
            "The optimizer parameter groups do not match the trainable model "
            "parameters exactly."
        )
    return optimizer


def move_batch_to_device(
    images: list[torch.Tensor],
    targets: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    non_blocking = device.type == "cuda"
    moved_images = [
        image.to(device, non_blocking=non_blocking)
        for image in images
    ]
    moved_targets: list[dict[str, Any]] = []
    for target in targets:
        moved_targets.append(
            {
                key: (
                    value.to(device, non_blocking=non_blocking)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in target.items()
            }
        )
    return moved_images, moved_targets


def select_non_empty_training_batch(
    train_loader: torch.utils.data.DataLoader,
    *,
    max_search_batches: int,
) -> tuple[list[torch.Tensor], list[dict[str, Any]], int]:
    for batch_index, (images, targets) in enumerate(train_loader):
        if batch_index >= max_search_batches:
            break
        if len(images) != 1 or len(targets) != 1:
            raise ValueError(
                "The smoke-test train loader must use batch_size=1."
            )
        if int(targets[0]["boxes"].shape[0]) > 0:
            return images, targets, batch_index

    raise RuntimeError(
        "No non-empty training sample was found within "
        f"{max_search_batches} batches."
    )


def validate_loss_dict(
    loss_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    actual_keys = set(loss_dict)
    if actual_keys != EXPECTED_LOSS_KEYS:
        raise ValueError(
            "Unexpected Faster R-CNN loss keys. "
            f"Expected {sorted(EXPECTED_LOSS_KEYS)}, found {sorted(actual_keys)}."
        )

    values: dict[str, float] = {}
    for name in sorted(EXPECTED_LOSS_KEYS):
        value = loss_dict[name]
        if not torch.is_tensor(value) or value.numel() != 1:
            raise TypeError(f"Loss '{name}' must be a scalar tensor.")
        if not torch.isfinite(value).all():
            raise ValueError(f"Loss '{name}' contains NaN or Inf.")
        scalar = float(value.detach().float().item())
        if scalar < 0:
            raise ValueError(f"Loss '{name}' is negative: {scalar}.")
        values[name] = scalar

    total_loss = sum(loss_dict.values())
    if not torch.isfinite(total_loss).all():
        raise ValueError("The total Faster R-CNN loss contains NaN or Inf.")
    if float(total_loss.detach().float().item()) <= 0:
        raise ValueError("The total Faster R-CNN loss must be positive.")
    return total_loss, values


def find_nonfinite_gradient_parameters(model: nn.Module) -> list[str]:
    """Return trainable parameter names whose gradients contain NaN or Inf."""
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not torch.isfinite(parameter.grad.detach()).all()
    ]


def summarize_gradients(model: nn.Module) -> dict[str, Any]:
    trainable_with_gradient: list[str] = []
    trainable_without_gradient: list[str] = []
    frozen_with_gradient: list[str] = []
    nonfinite_gradient_parameters: list[str] = []
    sum_squared_norms = 0.0
    maximum_absolute_gradient = 0.0

    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if parameter.requires_grad:
            if gradient is None:
                trainable_without_gradient.append(name)
                continue
            trainable_with_gradient.append(name)
            gradient_float = gradient.detach().float()
            if not torch.isfinite(gradient_float).all():
                nonfinite_gradient_parameters.append(name)
                continue
            gradient_norm = float(torch.linalg.vector_norm(gradient_float).item())
            sum_squared_norms += gradient_norm**2
            if gradient_float.numel() > 0:
                maximum_absolute_gradient = max(
                    maximum_absolute_gradient,
                    float(gradient_float.abs().max().item()),
                )
        elif gradient is not None:
            frozen_with_gradient.append(name)

    if not trainable_with_gradient:
        raise ValueError("No trainable parameter received a gradient.")
    if frozen_with_gradient:
        raise ValueError(
            "Frozen parameters unexpectedly received gradients: "
            f"{frozen_with_gradient[:10]}."
        )
    if nonfinite_gradient_parameters:
        raise ValueError(
            "Non-finite gradients found in: "
            f"{nonfinite_gradient_parameters[:10]}."
        )

    global_l2_norm = math.sqrt(sum_squared_norms)
    if not math.isfinite(global_l2_norm) or global_l2_norm <= 0:
        raise ValueError(
            f"Invalid global gradient norm: {global_l2_norm}."
        )

    return {
        "trainable_parameter_tensors_with_gradient": len(
            trainable_with_gradient
        ),
        "trainable_parameter_tensors_without_gradient": len(
            trainable_without_gradient
        ),
        "frozen_parameter_tensors_with_gradient": len(frozen_with_gradient),
        "nonfinite_gradient_parameter_tensors": len(
            nonfinite_gradient_parameters
        ),
        "global_l2_norm": global_l2_norm,
        "maximum_absolute_gradient": maximum_absolute_gradient,
        "first_trainable_parameters_with_gradient": trainable_with_gradient[:20],
        "first_trainable_parameters_without_gradient": (
            trainable_without_gradient[:20]
        ),
    }


def validate_predictions(
    predictions: Any,
    *,
    image_height: int,
    image_width: int,
) -> dict[str, Any]:
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise TypeError(
            "Inference must return a one-element list for a one-image batch."
        )
    prediction = predictions[0]
    required_keys = {"boxes", "labels", "scores"}
    missing_keys = required_keys.difference(prediction)
    if missing_keys:
        raise KeyError(
            f"Prediction is missing keys: {sorted(missing_keys)}."
        )

    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]

    if not torch.is_tensor(boxes) or boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(
            f"Predicted boxes must have shape [N,4], found {tuple(boxes.shape)}."
        )
    if not torch.is_floating_point(boxes):
        raise TypeError("Predicted boxes must use a floating-point dtype.")
    if labels.dtype != torch.int64 or labels.ndim != 1:
        raise TypeError("Predicted labels must be an int64 vector.")
    if not torch.is_floating_point(scores) or scores.ndim != 1:
        raise TypeError("Predicted scores must be a floating-point vector.")

    detection_count = int(boxes.shape[0])
    if labels.shape[0] != detection_count or scores.shape[0] != detection_count:
        raise ValueError(
            "Predicted boxes, labels and scores have inconsistent lengths."
        )
    if detection_count == 0:
        raise ValueError(
            "Inference returned zero detections even with score threshold 0.0."
        )

    boxes_float = boxes.detach().float()
    scores_float = scores.detach().float()
    if not torch.isfinite(boxes_float).all():
        raise ValueError("Predicted boxes contain NaN or Inf.")
    if not torch.isfinite(scores_float).all():
        raise ValueError("Predicted scores contain NaN or Inf.")

    x_min, y_min, x_max, y_max = boxes_float.unbind(dim=1)
    tolerance = 1e-3
    valid_geometry = (
        (x_min >= -tolerance).all()
        and (y_min >= -tolerance).all()
        and (x_max <= image_width + tolerance).all()
        and (y_max <= image_height + tolerance).all()
        and (x_max > x_min).all()
        and (y_max > y_min).all()
    )
    if not bool(valid_geometry):
        raise ValueError(
            "Predicted boxes are degenerate or outside the original image."
        )
    if not bool(
        ((labels >= 1) & (labels < NUM_DETECTOR_CLASSES)).all()
    ):
        raise ValueError(
            "Predicted foreground labels must be in the range 1..43."
        )
    if not bool(((scores_float >= 0) & (scores_float <= 1)).all()):
        raise ValueError("Predicted scores must be probabilities in [0,1].")

    preview_count = min(10, detection_count)
    preview = []
    for index in range(preview_count):
        preview.append(
            {
                "rank": index + 1,
                "box_xyxy": [
                    float(value)
                    for value in boxes_float[index].cpu().tolist()
                ],
                "label": int(labels[index].item()),
                "score": float(scores_float[index].item()),
            }
        )

    return {
        "prediction_keys": sorted(prediction.keys()),
        "detection_count": detection_count,
        "boxes_shape": list(boxes.shape),
        "boxes_dtype": str(boxes.dtype),
        "labels_shape": list(labels.shape),
        "labels_dtype": str(labels.dtype),
        "scores_shape": list(scores.shape),
        "scores_dtype": str(scores.dtype),
        "score_min": float(scores_float.min().item()),
        "score_max": float(scores_float.max().item()),
        "observed_labels": sorted(
            {int(value) for value in labels.detach().cpu().tolist()}
        ),
        "top_predictions": preview,
        "all_boxes_finite": True,
        "all_boxes_inside_image": True,
        "all_scores_finite": True,
        "all_scores_in_unit_interval": True,
        "all_labels_in_foreground_range": True,
    }


def run_smoke_test(arguments: argparse.Namespace) -> dict[str, Any]:
    validate_arguments(arguments)
    device = resolve_device(arguments.device)
    amp_enabled = resolve_amp_enabled(arguments.amp, device)
    set_reproducibility_seed(arguments.seed)

    if device.type == "cuda":
        cuda_index = 0 if device.index is None else device.index

        # Imposta esplicitamente la GPU corrente. Alcune versioni di PyTorch
        # su Windows non accettano correttamente torch.device in
        # reset_peak_memory_stats().
        torch.cuda.set_device(cuda_index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    dataset_dict, resolved_cache_dir = load_detection_dataset(
        arguments.cache_dir
    )
    loader_bundle = build_detection_dataloaders(
        dataset_dict,
        train_batch_size=1,
        evaluation_batch_size=1,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        seed=arguments.seed,
    )
    images_cpu, targets_cpu, selected_batch_index = (
        select_non_empty_training_batch(
            loader_bundle.train,
            max_search_batches=arguments.max_search_batches,
        )
    )

    sample_image_id = int(targets_cpu[0]["image_id"])
    sample_box_count = int(targets_cpu[0]["boxes"].shape[0])
    sample_height, sample_width = map(int, images_cpu[0].shape[-2:])
    sample_labels = [int(value) for value in targets_cpu[0]["labels"].tolist()]

    config = FasterRCNNBaselineConfig(
        weights=arguments.weights,
        seed=arguments.seed,
        progress=not arguments.no_progress,
    )
    model, construction_metadata = build_faster_rcnn_baseline(config)
    model.to(device)
    configure_model_for_training(model, freeze_backbone=True)

    optimizer = build_smoke_test_optimizer(
        model,
        learning_rate=arguments.learning_rate,
        momentum=arguments.momentum,
        weight_decay=arguments.weight_decay,
    )
    scaler = build_grad_scaler(
        device,
        amp_enabled,
        initial_scale=arguments.amp_initial_scale,
    )

    named_parameters = dict(model.named_parameters())
    backbone_parameter = named_parameters[REPRESENTATIVE_BACKBONE_PARAMETER]
    trainable_parameter = named_parameters[REPRESENTATIVE_TRAINABLE_PARAMETER]
    backbone_before = backbone_parameter.detach().cpu().clone()
    trainable_before = trainable_parameter.detach().cpu().clone()

    images, targets = move_batch_to_device(images_cpu, targets_cpu, device)
    memory_after_batch_move = capture_cuda_memory(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    amp_attempts: list[dict[str, Any]] = []
    forward_seconds = 0.0
    backward_seconds = 0.0
    optimizer_step_seconds = 0.0
    successful_training_step = False

    # GradScaler may legitimately find Inf/NaN gradients during its first
    # calibration iterations. In that case scaler.step() skips optimizer.step(),
    # scaler.update() lowers the scale, and we retry the same structural step.
    for attempt_index in range(arguments.max_amp_retries + 1):
        optimizer.zero_grad(set_to_none=True)
        scale_before_backward = float(scaler.get_scale())

        forward_start = time.perf_counter()
        with autocast_context(device, amp_enabled):
            loss_dict = model(images, targets)
            total_loss, scalar_losses = validate_loss_dict(loss_dict)
        synchronize_cuda(device)
        current_forward_seconds = time.perf_counter() - forward_start
        forward_seconds += current_forward_seconds
        memory_after_forward = capture_cuda_memory(device)

        backward_start = time.perf_counter()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        synchronize_cuda(device)
        current_backward_seconds = time.perf_counter() - backward_start
        backward_seconds += current_backward_seconds

        nonfinite_gradient_parameters = find_nonfinite_gradient_parameters(
            model
        )
        memory_after_backward = capture_cuda_memory(device)

        if nonfinite_gradient_parameters:
            if not amp_enabled:
                raise ValueError(
                    "Non-finite gradients were produced without AMP: "
                    f"{nonfinite_gradient_parameters[:10]}."
                )

            # GradScaler inspects the already-unscaled gradients, skips the
            # optimizer step, and reduces the scale for the next attempt.
            skipped_step_start = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            synchronize_cuda(device)
            skipped_step_seconds = time.perf_counter() - skipped_step_start
            scale_after_update = float(scaler.get_scale())

            amp_attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "scale_before_backward": scale_before_backward,
                    "scale_after_update": scale_after_update,
                    "optimizer_step_skipped": True,
                    "nonfinite_gradient_parameters": (
                        nonfinite_gradient_parameters
                    ),
                    "losses": scalar_losses,
                    "total_loss": float(
                        total_loss.detach().float().item()
                    ),
                    "forward_seconds": current_forward_seconds,
                    "backward_seconds": current_backward_seconds,
                    "scaler_step_seconds": skipped_step_seconds,
                }
            )

            del loss_dict, total_loss
            if attempt_index >= arguments.max_amp_retries:
                raise ValueError(
                    "AMP still produced non-finite gradients after "
                    f"{arguments.max_amp_retries + 1} attempts. "
                    "Retry with --no-amp or a smaller "
                    "--amp-initial-scale."
                )
            continue

        gradient_report = summarize_gradients(model)
        representative_gradient = trainable_parameter.grad
        if representative_gradient is None:
            raise ValueError(
                f"'{REPRESENTATIVE_TRAINABLE_PARAMETER}' received no gradient."
            )
        representative_gradient_norm = float(
            torch.linalg.vector_norm(
                representative_gradient.detach().float()
            ).item()
        )
        if (
            not math.isfinite(representative_gradient_norm)
            or representative_gradient_norm <= 0
        ):
            raise ValueError(
                "The representative predictor gradient is zero or non-finite."
            )

        optimizer_step_start = time.perf_counter()
        scaler.step(optimizer)
        scaler.update()
        synchronize_cuda(device)
        optimizer_step_seconds = (
            time.perf_counter() - optimizer_step_start
        )
        scale_after_update = float(scaler.get_scale())
        memory_after_optimizer_step = capture_cuda_memory(device)

        amp_attempts.append(
            {
                "attempt": attempt_index + 1,
                "scale_before_backward": scale_before_backward,
                "scale_after_update": scale_after_update,
                "optimizer_step_skipped": False,
                "nonfinite_gradient_parameters": [],
                "losses": scalar_losses,
                "total_loss": float(total_loss.detach().float().item()),
                "forward_seconds": current_forward_seconds,
                "backward_seconds": current_backward_seconds,
                "scaler_step_seconds": optimizer_step_seconds,
            }
        )
        successful_training_step = True
        break

    if not successful_training_step:
        raise RuntimeError("No successful optimizer step was completed.")

    trainable_after = trainable_parameter.detach().cpu()
    backbone_after = backbone_parameter.detach().cpu()
    trainable_max_abs_change = float(
        (trainable_after - trainable_before).abs().max().item()
    )
    backbone_max_abs_change = float(
        (backbone_after - backbone_before).abs().max().item()
    )
    if trainable_max_abs_change <= 0:
        raise ValueError(
            "The optimizer step did not change the representative trainable "
            "predictor parameter."
        )
    if backbone_max_abs_change != 0:
        raise ValueError(
            "The frozen backbone changed during the optimizer step."
        )

    training_peak_memory = capture_cuda_memory(device)
    scalar_total_loss = float(total_loss.detach().float().item())

    # Release the training graph before measuring inference memory separately.
    optimizer.zero_grad(set_to_none=True)
    del loss_dict, total_loss
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    original_score_threshold = float(model.roi_heads.score_thresh)
    model.roi_heads.score_thresh = arguments.inference_score_threshold
    model.eval()

    inference_start = time.perf_counter()
    with torch.inference_mode():
        with autocast_context(device, amp_enabled):
            predictions = model(images)
    synchronize_cuda(device)
    inference_seconds = time.perf_counter() - inference_start
    inference_memory = capture_cuda_memory(device)

    prediction_report = validate_predictions(
        predictions,
        image_height=sample_height,
        image_width=sample_width,
    )

    model.roi_heads.score_thresh = original_score_threshold

    report = {
        "runtime": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "amp_enabled": amp_enabled,
            "amp_dtype": "torch.float16" if amp_enabled else None,
            "amp_initial_scale": arguments.amp_initial_scale,
            "max_amp_retries": arguments.max_amp_retries,
        },
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "configuration": DATASET_CONFIGURATION,
            "revision": DATASET_REVISION,
            "resolved_cache_dir": str(resolved_cache_dir),
        },
        "sample": {
            "split": "train",
            "selected_loader_batch_index": selected_batch_index,
            "image_id": sample_image_id,
            "image_shape": [int(value) for value in images_cpu[0].shape],
            "image_dtype_before_device_move": str(images_cpu[0].dtype),
            "image_device_after_move": str(images[0].device),
            "box_count": sample_box_count,
            "labels": sample_labels,
            "boxes_xyxy": [
                [float(value) for value in row]
                for row in targets_cpu[0]["boxes"].tolist()
            ],
        },
        "model": construction_metadata,
        "optimizer": {
            "name": "SGD",
            "learning_rate": arguments.learning_rate,
            "momentum": arguments.momentum,
            "weight_decay": arguments.weight_decay,
            "contains_only_trainable_parameters": True,
        },
        "training_forward": {
            "model_training": True,
            "backbone_training": False,
            "loss_keys": sorted(scalar_losses),
            "losses": scalar_losses,
            "total_loss": scalar_total_loss,
            "all_losses_finite": True,
            "forward_seconds": forward_seconds,
        },
        "backward": {
            **gradient_report,
            "representative_parameter": REPRESENTATIVE_TRAINABLE_PARAMETER,
            "representative_gradient_l2_norm": representative_gradient_norm,
            "backward_seconds": backward_seconds,
            "grad_scaler_scale_before_backward": scale_before_backward,
            "amp_attempt_count": len(amp_attempts),
            "amp_retry_count": sum(
                int(item["optimizer_step_skipped"]) for item in amp_attempts
            ),
            "amp_attempts": amp_attempts,
        },
        "optimizer_step": {
            "executed": True,
            "representative_trainable_parameter": (
                REPRESENTATIVE_TRAINABLE_PARAMETER
            ),
            "representative_trainable_max_abs_change": (
                trainable_max_abs_change
            ),
            "representative_frozen_parameter": (
                REPRESENTATIVE_BACKBONE_PARAMETER
            ),
            "representative_frozen_max_abs_change": backbone_max_abs_change,
            "frozen_backbone_unchanged": True,
            "optimizer_step_seconds": optimizer_step_seconds,
            "grad_scaler_scale_after_update": scale_after_update,
        },
        "inference": {
            "model_training": False,
            "temporary_score_threshold": (
                arguments.inference_score_threshold
            ),
            "original_score_threshold": original_score_threshold,
            "inference_seconds": inference_seconds,
            **prediction_report,
        },
        "cuda_memory": {
            "after_batch_move": memory_after_batch_move,
            "after_training_forward": memory_after_forward,
            "after_backward": memory_after_backward,
            "after_optimizer_step": memory_after_optimizer_step,
            "training_peak": training_peak_memory,
            "inference_peak": inference_memory,
        },
        "checks": {
            "real_dataset_batch_used": True,
            "training_forward_executed": True,
            "expected_four_losses_returned": True,
            "all_losses_finite": True,
            "backward_executed": True,
            "trainable_gradients_finite": True,
            "frozen_parameters_received_no_gradients": True,
            "optimizer_step_executed": True,
            "trainable_parameter_changed": True,
            "frozen_backbone_unchanged": True,
            "inference_executed": True,
            "prediction_contract_valid": True,
            "all_checks_passed": True,
        },
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    runtime = report["runtime"]
    sample = report["sample"]
    training = report["training_forward"]
    backward = report["backward"]
    step = report["optimizer_step"]
    inference = report["inference"]
    memory = report["cuda_memory"]

    print("\n=== Exercise 3.3 - Step 10: Faster R-CNN smoke test ===")
    print(f"Device: {runtime['device']}")
    print(f"AMP enabled: {runtime['amp_enabled']}")
    if memory["after_batch_move"]["available"]:
        print(
            "CUDA device: "
            f"{memory['after_batch_move']['device_name']}"
        )

    print("\nSelected real training sample:")
    print(f"  loader batch index: {sample['selected_loader_batch_index']}")
    print(f"  image_id: {sample['image_id']}")
    print(f"  image shape: {sample['image_shape']}")
    print(f"  boxes: {sample['box_count']}")
    print(f"  labels: {sample['labels']}")

    print("\nTraining forward losses:")
    for name, value in training["losses"].items():
        print(f"  - {name}: {value:.6f}")
    print(f"  total_loss: {training['total_loss']:.6f}")
    print(f"  all finite: {training['all_losses_finite']}")

    print("\nBackward:")
    print(
        "  trainable tensors with gradient: "
        f"{backward['trainable_parameter_tensors_with_gradient']}"
    )
    print(
        "  trainable tensors without gradient: "
        f"{backward['trainable_parameter_tensors_without_gradient']}"
    )
    print(
        "  frozen tensors with gradient: "
        f"{backward['frozen_parameter_tensors_with_gradient']}"
    )
    print(f"  global gradient L2 norm: {backward['global_l2_norm']:.6f}")
    print(
        "  representative predictor gradient norm: "
        f"{backward['representative_gradient_l2_norm']:.6f}"
    )
    print(
        "  AMP attempts/retries: "
        f"{backward['amp_attempt_count']} / "
        f"{backward['amp_retry_count']}"
    )
    for attempt in backward["amp_attempts"]:
        if attempt["optimizer_step_skipped"]:
            print(
                "    - skipped attempt "
                f"{attempt['attempt']}: scale "
                f"{attempt['scale_before_backward']:.1f} -> "
                f"{attempt['scale_after_update']:.1f}, non-finite: "
                f"{attempt['nonfinite_gradient_parameters'][:3]}"
            )

    print("\nOptimizer step:")
    print(
        "  trainable predictor max |delta|: "
        f"{step['representative_trainable_max_abs_change']:.8e}"
    )
    print(
        "  frozen backbone max |delta|: "
        f"{step['representative_frozen_max_abs_change']:.8e}"
    )
    print(f"  frozen backbone unchanged: {step['frozen_backbone_unchanged']}")

    print("\nInference:")
    print(f"  prediction keys: {inference['prediction_keys']}")
    print(f"  detections: {inference['detection_count']}")
    print(
        "  boxes/labels/scores shapes: "
        f"{inference['boxes_shape']} / "
        f"{inference['labels_shape']} / "
        f"{inference['scores_shape']}"
    )
    print(
        "  score range: "
        f"[{inference['score_min']:.6f}, {inference['score_max']:.6f}]"
    )
    print(f"  observed labels: {inference['observed_labels']}")

    if memory["training_peak"]["available"]:
        print("\nCUDA peak memory:")
        print(
            "  training allocated peak: "
            f"{format_megabytes(memory['training_peak']['peak_allocated_bytes'])}"
        )
        print(
            "  training reserved peak:  "
            f"{format_megabytes(memory['training_peak']['peak_reserved_bytes'])}"
        )
        print(
            "  inference allocated peak: "
            f"{format_megabytes(memory['inference_peak']['peak_allocated_bytes'])}"
        )
        print(
            "  inference reserved peak:  "
            f"{format_megabytes(memory['inference_peak']['peak_reserved_bytes'])}"
        )

    print("\nAll smoke-test checks: PASSED")


def main() -> None:
    arguments = parse_arguments()
    try:
        report = run_smoke_test(arguments)
    except torch.OutOfMemoryError as error:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA out of memory during the one-image Faster R-CNN smoke test. "
            "The batch size is already 1. Run the structural check on CPU with "
            "--device cpu, or execute the CUDA smoke test on the RTX 5090 "
            "server before changing the scientific image-size configuration."
        ) from error

    output_path = arguments.output_path.expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_report(report)
    print(f"\nSmoke-test report saved to: {output_path}")


if __name__ == "__main__":
    main()
