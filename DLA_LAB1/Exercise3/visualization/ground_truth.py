"""Ground-truth visualization utilities for Exercise 3 object detection.

The functions in this module operate on the final PyTorch detection contract:
``image`` is a CHW tensor and ``target['boxes']`` contains absolute XYXY
coordinates.  Images are converted to uint8 only for drawing and saving; the
underlying dataset sample is never modified.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image as PILImage
from PIL import ImageDraw
from torchvision.utils import draw_bounding_boxes

from Exercise3.data_pipeline.taxonomy import build_detector_label_to_name


@dataclass(frozen=True)
class GroundTruthVisualizationRecord:
    """Serializable description of one saved visualization."""

    split: str
    sample_index: int
    image_id: int
    selection_reason: str
    object_count: int
    detector_labels: list[int]
    class_names: list[str]
    boxes_xyxy: list[list[float]]
    original_path: str
    transformed_path: str
    comparison_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _plain_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Drop a tv_tensor subclass while preserving values and device."""
    if type(tensor) is torch.Tensor:
        return tensor
    return tensor.as_subclass(torch.Tensor)


def image_to_uint8(image: torch.Tensor) -> torch.Tensor:
    """Convert a CHW RGB image to a plain uint8 tensor in [0, 255]."""
    tensor = _plain_tensor(image).detach().cpu()
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(
            f"Expected an RGB image with shape [3,H,W], found {tuple(tensor.shape)}."
        )

    if tensor.dtype == torch.uint8:
        return tensor.clone()

    if not tensor.is_floating_point():
        raise TypeError(
            "Visualization supports uint8 or floating-point images, found "
            f"{tensor.dtype}."
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("Image contains NaN or Inf values.")

    value_min = float(tensor.min().item())
    value_max = float(tensor.max().item())
    if value_min < 0.0 or value_max > 1.0:
        raise ValueError(
            "Floating-point images must be in [0,1] before visualization, "
            f"found [{value_min}, {value_max}]."
        )

    return tensor.mul(255.0).round().clamp(0, 255).to(torch.uint8)


def format_ground_truth_labels(
    target: Mapping[str, Any],
    label_to_name: Mapping[int, str] | None = None,
) -> tuple[list[int], list[str], list[str]]:
    """Return detector IDs, class names and display strings for one target."""
    if "labels" not in target:
        raise KeyError("The target does not contain a 'labels' field.")

    mapping = dict(label_to_name or build_detector_label_to_name())
    detector_labels = [int(value) for value in target["labels"].tolist()]

    class_names: list[str] = []
    display_labels: list[str] = []
    for detector_label in detector_labels:
        if detector_label not in mapping:
            raise KeyError(
                f"Detector label {detector_label} is missing from the taxonomy."
            )
        class_name = str(mapping[detector_label])
        class_names.append(class_name)
        display_labels.append(f"{detector_label}: {class_name}")

    return detector_labels, class_names, display_labels


def draw_ground_truth(
    image: torch.Tensor,
    target: Mapping[str, Any],
    *,
    box_width: int = 4,
    font_path: str | Path | None = None,
    font_size: int = 20,
) -> torch.Tensor:
    """Draw all ground-truth boxes and labels on one image.

    The returned tensor is a plain uint8 CHW tensor. Empty targets are valid and
    return an unmodified image.
    """
    if box_width <= 0:
        raise ValueError("box_width must be greater than zero.")
    if font_size <= 0:
        raise ValueError("font_size must be greater than zero.")
    if "boxes" not in target:
        raise KeyError("The target does not contain a 'boxes' field.")

    uint8_image = image_to_uint8(image)
    boxes = _plain_tensor(target["boxes"]).detach().cpu().to(torch.float32)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(
            f"Expected boxes with shape [N,4], found {tuple(boxes.shape)}."
        )

    detector_labels, _, display_labels = format_ground_truth_labels(target)
    if len(detector_labels) != boxes.shape[0]:
        raise ValueError(
            "The number of labels differs from the number of ground-truth boxes."
        )

    if boxes.shape[0] == 0:
        return uint8_image

    drawing_arguments: dict[str, Any] = {
        "image": uint8_image,
        "boxes": boxes,
        "labels": display_labels,
        "colors": "red",
        "width": box_width,
    }
    if font_path is not None:
        resolved_font_path = Path(font_path).expanduser().resolve()
        if not resolved_font_path.is_file():
            raise FileNotFoundError(
                f"The requested label font does not exist: {resolved_font_path}"
            )
        drawing_arguments["font"] = str(resolved_font_path)
        drawing_arguments["font_size"] = font_size

    return draw_bounding_boxes(**drawing_arguments)


def tensor_to_pil(image: torch.Tensor) -> PILImage.Image:
    """Convert a uint8 or [0,1] CHW tensor to a detached RGB PIL image."""
    uint8_image = image_to_uint8(image)
    array = uint8_image.permute(1, 2, 0).contiguous().numpy()
    return PILImage.fromarray(array, mode="RGB")


def save_tensor_png(image: torch.Tensor, output_path: str | Path) -> Path:
    """Save a CHW image tensor as PNG and return the resolved path."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(image).save(path, format="PNG")
    return path


def save_side_by_side_comparison(
    original_annotated: torch.Tensor,
    transformed_annotated: torch.Tensor,
    output_path: str | Path,
    *,
    left_title: str = "Originale + ground truth",
    right_title: str = "Trasformata + ground truth",
) -> Path:
    """Save two annotated images side by side without relying on GUI backends."""
    left = tensor_to_pil(original_annotated)
    right = tensor_to_pil(transformed_annotated)
    if left.size != right.size:
        raise ValueError(
            "The side-by-side comparison currently requires equal image sizes, "
            f"found {left.size} and {right.size}."
        )

    header_height = 34
    gap = 8
    canvas = PILImage.new(
        "RGB",
        (left.width + right.width + gap, left.height + header_height),
        color="white",
    )
    canvas.paste(left, (0, header_height))
    canvas.paste(right, (left.width + gap, header_height))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), left_title, fill="black")
    draw.text((left.width + gap + 8, 8), right_title, fill="black")

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    return path


def save_ground_truth_triplet(
    *,
    original_image: torch.Tensor,
    original_target: Mapping[str, Any],
    transformed_image: torch.Tensor,
    transformed_target: Mapping[str, Any],
    split: str,
    sample_index: int,
    selection_reason: str,
    output_dir: str | Path,
    box_width: int = 4,
    font_path: str | Path | None = None,
    font_size: int = 20,
) -> GroundTruthVisualizationRecord:
    """Save original, transformed and side-by-side ground-truth images."""
    output_directory = Path(output_dir).expanduser().resolve()
    split_directory = output_directory / "examples" / split
    split_directory.mkdir(parents=True, exist_ok=True)

    original_annotated = draw_ground_truth(
        original_image,
        original_target,
        box_width=box_width,
        font_path=font_path,
        font_size=font_size,
    )
    transformed_annotated = draw_ground_truth(
        transformed_image,
        transformed_target,
        box_width=box_width,
        font_path=font_path,
        font_size=font_size,
    )

    image_id = int(transformed_target["image_id"])
    stem = f"{split}_{sample_index:04d}_image-{image_id}_{selection_reason}"
    original_path = save_tensor_png(
        original_annotated,
        split_directory / f"{stem}_original.png",
    )
    transformed_path = save_tensor_png(
        transformed_annotated,
        split_directory / f"{stem}_transformed.png",
    )
    comparison_path = save_side_by_side_comparison(
        original_annotated,
        transformed_annotated,
        split_directory / f"{stem}_comparison.png",
    )

    detector_labels, class_names, _ = format_ground_truth_labels(
        transformed_target
    )
    boxes_xyxy = [
        [float(value) for value in row]
        for row in transformed_target["boxes"].tolist()
    ]

    return GroundTruthVisualizationRecord(
        split=split,
        sample_index=sample_index,
        image_id=image_id,
        selection_reason=selection_reason,
        object_count=len(detector_labels),
        detector_labels=detector_labels,
        class_names=class_names,
        boxes_xyxy=boxes_xyxy,
        original_path=str(original_path),
        transformed_path=str(transformed_path),
        comparison_path=str(comparison_path),
    )
