"""Qualitative ground-truth/prediction visualizations for Exercise 3."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image as PILImage
from PIL import ImageDraw
from torchvision.utils import draw_bounding_boxes

from Exercise3.data_pipeline.taxonomy import build_detector_label_to_name
from Exercise3.visualization.ground_truth import image_to_uint8, tensor_to_pil


def select_visualization_indices(
    rows: list[dict[str, Any]],
    *,
    sample_count: int,
    seed: int,
) -> list[tuple[int, str]]:
    """Select diverse, reproducible qualitative examples."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if not rows:
        return []

    selected: list[tuple[int, str]] = []
    used: set[int] = set()

    def add(row: dict[str, Any] | None, reason: str) -> None:
        if row is None or len(selected) >= sample_count:
            return
        index = int(row["dataset_index"])
        if index not in used:
            used.add(index)
            selected.append((index, reason))

    empty_fp = [row for row in rows if row.get("false_positive_on_empty_target")]
    add(
        max(empty_fp, key=lambda row: int(row.get("false_positives", 0)), default=None),
        "empty-fp",
    )
    add(
        max(rows, key=lambda row: int(row.get("false_negatives", 0))),
        "most-fn",
    )
    add(
        max(rows, key=lambda row: int(row.get("false_positives", 0))),
        "most-fp",
    )
    add(
        max(rows, key=lambda row: int(row.get("true_positives", 0))),
        "most-tp",
    )

    remaining = [row for row in rows if int(row["dataset_index"]) not in used]
    random.Random(seed).shuffle(remaining)
    for row in remaining:
        add(row, "random")
        if len(selected) >= min(sample_count, len(rows)):
            break
    return selected


def _font_arguments(font_path: str | None, font_size: int) -> dict[str, Any]:
    if font_size <= 0:
        raise ValueError("font_size must be positive.")
    if font_path is None:
        return {}
    path = Path(font_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Font not found: {path}")
    return {"font": str(path), "font_size": font_size}


def _draw_target(
    image: torch.Tensor,
    target: dict[str, Any],
    *,
    box_width: int,
    font_path: str | None,
    font_size: int,
) -> torch.Tensor:
    boxes = target["boxes"].detach().cpu().to(torch.float32)
    if boxes.shape[0] == 0:
        return image.clone()
    names = build_detector_label_to_name()
    labels = [
        f"GT {int(label)}: {names[int(label)]}"
        for label in target["labels"].detach().cpu().tolist()
    ]
    return draw_bounding_boxes(
        image=image,
        boxes=boxes,
        labels=labels,
        colors="red",
        width=box_width,
        **_font_arguments(font_path, font_size),
    )


def _draw_prediction(
    image: torch.Tensor,
    prediction: dict[str, torch.Tensor],
    *,
    score_threshold: float,
    box_width: int,
    font_path: str | None,
    font_size: int,
) -> torch.Tensor:
    keep = prediction["scores"] >= score_threshold
    boxes = prediction["boxes"][keep].detach().cpu().to(torch.float32)
    if boxes.shape[0] == 0:
        return image.clone()
    labels_tensor = prediction["labels"][keep].detach().cpu().to(torch.int64)
    scores = prediction["scores"][keep].detach().cpu().to(torch.float32)
    names = build_detector_label_to_name()
    labels = [
        f"P {int(label)}: {names[int(label)]} {float(score):.2f}"
        for label, score in zip(labels_tensor.tolist(), scores.tolist(), strict=True)
    ]
    return draw_bounding_boxes(
        image=image,
        boxes=boxes,
        labels=labels,
        colors="green",
        width=box_width,
        **_font_arguments(font_path, font_size),
    )


def save_comparison(
    *,
    image: torch.Tensor,
    target: dict[str, Any],
    prediction: dict[str, torch.Tensor],
    output_path: str | Path,
    score_threshold: float,
    box_width: int,
    font_path: str | None,
    font_size: int,
    title: str,
) -> Path:
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be within [0,1].")
    if box_width <= 0:
        raise ValueError("box_width must be positive.")

    uint8_image = image_to_uint8(image)
    ground_truth = _draw_target(
        uint8_image,
        target,
        box_width=box_width,
        font_path=font_path,
        font_size=font_size,
    )
    predicted = _draw_prediction(
        uint8_image,
        prediction,
        score_threshold=score_threshold,
        box_width=box_width,
        font_path=font_path,
        font_size=font_size,
    )
    left = tensor_to_pil(ground_truth)
    right = tensor_to_pil(predicted)
    header_height = 52
    gap = 8
    canvas = PILImage.new(
        "RGB",
        (left.width + right.width + gap, left.height + header_height),
        color="white",
    )
    canvas.paste(left, (0, header_height))
    canvas.paste(right, (left.width + gap, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill="black")
    draw.text((8, 28), "Ground truth", fill="red")
    draw.text((left.width + gap + 8, 28), "Predictions", fill="green")

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    return path
