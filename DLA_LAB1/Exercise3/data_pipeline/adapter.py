"""PyTorch dataset adapter for German traffic-sign object detection.

This module converts the Hugging Face dataset representation into the target
structure expected by Torchvision detection models.  It deliberately performs
no resize or augmentation: synchronized transforms are introduced in Step 6.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

from Exercise3.data_pipeline.loading import (
    ANNOTATIONS_FIELD,
    IMAGE_FIELD,
    extract_class_names,
    normalize_objects,
    validate_object_field_lengths,
)
from Exercise3.data_pipeline.taxonomy import (
    NUM_DETECTOR_CLASSES,
    build_source_to_detector_label,
)


DetectionTarget = dict[str, Any]
DetectionTransform = Callable[
    [tv_tensors.Image, DetectionTarget],
    tuple[torch.Tensor, DetectionTarget],
]


@dataclass(frozen=True)
class SampleAdapterDiagnostics:
    """Counts describing one raw-to-PyTorch sample conversion."""

    sample_index: int
    image_id: int
    source_object_count: int
    output_object_count: int
    exact_duplicates_removed: int
    image_width: int
    image_height: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible representation."""
        return asdict(self)


class GermanTrafficSignDetectionDataset(Dataset):
    """Adapt one Hugging Face split to Torchvision's detection interface.

    Each item is returned as ``(image, target)`` where ``image`` is a
    ``tv_tensors.Image`` with shape ``[3, H, W]`` and ``target`` contains:

    - ``boxes``: float32 ``BoundingBoxes[N, 4]`` in XYXY format;
    - ``labels``: int64 detector labels in the range 1..43;
    - ``image_id``: unique integer identifier within the split;
    - ``area``: float32 box areas computed after conversion;
    - ``iscrowd``: int64 zeros because this dataset has no crowd annotations.

    Label 0 is reserved for background and is never assigned to ground truth.
    """

    def __init__(
        self,
        split_dataset: Any,
        transforms: DetectionTransform | None = None,
        *,
        remove_exact_duplicates: bool = True,
        strict: bool = True,
    ) -> None:
        self.split_dataset = split_dataset
        self.transforms = transforms
        self.remove_exact_duplicates = remove_exact_duplicates
        self.strict = strict

        self.class_names = extract_class_names(split_dataset)
        if not self.class_names:
            raise ValueError(
                "The Hugging Face split does not expose detection class names."
            )

        self.source_to_detector_label = build_source_to_detector_label(
            self.class_names
        )

    def __len__(self) -> int:
        return len(self.split_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, DetectionTarget]:
        image, target, _ = self.get_item_with_diagnostics(index)
        return image, target

    def get_item_with_diagnostics(
        self,
        index: int,
    ) -> tuple[torch.Tensor, DetectionTarget, SampleAdapterDiagnostics]:
        """Return an adapted item plus conversion diagnostics for validation."""
        if not 0 <= index < len(self):
            raise IndexError(
                f"Sample index {index} is outside a split of size {len(self)}."
            )

        sample = self.split_dataset[index]
        image, target, diagnostics = self._adapt_sample(sample, index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, diagnostics

    def _adapt_sample(
        self,
        sample: Mapping[str, Any],
        index: int,
    ) -> tuple[tv_tensors.Image, DetectionTarget, SampleAdapterDiagnostics]:
        missing = {
            field
            for field in (IMAGE_FIELD, ANNOTATIONS_FIELD)
            if field not in sample
        }
        if missing:
            raise KeyError(
                f"Sample {index} is missing required field(s): {sorted(missing)}."
            )

        raw_image = sample[IMAGE_FIELD]
        if not hasattr(raw_image, "size"):
            raise TypeError(
                f"Sample {index} image does not expose a PIL-like size."
            )

        if getattr(raw_image, "mode", None) != "RGB":
            raw_image = raw_image.convert("RGB")

        image_width, image_height = map(int, raw_image.size)
        self._validate_declared_image_size(
            sample=sample,
            index=index,
            actual_width=image_width,
            actual_height=image_height,
        )

        image = tv_tensors.Image(raw_image)
        objects = normalize_objects(sample[ANNOTATIONS_FIELD])
        source_object_count = validate_object_field_lengths(objects)

        raw_boxes = objects.get("bbox", [])
        raw_categories = objects.get("category", [])
        if len(raw_boxes) != source_object_count:
            raise ValueError(
                f"Sample {index}: bbox count {len(raw_boxes)} differs from "
                f"object count {source_object_count}."
            )
        if len(raw_categories) != source_object_count:
            raise ValueError(
                f"Sample {index}: category count {len(raw_categories)} differs "
                f"from object count {source_object_count}."
            )

        boxes_xyxy: list[list[float]] = []
        labels: list[int] = []
        seen_exact_annotations: set[tuple[int, float, float, float, float]] = set()
        duplicates_removed = 0

        for object_index, (raw_box, raw_category) in enumerate(
            zip(raw_boxes, raw_categories, strict=True)
        ):
            source_category_id = int(raw_category)
            box_xywh = self._parse_and_validate_box(
                raw_box=raw_box,
                sample_index=index,
                object_index=object_index,
                image_width=image_width,
                image_height=image_height,
            )

            if source_category_id not in self.source_to_detector_label:
                raise ValueError(
                    f"Sample {index}, object {object_index}: unknown source "
                    f"category ID {source_category_id}."
                )

            duplicate_key = (source_category_id, *box_xywh)
            if self.remove_exact_duplicates and duplicate_key in seen_exact_annotations:
                duplicates_removed += 1
                continue
            seen_exact_annotations.add(duplicate_key)

            x_min, y_min, box_width, box_height = box_xywh
            boxes_xyxy.append(
                [
                    x_min,
                    y_min,
                    x_min + box_width,
                    y_min + box_height,
                ]
            )
            labels.append(self.source_to_detector_label[source_category_id])

        boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4)
        boxes = tv_tensors.BoundingBoxes(
            boxes_tensor,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(image_height, image_width),
        )
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        area = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        ).to(dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes_tensor),), dtype=torch.int64)

        image_id_value = sample.get("image_id", index)
        image_id = index if image_id_value is None else int(image_id_value)

        target: DetectionTarget = {
            "boxes": boxes,
            "labels": labels_tensor,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        diagnostics = SampleAdapterDiagnostics(
            sample_index=index,
            image_id=image_id,
            source_object_count=source_object_count,
            output_object_count=len(boxes_tensor),
            exact_duplicates_removed=duplicates_removed,
            image_width=image_width,
            image_height=image_height,
        )
        return image, target, diagnostics

    def _validate_declared_image_size(
        self,
        sample: Mapping[str, Any],
        index: int,
        actual_width: int,
        actual_height: int,
    ) -> None:
        declared_width = sample.get("width")
        declared_height = sample.get("height")

        if declared_width is not None and int(declared_width) != actual_width:
            raise ValueError(
                f"Sample {index}: declared width {declared_width} differs from "
                f"decoded width {actual_width}."
            )
        if declared_height is not None and int(declared_height) != actual_height:
            raise ValueError(
                f"Sample {index}: declared height {declared_height} differs "
                f"from decoded height {actual_height}."
            )

    def _parse_and_validate_box(
        self,
        raw_box: Any,
        sample_index: int,
        object_index: int,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float]:
        if not isinstance(raw_box, Sequence) or isinstance(
            raw_box,
            (str, bytes, bytearray),
        ):
            raise TypeError(
                f"Sample {sample_index}, object {object_index}: bbox must be a "
                "sequence of four numbers."
            )
        if len(raw_box) != 4:
            raise ValueError(
                f"Sample {sample_index}, object {object_index}: expected bbox "
                f"length 4, found {len(raw_box)}."
            )

        x_min, y_min, box_width, box_height = map(float, raw_box)
        values = (x_min, y_min, box_width, box_height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"Sample {sample_index}, object {object_index}: bbox contains "
                f"non-finite values {values}."
            )

        x_max = x_min + box_width
        y_max = y_min + box_height
        valid = (
            box_width > 0
            and box_height > 0
            and 0 <= x_min < x_max <= image_width
            and 0 <= y_min < y_max <= image_height
        )
        if not valid and self.strict:
            raise ValueError(
                f"Sample {sample_index}, object {object_index}: invalid xywh "
                f"bbox {values} for image {image_width}x{image_height}."
            )

        return values


def validate_detection_target(
    image: torch.Tensor,
    target: Mapping[str, Any],
) -> None:
    """Raise a descriptive error if one adapted sample violates the contract."""
    required_keys = {"boxes", "labels", "image_id", "area", "iscrowd"}
    missing_keys = required_keys - set(target)
    if missing_keys:
        raise KeyError(f"Detection target is missing keys: {sorted(missing_keys)}.")

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Image must be a Tensor, found {type(image).__name__}.")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Image must have shape [3, H, W], found {tuple(image.shape)}.")

    boxes = target["boxes"]
    labels = target["labels"]
    area = target["area"]
    iscrowd = target["iscrowd"]

    if not isinstance(boxes, torch.Tensor):
        raise TypeError("target['boxes'] must be a Tensor.")
    if boxes.dtype != torch.float32:
        raise TypeError(f"Boxes must be float32, found {boxes.dtype}.")
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"Boxes must have shape [N, 4], found {tuple(boxes.shape)}.")

    object_count = boxes.shape[0]
    expected_vector_shape = (object_count,)
    if labels.dtype != torch.int64 or tuple(labels.shape) != expected_vector_shape:
        raise TypeError(
            "Labels must be int64 with shape [N], found "
            f"dtype={labels.dtype}, shape={tuple(labels.shape)}."
        )
    if area.dtype != torch.float32 or tuple(area.shape) != expected_vector_shape:
        raise TypeError(
            "Area must be float32 with shape [N], found "
            f"dtype={area.dtype}, shape={tuple(area.shape)}."
        )
    if iscrowd.dtype != torch.int64 or tuple(iscrowd.shape) != expected_vector_shape:
        raise TypeError(
            "iscrowd must be int64 with shape [N], found "
            f"dtype={iscrowd.dtype}, shape={tuple(iscrowd.shape)}."
        )

    if not torch.isfinite(boxes).all():
        raise ValueError("Boxes contain NaN or Inf values.")
    if not torch.isfinite(area).all():
        raise ValueError("Areas contain NaN or Inf values.")

    if object_count == 0:
        return

    height, width = image.shape[-2:]
    x_min, y_min, x_max, y_max = boxes.unbind(dim=1)
    if not (
        (x_min >= 0).all()
        and (y_min >= 0).all()
        and (x_max <= width).all()
        and (y_max <= height).all()
        and (x_max > x_min).all()
        and (y_max > y_min).all()
    ):
        raise ValueError("Boxes are degenerate or outside the image boundaries.")

    if not ((labels >= 1) & (labels < NUM_DETECTOR_CLASSES)).all():
        raise ValueError(
            f"Foreground labels must be in [1, {NUM_DETECTOR_CLASSES - 1}]."
        )
    if not (area > 0).all():
        raise ValueError("Every retained box must have strictly positive area.")
    if not (iscrowd == 0).all():
        raise ValueError("This dataset should contain only iscrowd=0 instances.")
