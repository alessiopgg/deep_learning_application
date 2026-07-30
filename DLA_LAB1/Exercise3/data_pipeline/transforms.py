"""Step 6 transforms for the traffic-sign detection dataset.

This module introduces the first minimal, synchronized transform pipeline for
Exercise 3.3. At this stage we intentionally avoid data augmentation and avoid
manual resizing. The goal is simply to convert images from uint8 [0, 255] to
float32 [0, 1] while preserving tv_tensors metadata and leaving bounding boxes
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes, Image


@dataclass(frozen=True)
class TransformDescription:
    """Human-readable summary of one transform pipeline."""

    name: str
    split: str
    converts_image_dtype: str
    converts_image_range: str
    applies_geometric_augmentation: bool
    applies_photometric_augmentation: bool
    resizes_image: bool
    normalizes_with_imagenet_statistics: bool
    preserves_box_geometry: bool


class IdentityTargetTransform(torch.nn.Module):
    """Explicit no-op kept for readability and future extension."""

    def forward(self, image: Image, target: dict) -> tuple[Image, dict]:
        return image, target


class DetectionFloatTransform(torch.nn.Module):
    """Convert a tv_tensors.Image from uint8 [0,255] to float32 [0,1].

    The transform is intentionally minimal. Bounding boxes and target tensors are
    passed through unchanged. The image remains a tv_tensors.Image so that later
    transforms from torchvision v2 can still operate on it together with the
    associated BoundingBoxes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.image_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
        ])
        self.target_transform = IdentityTargetTransform()

    def forward(self, image: Image, target: dict) -> tuple[Image, dict]:
        transformed_image = self.image_transform(image)
        transformed_image, transformed_target = self.target_transform(
            transformed_image,
            target,
        )
        return transformed_image, transformed_target


class DetectionTransformPipeline(torch.nn.Module):
    """Wrapper exposing train/validation/test transform calls.

    At Step 6 all splits use the same minimal pipeline. The wrapper still keeps a
    split-aware interface because later steps may introduce different training and
    evaluation transforms without changing the dataset integration API.
    """

    def __init__(self) -> None:
        super().__init__()
        self.train_transform = DetectionFloatTransform()
        self.validation_transform = DetectionFloatTransform()
        self.test_transform = DetectionFloatTransform()

    def forward(
        self,
        image: Image,
        target: dict,
        split: str,
    ) -> tuple[Image, dict]:
        if split == "train":
            return self.train_transform(image, target)
        if split == "validation":
            return self.validation_transform(image, target)
        if split == "test":
            return self.test_transform(image, target)
        raise ValueError(
            f"Unknown split '{split}'. Expected one of: train, validation, test."
        )


class TransformedDetectionDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies synchronized transforms after adaptation."""

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        split: str,
        transform_pipeline: DetectionTransformPipeline | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.split = split
        self.transform_pipeline = transform_pipeline or DetectionTransformPipeline()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, target = self.base_dataset[index]
        return self.transform_pipeline(image=image, target=target, split=self.split)


def build_detection_transform_pipeline() -> DetectionTransformPipeline:
    """Create the minimal split-aware transform pipeline for Step 6."""
    return DetectionTransformPipeline()


def describe_transform_pipeline() -> list[TransformDescription]:
    """Return a stable textual description used by the validation script."""
    return [
        TransformDescription(
            name="DetectionFloatTransform",
            split=split,
            converts_image_dtype="torch.uint8 -> torch.float32",
            converts_image_range="[0, 255] -> [0.0, 1.0]",
            applies_geometric_augmentation=False,
            applies_photometric_augmentation=False,
            resizes_image=False,
            normalizes_with_imagenet_statistics=False,
            preserves_box_geometry=True,
        )
        for split in ("train", "validation", "test")
    ]


def image_value_range(image: torch.Tensor) -> tuple[float, float]:
    """Return the minimum and maximum scalar values of the transformed image."""
    if image.numel() == 0:
        return 0.0, 0.0
    return float(image.min().item()), float(image.max().item())


def boxes_to_list(boxes: BoundingBoxes) -> list[list[float]]:
    """Convert BoundingBoxes to a JSON-serializable nested list."""
    return [[float(value) for value in row] for row in boxes.tolist()]
