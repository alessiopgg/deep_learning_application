"""Minimal synchronized transforms for detection samples."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes, Image


@dataclass(frozen=True)
class TransformDescription:
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
    def forward(self, image: Image, target: dict) -> tuple[Image, dict]:
        return image, target


class DetectionFloatTransform(torch.nn.Module):
    """Convert uint8 [0,255] images to float32 [0,1]; targets are unchanged."""

    def __init__(self) -> None:
        super().__init__()
        self.to_float = v2.ToDtype(torch.float32, scale=True)

    def forward(self, image: Image, target: dict) -> tuple[Image, dict]:
        return self.to_float(image), target


class DetectionTransformPipeline(torch.nn.Module):
    """All official splits intentionally use the same baseline transform."""

    VALID_SPLITS = {"train", "validation", "test"}

    def __init__(self) -> None:
        super().__init__()
        self.transform = DetectionFloatTransform()

    def forward(
        self,
        image: Image,
        target: dict,
        split: str,
    ) -> tuple[Image, dict]:
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Unknown split {split!r}. Expected train, validation or test."
            )
        return self.transform(image, target)


class TransformedDetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        split: str,
        transform_pipeline: DetectionTransformPipeline | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.split = split
        self.transform_pipeline = (
            transform_pipeline or DetectionTransformPipeline()
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, target = self.base_dataset[index]
        return self.transform_pipeline(image, target, self.split)


def build_detection_transform_pipeline() -> DetectionTransformPipeline:
    return DetectionTransformPipeline()


def describe_transform_pipeline() -> list[TransformDescription]:
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
    if image.numel() == 0:
        return 0.0, 0.0
    return float(image.min()), float(image.max())


def boxes_to_list(boxes: BoundingBoxes) -> list[list[float]]:
    return [[float(value) for value in row] for row in boxes.tolist()]
