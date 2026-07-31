"""DataLoader construction for Exercise 3 object detection.

Detection samples cannot use PyTorch's default tensor stacking because every
image may contain a different number of objects.  The custom ``collate_fn``
therefore returns two parallel Python lists:

    images:  list[Tensor]
    targets: list[dict[str, Any]]

The detector will receive these lists directly in the later training steps.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from Exercise3.data_pipeline.adapter import (
    DetectionTarget,
    GermanTrafficSignDetectionDataset,
)
from Exercise3.data_pipeline.transforms import (
    DetectionTransformPipeline,
    TransformedDetectionDataset,
    build_detection_transform_pipeline,
)


DEFAULT_SEED = 42
DEFAULT_TRAIN_BATCH_SIZE = 2
DEFAULT_EVALUATION_BATCH_SIZE = 1
DEFAULT_NUM_WORKERS = 0


DetectionBatch = tuple[list[torch.Tensor], list[DetectionTarget]]


@dataclass(frozen=True)
class DetectionLoaderSettings:
    """Serializable configuration shared by the three DataLoaders."""

    train_batch_size: int
    evaluation_batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool
    seed: int
    train_shuffle: bool
    validation_shuffle: bool
    test_shuffle: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectionDatasetBundle:
    """Transformed datasets for the official train/validation/test splits."""

    train: TransformedDetectionDataset
    validation: TransformedDetectionDataset
    test: TransformedDetectionDataset

    def as_dict(self) -> dict[str, TransformedDetectionDataset]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True)
class DetectionDataLoaderBundle:
    """DataLoaders and the exact settings used to construct them."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader
    datasets: DetectionDatasetBundle
    settings: DetectionLoaderSettings

    def loaders_as_dict(self) -> dict[str, DataLoader]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, DetectionTarget]],
) -> DetectionBatch:
    """Keep variable-size images and targets as parallel lists.

    The default PyTorch collation attempts to stack tensors and dictionary
    fields. That is not appropriate for object detection because each target can
    contain a different number of boxes. This top-level function is deliberately
    picklable so it also works with Windows DataLoader workers.
    """
    if not batch:
        raise ValueError("Cannot collate an empty detection batch.")

    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def seed_detection_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the worker-specific PyTorch seed."""
    del worker_id  # The worker-specific value is already encoded in initial_seed.
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_detection_datasets(
    dataset_dict: Any,
    transform_pipeline: DetectionTransformPipeline | None = None,
) -> DetectionDatasetBundle:
    """Adapt and transform all official dataset splits."""
    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits.difference(dataset_dict.keys())
    if missing_splits:
        raise KeyError(
            "The detection dataset is missing required split(s): "
            f"{sorted(missing_splits)}."
        )

    pipeline = transform_pipeline or build_detection_transform_pipeline()

    transformed: dict[str, TransformedDetectionDataset] = {}
    for split_name in ("train", "validation", "test"):
        adapted_dataset = GermanTrafficSignDetectionDataset(
            dataset_dict[split_name]
        )
        transformed[split_name] = TransformedDetectionDataset(
            base_dataset=adapted_dataset,
            split=split_name,
            transform_pipeline=pipeline,
        )

    return DetectionDatasetBundle(
        train=transformed["train"],
        validation=transformed["validation"],
        test=transformed["test"],
    )


def _build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int,
) -> DataLoader:
    """Construct one DataLoader using the shared detection contract."""
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": detection_collate_fn,
        "pin_memory": pin_memory,
        "drop_last": False,
        "worker_init_fn": seed_detection_worker,
        "generator": _make_generator(seed),
        "persistent_workers": persistent_workers,
    }

    return DataLoader(**arguments)


def build_detection_dataloaders(
    dataset_dict: Any,
    *,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    evaluation_batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool | None = None,
    persistent_workers: bool = False,
    seed: int = DEFAULT_SEED,
    transform_pipeline: DetectionTransformPipeline | None = None,
) -> DetectionDataLoaderBundle:
    """Build reproducible train, validation and test detection DataLoaders."""
    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be greater than zero.")
    if evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be greater than zero.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if persistent_workers and num_workers == 0:
        raise ValueError(
            "persistent_workers=True requires num_workers greater than zero."
        )

    resolved_pin_memory = (
        torch.cuda.is_available() if pin_memory is None else pin_memory
    )
    datasets = build_detection_datasets(
        dataset_dict=dataset_dict,
        transform_pipeline=transform_pipeline,
    )

    settings = DetectionLoaderSettings(
        train_batch_size=train_batch_size,
        evaluation_batch_size=evaluation_batch_size,
        num_workers=num_workers,
        pin_memory=resolved_pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
        seed=seed,
        train_shuffle=True,
        validation_shuffle=False,
        test_shuffle=False,
    )

    train_loader = _build_loader(
        datasets.train,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=resolved_pin_memory,
        persistent_workers=persistent_workers,
        seed=seed,
    )
    validation_loader = _build_loader(
        datasets.validation,
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_pin_memory,
        persistent_workers=persistent_workers,
        seed=seed + 1,
    )
    test_loader = _build_loader(
        datasets.test,
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_pin_memory,
        persistent_workers=persistent_workers,
        seed=seed + 2,
    )

    return DetectionDataLoaderBundle(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        datasets=datasets,
        settings=settings,
    )
