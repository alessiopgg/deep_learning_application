"""Detection datasets and reproducible DataLoaders."""

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
    train: TransformedDetectionDataset
    validation: TransformedDetectionDataset
    test: TransformedDetectionDataset

    def as_dict(self):
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True)
class DetectionDataLoaderBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    datasets: DetectionDatasetBundle
    settings: DetectionLoaderSettings

    def loaders_as_dict(self):
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, DetectionTarget]],
) -> DetectionBatch:
    if not batch:
        raise ValueError("Cannot collate an empty detection batch.")
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def seed_detection_worker(_: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_detection_worker,
        generator=generator,
        persistent_workers=persistent_workers,
    )


def build_detection_datasets(
    dataset_dict: Any,
    transform_pipeline: DetectionTransformPipeline | None = None,
) -> DetectionDatasetBundle:
    missing = {"train", "validation", "test"} - set(dataset_dict)
    if missing:
        raise KeyError(f"Missing required split(s): {sorted(missing)}.")

    pipeline = transform_pipeline or build_detection_transform_pipeline()
    datasets = {
        split: TransformedDetectionDataset(
            GermanTrafficSignDetectionDataset(dataset_dict[split]),
            split,
            pipeline,
        )
        for split in ("train", "validation", "test")
    }
    return DetectionDatasetBundle(**datasets)


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
    if train_batch_size <= 0 or evaluation_batch_size <= 0:
        raise ValueError("Batch sizes must be greater than zero.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0.")

    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    datasets = build_detection_datasets(dataset_dict, transform_pipeline)
    settings = DetectionLoaderSettings(
        train_batch_size=train_batch_size,
        evaluation_batch_size=evaluation_batch_size,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        persistent_workers=persistent_workers,
        drop_last=False,
        seed=seed,
        train_shuffle=True,
        validation_shuffle=False,
        test_shuffle=False,
    )

    common = dict(
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        persistent_workers=persistent_workers,
    )
    return DetectionDataLoaderBundle(
        train=_loader(
            datasets.train,
            batch_size=train_batch_size,
            shuffle=True,
            seed=seed,
            **common,
        ),
        validation=_loader(
            datasets.validation,
            batch_size=evaluation_batch_size,
            shuffle=False,
            seed=seed + 1,
            **common,
        ),
        test=_loader(
            datasets.test,
            batch_size=evaluation_batch_size,
            shuffle=False,
            seed=seed + 2,
            **common,
        ),
        datasets=datasets,
        settings=settings,
    )
