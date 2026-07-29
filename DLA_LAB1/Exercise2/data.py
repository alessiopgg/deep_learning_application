"""GTSRB loading, stratified splitting and DataLoader construction."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import GTSRB


EXERCISE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetInfo:
    """Sizes and execution settings associated with the dataset splits."""

    data_dir: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    original_train_samples: int
    train_samples: int
    validation_samples: int
    test_samples: int
    train_batches: int
    validation_batches: int
    test_batches: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class DataLoaders:
    """DataLoader collection and metadata for one experiment."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader
    info: DatasetInfo


def resolve_exercise_path(path_value: str | Path) -> Path:
    """Resolve relative paths from the Exercise2 directory."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = EXERCISE_DIR / path
    return path.resolve()


def load_gtsrb(data_dir: Path, transform) -> tuple[GTSRB, GTSRB]:
    """Load the official GTSRB training and test splits."""
    train_dataset = GTSRB(
        root=str(data_dir),
        split="train",
        download=True,
        transform=transform,
    )
    test_dataset = GTSRB(
        root=str(data_dir),
        split="test",
        download=True,
        transform=transform,
    )
    return train_dataset, test_dataset


def extract_labels(dataset: Dataset) -> np.ndarray:
    """Extract class labels without loading and transforming every image."""
    samples = getattr(dataset, "_samples", None)
    if samples is not None:
        return np.asarray(
            [int(label) for _, label in samples],
            dtype=np.int64,
        )

    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return np.asarray(targets, dtype=np.int64)

    raise AttributeError(
        "The dataset exposes neither '_samples' nor 'targets'. "
        "A dataset-specific label extractor is required."
    )


def stratified_split_indices(
    labels: Sequence[int] | np.ndarray,
    validation_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create reproducible, stratified train and validation indices."""
    labels_array = np.asarray(labels, dtype=np.int64)
    all_indices = np.arange(labels_array.size)

    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=validation_size,
        random_state=seed,
        stratify=labels_array,
    )

    return train_indices, validation_indices


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python random state inside each DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    config: DictConfig,
    transform,
    device: torch.device,
) -> DataLoaders:
    """Build train, validation and test DataLoaders from the configuration."""
    data_dir = resolve_exercise_path(config.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = load_gtsrb(
        data_dir=data_dir,
        transform=transform,
    )

    labels = extract_labels(train_dataset)
    train_indices, validation_indices = stratified_split_indices(
        labels=labels,
        validation_size=config.data.validation_size,
        seed=config.experiment.seed,
    )

    train_subset = Subset(train_dataset, train_indices.tolist())
    validation_subset = Subset(
        train_dataset,
        validation_indices.tolist(),
    )

    pin_memory = bool(
        config.data.pin_memory and device.type == "cuda"
    )

    generator = torch.Generator()
    generator.manual_seed(config.experiment.seed)

    common_loader_arguments = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
    }

    train_loader = DataLoader(
        train_subset,
        shuffle=True,
        generator=generator,
        **common_loader_arguments,
    )
    validation_loader = DataLoader(
        validation_subset,
        shuffle=False,
        **common_loader_arguments,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_arguments,
    )

    info = DatasetInfo(
        data_dir=str(data_dir),
        batch_size=int(config.data.batch_size),
        num_workers=int(config.data.num_workers),
        pin_memory=pin_memory,
        original_train_samples=len(train_dataset),
        train_samples=len(train_subset),
        validation_samples=len(validation_subset),
        test_samples=len(test_dataset),
        train_batches=len(train_loader),
        validation_batches=len(validation_loader),
        test_batches=len(test_loader),
    )

    return DataLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        info=info,
    )


def print_data_summary(data_loaders: DataLoaders) -> None:
    """Print the relevant properties of the prepared dataset splits."""
    info = data_loaders.info

    print("\n=== Exercise 2 data preparation ===")
    print(f"Data directory: {info.data_dir}")
    print(f"Batch size: {info.batch_size}")
    print(f"DataLoader workers: {info.num_workers}")
    print(f"Pinned memory: {info.pin_memory}")
    print(f"Original training images: {info.original_train_samples}")
    print(f"Training subset images: {info.train_samples}")
    print(f"Validation subset images: {info.validation_samples}")
    print(f"Test images: {info.test_samples}")
    print(f"Training batches: {info.train_batches}")
    print(f"Validation batches: {info.validation_batches}")
    print(f"Test batches: {info.test_batches}")
