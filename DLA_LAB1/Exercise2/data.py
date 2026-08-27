"""GTSRB loading and reproducible train/validation/test DataLoaders."""

import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import GTSRB


EXERCISE_DIR = Path(__file__).resolve().parent


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (EXERCISE_DIR / path if not path.is_absolute() else path).resolve()


def extract_labels(dataset) -> np.ndarray:
    samples = getattr(dataset, "_samples", None)
    if samples is not None:
        return np.asarray([label for _, label in samples], dtype=np.int64)
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return np.asarray(targets, dtype=np.int64)
    raise AttributeError("GTSRB exposes neither '_samples' nor 'targets'.")


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(config: DictConfig, transform, device: torch.device):
    """Return train/validation/test loaders plus a compact dataset summary."""
    data_dir = resolve_path(config.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = GTSRB(
        root=str(data_dir), split="train", transform=transform, download=True
    )
    test_dataset = GTSRB(
        root=str(data_dir), split="test", transform=transform, download=True
    )

    labels = extract_labels(train_dataset)
    indices = np.arange(len(train_dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config.data.validation_size,
        random_state=config.experiment.seed,
        stratify=labels,
    )

    train_subset = Subset(train_dataset, train_idx.tolist())
    val_subset = Subset(train_dataset, val_idx.tolist())
    pin_memory = bool(config.data.pin_memory and device.type == "cuda")

    generator = torch.Generator().manual_seed(config.experiment.seed)
    common = dict(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )
    loaders = {
        "train": DataLoader(
            train_subset, shuffle=True, generator=generator, **common
        ),
        "validation": DataLoader(val_subset, shuffle=False, **common),
        "test": DataLoader(test_dataset, shuffle=False, **common),
    }

    info = {
        "data_dir": str(data_dir),
        "batch_size": int(config.data.batch_size),
        "num_workers": int(config.data.num_workers),
        "pin_memory": pin_memory,
        "original_train_samples": len(train_dataset),
        "train_samples": len(train_subset),
        "validation_samples": len(val_subset),
        "test_samples": len(test_dataset),
        "train_batches": len(loaders["train"]),
        "validation_batches": len(loaders["validation"]),
        "test_batches": len(loaders["test"]),
    }
    return loaders, info


def print_data_summary(info: dict) -> None:
    print("\n=== Exercise 2 data preparation ===")
    print(f"Data directory: {info['data_dir']}")
    print(f"Batch size: {info['batch_size']}")
    print(f"DataLoader workers: {info['num_workers']}")
    print(f"Pinned memory: {info['pin_memory']}")
    print(f"Original training images: {info['original_train_samples']}")
    print(f"Training subset images: {info['train_samples']}")
    print(f"Validation subset images: {info['validation_samples']}")
    print(f"Test images: {info['test_samples']}")
    print(f"Training batches: {info['train_batches']}")
    print(f"Validation batches: {info['validation_batches']}")
    print(f"Test batches: {info['test_batches']}")
