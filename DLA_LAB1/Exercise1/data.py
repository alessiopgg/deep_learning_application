from pathlib import Path

import numpy as np
from torchvision.datasets import GTSRB


def load_gtsrb(data_dir, transform=None):
    """
    Load the GTSRB training and test datasets.

    The transform parameter allows us to apply image preprocessing
    when required.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = GTSRB(
        root=str(data_dir),
        split="train",
        transform=transform,
        download=True,
    )

    test_dataset = GTSRB(
        root=str(data_dir),
        split="test",
        transform=transform,
        download=True,
    )

    return train_dataset, test_dataset


def extract_labels(dataset):
    """
    Extract all labels without loading the images.
    """
    return np.array(
        [label for _, label in dataset._samples]
    )