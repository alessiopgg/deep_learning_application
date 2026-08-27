from pathlib import Path

import numpy as np
from torchvision.datasets import GTSRB


def load_gtsrb(data_dir, transform=None):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    common_args = {
        "root": str(data_dir),
        "transform": transform,
        "download": True,
    }
    return (
        GTSRB(split="train", **common_args),
        GTSRB(split="test", **common_args),
    )


def extract_labels(dataset):
    return np.array([label for _, label in dataset._samples])
