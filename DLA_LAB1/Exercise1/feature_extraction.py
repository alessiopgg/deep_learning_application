from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)

from data import load_gtsrb

EXERCISE_DIR = Path(__file__).resolve().parent
DATA_DIR = EXERCISE_DIR.parent / "data"
FEATURES_DIR = EXERCISE_DIR / "outputs" / "exercise_1_2" / "features"

MODEL_CONFIGS = {
    "resnet18": {
        "constructor": resnet18,
        "weights": ResNet18_Weights.IMAGENET1K_V1,
        "batch_size": 32,
    },
    "resnet50": {
        "constructor": resnet50,
        "weights": ResNet50_Weights.IMAGENET1K_V2,
        "batch_size": 16,
    },
}


def create_feature_extractor(model_name, device):
    config = MODEL_CONFIGS[model_name]
    weights = config["weights"]
    model = config["constructor"](weights=weights)
    feature_dimension = model.fc.in_features
    model.fc = nn.Identity()
    model = model.to(device).eval()
    return model, weights.transforms(), config["batch_size"], feature_dimension


def extract_features(model, dataloader, device):
    features, labels = [], []

    with torch.inference_mode():
        for batch_number, (images, batch_labels) in enumerate(dataloader, start=1):
            batch_features = model(images.to(device, non_blocking=True))
            features.append(batch_features.cpu().numpy())
            labels.append(batch_labels.numpy())

            if batch_number % 100 == 0:
                print(f"Processed batch {batch_number}/{len(dataloader)}")

    return np.concatenate(features), np.concatenate(labels)


def run_feature_extraction(model_name, force=False):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    train_output = FEATURES_DIR / f"train_features_{model_name}.npz"
    test_output = FEATURES_DIR / f"test_features_{model_name}.npz"

    if train_output.exists() and test_output.exists() and not force:
        print(f"Features for {model_name} already exist. Extraction skipped.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPreparing feature extractor: {model_name}")
    print(f"Device: {device}")

    model, transform, batch_size, feature_dimension = create_feature_extractor(
        model_name, device
    )
    print(f"Batch size: {batch_size}")
    print(f"Feature dimension: {feature_dimension}")

    train_dataset, test_dataset = load_gtsrb(DATA_DIR, transform=transform)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    print("\nExtracting training features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    print("\nExtracting test features...")
    test_features, test_labels = extract_features(model, test_loader, device)

    np.savez_compressed(train_output, features=train_features, labels=train_labels)
    np.savez_compressed(test_output, features=test_features, labels=test_labels)

    print("\nExtraction completed")
    print("Training features shape:", train_features.shape)
    print("Training labels shape:", train_labels.shape)
    print("Test features shape:", test_features.shape)
    print("Test labels shape:", test_labels.shape)
    print(f"\nTraining features saved in: {train_output}")
    print(f"Test features saved in: {test_output}")


def main():
    run_feature_extraction("resnet18")


if __name__ == "__main__":
    main()
