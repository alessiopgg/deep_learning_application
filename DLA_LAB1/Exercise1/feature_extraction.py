from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet50_Weights,
)

from data import load_gtsrb


EXERCISE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXERCISE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"

FEATURES_DIR = (
        EXERCISE_DIR
        / "outputs"
        / "exercise_1_2"
        / "features"
)


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
    """
    Create the selected pretrained model without its final classifier.
    """
    config = MODEL_CONFIGS[model_name]

    weights = config["weights"]
    model = config["constructor"](weights=weights)

    feature_dimension = model.fc.in_features

    # Remove the original ImageNet classifier
    model.fc = nn.Identity()

    model = model.to(device)
    model.eval()

    transform = weights.transforms()
    batch_size = config["batch_size"]

    return model, transform, batch_size, feature_dimension


def extract_features(model, dataloader, device):
    """
    Extract features and labels from an entire dataset.
    """
    all_features = []
    all_labels = []

    with torch.inference_mode():
        for batch_number, (images, labels) in enumerate(
                dataloader,
                start=1,
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            features = model(images)

            all_features.append(
                features.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

            if batch_number % 100 == 0:
                print(
                    f"Processed batch "
                    f"{batch_number}/{len(dataloader)}"
                )

    all_features = np.concatenate(
        all_features,
        axis=0,
    )

    all_labels = np.concatenate(
        all_labels,
        axis=0,
    )

    return all_features, all_labels


def save_features(
        file_path,
        features,
        labels,
):
    """
    Save features and labels in an NPZ archive.
    """
    np.savez_compressed(
        file_path,
        features=features,
        labels=labels,
    )


def run_feature_extraction(
        model_name,
        force=False,
):
    """
    Extract and save train and test features for one pretrained model.

    If the files already exist, extraction is skipped unless force=True.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_output_path = (
            FEATURES_DIR
            / f"train_features_{model_name}.npz"
    )

    test_output_path = (
            FEATURES_DIR
            / f"test_features_{model_name}.npz"
    )

    if (
            train_output_path.exists()
            and test_output_path.exists()
            and not force
    ):
        print(
            f"Features for {model_name} already exist. "
            "Extraction skipped."
        )
        return

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nPreparing feature extractor: {model_name}")
    print(f"Device: {device}")

    model, transform, batch_size, feature_dimension = (
        create_feature_extractor(
            model_name=model_name,
            device=device,
        )
    )

    print(f"Batch size: {batch_size}")
    print(f"Feature dimension: {feature_dimension}")

    train_dataset, test_dataset = load_gtsrb(
        DATA_DIR,
        transform=transform,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    print("\nExtracting training features...")

    train_features, train_labels = extract_features(
        model=model,
        dataloader=train_loader,
        device=device,
    )

    print("\nExtracting test features...")

    test_features, test_labels = extract_features(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    print("\nExtraction completed")
    print(
        "Training features shape:",
        train_features.shape,
    )
    print(
        "Training labels shape:",
        train_labels.shape,
    )
    print(
        "Test features shape:",
        test_features.shape,
    )
    print(
        "Test labels shape:",
        test_labels.shape,
    )

    save_features(
        file_path=train_output_path,
        features=train_features,
        labels=train_labels,
    )

    save_features(
        file_path=test_output_path,
        features=test_features,
        labels=test_labels,
    )

    print(f"\nTraining features saved in: {train_output_path}")
    print(f"Test features saved in: {test_output_path}")


def main():
    """
    Default execution when this file is launched directly.
    """
    run_feature_extraction(
        model_name="resnet18"
    )


if __name__ == "__main__":
    main()