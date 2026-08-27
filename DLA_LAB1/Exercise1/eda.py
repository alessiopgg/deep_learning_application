from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from data import extract_labels

NUM_CLASSES = 43


def plot_random_samples(dataset, output_path, seed=42):
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(dataset), size=12, replace=False)

    figure, axes = plt.subplots(3, 4, figsize=(12, 9))
    for axis, sample_index in zip(axes.flat, selected_indices):
        image, label = dataset[int(sample_index)]
        axis.imshow(image)
        axis.set_title(f"Class {label}")
        axis.axis("off")

    figure.suptitle("Random samples from the GTSRB training set")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def compute_class_distribution(train_dataset, test_dataset):
    train_labels = extract_labels(train_dataset)
    test_labels = extract_labels(test_dataset)
    train_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    test_counts = np.bincount(test_labels, minlength=NUM_CLASSES)

    return pd.DataFrame(
        {
            "class_id": np.arange(NUM_CLASSES),
            "train_count": train_counts,
            "train_percentage": train_counts / len(train_labels) * 100,
            "test_count": test_counts,
            "test_percentage": test_counts / len(test_labels) * 100,
        }
    )


def plot_class_distribution(distribution, output_path):
    class_ids = distribution["class_id"].to_numpy()
    bar_width = 0.4
    figure, axis = plt.subplots(figsize=(16, 6))

    axis.bar(
        class_ids - bar_width / 2,
        distribution["train_percentage"],
        width=bar_width,
        label="Train",
    )
    axis.bar(
        class_ids + bar_width / 2,
        distribution["test_percentage"],
        width=bar_width,
        label="Test",
    )
    axis.set(
        title="GTSRB class distribution",
        xlabel="Class ID",
        ylabel="Percentage of images",
    )
    axis.set_xticks(class_ids)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def collect_image_metadata(dataset, split_name):
    print(f"Extracting metadata from the {split_name} split...")
    records = []

    for image_path, label in dataset._samples:
        with Image.open(image_path) as image:
            width, height = image.size

        records.append(
            {
                "split": split_name,
                "label": label,
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
                "area": width * height,
            }
        )

    return pd.DataFrame(records)


def plot_image_dimensions(metadata, output_path):
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))

    for split_name in ("train", "test"):
        split_data = metadata[metadata["split"] == split_name]
        for axis, column in zip(axes, ("width", "height", "aspect_ratio")):
            axis.hist(
                split_data[column],
                bins=50,
                density=True,
                alpha=0.5,
                label=split_name,
            )

    axes[0].set(title="Width distribution", xlabel="Width in pixels", ylabel="Density")
    axes[1].set(title="Height distribution", xlabel="Height in pixels", ylabel="Density")
    axes[2].set(
        title="Aspect-ratio distribution",
        xlabel="Width / height",
        ylabel="Density",
    )

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()

    figure.suptitle("GTSRB image dimensions")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def run_eda(train_dataset, test_dataset, output_dir, seed=42):
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    results_dir = output_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_random_samples(
        train_dataset,
        figures_dir / "gtsrb_train_samples.png",
        seed=seed,
    )

    class_distribution = compute_class_distribution(train_dataset, test_dataset)
    class_distribution.to_csv(results_dir / "class_distribution.csv", index=False)
    plot_class_distribution(
        class_distribution,
        figures_dir / "class_distribution.png",
    )

    metadata = pd.concat(
        [
            collect_image_metadata(train_dataset, "train"),
            collect_image_metadata(test_dataset, "test"),
        ],
        ignore_index=True,
    )
    columns = ["width", "height", "aspect_ratio", "area"]
    metadata_summary = metadata.groupby("split")[columns].describe(
        percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
    )
    metadata_summary.to_csv(results_dir / "image_metadata_summary.csv")
    plot_image_dimensions(metadata, figures_dir / "image_dimensions.png")

    train_counts = class_distribution["train_count"]
    least = class_distribution.loc[train_counts.idxmin()]
    most = class_distribution.loc[train_counts.idxmax()]
    imbalance_ratio = train_counts.max() / train_counts.min()
    correlation = class_distribution[
        ["train_percentage", "test_percentage"]
    ].corr().iloc[0, 1]

    print("\n=== Exercise 1.1 results ===")
    print(f"Training images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(
        f"Least represented training class: {int(least['class_id'])} "
        f"({int(least['train_count'])} images)"
    )
    print(
        f"Most represented training class: {int(most['class_id'])} "
        f"({int(most['train_count'])} images)"
    )
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    print(f"Train-test distribution correlation: {correlation:.4f}")
    print("\nImage metadata summary:")
    print(metadata_summary)
    print(f"\nFigures saved in: {figures_dir}")
    print(f"Results saved in: {results_dir}")
