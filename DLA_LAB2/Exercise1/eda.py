from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import DatasetDict


SPLIT_NAMES = ("train", "validation", "test")


def get_label_names(dataset: DatasetDict) -> list[str]:
    """Return the semantic names associated with the label IDs."""
    names = dataset["train"].features["label"].names
    return list(names)


def print_dataset_overview(dataset: DatasetDict) -> None:
    """Print the information requested in Exercise 1.1."""
    label_names = get_label_names(dataset)

    print("\n=== Exercise 1.1: dataset overview ===")
    print(f"Dataset type: {type(dataset).__name__}")
    print(f"Available splits: {list(dataset.keys())}")

    for split_name in SPLIT_NAMES:
        split = dataset[split_name]
        print(
            f"{split_name}: {len(split)} examples, "
            f"columns={split.column_names}"
        )

    print("\nLabel mapping:")
    for label_id, label_name in enumerate(label_names):
        print(f"{label_id} -> {label_name}")

    print("\nFirst training examples:")
    for example in dataset["train"].select(range(3)):
        print(f"label={example['label']} | text={example['text']!r}")


def build_class_distribution(
    dataset: DatasetDict,
    label_names: list[str],
) -> pd.DataFrame:
    """Compute class counts and percentages for each split."""
    records = []

    for split_name in SPLIT_NAMES:
        labels = np.asarray(dataset[split_name]["label"], dtype=np.int64)
        counts = np.bincount(labels, minlength=len(label_names))

        for label_id, count in enumerate(counts):
            records.append(
                {
                    "split": split_name,
                    "label_id": label_id,
                    "label_name": label_names[label_id],
                    "count": int(count),
                    "percentage": float(count / len(labels) * 100),
                }
            )

    return pd.DataFrame(records)


def build_text_length_summary(dataset: DatasetDict) -> pd.DataFrame:
    """Summarize sentence lengths in characters and words."""
    records = []

    for split_name in SPLIT_NAMES:
        texts = dataset[split_name]["text"]

        for measurement, values in {
            "characters": np.asarray([len(text) for text in texts]),
            "words": np.asarray([len(text.split()) for text in texts]),
        }.items():
            records.append(
                {
                    "split": split_name,
                    "measurement": measurement,
                    "mean": float(values.mean()),
                    "standard_deviation": float(values.std()),
                    "minimum": int(values.min()),
                    "median": float(np.median(values)),
                    "percentile_95": float(np.percentile(values, 95)),
                    "maximum": int(values.max()),
                }
            )

    return pd.DataFrame(records)


def plot_class_distribution(
    class_distribution: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot class percentages for train, validation and test."""
    pivot = class_distribution.pivot(
        index="split",
        columns="label_name",
        values="percentage",
    ).loc[list(SPLIT_NAMES)]

    axis = pivot.plot(kind="bar", figsize=(9, 5))
    axis.set_title("Rotten Tomatoes class distribution")
    axis.set_xlabel("Dataset split")
    axis.set_ylabel("Examples (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.3)

    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_text_length_distribution(
    dataset: DatasetDict,
    output_path: Path,
) -> None:
    """Plot word-length distributions for all official splits."""
    figure, axis = plt.subplots(figsize=(10, 6))

    for split_name in SPLIT_NAMES:
        word_lengths = [
            len(text.split())
            for text in dataset[split_name]["text"]
        ]
        axis.hist(
            word_lengths,
            bins=40,
            density=True,
            alpha=0.5,
            label=split_name,
        )

    axis.set_title("Rotten Tomatoes sentence-length distribution")
    axis.set_xlabel("Words")
    axis.set_ylabel("Density")
    axis.grid(alpha=0.3)
    axis.legend(title="Split")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def run_eda(dataset: DatasetDict, output_dir: Path) -> None:
    """Run and save the compact exploratory analysis for Exercise 1.1."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    results_dir = output_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    label_names = get_label_names(dataset)
    class_distribution = build_class_distribution(dataset, label_names)
    text_length_summary = build_text_length_summary(dataset)

    class_distribution.to_csv(
        results_dir / "class_distribution.csv",
        index=False,
    )
    text_length_summary.to_csv(
        results_dir / "text_length_summary.csv",
        index=False,
    )

    plot_class_distribution(
        class_distribution,
        figures_dir / "class_distribution.png",
    )
    plot_text_length_distribution(
        dataset,
        figures_dir / "text_length_distribution.png",
    )

    print("\n=== Class distribution ===")
    print(class_distribution.to_string(index=False))
    print("\n=== Text-length summary ===")
    print(text_length_summary.to_string(index=False, float_format="%.2f"))
    print(f"\nFigures saved in: {figures_dir}")
    print(f"Results saved in: {results_dir}")
