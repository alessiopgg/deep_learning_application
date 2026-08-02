from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import DatasetDict


REQUIRED_SPLITS = ("train", "validation", "test")
REQUIRED_COLUMNS = ("text", "label")


def validate_dataset_structure(dataset: DatasetDict) -> None:
    """
    Verify that the dataset contains the splits and columns
    required by Exercise 1.
    """
    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            "Expected a DatasetDict, "
            f"but received {type(dataset).__name__}."
        )

    missing_splits = [
        split_name
        for split_name in REQUIRED_SPLITS
        if split_name not in dataset
    ]

    if missing_splits:
        raise ValueError(
            f"Missing required splits: {missing_splits}"
        )

    for split_name in REQUIRED_SPLITS:
        split_dataset = dataset[split_name]

        missing_columns = [
            column_name
            for column_name in REQUIRED_COLUMNS
            if column_name not in split_dataset.column_names
        ]

        if missing_columns:
            raise ValueError(
                f"Split '{split_name}' is missing columns: "
                f"{missing_columns}"
            )


def get_label_names(dataset: DatasetDict) -> list[str]:
    """
    Return the semantic names associated with the label IDs.
    """
    label_feature = dataset["train"].features["label"]
    label_names = getattr(label_feature, "names", None)

    if not label_names:
        raise ValueError(
            "The label feature does not expose class names."
        )

    return list(label_names)


def print_dataset_overview(
        dataset: DatasetDict,
        declared_split_names: list[str],
) -> None:
    """
    Print the fundamental structure of the loaded dataset.
    """
    print("\n=== Exercise 1.1: dataset overview ===")

    print(f"Loaded object type: {type(dataset).__name__}")
    print(
        "Splits declared by the dataset repository: "
        f"{declared_split_names}"
    )
    print(
        "Splits present in the loaded DatasetDict: "
        f"{list(dataset.keys())}"
    )

    print("\n=== Split sizes ===")

    for split_name, split_dataset in dataset.items():
        print(
            f"{split_name}: "
            f"{len(split_dataset)} examples"
        )

    train_dataset = dataset["train"]

    print("\n=== Training split schema ===")
    print(f"Object type: {type(train_dataset).__name__}")
    print(f"Column names: {train_dataset.column_names}")
    print(f"Features: {train_dataset.features}")

    label_names = get_label_names(dataset)

    print("\n=== Label mapping ===")

    for label_id, label_name in enumerate(label_names):
        print(f"{label_id} -> {label_name}")

    first_example = train_dataset[0]

    print("\n=== First training example ===")
    print(f"text: {first_example['text']!r}")
    print(f"label: {first_example['label']}")


def build_integrity_summary(
        dataset: DatasetDict,
        number_of_labels: int,
) -> pd.DataFrame:
    """
    Check missing texts, empty texts, invalid labels
    and exact duplicates inside each split.
    """
    records = []

    for split_name in REQUIRED_SPLITS:
        split_dataset = dataset[split_name]

        texts = split_dataset["text"]
        labels = split_dataset["label"]

        missing_text_count = sum(
            text is None
            for text in texts
        )

        non_string_text_count = sum(
            text is not None and not isinstance(text, str)
            for text in texts
        )

        empty_text_count = sum(
            isinstance(text, str) and not text.strip()
            for text in texts
        )

        invalid_label_count = sum(
            not isinstance(label, int)
            or label < 0
            or label >= number_of_labels
            for label in labels
        )

        valid_string_texts = [
            text
            for text in texts
            if isinstance(text, str)
        ]

        duplicate_text_count = (
                len(valid_string_texts)
                - len(set(valid_string_texts))
        )

        records.append(
            {
                "split": split_name,
                "examples": len(split_dataset),
                "missing_texts": missing_text_count,
                "non_string_texts": non_string_text_count,
                "empty_texts": empty_text_count,
                "invalid_labels": invalid_label_count,
                "duplicate_texts_within_split": (
                    duplicate_text_count
                ),
            }
        )

    return pd.DataFrame(records)


def build_split_overlap_summary(
        dataset: DatasetDict,
) -> pd.DataFrame:
    """
    Count exact text matches between different dataset splits.

    Exact overlaps across train, validation and test are useful
    to identify possible leakage or repeated examples.
    """
    text_sets = {
        split_name: {
            text
            for text in dataset[split_name]["text"]
            if isinstance(text, str)
        }
        for split_name in REQUIRED_SPLITS
    }

    records = []

    for first_split, second_split in combinations(
            REQUIRED_SPLITS,
            2,
    ):
        overlap = (
                text_sets[first_split]
                & text_sets[second_split]
        )

        records.append(
            {
                "first_split": first_split,
                "second_split": second_split,
                "exact_text_overlap": len(overlap),
            }
        )

    return pd.DataFrame(records)


def build_class_distribution(
        dataset: DatasetDict,
        label_names: list[str],
) -> pd.DataFrame:
    """
    Compute class counts and percentages for every split.
    """
    records = []
    number_of_labels = len(label_names)

    for split_name in REQUIRED_SPLITS:
        labels = np.asarray(
            dataset[split_name]["label"],
            dtype=np.int64,
        )

        counts = np.bincount(
            labels,
            minlength=number_of_labels,
        )

        for label_id, count in enumerate(counts):
            records.append(
                {
                    "split": split_name,
                    "label_id": label_id,
                    "label_name": label_names[label_id],
                    "count": int(count),
                    "percentage": (
                            float(count) / len(labels) * 100
                    ),
                }
            )

    return pd.DataFrame(records)


def describe_values(
        values: np.ndarray,
        split_name: str,
        measurement: str,
) -> dict:
    """
    Compute a compact descriptive summary.
    """
    return {
        "split": split_name,
        "measurement": measurement,
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "minimum": int(np.min(values)),
        "percentile_25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "percentile_75": float(np.percentile(values, 75)),
        "percentile_95": float(np.percentile(values, 95)),
        "maximum": int(np.max(values)),
    }


def build_text_length_summary(
        dataset: DatasetDict,
) -> pd.DataFrame:
    """
    Measure text lengths in characters and whitespace-separated words.

    Token lengths are deliberately not computed here because they depend
    on the tokenizer introduced in Exercise 1.2.
    """
    records = []

    for split_name in REQUIRED_SPLITS:
        texts = [
            text
            for text in dataset[split_name]["text"]
            if isinstance(text, str)
        ]

        character_lengths = np.asarray(
            [len(text) for text in texts],
            dtype=np.int64,
        )

        word_lengths = np.asarray(
            [len(text.split()) for text in texts],
            dtype=np.int64,
        )

        records.append(
            describe_values(
                values=character_lengths,
                split_name=split_name,
                measurement="characters",
            )
        )

        records.append(
            describe_values(
                values=word_lengths,
                split_name=split_name,
                measurement="words",
            )
        )

    return pd.DataFrame(records)


def plot_class_distribution(
        class_distribution: pd.DataFrame,
        output_path: Path,
) -> None:
    """
    Plot negative and positive class percentages by split.
    """
    split_names = list(REQUIRED_SPLITS)
    label_names = (
        class_distribution["label_name"]
        .drop_duplicates()
        .tolist()
    )

    x_positions = np.arange(len(split_names))
    bar_width = 0.35

    figure, axis = plt.subplots(figsize=(9, 5))

    for label_index, label_name in enumerate(label_names):
        label_data = (
            class_distribution[
                class_distribution["label_name"] == label_name
                ]
            .set_index("split")
            .loc[split_names]
        )

        offset = (
                         label_index - (len(label_names) - 1) / 2
                 ) * bar_width

        axis.bar(
            x_positions + offset,
            label_data["percentage"],
            width=bar_width,
            label=label_name,
            )

    axis.set_title("Rotten Tomatoes class distribution")
    axis.set_xlabel("Dataset split")
    axis.set_ylabel("Examples (%)")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(split_names)
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(title="Sentiment")

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_text_length_distribution(
        dataset: DatasetDict,
        output_path: Path,
) -> None:
    """
    Plot the distribution of sentence lengths measured in words.
    """
    figure, axis = plt.subplots(figsize=(10, 6))

    for split_name in REQUIRED_SPLITS:
        word_lengths = [
            len(text.split())
            for text in dataset[split_name]["text"]
            if isinstance(text, str)
        ]

        axis.hist(
            word_lengths,
            bins=40,
            density=True,
            alpha=0.5,
            label=split_name,
        )

    axis.set_title(
        "Rotten Tomatoes sentence-length distribution"
    )
    axis.set_xlabel("Whitespace-separated words")
    axis.set_ylabel("Density")
    axis.grid(alpha=0.3)
    axis.legend(title="Split")

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def run_eda(
        dataset: DatasetDict,
        output_dir: Path,
) -> None:
    """
    Run the complete exploratory analysis for Exercise 1.1.
    """
    validate_dataset_structure(dataset)

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    results_dir = output_dir / "results"

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    label_names = get_label_names(dataset)

    integrity_summary = build_integrity_summary(
        dataset=dataset,
        number_of_labels=len(label_names),
    )

    if integrity_summary["invalid_labels"].sum() > 0:
        raise ValueError(
            "Invalid labels were found in the dataset."
        )

    split_overlap_summary = build_split_overlap_summary(
        dataset=dataset
    )

    class_distribution = build_class_distribution(
        dataset=dataset,
        label_names=label_names,
    )

    text_length_summary = build_text_length_summary(
        dataset=dataset
    )

    integrity_summary.to_csv(
        results_dir / "integrity_checks.csv",
        index=False,
        )

    split_overlap_summary.to_csv(
        results_dir / "split_overlap.csv",
        index=False,
        )

    class_distribution.to_csv(
        results_dir / "class_distribution.csv",
        index=False,
        )

    text_length_summary.to_csv(
        results_dir / "text_length_summary.csv",
        index=False,
        )

    plot_class_distribution(
        class_distribution=class_distribution,
        output_path=(
                figures_dir / "class_distribution.png"
        ),
    )

    plot_text_length_distribution(
        dataset=dataset,
        output_path=(
                figures_dir / "text_length_distribution.png"
        ),
    )

    print("\n=== Integrity checks ===")
    print(
        integrity_summary.to_string(index=False)
    )

    print("\n=== Exact overlaps between splits ===")
    print(
        split_overlap_summary.to_string(index=False)
    )

    print("\n=== Class distribution ===")
    print(
        class_distribution.to_string(
            index=False,
            formatters={
                "percentage": lambda value: f"{value:.2f}%"
            },
        )
    )

    print("\n=== Text-length summary ===")
    print(
        text_length_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print(f"\nFigures saved in: {figures_dir}")
    print(f"Results saved in: {results_dir}")