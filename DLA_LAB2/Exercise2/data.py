from collections.abc import Mapping
from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DATASET_NAME = "cornell-movie-review-data/rotten_tomatoes"
MODEL_CHECKPOINT = "distilbert/distilbert-base-uncased"


def load_rotten_tomatoes(
        dataset_name: str = DATASET_NAME,
) -> DatasetDict:
    """Load the official Rotten Tomatoes train, validation and test splits."""
    dataset = load_dataset(dataset_name)

    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected a Hugging Face DatasetDict.")

    return dataset


def load_tokenizer(
        model_checkpoint: str = MODEL_CHECKPOINT,
) -> PreTrainedTokenizerBase:
    """Load the tokenizer associated with the selected DistilBERT checkpoint."""
    return AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_batch(
        examples: Mapping[str, list[Any]],
        tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    """Tokenize a batch without applying fixed padding."""
    texts = examples.get("text")

    if texts is None:
        raise KeyError("The dataset batch does not contain the 'text' column.")

    return tokenizer(
        texts,
        truncation=True,
        return_token_type_ids=False,
    )


def tokenize_dataset(
        dataset: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
) -> DatasetDict:
    """Add input_ids and attention_mask to every dataset split."""
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_batch(examples, tokenizer),
        batched=True,
        desc="Tokenizing Rotten Tomatoes",
    )

    for split_name in tokenized_dataset:
        split = tokenized_dataset[split_name]

        if "token_type_ids" in split.column_names:
            tokenized_dataset[split_name] = split.remove_columns(
                ["token_type_ids"]
            )

    required_columns = {"text", "label", "input_ids", "attention_mask"}

    for split_name, split in tokenized_dataset.items():
        missing_columns = required_columns.difference(split.column_names)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Split {split_name!r} is missing required columns: {missing}"
            )

    return tokenized_dataset


def inspect_tokenized_example(
        raw_dataset: DatasetDict,
        tokenized_dataset: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        split_name: str = "train",
        example_index: int = 0,
) -> None:
    """Print one example before and after tokenization."""
    if split_name not in raw_dataset:
        available = ", ".join(raw_dataset.keys())
        raise ValueError(
            f"Unknown split {split_name!r}. Available splits: {available}"
        )

    split_size = len(raw_dataset[split_name])

    if not 0 <= example_index < split_size:
        raise IndexError(
            f"Example index {example_index} is outside split "
            f"{split_name!r}, which contains {split_size} examples."
        )

    raw_example = raw_dataset[split_name][example_index]
    tokenized_example = tokenized_dataset[split_name][example_index]

    input_ids = tokenized_example["input_ids"]
    attention_mask = tokenized_example["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print("\n=== Exercise 2.1: token preprocessing inspection ===")
    print(f"Dataset identifier: {DATASET_NAME}")
    print(f"Tokenizer checkpoint: {MODEL_CHECKPOINT}")
    print(f"Available splits: {list(raw_dataset.keys())}")

    print("\nSplit sizes:")
    for name, split in raw_dataset.items():
        print(f"- {name}: {len(split)}")

    print(f"\nSelected example: {split_name}/{example_index}")
    print(f"Columns before tokenization: {raw_dataset[split_name].column_names}")
    print(
        "Columns after tokenization: "
        f"{tokenized_dataset[split_name].column_names}"
    )
    print(f"Text: {raw_example['text']!r}")
    print(f"Label: {raw_example['label']}")
    print(f"Number of tokens: {len(input_ids)}")
    print(f"input_ids: {input_ids}")
    print(f"attention_mask: {attention_mask}")
    print(f"Tokens: {tokens}")

    if len(input_ids) != len(attention_mask):
        raise ValueError(
            "input_ids and attention_mask must have the same length."
        )

    if not all(value == 1 for value in attention_mask):
        raise ValueError(
            "Unexpected padding found during Dataset.map(). "
            "Padding must be added later by the data collator."
        )

    print(
        "\nCheck passed: each example contains text, label, input_ids "
        "and attention_mask, without fixed padding or token_type_ids."
    )