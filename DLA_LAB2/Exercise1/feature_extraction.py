import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from transformer_inspection import MODEL_CHECKPOINT


REQUIRED_SPLITS = (
    "train",
    "validation",
    "test",
)

TOKENIZATION_BATCH_SIZE = 512

DEFAULT_SMOKE_TEST_EXAMPLES = 8
DEFAULT_SMOKE_TEST_BATCH_SIZE = 4

DEFAULT_EXTRACTION_BATCH_SIZE = 32
DEFAULT_PROGRESS_INTERVAL = 25


def validate_feature_extraction_dataset(
        dataset: DatasetDict,
) -> None:
    """
    Verify that all required splits and columns are available.
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
            for column_name in ("text", "label")
            if column_name not in split_dataset.column_names
        ]

        if missing_columns:
            raise ValueError(
                f"Split '{split_name}' is missing columns: "
                f"{missing_columns}"
            )


def compute_token_lengths(
        texts: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = TOKENIZATION_BATCH_SIZE,
) -> np.ndarray:
    """
    Compute tokenized sequence lengths without padding or truncation.

    Special tokens are included, so every sequence length also
    accounts for [CLS] and [SEP].
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    token_lengths: list[int] = []

    for batch_start in range(
            0,
            len(texts),
            batch_size,
    ):
        batch_end = batch_start + batch_size

        text_batch = list(
            texts[batch_start:batch_end]
        )

        encoding = tokenizer(
            text_batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )

        input_ids_batch = encoding["input_ids"]

        token_lengths.extend(
            len(input_ids)
            for input_ids in input_ids_batch
        )

    return np.asarray(
        token_lengths,
        dtype=np.int64,
    )


def summarize_token_lengths(
        token_lengths: np.ndarray,
        split_name: str,
        model_sequence_limit: int,
) -> dict:
    """
    Build a descriptive token-length summary for one split.
    """
    if token_lengths.size == 0:
        raise ValueError(
            f"Split '{split_name}' produced no token lengths."
        )

    over_limit_count = int(
        np.sum(
            token_lengths > model_sequence_limit
        )
    )

    return {
        "split": split_name,
        "examples": int(token_lengths.size),
        "mean": float(np.mean(token_lengths)),
        "standard_deviation": float(
            np.std(token_lengths)
        ),
        "minimum": int(np.min(token_lengths)),
        "percentile_25": float(
            np.percentile(token_lengths, 25)
        ),
        "median": float(
            np.median(token_lengths)
        ),
        "percentile_75": float(
            np.percentile(token_lengths, 75)
        ),
        "percentile_95": float(
            np.percentile(token_lengths, 95)
        ),
        "maximum": int(np.max(token_lengths)),
        "model_sequence_limit": model_sequence_limit,
        "examples_over_limit": over_limit_count,
        "percentage_over_limit": (
                over_limit_count
                / token_lengths.size
                * 100
        ),
    }


def build_token_length_summary(
        dataset: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        model_sequence_limit: int,
) -> pd.DataFrame:
    """
    Measure tokenized lengths for train, validation and test.
    """
    records = []

    for split_name in REQUIRED_SPLITS:
        texts = dataset[split_name]["text"]

        print(
            f"Measuring token lengths for "
            f"'{split_name}' "
            f"({len(texts)} examples)..."
        )

        token_lengths = compute_token_lengths(
            texts=texts,
            tokenizer=tokenizer,
        )

        records.append(
            summarize_token_lengths(
                token_lengths=token_lengths,
                split_name=split_name,
                model_sequence_limit=(
                    model_sequence_limit
                ),
            )
        )

    return pd.DataFrame(records)


def run_token_length_preflight(
        dataset: DatasetDict,
        output_dir: Path,
        model_checkpoint: str = MODEL_CHECKPOINT,
) -> pd.DataFrame:
    """
    Run the token-length preflight for Exercise 1.3.

    No Transformer forward pass and no feature extraction are
    performed in this step.
    """
    validate_feature_extraction_dataset(dataset)

    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint
    )

    model_config = AutoConfig.from_pretrained(
        model_checkpoint
    )

    model_sequence_limit = int(
        model_config.max_position_embeddings
    )

    print("\n=== Exercise 1.3: token-length preflight ===")
    print(f"Checkpoint: {model_checkpoint}")
    print(
        f"Tokenizer type: "
        f"{type(tokenizer).__name__}"
    )
    print(
        f"Configuration type: "
        f"{type(model_config).__name__}"
    )
    print(
        f"Tokenizer model_max_length: "
        f"{tokenizer.model_max_length}"
    )
    print(
        f"Model maximum sequence length: "
        f"{model_sequence_limit}"
    )
    print(
        "Special tokens are included in every measured "
        "sequence length."
    )
    print(
        "Padding and truncation are disabled during "
        "this measurement.\n"
    )

    token_length_summary = (
        build_token_length_summary(
            dataset=dataset,
            tokenizer=tokenizer,
            model_sequence_limit=(
                model_sequence_limit
            ),
        )
    )

    output_dir = Path(output_dir)
    results_dir = output_dir / "results"

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
            results_dir
            / "token_length_summary.csv"
    )

    token_length_summary.to_csv(
        output_path,
        index=False,
    )

    print("\n=== Token-length summary ===")
    print(
        token_length_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    total_examples_over_limit = int(
        token_length_summary[
            "examples_over_limit"
        ].sum()
    )

    print("\n=== Truncation preflight ===")

    if total_examples_over_limit == 0:
        print(
            "No example exceeds the model sequence limit."
        )
        print(
            "For this dataset and tokenizer, truncation "
            "is not required to remain within the "
            "DistilBERT architectural limit."
        )
    else:
        print(
            f"{total_examples_over_limit} examples exceed "
            f"the model sequence limit of "
            f"{model_sequence_limit} tokens."
        )
        print(
            "A truncation strategy will be required before "
            "feature extraction."
        )

    print(
        f"\nResults saved in: {output_path}"
    )
    print(
        "No model forward pass was executed."
    )

    return token_length_summary


def resolve_device(
        requested_device: str,
) -> torch.device:
    """
    Resolve auto, cpu, cuda or an explicit device such as cuda:0.
    """
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    device = torch.device(requested_device)

    if (
            device.type == "cuda"
            and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "A CUDA device was requested, but CUDA is not "
            "available in the current PyTorch environment."
        )

    return device


def load_frozen_feature_extractor(
        model_checkpoint: str,
        device: torch.device,
) -> tuple[
    PreTrainedTokenizerBase,
    PreTrainedModel,
]:
    """
    Load the tokenizer and a frozen DistilBERT base encoder.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint
    )

    model = AutoModel.from_pretrained(
        model_checkpoint
    )

    model.requires_grad_(False)
    model.eval()
    model.to(device)

    return tokenizer, model


def extract_cls_features(
        texts: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        device: torch.device,
        batch_size: int,
        progress_label: str | None = None,
        progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> tuple[np.ndarray, dict]:
    """
    Extract the last-layer first-token representation for each text.

    Returns:
        features:
            NumPy matrix with shape
            [number_of_examples, hidden_size].

        first_batch_diagnostics:
            Shapes observed during the first processed batch.
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if progress_interval <= 0:
        raise ValueError(
            "progress_interval must be greater than zero."
        )

    if len(texts) == 0:
        raise ValueError(
            "At least one text is required for extraction."
        )

    feature_batches: list[np.ndarray] = []
    first_batch_diagnostics: dict = {}

    total_batches = math.ceil(
        len(texts) / batch_size
    )

    for batch_index, batch_start in enumerate(
            range(
                0,
                len(texts),
                batch_size,
            ),
            start=1,
    ):
        batch_end = min(
            batch_start + batch_size,
            len(texts),
            )

        text_batch = list(
            texts[batch_start:batch_end]
        )

        encoding = tokenizer(
            text_batch,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        model_inputs = {
            "input_ids": encoding[
                "input_ids"
            ].to(device),
            "attention_mask": encoding[
                "attention_mask"
            ].to(device),
        }

        with torch.inference_mode():
            outputs = model(**model_inputs)

        last_hidden_state = (
            outputs.last_hidden_state
        )

        cls_features = (
            last_hidden_state[:, 0, :]
        )

        expected_batch_size = len(text_batch)

        if cls_features.ndim != 2:
            raise ValueError(
                "Expected CLS features with shape "
                "[batch_size, hidden_size], "
                f"but received "
                f"{tuple(cls_features.shape)}."
            )

        if cls_features.shape[0] != expected_batch_size:
            raise ValueError(
                "The number of extracted feature rows does "
                "not match the current text batch."
            )

        if batch_start == 0:
            first_batch_diagnostics = {
                "input_ids_shape": tuple(
                    model_inputs[
                        "input_ids"
                    ].shape
                ),
                "attention_mask_shape": tuple(
                    model_inputs[
                        "attention_mask"
                    ].shape
                ),
                "last_hidden_state_shape": tuple(
                    last_hidden_state.shape
                ),
                "cls_features_shape": tuple(
                    cls_features.shape
                ),
                "input_device": str(
                    model_inputs[
                        "input_ids"
                    ].device
                ),
                "output_device": str(
                    cls_features.device
                ),
                "output_dtype": str(
                    cls_features.dtype
                ),
                "requires_gradient": bool(
                    cls_features.requires_grad
                ),
            }

        feature_batches.append(
            cls_features.cpu().numpy()
        )

        should_print_progress = (
                progress_label is not None
                and (
                        batch_index == 1
                        or batch_index % progress_interval == 0
                        or batch_index == total_batches
                )
        )

        if should_print_progress:
            print(
                f"{progress_label}: "
                f"batch {batch_index}/{total_batches}, "
                f"examples {batch_end}/{len(texts)}"
            )

    features = np.concatenate(
        feature_batches,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    hidden_size = int(
        model.config.hidden_size
    )

    expected_shape = (
        len(texts),
        hidden_size,
    )

    if features.shape != expected_shape:
        raise ValueError(
            "Unexpected final feature matrix shape: "
            f"{features.shape} instead of "
            f"{expected_shape}."
        )

    return features, first_batch_diagnostics


def save_feature_archive(
        output_path: Path,
        features: np.ndarray,
        labels: np.ndarray,
) -> None:
    """
    Save features, labels and original split indices atomically.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if features.ndim != 2:
        raise ValueError(
            "Features must be a two-dimensional matrix."
        )

    if labels.ndim != 1:
        raise ValueError(
            "Labels must be a one-dimensional array."
        )

    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "Features and labels contain different numbers "
            "of examples."
        )

    indices = np.arange(
        labels.shape[0],
        dtype=np.int64,
    )

    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    np.savez_compressed(
        temporary_path,
        features=features,
        labels=labels,
        indices=indices,
    )

    temporary_path.replace(output_path)


def save_json_atomically(
        data: dict,
        output_path: Path,
) -> None:
    """
    Save a JSON file through a temporary file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    with temporary_path.open(
            "w",
            encoding="utf-8",
    ) as output_file:
        json.dump(
            data,
            output_file,
            indent=2,
        )

    temporary_path.replace(output_path)


def run_feature_extraction_smoke_test(
        dataset: DatasetDict,
        max_examples: int = DEFAULT_SMOKE_TEST_EXAMPLES,
        batch_size: int = DEFAULT_SMOKE_TEST_BATCH_SIZE,
        requested_device: str = "auto",
        model_checkpoint: str = MODEL_CHECKPOINT,
) -> np.ndarray:
    """
    Extract CLS features from a small training subset.

    No features are saved and no classifier is trained.
    """
    validate_feature_extraction_dataset(dataset)

    if max_examples <= 0:
        raise ValueError(
            "max_examples must be greater than zero."
        )

    train_dataset = dataset["train"]

    selected_examples = min(
        max_examples,
        len(train_dataset),
    )

    smoke_dataset = train_dataset.select(
        range(selected_examples)
    )

    texts = smoke_dataset["text"]

    labels = np.asarray(
        smoke_dataset["label"],
        dtype=np.int64,
    )

    device = resolve_device(
        requested_device=requested_device
    )

    tokenizer, model = (
        load_frozen_feature_extractor(
            model_checkpoint=model_checkpoint,
            device=device,
        )
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "\n=== Exercise 1.3: feature-extraction "
        "smoke test ==="
    )
    print(f"Checkpoint: {model_checkpoint}")
    print(
        f"Tokenizer type: "
        f"{type(tokenizer).__name__}"
    )
    print(
        f"Model type: "
        f"{type(model).__name__}"
    )
    print(f"Requested device: {requested_device}")
    print(f"Resolved device: {device}")
    print(f"Model training mode: {model.training}")
    print(f"Total model parameters: {total_parameters:,}")
    print(
        f"Trainable model parameters: "
        f"{trainable_parameters:,}"
    )
    print("Selected split: train")
    print(f"Selected examples: {selected_examples}")
    print(f"Extraction batch size: {batch_size}")
    print("Padding: dynamic within each batch")
    print("Truncation: disabled")

    features, diagnostics = extract_cls_features(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
    )

    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "The feature and label counts do not match: "
            f"{features.shape[0]} != {labels.shape[0]}."
        )

    print("\n=== First batch diagnostics ===")
    print(
        "input_ids: "
        f"{diagnostics['input_ids_shape']}"
    )
    print(
        "attention_mask: "
        f"{diagnostics['attention_mask_shape']}"
    )
    print(
        "last_hidden_state: "
        f"{diagnostics['last_hidden_state_shape']}"
    )
    print(
        "CLS features: "
        f"{diagnostics['cls_features_shape']}"
    )
    print(
        f"Input device: "
        f"{diagnostics['input_device']}"
    )
    print(
        f"Output device before NumPy conversion: "
        f"{diagnostics['output_device']}"
    )
    print(
        f"Output tensor dtype: "
        f"{diagnostics['output_dtype']}"
    )
    print(
        f"Requires gradient: "
        f"{diagnostics['requires_gradient']}"
    )

    print("\n=== Final smoke-test outputs ===")
    print(
        f"Feature matrix type: "
        f"{type(features).__name__}"
    )
    print(
        f"Feature matrix shape: "
        f"{features.shape}"
    )
    print(
        f"Feature matrix dtype: "
        f"{features.dtype}"
    )
    print(
        f"Labels type: "
        f"{type(labels).__name__}"
    )
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    print(
        "First feature vector, first 10 components:"
    )
    print(features[0, :10])
    print(f"Selected labels: {labels.tolist()}")

    print(
        "\nSmoke test completed successfully."
    )
    print(
        "No feature file was saved and no classifier "
        "was trained."
    )

    return features


def run_full_feature_extraction(
        dataset: DatasetDict,
        output_dir: Path,
        batch_size: int = DEFAULT_EXTRACTION_BATCH_SIZE,
        requested_device: str = "auto",
        model_checkpoint: str = MODEL_CHECKPOINT,
        overwrite: bool = False,
) -> dict:
    """
    Extract and save CLS features for all official dataset splits.
    """
    validate_feature_extraction_dataset(dataset)

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    results_dir = output_dir / "results"

    archive_paths = {
        split_name: (
                features_dir
                / f"{split_name}_features.npz"
        )
        for split_name in REQUIRED_SPLITS
    }

    metadata_path = (
            results_dir
            / "feature_extraction_metadata.json"
    )

    expected_paths = [
        *archive_paths.values(),
        metadata_path,
    ]

    existing_paths = [
        path
        for path in expected_paths
        if path.exists()
    ]

    if existing_paths and not overwrite:
        formatted_paths = "\n".join(
            f"- {path}"
            for path in existing_paths
        )

        raise FileExistsError(
            "Feature-extraction outputs already exist:\n"
            f"{formatted_paths}\n"
            "Use --overwrite only if they should be replaced."
        )

    device = resolve_device(
        requested_device=requested_device
    )

    tokenizer, model = (
        load_frozen_feature_extractor(
            model_checkpoint=model_checkpoint,
            device=device,
        )
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "\n=== Exercise 1.3: full feature extraction ==="
    )
    print(f"Checkpoint: {model_checkpoint}")
    print(
        f"Tokenizer type: "
        f"{type(tokenizer).__name__}"
    )
    print(
        f"Model type: "
        f"{type(model).__name__}"
    )
    print(f"Requested device: {requested_device}")
    print(f"Resolved device: {device}")
    print(f"Model training mode: {model.training}")
    print(f"Total model parameters: {total_parameters:,}")
    print(
        f"Trainable model parameters: "
        f"{trainable_parameters:,}"
    )
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Extraction batch size: {batch_size}")
    print("Feature: last_hidden_state[:, 0, :]")
    print("Padding: dynamic within each batch")
    print("Truncation: disabled")
    print("Feature dtype on disk: float32")

    metadata = {
        "model_checkpoint": model_checkpoint,
        "tokenizer_type": type(tokenizer).__name__,
        "model_type": type(model).__name__,
        "requested_device": requested_device,
        "resolved_device": str(device),
        "batch_size": batch_size,
        "feature_source": (
            "last_hidden_state[:, 0, :]"
        ),
        "hidden_size": int(
            model.config.hidden_size
        ),
        "padding": "dynamic_per_batch",
        "truncation": False,
        "total_model_parameters": int(
            total_parameters
        ),
        "trainable_model_parameters": int(
            trainable_parameters
        ),
        "splits": {},
    }

    for split_name in REQUIRED_SPLITS:
        split_dataset = dataset[split_name]

        texts = split_dataset["text"]

        labels = np.asarray(
            split_dataset["label"],
            dtype=np.int64,
        )

        print(
            f"\n=== Extracting split: {split_name} ==="
        )
        print(f"Examples: {len(texts)}")

        start_time = time.perf_counter()

        features, diagnostics = extract_cls_features(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            progress_label=split_name,
        )

        elapsed_seconds = (
                time.perf_counter() - start_time
        )

        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Split '{split_name}' has "
                f"{features.shape[0]} feature rows but "
                f"{labels.shape[0]} labels."
            )

        if not np.isfinite(features).all():
            raise ValueError(
                f"Split '{split_name}' contains non-finite "
                "feature values."
            )

        archive_path = archive_paths[split_name]

        save_feature_archive(
            output_path=archive_path,
            features=features,
            labels=labels,
        )

        metadata["splits"][split_name] = {
            "examples": int(labels.shape[0]),
            "feature_shape": list(features.shape),
            "label_shape": list(labels.shape),
            "feature_dtype": str(features.dtype),
            "label_dtype": str(labels.dtype),
            "elapsed_seconds": float(
                elapsed_seconds
            ),
            "archive_path": str(archive_path),
            "first_batch_diagnostics": diagnostics,
        }

        print(
            f"Feature shape: {features.shape}"
        )
        print(f"Label shape: {labels.shape}")
        print(f"Feature dtype: {features.dtype}")
        print(f"Finite values: True")
        print(
            f"Elapsed time: "
            f"{elapsed_seconds:.2f} seconds"
        )
        print(f"Saved archive: {archive_path}")

    save_json_atomically(
        data=metadata,
        output_path=metadata_path,
    )

    print(
        "\n=== Feature extraction completed ==="
    )

    for split_name in REQUIRED_SPLITS:
        split_metadata = metadata[
            "splits"
        ][split_name]

        print(
            f"{split_name}: "
            f"{tuple(split_metadata['feature_shape'])}"
        )

    print(f"Metadata saved in: {metadata_path}")
    print(
        "No classifier was trained in this step."
    )

    return metadata