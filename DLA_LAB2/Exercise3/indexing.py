from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from PIL import Image
from transformers import __version__ as transformers_version

from Exercise3.data import DEFAULT_DATASET_ID, find_caption_fields
from Exercise3.io_utils import (
    load_json,
    prepare_artifact_paths,
    save_json_atomic,
    save_numpy_atomic,
)
from Exercise3.model import (
    DEFAULT_CLIP_MODEL_ID,
    extract_image_embeddings,
    load_clip,
    normalize_embeddings,
    prepare_image_inputs,
    resolve_device,
)


DEFAULT_DATASET_CONFIG = "default"
DEFAULT_INDEX_SPLITS = ("train", "dev", "test")
DEFAULT_INDEX_BATCH_SIZE = 16
DEFAULT_INDEX_OUTPUT_DIR = "Exercise3/outputs/index"

EMBEDDINGS_FILENAME = "image_embeddings.npy"
METADATA_FILENAME = "image_metadata.json"
CONFIG_FILENAME = "index_config.json"


def _artifact_paths(index_dir: str | Path) -> tuple[Path, Path, Path]:
    root = Path(index_dir)
    return (
        root / EMBEDDINGS_FILENAME,
        root / METADATA_FILENAME,
        root / CONFIG_FILENAME,
    )


def _validate_splits(splits: Sequence[str]) -> list[str]:
    cleaned = [split.strip() for split in splits]
    if not cleaned or any(not split for split in cleaned):
        raise ValueError("At least one non-empty dataset split is required.")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"Dataset splits must be unique: {cleaned}")
    return cleaned


def _selected_rows_per_split(
    dataset: DatasetDict,
    splits: Sequence[str],
    limit: int | None,
) -> dict[str, int]:
    missing = [split for split in splits if split not in dataset]
    if missing:
        raise KeyError(
            f"Dataset splits not found: {missing}. "
            f"Available splits: {list(dataset.keys())}"
        )

    total_available = sum(len(dataset[split]) for split in splits)
    if total_available == 0:
        raise ValueError("The selected dataset splits are empty.")
    if limit is not None and not 0 < limit <= total_available:
        raise ValueError(
            f"limit must be between 1 and {total_available}, received {limit}."
        )

    remaining = total_available if limit is None else limit
    selected: dict[str, int] = {}
    for split in splits:
        selected[split] = min(len(dataset[split]), remaining)
        remaining -= selected[split]
        if remaining == 0:
            break
    return selected


def _batch_metadata(
    *,
    batch: dict[str, list[Any]],
    split: str,
    first_row_index: int,
    caption_fields: Sequence[str],
    first_index_position: int,
) -> list[dict[str, Any]]:
    images = batch.get("image")
    if not isinstance(images, list):
        raise TypeError(f"Expected a list of images for split '{split}'.")

    records: list[dict[str, Any]] = []
    for offset, image in enumerate(images):
        row_index = first_row_index + offset
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected a PIL image at {split}[{row_index}], "
                f"received {type(image).__name__}."
            )

        captions = [
            values[offset].strip()
            for field in caption_fields
            if isinstance((values := batch.get(field)), list)
            and isinstance(values[offset], str)
            and values[offset].strip()
        ]
        if not captions:
            raise ValueError(
                f"No non-empty captions were found at {split}[{row_index}]."
            )

        records.append(
            {
                "index_position": first_index_position + offset,
                "image_id": f"{split}:{row_index}",
                "split": split,
                "dataset_row_index": row_index,
                "original_size": [image.width, image.height],
                "captions": captions,
            }
        )
    return records


def build_image_index(
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    splits: Sequence[str] = DEFAULT_INDEX_SPLITS,
    model_id: str = DEFAULT_CLIP_MODEL_ID,
    limit: int | None = None,
    batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
    requested_device: str = "auto",
    output_dir: str | Path = DEFAULT_INDEX_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """Build and persist an L2-normalized CLIP image index."""

    for name, value in {
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "model_id": model_id,
    }.items():
        if not value.strip():
            raise ValueError(f"{name} cannot be empty.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    selected_splits = _validate_splits(splits)
    embeddings_path, metadata_path, config_path = prepare_artifact_paths(
        output_dir,
        (EMBEDDINGS_FILENAME, METADATA_FILENAME, CONFIG_FILENAME),
        force=force,
        artifact_name="Index",
    )
    device = resolve_device(requested_device)

    print("=== Exercise 3.3: image index construction ===")
    print(f"Dataset identifier: {dataset_id}")
    print(f"Dataset configuration: {dataset_config}")
    print(f"Requested splits: {selected_splits}")
    print(f"Model checkpoint: {model_id}")
    print(f"Limit: {limit if limit is not None else 'none'}")
    print(f"Batch size: {batch_size}")
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"Output directory: {Path(output_dir)}")

    print("\nLoading dataset...")
    dataset = load_dataset(dataset_id, name=dataset_config)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict, received {type(dataset).__name__}."
        )

    selected_rows = _selected_rows_per_split(
        dataset,
        selected_splits,
        limit,
    )
    num_images = sum(selected_rows.values())
    total_batches = sum(
        math.ceil(count / batch_size)
        for count in selected_rows.values()
        if count
    )
    print(f"Selected rows per split: {selected_rows}")
    print(f"Total selected images: {num_images}")
    print(f"Total batches: {total_batches}")

    print("\nLoading CLIP processor and model...")
    processor, model = load_clip(model_id=model_id, device=device)
    projection_dim = int(model.config.projection_dim)
    print(f"Projection dimension: {projection_dim}")
    print(f"Model training mode: {model.training}")

    embedding_batches: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    processed_images = 0
    completed_batches = 0
    progress_interval = max(1, total_batches // 10)
    start_time = time.perf_counter()

    for split, split_count in selected_rows.items():
        if split_count == 0:
            continue

        split_dataset = dataset[split]
        caption_fields = find_caption_fields(split_dataset.column_names)
        if "image" not in split_dataset.column_names or not caption_fields:
            raise KeyError(
                f"Split '{split}' must contain image and caption fields."
            )

        for start in range(0, split_count, batch_size):
            stop = min(start + batch_size, split_count)
            batch = split_dataset[start:stop]
            images = batch["image"]

            image_inputs = prepare_image_inputs(processor, images, device)
            normalized = normalize_embeddings(
                extract_image_embeddings(model, image_inputs)
            )
            if normalized.shape != (len(images), projection_dim):
                raise ValueError(
                    f"Unexpected embedding shape in {split}[{start}:{stop}]: "
                    f"{tuple(normalized.shape)}."
                )

            embedding_batches.append(
                normalized.detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
            )
            metadata.extend(
                _batch_metadata(
                    batch=batch,
                    split=split,
                    first_row_index=start,
                    caption_fields=caption_fields,
                    first_index_position=processed_images,
                )
            )

            processed_images += len(images)
            completed_batches += 1
            if (
                completed_batches == 1
                or completed_batches == total_batches
                or completed_batches % progress_interval == 0
            ):
                print(
                    f"Processed {processed_images}/{num_images} images "
                    f"({completed_batches}/{total_batches} batches)"
                )

    if not embedding_batches:
        raise ValueError("No embedding batches were produced.")

    image_embeddings = np.concatenate(embedding_batches, axis=0)
    expected_shape = (num_images, projection_dim)
    if image_embeddings.shape != expected_shape or len(metadata) != num_images:
        raise ValueError(
            "The final embeddings and metadata do not match the "
            f"expected image count {num_images}."
        )
    if image_embeddings.dtype != np.float32:
        raise TypeError(
            f"Expected float32 index, received {image_embeddings.dtype}."
        )
    if not np.isfinite(image_embeddings).all():
        raise ValueError("The final image index contains non-finite values.")

    elapsed_seconds = time.perf_counter() - start_time
    config = {
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "splits": selected_splits,
        "selected_rows_per_split": selected_rows,
        "selection_policy": "first rows in the requested split order",
        "model_id": model_id,
        "num_images": num_images,
        "embedding_dimension": projection_dim,
        "embedding_dtype": str(image_embeddings.dtype),
        "normalized": True,
        "similarity": "cosine similarity via dot product",
        "limit": limit,
        "batch_size": batch_size,
        "requested_device": requested_device,
        "resolved_device": str(device),
        "elapsed_seconds": elapsed_seconds,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "torch": torch.__version__,
            "transformers": transformers_version,
            "numpy": np.__version__,
        },
        "artifacts": {
            "embeddings": EMBEDDINGS_FILENAME,
            "metadata": METADATA_FILENAME,
            "config": CONFIG_FILENAME,
        },
    }

    save_numpy_atomic(image_embeddings, embeddings_path)
    save_json_atomic(metadata, metadata_path)
    save_json_atomic(config, config_path)

    print("\n=== Index construction completed ===")
    print(f"Images indexed: {num_images}")
    print(f"Final embedding shape: {image_embeddings.shape}")
    print(f"Embedding dimension: {projection_dim}")
    print(f"Embedding dtype: {image_embeddings.dtype}")
    print(f"Elapsed time: {elapsed_seconds:.3f} seconds")
    print(f"Embeddings saved in: {embeddings_path}")
    print(f"Metadata saved in: {metadata_path}")
    print(f"Configuration saved in: {config_path}")
    return config


def _validate_metadata(
    metadata: list[Any],
    num_images: int,
) -> list[dict[str, Any]]:
    if len(metadata) != num_images:
        raise ValueError(
            f"Expected {num_images} metadata records, received {len(metadata)}."
        )

    image_ids: list[str] = []
    validated: list[dict[str, Any]] = []
    for position, record in enumerate(metadata):
        if not isinstance(record, dict):
            raise TypeError(
                f"Metadata record {position} must be a JSON object."
            )
        if record.get("index_position") != position:
            raise ValueError(
                f"Metadata record {position} has an inconsistent index_position."
            )

        image_id = record.get("image_id")
        split = record.get("split")
        row_index = record.get("dataset_row_index")
        captions = record.get("captions")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(
                f"Metadata record {position} has an invalid image_id."
            )
        if not isinstance(split, str) or not split:
            raise ValueError(
                f"Metadata record {position} has an invalid split."
            )
        if not isinstance(row_index, int) or row_index < 0:
            raise ValueError(
                f"Metadata record {position} has an invalid row index."
            )
        if (
            not isinstance(captions, list)
            or not captions
            or any(
                not isinstance(caption, str) or not caption.strip()
                for caption in captions
            )
        ):
            raise ValueError(
                f"Metadata record {position} has invalid captions."
            )

        image_ids.append(image_id)
        validated.append(record)

    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate image identifiers were found in the index.")
    return validated


def load_image_index(
    index_dir: str | Path = DEFAULT_INDEX_OUTPUT_DIR,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Load and validate embeddings, metadata and index configuration."""

    embeddings_path, metadata_path, config_path = _artifact_paths(index_dir)
    missing = [
        path
        for path in (embeddings_path, metadata_path, config_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing index artifacts: "
            + ", ".join(str(path) for path in missing)
        )

    image_embeddings = np.load(embeddings_path, allow_pickle=False)
    metadata = load_json(metadata_path)
    config = load_json(config_path)
    if not isinstance(config, dict) or not isinstance(metadata, list):
        raise TypeError(
            "Index configuration must be an object and metadata a list."
        )

    required = {
        "dataset_id",
        "dataset_config",
        "model_id",
        "num_images",
        "embedding_dimension",
        "normalized",
    }
    missing_fields = sorted(required.difference(config))
    if missing_fields:
        raise KeyError(
            f"Missing required index configuration fields: {missing_fields}"
        )

    num_images = config["num_images"]
    embedding_dimension = config["embedding_dimension"]
    if not isinstance(num_images, int) or num_images <= 0:
        raise ValueError("Index field num_images is invalid.")
    if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
        raise ValueError("Index field embedding_dimension is invalid.")
    if not isinstance(config["model_id"], str) or not config["model_id"].strip():
        raise ValueError("Index field model_id is invalid.")
    if config["normalized"] is not True:
        raise ValueError("The saved image index is not marked as normalized.")

    expected_shape = (num_images, embedding_dimension)
    if image_embeddings.shape != expected_shape:
        raise ValueError(
            f"Expected index shape {expected_shape}, "
            f"received {image_embeddings.shape}."
        )
    if image_embeddings.dtype != np.float32:
        raise TypeError(
            f"Expected float32 image embeddings, "
            f"received {image_embeddings.dtype}."
        )
    if not np.isfinite(image_embeddings).all():
        raise ValueError("The image index contains non-finite values.")
    if not np.allclose(
        np.linalg.norm(image_embeddings, axis=1),
        1.0,
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError("The saved image embeddings are not L2-normalized.")

    validated_metadata = _validate_metadata(metadata, num_images)
    return image_embeddings, validated_metadata, config
