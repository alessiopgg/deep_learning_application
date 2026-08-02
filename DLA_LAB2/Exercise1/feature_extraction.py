import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from datasets import DatasetDict
from transformers import AutoModel, AutoTokenizer

from transformer_inspection import MODEL_CHECKPOINT


REQUIRED_SPLITS = ("train", "validation", "test")
DEFAULT_EXTRACTION_BATCH_SIZE = 32
DEFAULT_PROGRESS_INTERVAL = 25


def resolve_device(requested_device: str) -> torch.device:
    """Resolve auto, cpu, cuda or an explicit CUDA device."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def extract_cls_features(
    texts: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    progress_label: str,
) -> np.ndarray:
    """Extract the last-layer representation of the first token."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    feature_batches = []
    total_batches = math.ceil(len(texts) / batch_size)
    max_length = int(model.config.max_position_embeddings)

    for batch_number, start in enumerate(
        range(0, len(texts), batch_size),
        start=1,
    ):
        batch_texts = list(texts[start : start + batch_size])
        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            hidden_states = model(**encoding).last_hidden_state

        feature_batches.append(
            hidden_states[:, 0, :].cpu().numpy()
        )

        if (
            batch_number == 1
            or batch_number % DEFAULT_PROGRESS_INTERVAL == 0
            or batch_number == total_batches
        ):
            processed = min(start + batch_size, len(texts))
            print(
                f"{progress_label}: batch {batch_number}/{total_batches}, "
                f"examples {processed}/{len(texts)}"
            )

    return np.concatenate(feature_batches).astype(np.float32, copy=False)


def run_full_feature_extraction(
    dataset: DatasetDict,
    output_dir: Path,
    batch_size: int = DEFAULT_EXTRACTION_BATCH_SIZE,
    requested_device: str = "auto",
    model_checkpoint: str = MODEL_CHECKPOINT,
    overwrite: bool = False,
) -> dict:
    """Extract and save DistilBERT CLS features for all official splits."""
    missing_splits = [
        split_name
        for split_name in REQUIRED_SPLITS
        if split_name not in dataset
    ]
    if missing_splits:
        raise ValueError(f"Missing dataset splits: {missing_splits}")

    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    results_dir = output_dir / "results"
    features_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    archive_paths = {
        split_name: features_dir / f"{split_name}_features.npz"
        for split_name in REQUIRED_SPLITS
    }
    metadata_path = results_dir / "feature_extraction_metadata.json"

    existing_outputs = [
        path
        for path in [*archive_paths.values(), metadata_path]
        if path.exists()
    ]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Feature outputs already exist. Use --overwrite to replace them."
        )

    device = resolve_device(requested_device)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModel.from_pretrained(model_checkpoint)
    model.requires_grad_(False)
    model.eval()
    model.to(device)

    print("\n=== Exercise 1.3: DistilBERT feature extraction ===")
    print(f"Checkpoint: {model_checkpoint}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Feature: last_hidden_state[:, 0, :]")

    metadata = {
        "model_checkpoint": model_checkpoint,
        "device": str(device),
        "batch_size": batch_size,
        "feature_source": "last_hidden_state[:, 0, :]",
        "hidden_size": int(model.config.hidden_size),
        "max_length": int(model.config.max_position_embeddings),
        "splits": {},
    }

    for split_name in REQUIRED_SPLITS:
        split = dataset[split_name]
        texts = split["text"]
        labels = np.asarray(split["label"], dtype=np.int64)

        start_time = time.perf_counter()
        features = extract_cls_features(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            progress_label=split_name,
        )
        elapsed_seconds = time.perf_counter() - start_time

        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Feature/label count mismatch for split '{split_name}'."
            )

        indices = np.arange(len(labels), dtype=np.int64)
        np.savez_compressed(
            archive_paths[split_name],
            features=features,
            labels=labels,
            indices=indices,
        )

        metadata["splits"][split_name] = {
            "examples": int(len(labels)),
            "feature_shape": list(features.shape),
            "elapsed_seconds": float(elapsed_seconds),
            "archive_path": str(archive_paths[split_name]),
        }

        print(
            f"{split_name}: features={features.shape}, "
            f"time={elapsed_seconds:.2f}s"
        )

    with metadata_path.open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2)

    print(f"\nMetadata saved in: {metadata_path}")
    return metadata
