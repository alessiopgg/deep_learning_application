from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import __version__ as transformers_version

from Exercise3.indexing import DEFAULT_INDEX_OUTPUT_DIR, load_image_index
from Exercise3.io_utils import (
    prepare_artifact_paths,
    save_csv_atomic,
    save_json_atomic,
)
from Exercise3.model import (
    extract_text_embeddings,
    load_clip,
    normalize_embeddings,
    prepare_text_inputs,
    resolve_device,
)


DEFAULT_EVALUATION_SPLIT = "test"
DEFAULT_EVALUATION_BATCH_SIZE = 64
DEFAULT_EVALUATION_OUTPUT_DIR = "Exercise3/outputs/evaluation"

METRICS_FILENAME = "text_to_image_metrics.json"
RANKS_FILENAME = "text_to_image_query_ranks.csv"

RANK_FIELDNAMES = [
    "query_id",
    "image_id",
    "split",
    "dataset_row_index",
    "caption_index",
    "caption",
    "target_candidate_position",
    "target_index_position",
    "rank",
    "target_score",
    "top1_image_id",
    "top1_score",
    "top1_is_target",
]


def _select_candidates(
    image_embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    split: str,
    limit: int | None,
) -> tuple[np.ndarray, list[dict[str, Any]], list[int]]:
    split = split.strip()
    if not split:
        raise ValueError("split cannot be empty.")

    positions = sorted(
        (
            position
            for position, record in enumerate(metadata)
            if record["split"] == split
        ),
        key=lambda position: metadata[position]["dataset_row_index"],
    )
    if not positions:
        raise ValueError(
            f"The saved index does not contain split '{split}'."
        )
    if limit is not None:
        if not 0 < limit <= len(positions):
            raise ValueError(
                f"limit must be between 1 and {len(positions)}, "
                f"received {limit}."
            )
        positions = positions[:limit]

    selected_metadata = [metadata[position] for position in positions]
    row_indices = [
        record["dataset_row_index"] for record in selected_metadata
    ]
    if len(row_indices) != len(set(row_indices)):
        raise ValueError(
            f"Duplicate dataset row indices found in split '{split}'."
        )
    return image_embeddings[positions], selected_metadata, positions


def _build_queries(
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for candidate_position, record in enumerate(metadata):
        for caption_index, caption in enumerate(record["captions"]):
            caption = caption.strip()
            if not caption:
                raise ValueError(
                    f"Image '{record['image_id']}' contains an empty caption."
                )
            queries.append(
                {
                    "query_id": (
                        f"{record['image_id']}:caption_{caption_index}"
                    ),
                    "image_id": record["image_id"],
                    "split": record["split"],
                    "dataset_row_index": record["dataset_row_index"],
                    "caption_index": caption_index,
                    "caption": caption,
                    "target_candidate_position": candidate_position,
                    "target_index_position": record["index_position"],
                }
            )
    if not queries:
        raise ValueError("No text queries were generated for evaluation.")
    return queries


def _stable_target_ranks(
    scores: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Return one-based ranks with stable candidate-order tie breaking."""

    rows = np.arange(scores.shape[0])
    target_scores = scores[rows, targets]
    greater = np.sum(scores > target_scores[:, None], axis=1)
    candidates = np.arange(scores.shape[1])
    tied_before = np.sum(
        (scores == target_scores[:, None])
        & (candidates[None, :] < targets[:, None]),
        axis=1,
    )
    return 1 + greater + tied_before


def _batch_rows(
    *,
    records: list[dict[str, Any]],
    scores: np.ndarray,
    ranks: np.ndarray,
    candidate_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    targets = np.asarray(
        [record["target_candidate_position"] for record in records],
        dtype=np.int64,
    )
    top1_positions = np.argmax(scores, axis=1)
    rows: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        target = int(targets[index])
        top1 = int(top1_positions[index])
        rows.append(
            {
                **record,
                "rank": int(ranks[index]),
                "target_score": float(scores[index, target]),
                "top1_image_id": candidate_metadata[top1]["image_id"],
                "top1_score": float(scores[index, top1]),
                "top1_is_target": top1 == target,
            }
        )
    return rows


def evaluate_text_to_image(
    *,
    index_dir: str | Path = DEFAULT_INDEX_OUTPUT_DIR,
    split: str = DEFAULT_EVALUATION_SPLIT,
    limit: int | None = None,
    batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
    requested_device: str = "auto",
    output_dir: str | Path = DEFAULT_EVALUATION_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate caption-to-image retrieval on one indexed split."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    metrics_path, ranks_path = prepare_artifact_paths(
        output_dir,
        (METRICS_FILENAME, RANKS_FILENAME),
        force=force,
        artifact_name="Evaluation",
    )

    print("=== Exercise 3.3: text-to-image evaluation ===")
    print(f"Index directory: {Path(index_dir)}")
    print(f"Evaluation split: {split}")
    print(f"Image limit: {limit if limit is not None else 'none'}")
    print(f"Text batch size: {batch_size}")
    print(f"Output directory: {Path(output_dir)}")

    total_start = time.perf_counter()

    print("\nLoading and validating image index...")
    load_start = time.perf_counter()
    image_embeddings, metadata, config = load_image_index(index_dir)
    index_load_seconds = time.perf_counter() - load_start

    candidate_embeddings, candidate_metadata, global_positions = (
        _select_candidates(
            image_embeddings,
            metadata,
            split,
            limit,
        )
    )
    queries = _build_queries(candidate_metadata)
    num_candidates, embedding_dimension = candidate_embeddings.shape
    num_queries = len(queries)
    num_batches = (num_queries + batch_size - 1) // batch_size

    print(f"Candidate images: {num_candidates}")
    print(f"Caption queries: {num_queries}")
    print(f"Embedding dimension: {embedding_dimension}")
    print(f"Text batches: {num_batches}")

    device = resolve_device(requested_device)
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    print("\nLoading CLIP processor and model...")
    model_load_start = time.perf_counter()
    processor, model = load_clip(
        model_id=config["model_id"],
        device=device,
    )
    model_load_seconds = time.perf_counter() - model_load_start
    if int(model.config.projection_dim) != embedding_dimension:
        raise ValueError(
            "CLIP projection dimension does not match the image index."
        )

    result_rows: list[dict[str, Any]] = []
    ranks_list: list[int] = []
    progress_interval = max(1, num_batches // 10)
    evaluation_start = time.perf_counter()

    for batch_number, start in enumerate(
        range(0, num_queries, batch_size),
        start=1,
    ):
        records = queries[start : start + batch_size]
        text_inputs = prepare_text_inputs(
            processor,
            [record["caption"] for record in records],
            device,
        )
        text_embeddings = normalize_embeddings(
            extract_text_embeddings(model, text_inputs)
        )
        if text_embeddings.shape != (
            len(records),
            embedding_dimension,
        ):
            raise ValueError(
                f"Unexpected text embedding shape in batch {batch_number}."
            )

        scores = (
            text_embeddings.detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
            @ candidate_embeddings.T
        )
        targets = np.asarray(
            [
                record["target_candidate_position"]
                for record in records
            ],
            dtype=np.int64,
        )
        ranks = _stable_target_ranks(scores, targets)
        result_rows.extend(
            _batch_rows(
                records=records,
                scores=scores,
                ranks=ranks,
                candidate_metadata=candidate_metadata,
            )
        )
        ranks_list.extend(ranks.astype(int).tolist())

        processed = min(start + batch_size, num_queries)
        if (
            batch_number == 1
            or batch_number == num_batches
            or batch_number % progress_interval == 0
        ):
            print(
                f"Processed {processed}/{num_queries} queries "
                f"({batch_number}/{num_batches} batches)"
            )

    ranks_array = np.asarray(ranks_list, dtype=np.int64)
    if ranks_array.shape != (num_queries,) or len(result_rows) != num_queries:
        raise ValueError("The evaluation produced an invalid number of results.")

    evaluation_seconds = time.perf_counter() - evaluation_start
    total_seconds = time.perf_counter() - total_start
    metrics = {
        "recall_at_1": float(np.mean(ranks_array <= 1)),
        "recall_at_5": float(np.mean(ranks_array <= 5)),
        "recall_at_10": float(np.mean(ranks_array <= 10)),
        "median_rank": float(np.median(ranks_array)),
        "mean_rank": float(np.mean(ranks_array)),
        "min_rank": int(np.min(ranks_array)),
        "max_rank": int(np.max(ranks_array)),
    }

    summary = {
        "task": "text-to-image retrieval",
        "protocol": {
            "candidate_gallery": "images from the selected indexed split",
            "queries": (
                "all stored captions associated with each selected image"
            ),
            "ground_truth": "the image paired with each caption",
            "rank_indexing": "one-based",
            "recall_values": "fractions in the interval [0, 1]",
        },
        "dataset_id": config["dataset_id"],
        "dataset_config": config["dataset_config"],
        "split": split,
        "limit": limit,
        "model_id": config["model_id"],
        "num_candidate_images": num_candidates,
        "num_queries": num_queries,
        "captions_per_image": num_queries / num_candidates,
        "embedding_dimension": embedding_dimension,
        "text_batch_size": batch_size,
        "metrics": metrics,
        "timing_seconds": {
            "index_loading_and_validation": index_load_seconds,
            "model_loading": model_load_seconds,
            "text_encoding_and_ranking": evaluation_seconds,
            "total": total_seconds,
        },
        "software": {
            "torch": torch.__version__,
            "transformers": transformers_version,
            "numpy": np.__version__,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_global_index_positions": {
            "first": int(global_positions[0]),
            "last": int(global_positions[-1]),
        },
        "artifacts": {
            "metrics": METRICS_FILENAME,
            "query_ranks": RANKS_FILENAME,
        },
    }

    save_json_atomic(summary, metrics_path)
    save_csv_atomic(result_rows, RANK_FIELDNAMES, ranks_path)

    print("\n=== Evaluation completed ===")
    print(f"Candidate images: {num_candidates}")
    print(f"Caption queries: {num_queries}")
    print(f"Recall@1: {metrics['recall_at_1']:.4f}")
    print(f"Recall@5: {metrics['recall_at_5']:.4f}")
    print(f"Recall@10: {metrics['recall_at_10']:.4f}")
    print(f"Median Rank: {metrics['median_rank']:.2f}")
    print(f"Mean Rank: {metrics['mean_rank']:.2f}")
    print(
        "Text encoding and ranking time: "
        f"{evaluation_seconds:.3f} seconds"
    )
    print(f"Metrics saved in: {metrics_path}")
    print(f"Per-query ranks saved in: {ranks_path}")
    return summary
