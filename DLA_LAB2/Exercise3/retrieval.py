from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Exercise3.indexing import DEFAULT_INDEX_OUTPUT_DIR, load_image_index
from Exercise3.model import (
    extract_text_embeddings,
    load_clip,
    normalize_embeddings,
    prepare_text_inputs,
    resolve_device,
)


DEFAULT_TOP_K = 10


def validate_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The search query cannot be empty.")
    return query.strip()


def validate_model_compatibility(
    index_model_id: str,
    requested_model_id: str | None,
) -> str:
    if requested_model_id is None:
        return index_model_id

    model_id = requested_model_id.strip()
    if not model_id:
        raise ValueError("model_id cannot be empty.")
    if model_id != index_model_id:
        raise ValueError(
            "The query model does not match the indexed model: "
            f"query='{model_id}', index='{index_model_id}'."
        )
    return model_id


def encode_text_query(
    *,
    query: str,
    processor: Any,
    model: Any,
    device: torch.device,
    embedding_dimension: int,
) -> np.ndarray:
    """Encode and L2-normalize one text query."""

    text_inputs = prepare_text_inputs(
        processor,
        [validate_query(query)],
        device,
    )
    normalized = normalize_embeddings(
        extract_text_embeddings(model, text_inputs)
    )
    if normalized.shape != (1, embedding_dimension):
        raise ValueError(
            f"Expected text embedding shape {(1, embedding_dimension)}, "
            f"received {tuple(normalized.shape)}."
        )

    embedding = (
        normalized[0]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    if not np.isfinite(embedding).all():
        raise ValueError("The query embedding contains non-finite values.")
    return embedding


def rank_image_embeddings(
    *,
    query_embedding: np.ndarray,
    image_embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Return the highest-scoring images for one normalized query."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if (
        image_embeddings.ndim != 2
        or query_embedding.shape != (image_embeddings.shape[1],)
        or len(metadata) != image_embeddings.shape[0]
    ):
        raise ValueError("Query, embeddings and metadata are incompatible.")

    scores = image_embeddings @ query_embedding
    positions = np.argsort(-scores, kind="stable")[: min(top_k, len(metadata))]

    return [
        {
            "rank": rank,
            "score": float(scores[position]),
            "index_position": int(position),
            "image_id": metadata[position]["image_id"],
            "split": metadata[position]["split"],
            "dataset_row_index": metadata[position]["dataset_row_index"],
            "original_size": metadata[position].get("original_size"),
            "captions": metadata[position]["captions"],
        }
        for rank, position in enumerate(positions, start=1)
    ]


def _print_results(results: list[dict[str, Any]]) -> None:
    print("\nTop results")
    for result in results:
        print(
            f"\nRank {result['rank']} | id={result['image_id']} | "
            f"score={result['score']:.6f}"
        )
        print(
            "  Dataset lookup: "
            f"dataset['{result['split']}']"
            f"[{result['dataset_row_index']}]['image']"
        )
        if result["original_size"] is not None:
            print(f"  Original size: {tuple(result['original_size'])}")
        for index, caption in enumerate(result["captions"], start=1):
            print(f"  Caption {index}: {caption}")


def search_image_index(
    *,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    index_dir: str | Path = DEFAULT_INDEX_OUTPUT_DIR,
    requested_model_id: str | None = None,
    requested_device: str = "auto",
) -> list[dict[str, Any]]:
    """Load an existing index and search it with one text query."""

    cleaned_query = validate_query(query)
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    print("=== Exercise 3.3: text-to-image search ===")
    print(f"Query: {cleaned_query}")
    print(f"Requested top-k: {top_k}")
    print(f"Index directory: {Path(index_dir)}")

    load_start = time.perf_counter()
    image_embeddings, metadata, config = load_image_index(index_dir)
    load_seconds = time.perf_counter() - load_start

    model_id = validate_model_compatibility(
        config["model_id"],
        requested_model_id,
    )
    device = resolve_device(requested_device)

    print(f"Images in index: {len(metadata)}")
    print(f"Embedding dimension: {image_embeddings.shape[1]}")
    print(f"Model checkpoint: {model_id}")
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    if top_k > len(metadata):
        print(
            f"Requested top-k exceeds the index size; "
            f"using top-k={len(metadata)}."
        )

    print("\nLoading CLIP processor and model...")
    processor, model = load_clip(model_id=model_id, device=device)

    search_start = time.perf_counter()
    query_embedding = encode_text_query(
        query=cleaned_query,
        processor=processor,
        model=model,
        device=device,
        embedding_dimension=image_embeddings.shape[1],
    )
    results = rank_image_embeddings(
        query_embedding=query_embedding,
        image_embeddings=image_embeddings,
        metadata=metadata,
        top_k=top_k,
    )
    search_seconds = time.perf_counter() - search_start

    _print_results(results)
    print("\n=== Search completed ===")
    print(f"Index loading and validation: {load_seconds:.3f} seconds")
    print(f"Text encoding and ranking: {search_seconds:.3f} seconds")
    return results
