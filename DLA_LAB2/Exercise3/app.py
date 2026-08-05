from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import gradio as gr
import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from PIL import Image

from Exercise3.indexing import DEFAULT_INDEX_OUTPUT_DIR, load_image_index
from Exercise3.model import load_clip, resolve_device
from Exercise3.retrieval import (
    DEFAULT_TOP_K,
    encode_text_query,
    rank_image_embeddings,
    validate_query,
)


DEFAULT_SERVER_NAME = "127.0.0.1"


def _validate_dataset_lookup(
    dataset: DatasetDict,
    metadata: list[dict[str, Any]],
) -> None:
    largest_rows: dict[str, int] = {}
    for record in metadata:
        split = record["split"]
        largest_rows[split] = max(
            largest_rows.get(split, -1),
            record["dataset_row_index"],
        )

    for split, largest_row in largest_rows.items():
        if split not in dataset:
            raise KeyError(f"Metadata refers to missing split '{split}'.")
        if largest_row >= len(dataset[split]):
            raise IndexError(
                f"Metadata refers to {split}[{largest_row}], but the split "
                f"contains only {len(dataset[split])} rows."
            )


def _gallery_caption(result: dict[str, Any]) -> str:
    return (
        f"#{result['rank']} | score={result['score']:.4f} | "
        f"{result['image_id']}\n{result['captions'][0]}"
    )


def _build_search_callback(
    *,
    image_embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    dataset: DatasetDict,
    processor: Any,
    model: Any,
    device: torch.device,
    top_k: int,
) -> Callable[[str], tuple[list[tuple[Image.Image, str]], str]]:
    embedding_dimension = image_embeddings.shape[1]

    def search(query: str) -> tuple[list[tuple[Image.Image, str]], str]:
        try:
            query = validate_query(query)
        except ValueError as error:
            return [], str(error)

        start = time.perf_counter()
        results = rank_image_embeddings(
            query_embedding=encode_text_query(
                query=query,
                processor=processor,
                model=model,
                device=device,
                embedding_dimension=embedding_dimension,
            ),
            image_embeddings=image_embeddings,
            metadata=metadata,
            top_k=top_k,
        )

        gallery: list[tuple[Image.Image, str]] = []
        for result in results:
            image = dataset[result["split"]][
                result["dataset_row_index"]
            ]["image"]
            if not isinstance(image, Image.Image):
                raise TypeError(
                    f"Expected a PIL image for {result['image_id']}."
                )
            gallery.append((image.copy(), _gallery_caption(result)))

        elapsed = time.perf_counter() - start
        status = (
            f"Found {len(gallery)} results for "
            f"**{query}** in {elapsed:.3f} seconds."
        )
        return gallery, status

    return search


def create_app(
    *,
    index_dir: str | Path = DEFAULT_INDEX_OUTPUT_DIR,
    requested_device: str = "auto",
    top_k: int = DEFAULT_TOP_K,
) -> gr.Blocks:
    """Load resources once and construct the Gradio application."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    print("=== Exercise 3.3: Gradio application startup ===")
    print(f"Index directory: {Path(index_dir)}")
    print(f"Top-k results: {top_k}")
    startup_start = time.perf_counter()

    print("\nLoading and validating image index...")
    image_embeddings, metadata, config = load_image_index(index_dir)

    device = resolve_device(requested_device)
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    print("\nLoading Flickr8k...")
    dataset = load_dataset(
        config["dataset_id"],
        name=config["dataset_config"],
    )
    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict, received {type(dataset).__name__}."
        )
    _validate_dataset_lookup(dataset, metadata)

    print("\nLoading CLIP processor and model...")
    processor, model = load_clip(
        model_id=config["model_id"],
        device=device,
    )
    if int(model.config.projection_dim) != image_embeddings.shape[1]:
        raise ValueError(
            "CLIP projection dimension does not match the saved index."
        )

    search = _build_search_callback(
        image_embeddings=image_embeddings,
        metadata=metadata,
        dataset=dataset,
        processor=processor,
        model=model,
        device=device,
        top_k=min(top_k, len(metadata)),
    )
    print(
        "Application resources loaded in "
        f"{time.perf_counter() - startup_start:.3f} seconds"
    )

    with gr.Blocks(title="Flickr8k Text-to-Image Retrieval") as demo:
        gr.Markdown(
            "# Flickr8k Text-to-Image Retrieval\n"
            "Enter a short English description. CLIP compares its text "
            "embedding with 8,000 saved Flickr8k image embeddings and "
            "returns the ten most similar results."
        )
        with gr.Row():
            query_input = gr.Textbox(
                label="Text query",
                placeholder="e.g. a dog playing outside",
                lines=1,
            )
            search_button = gr.Button("Search", variant="primary")

        status_output = gr.Markdown(
            "Enter a description and start the search."
        )
        gallery_output = gr.Gallery(
            label="Top matches",
            columns=5,
            rows=2,
        )

        for trigger in (search_button.click, query_input.submit):
            trigger(
                fn=search,
                inputs=query_input,
                outputs=[gallery_output, status_output],
            )

    return demo


def launch_app(
    *,
    index_dir: str | Path = DEFAULT_INDEX_OUTPUT_DIR,
    requested_device: str = "auto",
    top_k: int = DEFAULT_TOP_K,
    server_name: str = DEFAULT_SERVER_NAME,
    server_port: int | None = None,
    share: bool = False,
    inbrowser: bool = False,
) -> None:
    if not server_name.strip():
        raise ValueError("server_name cannot be empty.")
    if server_port is not None and not 1 <= server_port <= 65535:
        raise ValueError("server_port must be between 1 and 65535.")

    create_app(
        index_dir=index_dir,
        requested_device=requested_device,
        top_k=top_k,
    ).launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        inbrowser=inbrowser,
    )
