from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel, __version__ as transformers_version

from Exercise3.data import DEFAULT_DATASET_ID, load_image_caption_samples


DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_CLIP_SPLIT = "dev"
DEFAULT_CLIP_NUM_SAMPLES = 3


def resolve_device(requested_device: str = "auto") -> torch.device:
    """Resolve and validate the device used for CLIP inference."""

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(requested_device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid PyTorch device: {requested_device}") from error

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA is not available.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is unavailable. "
                f"Visible CUDA devices: {torch.cuda.device_count()}."
            )
    elif device.type != "cpu":
        raise ValueError("Only CPU and CUDA devices are supported in this project.")

    return device


def load_clip(
    *,
    model_id: str = DEFAULT_CLIP_MODEL_ID,
    device: torch.device,
) -> tuple[Any, CLIPModel]:
    """Load the CLIP processor and model for inference."""

    processor = AutoProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


def prepare_image_inputs(
    processor: Any,
    images: Sequence[Image.Image],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert PIL images into the tensors expected by CLIP."""

    if not images:
        raise ValueError("At least one image is required.")

    inputs = processor(images=list(images), return_tensors="pt")
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def prepare_text_inputs(
    processor: Any,
    texts: Sequence[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Tokenize and pad text queries for CLIP."""

    cleaned_texts = [text.strip() for text in texts]
    if not cleaned_texts or any(not text for text in cleaned_texts):
        raise ValueError("All text queries must be non-empty.")

    inputs = processor(
        text=cleaned_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def _projected_embeddings(output: Any, modality: str) -> torch.Tensor:
    """Extract projected features across supported Transformers return formats."""

    if isinstance(output, torch.Tensor):
        embeddings = output
    else:
        embeddings = getattr(output, "pooler_output", None)

    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(
            f"CLIP returned an unsupported {modality} feature type: "
            f"{type(output).__name__}."
        )
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected two-dimensional {modality} embeddings, "
            f"received shape {tuple(embeddings.shape)}."
        )

    return embeddings


def extract_image_embeddings(
    model: CLIPModel,
    image_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute projected CLIP image embeddings without gradients."""

    with torch.inference_mode():
        output = model.get_image_features(**image_inputs)

    return _projected_embeddings(output, modality="image")


def extract_text_embeddings(
    model: CLIPModel,
    text_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute projected CLIP text embeddings without gradients."""

    with torch.inference_mode():
        output = model.get_text_features(**text_inputs)

    return _projected_embeddings(output, modality="text")


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """L2-normalize each embedding so dot products equal cosine similarity."""

    return F.normalize(embeddings, p=2, dim=-1)


def _print_tensor(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name}: shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, device={tensor.device}"
    )


def _print_similarity_matrix(
    similarities: torch.Tensor,
    samples: list[dict[str, Any]],
    queries: list[str],
) -> None:
    matrix = similarities.detach().cpu()

    print("\nCosine similarity matrix")
    header = "query\\image" + "".join(
        f" | {sample['split']}[{sample['dataset_row_index']}]"
        for sample in samples
    )
    print(header)

    for query_index, row in enumerate(matrix):
        scores = "".join(f" | {score.item():.4f}" for score in row)
        print(f"query {query_index}{scores}")

    print("\nTop match for each query")
    for query_index, query in enumerate(queries):
        best_image_index = int(torch.argmax(matrix[query_index]).item())
        best_sample = samples[best_image_index]
        expected_sample = samples[query_index]
        best_score = float(matrix[query_index, best_image_index].item())

        print(f"\nQuery {query_index}: {query}")
        print(
            "  Expected source image: "
            f"{expected_sample['split']}[{expected_sample['dataset_row_index']}]"
        )
        print(
            "  Highest-scoring image: "
            f"{best_sample['split']}[{best_sample['dataset_row_index']}]"
        )
        print(f"  Cosine similarity: {best_score:.6f}")


def inspect_clip(
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = DEFAULT_CLIP_SPLIT,
    num_samples: int = DEFAULT_CLIP_NUM_SAMPLES,
    model_id: str = DEFAULT_CLIP_MODEL_ID,
    requested_device: str = "auto",
) -> None:
    """Inspect CLIP preprocessing, embeddings and cross-modal similarities."""

    if num_samples < 2:
        raise ValueError("num_samples must be at least 2 for a similarity comparison.")

    device = resolve_device(requested_device)

    print("=== Exercise 3.3: CLIP inspection ===")
    print(f"Dataset identifier: {dataset_id}")
    print(f"Dataset split: {split}")
    print(f"Number of samples: {num_samples}")
    print(f"Model checkpoint: {model_id}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers_version}")
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    samples = load_image_caption_samples(
        dataset_id=dataset_id,
        split=split,
        num_samples=num_samples,
    )
    images = [sample["image"] for sample in samples]
    queries = [sample["captions"][0] for sample in samples]

    print("\nSelected image-query pairs")
    for sample, query in zip(samples, queries, strict=True):
        print(
            f"- {sample['split']}[{sample['dataset_row_index']}], "
            f"original_size={sample['image'].size}, query={query}"
        )

    print("\nLoading CLIP processor and model...")
    processor, model = load_clip(model_id=model_id, device=device)
    print(f"Model class: {type(model).__name__}")
    print(f"Processor class: {type(processor).__name__}")
    print(f"Configured projection dimension: {model.config.projection_dim}")
    print(f"Model training mode: {model.training}")

    image_inputs = prepare_image_inputs(processor, images, device)
    text_inputs = prepare_text_inputs(processor, queries, device)

    print("\nProcessed model inputs")
    for name, tensor in image_inputs.items():
        _print_tensor(name, tensor)
    for name, tensor in text_inputs.items():
        _print_tensor(name, tensor)

    image_embeddings = extract_image_embeddings(model, image_inputs)
    text_embeddings = extract_text_embeddings(model, text_inputs)

    if image_embeddings.shape[0] != num_samples:
        raise ValueError(
            "The number of image embeddings does not match the number of samples."
        )
    if text_embeddings.shape[0] != num_samples:
        raise ValueError(
            "The number of text embeddings does not match the number of queries."
        )
    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            "Image and text embeddings have different dimensions: "
            f"{image_embeddings.shape[1]} and {text_embeddings.shape[1]}."
        )

    print("\nRaw embeddings")
    _print_tensor("image_embeddings", image_embeddings)
    _print_tensor("text_embeddings", text_embeddings)

    normalized_images = normalize_embeddings(image_embeddings)
    normalized_texts = normalize_embeddings(text_embeddings)

    image_norms = torch.linalg.vector_norm(normalized_images, dim=-1)
    text_norms = torch.linalg.vector_norm(normalized_texts, dim=-1)

    print("\nNormalized embeddings")
    _print_tensor("normalized_image_embeddings", normalized_images)
    _print_tensor("normalized_text_embeddings", normalized_texts)
    print(
        "Image embedding norms: "
        f"min={image_norms.min().item():.6f}, max={image_norms.max().item():.6f}"
    )
    print(
        "Text embedding norms: "
        f"min={text_norms.min().item():.6f}, max={text_norms.max().item():.6f}"
    )

    similarities = normalized_texts @ normalized_images.T
    expected_shape = (num_samples, num_samples)
    if tuple(similarities.shape) != expected_shape:
        raise ValueError(
            f"Expected similarity shape {expected_shape}, "
            f"received {tuple(similarities.shape)}."
        )

    _print_tensor("similarity_scores", similarities)
    _print_similarity_matrix(similarities, samples, queries)

    print("\n=== CLIP inspection completed ===")
