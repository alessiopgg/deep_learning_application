from __future__ import annotations

import json
import statistics
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from datasets import DatasetDict, IterableDatasetDict, get_dataset_config_names, load_dataset
from PIL import Image


DEFAULT_DATASET_ID = "intro/flickr8k"
DEFAULT_EDA_LIMIT = 200
DEFAULT_NUM_EXAMPLES = 3


def find_caption_fields(column_names: Iterable[str]) -> list[str]:
    """Return caption columns ordered by their numeric suffix when possible."""

    fields = [name for name in column_names if name.startswith("caption_")]

    def sort_key(name: str) -> tuple[int, str]:
        suffix = name.removeprefix("caption_")
        return (int(suffix), name) if suffix.isdigit() else (10_000, name)

    return sorted(fields, key=sort_key)


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
        }

    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _iter_examples(split_dataset: Any, limit: int, streaming: bool) -> Iterable[dict[str, Any]]:
    if streaming:
        return islice(iter(split_dataset), limit)

    actual_limit = min(limit, len(split_dataset))
    return (split_dataset[index] for index in range(actual_limit))


def _save_preview(
    image: Image.Image,
    split_name: str,
    example_index: int,
    output_dir: Path,
) -> str:
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    output_path = preview_dir / f"{split_name}_{example_index:03d}.jpg"
    image.convert("RGB").save(output_path, format="JPEG", quality=90)
    return str(output_path)


def _inspect_split(
    split_name: str,
    split_dataset: Any,
    *,
    streaming: bool,
    eda_limit: int,
    num_examples: int,
    output_dir: Path,
) -> dict[str, Any]:
    column_names = list(split_dataset.column_names)
    caption_fields = find_caption_fields(column_names)

    widths: list[float] = []
    heights: list[float] = []
    aspect_ratios: list[float] = []
    caption_counts: list[float] = []
    caption_word_lengths: list[float] = []
    image_modes: dict[str, int] = {}
    image_formats: dict[str, int] = {}
    image_python_types: dict[str, int] = {}
    shown_examples: list[dict[str, Any]] = []

    for sample_index, example in enumerate(
        _iter_examples(split_dataset, eda_limit, streaming)
    ):
        if "image" not in example:
            raise KeyError(
                f"The split '{split_name}' does not contain the required 'image' field. "
                f"Available fields: {column_names}"
            )

        image = example["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected a PIL image in split '{split_name}', "
                f"but received {type(image).__name__}."
            )

        width, height = image.size
        widths.append(float(width))
        heights.append(float(height))
        aspect_ratios.append(float(width / height))

        image_type = f"{type(image).__module__}.{type(image).__name__}"
        image_python_types[image_type] = image_python_types.get(image_type, 0) + 1

        mode = str(image.mode)
        image_modes[mode] = image_modes.get(mode, 0) + 1

        image_format = str(image.format or "unknown")
        image_formats[image_format] = image_formats.get(image_format, 0) + 1

        captions = [
            example[field]
            for field in caption_fields
            if isinstance(example.get(field), str) and example[field].strip()
        ]
        caption_counts.append(float(len(captions)))
        caption_word_lengths.extend(float(len(caption.split())) for caption in captions)

        if sample_index < num_examples:
            preview_path = _save_preview(
                image=image,
                split_name=split_name,
                example_index=sample_index,
                output_dir=output_dir,
            )

            metadata = {
                key: value
                for key, value in example.items()
                if key != "image" and key not in caption_fields
            }

            shown_examples.append(
                {
                    "dataset_row_index": sample_index,
                    "image_size": [width, height],
                    "image_mode": mode,
                    "image_format": image_format,
                    "metadata": metadata,
                    "captions": captions,
                    "preview_path": preview_path,
                }
            )

    split_length = None if streaming else len(split_dataset)

    return {
        "split": split_name,
        "num_rows": split_length,
        "columns": column_names,
        "features": str(split_dataset.features),
        "caption_fields": caption_fields,
        "eda_sample_size": len(widths),
        "image_width": _numeric_summary(widths),
        "image_height": _numeric_summary(heights),
        "image_aspect_ratio": _numeric_summary(aspect_ratios),
        "captions_per_image": _numeric_summary(caption_counts),
        "caption_length_words": _numeric_summary(caption_word_lengths),
        "image_modes": image_modes,
        "image_formats": image_formats,
        "image_python_types": image_python_types,
        "examples": shown_examples,
    }


def _print_numeric_summary(label: str, summary: dict[str, float | int | None]) -> None:
    if summary["count"] == 0:
        print(f"{label}: no observations")
        return

    print(
        f"{label}: "
        f"min={summary['min']:.3f}, "
        f"mean={summary['mean']:.3f}, "
        f"median={summary['median']:.3f}, "
        f"max={summary['max']:.3f}"
    )


def _print_split_report(report: dict[str, Any]) -> None:
    print(f"\n--- Split: {report['split']} ---")
    if report["num_rows"] is None:
        print("Rows: not available in streaming mode")
    else:
        print(f"Rows: {report['num_rows']}")

    print(f"Columns: {report['columns']}")
    print(f"Features: {report['features']}")
    print(f"Caption fields: {report['caption_fields']}")
    print(f"EDA sample size: {report['eda_sample_size']}")

    _print_numeric_summary("Image width [px]", report["image_width"])
    _print_numeric_summary("Image height [px]", report["image_height"])
    _print_numeric_summary("Aspect ratio width/height", report["image_aspect_ratio"])
    _print_numeric_summary("Captions per image", report["captions_per_image"])
    _print_numeric_summary("Caption length [words]", report["caption_length_words"])

    print(f"Image Python types: {report['image_python_types']}")
    print(f"Image modes: {report['image_modes']}")
    print(f"Image formats: {report['image_formats']}")

    for example_number, example in enumerate(report["examples"], start=1):
        print(f"\nExample {example_number}")
        print(f"  Dataset row index: {example['dataset_row_index']}")
        print(f"  Image size: {tuple(example['image_size'])}")
        print(f"  Image mode: {example['image_mode']}")
        print(f"  Image format: {example['image_format']}")
        print(f"  Metadata: {example['metadata']}")
        print(f"  Preview saved in: {example['preview_path']}")
        for caption_index, caption in enumerate(example["captions"], start=1):
            print(f"  Caption {caption_index}: {caption}")


def inspect_dataset(
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    streaming: bool = False,
    eda_limit: int = DEFAULT_EDA_LIMIT,
    num_examples: int = DEFAULT_NUM_EXAMPLES,
    output_dir: str | Path = "Exercise3/outputs/dataset_inspection",
) -> dict[str, Any]:
    """Load Flickr8k, inspect its schema and run a compact deterministic EDA."""

    if eda_limit <= 0:
        raise ValueError("eda_limit must be greater than zero.")
    if num_examples <= 0:
        raise ValueError("num_examples must be greater than zero.")
    if num_examples > eda_limit:
        raise ValueError("num_examples cannot be greater than eda_limit.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=== Exercise 3.3: Flickr8k dataset inspection ===")
    print(f"Dataset identifier: {dataset_id}")
    print(f"Streaming mode: {streaming}")
    print(f"EDA limit per split: {eda_limit}")
    print(f"Examples saved per split: {num_examples}")

    print("\nAvailable configurations:")
    config_names = get_dataset_config_names(dataset_id)
    if config_names:
        for config_name in config_names:
            print(f"- {config_name}")
    else:
        print("- No named configurations reported")

    print("\nLoading dataset...")
    dataset = load_dataset(dataset_id, streaming=streaming)

    expected_type = IterableDatasetDict if streaming else DatasetDict
    if not isinstance(dataset, expected_type):
        raise TypeError(
            f"Expected {expected_type.__name__}, received {type(dataset).__name__}."
        )

    split_names = list(dataset.keys())
    if not split_names:
        raise ValueError("The loaded dataset does not contain any split.")

    print(f"Loaded object type: {type(dataset).__name__}")
    print(f"Available splits: {split_names}")

    split_reports = []
    for split_name in split_names:
        split_report = _inspect_split(
            split_name=split_name,
            split_dataset=dataset[split_name],
            streaming=streaming,
            eda_limit=eda_limit,
            num_examples=num_examples,
            output_dir=output_path,
        )
        split_reports.append(split_report)
        _print_split_report(split_report)

    report = {
        "dataset_id": dataset_id,
        "streaming": streaming,
        "configurations": config_names,
        "splits": split_names,
        "eda_limit_per_split": eda_limit,
        "num_examples_per_split": num_examples,
        "split_reports": split_reports,
    }

    report_name = (
        "dataset_inspection_streaming.json"
        if streaming
        else "dataset_inspection.json"
    )
    report_path = output_path / report_name
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n=== Inspection completed ===")
    print(f"JSON report saved in: {report_path}")
    print(f"Image previews saved in: {output_path / 'previews'}")

    return report


def load_image_caption_samples(
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = "dev",
    num_samples: int = 3,
) -> list[dict[str, Any]]:
    """Load a small deterministic group of image-caption samples."""

    if not split.strip():
        raise ValueError("split cannot be empty.")
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero.")

    split_dataset = load_dataset(dataset_id, split=split)
    if num_samples > len(split_dataset):
        raise ValueError(
            f"Requested {num_samples} samples from split '{split}', "
            f"which contains only {len(split_dataset)} rows."
        )

    column_names = list(split_dataset.column_names)
    caption_fields = find_caption_fields(column_names)
    if "image" not in column_names:
        raise KeyError(
            f"The split '{split}' does not contain the required 'image' field. "
            f"Available fields: {column_names}"
        )
    if not caption_fields:
        raise KeyError(
            f"No caption fields were found in split '{split}'. "
            f"Available fields: {column_names}"
        )

    samples: list[dict[str, Any]] = []
    for row_index in range(num_samples):
        example = split_dataset[row_index]
        image = example["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected a PIL image at {split}[{row_index}], "
                f"but received {type(image).__name__}."
            )

        captions = [
            example[field].strip()
            for field in caption_fields
            if isinstance(example.get(field), str) and example[field].strip()
        ]
        if not captions:
            raise ValueError(
                f"No non-empty captions were found at {split}[{row_index}]."
            )

        samples.append(
            {
                "image": image,
                "split": split,
                "dataset_row_index": row_index,
                "captions": captions,
            }
        )

    return samples
