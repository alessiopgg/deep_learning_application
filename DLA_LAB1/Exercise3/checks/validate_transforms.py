"""Validation script for Step 6 minimal transforms."""

from __future__ import annotations

from Exercise3.paths import EXERCISE_DIR

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torchvision.tv_tensors import BoundingBoxes, Image

from Exercise3.data_pipeline.loading import DEFAULT_CACHE_DIR, load_detection_dataset
from Exercise3.data_pipeline.adapter import GermanTrafficSignDetectionDataset
from Exercise3.data_pipeline.transforms import (
    TransformedDetectionDataset,
    boxes_to_list,
    build_detection_transform_pipeline,
    describe_transform_pipeline,
    image_value_range,
)


DEFAULT_OUTPUT_PATH = EXERCISE_DIR / "outputs" / "step_6" / "transform_validation.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Step 6 synchronized transforms for the traffic-sign "
            "detection dataset."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Hugging Face cache directory.",
    )
    parser.add_argument(
        "--sample-split",
        type=str,
        default="validation",
        help="Split used for the detailed sample preview.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=3,
        help="Zero-based sample index inside the selected split.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSON report path.",
    )
    return parser.parse_args()


def ensure_target_contract(image: Image, target: dict[str, Any]) -> None:
    if not isinstance(image, Image):
        raise TypeError(f"Expected tv_tensors.Image, found {type(image).__name__}.")

    if image.dtype != torch.float32:
        raise TypeError(f"Expected image dtype torch.float32, found {image.dtype}.")

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected image shape [3,H,W], found {tuple(image.shape)}.")

    value_min, value_max = image_value_range(image)
    if value_min < 0.0 or value_max > 1.0:
        raise ValueError(
            "Transformed image values must stay within [0, 1], found "
            f"range [{value_min}, {value_max}]."
        )

    required_keys = {"boxes", "labels", "image_id", "area", "iscrowd"}
    missing_keys = required_keys.difference(target)
    if missing_keys:
        raise KeyError(f"Missing target keys: {sorted(missing_keys)}")

    boxes = target["boxes"]
    labels = target["labels"]
    area = target["area"]
    iscrowd = target["iscrowd"]

    if not isinstance(boxes, BoundingBoxes):
        raise TypeError(
            f"Expected target['boxes'] to be BoundingBoxes, found {type(boxes).__name__}."
        )

    if boxes.dtype != torch.float32:
        raise TypeError(f"Expected boxes dtype torch.float32, found {boxes.dtype}.")

    if tuple(boxes.shape)[-1] != 4:
        raise ValueError(f"Expected boxes shape [N,4], found {tuple(boxes.shape)}.")

    if labels.dtype != torch.int64:
        raise TypeError(f"Expected labels dtype torch.int64, found {labels.dtype}.")

    if area.dtype != torch.float32:
        raise TypeError(f"Expected area dtype torch.float32, found {area.dtype}.")

    if iscrowd.dtype != torch.int64:
        raise TypeError(
            f"Expected iscrowd dtype torch.int64, found {iscrowd.dtype}."
        )

    object_count = len(labels)
    if boxes.shape[0] != object_count or len(area) != object_count or len(iscrowd) != object_count:
        raise ValueError("Target tensors do not describe the same number of objects.")


def summarize_split(dataset: TransformedDetectionDataset) -> dict[str, Any]:
    image_dtype_counts: dict[str, int] = {}
    min_boxes_per_image: int | None = None
    max_boxes_per_image: int | None = None
    observed_labels: set[int] = set()
    image_shapes: set[tuple[int, int, int]] = set()
    min_value = 1.0
    max_value = 0.0

    for index in range(len(dataset)):
        image, target = dataset[index]
        ensure_target_contract(image, target)

        dtype_name = str(image.dtype)
        image_dtype_counts[dtype_name] = image_dtype_counts.get(dtype_name, 0) + 1
        image_shapes.add(tuple(int(v) for v in image.shape))

        value_lo, value_hi = image_value_range(image)
        min_value = min(min_value, value_lo)
        max_value = max(max_value, value_hi)

        box_count = int(target["boxes"].shape[0])
        min_boxes_per_image = box_count if min_boxes_per_image is None else min(min_boxes_per_image, box_count)
        max_boxes_per_image = box_count if max_boxes_per_image is None else max(max_boxes_per_image, box_count)

        observed_labels.update(int(v) for v in target["labels"].tolist())

    return {
        "images": len(dataset),
        "image_dtype_counts": image_dtype_counts,
        "unique_image_shapes": [list(shape) for shape in sorted(image_shapes)],
        "image_value_range": {
            "min": float(min_value),
            "max": float(max_value),
        },
        "minimum_boxes_per_image": int(min_boxes_per_image or 0),
        "maximum_boxes_per_image": int(max_boxes_per_image or 0),
        "observed_detector_labels": sorted(observed_labels),
        "observed_label_count": len(observed_labels),
        "target_contract_valid": True,
    }


def build_sample_preview(dataset: TransformedDetectionDataset, split: str, sample_index: int) -> dict[str, Any]:
    image, target = dataset[sample_index]
    ensure_target_contract(image, target)
    value_lo, value_hi = image_value_range(image)

    return {
        "split": split,
        "sample_index": sample_index,
        "image_id": int(target["image_id"]),
        "image_type": type(image).__name__,
        "image_dtype": str(image.dtype),
        "image_shape": [int(v) for v in image.shape],
        "image_value_min": float(value_lo),
        "image_value_max": float(value_hi),
        "boxes_type": type(target["boxes"]).__name__,
        "boxes_format": str(target["boxes"].format),
        "boxes_canvas_size": [int(v) for v in target["boxes"].canvas_size],
        "boxes_dtype": str(target["boxes"].dtype),
        "boxes_shape": [int(v) for v in target["boxes"].shape],
        "boxes_xyxy": boxes_to_list(target["boxes"]),
        "labels_dtype": str(target["labels"].dtype),
        "labels_shape": [int(v) for v in target["labels"].shape],
        "labels": [int(v) for v in target["labels"].tolist()],
        "area_dtype": str(target["area"].dtype),
        "area": [float(v) for v in target["area"].tolist()],
        "iscrowd_dtype": str(target["iscrowd"].dtype),
        "iscrowd": [int(v) for v in target["iscrowd"].tolist()],
    }


def main() -> None:
    arguments = parse_arguments()

    if arguments.sample_index < 0:
        raise ValueError("--sample-index must be greater than or equal to zero.")

    dataset_dict, _ = load_detection_dataset(arguments.cache_dir)
    transform_pipeline = build_detection_transform_pipeline()

    transformed_datasets: dict[str, TransformedDetectionDataset] = {}
    for split_name in ("train", "validation", "test"):
        base_dataset = GermanTrafficSignDetectionDataset(dataset_dict[split_name])
        transformed_datasets[split_name] = TransformedDetectionDataset(
            base_dataset=base_dataset,
            split=split_name,
            transform_pipeline=transform_pipeline,
        )

    if arguments.sample_split not in transformed_datasets:
        raise ValueError(
            f"Unknown sample split '{arguments.sample_split}'. Expected train, validation or test."
        )

    if not 0 <= arguments.sample_index < len(transformed_datasets[arguments.sample_split]):
        raise IndexError(
            f"Sample index {arguments.sample_index} is outside split '{arguments.sample_split}'."
        )

    split_summaries = {
        split_name: summarize_split(dataset)
        for split_name, dataset in transformed_datasets.items()
    }

    sample_preview = build_sample_preview(
        dataset=transformed_datasets[arguments.sample_split],
        split=arguments.sample_split,
        sample_index=arguments.sample_index,
    )

    report = {
        "transform_pipeline": [asdict(item) for item in describe_transform_pipeline()],
        "splits": split_summaries,
        "sample_preview": sample_preview,
        "all_target_contracts_valid": True,
    }

    output_path = arguments.output_path.expanduser()
    if not output_path.is_absolute():
        output_path = EXERCISE_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Exercise 3.3 - Step 6: minimal transforms ===")
    print("Transform policy:")
    print("  - image dtype: torch.uint8 -> torch.float32")
    print("  - image range: [0, 255] -> [0.0, 1.0]")
    print("  - geometric augmentation: no")
    print("  - photometric augmentation: no")
    print("  - explicit resize: no")
    print("  - explicit ImageNet normalization: no")
    print("  - bounding boxes preserved: yes")
    print()
    print(f"All target contracts valid: {report['all_target_contracts_valid']}")
    print()
    print("Per split:")
    for split_name, summary in split_summaries.items():
        print(
            f"  - {split_name}: {summary['images']} images, "
            f"image dtype counts {summary['image_dtype_counts']}, "
            f"value range [{summary['image_value_range']['min']:.4f}, "
            f"{summary['image_value_range']['max']:.4f}], "
            f"boxes/image min={summary['minimum_boxes_per_image']}, "
            f"max={summary['maximum_boxes_per_image']}, "
            f"observed labels={summary['observed_label_count']}"
        )
    print()
    print("Sample preview:")
    print(
        f"  split/index: {sample_preview['split']}[{sample_preview['sample_index']}]"
    )
    print(f"  image_id: {sample_preview['image_id']}")
    print(
        "  image: "
        f"{sample_preview['image_type']} {sample_preview['image_shape']} "
        f"{sample_preview['image_dtype']} range=[{sample_preview['image_value_min']:.4f}, "
        f"{sample_preview['image_value_max']:.4f}]"
    )
    print(
        "  boxes: "
        f"{sample_preview['boxes_type']} {sample_preview['boxes_shape']} "
        f"{sample_preview['boxes_dtype']} {sample_preview['boxes_format']}"
    )
    print(f"  labels: {sample_preview['labels']} ({sample_preview['labels_dtype']})")
    print(f"  area: {sample_preview['area']} ({sample_preview['area_dtype']})")
    print(f"  iscrowd: {sample_preview['iscrowd']} ({sample_preview['iscrowd_dtype']})")
    print()
    print(f"Transform validation report saved to: {output_path}")


if __name__ == "__main__":
    main()
