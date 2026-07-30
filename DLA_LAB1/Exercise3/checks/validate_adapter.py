"""Validate the Step 5 PyTorch detection adapter on every official split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from Exercise3.data_pipeline.loading import DEFAULT_CACHE_DIR, load_detection_dataset, resolve_exercise_path
from Exercise3.data_pipeline.adapter import (
    GermanTrafficSignDetectionDataset,
    validate_detection_target,
)
from Exercise3.data_pipeline.taxonomy import NUM_DETECTOR_CLASSES


DEFAULT_OUTPUT_PATH = Path("outputs/step_5/adapter_validation.json")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate image tensors and Faster R-CNN target dictionaries for "
            "all German traffic-sign detection splits."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Hugging Face cache directory, relative to Exercise3 if needed.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSON report path, relative to Exercise3 if needed.",
    )
    parser.add_argument(
        "--sample-split",
        type=str,
        default="validation",
        help="Split whose sample details are printed.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=3,
        help="Zero-based sample index printed after full validation.",
    )
    return parser.parse_args()


def validate_split(split_name: str, split_dataset: Any) -> dict[str, Any]:
    adapter = GermanTrafficSignDetectionDataset(split_dataset)

    source_objects = 0
    output_objects = 0
    duplicates_removed = 0
    empty_images = 0
    image_ids: list[int] = []
    image_dtype_counts: Counter[str] = Counter()
    box_counts: list[int] = []
    observed_labels: set[int] = set()

    for index in range(len(adapter)):
        image, target, diagnostics = adapter.get_item_with_diagnostics(index)
        validate_detection_target(image, target)

        source_objects += diagnostics.source_object_count
        output_objects += diagnostics.output_object_count
        duplicates_removed += diagnostics.exact_duplicates_removed
        empty_images += int(diagnostics.output_object_count == 0)
        image_ids.append(int(target["image_id"]))
        image_dtype_counts[str(image.dtype)] += 1
        box_counts.append(int(target["boxes"].shape[0]))
        observed_labels.update(int(label) for label in target["labels"].tolist())

    unique_image_ids = len(set(image_ids)) == len(image_ids)
    if not unique_image_ids:
        raise ValueError(f"Split '{split_name}' contains duplicate image IDs.")

    return {
        "images": len(adapter),
        "source_objects": source_objects,
        "output_objects": output_objects,
        "exact_duplicates_removed": duplicates_removed,
        "empty_images_after_adapter": empty_images,
        "image_ids_unique": unique_image_ids,
        "image_dtype_counts": dict(sorted(image_dtype_counts.items())),
        "minimum_boxes_per_image": min(box_counts, default=0),
        "maximum_boxes_per_image": max(box_counts, default=0),
        "observed_detector_labels": sorted(observed_labels),
        "observed_label_count": len(observed_labels),
        "target_contract_valid": True,
    }


def build_sample_preview(
    split_name: str,
    split_dataset: Any,
    sample_index: int,
) -> dict[str, Any]:
    adapter = GermanTrafficSignDetectionDataset(split_dataset)
    image, target, diagnostics = adapter.get_item_with_diagnostics(sample_index)
    validate_detection_target(image, target)

    return {
        "split": split_name,
        "sample_index": sample_index,
        "image_id": int(target["image_id"]),
        "image_type": type(image).__name__,
        "image_dtype": str(image.dtype),
        "image_shape": list(image.shape),
        "image_value_min": int(image.min().item()),
        "image_value_max": int(image.max().item()),
        "boxes_type": type(target["boxes"]).__name__,
        "boxes_format": str(target["boxes"].format),
        "boxes_canvas_size": list(target["boxes"].canvas_size),
        "boxes_dtype": str(target["boxes"].dtype),
        "boxes_shape": list(target["boxes"].shape),
        "boxes_xyxy": target["boxes"].tolist(),
        "labels_dtype": str(target["labels"].dtype),
        "labels_shape": list(target["labels"].shape),
        "labels": target["labels"].tolist(),
        "area_dtype": str(target["area"].dtype),
        "area": target["area"].tolist(),
        "iscrowd_dtype": str(target["iscrowd"].dtype),
        "iscrowd": target["iscrowd"].tolist(),
        "diagnostics": diagnostics.to_dict(),
    }


def print_report(report: dict[str, Any]) -> None:
    totals = report["totals"]
    print("\n=== Exercise 3.3 - Step 5: PyTorch dataset adapter ===")
    print(f"Images validated: {totals['images']}")
    print(f"Raw annotations: {totals['source_objects']}")
    print(f"Adapter annotations: {totals['output_objects']}")
    print(f"Exact duplicate copies removed: {totals['duplicates_removed']}")
    print(f"Empty images preserved: {totals['empty_images']}")
    print(f"All target contracts valid: {report['all_target_contracts_valid']}")
    print(f"Detector class count including background: {NUM_DETECTOR_CLASSES}")

    print("\nPer split:")
    for split_name, split_report in report["splits"].items():
        print(
            f"  - {split_name}: {split_report['images']} images, "
            f"{split_report['output_objects']} retained objects, "
            f"{split_report['empty_images_after_adapter']} empty images, "
            f"{split_report['exact_duplicates_removed']} duplicates removed"
        )

    sample = report["sample_preview"]
    print("\nSample preview:")
    print(f"  split/index: {sample['split']}[{sample['sample_index']}]")
    print(f"  image_id: {sample['image_id']}")
    print(
        f"  image: {sample['image_type']} {sample['image_shape']} "
        f"{sample['image_dtype']} range="
        f"[{sample['image_value_min']}, {sample['image_value_max']}]"
    )
    print(
        f"  boxes: {sample['boxes_type']} {sample['boxes_shape']} "
        f"{sample['boxes_dtype']} {sample['boxes_format']}"
    )
    print(f"  labels: {sample['labels']} ({sample['labels_dtype']})")
    print(f"  area: {sample['area']} ({sample['area_dtype']})")
    print(f"  iscrowd: {sample['iscrowd']} ({sample['iscrowd_dtype']})")


def main() -> None:
    arguments = parse_arguments()
    dataset, _ = load_detection_dataset(arguments.cache_dir)

    if arguments.sample_split not in dataset:
        raise KeyError(
            f"Unknown sample split '{arguments.sample_split}'. Available: "
            f"{sorted(dataset.keys())}."
        )
    if not 0 <= arguments.sample_index < len(dataset[arguments.sample_split]):
        raise IndexError(
            f"Sample index {arguments.sample_index} is outside split "
            f"'{arguments.sample_split}'."
        )

    split_reports = {
        str(split_name): validate_split(str(split_name), split_dataset)
        for split_name, split_dataset in dataset.items()
    }

    totals = {
        "images": sum(item["images"] for item in split_reports.values()),
        "source_objects": sum(
            item["source_objects"] for item in split_reports.values()
        ),
        "output_objects": sum(
            item["output_objects"] for item in split_reports.values()
        ),
        "duplicates_removed": sum(
            item["exact_duplicates_removed"] for item in split_reports.values()
        ),
        "empty_images": sum(
            item["empty_images_after_adapter"] for item in split_reports.values()
        ),
    }

    report = {
        "adapter_contract": {
            "image": "tv_tensors.Image[3,H,W], uint8 before Step 6 transforms",
            "boxes": "float32 BoundingBoxes[N,4], XYXY",
            "labels": "int64 Tensor[N], foreground labels 1..43",
            "image_id": "unique Python int within each split",
            "area": "float32 Tensor[N], recomputed from retained XYXY boxes",
            "iscrowd": "int64 Tensor[N], all zeros",
            "background_label": 0,
            "num_detector_classes": NUM_DETECTOR_CLASSES,
        },
        "duplicate_policy": (
            "Within each image, retain the first annotation for an exact "
            "(source category, x, y, width, height) duplicate and discard only "
            "subsequent exact copies."
        ),
        "splits": split_reports,
        "totals": totals,
        "all_target_contracts_valid": all(
            item["target_contract_valid"] for item in split_reports.values()
        ),
        "sample_preview": build_sample_preview(
            arguments.sample_split,
            dataset[arguments.sample_split],
            arguments.sample_index,
        ),
    }

    output_path = resolve_exercise_path(arguments.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_report(report)
    print(f"\nValidation report saved to: {output_path}")


if __name__ == "__main__":
    main()
