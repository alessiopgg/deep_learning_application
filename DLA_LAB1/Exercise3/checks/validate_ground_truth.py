"""Validate and save Step 8 ground-truth visualizations."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from Exercise3.data_pipeline.adapter import validate_detection_target
from Exercise3.data_pipeline.loaders import build_detection_datasets
from Exercise3.data_pipeline.loading import (
    DEFAULT_CACHE_DIR,
    load_detection_dataset,
    resolve_exercise_path,
)
from Exercise3.visualization.ground_truth import save_ground_truth_triplet


DEFAULT_OUTPUT_DIR = Path("outputs/step_8")
DEFAULT_SAMPLES_PER_SPLIT = 3
DEFAULT_SEED = 42
PIXEL_SCALING_TOLERANCE = 1e-7


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that Step 6 preserves detection annotations and save "
            "representative ground-truth visualizations for every split."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Hugging Face cache directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory relative to Exercise3 unless absolute.",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=DEFAULT_SAMPLES_PER_SPLIT,
        help="Number of representative examples saved for each split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used only when additional representative indices are needed.",
    )
    parser.add_argument(
        "--box-width",
        type=int,
        default=4,
        help="Ground-truth box line width in pixels.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help=(
            "Optional TrueType font path for larger labels. When omitted, "
            "Torchvision uses its portable default font."
        ),
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=20,
        help="Label font size used only together with --font-path.",
    )
    return parser.parse_args()


def _plain_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if type(tensor) is torch.Tensor:
        return tensor
    return tensor.as_subclass(torch.Tensor)


def _assert_tensor_equal(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    split: str,
    sample_index: int,
    field: str,
) -> None:
    if left.dtype != right.dtype:
        raise TypeError(
            f"{split}[{sample_index}] field '{field}' changed dtype: "
            f"{left.dtype} -> {right.dtype}."
        )
    if left.shape != right.shape:
        raise ValueError(
            f"{split}[{sample_index}] field '{field}' changed shape: "
            f"{tuple(left.shape)} -> {tuple(right.shape)}."
        )
    if not torch.equal(_plain_tensor(left), _plain_tensor(right)):
        raise ValueError(
            f"{split}[{sample_index}] field '{field}' changed values."
        )


def validate_split_preservation(
    split: str,
    transformed_dataset: Any,
) -> dict[str, Any]:
    """Compare every adapted sample before and after the Step 6 transform."""
    base_dataset = transformed_dataset.base_dataset
    images_checked = 0
    objects_checked = 0
    empty_images = 0
    maximum_pixel_scaling_error = 0.0

    for sample_index in range(len(transformed_dataset)):
        original_image, original_target = base_dataset[sample_index]
        transformed_image, transformed_target = transformed_dataset[sample_index]

        validate_detection_target(original_image, original_target)
        validate_detection_target(transformed_image, transformed_target)

        if tuple(original_image.shape) != tuple(transformed_image.shape):
            raise ValueError(
                f"{split}[{sample_index}] image shape changed from "
                f"{tuple(original_image.shape)} to {tuple(transformed_image.shape)}."
            )
        if int(original_target["image_id"]) != int(transformed_target["image_id"]):
            raise ValueError(f"{split}[{sample_index}] image_id changed.")

        for field in ("boxes", "labels", "area", "iscrowd"):
            _assert_tensor_equal(
                original_target[field],
                transformed_target[field],
                split=split,
                sample_index=sample_index,
                field=field,
            )

        expected_float_image = _plain_tensor(original_image).to(torch.float32).div(255.0)
        actual_float_image = _plain_tensor(transformed_image)
        pixel_error = float(
            (expected_float_image - actual_float_image).abs().max().item()
        )
        maximum_pixel_scaling_error = max(
            maximum_pixel_scaling_error,
            pixel_error,
        )
        if not math.isfinite(pixel_error) or pixel_error > PIXEL_SCALING_TOLERANCE:
            raise ValueError(
                f"{split}[{sample_index}] float conversion error {pixel_error} "
                f"exceeds tolerance {PIXEL_SCALING_TOLERANCE}."
            )

        object_count = int(transformed_target["boxes"].shape[0])
        images_checked += 1
        objects_checked += object_count
        empty_images += int(object_count == 0)

    return {
        "images_checked": images_checked,
        "objects_checked": objects_checked,
        "empty_images_checked": empty_images,
        "image_shapes_preserved": True,
        "image_ids_preserved": True,
        "boxes_preserved": True,
        "labels_preserved": True,
        "areas_preserved": True,
        "iscrowd_preserved": True,
        "maximum_pixel_scaling_error": maximum_pixel_scaling_error,
        "pixel_scaling_tolerance": PIXEL_SCALING_TOLERANCE,
        "contract_valid": True,
    }


def select_representative_indices(
    transformed_dataset: Any,
    *,
    count: int,
    seed: int,
) -> list[tuple[str, int]]:
    """Select crowded, smallest-object and empty examples deterministically."""
    if count <= 0:
        return []

    max_objects_index: int | None = None
    max_objects = -1
    smallest_box_index: int | None = None
    smallest_area = math.inf
    empty_index: int | None = None

    for sample_index in range(len(transformed_dataset)):
        _, target = transformed_dataset[sample_index]
        object_count = int(target["boxes"].shape[0])

        if object_count > max_objects:
            max_objects = object_count
            max_objects_index = sample_index

        if object_count == 0 and empty_index is None:
            empty_index = sample_index

        if object_count > 0:
            sample_smallest_area = float(target["area"].min().item())
            if sample_smallest_area < smallest_area:
                smallest_area = sample_smallest_area
                smallest_box_index = sample_index

    candidates = [
        ("most-objects", max_objects_index),
        ("smallest-box", smallest_box_index),
        ("empty", empty_index),
    ]

    selected: list[tuple[str, int]] = []
    used_indices: set[int] = set()
    for reason, sample_index in candidates:
        if sample_index is None or sample_index in used_indices:
            continue
        selected.append((reason, sample_index))
        used_indices.add(sample_index)
        if len(selected) == count:
            return selected

    random_generator = random.Random(seed)
    remaining_indices = [
        index for index in range(len(transformed_dataset)) if index not in used_indices
    ]
    random_generator.shuffle(remaining_indices)
    for sample_index in remaining_indices:
        selected.append(("seeded-extra", sample_index))
        if len(selected) == count:
            break

    return selected


def save_split_visualizations(
    *,
    split: str,
    transformed_dataset: Any,
    output_dir: Path,
    samples_per_split: int,
    seed: int,
    box_width: int,
    font_path: Path | None,
    font_size: int,
) -> list[dict[str, Any]]:
    base_dataset = transformed_dataset.base_dataset
    selected = select_representative_indices(
        transformed_dataset,
        count=samples_per_split,
        seed=seed,
    )

    records: list[dict[str, Any]] = []
    for selection_reason, sample_index in selected:
        original_image, original_target = base_dataset[sample_index]
        transformed_image, transformed_target = transformed_dataset[sample_index]

        record = save_ground_truth_triplet(
            original_image=original_image,
            original_target=original_target,
            transformed_image=transformed_image,
            transformed_target=transformed_target,
            split=split,
            sample_index=sample_index,
            selection_reason=selection_reason,
            output_dir=output_dir,
            box_width=box_width,
            font_path=font_path,
            font_size=font_size,
        )
        records.append(record.to_dict())

    return records


def print_report(report: dict[str, Any], output_dir: Path) -> None:
    print("\n=== Exercise 3.3 - Step 8: ground-truth visualization ===")
    print("Annotation source: final transformed detection datasets")
    print("Box format: absolute XYXY")
    print("Ground-truth color: red")
    print("Saved views: original, transformed, side-by-side comparison")
    print()

    print("Full preservation checks:")
    for split, summary in report["preservation_checks"].items():
        print(
            f"  - {split}: {summary['images_checked']} images, "
            f"{summary['objects_checked']} objects, "
            f"{summary['empty_images_checked']} empty images, "
            f"max pixel scaling error="
            f"{summary['maximum_pixel_scaling_error']:.3g}"
        )

    print("\nSaved representative examples:")
    for split, records in report["visualizations"].items():
        descriptions = [
            f"{item['sample_index']} ({item['selection_reason']}, "
            f"{item['object_count']} objects)"
            for item in records
        ]
        print(f"  - {split}: " + ", ".join(descriptions))

    print(f"\nAll annotation preservation checks valid: "
          f"{report['all_preservation_checks_valid']}")
    print(f"Ground-truth outputs saved to: {output_dir}")


def main() -> None:
    arguments = parse_arguments()
    if arguments.samples_per_split <= 0:
        raise ValueError("--samples-per-split must be greater than zero.")
    if arguments.box_width <= 0:
        raise ValueError("--box-width must be greater than zero.")
    if arguments.font_size <= 0:
        raise ValueError("--font-size must be greater than zero.")

    dataset_dict, _ = load_detection_dataset(arguments.cache_dir)
    datasets = build_detection_datasets(dataset_dict)
    transformed_datasets = datasets.as_dict()

    output_dir = resolve_exercise_path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preservation_checks = {
        split: validate_split_preservation(split, dataset)
        for split, dataset in transformed_datasets.items()
    }
    visualizations = {
        split: save_split_visualizations(
            split=split,
            transformed_dataset=dataset,
            output_dir=output_dir,
            samples_per_split=arguments.samples_per_split,
            seed=arguments.seed + split_index,
            box_width=arguments.box_width,
            font_path=arguments.font_path,
            font_size=arguments.font_size,
        )
        for split_index, (split, dataset) in enumerate(
            transformed_datasets.items()
        )
    }

    report = {
        "visualization_policy": {
            "box_format": "XYXY",
            "coordinate_type": "absolute pixels",
            "ground_truth_color": "red",
            "label_format": "<detector_label>: <GTSRB class name>",
            "saved_views": ["original", "transformed", "comparison"],
            "selection_policy": ["most-objects", "smallest-box", "empty"],
            "samples_per_split": arguments.samples_per_split,
            "font_path": (
                str(arguments.font_path.expanduser().resolve())
                if arguments.font_path is not None
                else None
            ),
            "font_size": arguments.font_size,
            "seed": arguments.seed,
        },
        "preservation_checks": preservation_checks,
        "visualizations": visualizations,
        "all_preservation_checks_valid": all(
            item["contract_valid"] for item in preservation_checks.values()
        ),
    }

    report_path = output_dir / "ground_truth_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_report(report, output_dir)
    print(f"Validation report saved to: {report_path}")


if __name__ == "__main__":
    main()
