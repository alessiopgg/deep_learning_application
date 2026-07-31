"""Validate Step 7 detection DataLoaders and custom collation."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from Exercise3.data_pipeline.adapter import validate_detection_target
from Exercise3.data_pipeline.loaders import (
    DEFAULT_EVALUATION_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    DEFAULT_TRAIN_BATCH_SIZE,
    DetectionDataLoaderBundle,
    build_detection_dataloaders,
)
from Exercise3.data_pipeline.loading import (
    DEFAULT_CACHE_DIR,
    load_detection_dataset,
    resolve_exercise_path,
)


DEFAULT_OUTPUT_PATH = Path("outputs/step_7/dataloader_validation.json")
REPRODUCIBILITY_PREVIEW_IMAGES = 12


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate train, validation and test DataLoaders for "
            "Exercise 3 object detection."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Hugging Face cache directory.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help="Training batch size. Default: 2.",
    )
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=DEFAULT_EVALUATION_BATCH_SIZE,
        help="Validation and test batch size. Default: 1.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="DataLoader subprocess count. Default: 0 for Windows safety.",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep worker processes alive between epochs; requires workers > 0.",
    )

    pin_group = parser.add_mutually_exclusive_group()
    pin_group.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="Force CUDA-pinned host memory.",
    )
    pin_group.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable CUDA-pinned host memory.",
    )
    parser.set_defaults(pin_memory=None)

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed controlling train shuffling and DataLoader workers.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSON report path relative to Exercise3 unless absolute.",
    )
    return parser.parse_args()


def _validate_batch_structure(
    images: Any,
    targets: Any,
    *,
    split_name: str,
    batch_index: int,
) -> None:
    if not isinstance(images, list):
        raise TypeError(
            f"{split_name} batch {batch_index}: images must be a list, "
            f"found {type(images).__name__}."
        )
    if not isinstance(targets, list):
        raise TypeError(
            f"{split_name} batch {batch_index}: targets must be a list, "
            f"found {type(targets).__name__}."
        )
    if len(images) != len(targets):
        raise ValueError(
            f"{split_name} batch {batch_index}: images and targets have "
            f"different lengths ({len(images)} != {len(targets)})."
        )
    if not images:
        raise ValueError(f"{split_name} batch {batch_index} is empty.")


def _validate_transformed_image(
    image: torch.Tensor,
    *,
    split_name: str,
    batch_index: int,
    sample_index: int,
) -> tuple[float, float]:
    if image.dtype != torch.float32:
        raise TypeError(
            f"{split_name} batch {batch_index} sample {sample_index}: "
            f"expected float32 image, found {image.dtype}."
        )
    if image.device.type != "cpu":
        raise ValueError(
            f"DataLoader must return CPU tensors before the training loop, "
            f"found device {image.device}."
        )
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(
            f"Expected image shape [3,H,W], found {tuple(image.shape)}."
        )

    value_min = float(image.min().item())
    value_max = float(image.max().item())
    if value_min < 0.0 or value_max > 1.0:
        raise ValueError(
            f"Image range must be within [0,1], found [{value_min}, {value_max}]."
        )
    return value_min, value_max


def validate_loader(
    *,
    split_name: str,
    loader: torch.utils.data.DataLoader,
    expected_dataset_size: int,
    configured_batch_size: int,
) -> dict[str, Any]:
    """Iterate through a complete loader and validate every returned sample."""
    image_ids: list[int] = []
    object_counts: list[int] = []
    batch_sizes: list[int] = []
    image_shapes: Counter[tuple[int, int, int]] = Counter()
    image_dtypes: Counter[str] = Counter()
    image_devices: Counter[str] = Counter()
    boxes_dtypes: Counter[str] = Counter()
    labels_dtypes: Counter[str] = Counter()
    pinned_images = 0
    empty_images = 0
    global_min = 1.0
    global_max = 0.0
    first_batch_preview: dict[str, Any] | None = None

    for batch_index, (images, targets) in enumerate(loader):
        _validate_batch_structure(
            images,
            targets,
            split_name=split_name,
            batch_index=batch_index,
        )
        batch_sizes.append(len(images))

        if len(images) > configured_batch_size:
            raise ValueError(
                f"{split_name} batch {batch_index} exceeds configured batch size."
            )

        for sample_index, (image, target) in enumerate(
            zip(images, targets, strict=True)
        ):
            validate_detection_target(image, target)
            value_min, value_max = _validate_transformed_image(
                image,
                split_name=split_name,
                batch_index=batch_index,
                sample_index=sample_index,
            )

            global_min = min(global_min, value_min)
            global_max = max(global_max, value_max)
            image_ids.append(int(target["image_id"]))
            object_count = int(target["boxes"].shape[0])
            object_counts.append(object_count)
            empty_images += int(object_count == 0)
            image_shapes[tuple(int(value) for value in image.shape)] += 1
            image_dtypes[str(image.dtype)] += 1
            image_devices[str(image.device)] += 1
            boxes_dtypes[str(target["boxes"].dtype)] += 1
            labels_dtypes[str(target["labels"].dtype)] += 1
            pinned_images += int(image.is_pinned())

        if first_batch_preview is None:
            first_batch_preview = {
                "batch_size": len(images),
                "image_ids": [int(target["image_id"]) for target in targets],
                "image_shapes": [
                    [int(value) for value in image.shape] for image in images
                ],
                "boxes_per_image": [
                    int(target["boxes"].shape[0]) for target in targets
                ],
                "labels_per_image": [
                    [int(value) for value in target["labels"].tolist()]
                    for target in targets
                ],
            }

    observed_images = len(image_ids)
    expected_batches = math.ceil(expected_dataset_size / configured_batch_size)
    if observed_images != expected_dataset_size:
        raise ValueError(
            f"Split '{split_name}' yielded {observed_images} images, expected "
            f"{expected_dataset_size}."
        )
    if len(set(image_ids)) != observed_images:
        raise ValueError(f"Split '{split_name}' yielded duplicate image IDs.")
    if len(batch_sizes) != expected_batches:
        raise ValueError(
            f"Split '{split_name}' yielded {len(batch_sizes)} batches, expected "
            f"{expected_batches}."
        )

    expected_last_batch_size = (
        expected_dataset_size % configured_batch_size or configured_batch_size
    )
    if batch_sizes[-1] != expected_last_batch_size:
        raise ValueError(
            f"Split '{split_name}' last batch has size {batch_sizes[-1]}, "
            f"expected {expected_last_batch_size}."
        )

    return {
        "dataset_size": expected_dataset_size,
        "configured_batch_size": configured_batch_size,
        "batch_count": len(batch_sizes),
        "expected_batch_count": expected_batches,
        "observed_batch_sizes": sorted(set(batch_sizes)),
        "last_batch_size": batch_sizes[-1],
        "images_seen": observed_images,
        "image_ids_unique": True,
        "total_objects": sum(object_counts),
        "empty_images": empty_images,
        "minimum_objects_per_image": min(object_counts, default=0),
        "maximum_objects_per_image": max(object_counts, default=0),
        "image_shapes": {
            str(shape): count for shape, count in sorted(image_shapes.items())
        },
        "image_dtypes": dict(sorted(image_dtypes.items())),
        "image_devices": dict(sorted(image_devices.items())),
        "boxes_dtypes": dict(sorted(boxes_dtypes.items())),
        "labels_dtypes": dict(sorted(labels_dtypes.items())),
        "image_value_range": {"min": global_min, "max": global_max},
        "pinned_images": pinned_images,
        "first_batch": first_batch_preview,
        "contract_valid": True,
    }


def _preview_train_order(
    bundle: DetectionDataLoaderBundle,
    limit: int,
) -> list[int]:
    image_ids: list[int] = []
    for _, targets in bundle.train:
        image_ids.extend(int(target["image_id"]) for target in targets)
        if len(image_ids) >= limit:
            return image_ids[:limit]
    return image_ids


def validate_train_shuffle_reproducibility(
    *,
    dataset_dict: Any,
    train_batch_size: int,
    evaluation_batch_size: int,
    num_workers: int,
    pin_memory: bool | None,
    persistent_workers: bool,
    seed: int,
) -> dict[str, Any]:
    """Confirm that two fresh loaders with the same seed start identically."""
    first_bundle = build_detection_dataloaders(
        dataset_dict,
        train_batch_size=train_batch_size,
        evaluation_batch_size=evaluation_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        seed=seed,
    )
    second_bundle = build_detection_dataloaders(
        dataset_dict,
        train_batch_size=train_batch_size,
        evaluation_batch_size=evaluation_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        seed=seed,
    )

    first_order = _preview_train_order(
        first_bundle,
        REPRODUCIBILITY_PREVIEW_IMAGES,
    )
    second_order = _preview_train_order(
        second_bundle,
        REPRODUCIBILITY_PREVIEW_IMAGES,
    )
    reproducible = first_order == second_order
    if not reproducible:
        raise RuntimeError(
            "Two train DataLoaders created with the same seed produced "
            "different initial sample orders."
        )

    return {
        "seed": seed,
        "preview_length": len(first_order),
        "first_order": first_order,
        "second_order": second_order,
        "same_seed_same_initial_order": True,
    }


def print_report(report: dict[str, Any]) -> None:
    settings = report["settings"]
    print("\n=== Exercise 3.3 - Step 7: detection DataLoaders ===")
    print(f"Train batch size: {settings['train_batch_size']}")
    print(f"Evaluation batch size: {settings['evaluation_batch_size']}")
    print(f"Number of workers: {settings['num_workers']}")
    print(f"Pin memory: {settings['pin_memory']}")
    print(f"Persistent workers: {settings['persistent_workers']}")
    print(f"Drop last: {settings['drop_last']}")
    print(f"Seed: {settings['seed']}")
    print("Batch structure: list[Tensor], list[dict]")

    print("\nPer split:")
    for split_name, split_report in report["splits"].items():
        print(
            f"  - {split_name}: {split_report['batch_count']} batches, "
            f"{split_report['images_seen']} images, "
            f"batch sizes {split_report['observed_batch_sizes']}, "
            f"{split_report['total_objects']} objects, "
            f"{split_report['empty_images']} empty images"
        )

    reproducibility = report["train_shuffle_reproducibility"]
    print("\nTrain shuffle reproducibility:")
    print(
        "  same seed -> same initial order: "
        f"{reproducibility['same_seed_same_initial_order']}"
    )
    print(f"  initial image IDs: {reproducibility['first_order']}")

    print("\nFirst batch previews:")
    for split_name, split_report in report["splits"].items():
        preview = split_report["first_batch"]
        print(
            f"  - {split_name}: batch_size={preview['batch_size']}, "
            f"image_ids={preview['image_ids']}, "
            f"shapes={preview['image_shapes']}, "
            f"boxes/image={preview['boxes_per_image']}"
        )

    print(f"\nAll DataLoader contracts valid: {report['all_contracts_valid']}")


def main() -> None:
    arguments = parse_arguments()
    dataset_dict, _ = load_detection_dataset(arguments.cache_dir)

    reproducibility_report = validate_train_shuffle_reproducibility(
        dataset_dict=dataset_dict,
        train_batch_size=arguments.train_batch_size,
        evaluation_batch_size=arguments.evaluation_batch_size,
        num_workers=arguments.num_workers,
        pin_memory=arguments.pin_memory,
        persistent_workers=arguments.persistent_workers,
        seed=arguments.seed,
    )

    bundle = build_detection_dataloaders(
        dataset_dict,
        train_batch_size=arguments.train_batch_size,
        evaluation_batch_size=arguments.evaluation_batch_size,
        num_workers=arguments.num_workers,
        pin_memory=arguments.pin_memory,
        persistent_workers=arguments.persistent_workers,
        seed=arguments.seed,
    )

    split_reports = {
        "train": validate_loader(
            split_name="train",
            loader=bundle.train,
            expected_dataset_size=len(bundle.datasets.train),
            configured_batch_size=bundle.settings.train_batch_size,
        ),
        "validation": validate_loader(
            split_name="validation",
            loader=bundle.validation,
            expected_dataset_size=len(bundle.datasets.validation),
            configured_batch_size=bundle.settings.evaluation_batch_size,
        ),
        "test": validate_loader(
            split_name="test",
            loader=bundle.test,
            expected_dataset_size=len(bundle.datasets.test),
            configured_batch_size=bundle.settings.evaluation_batch_size,
        ),
    }

    report = {
        "settings": bundle.settings.to_dict(),
        "batch_contract": {
            "images": "list[torch.Tensor], one [3,H,W] float32 CPU tensor per image",
            "targets": "list[dict], one variable-length detection target per image",
            "images_are_stacked": False,
            "targets_are_stacked": False,
            "device_transfer": "performed later by the training loop",
        },
        "splits": split_reports,
        "train_shuffle_reproducibility": reproducibility_report,
        "all_contracts_valid": all(
            split_report["contract_valid"]
            for split_report in split_reports.values()
        ),
    }

    output_path = resolve_exercise_path(arguments.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_report(report)
    print(f"\nDataLoader validation report saved to: {output_path}")


if __name__ == "__main__":
    main()
