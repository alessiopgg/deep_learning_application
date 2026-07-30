"""Loading and structural inspection of the traffic-sign detection dataset."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from pprint import pformat
from typing import Any

try:
    from datasets import Dataset, DatasetDict, load_dataset
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Exercise 3 requires Hugging Face 'datasets' version 3.x."
    ) from exc


DATASET_REPOSITORY = "keremberke/german-traffic-sign-detection"
DATASET_CONFIGURATION = "full"
IMAGE_FIELD = "image"
ANNOTATIONS_FIELD = "objects"
SOURCE_BBOX_FORMAT = "COCO xywh: [x_min, y_min, width, height]"

EXERCISE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = Path("../data/huggingface")
DEFAULT_REPORT_PATH = Path("outputs/step_2/dataset_inspection.json")


def resolve_exercise_path(path_value: str | Path) -> Path:
    """Resolve relative paths from the Exercise3 directory."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = EXERCISE_DIR / path
    return path.resolve()


def load_detection_dataset(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> tuple[DatasetDict, Path]:
    """Download the dataset or reuse the local Hugging Face cache."""
    resolved_cache_dir = resolve_exercise_path(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        DATASET_REPOSITORY,
        name=DATASET_CONFIGURATION,
        cache_dir=str(resolved_cache_dir),
    )

    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            "Expected load_dataset(...) to return a DatasetDict, "
            f"but received {type(dataset).__name__}."
        )

    return dataset, resolved_cache_dir


def _unwrap_feature(feature: Any) -> Any:
    """Remove nested List/Sequence wrappers from a Hugging Face feature."""
    current = feature
    visited_ids: set[int] = set()

    while hasattr(current, "feature") and id(current) not in visited_ids:
        visited_ids.add(id(current))
        current = current.feature

    return current


def extract_class_names(split_dataset: Dataset) -> list[str]:
    """Extract category names from the nested ClassLabel metadata."""
    objects_feature = split_dataset.features.get(ANNOTATIONS_FIELD)
    objects_feature = _unwrap_feature(objects_feature)

    if not isinstance(objects_feature, Mapping):
        return []

    category_feature = _unwrap_feature(objects_feature.get("category"))
    names = getattr(category_feature, "names", None)
    if names is None:
        return []

    return [str(name) for name in names]


def normalize_objects(objects: Any) -> dict[str, list[Any]]:
    """Convert the objects field to a dictionary of equally sized lists."""
    if isinstance(objects, Mapping):
        normalized = {}
        for key, value in objects.items():
            if isinstance(value, Sequence) and not isinstance(
                value,
                (str, bytes, bytearray),
            ):
                normalized[str(key)] = list(value)
            else:
                normalized[str(key)] = [value]
        return normalized

    if isinstance(objects, Sequence) and not isinstance(
        objects,
        (str, bytes, bytearray),
    ):
        normalized: dict[str, list[Any]] = {}
        for object_record in objects:
            if not isinstance(object_record, Mapping):
                raise TypeError("Each object annotation must be a mapping.")
            for key, value in object_record.items():
                normalized.setdefault(str(key), []).append(value)
        return normalized

    raise TypeError(
        "The objects field must be a mapping or a sequence of mappings."
    )


def count_objects(objects: Mapping[str, list[Any]]) -> int:
    """Check annotation-array lengths and return the number of objects."""
    if not objects:
        return 0

    lengths = {field: len(values) for field, values in objects.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "The annotation fields have inconsistent lengths: "
            f"{lengths}."
        )

    return next(iter(lengths.values()))


def inspect_dataset(
    dataset: DatasetDict,
    cache_dir: Path,
    split_name: str,
    sample_index: int,
    max_objects: int,
) -> dict[str, Any]:
    """Inspect splits, features, classes and one decoded sample."""
    if split_name not in dataset:
        raise KeyError(
            f"Unknown split '{split_name}'. Available splits: "
            f"{sorted(dataset.keys())}."
        )

    split_dataset = dataset[split_name]
    required_fields = {IMAGE_FIELD, ANNOTATIONS_FIELD}
    missing_fields = required_fields.difference(split_dataset.column_names)
    if missing_fields:
        raise KeyError(
            f"Missing required dataset fields: {sorted(missing_fields)}."
        )

    if not 0 <= sample_index < len(split_dataset):
        raise IndexError(
            f"Sample index {sample_index} is outside split '{split_name}' "
            f"with {len(split_dataset)} samples."
        )

    class_names = extract_class_names(split_dataset)
    sample = split_dataset[sample_index]
    image = sample[IMAGE_FIELD]
    objects = normalize_objects(sample[ANNOTATIONS_FIELD])
    object_count = count_objects(objects)

    boxes = [
        [float(coordinate) for coordinate in box]
        for box in objects.get("bbox", [])
    ]
    category_ids = [int(value) for value in objects.get("category", [])]

    if len(boxes) != object_count or len(category_ids) != object_count:
        raise ValueError(
            "Every object must have one bounding box and one category ID."
        )

    visible_category_ids = category_ids[:max_objects]
    visible_category_names = [
        class_names[category_id]
        if 0 <= category_id < len(class_names)
        else f"<unknown category {category_id}>"
        for category_id in visible_category_ids
    ]

    image_size = getattr(image, "size", None)
    if image_size is not None:
        image_size = [int(image_size[0]), int(image_size[1])]

    return {
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_configuration": DATASET_CONFIGURATION,
        "cache_dir": str(cache_dir),
        "split_sizes": {
            name: len(split) for name, split in dataset.items()
        },
        "inspected_split": split_name,
        "columns": list(split_dataset.column_names),
        "features": pformat(split_dataset.features, sort_dicts=False),
        "image_field": IMAGE_FIELD,
        "annotations_field": ANNOTATIONS_FIELD,
        "source_bbox_format": SOURCE_BBOX_FORMAT,
        "class_count": len(class_names) if class_names else None,
        "class_names": class_names,
        "sample": {
            "index": sample_index,
            "image_id": (
                int(sample["image_id"])
                if sample.get("image_id") is not None
                else None
            ),
            "image_type": type(image).__name__,
            "image_mode": getattr(image, "mode", None),
            "decoded_image_size_width_height": image_size,
            "declared_width": (
                int(sample["width"])
                if sample.get("width") is not None
                else None
            ),
            "declared_height": (
                int(sample["height"])
                if sample.get("height") is not None
                else None
            ),
            "object_count": object_count,
            "bounding_boxes_xywh": boxes[:max_objects],
            "category_ids": visible_category_ids,
            "category_names": visible_category_names,
            "objects_truncated": object_count > max_objects,
        },
    }


def print_dataset_report(report: Mapping[str, Any]) -> None:
    """Print the structural information required by step 2."""
    print("\n=== Exercise 3.3 - Step 2: dataset inspection ===")
    print(f"Dataset: {report['dataset_repository']}")
    print(f"Configuration: {report['dataset_configuration']}")
    print(f"Cache directory: {report['cache_dir']}")

    print("\nSplits:")
    for split_name, split_size in report["split_sizes"].items():
        print(f"  - {split_name}: {split_size} images")

    print(f"\nInspected split: {report['inspected_split']}")
    print(f"Columns: {report['columns']}")
    print(f"Image field: {report['image_field']}")
    print(f"Annotations field: {report['annotations_field']}")
    print(f"Bounding-box format: {report['source_bbox_format']}")

    print("\nFeatures:")
    print(report["features"])

    class_names = report["class_names"]
    if class_names:
        print(f"\nClasses ({len(class_names)}):")
        for category_id, category_name in enumerate(class_names):
            print(f"  {category_id:>2}: {category_name}")
    else:
        print("\nClass names are not available in the feature metadata.")

    sample = report["sample"]
    print("\nSample:")
    print(f"  index: {sample['index']}")
    print(f"  image_id: {sample['image_id']}")
    print(f"  image type: {sample['image_type']}")
    print(f"  image mode: {sample['image_mode']}")
    print(
        "  decoded size (W, H): "
        f"{sample['decoded_image_size_width_height']}"
    )
    print(
        "  declared size (W, H): "
        f"({sample['declared_width']}, {sample['declared_height']})"
    )
    print(f"  number of objects: {sample['object_count']}")
    print(f"  boxes xywh: {sample['bounding_boxes_xywh']}")
    print(f"  category IDs: {sample['category_ids']}")
    print(f"  category names: {sample['category_names']}")
    if sample["objects_truncated"]:
        print("  note: printed annotations were truncated.")


def save_dataset_report(
    report: Mapping[str, Any],
    output_path: str | Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Save the dataset inspection as JSON."""
    resolved_output_path = resolve_exercise_path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return resolved_output_path
