"""Loading and structural inspection for the traffic-sign detection dataset."""

from __future__ import annotations

from Exercise3.paths import EXERCISE_DIR

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

try:
    import datasets
    from datasets import Dataset, DatasetDict, load_dataset
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The Hugging Face 'datasets' package is required for Exercise 3. "
        "Install datasets==3.6.0 before running this script."
    ) from exc


DATASET_REPOSITORY = "keremberke/german-traffic-sign-detection"
DATASET_CONFIGURATION = "full"
DATASET_REVISION = "a549a284a1fefdc761ad459ee85f50c5ad8138ef"
IMAGE_FIELD = "image"
ANNOTATIONS_FIELD = "objects"
SOURCE_BBOX_FORMAT = "COCO xywh: [x_min, y_min, width, height]"

DEFAULT_CACHE_DIR = Path("../data/huggingface")
DEFAULT_REPORT_PATH = Path("outputs/step_2/dataset_inspection.json")


@dataclass(frozen=True)
class SampleInspection:
    """Serializable description of one dataset sample."""

    split: str
    index: int
    image_id: int | str | None
    image_type: str
    image_mode: str | None
    image_size: tuple[int, int] | None
    declared_width: int | None
    declared_height: int | None
    object_count: int
    bounding_boxes_xywh: list[list[float]]
    category_ids: list[int]
    category_names: list[str]
    objects_truncated: bool


@dataclass(frozen=True)
class DatasetInspection:
    """Serializable structural report for the Hugging Face dataset."""

    dataset_repository: str
    dataset_configuration: str
    dataset_revision: str
    datasets_version: str
    cache_dir: str
    split_sizes: dict[str, int]
    inspected_split: str
    columns: list[str]
    features: str
    image_field: str
    annotations_field: str
    source_bbox_format: str
    class_count: int | None
    class_names: list[str]
    sample: SampleInspection

    def to_dict(self) -> dict[str, Any]:
        """Return the report as JSON-compatible built-in objects."""
        return asdict(self)


def resolve_exercise_path(path_value: str | Path) -> Path:
    """Resolve relative paths from the Exercise3 directory."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = EXERCISE_DIR / path
    return path.resolve()


def _validate_datasets_version() -> None:
    """Reject versions that no longer execute this legacy dataset script."""
    version_text = datasets.__version__
    try:
        major_version = int(version_text.split(".", maxsplit=1)[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Could not parse Hugging Face datasets version: {version_text!r}."
        ) from exc

    if major_version >= 4:
        raise RuntimeError(
            "This dataset is distributed through a legacy loading script, "
            f"which is not supported by datasets {version_text}. "
            "Install the laboratory-compatible version with: "
            "python -m pip install datasets==3.6.0"
        )


def load_detection_dataset(
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> tuple[DatasetDict, Path]:
    """Download or reuse the official train, validation and test splits."""
    _validate_datasets_version()

    resolved_cache_dir = resolve_exercise_path(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        DATASET_REPOSITORY,
        name=DATASET_CONFIGURATION,
        revision=DATASET_REVISION,
        trust_remote_code=True,
        cache_dir=str(resolved_cache_dir),
    )

    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            "Expected load_dataset(...) to return a DatasetDict, "
            f"but received {type(dataset).__name__}."
        )

    return dataset, resolved_cache_dir


def _unwrap_feature(feature: Any) -> Any:
    """Remove List/Sequence wrappers until the underlying feature is reached."""
    current = feature
    visited_ids: set[int] = set()

    while hasattr(current, "feature") and id(current) not in visited_ids:
        visited_ids.add(id(current))
        current = current.feature

    return current


def _get_objects_feature_mapping(split_dataset: Dataset) -> Mapping[str, Any] | None:
    """Return the nested feature mapping stored under the objects field."""
    if ANNOTATIONS_FIELD not in split_dataset.features:
        return None

    feature = _unwrap_feature(split_dataset.features[ANNOTATIONS_FIELD])
    if isinstance(feature, Mapping):
        return feature
    return None


def extract_class_names(split_dataset: Dataset) -> list[str]:
    """Read category names from the nested ClassLabel metadata when available."""
    objects_features = _get_objects_feature_mapping(split_dataset)
    if objects_features is None or "category" not in objects_features:
        return []

    category_feature = _unwrap_feature(objects_features["category"])
    names = getattr(category_feature, "names", None)
    if names is None:
        return []

    return [str(name) for name in names]


def normalize_objects(objects: Any) -> dict[str, list[Any]]:
    """Normalize object annotations to a dictionary of Python lists."""
    if isinstance(objects, Mapping):
        normalized: dict[str, list[Any]] = {}
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
        normalized = {}
        for object_record in objects:
            if not isinstance(object_record, Mapping):
                raise TypeError(
                    "Each object annotation must be a mapping, but found "
                    f"{type(object_record).__name__}."
                )
            for key, value in object_record.items():
                normalized.setdefault(str(key), []).append(value)
        return normalized

    raise TypeError(
        "The objects field must be a mapping or a sequence of mappings, "
        f"but received {type(objects).__name__}."
    )


def annotation_field_lengths(objects: Mapping[str, list[Any]]) -> dict[str, int]:
    """Return the number of entries stored in every annotation field."""
    return {field: len(values) for field, values in objects.items()}


def validate_object_field_lengths(objects: Mapping[str, list[Any]]) -> int:
    """Validate that all annotation arrays describe the same object count."""
    if not objects:
        return 0

    lengths = annotation_field_lengths(objects)
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(
            "Inconsistent annotation lengths inside the objects field: "
            f"{lengths}."
        )

    return next(iter(unique_lengths))


def _image_size(image: Any) -> tuple[int, int] | None:
    """Return a width-height pair when the decoded image exposes one."""
    size = getattr(image, "size", None)
    if (
            isinstance(size, Sequence)
            and not isinstance(size, (str, bytes, bytearray))
            and len(size) == 2
    ):
        return int(size[0]), int(size[1])
    return None


def _resolve_category_name(
        category_id: int,
        class_names: Sequence[str],
) -> str:
    """Convert a category ID to a name without hiding invalid identifiers."""
    if 0 <= category_id < len(class_names):
        return class_names[category_id]
    return f"<unknown category {category_id}>"


def inspect_sample(
        split_dataset: Dataset,
        split_name: str,
        sample_index: int,
        class_names: Sequence[str],
        max_objects: int = 10,
) -> SampleInspection:
    """Inspect one decoded image and its object annotations."""
    if not 0 <= sample_index < len(split_dataset):
        raise IndexError(
            f"Sample index {sample_index} is outside split '{split_name}' "
            f"with {len(split_dataset)} samples."
        )

    sample = split_dataset[sample_index]
    missing_fields = {
        field
        for field in (IMAGE_FIELD, ANNOTATIONS_FIELD)
        if field not in sample
    }
    if missing_fields:
        raise KeyError(
            "The selected sample is missing required field(s): "
            f"{sorted(missing_fields)}. Available fields: {sorted(sample)}."
        )

    image = sample[IMAGE_FIELD]
    objects = normalize_objects(sample[ANNOTATIONS_FIELD])
    object_count = validate_object_field_lengths(objects)

    boxes = [
        [float(value) for value in box]
        for box in objects.get("bbox", [])
    ]
    category_ids = [int(value) for value in objects.get("category", [])]

    if len(boxes) != object_count:
        raise ValueError(
            "The bbox field does not contain one box for each object: "
            f"objects={object_count}, boxes={len(boxes)}."
        )
    if len(category_ids) != object_count:
        raise ValueError(
            "The category field does not contain one label for each object: "
            f"objects={object_count}, labels={len(category_ids)}."
        )

    visible_boxes = boxes[:max_objects]
    visible_category_ids = category_ids[:max_objects]
    visible_category_names = [
        _resolve_category_name(category_id, class_names)
        for category_id in visible_category_ids
    ]

    return SampleInspection(
        split=split_name,
        index=sample_index,
        image_id=sample.get("image_id"),
        image_type=type(image).__name__,
        image_mode=getattr(image, "mode", None),
        image_size=_image_size(image),
        declared_width=(
            int(sample["width"]) if sample.get("width") is not None else None
        ),
        declared_height=(
            int(sample["height"]) if sample.get("height") is not None else None
        ),
        object_count=object_count,
        bounding_boxes_xywh=visible_boxes,
        category_ids=visible_category_ids,
        category_names=visible_category_names,
        objects_truncated=object_count > max_objects,
    )


def inspect_dataset(
        dataset: DatasetDict,
        cache_dir: Path,
        split_name: str = "train",
        sample_index: int = 0,
        max_objects: int = 10,
) -> DatasetInspection:
    """Build the structural report required by Exercise 3, step 2."""
    if split_name not in dataset:
        raise KeyError(
            f"Unknown split '{split_name}'. Available splits: "
            f"{sorted(dataset.keys())}."
        )

    split_dataset = dataset[split_name]
    missing_columns = {
        field
        for field in (IMAGE_FIELD, ANNOTATIONS_FIELD)
        if field not in split_dataset.column_names
    }
    if missing_columns:
        raise KeyError(
            "The dataset schema is missing required field(s): "
            f"{sorted(missing_columns)}. Available columns: "
            f"{split_dataset.column_names}."
        )

    class_names = extract_class_names(split_dataset)
    sample = inspect_sample(
        split_dataset=split_dataset,
        split_name=split_name,
        sample_index=sample_index,
        class_names=class_names,
        max_objects=max_objects,
    )

    return DatasetInspection(
        dataset_repository=DATASET_REPOSITORY,
        dataset_configuration=DATASET_CONFIGURATION,
        dataset_revision=DATASET_REVISION,
        datasets_version=datasets.__version__,
        cache_dir=str(cache_dir),
        split_sizes={name: len(split) for name, split in dataset.items()},
        inspected_split=split_name,
        columns=list(split_dataset.column_names),
        features=pformat(split_dataset.features, sort_dicts=False),
        image_field=IMAGE_FIELD,
        annotations_field=ANNOTATIONS_FIELD,
        source_bbox_format=SOURCE_BBOX_FORMAT,
        class_count=len(class_names) if class_names else None,
        class_names=class_names,
        sample=sample,
    )


def print_dataset_inspection(report: DatasetInspection) -> None:
    """Print a readable summary without inventing values."""
    print("\n=== Exercise 3.3 - Step 2: dataset inspection ===")
    print(f"Dataset: {report.dataset_repository}")
    print(f"Configuration: {report.dataset_configuration}")
    print(f"Revision: {report.dataset_revision}")
    print(f"datasets version: {report.datasets_version}")
    print(f"Cache directory: {report.cache_dir}")

    print("\nSplits:")
    for split_name, split_size in report.split_sizes.items():
        print(f"  - {split_name}: {split_size} images")

    print(f"\nInspected split: {report.inspected_split}")
    print(f"Columns: {report.columns}")
    print(f"Image field: {report.image_field}")
    print(f"Annotations field: {report.annotations_field}")
    print(f"Bounding-box format: {report.source_bbox_format}")

    print("\nFeatures:")
    print(report.features)

    if report.class_names:
        print(f"\nClasses ({len(report.class_names)}):")
        for category_id, category_name in enumerate(report.class_names):
            print(f"  {category_id:>2}: {category_name}")
    else:
        print("\nClass names are not exposed as ClassLabel metadata.")

    sample = report.sample
    print("\nSample:")
    print(f"  split/index: {sample.split}[{sample.index}]")
    print(f"  image_id: {sample.image_id}")
    print(f"  decoded image type: {sample.image_type}")
    print(f"  decoded image mode: {sample.image_mode}")
    print(f"  decoded image size (W, H): {sample.image_size}")
    print(
        "  declared image size (W, H): "
        f"({sample.declared_width}, {sample.declared_height})"
    )
    print(f"  number of objects: {sample.object_count}")
    print(f"  boxes in source xywh format: {sample.bounding_boxes_xywh}")
    print(f"  category IDs: {sample.category_ids}")
    print(f"  category names: {sample.category_names}")
    if sample.objects_truncated:
        print("  note: the printed annotations were truncated.")


def save_dataset_inspection(
        report: DatasetInspection,
        output_path: str | Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Save the inspection report as indented UTF-8 JSON."""
    resolved_output_path = resolve_exercise_path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return resolved_output_path