"""Exploratory data analysis for the traffic-sign detection dataset.

This module does not modify annotations and does not build a PyTorch dataset.
It records the source COCO-style xywh boxes, derives diagnostic quantities,
checks annotation consistency, and saves tables, summaries, plots and examples.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from Exercise3.data_pipeline.loading import (
    ANNOTATIONS_FIELD,
    DATASET_CONFIGURATION,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    DEFAULT_CACHE_DIR,
    IMAGE_FIELD,
    annotation_field_lengths,
    extract_class_names,
    load_detection_dataset,
    normalize_objects,
    resolve_exercise_path,
)


DEFAULT_OUTPUT_DIR = Path("outputs/step_3")
DEFAULT_EXAMPLES_PER_SPLIT = 3
DEFAULT_SEED = 42
COCO_SMALL_MAX_AREA = 32**2
COCO_MEDIUM_MAX_AREA = 96**2
DUPLICATE_ROUND_DECIMALS = 4
BOUNDARY_TOLERANCE = 1e-6


IMAGE_COLUMNS = [
    "split",
    "sample_index",
    "image_id",
    "width",
    "height",
    "image_area",
    "image_aspect_ratio",
    "object_count",
    "has_objects",
    "annotation_lengths_consistent",
    "invalid_box_count",
    "degenerate_box_count",
    "nonfinite_box_count",
    "out_of_bounds_box_count",
    "invalid_category_count",
    "exact_duplicate_box_count",
    "duplicate_geometry_box_count",
    "area_mismatch_count",
]

BOX_COLUMNS = [
    "split",
    "sample_index",
    "image_id",
    "object_index",
    "annotation_id",
    "category_id",
    "category_name",
    "x_min",
    "y_min",
    "box_width",
    "box_height",
    "x_max",
    "y_max",
    "declared_area",
    "computed_area",
    "absolute_area_difference",
    "relative_area_difference",
    "area_mismatch",
    "relative_area",
    "box_aspect_ratio",
    "scale_coco",
    "bbox_structure_valid",
    "coordinates_finite",
    "degenerate",
    "out_of_bounds",
    "category_valid",
    "valid_box",
    "exact_duplicate",
    "duplicate_geometry",
]


def parse_arguments() -> argparse.Namespace:
    """Parse EDA-specific command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Run structural and statistical EDA on the German traffic-sign "
            "detection dataset."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=(
            "Hugging Face cache directory. Relative paths are resolved from "
            "the Exercise3 directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "EDA output directory. Relative paths are resolved from the "
            "Exercise3 directory."
        ),
    )
    parser.add_argument(
        "--examples-per-split",
        type=int,
        default=DEFAULT_EXAMPLES_PER_SPLIT,
        help="Number of annotated ground-truth examples saved per split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used only for reproducible example selection.",
    )
    return parser.parse_args()


def _sequence_value(
    values: Sequence[Any] | None,
    index: int,
    default: Any = None,
) -> Any:
    """Read one value from a possibly missing or shorter annotation field."""
    if values is None or index >= len(values):
        return default
    return values[index]


def _optional_int(value: Any) -> int | None:
    """Convert a scalar to int while preserving missing values."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_float(value: Any) -> float:
    """Convert a scalar to float, returning NaN for malformed values."""
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan


def _parse_bbox(value: Any) -> tuple[float, float, float, float, bool]:
    """Parse one source xywh box without silently correcting malformed data."""
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return math.nan, math.nan, math.nan, math.nan, False
    if len(value) != 4:
        return math.nan, math.nan, math.nan, math.nan, False

    x_min, y_min, width, height = (_optional_float(item) for item in value)
    return x_min, y_min, width, height, True


def _box_scale(computed_area: float) -> str | None:
    """Classify a positive box area using the standard COCO area ranges."""
    if not math.isfinite(computed_area) or computed_area <= 0:
        return None
    if computed_area < COCO_SMALL_MAX_AREA:
        return "small"
    if computed_area < COCO_MEDIUM_MAX_AREA:
        return "medium"
    return "large"


def _category_name(category_id: int | None, class_names: Sequence[str]) -> str:
    """Resolve a category ID without hiding invalid or missing identifiers."""
    if category_id is None:
        return "<missing category>"
    if 0 <= category_id < len(class_names):
        return class_names[category_id]
    return f"<unknown category {category_id}>"


def _annotation_count(objects: Mapping[str, list[Any]]) -> tuple[int, bool]:
    """Return maximum field length and whether all field lengths agree."""
    lengths = annotation_field_lengths(objects)
    if not lengths:
        return 0, True
    return max(lengths.values()), len(set(lengths.values())) == 1


def _duplicate_keys(
    category_id: int | None,
    x_min: float,
    y_min: float,
    width: float,
    height: float,
) -> tuple[tuple[Any, ...] | None, tuple[Any, ...] | None]:
    """Build exact-label and geometry-only keys for duplicate checks."""
    values = (x_min, y_min, width, height)
    if not all(math.isfinite(value) for value in values):
        return None, None

    geometry_key = tuple(
        round(value, DUPLICATE_ROUND_DECIMALS) for value in values
    )
    exact_key = (category_id, *geometry_key)
    return exact_key, geometry_key


def _build_box_record(
    *,
    split_name: str,
    sample_index: int,
    image_id: int | str | None,
    image_width: int,
    image_height: int,
    object_index: int,
    objects: Mapping[str, list[Any]],
    class_names: Sequence[str],
) -> tuple[dict[str, Any], tuple[Any, ...] | None, tuple[Any, ...] | None]:
    """Create one box row and the keys used for duplicate detection."""
    annotation_id = _optional_int(
        _sequence_value(objects.get("id"), object_index)
    )
    category_id = _optional_int(
        _sequence_value(objects.get("category"), object_index)
    )
    declared_area = _optional_float(
        _sequence_value(objects.get("area"), object_index)
    )

    raw_bbox = _sequence_value(objects.get("bbox"), object_index)
    x_min, y_min, width, height, bbox_structure_valid = _parse_bbox(raw_bbox)

    coordinates = (x_min, y_min, width, height)
    coordinates_finite = bbox_structure_valid and all(
        math.isfinite(value) for value in coordinates
    )
    degenerate = coordinates_finite and (width <= 0 or height <= 0)

    if coordinates_finite:
        x_max = x_min + width
        y_max = y_min + height
        computed_area = width * height
    else:
        x_max = math.nan
        y_max = math.nan
        computed_area = math.nan

    out_of_bounds = coordinates_finite and (
        x_min < -BOUNDARY_TOLERANCE
        or y_min < -BOUNDARY_TOLERANCE
        or x_max > image_width + BOUNDARY_TOLERANCE
        or y_max > image_height + BOUNDARY_TOLERANCE
    )
    category_valid = (
        category_id is not None and 0 <= category_id < len(class_names)
    )

    if math.isfinite(declared_area) and math.isfinite(computed_area):
        absolute_area_difference = abs(declared_area - computed_area)
        denominator = max(abs(computed_area), BOUNDARY_TOLERANCE)
        relative_area_difference = absolute_area_difference / denominator
        area_mismatch = (
            absolute_area_difference > max(1.0, 0.01 * denominator)
        )
    else:
        absolute_area_difference = math.nan
        relative_area_difference = math.nan
        area_mismatch = False

    image_area = image_width * image_height
    relative_area = (
        computed_area / image_area
        if math.isfinite(computed_area) and image_area > 0
        else math.nan
    )
    box_aspect_ratio = (
        width / height
        if coordinates_finite and height > 0
        else math.nan
    )

    valid_box = (
        bbox_structure_valid
        and coordinates_finite
        and not degenerate
        and not out_of_bounds
        and category_valid
    )

    exact_key, geometry_key = _duplicate_keys(
        category_id=category_id,
        x_min=x_min,
        y_min=y_min,
        width=width,
        height=height,
    )

    return (
        {
            "split": split_name,
            "sample_index": sample_index,
            "image_id": image_id,
            "object_index": object_index,
            "annotation_id": annotation_id,
            "category_id": category_id,
            "category_name": _category_name(category_id, class_names),
            "x_min": x_min,
            "y_min": y_min,
            "box_width": width,
            "box_height": height,
            "x_max": x_max,
            "y_max": y_max,
            "declared_area": declared_area,
            "computed_area": computed_area,
            "absolute_area_difference": absolute_area_difference,
            "relative_area_difference": relative_area_difference,
            "area_mismatch": area_mismatch,
            "relative_area": relative_area,
            "box_aspect_ratio": box_aspect_ratio,
            "scale_coco": _box_scale(computed_area),
            "bbox_structure_valid": bbox_structure_valid,
            "coordinates_finite": coordinates_finite,
            "degenerate": degenerate,
            "out_of_bounds": out_of_bounds,
            "category_valid": category_valid,
            "valid_box": valid_box,
            "exact_duplicate": False,
            "duplicate_geometry": False,
        },
        exact_key,
        geometry_key,
    )


def collect_eda_tables(
    dataset: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Collect image-, box- and class-level tables for all official splits."""
    if "train" not in dataset:
        raise KeyError("The dataset must expose a train split to read classes.")

    class_names = extract_class_names(dataset["train"])
    if not class_names:
        raise RuntimeError(
            "Class names could not be extracted from the dataset metadata."
        )

    image_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []

    for split_name, split_dataset in dataset.items():
        required_columns = {
            "image_id",
            "width",
            "height",
            ANNOTATIONS_FIELD,
        }
        missing_columns = required_columns.difference(split_dataset.column_names)
        if missing_columns:
            raise KeyError(
                f"Split '{split_name}' is missing columns: "
                f"{sorted(missing_columns)}."
            )

        metadata_columns = [
            "image_id",
            "width",
            "height",
            ANNOTATIONS_FIELD,
        ]
        metadata_dataset = split_dataset.select_columns(metadata_columns)

        for sample_index, sample in enumerate(metadata_dataset):
            image_id = sample.get("image_id")
            image_width = int(sample["width"])
            image_height = int(sample["height"])
            objects = normalize_objects(sample[ANNOTATIONS_FIELD])
            object_count, lengths_consistent = _annotation_count(objects)

            local_box_rows: list[dict[str, Any]] = []
            exact_keys: list[tuple[Any, ...] | None] = []
            geometry_keys: list[tuple[Any, ...] | None] = []

            for object_index in range(object_count):
                box_record, exact_key, geometry_key = _build_box_record(
                    split_name=split_name,
                    sample_index=sample_index,
                    image_id=image_id,
                    image_width=image_width,
                    image_height=image_height,
                    object_index=object_index,
                    objects=objects,
                    class_names=class_names,
                )
                local_box_rows.append(box_record)
                exact_keys.append(exact_key)
                geometry_keys.append(geometry_key)

            exact_counts = Counter(key for key in exact_keys if key is not None)
            geometry_counts = Counter(
                key for key in geometry_keys if key is not None
            )

            for local_index, box_record in enumerate(local_box_rows):
                exact_key = exact_keys[local_index]
                geometry_key = geometry_keys[local_index]
                box_record["exact_duplicate"] = (
                    exact_key is not None and exact_counts[exact_key] > 1
                )
                box_record["duplicate_geometry"] = (
                    geometry_key is not None
                    and geometry_counts[geometry_key] > 1
                )
                box_rows.append(box_record)

            image_area = image_width * image_height
            image_rows.append(
                {
                    "split": split_name,
                    "sample_index": sample_index,
                    "image_id": image_id,
                    "width": image_width,
                    "height": image_height,
                    "image_area": image_area,
                    "image_aspect_ratio": (
                        image_width / image_height
                        if image_height > 0
                        else math.nan
                    ),
                    "object_count": object_count,
                    "has_objects": object_count > 0,
                    "annotation_lengths_consistent": lengths_consistent,
                    "invalid_box_count": sum(
                        not bool(row["valid_box"]) for row in local_box_rows
                    ),
                    "degenerate_box_count": sum(
                        bool(row["degenerate"]) for row in local_box_rows
                    ),
                    "nonfinite_box_count": sum(
                        not bool(row["coordinates_finite"])
                        for row in local_box_rows
                    ),
                    "out_of_bounds_box_count": sum(
                        bool(row["out_of_bounds"]) for row in local_box_rows
                    ),
                    "invalid_category_count": sum(
                        not bool(row["category_valid"])
                        for row in local_box_rows
                    ),
                    "exact_duplicate_box_count": sum(
                        bool(row["exact_duplicate"])
                        for row in local_box_rows
                    ),
                    "duplicate_geometry_box_count": sum(
                        bool(row["duplicate_geometry"])
                        for row in local_box_rows
                    ),
                    "area_mismatch_count": sum(
                        bool(row["area_mismatch"])
                        for row in local_box_rows
                    ),
                }
            )

    images_df = pd.DataFrame(image_rows, columns=IMAGE_COLUMNS)
    boxes_df = pd.DataFrame(box_rows, columns=BOX_COLUMNS)
    class_distribution_df = build_class_distribution(
        images_df=images_df,
        boxes_df=boxes_df,
        class_names=class_names,
        split_names=list(dataset.keys()),
    )
    return images_df, boxes_df, class_distribution_df, class_names


def build_class_distribution(
    *,
    images_df: pd.DataFrame,
    boxes_df: pd.DataFrame,
    class_names: Sequence[str],
    split_names: Sequence[str],
) -> pd.DataFrame:
    """Build a zero-complete class distribution for every split."""
    rows: list[dict[str, Any]] = []

    for split_name in split_names:
        split_boxes = boxes_df[
            (boxes_df["split"] == split_name)
            & boxes_df["category_valid"].fillna(False)
        ]
        total_objects = int(len(split_boxes))

        for category_id, category_name in enumerate(class_names):
            class_boxes = split_boxes[
                split_boxes["category_id"] == category_id
            ]
            object_count = int(len(class_boxes))
            image_count = int(
                class_boxes[["sample_index"]].drop_duplicates().shape[0]
            )
            valid_box_count = int(class_boxes["valid_box"].fillna(False).sum())

            rows.append(
                {
                    "split": split_name,
                    "category_id": category_id,
                    "category_name": category_name,
                    "object_count": object_count,
                    "image_count": image_count,
                    "valid_box_count": valid_box_count,
                    "object_percentage": (
                        100.0 * object_count / total_objects
                        if total_objects > 0
                        else 0.0
                    ),
                    "split_image_count": int(
                        (images_df["split"] == split_name).sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def _finite_series(series: pd.Series) -> pd.Series:
    """Return numeric finite values only."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[np.isfinite(numeric)]


def describe_series(series: pd.Series) -> dict[str, float | int | None]:
    """Create compact JSON-safe descriptive statistics."""
    values = _finite_series(series)
    if values.empty:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p95": None,
            "max": None,
        }

    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def _class_imbalance_summary(
    class_distribution_df: pd.DataFrame,
    split_name: str,
) -> dict[str, Any]:
    """Summarize observed object imbalance without discarding zero classes."""
    split_distribution = class_distribution_df[
        class_distribution_df["split"] == split_name
    ].copy()
    nonzero = split_distribution[split_distribution["object_count"] > 0]
    zero_classes = split_distribution[
        split_distribution["object_count"] == 0
    ]

    if nonzero.empty:
        return {
            "observed_class_count": 0,
            "zero_object_classes": zero_classes["category_name"].tolist(),
            "least_frequent": None,
            "most_frequent": None,
            "max_to_min_nonzero_ratio": None,
        }

    least = nonzero.loc[nonzero["object_count"].idxmin()]
    most = nonzero.loc[nonzero["object_count"].idxmax()]
    return {
        "observed_class_count": int(len(nonzero)),
        "zero_object_classes": zero_classes["category_name"].tolist(),
        "least_frequent": {
            "category_id": int(least["category_id"]),
            "category_name": str(least["category_name"]),
            "object_count": int(least["object_count"]),
        },
        "most_frequent": {
            "category_id": int(most["category_id"]),
            "category_name": str(most["category_name"]),
            "object_count": int(most["object_count"]),
        },
        "max_to_min_nonzero_ratio": float(
            most["object_count"] / least["object_count"]
        ),
    }


def build_summary(
    *,
    images_df: pd.DataFrame,
    boxes_df: pd.DataFrame,
    class_distribution_df: pd.DataFrame,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Build the machine-readable EDA report."""
    summary: dict[str, Any] = {
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "configuration": DATASET_CONFIGURATION,
            "revision": DATASET_REVISION,
            "class_count": len(class_names),
            "class_names": list(class_names),
            "source_bbox_format": "xywh",
        },
        "scale_definition": {
            "convention": "COCO absolute pixel area",
            "small": f"area < {COCO_SMALL_MAX_AREA}",
            "medium": (
                f"{COCO_SMALL_MAX_AREA} <= area < "
                f"{COCO_MEDIUM_MAX_AREA}"
            ),
            "large": f"area >= {COCO_MEDIUM_MAX_AREA}",
        },
        "totals": {
            "images": int(len(images_df)),
            "objects": int(len(boxes_df)),
            "images_without_objects": int((~images_df["has_objects"]).sum()),
            "images_with_inconsistent_annotation_lengths": int(
                (~images_df["annotation_lengths_consistent"]).sum()
            ),
            "valid_boxes": int(boxes_df["valid_box"].fillna(False).sum()),
            "invalid_boxes": int((~boxes_df["valid_box"].fillna(False)).sum()),
            "degenerate_boxes": int(boxes_df["degenerate"].fillna(False).sum()),
            "nonfinite_boxes": int(
                (~boxes_df["coordinates_finite"].fillna(False)).sum()
            ),
            "out_of_bounds_boxes": int(
                boxes_df["out_of_bounds"].fillna(False).sum()
            ),
            "invalid_categories": int(
                (~boxes_df["category_valid"].fillna(False)).sum()
            ),
            "exact_duplicate_annotations": int(
                boxes_df["exact_duplicate"].fillna(False).sum()
            ),
            "duplicate_geometry_annotations": int(
                boxes_df["duplicate_geometry"].fillna(False).sum()
            ),
            "area_mismatches": int(
                boxes_df["area_mismatch"].fillna(False).sum()
            ),
        },
        "splits": {},
        "image_statistics": {
            "width": describe_series(images_df["width"]),
            "height": describe_series(images_df["height"]),
            "aspect_ratio": describe_series(images_df["image_aspect_ratio"]),
            "objects_per_image": describe_series(images_df["object_count"]),
        },
        "box_statistics": {
            "width": describe_series(boxes_df["box_width"]),
            "height": describe_series(boxes_df["box_height"]),
            "absolute_area": describe_series(boxes_df["computed_area"]),
            "relative_area": describe_series(boxes_df["relative_area"]),
            "aspect_ratio": describe_series(boxes_df["box_aspect_ratio"]),
        },
    }

    for split_name in images_df["split"].drop_duplicates().tolist():
        split_images = images_df[images_df["split"] == split_name]
        split_boxes = boxes_df[boxes_df["split"] == split_name]
        scale_counts = (
            split_boxes["scale_coco"].value_counts(dropna=False).to_dict()
        )
        summary["splits"][split_name] = {
            "images": int(len(split_images)),
            "objects": int(len(split_boxes)),
            "images_without_objects": int(
                (~split_images["has_objects"]).sum()
            ),
            "objects_per_image": describe_series(split_images["object_count"]),
            "valid_boxes": int(split_boxes["valid_box"].fillna(False).sum()),
            "invalid_boxes": int(
                (~split_boxes["valid_box"].fillna(False)).sum()
            ),
            "scale_counts": {
                str(key): int(value) for key, value in scale_counts.items()
            },
            "class_imbalance": _class_imbalance_summary(
                class_distribution_df,
                split_name,
            ),
        }

    return summary


def save_tables(
    *,
    images_df: pd.DataFrame,
    boxes_df: pd.DataFrame,
    class_distribution_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Save complete tables and focused diagnostic subsets."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "images": tables_dir / "images.csv",
        "boxes": tables_dir / "boxes.csv",
        "class_distribution": tables_dir / "class_distribution.csv",
        "invalid_boxes": tables_dir / "invalid_boxes.csv",
        "duplicate_boxes": tables_dir / "duplicate_boxes.csv",
        "empty_images": tables_dir / "empty_images.csv",
    }

    images_df.to_csv(paths["images"], index=False)
    boxes_df.to_csv(paths["boxes"], index=False)
    class_distribution_df.to_csv(paths["class_distribution"], index=False)
    boxes_df[~boxes_df["valid_box"].fillna(False)].to_csv(
        paths["invalid_boxes"],
        index=False,
    )
    boxes_df[
        boxes_df["exact_duplicate"].fillna(False)
        | boxes_df["duplicate_geometry"].fillna(False)
    ].to_csv(paths["duplicate_boxes"], index=False)
    images_df[~images_df["has_objects"]].to_csv(
        paths["empty_images"],
        index=False,
    )
    return paths


def _save_figure(figure: plt.Figure, path: Path) -> None:
    """Save and close one Matplotlib figure."""
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_split_sizes(images_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot image counts for the official splits."""
    counts = images_df["split"].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(counts.index, counts.values)
    axis.set_title("Images per split")
    axis.set_xlabel("Split")
    axis.set_ylabel("Number of images")
    axis.grid(axis="y", alpha=0.25)
    path = figures_dir / "split_sizes.png"
    _save_figure(figure, path)
    return path


def plot_objects_per_image(images_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot the number of annotated objects per image."""
    max_count = int(images_df["object_count"].max())
    bins = np.arange(-0.5, max_count + 1.5, 1.0)
    figure, axis = plt.subplots(figsize=(8, 5))
    for split_name, split_rows in images_df.groupby("split", sort=True):
        axis.hist(
            split_rows["object_count"],
            bins=bins,
            alpha=0.55,
            label=split_name,
        )
    axis.set_title("Objects per image")
    axis.set_xlabel("Number of objects")
    axis.set_ylabel("Number of images")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    path = figures_dir / "objects_per_image.png"
    _save_figure(figure, path)
    return path


def plot_class_distribution(
    class_distribution_df: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Save one horizontal class-frequency plot for every split."""
    paths: list[Path] = []
    for split_name, split_rows in class_distribution_df.groupby(
        "split",
        sort=True,
    ):
        ordered = split_rows.sort_values("object_count", ascending=True)
        figure_height = max(8.0, 0.25 * len(ordered))
        figure, axis = plt.subplots(figsize=(10, figure_height))
        axis.barh(ordered["category_name"], ordered["object_count"])
        axis.set_title(f"Class distribution - {split_name}")
        axis.set_xlabel("Number of annotated objects")
        axis.set_ylabel("Class")
        axis.grid(axis="x", alpha=0.25)
        path = figures_dir / f"class_distribution_{split_name}.png"
        _save_figure(figure, path)
        paths.append(path)
    return paths


def plot_image_dimensions(images_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot image width against image height."""
    figure, axis = plt.subplots(figsize=(7, 6))
    for split_name, split_rows in images_df.groupby("split", sort=True):
        axis.scatter(
            split_rows["width"],
            split_rows["height"],
            alpha=0.6,
            label=split_name,
        )
    axis.set_title("Image dimensions")
    axis.set_xlabel("Width [px]")
    axis.set_ylabel("Height [px]")
    axis.legend()
    axis.grid(alpha=0.25)
    path = figures_dir / "image_dimensions.png"
    _save_figure(figure, path)
    return path


def plot_box_dimensions(boxes_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot positive finite box width against height on logarithmic axes."""
    valid_dimensions = boxes_df[
        boxes_df["coordinates_finite"].fillna(False)
        & (boxes_df["box_width"] > 0)
        & (boxes_df["box_height"] > 0)
    ]
    figure, axis = plt.subplots(figsize=(7, 6))
    for split_name, split_rows in valid_dimensions.groupby("split", sort=True):
        axis.scatter(
            split_rows["box_width"],
            split_rows["box_height"],
            alpha=0.5,
            label=split_name,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title("Bounding-box dimensions")
    axis.set_xlabel("Box width [px, log scale]")
    axis.set_ylabel("Box height [px, log scale]")
    axis.legend()
    axis.grid(alpha=0.25)
    path = figures_dir / "box_dimensions.png"
    _save_figure(figure, path)
    return path


def plot_relative_area(boxes_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot the fraction of image area covered by each valid positive box."""
    values = _finite_series(boxes_df["relative_area"])
    values = values[values > 0]
    figure, axis = plt.subplots(figsize=(8, 5))
    if not values.empty:
        axis.hist(values, bins=40)
        axis.set_xscale("log")
    axis.set_title("Bounding-box relative area")
    axis.set_xlabel("Box area / image area [log scale]")
    axis.set_ylabel("Number of boxes")
    axis.grid(axis="y", alpha=0.25)
    path = figures_dir / "box_relative_area.png"
    _save_figure(figure, path)
    return path


def plot_box_aspect_ratio(boxes_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot box aspect ratios while limiting only the display range."""
    values = _finite_series(boxes_df["box_aspect_ratio"])
    values = values[values > 0]
    figure, axis = plt.subplots(figsize=(8, 5))
    if not values.empty:
        display_max = float(values.quantile(0.99))
        displayed = values[values <= display_max]
        axis.hist(displayed, bins=40)
        axis.set_xlabel(
            f"Box width / height (display clipped at p99={display_max:.3g})"
        )
    else:
        axis.set_xlabel("Box width / height")
    axis.set_title("Bounding-box aspect ratio")
    axis.set_ylabel("Number of boxes")
    axis.grid(axis="y", alpha=0.25)
    path = figures_dir / "box_aspect_ratio.png"
    _save_figure(figure, path)
    return path


def plot_scale_distribution(boxes_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot COCO small, medium and large counts by split."""
    counts = (
        boxes_df.dropna(subset=["scale_coco"])
        .groupby(["split", "scale_coco"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["small", "medium", "large"], fill_value=0)
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=axis)
    axis.set_title("Object scale distribution (COCO area ranges)")
    axis.set_xlabel("Split")
    axis.set_ylabel("Number of boxes")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.25)
    path = figures_dir / "object_scale_distribution.png"
    _save_figure(figure, path)
    return path


def plot_annotation_checks(
    images_df: pd.DataFrame,
    boxes_df: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """Plot counts for the main annotation integrity checks."""
    check_counts = pd.Series(
        {
            "empty images": int((~images_df["has_objects"]).sum()),
            "inconsistent fields": int(
                (~images_df["annotation_lengths_consistent"]).sum()
            ),
            "invalid boxes": int(
                (~boxes_df["valid_box"].fillna(False)).sum()
            ),
            "degenerate boxes": int(
                boxes_df["degenerate"].fillna(False).sum()
            ),
            "out-of-bounds boxes": int(
                boxes_df["out_of_bounds"].fillna(False).sum()
            ),
            "invalid categories": int(
                (~boxes_df["category_valid"].fillna(False)).sum()
            ),
            "exact duplicates": int(
                boxes_df["exact_duplicate"].fillna(False).sum()
            ),
            "area mismatches": int(
                boxes_df["area_mismatch"].fillna(False).sum()
            ),
        }
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(check_counts.index, check_counts.values)
    axis.set_title("Annotation integrity checks")
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    path = figures_dir / "annotation_integrity_checks.png"
    _save_figure(figure, path)
    return path


def save_plots(
    *,
    images_df: pd.DataFrame,
    boxes_df: pd.DataFrame,
    class_distribution_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Save all statistical EDA figures."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_split_sizes(images_df, figures_dir),
        plot_objects_per_image(images_df, figures_dir),
        plot_image_dimensions(images_df, figures_dir),
        plot_box_dimensions(boxes_df, figures_dir),
        plot_relative_area(boxes_df, figures_dir),
        plot_box_aspect_ratio(boxes_df, figures_dir),
        plot_scale_distribution(boxes_df, figures_dir),
        plot_annotation_checks(images_df, boxes_df, figures_dir),
    ]
    paths.extend(plot_class_distribution(class_distribution_df, figures_dir))
    return paths


def _select_example_indices(
    split_images: pd.DataFrame,
    count: int,
    random_generator: np.random.Generator,
) -> list[int]:
    """Select reproducible examples, preferring images with annotations."""
    nonempty = split_images[split_images["object_count"] > 0]
    candidates = nonempty if not nonempty.empty else split_images
    if candidates.empty or count == 0:
        return []

    selected_count = min(count, len(candidates))
    selected = random_generator.choice(
        candidates["sample_index"].to_numpy(dtype=int),
        size=selected_count,
        replace=False,
    )
    return sorted(int(index) for index in np.atleast_1d(selected))


def save_ground_truth_examples(
    *,
    dataset: Any,
    images_df: pd.DataFrame,
    boxes_df: pd.DataFrame,
    output_dir: Path,
    examples_per_split: int,
    seed: int,
) -> list[Path]:
    """Save reproducibly selected images with their source ground-truth boxes."""
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    random_generator = np.random.default_rng(seed)
    saved_paths: list[Path] = []

    for split_name, split_dataset in dataset.items():
        split_images = images_df[images_df["split"] == split_name]
        indices = _select_example_indices(
            split_images=split_images,
            count=examples_per_split,
            random_generator=random_generator,
        )

        for sample_index in indices:
            sample = split_dataset[sample_index]
            image = sample[IMAGE_FIELD]
            sample_boxes = boxes_df[
                (boxes_df["split"] == split_name)
                & (boxes_df["sample_index"] == sample_index)
            ]

            figure, axis = plt.subplots(figsize=(12, 7))
            axis.imshow(image)
            axis.set_title(
                f"{split_name}[{sample_index}] - "
                f"{len(sample_boxes)} annotated object(s)"
            )
            axis.axis("off")

            for row in sample_boxes.itertuples(index=False):
                if not bool(row.bbox_structure_valid) or not bool(
                    row.coordinates_finite
                ):
                    continue

                rectangle = Rectangle(
                    (row.x_min, row.y_min),
                    row.box_width,
                    row.box_height,
                    fill=False,
                    linewidth=2,
                )
                axis.add_patch(rectangle)
                axis.text(
                    row.x_min,
                    max(0.0, row.y_min - 4.0),
                    f"{row.category_id}: {row.category_name}",
                    fontsize=8,
                    bbox={"alpha": 0.7, "pad": 2},
                )

            path = examples_dir / f"{split_name}_{sample_index:04d}.png"
            _save_figure(figure, path)
            saved_paths.append(path)

    return saved_paths


def save_summary(summary: Mapping[str, Any], output_dir: Path) -> Path:
    """Save the complete EDA report as UTF-8 JSON."""
    path = output_dir / "eda_summary.json"
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def print_summary(summary: Mapping[str, Any], output_dir: Path) -> None:
    """Print only computed values from the real dataset."""
    totals = summary["totals"]
    print("\n=== Exercise 3.3 - Step 3: detection dataset EDA ===")
    print(f"Images: {totals['images']}")
    print(f"Objects: {totals['objects']}")
    print(f"Images without objects: {totals['images_without_objects']}")
    print(f"Valid boxes: {totals['valid_boxes']}")
    print(f"Invalid boxes: {totals['invalid_boxes']}")
    print(f"Out-of-bounds boxes: {totals['out_of_bounds_boxes']}")
    print(f"Degenerate boxes: {totals['degenerate_boxes']}")
    print(f"Exact duplicate annotations: {totals['exact_duplicate_annotations']}")
    print(f"Area mismatches: {totals['area_mismatches']}")

    print("\nPer split:")
    for split_name, split_summary in summary["splits"].items():
        print(
            f"  - {split_name}: {split_summary['images']} images, "
            f"{split_summary['objects']} objects, "
            f"{split_summary['images_without_objects']} empty images"
        )

    print(f"\nEDA outputs saved to: {output_dir}")


def main() -> None:
    """Run the complete step-3 EDA and persist every artifact."""
    arguments = parse_arguments()
    if arguments.examples_per_split < 0:
        raise ValueError("--examples-per-split cannot be negative.")

    output_dir = resolve_exercise_path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, _ = load_detection_dataset(arguments.cache_dir)
    images_df, boxes_df, class_distribution_df, class_names = (
        collect_eda_tables(dataset)
    )
    summary = build_summary(
        images_df=images_df,
        boxes_df=boxes_df,
        class_distribution_df=class_distribution_df,
        class_names=class_names,
    )

    save_tables(
        images_df=images_df,
        boxes_df=boxes_df,
        class_distribution_df=class_distribution_df,
        output_dir=output_dir,
    )
    save_plots(
        images_df=images_df,
        boxes_df=boxes_df,
        class_distribution_df=class_distribution_df,
        output_dir=output_dir,
    )
    save_ground_truth_examples(
        dataset=dataset,
        images_df=images_df,
        boxes_df=boxes_df,
        output_dir=output_dir,
        examples_per_split=arguments.examples_per_split,
        seed=arguments.seed,
    )
    save_summary(summary, output_dir)
    print_summary(summary, output_dir)


if __name__ == "__main__":
    main()
