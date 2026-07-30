"""Build and save the verified detection-to-GTSRB class mapping."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from Exercise3.data_pipeline.taxonomy import (
    BACKGROUND_LABEL,
    GTSRB_CLASS_NAMES,
    NUM_DETECTOR_CLASSES,
    NUM_GTSRB_CLASSES,
    TaxonomyRecord,
    build_taxonomy_records,
)


DEFAULT_OUTPUT_DIR = Path("outputs/step_4")
EXPECTED_SPLITS = ("train", "validation", "test")


def parse_arguments() -> argparse.Namespace:
    """Parse paths used to validate and export the class taxonomy."""
    from Exercise3.data_pipeline.loading import DEFAULT_CACHE_DIR

    parser = argparse.ArgumentParser(
        description=(
            "Validate the 43 detection categories against the canonical GTSRB "
            "class order and reserve detector label 0 for background."
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
            "Mapping output directory. Relative paths are resolved from the "
            "Exercise3 directory."
        ),
    )
    return parser.parse_args()


def _extract_and_validate_class_names(dataset: Mapping[str, Any]) -> list[str]:
    """Require every split to expose the same ordered ClassLabel taxonomy."""
    from Exercise3.data_pipeline.loading import extract_class_names

    split_taxonomies: dict[str, list[str]] = {}
    for split_name, split_dataset in dataset.items():
        class_names = extract_class_names(split_dataset)
        if not class_names:
            raise ValueError(
                f"Split '{split_name}' does not expose ClassLabel names."
            )
        split_taxonomies[str(split_name)] = class_names

    if not split_taxonomies:
        raise ValueError("The dataset does not contain any splits.")

    reference_split = next(iter(split_taxonomies))
    reference_names = split_taxonomies[reference_split]
    for split_name, class_names in split_taxonomies.items():
        if class_names != reference_names:
            raise ValueError(
                "ClassLabel order differs between splits: "
                f"'{reference_split}' and '{split_name}'."
            )

    return reference_names


def collect_split_class_counts(
    dataset: Mapping[str, Any],
    class_count: int,
) -> dict[str, Counter[int]]:
    """Count source category IDs in every split without changing annotations."""
    from Exercise3.data_pipeline.loading import ANNOTATIONS_FIELD, normalize_objects

    split_counts: dict[str, Counter[int]] = {}
    for split_name, split_dataset in dataset.items():
        counter: Counter[int] = Counter()
        for sample in split_dataset:
            objects = normalize_objects(sample[ANNOTATIONS_FIELD])
            for raw_category_id in objects.get("category", []):
                category_id = int(raw_category_id)
                if not 0 <= category_id < class_count:
                    raise ValueError(
                        f"Invalid category ID {category_id} in split "
                        f"'{split_name}'."
                    )
                counter[category_id] += 1
        split_counts[str(split_name)] = counter

    return split_counts


def build_mapping_rows(
    records: Sequence[TaxonomyRecord],
    split_counts: Mapping[str, Counter[int]],
) -> list[dict[str, Any]]:
    """Combine semantic mapping and observed split support in one table."""
    rows: list[dict[str, Any]] = []
    for record in records:
        counts = {
            split_name: int(split_counts.get(split_name, Counter()).get(
                record.source_category_id,
                0,
            ))
            for split_name in EXPECTED_SPLITS
        }
        rows.append(
            {
                **record.to_dict(),
                "train_count": counts["train"],
                "validation_count": counts["validation"],
                "test_count": counts["test"],
                "total_count": sum(counts.values()),
                "present_in_train": counts["train"] > 0,
                "present_in_validation": counts["validation"] > 0,
                "present_in_test": counts["test"] > 0,
            }
        )
    return rows


def build_mapping_report(
    detection_class_names: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    dataset_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the machine-readable mapping and validation report."""
    mapped_source_ids = [int(row["source_category_id"]) for row in rows]
    mapped_gtsrb_ids = [int(row["gtsrb_class_id"]) for row in rows]
    detector_labels = [int(row["detector_label"]) for row in rows]

    expected_source_ids = set(range(len(detection_class_names)))
    expected_gtsrb_ids = set(range(NUM_GTSRB_CLASSES))
    expected_detector_labels = set(range(1, NUM_DETECTOR_CLASSES))

    train_missing = [
        {
            "source_category_id": int(row["source_category_id"]),
            "source_category_name": str(row["source_category_name"]),
            "gtsrb_class_id": int(row["gtsrb_class_id"]),
            "detector_label": int(row["detector_label"]),
        }
        for row in rows
        if not bool(row["present_in_train"])
    ]

    validation = {
        "source_class_count": len(detection_class_names),
        "gtsrb_class_count": len(GTSRB_CLASS_NAMES),
        "mapped_class_count": len(rows),
        "unmapped_source_ids": sorted(expected_source_ids - set(mapped_source_ids)),
        "unused_gtsrb_ids": sorted(expected_gtsrb_ids - set(mapped_gtsrb_ids)),
        "missing_detector_labels": sorted(
            expected_detector_labels - set(detector_labels)
        ),
        "duplicate_source_ids": sorted(
            source_id
            for source_id in set(mapped_source_ids)
            if mapped_source_ids.count(source_id) > 1
        ),
        "duplicate_gtsrb_ids": sorted(
            class_id
            for class_id in set(mapped_gtsrb_ids)
            if mapped_gtsrb_ids.count(class_id) > 1
        ),
        "duplicate_detector_labels": sorted(
            label
            for label in set(detector_labels)
            if detector_labels.count(label) > 1
        ),
        "is_bijective": (
            set(mapped_source_ids) == expected_source_ids
            and set(mapped_gtsrb_ids) == expected_gtsrb_ids
            and set(detector_labels) == expected_detector_labels
            and len(rows) == NUM_GTSRB_CLASSES
        ),
    }

    if not validation["is_bijective"]:
        raise ValueError(f"The generated mapping is not bijective: {validation}")

    return {
        "dataset": dict(dataset_metadata),
        "label_policy": {
            "background_label": BACKGROUND_LABEL,
            "foreground_rule": "detector_label = gtsrb_class_id + 1",
            "foreground_label_range": [1, NUM_GTSRB_CLASSES],
            "num_real_classes": NUM_GTSRB_CLASSES,
            "num_detector_classes_including_background": NUM_DETECTOR_CLASSES,
            "rationale": (
                "The detector head is new, but using canonical GTSRB order makes "
                "labels, qualitative outputs and checkpoint comparisons explicit."
            ),
        },
        "validation": validation,
        "classes_missing_from_train": train_missing,
        "mappings": [dict(row) for row in rows],
    }


def save_mapping_outputs(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save JSON and CSV versions of the validated taxonomy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "class_mapping.json"
    csv_path = output_dir / "class_mapping.csv"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not rows:
        raise ValueError("Cannot save an empty class mapping table.")
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def print_mapping_summary(report: Mapping[str, Any]) -> None:
    """Print only the checks needed to approve Step 4."""
    validation = report["validation"]
    label_policy = report["label_policy"]
    missing_from_train = report["classes_missing_from_train"]

    print("\n=== Exercise 3.3 - Step 4: class mapping ===")
    print(f"Detection source classes: {validation['source_class_count']}")
    print(f"Canonical GTSRB classes: {validation['gtsrb_class_count']}")
    print(f"Mapped classes: {validation['mapped_class_count']}")
    print(f"Bijective mapping: {validation['is_bijective']}")
    print(f"Background label: {label_policy['background_label']}")
    print(
        "Foreground detector labels: "
        f"{label_policy['foreground_label_range'][0]}.."
        f"{label_policy['foreground_label_range'][1]}"
    )
    print(
        "Faster R-CNN num_classes: "
        f"{label_policy['num_detector_classes_including_background']}"
    )

    if missing_from_train:
        print("\nClasses without positive examples in train:")
        for item in missing_from_train:
            print(
                "  - "
                f"{item['source_category_name']} "
                f"(source ID {item['source_category_id']}, "
                f"GTSRB ID {item['gtsrb_class_id']}, "
                f"detector label {item['detector_label']})"
            )
    else:
        print("\nEvery class has at least one positive training example.")


def main() -> None:
    """Load the cached dataset, validate taxonomy and save mapping artifacts."""
    from Exercise3.data_pipeline.loading import (
        DATASET_CONFIGURATION,
        DATASET_REPOSITORY,
        DATASET_REVISION,
        load_detection_dataset,
        resolve_exercise_path,
    )

    arguments = parse_arguments()
    dataset, _ = load_detection_dataset(arguments.cache_dir)
    detection_class_names = _extract_and_validate_class_names(dataset)
    records = build_taxonomy_records(detection_class_names)
    split_counts = collect_split_class_counts(
        dataset=dataset,
        class_count=len(detection_class_names),
    )
    rows = build_mapping_rows(records, split_counts)

    report = build_mapping_report(
        detection_class_names=detection_class_names,
        rows=rows,
        dataset_metadata={
            "repository": DATASET_REPOSITORY,
            "configuration": DATASET_CONFIGURATION,
            "revision": DATASET_REVISION,
            "source_category_order": detection_class_names,
        },
    )

    output_dir = resolve_exercise_path(arguments.output_dir)
    json_path, csv_path = save_mapping_outputs(report, rows, output_dir)
    print_mapping_summary(report)
    print(f"\nJSON mapping saved to: {json_path}")
    print(f"CSV mapping saved to: {csv_path}")


if __name__ == "__main__":
    main()
