"""Explicit class taxonomy used by the Exercise 3 detector.

The Hugging Face detection dataset stores the same 43 traffic-sign concepts as
GTSRB, but in a different category order and with shorter English names.  This
module defines the semantic permutation explicitly and reserves detector label
0 for Faster R-CNN background.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


BACKGROUND_LABEL = 0
NUM_GTSRB_CLASSES = 43
NUM_DETECTOR_CLASSES = NUM_GTSRB_CLASSES + 1

# Conventional GTSRB class-ID order, IDs 0 through 42.
GTSRB_CLASS_NAMES: tuple[str, ...] = (
    "Speed limit (20 km/h)",
    "Speed limit (30 km/h)",
    "Speed limit (50 km/h)",
    "Speed limit (60 km/h)",
    "Speed limit (70 km/h)",
    "Speed limit (80 km/h)",
    "End of speed limit (80 km/h)",
    "Speed limit (100 km/h)",
    "Speed limit (120 km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
)

# Project-normalized names in the same canonical GTSRB order.
GTSRB_NORMALIZED_NAMES: tuple[str, ...] = (
    "speed limit 20",
    "speed limit 30",
    "speed limit 50",
    "speed limit 60",
    "speed limit 70",
    "speed limit 80",
    "restriction ends 80",
    "speed limit 100",
    "speed limit 120",
    "no overtaking",
    "no overtaking -trucks-",
    "priority at next intersection",
    "priority road",
    "give way",
    "stop",
    "no traffic both ways",
    "no trucks",
    "no entry",
    "danger",
    "bend left",
    "bend right",
    "bend",
    "uneven road",
    "slippery road",
    "road narrows",
    "construction",
    "traffic signal",
    "pedestrian crossing",
    "school crossing",
    "cycles crossing",
    "snow",
    "animals",
    "restriction ends",
    "go right",
    "go left",
    "go straight",
    "go right or straight",
    "go left or straight",
    "keep right",
    "keep left",
    "roundabout",
    "restriction ends -overtaking-",
    "restriction ends -overtaking -trucks--",
)

# Explicit semantic permutation: detection category name -> canonical GTSRB ID.
DETECTION_NAME_TO_GTSRB_ID: dict[str, int] = {
    "animals": 31,
    "construction": 25,
    "cycles crossing": 29,
    "danger": 18,
    "no entry": 17,
    "pedestrian crossing": 27,
    "school crossing": 28,
    "snow": 30,
    "stop": 14,
    "bend": 21,
    "bend left": 19,
    "bend right": 20,
    "give way": 13,
    "go left": 34,
    "go left or straight": 37,
    "go right": 33,
    "go right or straight": 36,
    "go straight": 35,
    "keep left": 39,
    "keep right": 38,
    "no overtaking": 9,
    "no overtaking -trucks-": 10,
    "no traffic both ways": 15,
    "no trucks": 16,
    "priority at next intersection": 11,
    "priority road": 12,
    "restriction ends": 32,
    "restriction ends -overtaking -trucks--": 42,
    "restriction ends -overtaking-": 41,
    "restriction ends 80": 6,
    "road narrows": 24,
    "roundabout": 40,
    "slippery road": 23,
    "speed limit 100": 7,
    "speed limit 120": 8,
    "speed limit 20": 0,
    "speed limit 30": 1,
    "speed limit 50": 2,
    "speed limit 60": 3,
    "speed limit 70": 4,
    "speed limit 80": 5,
    "traffic signal": 26,
    "uneven road": 22,
}

MAPPING_NOTES: dict[str, str] = {
    "animals": "Detection shorthand for GTSRB 'Wild animals crossing'.",
    "construction": "Detection shorthand for GTSRB 'Road work'.",
    "danger": "Detection shorthand for GTSRB 'General caution'.",
    "bend": "Detection shorthand for GTSRB 'Double curve'.",
    "give way": "Equivalent to the GTSRB class commonly named 'Yield'.",
    "no traffic both ways": "Detection shorthand for GTSRB 'No vehicles'.",
    "no trucks": (
        "Detection shorthand for vehicles over 3.5 metric tons prohibited."
    ),
    "restriction ends": (
        "Detection shorthand for end of all speed and passing limits."
    ),
    "road narrows": "Mapped to the GTSRB road-narrows-on-the-right class.",
    "snow": "Detection shorthand for GTSRB 'Beware of ice/snow'.",
    "traffic signal": "Equivalent to the GTSRB class 'Traffic signals'.",
}


@dataclass(frozen=True)
class TaxonomyRecord:
    """One verified correspondence between source and detector taxonomies."""

    source_category_id: int
    source_category_name: str
    gtsrb_class_id: int
    gtsrb_class_name: str
    normalized_name: str
    detector_label: int
    mapping_note: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def validate_static_taxonomy() -> None:
    """Validate constants that must remain a complete one-to-one mapping."""
    if len(GTSRB_CLASS_NAMES) != NUM_GTSRB_CLASSES:
        raise ValueError(
            "GTSRB_CLASS_NAMES must contain exactly "
            f"{NUM_GTSRB_CLASSES} entries."
        )
    if len(GTSRB_NORMALIZED_NAMES) != NUM_GTSRB_CLASSES:
        raise ValueError(
            "GTSRB_NORMALIZED_NAMES must contain exactly "
            f"{NUM_GTSRB_CLASSES} entries."
        )
    if len(set(GTSRB_CLASS_NAMES)) != NUM_GTSRB_CLASSES:
        raise ValueError("GTSRB_CLASS_NAMES contains duplicate names.")
    if len(set(GTSRB_NORMALIZED_NAMES)) != NUM_GTSRB_CLASSES:
        raise ValueError("GTSRB_NORMALIZED_NAMES contains duplicate names.")

    mapped_ids = list(DETECTION_NAME_TO_GTSRB_ID.values())
    if len(DETECTION_NAME_TO_GTSRB_ID) != NUM_GTSRB_CLASSES:
        raise ValueError(
            "DETECTION_NAME_TO_GTSRB_ID must contain exactly "
            f"{NUM_GTSRB_CLASSES} entries."
        )
    if set(mapped_ids) != set(range(NUM_GTSRB_CLASSES)):
        missing = sorted(set(range(NUM_GTSRB_CLASSES)) - set(mapped_ids))
        duplicates = sorted(
            class_id
            for class_id in set(mapped_ids)
            if mapped_ids.count(class_id) > 1
        )
        raise ValueError(
            "The detection-to-GTSRB mapping is not bijective. "
            f"Missing target IDs: {missing}; duplicated target IDs: {duplicates}."
        )


def build_taxonomy_records(
    detection_class_names: Sequence[str],
) -> tuple[TaxonomyRecord, ...]:
    """Build and validate the mapping using the dataset's actual class order."""
    validate_static_taxonomy()

    source_names = [str(name) for name in detection_class_names]
    if len(source_names) != NUM_GTSRB_CLASSES:
        raise ValueError(
            "The detection dataset must expose exactly "
            f"{NUM_GTSRB_CLASSES} classes, but found {len(source_names)}."
        )
    if len(set(source_names)) != len(source_names):
        raise ValueError("The detection dataset contains duplicate class names.")

    expected_names = set(DETECTION_NAME_TO_GTSRB_ID)
    actual_names = set(source_names)
    missing_names = sorted(expected_names - actual_names)
    unexpected_names = sorted(actual_names - expected_names)
    if missing_names or unexpected_names:
        raise ValueError(
            "The dataset taxonomy differs from the verified mapping. "
            f"Missing expected names: {missing_names}; "
            f"unexpected names: {unexpected_names}."
        )

    records = []
    for source_id, source_name in enumerate(source_names):
        gtsrb_id = DETECTION_NAME_TO_GTSRB_ID[source_name]
        records.append(
            TaxonomyRecord(
                source_category_id=source_id,
                source_category_name=source_name,
                gtsrb_class_id=gtsrb_id,
                gtsrb_class_name=GTSRB_CLASS_NAMES[gtsrb_id],
                normalized_name=GTSRB_NORMALIZED_NAMES[gtsrb_id],
                detector_label=gtsrb_id + 1,
                mapping_note=MAPPING_NOTES.get(
                    source_name,
                    "Direct semantic equivalence after name normalization.",
                ),
            )
        )

    detector_labels = {record.detector_label for record in records}
    expected_labels = set(range(1, NUM_DETECTOR_CLASSES))
    if detector_labels != expected_labels:
        raise ValueError(
            "Detector labels must cover every foreground label from 1 to 43."
        )

    return tuple(records)


def build_source_to_detector_label(
    detection_class_names: Sequence[str],
) -> dict[int, int]:
    """Return source category ID -> Faster R-CNN foreground label."""
    return {
        record.source_category_id: record.detector_label
        for record in build_taxonomy_records(detection_class_names)
    }


def build_detector_label_to_name() -> dict[int, str]:
    """Return labels used by Faster R-CNN, including background label 0."""
    validate_static_taxonomy()
    return {
        BACKGROUND_LABEL: "background",
        **{
            gtsrb_id + 1: class_name
            for gtsrb_id, class_name in enumerate(GTSRB_CLASS_NAMES)
        },
    }
