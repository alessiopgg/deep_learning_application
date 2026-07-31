"""Entry point for Exercise 3.3 dataset inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

from Exercise3.data_pipeline.loading import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_PATH,
    inspect_dataset,
    load_detection_dataset,
    print_dataset_inspection,
    save_dataset_inspection,
)


def parse_arguments() -> argparse.Namespace:
    """Parse the options needed by step 2."""
    parser = argparse.ArgumentParser(
        description=(
            "Load and inspect the German traffic-sign detection dataset "
            "without constructing a detector."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Hugging Face cache path, relative to Exercise3 if needed.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to inspect: train, validation or test.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based sample index inside the selected split.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=10,
        help="Maximum annotations printed for the selected sample.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="JSON report path, relative to Exercise3 if needed.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the dataset, inspect it and save the resulting report."""
    arguments = parse_arguments()

    if arguments.sample_index < 0:
        raise ValueError("--sample-index must be greater than or equal to zero.")
    if arguments.max_objects <= 0:
        raise ValueError("--max-objects must be greater than zero.")

    dataset, cache_dir = load_detection_dataset(arguments.cache_dir)
    report = inspect_dataset(
        dataset=dataset,
        cache_dir=cache_dir,
        split_name=arguments.split,
        sample_index=arguments.sample_index,
        max_objects=arguments.max_objects,
    )

    print_dataset_inspection(report)
    report_path = save_dataset_inspection(report, arguments.report_path)
    print(f"\nInspection report saved to: {report_path}")


if __name__ == "__main__":
    main()
