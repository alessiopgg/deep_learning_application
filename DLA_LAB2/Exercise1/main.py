import argparse
from pathlib import Path

from data import (
    DATASET_NAME,
    get_available_split_names,
    load_rotten_tomatoes,
)
from eda import (
    print_dataset_overview,
    run_eda,
)


EXERCISE_DIR = Path(__file__).resolve().parent

EDA_OUTPUT_DIR = (
        EXERCISE_DIR
        / "outputs"
        / "exercise_1_1"
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse the command-line arguments for Exercise 1.
    """
    parser = argparse.ArgumentParser(
        description="DLA Lab 2 - Exercise 1 experiments"
    )

    subparsers = parser.add_subparsers(
        dest="experiment",
        required=True,
    )

    subparsers.add_parser(
        "eda",
        help="Run the Rotten Tomatoes exploratory analysis.",
    )

    return parser.parse_args()


def run_eda_experiment() -> None:
    """
    Execute Exercise 1.1.
    """
    print("Running Exercise 1.1 - dataset exploration...")
    print(f"Dataset identifier: {DATASET_NAME}")

    declared_split_names = get_available_split_names()
    dataset = load_rotten_tomatoes()

    print_dataset_overview(
        dataset=dataset,
        declared_split_names=declared_split_names,
    )

    run_eda(
        dataset=dataset,
        output_dir=EDA_OUTPUT_DIR,
    )


def main() -> None:
    args = parse_arguments()

    if args.experiment == "eda":
        run_eda_experiment()


if __name__ == "__main__":
    main()