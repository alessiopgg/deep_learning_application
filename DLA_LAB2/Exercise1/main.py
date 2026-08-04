import argparse
from pathlib import Path

from baseline_classifier import (
    DEFAULT_LINEAR_SVC_C_VALUES,
    DEFAULT_LINEAR_SVC_MAX_ITER,
    DEFAULT_LOGISTIC_REGRESSION_C,
    DEFAULT_LOGISTIC_REGRESSION_MAX_ITER,
    run_logistic_regression_validation_experiment,
    run_selected_baseline_test_evaluation,
    run_validation_model_selection,
)
from data import DATASET_NAME, load_rotten_tomatoes
from eda import print_dataset_overview, run_eda
from feature_extraction import (
    DEFAULT_EXTRACTION_BATCH_SIZE,
    run_full_feature_extraction,
)
from transformer_inspection import (
    MODEL_CHECKPOINT,
    run_transformer_batch_inspection,
    run_transformer_inspection,
)


EXERCISE_DIR = Path(__file__).resolve().parent
EDA_OUTPUT_DIR = EXERCISE_DIR / "outputs" / "exercise_1_1"
EXERCISE_1_3_OUTPUT_DIR = EXERCISE_DIR / "outputs" / "exercise_1_3"


def run_eda_command(_: argparse.Namespace) -> None:
    dataset = load_rotten_tomatoes()
    print(f"Dataset identifier: {DATASET_NAME}")
    print_dataset_overview(dataset)
    run_eda(dataset, EDA_OUTPUT_DIR)


def run_single_inspection_command(_: argparse.Namespace) -> None:
    dataset = load_rotten_tomatoes()
    example = dataset["train"][0]
    run_transformer_inspection(
        text=example["text"],
        label=example["label"],
        split_name="train",
        example_index=0,
    )


def run_batch_inspection_command(_: argparse.Namespace) -> None:
    dataset = load_rotten_tomatoes()
    train_split = dataset["train"]
    candidate_indices = range(min(50, len(train_split)))

    shortest = min(
        candidate_indices,
        key=lambda index: len(train_split[index]["text"].split()),
    )
    longest = max(
        candidate_indices,
        key=lambda index: len(train_split[index]["text"].split()),
    )
    indices = [shortest, longest]
    examples = [train_split[index] for index in indices]

    run_transformer_batch_inspection(
        texts=[example["text"] for example in examples],
        labels=[example["label"] for example in examples],
        split_name="train",
        example_indices=indices,
    )


def run_feature_extraction_command(args: argparse.Namespace) -> None:
    dataset = load_rotten_tomatoes()
    run_full_feature_extraction(
        dataset=dataset,
        output_dir=EXERCISE_1_3_OUTPUT_DIR,
        batch_size=args.batch_size,
        requested_device=args.device,
        model_checkpoint=MODEL_CHECKPOINT,
        overwrite=args.overwrite,
    )


def run_model_selection_command(args: argparse.Namespace) -> None:
    run_validation_model_selection(
        output_dir=EXERCISE_1_3_OUTPUT_DIR,
        c_values=args.c_values,
        max_iter=args.max_iter,
        overwrite=args.overwrite,
    )



def run_logistic_regression_validation_command(
    args: argparse.Namespace,
) -> None:
    run_logistic_regression_validation_experiment(
        output_dir=EXERCISE_1_3_OUTPUT_DIR,
        c_value=args.c,
        max_iter=args.max_iter,
        overwrite=args.overwrite,
    )


def run_test_evaluation_command(args: argparse.Namespace) -> None:
    run_selected_baseline_test_evaluation(
        output_dir=EXERCISE_1_3_OUTPUT_DIR,
        overwrite=args.overwrite,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DLA Lab 2 - Exercise 1"
    )
    subparsers = parser.add_subparsers(required=True)

    eda_parser = subparsers.add_parser("eda", help="Run Exercise 1.1.")
    eda_parser.set_defaults(handler=run_eda_command)

    inspect_parser = subparsers.add_parser(
        "inspect-transformer",
        help="Inspect DistilBERT on one dataset example.",
    )
    inspect_parser.set_defaults(handler=run_single_inspection_command)

    batch_parser = subparsers.add_parser(
        "inspect-transformer-batch",
        help="Inspect dynamic padding on two dataset examples.",
    )
    batch_parser.set_defaults(handler=run_batch_inspection_command)

    extraction_parser = subparsers.add_parser(
        "extract-features",
        help="Extract and save DistilBERT CLS features.",
    )
    extraction_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EXTRACTION_BATCH_SIZE,
    )
    extraction_parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda or an explicit device such as cuda:0",
    )
    extraction_parser.add_argument("--overwrite", action="store_true")
    extraction_parser.set_defaults(handler=run_feature_extraction_command)

    selection_parser = subparsers.add_parser(
        "select-baseline",
        help="Select LinearSVC C using validation.",
    )
    selection_parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_LINEAR_SVC_C_VALUES),
    )
    selection_parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_LINEAR_SVC_MAX_ITER,
    )
    selection_parser.add_argument("--overwrite", action="store_true")
    selection_parser.set_defaults(handler=run_model_selection_command)


    logistic_parser = subparsers.add_parser(
        "evaluate-logistic",
        help=(
            "Train StandardScaler + LogisticRegression on cached "
            "train features and evaluate validation only."
        ),
    )
    logistic_parser.add_argument(
        "--c",
        type=float,
        default=DEFAULT_LOGISTIC_REGRESSION_C,
    )
    logistic_parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_LOGISTIC_REGRESSION_MAX_ITER,
    )
    logistic_parser.add_argument("--overwrite", action="store_true")
    logistic_parser.set_defaults(
        handler=run_logistic_regression_validation_command
    )

    test_parser = subparsers.add_parser(
        "evaluate-test",
        help="Evaluate the selected baseline on test.",
    )
    test_parser.add_argument("--overwrite", action="store_true")
    test_parser.set_defaults(handler=run_test_evaluation_command)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.handler(args)


if __name__ == "__main__":
    main()
