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
from feature_extraction import (
    DEFAULT_EXTRACTION_BATCH_SIZE,
    DEFAULT_SMOKE_TEST_BATCH_SIZE,
    DEFAULT_SMOKE_TEST_EXAMPLES,
    run_feature_extraction_smoke_test,
    run_full_feature_extraction,
    run_token_length_preflight,
)
from transformer_inspection import (
    MODEL_CHECKPOINT,
    run_transformer_batch_inspection,
    run_transformer_inspection,
)

from baseline_classifier import (
    DEFAULT_LINEAR_SVC_C,
    DEFAULT_LINEAR_SVC_MAX_ITER,
    run_validation_baseline,
)

from baseline_classifier import (
    DEFAULT_LINEAR_SVC_C,
    DEFAULT_LINEAR_SVC_C_VALUES,
    DEFAULT_LINEAR_SVC_MAX_ITER,
    run_validation_baseline,
    run_validation_model_selection,
)

from baseline_classifier import (
    DEFAULT_LINEAR_SVC_C,
    DEFAULT_LINEAR_SVC_C_VALUES,
    DEFAULT_LINEAR_SVC_MAX_ITER,
    run_selected_baseline_test_evaluation,
    run_validation_baseline,
    run_validation_model_selection,
)


EXERCISE_DIR = Path(__file__).resolve().parent

EDA_OUTPUT_DIR = (
        EXERCISE_DIR
        / "outputs"
        / "exercise_1_1"
)

EXERCISE_1_3_OUTPUT_DIR = (
        EXERCISE_DIR
        / "outputs"
        / "exercise_1_3"
)

BATCH_SELECTION_CANDIDATES = 50


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

    subparsers.add_parser(
        "inspect-transformer",
        help=(
            "Inspect the DistilBERT tokenizer and model "
            "using one real training example."
        ),
    )

    subparsers.add_parser(
        "inspect-transformer-batch",
        help=(
            "Inspect dynamic padding and DistilBERT outputs "
            "using two real training examples."
        ),
    )

    subparsers.add_parser(
        "feature-preflight",
        help=(
            "Measure tokenized sequence lengths before "
            "extracting DistilBERT features."
        ),
    )

    smoke_parser = subparsers.add_parser(
        "feature-smoke-test",
        help=(
            "Extract DistilBERT CLS features from a small "
            "training subset without saving them."
        ),
    )

    smoke_parser.add_argument(
        "--max-examples",
        type=int,
        default=DEFAULT_SMOKE_TEST_EXAMPLES,
        help=(
            "Maximum number of training examples used in "
            "the smoke test."
        ),
    )

    smoke_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_SMOKE_TEST_BATCH_SIZE,
        help="Batch size used for feature extraction.",
    )

    smoke_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Execution device: auto, cpu, cuda or an "
            "explicit device such as cuda:0."
        ),
    )

    extraction_parser = subparsers.add_parser(
        "extract-features",
        help=(
            "Extract and save DistilBERT CLS features for "
            "train, validation and test."
        ),
    )

    extraction_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EXTRACTION_BATCH_SIZE,
        help=(
            "Batch size used for complete feature extraction."
        ),
    )

    extraction_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Execution device: auto, cpu, cuda or an "
            "explicit device such as cuda:0."
        ),
    )

    extraction_parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing feature archives and metadata."
        ),
    )

    baseline_parser = subparsers.add_parser(
        "train-baseline",
        help=(
            "Train a StandardScaler + LinearSVC pipeline "
            "and evaluate it on validation."
        ),
    )

    baseline_parser.add_argument(
        "--c",
        type=float,
        default=DEFAULT_LINEAR_SVC_C,
        help="LinearSVC regularization parameter C.",
    )

    baseline_parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_LINEAR_SVC_MAX_ITER,
        help="Maximum number of LinearSVC iterations.",
    )

    baseline_parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing validation-baseline artifacts."
        ),
    )

    selection_parser = subparsers.add_parser(
        "select-baseline",
        help=(
            "Select the LinearSVC C value using validation "
            "without evaluating the test split."
        ),
    )

    selection_parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=list(
            DEFAULT_LINEAR_SVC_C_VALUES
        ),
        help=(
            "Candidate LinearSVC C values evaluated "
            "on validation."
        ),
    )

    selection_parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_LINEAR_SVC_MAX_ITER,
        help="Maximum LinearSVC iterations.",
    )

    selection_parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing model-selection artifacts."
        ),
    )

    test_parser = subparsers.add_parser(
        "evaluate-test",
        help=(
            "Evaluate the validation-selected LinearSVC "
            "pipeline once on the test split."
        ),
    )

    test_parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Reproduce and replace an existing test "
            "evaluation using the same selected pipeline."
        ),
    )

    return parser.parse_args()

def run_validation_model_selection_experiment(
        args: argparse.Namespace,
) -> None:
    """
    Select the classical baseline configuration on validation.
    """
    print(
        "Running Exercise 1.3 - "
        "LinearSVC validation model selection..."
    )
    print(f"Dataset identifier: {DATASET_NAME}")
    print(f"Model checkpoint: {MODEL_CHECKPOINT}")
    print(
        "The test split will not be loaded or evaluated."
    )

    run_validation_model_selection(
        output_dir=EXERCISE_1_3_OUTPUT_DIR,
        c_values=args.c_values,
        max_iter=args.max_iter,
        overwrite=args.overwrite,
    )
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


def run_transformer_inspection_experiment() -> None:
    """
    Execute the single-example inspection of Exercise 1.2.
    """
    print(
        "Running Exercise 1.2 - "
        "pre-trained tokenizer and model inspection..."
    )
    print(f"Dataset identifier: {DATASET_NAME}")
    print(f"Model checkpoint: {MODEL_CHECKPOINT}")

    dataset = load_rotten_tomatoes()

    split_name = "train"
    example_index = 0
    example = dataset[split_name][example_index]

    run_transformer_inspection(
        text=example["text"],
        label=example["label"],
        split_name=split_name,
        example_index=example_index,
    )


def select_contrasting_example_indices(
        train_dataset,
        candidate_limit: int = BATCH_SELECTION_CANDIDATES,
) -> list[int]:
    """
    Select a short and a long real text from the first candidates.
    """
    candidate_count = min(
        candidate_limit,
        len(train_dataset),
    )

    if candidate_count < 2:
        raise ValueError(
            "At least two training examples are required."
        )

    candidate_indices = list(
        range(candidate_count)
    )

    shortest_index = min(
        candidate_indices,
        key=lambda index: len(
            train_dataset[index]["text"].split()
        ),
    )

    longest_index = max(
        candidate_indices,
        key=lambda index: len(
            train_dataset[index]["text"].split()
        ),
    )

    if shortest_index == longest_index:
        raise ValueError(
            "Could not find two examples with contrasting "
            "word lengths."
        )

    return [
        shortest_index,
        longest_index,
    ]


def run_transformer_batch_inspection_experiment() -> None:
    """
    Execute the batch and padding inspection of Exercise 1.2.
    """
    print(
        "Running Exercise 1.2 - "
        "batch tokenization and padding inspection..."
    )
    print(f"Dataset identifier: {DATASET_NAME}")
    print(f"Model checkpoint: {MODEL_CHECKPOINT}")

    dataset = load_rotten_tomatoes()

    split_name = "train"
    train_dataset = dataset[split_name]

    example_indices = (
        select_contrasting_example_indices(
            train_dataset=train_dataset,
        )
    )

    examples = [
        train_dataset[example_index]
        for example_index in example_indices
    ]

    run_transformer_batch_inspection(
        texts=[
            example["text"]
            for example in examples
        ],
        labels=[
            example["label"]
            for example in examples
        ],
        split_name=split_name,
        example_indices=example_indices,
    )


def run_feature_preflight_experiment() -> None:
    """
    Execute the token-length preflight for Exercise 1.3.
    """
    print(
        "Running Exercise 1.3 - "
        "feature-extraction preflight..."
    )
    print(f"Dataset identifier: {DATASET_NAME}")
    print(f"Model checkpoint: {MODEL_CHECKPOINT}")

    dataset = load_rotten_tomatoes()

    run_token_length_preflight(
        dataset=dataset,
        output_dir=EXERCISE_1_3_OUTPUT_DIR,
        model_checkpoint=MODEL_CHECKPOINT,
    )


def run_feature_smoke_test_experiment(
        args: argparse.Namespace,
) -> None:
    """
    Execute a small DistilBERT feature-extraction test.
    """
    print(
        "Running Exercise 1.3 - "
        "feature-extraction smoke test..."
    )
    print(f"Dataset identifier: {DATASET_NAME}")
    print(f"Model checkpoint: {MODEL_CHECKPOINT}")

    dataset = load_rotten_tomatoes()

    run_feature_extraction_smoke_test(
        dataset=dataset,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        requested_device=args.device,
        model_checkpoint=MODEL_CHECKPOINT,
    )


def run_full_feature_extraction_experiment(
        args: argparse.Namespace,
) -> None:
    """
    Extract DistilBERT features for every official split.
    """
    print(
        "Running Exercise 1.3 - "
        "complete DistilBERT feature extraction..."
    )
    print(f"Dataset identifier: {DATASET_NAME}")
    print(f"Model checkpoint: {MODEL_CHECKPOINT}")

    dataset = load_rotten_tomatoes()

    run_full_feature_extraction(
        dataset=dataset,
        output_dir=EXERCISE_1_3_OUTPUT_DIR,
        batch_size=args.batch_size,
        requested_device=args.device,
        model_checkpoint=MODEL_CHECKPOINT,
        overwrite=args.overwrite,
    )


def run_validation_baseline_experiment(
        args: argparse.Namespace,
) -> None:
    """
    Train the classical Exercise 1.3 baseline and evaluate validation.
    """
    print(
        "Running Exercise 1.3 - "
        "LinearSVC validation baseline..."
    )
    print(f"Dataset identifier: {DATASET_NAME}")
    print(f"Model checkpoint: {MODEL_CHECKPOINT}")
    print(
        "The test split will not be evaluated in this command."
    )

    run_validation_baseline(
        output_dir=EXERCISE_1_3_OUTPUT_DIR,
        c_value=args.c,
        max_iter=args.max_iter,
        overwrite=args.overwrite,
    )

def run_selected_test_evaluation_experiment(
        args: argparse.Namespace,
) -> None:
    """
    Evaluate the validation-selected pipeline on the test split.
    """
    print(
        "Running Exercise 1.3 - "
        "final selected-baseline test evaluation..."
    )
    print(f"Dataset identifier: {DATASET_NAME}")
    print(f"Model checkpoint: {MODEL_CHECKPOINT}")
    print(
        "The saved validation-selected pipeline will be "
        "used without retraining."
    )

    run_selected_baseline_test_evaluation(
        output_dir=EXERCISE_1_3_OUTPUT_DIR,
        overwrite=args.overwrite,
    )


def main() -> None:
    args = parse_arguments()

    if args.experiment == "eda":
        run_eda_experiment()

    elif args.experiment == "inspect-transformer":
        run_transformer_inspection_experiment()

    elif args.experiment == "inspect-transformer-batch":
        run_transformer_batch_inspection_experiment()

    elif args.experiment == "feature-preflight":
        run_feature_preflight_experiment()

    elif args.experiment == "feature-smoke-test":
        run_feature_smoke_test_experiment(args)

    elif args.experiment == "extract-features":
        run_full_feature_extraction_experiment(args)

    elif args.experiment == "train-baseline":
        run_validation_baseline_experiment(args)

    elif args.experiment == "select-baseline":
        run_validation_model_selection_experiment(args)

    elif args.experiment == "evaluate-test":
        run_selected_test_evaluation_experiment(args)



if __name__ == "__main__":
    main()