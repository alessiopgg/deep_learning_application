import argparse
from pathlib import Path

from data import (
    inspect_tokenized_example,
    load_rotten_tomatoes,
    load_tokenizer,
    tokenize_dataset,
)
from model import (
    inspect_sequence_classifier,
    load_sequence_classifier,
)
from training import (
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_WEIGHT_DECAY,
    run_full_fine_tuning,
    run_test_evaluation,
)


EXERCISE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXERCISE_DIR / "outputs" / "exercise_2_3"


def load_tokenized_data():
    raw_dataset = load_rotten_tomatoes()
    tokenizer = load_tokenizer()
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer)
    return raw_dataset, tokenizer, tokenized_dataset


def run_tokenization_inspection(args: argparse.Namespace) -> None:
    raw_dataset, tokenizer, tokenized_dataset = load_tokenized_data()

    inspect_tokenized_example(
        raw_dataset=raw_dataset,
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        split_name=args.split,
        example_index=args.index,
    )


def run_model_inspection(_: argparse.Namespace) -> None:
    dataset = load_rotten_tomatoes()
    tokenizer = load_tokenizer()
    model = load_sequence_classifier()

    texts = [
        dataset["train"][0]["text"],
        dataset["train"][1]["text"],
    ]

    inspect_sequence_classifier(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
    )


def run_training_command(args: argparse.Namespace) -> None:
    _, tokenizer, tokenized_dataset = load_tokenized_data()

    run_full_fine_tuning(
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


def run_test_command(args: argparse.Namespace) -> None:
    _, tokenizer, tokenized_dataset = load_tokenized_data()

    run_test_evaluation(
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DLA Lab 2 - Exercise 2"
    )
    subparsers = parser.add_subparsers(required=True)

    inspection_parser = subparsers.add_parser(
        "inspect-tokenization",
        help="Run Exercise 2.1 token preprocessing and inspect one example.",
    )
    inspection_parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    inspection_parser.add_argument("--index", type=int, default=0)
    inspection_parser.set_defaults(handler=run_tokenization_inspection)

    model_parser = subparsers.add_parser(
        "inspect-model",
        help="Run Exercise 2.2 and inspect classification logits.",
    )
    model_parser.set_defaults(handler=run_model_inspection)

    training_parser = subparsers.add_parser(
        "train",
        help="Run Exercise 2.3 full DistilBERT fine-tuning.",
    )
    training_parser.add_argument(
        "--epochs",
        type=float,
        default=DEFAULT_EPOCHS,
    )
    training_parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    training_parser.add_argument(
        "--train-batch-size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
    )
    training_parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
    )
    training_parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    training_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    training_parser.set_defaults(handler=run_training_command)

    test_parser = subparsers.add_parser(
        "evaluate-test",
        help="Evaluate the selected fine-tuned model on the test split.",
    )
    test_parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
    )
    test_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    test_parser.set_defaults(handler=run_test_command)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.handler(args)


if __name__ == "__main__":
    main()