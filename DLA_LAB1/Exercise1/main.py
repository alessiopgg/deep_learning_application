import argparse
from pathlib import Path

from classical_baseline import CLASSIFIER_NAMES, run_classical_baseline
from data import load_gtsrb
from eda import run_eda
from feature_extraction import MODEL_CONFIGS, run_feature_extraction
from fine_tuning import (
    CLASSIFIER_TYPES,
    DEFAULT_CLASSIFIER_TYPE,
    DEFAULT_FINE_TUNING_STRATEGY,
    FINE_TUNING_STRATEGIES,
    NUM_EPOCHS,
    run_fine_tuning,
)

SEED = 42
EXERCISE_DIR = Path(__file__).resolve().parent
DATA_DIR = EXERCISE_DIR.parent / "data"
EDA_OUTPUT_DIR = EXERCISE_DIR / "outputs" / "exercise_1_1"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Exercise 1 experiments")
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    subparsers.add_parser("eda", help="Run the exploratory data analysis.")

    baseline_parser = subparsers.add_parser(
        "baseline", help="Run the classical classification baseline."
    )
    baseline_parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18"],
        choices=[*MODEL_CONFIGS, "all"],
    )
    baseline_parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["linear_svc"],
        choices=[*CLASSIFIER_NAMES, "all"],
    )
    baseline_parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log Exercise 1.2 experiments to Weights & Biases.",
    )

    finetune_parser = subparsers.add_parser(
        "finetune", help="Run the fine-tuning baseline."
    )
    finetune_parser.add_argument(
        "--model", default="resnet18", choices=list(MODEL_CONFIGS)
    )
    finetune_parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    finetune_parser.add_argument(
        "--strategy",
        default=DEFAULT_FINE_TUNING_STRATEGY,
        choices=FINE_TUNING_STRATEGIES,
        help="Choose which parts of the network are trainable.",
    )
    finetune_parser.add_argument(
        "--classifier",
        default=DEFAULT_CLASSIFIER_TYPE,
        choices=CLASSIFIER_TYPES,
        help="Choose the final classifier architecture.",
    )
    finetune_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the default batch size (resnet18=32, resnet50=16).",
    )
    finetune_parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log the Exercise 1.3 fine-tuning run to Weights & Biases.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.experiment == "eda":
        train_dataset, test_dataset = load_gtsrb(DATA_DIR)
        run_eda(train_dataset, test_dataset, EDA_OUTPUT_DIR, seed=SEED)
        return

    if args.experiment == "baseline":
        model_names = list(MODEL_CONFIGS) if "all" in args.models else args.models
        for model_name in model_names:
            run_feature_extraction(model_name)
        run_classical_baseline(model_names, args.classifiers, use_wandb=args.wandb)
        return

    run_fine_tuning(
        model_name=args.model,
        num_epochs=args.epochs,
        strategy=args.strategy,
        classifier_type=args.classifier,
        batch_size=args.batch_size,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
