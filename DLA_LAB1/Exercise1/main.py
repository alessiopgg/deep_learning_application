import argparse
from pathlib import Path

from classical_baseline import run_classical_baseline
from data import load_gtsrb
from eda import run_eda
from feature_extraction import (
    MODEL_CONFIGS as FEATURE_EXTRACTOR_CONFIGS,
    run_feature_extraction,
)
from fine_tuning import (
    CLASSIFIER_TYPES as FINE_TUNING_CLASSIFIER_TYPES,
    DEFAULT_CLASSIFIER_TYPE,
    DEFAULT_FINE_TUNING_STRATEGY,
    FINE_TUNING_STRATEGIES,
    MODEL_CONFIGS as FINE_TUNING_MODEL_CONFIGS,
    NUM_EPOCHS as DEFAULT_NUM_EPOCHS,
    run_fine_tuning,
)


SEED = 42

EXERCISE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXERCISE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"

EDA_OUTPUT_DIR = (
        EXERCISE_DIR
        / "outputs"
        / "exercise_1_1"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Exercise 1 experiments"
    )

    subparsers = parser.add_subparsers(
        dest="experiment",
        required=True,
    )

    # Exercise 1.1
    subparsers.add_parser(
        "eda",
        help="Run the exploratory data analysis.",
    )

    # Exercise 1.2
    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Run the classical classification baseline.",
    )

    baseline_parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18"],
        choices=[
            "resnet18",
            "resnet50",
            "all",
        ],
    )

    baseline_parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["linear_svc"],
        choices=[
            "linear_svc",
            "knn",
            "lda",
            "all",
        ],
    )

    baseline_parser.add_argument(
        "--wandb",
        action="store_true",
        help=(
            "Log Exercise 1.2 experiments "
            "to Weights & Biases."
        ),
    )

    # Exercise 1.3
    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Run the fine-tuning baseline.",
    )

    finetune_parser.add_argument(
        "--model",
        default="resnet18",
        choices=list(FINE_TUNING_MODEL_CONFIGS.keys()),
    )

    finetune_parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
    )

    finetune_parser.add_argument(
        "--strategy",
        default=DEFAULT_FINE_TUNING_STRATEGY,
        choices=FINE_TUNING_STRATEGIES,
        help="Choose which parts of the network are trainable.",
    )

    finetune_parser.add_argument(
        "--classifier",
        default=DEFAULT_CLASSIFIER_TYPE,
        choices=FINE_TUNING_CLASSIFIER_TYPES,
        help="Choose the final classifier architecture.",
    )

    finetune_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Override the default model batch size. "
            "Defaults: resnet18=32, resnet50=16."
        ),
    )

    finetune_parser.add_argument(
        "--wandb",
        action="store_true",
        help=(
            "Log the Exercise 1.3 fine-tuning run "
            "to Weights & Biases."
        ),
    )

    return parser.parse_args()


def run_eda_experiment():
    print("Running Exercise 1.1 - EDA...")

    train_dataset, test_dataset = load_gtsrb(DATA_DIR)

    run_eda(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        output_dir=EDA_OUTPUT_DIR,
        seed=SEED,
    )


def main():
    args = parse_arguments()

    if args.experiment == "eda":
        run_eda_experiment()

    elif args.experiment == "baseline":
        model_names = args.models

        if "all" in model_names:
            model_names = list(
                FEATURE_EXTRACTOR_CONFIGS.keys()
            )

        for model_name in model_names:
            run_feature_extraction(
                model_name=model_name
            )

        run_classical_baseline(
            model_names=model_names,
            classifier_names=args.classifiers,
            use_wandb=args.wandb,
        )

    elif args.experiment == "finetune":
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