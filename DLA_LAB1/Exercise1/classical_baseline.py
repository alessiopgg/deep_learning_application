import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


SEED = 42
NUM_CLASSES = 43
WANDB_PROJECT = "dla-lab1"

EXERCISE_DIR = Path(__file__).resolve().parent

FEATURES_DIR = (
        EXERCISE_DIR
        / "outputs"
        / "exercise_1_2"
        / "features"
)

RESULTS_DIR = (
        EXERCISE_DIR
        / "outputs"
        / "exercise_1_2"
        / "results"
)

RUNS_DIR = RESULTS_DIR / "runs"
EXPERIMENTS_CSV_PATH = RESULTS_DIR / "experiments.csv"

MODEL_NAMES = [
    "resnet18",
    "resnet50",
]

CLASSIFIER_NAMES = [
    "linear_svc",
    "knn",
    "lda",
]


def load_features(file_path):
    """
    Load features and labels from an NPZ archive.
    """
    archive = np.load(file_path)

    features = archive["features"]
    labels = archive["labels"]

    return features, labels


def get_feature_paths(model_name):
    """
    Return the training and test feature paths for a model.
    """
    train_path = (
            FEATURES_DIR
            / f"train_features_{model_name}.npz"
    )

    test_path = (
            FEATURES_DIR
            / f"test_features_{model_name}.npz"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training features not found: {train_path}\n"
            f"Extract the features for {model_name} first."
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test features not found: {test_path}\n"
            f"Extract the features for {model_name} first."
        )

    return train_path, test_path


def create_classifier(classifier_name):
    """
    Create the requested Scikit-learn classifier.
    """
    if classifier_name == "linear_svc":
        return LinearSVC(
            C=1.0,
            max_iter=10000,
            random_state=SEED,
        )

    if classifier_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        )

    if classifier_name == "lda":
        return LinearDiscriminantAnalysis()

    raise ValueError(
        f"Unknown classifier: {classifier_name}"
    )


def make_serializable(value):
    """
    Convert common Python and NumPy values to JSON-compatible values.
    """
    if isinstance(value, dict):
        return {
            str(key): make_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_serializable(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(
            value,
            (str, int, float, bool),
    ) or value is None:
        return value

    return str(value)


def save_json(file_path, data):
    """
    Save data as a formatted JSON file.
    """
    with file_path.open(
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            make_serializable(data),
            file,
            indent=4,
            ensure_ascii=False,
        )


def save_classification_report(
        file_path,
        report,
):
    """
    Save the classification report as a CSV file.
    """
    fieldnames = [
        "label",
        "precision",
        "recall",
        "f1_score",
        "support",
    ]

    rows = []

    for label, values in report.items():
        if not isinstance(values, dict):
            continue

        rows.append({
            "label": label,
            "precision": values["precision"],
            "recall": values["recall"],
            "f1_score": values["f1-score"],
            "support": values["support"],
        })

    with file_path.open(
            "w",
            newline="",
            encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def append_experiment_summary(result):
    """
    Append one experiment to the global experiments CSV file.
    """
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "run_id",
        "timestamp",
        "model",
        "classifier",
        "feature_dimension",
        "train_samples",
        "test_samples",
        "accuracy",
        "macro_f1",
        "training_seconds",
        "prediction_seconds",
        "classifier_parameters",
        "wandb_run_id",
    ]

    file_already_exists = (
        EXPERIMENTS_CSV_PATH.exists()
    )

    with EXPERIMENTS_CSV_PATH.open(
            "a",
            newline="",
            encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_already_exists:
            writer.writeheader()

        writer.writerow({
            field: result[field]
            for field in fieldnames
        })


def initialize_wandb_run(
        model_name,
        classifier_name,
        run_id,
        config,
):
    """
    Initialize one optional W&B run.
    """
    try:
        import wandb
    except ImportError as error:
        raise ImportError(
            "W&B is not installed. Run: pip install wandb"
        ) from error

    wandb_config = {
        "local_run_id": run_id,
        "exercise": config["exercise"],
        "dataset": config["dataset"],
        "feature_extractor": model_name,
        "classifier": classifier_name,
        "feature_dimension": config["feature_dimension"],
        "train_samples": config["train_samples"],
        "test_samples": config["test_samples"],
        "standardization": config["standardization"],
        "seed": config["seed"],
    }

    for parameter_name, parameter_value in (
            config["classifier_parameters"].items()
    ):
        wandb_config[
            f"classifier_{parameter_name}"
        ] = parameter_value

    run = wandb.init(
        project=WANDB_PROJECT,
        name=run_id,
        group="exercise-1-2",
        job_type="classical-baseline",
        tags=[
            "exercise-1-2",
            model_name,
            classifier_name,
        ],
        config=wandb_config,
    )

    return wandb, run


def log_results_to_wandb(
        wandb_module,
        wandb_run,
        test_labels,
        test_predictions,
        report,
        metrics,
):
    """
    Log final metrics and evaluation tables to W&B.
    """
    report_rows = []

    for label, values in report.items():
        if not isinstance(values, dict):
            continue

        report_rows.append([
            str(label),
            float(values["precision"]),
            float(values["recall"]),
            float(values["f1-score"]),
            int(values["support"]),
        ])

    report_table = wandb_module.Table(
        columns=[
            "label",
            "precision",
            "recall",
            "f1_score",
            "support",
        ],
        data=report_rows,
    )

    confusion_matrix = (
        wandb_module.plot.confusion_matrix(
            y_true=test_labels.tolist(),
            preds=test_predictions.tolist(),
            class_names=[
                str(class_id)
                for class_id in range(NUM_CLASSES)
            ],
            title="GTSRB test confusion matrix",
        )
    )

    wandb_run.log({
        "test/accuracy": metrics["accuracy"],
        "test/macro_f1": metrics["macro_f1"],
        "timing/training_seconds": (
            metrics["training_seconds"]
        ),
        "timing/prediction_seconds": (
            metrics["prediction_seconds"]
        ),
        "test/classification_report": report_table,
        "test/confusion_matrix": confusion_matrix,
    })


def evaluate_classifier(
        model_name,
        classifier_name,
        train_features_scaled,
        train_labels,
        test_features_scaled,
        test_labels,
        use_wandb=False,
):
    """
    Train, evaluate, save and optionally log one classifier run.
    """
    classifier = create_classifier(
        classifier_name
    )

    run_datetime = datetime.now()

    timestamp = run_datetime.isoformat(
        timespec="seconds"
    )

    run_id = (
        f"{run_datetime.strftime('%Y%m%d_%H%M%S_%f')}_"
        f"{model_name}_{classifier_name}"
    )

    run_dir = RUNS_DIR / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    classifier_parameters = make_serializable(
        classifier.get_params(deep=False)
    )

    config = {
        "run_id": run_id,
        "timestamp": timestamp,
        "exercise": "1.2",
        "dataset": "GTSRB",
        "model": model_name,
        "classifier": classifier_name,
        "feature_dimension": int(
            train_features_scaled.shape[1]
        ),
        "train_samples": int(
            len(train_labels)
        ),
        "test_samples": int(
            len(test_labels)
        ),
        "standardization": True,
        "seed": SEED,
        "classifier_parameters": (
            classifier_parameters
        ),
    }

    wandb_module = None
    wandb_run = None

    if use_wandb:
        wandb_module, wandb_run = (
            initialize_wandb_run(
                model_name=model_name,
                classifier_name=classifier_name,
                run_id=run_id,
                config=config,
            )
        )

    try:
        print(
            f"\nTraining {classifier_name} "
            f"on {model_name} features..."
        )

        training_start = perf_counter()

        classifier.fit(
            train_features_scaled,
            train_labels,
        )

        training_seconds = (
                perf_counter() - training_start
        )

        print("Training completed")
        print(
            f"Training time: "
            f"{training_seconds:.2f} seconds"
        )

        if hasattr(classifier, "classes_"):
            print(
                "Number of classes:",
                len(classifier.classes_),
            )

        if hasattr(classifier, "n_iter_"):
            print(
                "Iterations:",
                classifier.n_iter_,
            )

        print("Predicting test classes...")

        prediction_start = perf_counter()

        test_predictions = classifier.predict(
            test_features_scaled
        )

        prediction_seconds = (
                perf_counter() - prediction_start
        )

        accuracy = accuracy_score(
            test_labels,
            test_predictions,
        )

        macro_f1 = f1_score(
            test_labels,
            test_predictions,
            average="macro",
        )

        report = classification_report(
            test_labels,
            test_predictions,
            digits=4,
            output_dict=True,
            zero_division=0,
        )

        printable_report = classification_report(
            test_labels,
            test_predictions,
            digits=4,
            zero_division=0,
        )

        metrics = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "training_seconds": float(
                training_seconds
            ),
            "prediction_seconds": float(
                prediction_seconds
            ),
        }

        save_json(
            run_dir / "config.json",
            config,
            )

        save_json(
            run_dir / "metrics.json",
            metrics,
            )

        save_classification_report(
            run_dir
            / "classification_report.csv",
            report,
            )

        np.savez_compressed(
            run_dir / "predictions.npz",
            true_labels=test_labels,
            predictions=test_predictions,
            )

        wandb_run_id = (
            wandb_run.id
            if wandb_run is not None
            else ""
        )

        summary_result = {
            "run_id": run_id,
            "timestamp": timestamp,
            "model": model_name,
            "classifier": classifier_name,
            "feature_dimension": (
                config["feature_dimension"]
            ),
            "train_samples": (
                config["train_samples"]
            ),
            "test_samples": (
                config["test_samples"]
            ),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "training_seconds": (
                metrics["training_seconds"]
            ),
            "prediction_seconds": (
                metrics["prediction_seconds"]
            ),
            "classifier_parameters": json.dumps(
                classifier_parameters,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "wandb_run_id": wandb_run_id,
        }

        append_experiment_summary(
            summary_result
        )

        if wandb_run is not None:
            log_results_to_wandb(
                wandb_module=wandb_module,
                wandb_run=wandb_run,
                test_labels=test_labels,
                test_predictions=test_predictions,
                report=report,
                metrics=metrics,
            )

        print("Prediction completed")
        print(
            "Predictions shape:",
            test_predictions.shape,
        )

        print("\n=== Exercise 1.2 results ===")
        print(f"Feature extractor: {model_name}")
        print(f"Classifier: {classifier_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-score: {macro_f1:.4f}")
        print(
            f"Prediction time: "
            f"{prediction_seconds:.2f} seconds"
        )

        print("\n=== Classification report ===")
        print(printable_report)

        print(
            "\nLocal results saved in:",
            run_dir,
        )

        if wandb_run is not None:
            print(
                "W&B run ID:",
                wandb_run.id,
            )

        return summary_result

    finally:
        if wandb_run is not None:
            wandb_run.finish()


def run_classical_baseline(
        model_names,
        classifier_names,
        use_wandb=False,
):
    """
    Run the requested combinations of models and classifiers.
    """
    if isinstance(model_names, str):
        model_names = [model_names]

    if isinstance(classifier_names, str):
        classifier_names = [classifier_names]

    if "all" in model_names:
        model_names = MODEL_NAMES

    if "all" in classifier_names:
        classifier_names = CLASSIFIER_NAMES

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for model_name in model_names:
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

        train_path, test_path = get_feature_paths(
            model_name
        )

        train_features, train_labels = load_features(
            train_path
        )

        test_features, test_labels = load_features(
            test_path
        )

        print(f"\n{'=' * 60}")
        print(f"Feature extractor: {model_name}")
        print(f"{'=' * 60}")

        print(
            "Training features shape:",
            train_features.shape,
        )
        print(
            "Training labels shape:",
            train_labels.shape,
        )
        print(
            "Test features shape:",
            test_features.shape,
        )
        print(
            "Test labels shape:",
            test_labels.shape,
        )

        # Fit the scaler only on training features to avoid data leakage.
        scaler = StandardScaler()

        train_features_scaled = scaler.fit_transform(
            train_features
        )

        test_features_scaled = scaler.transform(
            test_features
        )

        print("\nStandardization completed")
        print(
            "Training features mean:",
            train_features_scaled.mean(),
        )
        print(
            "Training features standard deviation:",
            train_features_scaled.std(),
        )

        for classifier_name in classifier_names:
            if classifier_name not in CLASSIFIER_NAMES:
                raise ValueError(
                    f"Unknown classifier: {classifier_name}"
                )

            result = evaluate_classifier(
                model_name=model_name,
                classifier_name=classifier_name,
                train_features_scaled=(
                    train_features_scaled
                ),
                train_labels=train_labels,
                test_features_scaled=(
                    test_features_scaled
                ),
                test_labels=test_labels,
                use_wandb=use_wandb,
            )

            results.append(result)

    print("\n=== Experiments summary ===")

    for result in results:
        print(
            f"{result['model']} + "
            f"{result['classifier']} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"Macro F1: {result['macro_f1']:.4f} | "
            f"Train: {result['training_seconds']:.2f}s | "
            f"Predict: {result['prediction_seconds']:.2f}s"
        )

    print(
        "\nGlobal experiments CSV:",
        EXPERIMENTS_CSV_PATH,
    )

    return results


def main():
    """
    Default execution when this file is launched directly.
    """
    run_classical_baseline(
        model_names=["resnet18"],
        classifier_names=["linear_svc"],
        use_wandb=False,
    )


if __name__ == "__main__":
    main()
