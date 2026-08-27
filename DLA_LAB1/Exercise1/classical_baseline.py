import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from feature_extraction import FEATURES_DIR, MODEL_CONFIGS

SEED = 42
NUM_CLASSES = 43
WANDB_PROJECT = "dla-lab1"

EXERCISE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXERCISE_DIR / "outputs" / "exercise_1_2" / "results"
RUNS_DIR = RESULTS_DIR / "runs"
EXPERIMENTS_CSV_PATH = RESULTS_DIR / "experiments.csv"

MODEL_NAMES = tuple(MODEL_CONFIGS)
CLASSIFIER_NAMES = ("linear_svc", "knn", "lda")


def load_features(file_path):
    with np.load(file_path) as archive:
        return archive["features"], archive["labels"]


def get_feature_paths(model_name):
    train_path = FEATURES_DIR / f"train_features_{model_name}.npz"
    test_path = FEATURES_DIR / f"test_features_{model_name}.npz"

    for split_name, path in (("Training", train_path), ("Test", test_path)):
        if not path.exists():
            raise FileNotFoundError(
                f"{split_name} features not found: {path}\n"
                f"Extract the features for {model_name} first."
            )

    return train_path, test_path


def create_classifier(classifier_name):
    classifiers = {
        "linear_svc": lambda: LinearSVC(
            C=1.0,
            max_iter=10000,
            random_state=SEED,
        ),
        "knn": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "lda": LinearDiscriminantAnalysis,
    }
    try:
        return classifiers[classifier_name]()
    except KeyError as error:
        raise ValueError(f"Unknown classifier: {classifier_name}") from error


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(file_path, data):
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False, default=_json_default)


def save_classification_report(file_path, report):
    fieldnames = ["label", "precision", "recall", "f1_score", "support"]
    rows = [
        {
            "label": label,
            "precision": values["precision"],
            "recall": values["recall"],
            "f1_score": values["f1-score"],
            "support": values["support"],
        }
        for label, values in report.items()
        if isinstance(values, dict)
    ]

    with file_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_experiment_summary(result):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = EXPERIMENTS_CSV_PATH.exists()

    with EXPERIMENTS_CSV_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def initialize_wandb_run(model_name, classifier_name, run_id, config):
    try:
        import wandb
    except ImportError as error:
        raise ImportError("W&B is not installed. Run: pip install wandb") from error

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
    wandb_config.update(
        {
            f"classifier_{name}": value
            for name, value in config["classifier_parameters"].items()
        }
    )

    run = wandb.init(
        project=WANDB_PROJECT,
        name=run_id,
        group="exercise-1-2",
        job_type="classical-baseline",
        tags=["exercise-1-2", model_name, classifier_name],
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
    rows = [
        [
            str(label),
            float(values["precision"]),
            float(values["recall"]),
            float(values["f1-score"]),
            int(values["support"]),
        ]
        for label, values in report.items()
        if isinstance(values, dict)
    ]
    report_table = wandb_module.Table(
        columns=["label", "precision", "recall", "f1_score", "support"],
        data=rows,
    )
    confusion_matrix = wandb_module.plot.confusion_matrix(
        y_true=test_labels.tolist(),
        preds=test_predictions.tolist(),
        class_names=[str(class_id) for class_id in range(NUM_CLASSES)],
        title="GTSRB test confusion matrix",
    )
    wandb_run.log(
        {
            "test/accuracy": metrics["accuracy"],
            "test/macro_f1": metrics["macro_f1"],
            "timing/training_seconds": metrics["training_seconds"],
            "timing/prediction_seconds": metrics["prediction_seconds"],
            "test/classification_report": report_table,
            "test/confusion_matrix": confusion_matrix,
        }
    )


def evaluate_classifier(
    model_name,
    classifier_name,
    train_features_scaled,
    train_labels,
    test_features_scaled,
    test_labels,
    use_wandb=False,
):
    classifier = create_classifier(classifier_name)
    run_datetime = datetime.now()
    timestamp = run_datetime.isoformat(timespec="seconds")
    run_id = (
        f"{run_datetime.strftime('%Y%m%d_%H%M%S_%f')}_"
        f"{model_name}_{classifier_name}"
    )
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    classifier_parameters = classifier.get_params(deep=False)
    config = {
        "run_id": run_id,
        "timestamp": timestamp,
        "exercise": "1.2",
        "dataset": "GTSRB",
        "model": model_name,
        "classifier": classifier_name,
        "feature_dimension": int(train_features_scaled.shape[1]),
        "train_samples": int(len(train_labels)),
        "test_samples": int(len(test_labels)),
        "standardization": True,
        "seed": SEED,
        "classifier_parameters": classifier_parameters,
    }

    wandb_module = wandb_run = None
    if use_wandb:
        wandb_module, wandb_run = initialize_wandb_run(
            model_name, classifier_name, run_id, config
        )

    try:
        print(f"\nTraining {classifier_name} on {model_name} features...")
        start = perf_counter()
        classifier.fit(train_features_scaled, train_labels)
        training_seconds = perf_counter() - start
        print("Training completed")
        print(f"Training time: {training_seconds:.2f} seconds")

        if hasattr(classifier, "classes_"):
            print("Number of classes:", len(classifier.classes_))
        if hasattr(classifier, "n_iter_"):
            print("Iterations:", classifier.n_iter_)

        print("Predicting test classes...")
        start = perf_counter()
        test_predictions = classifier.predict(test_features_scaled)
        prediction_seconds = perf_counter() - start

        accuracy = accuracy_score(test_labels, test_predictions)
        macro_f1 = f1_score(test_labels, test_predictions, average="macro")
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
            "training_seconds": float(training_seconds),
            "prediction_seconds": float(prediction_seconds),
        }

        save_json(run_dir / "config.json", config)
        save_json(run_dir / "metrics.json", metrics)
        save_classification_report(run_dir / "classification_report.csv", report)
        np.savez_compressed(
            run_dir / "predictions.npz",
            true_labels=test_labels,
            predictions=test_predictions,
        )

        summary_result = {
            "run_id": run_id,
            "timestamp": timestamp,
            "model": model_name,
            "classifier": classifier_name,
            "feature_dimension": config["feature_dimension"],
            "train_samples": config["train_samples"],
            "test_samples": config["test_samples"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "training_seconds": metrics["training_seconds"],
            "prediction_seconds": metrics["prediction_seconds"],
            "classifier_parameters": json.dumps(
                classifier_parameters,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            ),
            "wandb_run_id": wandb_run.id if wandb_run is not None else "",
        }
        append_experiment_summary(summary_result)

        if wandb_run is not None:
            log_results_to_wandb(
                wandb_module,
                wandb_run,
                test_labels,
                test_predictions,
                report,
                metrics,
            )

        print("Prediction completed")
        print("Predictions shape:", test_predictions.shape)
        print("\n=== Exercise 1.2 results ===")
        print(f"Feature extractor: {model_name}")
        print(f"Classifier: {classifier_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-score: {macro_f1:.4f}")
        print(f"Prediction time: {prediction_seconds:.2f} seconds")
        print("\n=== Classification report ===")
        print(printable_report)
        print("\nLocal results saved in:", run_dir)
        if wandb_run is not None:
            print("W&B run ID:", wandb_run.id)

        return summary_result
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def run_classical_baseline(model_names, classifier_names, use_wandb=False):
    if isinstance(model_names, str):
        model_names = [model_names]
    if isinstance(classifier_names, str):
        classifier_names = [classifier_names]

    model_names = list(MODEL_NAMES) if "all" in model_names else model_names
    classifier_names = (
        list(CLASSIFIER_NAMES) if "all" in classifier_names else classifier_names
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for model_name in model_names:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {model_name}")

        train_path, test_path = get_feature_paths(model_name)
        train_features, train_labels = load_features(train_path)
        test_features, test_labels = load_features(test_path)

        print(f"\n{'=' * 60}")
        print(f"Feature extractor: {model_name}")
        print(f"{'=' * 60}")
        print("Training features shape:", train_features.shape)
        print("Training labels shape:", train_labels.shape)
        print("Test features shape:", test_features.shape)
        print("Test labels shape:", test_labels.shape)

        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        print("\nStandardization completed")
        print("Training features mean:", train_features_scaled.mean())
        print("Training features standard deviation:", train_features_scaled.std())

        for classifier_name in classifier_names:
            if classifier_name not in CLASSIFIER_NAMES:
                raise ValueError(f"Unknown classifier: {classifier_name}")
            results.append(
                evaluate_classifier(
                    model_name,
                    classifier_name,
                    train_features_scaled,
                    train_labels,
                    test_features_scaled,
                    test_labels,
                    use_wandb,
                )
            )

    print("\n=== Experiments summary ===")
    for result in results:
        print(
            f"{result['model']} + {result['classifier']} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"Macro F1: {result['macro_f1']:.4f} | "
            f"Train: {result['training_seconds']:.2f}s | "
            f"Predict: {result['prediction_seconds']:.2f}s"
        )
    print("\nGlobal experiments CSV:", EXPERIMENTS_CSV_PATH)
    return results


def main():
    run_classical_baseline(["resnet18"], ["linear_svc"], use_wandb=False)


if __name__ == "__main__":
    main()
