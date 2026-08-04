import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


SEED = 42
CLASS_IDS = (0, 1)
CLASS_NAMES = ("neg", "pos")
DEFAULT_LINEAR_SVC_C_VALUES = (0.01, 0.1, 1.0, 10.0)
DEFAULT_LINEAR_SVC_MAX_ITER = 10_000
DEFAULT_LOGISTIC_REGRESSION_C = 1.0
DEFAULT_LOGISTIC_REGRESSION_MAX_ITER = 1_000


def load_feature_archive(
    archive_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load features, labels and original split indices."""
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Feature archive not found: {archive_path}")

    with np.load(archive_path) as archive:
        features = archive["features"].astype(np.float32, copy=False)
        labels = archive["labels"].astype(np.int64, copy=False)
        indices = archive["indices"].astype(np.int64, copy=False)

    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"Feature/label count mismatch in {archive_path}.")

    return features, labels, indices


def build_pipeline(c_value: float, max_iter: int) -> Pipeline:
    """Build the leakage-safe StandardScaler + LinearSVC baseline."""
    if c_value <= 0:
        raise ValueError("LinearSVC C must be greater than zero.")
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than zero.")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearSVC(
                    C=c_value,
                    max_iter=max_iter,
                    dual=False,
                    random_state=SEED,
                ),
            ),
        ]
    )



def build_logistic_regression_pipeline(
    c_value: float = DEFAULT_LOGISTIC_REGRESSION_C,
    max_iter: int = DEFAULT_LOGISTIC_REGRESSION_MAX_ITER,
) -> Pipeline:
    """Build a StandardScaler + LogisticRegression pipeline."""
    if c_value <= 0:
        raise ValueError("LogisticRegression C must be greater than zero.")
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than zero.")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )


def evaluate_pipeline(
    pipeline: Pipeline,
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """Evaluate a fitted pipeline and return metrics and predictions."""
    start_time = time.perf_counter()
    predictions = pipeline.predict(features)
    decision_scores = pipeline.decision_function(features)
    prediction_seconds = time.perf_counter() - start_time

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "prediction_seconds": float(prediction_seconds),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=list(CLASS_IDS),
        ).tolist(),
    }

    report = classification_report(
        labels,
        predictions,
        labels=list(CLASS_IDS),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )

    return metrics, predictions, decision_scores, report


def ensure_outputs_can_be_written(
    output_paths: list[Path],
    overwrite: bool,
) -> None:
    """Avoid replacing experimental artifacts unintentionally."""
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output artifacts already exist. Use --overwrite to replace them."
        )


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)


def run_validation_model_selection(
    output_dir: Path,
    c_values: list[float] | tuple[float, ...] = DEFAULT_LINEAR_SVC_C_VALUES,
    max_iter: int = DEFAULT_LINEAR_SVC_MAX_ITER,
    overwrite: bool = False,
) -> dict:
    """Select LinearSVC C on validation without opening the test split."""
    if not c_values:
        raise ValueError("At least one C value is required.")

    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    results_dir = output_dir / "results"
    models_dir = output_dir / "models"
    predictions_dir = output_dir / "predictions"

    comparison_path = results_dir / "validation_model_selection.csv"
    selected_config_path = results_dir / "selected_baseline.json"
    selected_report_path = (
        results_dir / "selected_validation_classification_report.json"
    )
    selected_model_path = models_dir / "selected_linear_svc_pipeline.joblib"
    selected_predictions_path = (
        predictions_dir / "selected_validation_predictions.npz"
    )

    output_paths = [
        comparison_path,
        selected_config_path,
        selected_report_path,
        selected_model_path,
        selected_predictions_path,
    ]
    ensure_outputs_can_be_written(output_paths, overwrite)

    train_features, train_labels, _ = load_feature_archive(
        features_dir / "train_features.npz"
    )
    validation_features, validation_labels, validation_indices = (
        load_feature_archive(features_dir / "validation_features.npz")
    )

    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("Train and validation feature dimensions differ.")

    print("\n=== Exercise 1.3: validation model selection ===")
    print(f"Train features: {train_features.shape}")
    print(f"Validation features: {validation_features.shape}")
    print(f"Candidate C values: {list(c_values)}")
    print("Selection metric: validation macro-F1")
    print("Test split loaded: False")

    records = []
    best = None

    for c_value in c_values:
        pipeline = build_pipeline(float(c_value), max_iter)

        fit_start = time.perf_counter()
        pipeline.fit(train_features, train_labels)
        fit_seconds = time.perf_counter() - fit_start

        metrics, predictions, scores, report = evaluate_pipeline(
            pipeline,
            validation_features,
            validation_labels,
        )

        record = {
            "c": float(c_value),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "fit_seconds": float(fit_seconds),
            "prediction_seconds": metrics["prediction_seconds"],
            "iterations": int(pipeline.named_steps["classifier"].n_iter_),
        }
        records.append(record)

        candidate_key = (
            record["macro_f1"],
            record["accuracy"],
            -record["c"],
        )
        if best is None or candidate_key > best["key"]:
            best = {
                "key": candidate_key,
                "record": record,
                "pipeline": pipeline,
                "predictions": predictions,
                "scores": scores,
                "report": report,
                "confusion_matrix": metrics["confusion_matrix"],
            }

        print(
            f"C={float(c_value):g} | "
            f"accuracy={record['accuracy']:.6f} | "
            f"macro-F1={record['macro_f1']:.6f}"
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records).to_csv(comparison_path, index=False)

    selected = {
        "selection_split": "validation",
        "selection_metric": "macro_f1",
        "candidate_c_values": [float(value) for value in c_values],
        "selected_c": best["record"]["c"],
        "validation_accuracy": best["record"]["accuracy"],
        "validation_macro_f1": best["record"]["macro_f1"],
        "feature_dimension": int(train_features.shape[1]),
        "max_iter": int(max_iter),
        "confusion_matrix": best["confusion_matrix"],
        "test_used_for_model_selection": False,
    }

    save_json(selected, selected_config_path)
    save_json(best["report"], selected_report_path)
    joblib.dump(best["pipeline"], selected_model_path)
    np.savez_compressed(
        selected_predictions_path,
        indices=validation_indices,
        labels=validation_labels,
        predictions=best["predictions"].astype(np.int64),
        decision_scores=np.asarray(best["scores"], dtype=np.float64),
    )

    print("\n=== Selected baseline ===")
    print(f"C: {selected['selected_c']:g}")
    print(f"Validation accuracy: {selected['validation_accuracy']:.6f}")
    print(f"Validation macro-F1: {selected['validation_macro_f1']:.6f}")
    print(f"Pipeline saved in: {selected_model_path}")

    return selected



def run_logistic_regression_validation_experiment(
    output_dir: Path,
    c_value: float = DEFAULT_LOGISTIC_REGRESSION_C,
    max_iter: int = DEFAULT_LOGISTIC_REGRESSION_MAX_ITER,
    overwrite: bool = False,
) -> dict:
    """Train Logistic Regression on train features and evaluate validation."""
    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    results_dir = output_dir / "results"
    models_dir = output_dir / "models"
    predictions_dir = output_dir / "predictions"

    metrics_path = (
        results_dir / "logistic_regression_validation_metrics.json"
    )
    report_path = (
        results_dir
        / "logistic_regression_validation_classification_report.json"
    )
    model_path = models_dir / "logistic_regression_pipeline.joblib"
    predictions_path = (
        predictions_dir
        / "logistic_regression_validation_predictions.npz"
    )

    ensure_outputs_can_be_written(
        [metrics_path, report_path, model_path, predictions_path],
        overwrite,
    )

    train_features, train_labels, _ = load_feature_archive(
        features_dir / "train_features.npz"
    )
    validation_features, validation_labels, validation_indices = (
        load_feature_archive(features_dir / "validation_features.npz")
    )

    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("Train and validation feature dimensions differ.")

    pipeline = build_logistic_regression_pipeline(
        c_value=c_value,
        max_iter=max_iter,
    )

    print(
        "\n=== Exercise 1.3: Logistic Regression "
        "validation experiment ==="
    )
    print(f"Train features: {train_features.shape}")
    print(f"Validation features: {validation_features.shape}")
    print("Pipeline: StandardScaler + LogisticRegression")
    print("Solver: lbfgs")
    print(f"C: {c_value}")
    print(f"Maximum iterations: {max_iter}")
    print("Test split loaded: False")

    fit_start = time.perf_counter()
    pipeline.fit(train_features, train_labels)
    fit_seconds = time.perf_counter() - fit_start

    metrics, predictions, scores, report = evaluate_pipeline(
        pipeline,
        validation_features,
        validation_labels,
    )

    classifier = pipeline.named_steps["classifier"]

    result = {
        "split": "validation",
        "classifier": "LogisticRegression",
        "preprocessing": "StandardScaler",
        "solver": "lbfgs",
        "c": float(c_value),
        "max_iter": int(max_iter),
        "iterations": int(classifier.n_iter_[0]),
        "train_examples": int(train_features.shape[0]),
        "validation_examples": int(validation_features.shape[0]),
        "feature_dimension": int(train_features.shape[1]),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "fit_seconds": float(fit_seconds),
        "prediction_seconds": metrics["prediction_seconds"],
        "confusion_matrix": metrics["confusion_matrix"],
        "test_split_loaded": False,
    }

    save_json(result, metrics_path)
    save_json(report, report_path)
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    np.savez_compressed(
        predictions_path,
        indices=validation_indices,
        labels=validation_labels,
        predictions=predictions.astype(np.int64),
        decision_scores=np.asarray(scores, dtype=np.float64),
    )

    print("\n=== Logistic Regression validation results ===")
    print(f"Accuracy: {result['accuracy']:.6f}")
    print(f"Macro-F1: {result['macro_f1']:.6f}")
    print(f"Fit time: {result['fit_seconds']:.3f} seconds")
    print(
        "Prediction time: "
        f"{result['prediction_seconds']:.3f} seconds"
    )
    print(
        f"Iterations: {result['iterations']}/{result['max_iter']}"
    )
    print(f"Confusion matrix: {result['confusion_matrix']}")
    print(f"Metrics saved in: {metrics_path}")
    print("The test split was not loaded or evaluated.")

    return result


def run_selected_baseline_test_evaluation(
    output_dir: Path,
    overwrite: bool = False,
) -> dict:
    """Evaluate the validation-selected pipeline once on the test split."""
    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    results_dir = output_dir / "results"
    models_dir = output_dir / "models"
    predictions_dir = output_dir / "predictions"

    selected_model_path = models_dir / "selected_linear_svc_pipeline.joblib"
    selected_config_path = results_dir / "selected_baseline.json"
    test_metrics_path = results_dir / "test_metrics.json"
    test_report_path = results_dir / "test_classification_report.json"
    test_predictions_path = predictions_dir / "test_predictions.npz"

    required_inputs = [
        selected_model_path,
        selected_config_path,
        features_dir / "test_features.npz",
    ]
    missing_inputs = [path for path in required_inputs if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing required inputs: {missing_inputs}")

    ensure_outputs_can_be_written(
        [test_metrics_path, test_report_path, test_predictions_path],
        overwrite,
    )

    with selected_config_path.open("r", encoding="utf-8") as input_file:
        selected = json.load(input_file)

    pipeline = joblib.load(selected_model_path)
    test_features, test_labels, test_indices = load_feature_archive(
        features_dir / "test_features.npz"
    )

    if test_features.shape[1] != selected["feature_dimension"]:
        raise ValueError("Test feature dimension differs from the selected model.")

    metrics, predictions, scores, report = evaluate_pipeline(
        pipeline,
        test_features,
        test_labels,
    )

    test_metrics = {
        "split": "test",
        "selected_c": float(selected["selected_c"]),
        "validation_accuracy": float(selected["validation_accuracy"]),
        "validation_macro_f1": float(selected["validation_macro_f1"]),
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_examples": int(len(test_labels)),
        "feature_dimension": int(test_features.shape[1]),
        "prediction_seconds": metrics["prediction_seconds"],
        "confusion_matrix": metrics["confusion_matrix"],
        "pipeline_refitted_before_test": False,
        "test_used_for_model_selection": False,
    }

    save_json(test_metrics, test_metrics_path)
    save_json(report, test_report_path)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        test_predictions_path,
        indices=test_indices,
        labels=test_labels,
        predictions=predictions.astype(np.int64),
        decision_scores=np.asarray(scores, dtype=np.float64),
    )

    print("\n=== Final test evaluation ===")
    print(f"Accuracy: {test_metrics['test_accuracy']:.6f}")
    print(f"Macro-F1: {test_metrics['test_macro_f1']:.6f}")
    print(f"Confusion matrix: {test_metrics['confusion_matrix']}")
    print(f"Metrics saved in: {test_metrics_path}")

    return test_metrics
