import json
import time
import csv
from pathlib import Path

import joblib
import numpy as np
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

CLASS_IDS = (
    0,
    1,
)

CLASS_NAMES = (
    "neg",
    "pos",
)

DEFAULT_LINEAR_SVC_C = 1.0
DEFAULT_LINEAR_SVC_MAX_ITER = 10_000

DEFAULT_LINEAR_SVC_C_VALUES = (
    0.01,
    0.1,
    1.0,
    10.0,
)

def load_feature_archive(
        archive_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and validate one feature archive.

    The archive must contain:
        features: [number_of_examples, hidden_size]
        labels:   [number_of_examples]
        indices:  [number_of_examples]
    """
    archive_path = Path(archive_path)

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Feature archive not found: {archive_path}"
        )

    with np.load(archive_path) as archive:
        required_fields = {
            "features",
            "labels",
            "indices",
        }

        missing_fields = (
                required_fields
                - set(archive.files)
        )

        if missing_fields:
            raise ValueError(
                f"Archive '{archive_path}' is missing fields: "
                f"{sorted(missing_fields)}"
            )

        features = np.asarray(
            archive["features"],
            dtype=np.float32,
        )

        labels = np.asarray(
            archive["labels"],
            dtype=np.int64,
        )

        indices = np.asarray(
            archive["indices"],
            dtype=np.int64,
        )

    if features.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional feature matrix in "
            f"'{archive_path}', but received "
            f"shape {features.shape}."
        )

    if labels.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional labels in "
            f"'{archive_path}', but received "
            f"shape {labels.shape}."
        )

    if indices.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional indices in "
            f"'{archive_path}', but received "
            f"shape {indices.shape}."
        )

    number_of_examples = features.shape[0]

    if labels.shape[0] != number_of_examples:
        raise ValueError(
            f"Feature and label counts differ in "
            f"'{archive_path}': "
            f"{number_of_examples} != {labels.shape[0]}."
        )

    if indices.shape[0] != number_of_examples:
        raise ValueError(
            f"Feature and index counts differ in "
            f"'{archive_path}': "
            f"{number_of_examples} != {indices.shape[0]}."
        )

    if not np.isfinite(features).all():
        raise ValueError(
            f"Non-finite feature values found in "
            f"'{archive_path}'."
        )

    expected_indices = np.arange(
        number_of_examples,
        dtype=np.int64,
    )

    if not np.array_equal(
            indices,
            expected_indices,
    ):
        raise ValueError(
            f"The indices stored in '{archive_path}' do not "
            "match the original split ordering."
        )

    observed_labels = set(
        np.unique(labels).tolist()
    )

    valid_labels = set(CLASS_IDS)

    if not observed_labels.issubset(valid_labels):
        raise ValueError(
            f"Invalid labels found in '{archive_path}': "
            f"{sorted(observed_labels)}."
        )

    return features, labels, indices


def validate_train_validation_data(
        train_features: np.ndarray,
        train_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
) -> None:
    """
    Verify compatibility between training and validation data.
    """
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError(
            "Train and validation feature dimensions differ: "
            f"{train_features.shape[1]} != "
            f"{validation_features.shape[1]}."
        )

    expected_classes = set(CLASS_IDS)

    train_classes = set(
        np.unique(train_labels).tolist()
    )

    validation_classes = set(
        np.unique(validation_labels).tolist()
    )

    if train_classes != expected_classes:
        raise ValueError(
            "The training split does not contain both expected "
            f"classes: {sorted(train_classes)}."
        )

    if validation_classes != expected_classes:
        raise ValueError(
            "The validation split does not contain both expected "
            f"classes: {sorted(validation_classes)}."
        )


def build_linear_svc_pipeline(
        c_value: float = DEFAULT_LINEAR_SVC_C,
        max_iter: int = DEFAULT_LINEAR_SVC_MAX_ITER,
) -> Pipeline:
    """
    Build a leakage-safe StandardScaler + LinearSVC pipeline.

    Calling fit() on this pipeline fits both components using only
    the supplied training data.
    """
    if c_value <= 0:
        raise ValueError(
            "LinearSVC C must be greater than zero."
        )

    if max_iter <= 0:
        raise ValueError(
            "LinearSVC max_iter must be greater than zero."
        )

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
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


def save_json_atomically(
        data: dict,
        output_path: Path,
) -> None:
    """
    Save a JSON artifact through a temporary file.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    with temporary_path.open(
            "w",
            encoding="utf-8",
    ) as output_file:
        json.dump(
            data,
            output_file,
            indent=2,
        )

    temporary_path.replace(output_path)


def save_predictions_atomically(
        output_path: Path,
        indices: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        decision_scores: np.ndarray,
) -> None:
    """
    Save validation labels, predictions and decision scores.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    np.savez_compressed(
        temporary_path,
        indices=indices,
        labels=labels,
        predictions=predictions,
        decision_scores=decision_scores,
    )

    temporary_path.replace(output_path)


def save_pipeline_atomically(
        pipeline: Pipeline,
        output_path: Path,
) -> None:
    """
    Save the fitted Scikit-learn pipeline atomically.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    joblib.dump(
        pipeline,
        temporary_path,
    )

    temporary_path.replace(output_path)


def run_validation_baseline(
        output_dir: Path,
        c_value: float = DEFAULT_LINEAR_SVC_C,
        max_iter: int = DEFAULT_LINEAR_SVC_MAX_ITER,
        overwrite: bool = False,
) -> dict:
    """
    Train LinearSVC on training features and evaluate validation.

    The test feature archive is deliberately not loaded.
    """
    output_dir = Path(output_dir)

    features_dir = output_dir / "features"
    results_dir = output_dir / "results"
    models_dir = output_dir / "models"
    predictions_dir = output_dir / "predictions"

    train_archive_path = (
            features_dir
            / "train_features.npz"
    )

    validation_archive_path = (
            features_dir
            / "validation_features.npz"
    )

    model_path = (
            models_dir
            / "linear_svc_pipeline.joblib"
    )

    metrics_path = (
            results_dir
            / "validation_metrics.json"
    )

    report_path = (
            results_dir
            / "validation_classification_report.json"
    )

    predictions_path = (
            predictions_dir
            / "validation_predictions.npz"
    )

    expected_output_paths = [
        model_path,
        metrics_path,
        report_path,
        predictions_path,
    ]

    existing_output_paths = [
        path
        for path in expected_output_paths
        if path.exists()
    ]

    if existing_output_paths and not overwrite:
        formatted_paths = "\n".join(
            f"- {path}"
            for path in existing_output_paths
        )

        raise FileExistsError(
            "Validation-baseline outputs already exist:\n"
            f"{formatted_paths}\n"
            "Use --overwrite only to replace them."
        )

    print(
        "\n=== Exercise 1.3: loading feature archives ==="
    )

    (
        train_features,
        train_labels,
        _,
    ) = load_feature_archive(
        train_archive_path
    )

    (
        validation_features,
        validation_labels,
        validation_indices,
    ) = load_feature_archive(
        validation_archive_path
    )

    validate_train_validation_data(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
    )

    print(
        f"Train features: {train_features.shape}, "
        f"dtype={train_features.dtype}"
    )
    print(
        f"Train labels: {train_labels.shape}, "
        f"dtype={train_labels.dtype}"
    )
    print(
        f"Validation features: "
        f"{validation_features.shape}, "
        f"dtype={validation_features.dtype}"
    )
    print(
        f"Validation labels: "
        f"{validation_labels.shape}, "
        f"dtype={validation_labels.dtype}"
    )
    print("Finite feature values: True")
    print(
        "Test archive loaded: False"
    )

    pipeline = build_linear_svc_pipeline(
        c_value=c_value,
        max_iter=max_iter,
    )

    print(
        "\n=== Exercise 1.3: training configuration ==="
    )
    print("Classifier: LinearSVC")
    print("Preprocessing: StandardScaler")
    print(
        "Scaler fit data: training split only"
    )
    print(f"C: {c_value}")
    print(f"max_iter: {max_iter}")
    print("dual: False")
    print(f"random_state: {SEED}")
    print("class_weight: None")

    fit_start_time = time.perf_counter()

    pipeline.fit(
        train_features,
        train_labels,
    )

    fit_seconds = (
            time.perf_counter()
            - fit_start_time
    )

    prediction_start_time = time.perf_counter()

    validation_predictions = pipeline.predict(
        validation_features
    )

    validation_decision_scores = (
        pipeline.decision_function(
            validation_features
        )
    )

    prediction_seconds = (
            time.perf_counter()
            - prediction_start_time
    )

    validation_accuracy = accuracy_score(
        validation_labels,
        validation_predictions,
    )

    validation_macro_f1 = f1_score(
        validation_labels,
        validation_predictions,
        average="macro",
    )

    report = classification_report(
        validation_labels,
        validation_predictions,
        labels=list(CLASS_IDS),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        validation_labels,
        validation_predictions,
        labels=list(CLASS_IDS),
    )

    classifier = pipeline.named_steps[
        "classifier"
    ]

    number_of_iterations = int(
        classifier.n_iter_
    )

    metrics = {
        "split": "validation",
        "classifier": "LinearSVC",
        "preprocessing": "StandardScaler",
        "c": float(c_value),
        "max_iter": int(max_iter),
        "dual": False,
        "random_state": SEED,
        "class_weight": None,
        "train_examples": int(
            train_features.shape[0]
        ),
        "validation_examples": int(
            validation_features.shape[0]
        ),
        "feature_dimension": int(
            train_features.shape[1]
        ),
        "accuracy": float(
            validation_accuracy
        ),
        "macro_f1": float(
            validation_macro_f1
        ),
        "fit_seconds": float(
            fit_seconds
        ),
        "prediction_seconds": float(
            prediction_seconds
        ),
        "number_of_iterations": (
            number_of_iterations
        ),
        "converged_within_max_iter": (
                number_of_iterations < max_iter
        ),
        "confusion_matrix": matrix.tolist(),
        "test_archive_loaded": False,
    }

    save_pipeline_atomically(
        pipeline=pipeline,
        output_path=model_path,
    )

    save_json_atomically(
        data=metrics,
        output_path=metrics_path,
    )

    save_json_atomically(
        data=report,
        output_path=report_path,
    )

    save_predictions_atomically(
        output_path=predictions_path,
        indices=validation_indices,
        labels=validation_labels,
        predictions=validation_predictions.astype(
            np.int64
        ),
        decision_scores=np.asarray(
            validation_decision_scores,
            dtype=np.float64,
        ),
    )

    print(
        "\n=== Validation results ==="
    )
    print(
        f"Accuracy: "
        f"{validation_accuracy:.6f}"
    )
    print(
        f"Macro-F1: "
        f"{validation_macro_f1:.6f}"
    )
    print(
        f"Fit time: {fit_seconds:.3f} seconds"
    )
    print(
        "Prediction time: "
        f"{prediction_seconds:.3f} seconds"
    )
    print(
        f"LinearSVC iterations: "
        f"{number_of_iterations}/{max_iter}"
    )
    print(
        "Converged within max_iter: "
        f"{number_of_iterations < max_iter}"
    )

    print(
        "\nConfusion matrix "
        "[[true neg, false pos], "
        "[false neg, true pos]]:"
    )
    print(matrix)

    print(
        "\n=== Saved artifacts ==="
    )
    print(f"Pipeline: {model_path}")
    print(f"Metrics: {metrics_path}")
    print(
        f"Classification report: {report_path}"
    )
    print(
        f"Validation predictions: "
        f"{predictions_path}"
    )

    print(
        "\nValidation completed."
    )
    print(
        "The test feature archive was not loaded or evaluated."
    )

    return metrics

def save_csv_atomically(
        records: list[dict],
        field_names: list[str],
        output_path: Path,
) -> None:
    """
    Save tabular experiment results through a temporary CSV file.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=field_names,
        )

        writer.writeheader()
        writer.writerows(records)

    temporary_path.replace(output_path)


def run_validation_model_selection(
        output_dir: Path,
        c_values: list[float] | tuple[float, ...] = (
                DEFAULT_LINEAR_SVC_C_VALUES
        ),
        max_iter: int = DEFAULT_LINEAR_SVC_MAX_ITER,
        overwrite: bool = False,
) -> dict:
    """
    Select the LinearSVC C value using only the validation split.

    The test feature archive is deliberately not loaded.
    """
    output_dir = Path(output_dir)

    if not c_values:
        raise ValueError(
            "At least one C value is required."
        )

    normalized_c_values = [
        float(c_value)
        for c_value in c_values
    ]

    if any(
            c_value <= 0
            for c_value in normalized_c_values
    ):
        raise ValueError(
            "Every LinearSVC C value must be greater than zero."
        )

    if len(set(normalized_c_values)) != len(
            normalized_c_values
    ):
        raise ValueError(
            "Duplicate C values were provided."
        )

    if max_iter <= 0:
        raise ValueError(
            "max_iter must be greater than zero."
        )

    features_dir = output_dir / "features"
    results_dir = output_dir / "results"
    models_dir = output_dir / "models"
    predictions_dir = output_dir / "predictions"

    train_archive_path = (
            features_dir
            / "train_features.npz"
    )

    validation_archive_path = (
            features_dir
            / "validation_features.npz"
    )

    comparison_path = (
            results_dir
            / "validation_model_selection.csv"
    )

    selected_config_path = (
            results_dir
            / "selected_baseline.json"
    )

    selected_report_path = (
            results_dir
            / "selected_validation_classification_report.json"
    )

    selected_model_path = (
            models_dir
            / "selected_linear_svc_pipeline.joblib"
    )

    selected_predictions_path = (
            predictions_dir
            / "selected_validation_predictions.npz"
    )

    output_paths = [
        comparison_path,
        selected_config_path,
        selected_report_path,
        selected_model_path,
        selected_predictions_path,
    ]

    existing_paths = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing_paths and not overwrite:
        formatted_paths = "\n".join(
            f"- {path}"
            for path in existing_paths
        )

        raise FileExistsError(
            "Model-selection outputs already exist:\n"
            f"{formatted_paths}\n"
            "Use --overwrite only to replace them."
        )

    print(
        "\n=== Exercise 1.3: loading model-selection data ==="
    )

    (
        train_features,
        train_labels,
        _,
    ) = load_feature_archive(
        train_archive_path
    )

    (
        validation_features,
        validation_labels,
        validation_indices,
    ) = load_feature_archive(
        validation_archive_path
    )

    validate_train_validation_data(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
    )

    print(f"Train features: {train_features.shape}")
    print(
        f"Validation features: "
        f"{validation_features.shape}"
    )
    print(
        f"Candidate C values: "
        f"{normalized_c_values}"
    )
    print(f"Maximum iterations: {max_iter}")
    print("Selection metric: validation macro-F1")
    print("Test archive loaded: False")

    experiment_records: list[dict] = []

    best_record: dict | None = None
    best_pipeline: Pipeline | None = None
    best_predictions: np.ndarray | None = None
    best_decision_scores: np.ndarray | None = None

    print(
        "\n=== Validation model selection ==="
    )

    for c_value in normalized_c_values:
        pipeline = build_linear_svc_pipeline(
            c_value=c_value,
            max_iter=max_iter,
        )

        fit_start_time = time.perf_counter()

        pipeline.fit(
            train_features,
            train_labels,
        )

        fit_seconds = (
                time.perf_counter()
                - fit_start_time
        )

        prediction_start_time = time.perf_counter()

        validation_predictions = pipeline.predict(
            validation_features
        )

        validation_decision_scores = (
            pipeline.decision_function(
                validation_features
            )
        )

        prediction_seconds = (
                time.perf_counter()
                - prediction_start_time
        )

        validation_accuracy = accuracy_score(
            validation_labels,
            validation_predictions,
        )

        validation_macro_f1 = f1_score(
            validation_labels,
            validation_predictions,
            average="macro",
        )

        classifier = pipeline.named_steps[
            "classifier"
        ]

        number_of_iterations = int(
            classifier.n_iter_
        )

        record = {
            "c": float(c_value),
            "accuracy": float(
                validation_accuracy
            ),
            "macro_f1": float(
                validation_macro_f1
            ),
            "fit_seconds": float(
                fit_seconds
            ),
            "prediction_seconds": float(
                prediction_seconds
            ),
            "iterations": number_of_iterations,
            "max_iter": int(max_iter),
            "converged": (
                    number_of_iterations < max_iter
            ),
        }

        experiment_records.append(record)

        print(
            f"C={c_value:g} | "
            f"accuracy={validation_accuracy:.6f} | "
            f"macro-F1={validation_macro_f1:.6f} | "
            f"iterations={number_of_iterations} | "
            f"fit={fit_seconds:.3f}s"
        )

        candidate_key = (
            record["macro_f1"],
            record["accuracy"],
            -record["c"],
        )

        if best_record is None:
            is_better = True
        else:
            best_key = (
                best_record["macro_f1"],
                best_record["accuracy"],
                -best_record["c"],
            )

            is_better = candidate_key > best_key

        if is_better:
            best_record = record
            best_pipeline = pipeline
            best_predictions = (
                validation_predictions.astype(
                    np.int64
                )
            )
            best_decision_scores = np.asarray(
                validation_decision_scores,
                dtype=np.float64,
            )

    if (
            best_record is None
            or best_pipeline is None
            or best_predictions is None
            or best_decision_scores is None
    ):
        raise RuntimeError(
            "No valid model was selected."
        )

    selected_matrix = confusion_matrix(
        validation_labels,
        best_predictions,
        labels=list(CLASS_IDS),
    )

    selected_report = classification_report(
        validation_labels,
        best_predictions,
        labels=list(CLASS_IDS),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )

    selected_configuration = {
        "selection_split": "validation",
        "selection_metric": "macro_f1",
        "tie_breakers": [
            "accuracy",
            "smaller_c",
        ],
        "classifier": "LinearSVC",
        "preprocessing": "StandardScaler",
        "scaler_fit_data": "train_only",
        "candidate_c_values": (
            normalized_c_values
        ),
        "selected_c": best_record["c"],
        "validation_accuracy": (
            best_record["accuracy"]
        ),
        "validation_macro_f1": (
            best_record["macro_f1"]
        ),
        "fit_seconds": (
            best_record["fit_seconds"]
        ),
        "prediction_seconds": (
            best_record["prediction_seconds"]
        ),
        "iterations": (
            best_record["iterations"]
        ),
        "max_iter": int(max_iter),
        "converged": (
            best_record["converged"]
        ),
        "confusion_matrix": (
            selected_matrix.tolist()
        ),
        "train_examples": int(
            train_features.shape[0]
        ),
        "validation_examples": int(
            validation_features.shape[0]
        ),
        "feature_dimension": int(
            train_features.shape[1]
        ),
        "test_archive_loaded": False,
    }

    save_csv_atomically(
        records=experiment_records,
        field_names=[
            "c",
            "accuracy",
            "macro_f1",
            "fit_seconds",
            "prediction_seconds",
            "iterations",
            "max_iter",
            "converged",
        ],
        output_path=comparison_path,
    )

    save_json_atomically(
        data=selected_configuration,
        output_path=selected_config_path,
    )

    save_json_atomically(
        data=selected_report,
        output_path=selected_report_path,
    )

    save_pipeline_atomically(
        pipeline=best_pipeline,
        output_path=selected_model_path,
    )

    save_predictions_atomically(
        output_path=selected_predictions_path,
        indices=validation_indices,
        labels=validation_labels,
        predictions=best_predictions,
        decision_scores=best_decision_scores,
    )

    print(
        "\n=== Selected validation configuration ==="
    )
    print(
        f"Selected C: {best_record['c']:g}"
    )
    print(
        "Validation accuracy: "
        f"{best_record['accuracy']:.6f}"
    )
    print(
        "Validation macro-F1: "
        f"{best_record['macro_f1']:.6f}"
    )
    print(
        f"Iterations: "
        f"{best_record['iterations']}/{max_iter}"
    )
    print(
        f"Converged: {best_record['converged']}"
    )

    print(
        "\nSelected confusion matrix:"
    )
    print(selected_matrix)

    print(
        "\n=== Saved model-selection artifacts ==="
    )
    print(f"Comparison: {comparison_path}")
    print(
        f"Selected configuration: "
        f"{selected_config_path}"
    )
    print(
        f"Selected classification report: "
        f"{selected_report_path}"
    )
    print(
        f"Selected pipeline: "
        f"{selected_model_path}"
    )
    print(
        f"Selected validation predictions: "
        f"{selected_predictions_path}"
    )

    print(
        "\nModel selection completed."
    )
    print(
        "The test feature archive was not loaded or evaluated."
    )

    return selected_configuration

def load_json(
        input_path: Path,
) -> dict:
    """
    Load a JSON object from disk.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {input_path}"
        )

    with input_path.open(
            "r",
            encoding="utf-8",
    ) as input_file:
        data = json.load(input_file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in '{input_path}'."
        )

    return data


def run_selected_baseline_test_evaluation(
        output_dir: Path,
        overwrite: bool = False,
) -> dict:
    """
    Evaluate the validation-selected pipeline on the test split.

    The saved pipeline is used without fitting or modifying it.
    No hyperparameter selection is performed on the test data.
    """
    output_dir = Path(output_dir)

    features_dir = output_dir / "features"
    models_dir = output_dir / "models"
    results_dir = output_dir / "results"
    predictions_dir = output_dir / "predictions"

    selected_model_path = (
            models_dir
            / "selected_linear_svc_pipeline.joblib"
    )

    selected_config_path = (
            results_dir
            / "selected_baseline.json"
    )

    test_archive_path = (
            features_dir
            / "test_features.npz"
    )

    test_metrics_path = (
            results_dir
            / "test_metrics.json"
    )

    test_report_path = (
            results_dir
            / "test_classification_report.json"
    )

    test_predictions_path = (
            predictions_dir
            / "test_predictions.npz"
    )

    required_input_paths = [
        selected_model_path,
        selected_config_path,
        test_archive_path,
    ]

    missing_input_paths = [
        path
        for path in required_input_paths
        if not path.exists()
    ]

    if missing_input_paths:
        formatted_paths = "\n".join(
            f"- {path}"
            for path in missing_input_paths
        )

        raise FileNotFoundError(
            "Required test-evaluation inputs are missing:\n"
            f"{formatted_paths}"
        )

    output_paths = [
        test_metrics_path,
        test_report_path,
        test_predictions_path,
    ]

    existing_output_paths = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing_output_paths and not overwrite:
        formatted_paths = "\n".join(
            f"- {path}"
            for path in existing_output_paths
        )

        raise FileExistsError(
            "Test-evaluation outputs already exist:\n"
            f"{formatted_paths}\n"
            "The test should normally be evaluated only once. "
            "Use --overwrite only to reproduce the exact same "
            "evaluation intentionally."
        )

    print(
        "\n=== Exercise 1.3: loading selected baseline ==="
    )

    selected_configuration = load_json(
        selected_config_path
    )

    pipeline = joblib.load(
        selected_model_path
    )

    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            "The selected model artifact is not a "
            "Scikit-learn Pipeline."
        )

    required_pipeline_steps = {
        "scaler",
        "classifier",
    }

    observed_pipeline_steps = set(
        pipeline.named_steps
    )

    if not required_pipeline_steps.issubset(
            observed_pipeline_steps
    ):
        raise ValueError(
            "The selected pipeline does not contain the "
            "required scaler and classifier steps."
        )

    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]

    if not isinstance(scaler, StandardScaler):
        raise TypeError(
            "The selected pipeline scaler is not "
            "StandardScaler."
        )

    if not isinstance(classifier, LinearSVC):
        raise TypeError(
            "The selected pipeline classifier is not "
            "LinearSVC."
        )

    selected_c = float(
        selected_configuration["selected_c"]
    )

    fitted_c = float(classifier.C)

    if not np.isclose(
            selected_c,
            fitted_c,
    ):
        raise ValueError(
            "The C value stored in the selected configuration "
            "does not match the fitted classifier: "
            f"{selected_c} != {fitted_c}."
        )

    print(f"Pipeline type: {type(pipeline).__name__}")
    print(f"Scaler type: {type(scaler).__name__}")
    print(
        f"Classifier type: "
        f"{type(classifier).__name__}"
    )
    print(f"Selected C: {selected_c}")
    print(
        "Model-selection split: "
        f"{selected_configuration['selection_split']}"
    )
    print(
        "Model-selection metric: "
        f"{selected_configuration['selection_metric']}"
    )
    print(
        "Pipeline refitted before test: False"
    )

    print(
        "\n=== Exercise 1.3: loading test features ==="
    )

    (
        test_features,
        test_labels,
        test_indices,
    ) = load_feature_archive(
        test_archive_path
    )

    expected_feature_dimension = int(
        selected_configuration[
            "feature_dimension"
        ]
    )

    if test_features.shape[1] != expected_feature_dimension:
        raise ValueError(
            "The test feature dimension does not match the "
            "selected model configuration: "
            f"{test_features.shape[1]} != "
            f"{expected_feature_dimension}."
        )

    if hasattr(
            scaler,
            "n_features_in_",
    ):
        if (
                int(scaler.n_features_in_)
                != test_features.shape[1]
        ):
            raise ValueError(
                "The fitted scaler expects a different "
                "feature dimension."
            )

    print(
        f"Test features: {test_features.shape}, "
        f"dtype={test_features.dtype}"
    )
    print(
        f"Test labels: {test_labels.shape}, "
        f"dtype={test_labels.dtype}"
    )
    print("Finite feature values: True")
    print(
        "No fit operation will be executed."
    )

    prediction_start_time = time.perf_counter()

    test_predictions = pipeline.predict(
        test_features
    )

    test_decision_scores = (
        pipeline.decision_function(
            test_features
        )
    )

    prediction_seconds = (
            time.perf_counter()
            - prediction_start_time
    )

    test_accuracy = accuracy_score(
        test_labels,
        test_predictions,
    )

    test_macro_f1 = f1_score(
        test_labels,
        test_predictions,
        average="macro",
    )

    test_matrix = confusion_matrix(
        test_labels,
        test_predictions,
        labels=list(CLASS_IDS),
    )

    test_report = classification_report(
        test_labels,
        test_predictions,
        labels=list(CLASS_IDS),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )

    test_metrics = {
        "split": "test",
        "classifier": "LinearSVC",
        "preprocessing": "StandardScaler",
        "selected_c": selected_c,
        "selection_split": (
            selected_configuration[
                "selection_split"
            ]
        ),
        "selection_metric": (
            selected_configuration[
                "selection_metric"
            ]
        ),
        "validation_accuracy": float(
            selected_configuration[
                "validation_accuracy"
            ]
        ),
        "validation_macro_f1": float(
            selected_configuration[
                "validation_macro_f1"
            ]
        ),
        "test_accuracy": float(
            test_accuracy
        ),
        "test_macro_f1": float(
            test_macro_f1
        ),
        "test_examples": int(
            test_features.shape[0]
        ),
        "feature_dimension": int(
            test_features.shape[1]
        ),
        "prediction_seconds": float(
            prediction_seconds
        ),
        "confusion_matrix": (
            test_matrix.tolist()
        ),
        "pipeline_refitted_before_test": False,
        "test_used_for_model_selection": False,
    }

    save_json_atomically(
        data=test_metrics,
        output_path=test_metrics_path,
    )

    save_json_atomically(
        data=test_report,
        output_path=test_report_path,
    )

    save_predictions_atomically(
        output_path=test_predictions_path,
        indices=test_indices,
        labels=test_labels,
        predictions=test_predictions.astype(
            np.int64
        ),
        decision_scores=np.asarray(
            test_decision_scores,
            dtype=np.float64,
        ),
    )

    print("\n=== Final test results ===")
    print(
        f"Accuracy: {test_accuracy:.6f}"
    )
    print(
        f"Macro-F1: {test_macro_f1:.6f}"
    )
    print(
        "Prediction time: "
        f"{prediction_seconds:.3f} seconds"
    )

    print(
        "\nConfusion matrix "
        "[[true neg, false pos], "
        "[false neg, true pos]]:"
    )
    print(test_matrix)

    print(
        "\n=== Validation-test comparison ==="
    )
    print(
        "Validation accuracy: "
        f"{selected_configuration['validation_accuracy']:.6f}"
    )
    print(
        f"Test accuracy: {test_accuracy:.6f}"
    )
    print(
        "Validation macro-F1: "
        f"{selected_configuration['validation_macro_f1']:.6f}"
    )
    print(
        f"Test macro-F1: {test_macro_f1:.6f}"
    )

    print("\n=== Saved test artifacts ===")
    print(f"Test metrics: {test_metrics_path}")
    print(
        f"Test classification report: "
        f"{test_report_path}"
    )
    print(
        f"Test predictions: "
        f"{test_predictions_path}"
    )

    print(
        "\nFinal test evaluation completed."
    )
    print(
        "The selected pipeline was not refitted or modified."
    )

    return test_metrics