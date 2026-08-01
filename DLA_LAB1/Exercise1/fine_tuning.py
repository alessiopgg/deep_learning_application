import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)

from data import extract_labels, load_gtsrb


SEED = 42
VALIDATION_SIZE = 0.20
NUM_CLASSES = 43

BACKBONE_LEARNING_RATE = 1e-4
CLASSIFIER_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 5

# Print and optionally log the recent training metrics every N batches.
LOG_INTERVAL = 50

FINE_TUNING_STRATEGIES = (
    "classifier",
    "last_block",
    "full",
)
DEFAULT_FINE_TUNING_STRATEGY = "last_block"

CLASSIFIER_TYPES = (
    "linear",
    "mlp",
)
DEFAULT_CLASSIFIER_TYPE = "linear"

MLP_HIDDEN_FEATURES = 256
MLP_DROPOUT = 0.3

WANDB_PROJECT = "dla-lab1"
WANDB_GROUP = "exercise-1-3"

EXERCISE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXERCISE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"

RESULTS_DIR = (
        EXERCISE_DIR
        / "outputs"
        / "exercise_1_3"
        / "results"
)
RUNS_DIR = RESULTS_DIR / "runs"
EXPERIMENTS_CSV_PATH = RESULTS_DIR / "experiments.csv"

MODEL_CONFIGS = {
    "resnet18": {
        "constructor": resnet18,
        "weights": ResNet18_Weights.IMAGENET1K_V1,
        "batch_size": 32,
    },
    "resnet50": {
        "constructor": resnet50,
        "weights": ResNet50_Weights.IMAGENET1K_V2,
        "batch_size": 16,
    },
}


def set_random_seed(seed=SEED):
    """Set NumPy and PyTorch seeds for reproducible experiments."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(
        model_name,
        validation_size=VALIDATION_SIZE,
        seed=SEED,
        batch_size=None,
):
    """
    Create stratified training/validation splits and the test DataLoader.

    The preprocessing associated with the selected pretrained model is
    applied to all three splits.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODEL_CONFIGS[model_name]
    transform = config["weights"].transforms()
    if batch_size is None:
        batch_size = config["batch_size"]

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )
    train_dataset, test_dataset = load_gtsrb(
        DATA_DIR,
        transform=transform,
    )

    labels = extract_labels(train_dataset)
    all_indices = np.arange(len(train_dataset))

    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=validation_size,
        random_state=seed,
        stratify=labels,
    )

    train_subset = Subset(
        train_dataset,
        train_indices,
    )
    validation_subset = Subset(
        train_dataset,
        validation_indices,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    pin_memory = device.type == "cuda"

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    print("\n=== Exercise 1.3 data preparation ===")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Original training images: {len(train_dataset)}")
    print(f"Training subset images: {len(train_subset)}")
    print(f"Validation subset images: {len(validation_subset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Test batches: {len(test_loader)}")

    dataset_info = {
        "batch_size": int(batch_size),
        "original_train_samples": int(len(train_dataset)),
        "train_samples": int(len(train_subset)),
        "validation_samples": int(len(validation_subset)),
        "test_samples": int(len(test_dataset)),
    }

    return (
        train_loader,
        validation_loader,
        test_loader,
        device,
        dataset_info,
    )


def configure_trainable_layers(model, strategy):
    """Configure which ResNet parameters are trainable."""
    if strategy not in FINE_TUNING_STRATEGIES:
        raise ValueError(
            f"Unknown fine-tuning strategy: {strategy}"
        )

    for parameter in model.parameters():
        parameter.requires_grad = False

    if strategy == "classifier":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    elif strategy == "last_block":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    elif strategy == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True


def create_fine_tuning_model(
        model_name,
        device,
        strategy=DEFAULT_FINE_TUNING_STRATEGY,
        classifier_type=DEFAULT_CLASSIFIER_TYPE,
        num_classes=NUM_CLASSES,
):
    """Create the selected pretrained ResNet and replace its classifier."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    if classifier_type not in CLASSIFIER_TYPES:
        raise ValueError(
            f"Unknown classifier type: {classifier_type}"
        )

    config = MODEL_CONFIGS[model_name]
    model = config["constructor"](
        weights=config["weights"]
    )

    classifier_input_features = model.fc.in_features

    if classifier_type == "linear":
        model.fc = nn.Linear(
            in_features=classifier_input_features,
            out_features=num_classes,
        )

    elif classifier_type == "mlp":
        model.fc = nn.Sequential(
            nn.Linear(
                in_features=classifier_input_features,
                out_features=MLP_HIDDEN_FEATURES,
            ),
            nn.ReLU(),
            nn.Dropout(p=MLP_DROPOUT),
            nn.Linear(
                in_features=MLP_HIDDEN_FEATURES,
                out_features=num_classes,
            ),
        )

    configure_trainable_layers(
        model=model,
        strategy=strategy,
    )

    model = model.to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    trainable_percentage = (
            trainable_parameters / total_parameters * 100
    )

    trainable_modules = sorted(
        {
            name.split(".")[0]
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
    )

    print("\n=== Exercise 1.3 model preparation ===")
    print(f"Model: {model_name}")
    print(f"Classifier type: {classifier_type}")
    print(f"Fine-tuning strategy: {strategy}")
    print(
        "Classifier input features:",
        classifier_input_features,
    )

    if classifier_type == "mlp":
        print("MLP hidden features:", MLP_HIDDEN_FEATURES)
        print("MLP dropout:", MLP_DROPOUT)

    print(f"Classifier output classes: {num_classes}")
    print(f"Trainable modules: {trainable_modules}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Trainable percentage: {trainable_percentage:.2f}%")

    model_info = {
        "classifier_input_features": int(
            classifier_input_features
        ),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "trainable_percentage": float(trainable_percentage),
        "trainable_modules": trainable_modules,
    }

    return model, model_info


def set_model_training_mode(model, strategy):
    """
    Keep frozen modules in evaluation mode during selective fine-tuning.

    This prevents the running statistics of frozen BatchNorm layers from
    being updated.
    """
    model.train()

    if strategy == "full":
        return

    frozen_modules = [
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
    ]

    if strategy == "classifier":
        frozen_modules.append(model.layer4)

    for module in frozen_modules:
        module.eval()

    if strategy == "last_block":
        model.layer4.train()

    model.fc.train()


def create_training_components(
        model,
        strategy,
        backbone_learning_rate=BACKBONE_LEARNING_RATE,
        classifier_learning_rate=CLASSIFIER_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
):
    """Create CrossEntropyLoss and an AdamW optimizer."""
    criterion = nn.CrossEntropyLoss()

    if strategy == "classifier":
        parameter_groups = [
            {
                "params": model.fc.parameters(),
                "lr": classifier_learning_rate,
            },
        ]

    elif strategy == "last_block":
        parameter_groups = [
            {
                "params": model.layer4.parameters(),
                "lr": backbone_learning_rate,
            },
            {
                "params": model.fc.parameters(),
                "lr": classifier_learning_rate,
            },
        ]

    elif strategy == "full":
        backbone_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("fc.")
        ]

        parameter_groups = [
            {
                "params": backbone_parameters,
                "lr": backbone_learning_rate,
            },
            {
                "params": model.fc.parameters(),
                "lr": classifier_learning_rate,
            },
        ]

    else:
        raise ValueError(
            f"Unknown fine-tuning strategy: {strategy}"
        )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=weight_decay,
    )

    print("\n=== Exercise 1.3 training configuration ===")
    print("Loss function: CrossEntropyLoss")
    print("Optimizer: AdamW")
    print("Backbone learning rate:", backbone_learning_rate)
    print("Classifier learning rate:", classifier_learning_rate)
    print(f"Weight decay: {weight_decay}")
    print(f"Batch logging interval: {LOG_INTERVAL}")

    return criterion, optimizer


def train_one_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
        strategy,
        wandb_run=None,
        global_step=0,
        log_interval=LOG_INTERVAL,
):
    """
    Train the model for one complete epoch.

    Epoch-level metrics are calculated over the complete training set.
    Recent loss and accuracy are printed and optionally logged to W&B
    every ``log_interval`` batches.
    """
    if log_interval <= 0:
        raise ValueError(
            "log_interval must be greater than zero."
        )

    set_model_training_mode(
        model=model,
        strategy=strategy,
    )

    total_loss = 0.0
    total_correct_predictions = 0
    total_images = 0

    interval_loss = 0.0
    interval_correct_predictions = 0
    interval_images = 0

    for batch_number, (images, labels) in enumerate(
            dataloader,
            start=1,
    ):
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)
        batch_size = labels.size(0)

        correct_predictions = (
                predictions == labels
        ).sum().item()

        # Statistics for the complete epoch.
        total_loss += loss.item() * batch_size
        total_correct_predictions += correct_predictions
        total_images += batch_size

        # Statistics for the current logging interval.
        interval_loss += loss.item() * batch_size
        interval_correct_predictions += correct_predictions
        interval_images += batch_size

        global_step += 1

        should_log = (
                batch_number % log_interval == 0
                or batch_number == len(dataloader)
        )

        if should_log:
            recent_loss = interval_loss / interval_images
            recent_accuracy = (
                    interval_correct_predictions / interval_images
            )

            print(
                f"Batch {batch_number}/{len(dataloader)} | "
                f"Recent loss: {recent_loss:.4f} | "
                f"Recent accuracy: {recent_accuracy:.4f}"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "global_step": global_step,
                        "train_batch/loss": float(
                            recent_loss
                        ),
                        "train_batch/accuracy": float(
                            recent_accuracy
                        ),
                    }
                )

            interval_loss = 0.0
            interval_correct_predictions = 0
            interval_images = 0

    average_loss = total_loss / total_images
    accuracy = total_correct_predictions / total_images

    return average_loss, accuracy, global_step


def evaluate(
        model,
        dataloader,
        criterion,
        device,
        return_predictions=False,
):
    """Evaluate the model without gradients or parameter updates."""
    model.eval()

    total_loss = 0.0
    total_correct_predictions = 0
    total_images = 0

    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct_predictions += (
                    predictions == labels
            ).sum().item()
            total_images += batch_size

            if return_predictions:
                all_labels.append(
                    labels.cpu().numpy()
                )
                all_predictions.append(
                    predictions.cpu().numpy()
                )

    average_loss = total_loss / total_images
    accuracy = total_correct_predictions / total_images

    if not return_predictions:
        return average_loss, accuracy

    true_labels = np.concatenate(
        all_labels,
        axis=0,
    )
    predicted_labels = np.concatenate(
        all_predictions,
        axis=0,
    )

    return (
        average_loss,
        accuracy,
        true_labels,
        predicted_labels,
    )


def move_to_cpu(value):
    """Recursively move tensors in a nested structure to the CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()

    if isinstance(value, dict):
        return {
            key: move_to_cpu(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            move_to_cpu(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            move_to_cpu(item)
            for item in value
        )

    return value


def save_json(file_path, data):
    """Save a dictionary as an indented JSON file."""
    with file_path.open(
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def save_history(file_path, history):
    """Save epoch-level training history as CSV."""
    fieldnames = [
        "epoch",
        "training_loss",
        "training_accuracy",
        "validation_loss",
        "validation_accuracy",
        "epoch_seconds",
    ]

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

        for epoch_index in range(
                len(history["epoch"])
        ):
            writer.writerow(
                {
                    key: history[key][epoch_index]
                    for key in fieldnames
                }
            )


def save_classification_report(file_path, report):
    """Save class-level precision, recall and F1-score as CSV."""
    fieldnames = [
        "label",
        "precision",
        "recall",
        "f1_score",
        "support",
    ]

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

        for label, values in report.items():
            if not isinstance(values, dict):
                continue

            writer.writerow(
                {
                    "label": label,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1_score": values["f1-score"],
                    "support": values["support"],
                }
            )


def append_experiment_summary(result):
    """Append one completed fine-tuning run to experiments.csv."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "run_id",
        "timestamp",
        "model",
        "classifier_type",
        "strategy",
        "num_epochs",
        "best_epoch",
        "best_validation_loss",
        "best_validation_accuracy",
        "test_loss",
        "test_accuracy",
        "test_macro_f1",
        "total_training_seconds",
        "trainable_parameters",
        "wandb_run_id",
    ]

    file_exists = EXPERIMENTS_CSV_PATH.exists()

    with EXPERIMENTS_CSV_PATH.open(
            "a",
            newline="",
            encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


def save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        config,
        history,
        best_epoch,
        best_validation_loss,
        best_validation_accuracy,
):
    """Save the best model and the information needed to reuse it."""
    checkpoint = {
        "model_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "optimizer_state_dict": move_to_cpu(
            optimizer.state_dict()
        ),
        "config": config,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "best_validation_accuracy": float(
            best_validation_accuracy
        ),
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


def train_model(
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        device,
        strategy,
        run_dir,
        config,
        wandb_run=None,
        num_epochs=NUM_EPOCHS,
):
    """
    Train the model and select the best checkpoint by validation loss.
    """
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_epoch = 0

    history = {
        "epoch": [],
        "training_loss": [],
        "training_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "epoch_seconds": [],
    }

    checkpoint_path = run_dir / "best_model.pt"
    training_start = perf_counter()

    # Counts all training batches across every epoch.
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        epoch_start = perf_counter()

        print(f"\n=== Epoch {epoch}/{num_epochs} ===")

        (
            training_loss,
            training_accuracy,
            global_step,
        ) = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            strategy=strategy,
            wandb_run=wandb_run,
            global_step=global_step,
            log_interval=LOG_INTERVAL,
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        epoch_seconds = perf_counter() - epoch_start

        history["epoch"].append(epoch)
        history["training_loss"].append(
            float(training_loss)
        )
        history["training_accuracy"].append(
            float(training_accuracy)
        )
        history["validation_loss"].append(
            float(validation_loss)
        )
        history["validation_accuracy"].append(
            float(validation_accuracy)
        )
        history["epoch_seconds"].append(
            float(epoch_seconds)
        )

        print(f"Training loss: {training_loss:.4f}")
        print(f"Training accuracy: {training_accuracy:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation accuracy: {validation_accuracy:.4f}")
        print(f"Epoch time: {epoch_seconds:.2f} seconds")

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": float(
                        training_loss
                    ),
                    "train/accuracy": float(
                        training_accuracy
                    ),
                    "validation/loss": float(
                        validation_loss
                    ),
                    "validation/accuracy": float(
                        validation_accuracy
                    ),
                    "timing/epoch_seconds": float(
                        epoch_seconds
                    ),
                }
            )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch

            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                history=history,
                best_epoch=best_epoch,
                best_validation_loss=(
                    best_validation_loss
                ),
                best_validation_accuracy=(
                    best_validation_accuracy
                ),
            )

            print(
                "New best model saved "
                f"(epoch {best_epoch}, validation loss: "
                f"{best_validation_loss:.4f})"
            )

    total_training_seconds = (
            perf_counter() - training_start
    )

    if best_epoch == 0 or not checkpoint_path.exists():
        raise RuntimeError(
            "Training finished without saving a best model."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    save_history(
        run_dir / "history.csv",
        history,
        )

    print("\n=== Training completed ===")
    print(f"Best epoch: {best_epoch}")
    print(
        "Best validation loss:",
        f"{best_validation_loss:.4f}",
    )
    print(
        "Best validation accuracy:",
        f"{best_validation_accuracy:.4f}",
    )
    print(
        "Total training time:",
        f"{total_training_seconds:.2f} seconds",
    )

    training_summary = {
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "best_validation_accuracy": float(
            best_validation_accuracy
        ),
        "total_training_seconds": float(
            total_training_seconds
        ),
    }

    return model, history, training_summary


def create_wandb_run(config, run_id, use_wandb):
    """Create a W&B run only when tracking was requested."""
    if not use_wandb:
        return None, None

    try:
        import wandb
    except ImportError as error:
        raise ImportError(
            "W&B is not installed. Run: pip install wandb"
        ) from error

    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        name=run_id,
        group=WANDB_GROUP,
        job_type="fine-tuning",
        tags=[
            "exercise-1-3",
            config["model"],
            config["strategy"],
            config["classifier_type"],
            f"batch-size-{config['batch_size']}",
        ],
        config=config,
    )

    # Batch-level metrics use the number of processed training batches.
    wandb_run.define_metric("global_step")
    wandb_run.define_metric(
        "train_batch/*",
        step_metric="global_step",
    )

    # Epoch-level metrics use the epoch number.
    wandb_run.define_metric("epoch")
    wandb_run.define_metric(
        "train/*",
        step_metric="epoch",
    )
    wandb_run.define_metric(
        "validation/*",
        step_metric="epoch",
    )
    wandb_run.define_metric(
        "timing/epoch_seconds",
        step_metric="epoch",
    )

    return wandb, wandb_run


def log_test_results_to_wandb(
        wandb_module,
        wandb_run,
        report,
        true_labels,
        predictions,
        metrics,
        checkpoint_path,
        model_name,
        strategy,
        classifier_type,
        run_id,
):
    """Log final test outputs and the best checkpoint to W&B."""
    if wandb_run is None:
        return

    report_rows = []

    for label, values in report.items():
        if not isinstance(values, dict):
            continue

        report_rows.append(
            [
                str(label),
                values["precision"],
                values["recall"],
                values["f1-score"],
                values["support"],
            ]
        )

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

    wandb_run.log(
        {
            "test/classification_report": report_table,
            "test/confusion_matrix": (
                wandb_module.plot.confusion_matrix(
                    y_true=true_labels,
                    preds=predictions,
                    class_names=[
                        str(class_id)
                        for class_id in range(
                            NUM_CLASSES
                        )
                    ],
                )
            ),
        }
    )

    for metric_name, metric_value in metrics.items():
        wandb_run.summary[metric_name] = metric_value

    artifact = wandb_module.Artifact(
        name=(
            f"fine-tuned-{model_name}-"
            f"{strategy}-{classifier_type}"
        ),
        type="model",
        metadata={
            "run_id": run_id,
            "model": model_name,
            "strategy": strategy,
            "classifier_type": classifier_type,
        },
    )
    artifact.add_file(
        local_path=str(checkpoint_path),
        name="best_model.pt",
    )
    wandb_run.log_artifact(artifact)


def run_fine_tuning(
        model_name,
        num_epochs=NUM_EPOCHS,
        strategy=DEFAULT_FINE_TUNING_STRATEGY,
        classifier_type=DEFAULT_CLASSIFIER_TYPE,
        batch_size=None,
        seed=SEED,
        use_wandb=False,
):
    """Run, save and optionally track one Exercise 1.3 experiment."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    effective_batch_size = (
        MODEL_CONFIGS[model_name]["batch_size"]
        if batch_size is None
        else batch_size
    )

    if effective_batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if num_epochs <= 0:
        raise ValueError(
            "num_epochs must be greater than zero."
        )

    if strategy not in FINE_TUNING_STRATEGIES:
        raise ValueError(
            f"Unknown fine-tuning strategy: {strategy}"
        )

    if classifier_type not in CLASSIFIER_TYPES:
        raise ValueError(
            f"Unknown classifier type: {classifier_type}"
        )

    set_random_seed(seed)

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )
    timestamp_for_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    run_id = (
        f"{timestamp_for_id}_{model_name}_"
        f"{strategy}_{classifier_type}_"
        f"bs{effective_batch_size}"
    )
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        train_loader,
        validation_loader,
        test_loader,
        device,
        dataset_info,
    ) = create_dataloaders(
        model_name=model_name,
        validation_size=VALIDATION_SIZE,
        seed=seed,
        batch_size=effective_batch_size,
    )

    model, model_info = create_fine_tuning_model(
        model_name=model_name,
        device=device,
        strategy=strategy,
        classifier_type=classifier_type,
    )

    criterion, optimizer = create_training_components(
        model=model,
        strategy=strategy,
        backbone_learning_rate=BACKBONE_LEARNING_RATE,
        classifier_learning_rate=CLASSIFIER_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config = {
        "run_id": run_id,
        "timestamp": timestamp,
        "exercise": "1.3",
        "dataset": "GTSRB",
        "model": model_name,
        "pretrained_weights": str(
            MODEL_CONFIGS[model_name]["weights"]
        ),
        "classifier_type": classifier_type,
        "strategy": strategy,
        "num_classes": NUM_CLASSES,
        "num_epochs": int(num_epochs),
        "validation_size": float(VALIDATION_SIZE),
        "batch_size": dataset_info["batch_size"],
        "log_interval": int(LOG_INTERVAL),
        "backbone_learning_rate": float(
            BACKBONE_LEARNING_RATE
        ),
        "classifier_learning_rate": float(
            CLASSIFIER_LEARNING_RATE
        ),
        "weight_decay": float(WEIGHT_DECAY),
        "optimizer": "AdamW",
        "loss_function": "CrossEntropyLoss",
        "seed": int(seed),
        "device": str(device),
        "mlp_hidden_features": (
            MLP_HIDDEN_FEATURES
            if classifier_type == "mlp"
            else None
        ),
        "mlp_dropout": (
            MLP_DROPOUT
            if classifier_type == "mlp"
            else None
        ),
        **dataset_info,
        **model_info,
    }

    save_json(
        run_dir / "config.json",
        config,
        )

    wandb_module = None
    wandb_run = None

    try:
        wandb_module, wandb_run = create_wandb_run(
            config=config,
            run_id=run_id,
            use_wandb=use_wandb,
        )

        model, history, training_summary = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            strategy=strategy,
            run_dir=run_dir,
            config=config,
            wandb_run=wandb_run,
            num_epochs=num_epochs,
        )

        print("\n=== Training history ===")

        for epoch_index in range(
                len(history["epoch"])
        ):
            print(
                f"Epoch {history['epoch'][epoch_index]}: "
                f"train loss="
                f"{history['training_loss'][epoch_index]:.4f}, "
                f"train accuracy="
                f"{history['training_accuracy'][epoch_index]:.4f}, "
                f"validation loss="
                f"{history['validation_loss'][epoch_index]:.4f}, "
                f"validation accuracy="
                f"{history['validation_accuracy'][epoch_index]:.4f}"
            )

        (
            test_loss,
            test_accuracy,
            true_labels,
            predictions,
        ) = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            return_predictions=True,
        )

        test_macro_f1 = f1_score(
            true_labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        report = classification_report(
            true_labels,
            predictions,
            labels=np.arange(NUM_CLASSES),
            digits=4,
            output_dict=True,
            zero_division=0,
        )

        printable_report = classification_report(
            true_labels,
            predictions,
            labels=np.arange(NUM_CLASSES),
            digits=4,
            zero_division=0,
        )

        metrics = {
            "best_epoch": training_summary[
                "best_epoch"
            ],
            "best_validation_loss": training_summary[
                "best_validation_loss"
            ],
            "best_validation_accuracy": training_summary[
                "best_validation_accuracy"
            ],
            "test_loss": float(test_loss),
            "test_accuracy": float(
                accuracy_score(
                    true_labels,
                    predictions,
                )
            ),
            "test_macro_f1": float(test_macro_f1),
            "total_training_seconds": training_summary[
                "total_training_seconds"
            ],
        }

        save_json(
            run_dir / "metrics.json",
            metrics,
            )
        save_classification_report(
            run_dir / "classification_report.csv",
            report,
            )
        np.savez_compressed(
            run_dir / "predictions.npz",
            true_labels=true_labels,
            predictions=predictions,
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
            "classifier_type": classifier_type,
            "strategy": strategy,
            "num_epochs": int(num_epochs),
            "best_epoch": metrics["best_epoch"],
            "best_validation_loss": metrics[
                "best_validation_loss"
            ],
            "best_validation_accuracy": metrics[
                "best_validation_accuracy"
            ],
            "test_loss": metrics["test_loss"],
            "test_accuracy": metrics["test_accuracy"],
            "test_macro_f1": metrics["test_macro_f1"],
            "total_training_seconds": metrics[
                "total_training_seconds"
            ],
            "trainable_parameters": model_info[
                "trainable_parameters"
            ],
            "wandb_run_id": wandb_run_id,
        }

        append_experiment_summary(
            summary_result
        )

        log_test_results_to_wandb(
            wandb_module=wandb_module,
            wandb_run=wandb_run,
            report=report,
            true_labels=true_labels,
            predictions=predictions,
            metrics=metrics,
            checkpoint_path=run_dir / "best_model.pt",
            model_name=model_name,
            strategy=strategy,
            classifier_type=classifier_type,
            run_id=run_id,
        )

        print("\n=== Final test results ===")
        print(f"Model: {model_name}")
        print(f"Classifier type: {classifier_type}")
        print(f"Fine-tuning strategy: {strategy}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(
            f"Test macro F1-score: "
            f"{test_macro_f1:.4f}"
        )

        print("\n=== Classification report ===")
        print(printable_report)

        print(
            "\nLocal results saved in:",
            run_dir,
        )

        return model, history

    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main():
    run_fine_tuning(
        model_name="resnet18",
        num_epochs=NUM_EPOCHS,
        strategy=DEFAULT_FINE_TUNING_STRATEGY,
        classifier_type=DEFAULT_CLASSIFIER_TYPE,
        use_wandb=False,
    )


if __name__ == "__main__":
    main()