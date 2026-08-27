"""OmegaConf loading and validation for Exercise 2."""

import sys
from pathlib import Path
from typing import Optional, Sequence

from omegaconf import DictConfig, OmegaConf


EXERCISE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXERCISE_DIR / "configs" / "default.yaml"

VALID_VALUES = {
    "model.name": {"resnet18", "resnet50"},
    "model.classifier_type": {"linear", "mlp"},
    "model.fine_tuning_strategy": {"classifier", "last_block", "full"},
    "training.loss_function": {"cross_entropy"},
    "training.optimizer": {"adamw"},
    "checkpoint.monitor": {
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
    },
    "checkpoint.mode": {"min", "max"},
}


def load_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    cli_args: Optional[Sequence[str]] = None,
) -> DictConfig:
    """Load default YAML values and merge command-line overrides."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    overrides = OmegaConf.from_cli(
        list(sys.argv[1:] if cli_args is None else cli_args)
    )
    config = OmegaConf.merge(OmegaConf.load(path), overrides)
    OmegaConf.resolve(config)
    validate_config(config)
    return config


def validate_config(config: DictConfig) -> None:
    """Validate supported values and the few relevant numeric constraints."""
    for key, allowed in VALID_VALUES.items():
        value = OmegaConf.select(config, key)
        if value not in allowed:
            raise ValueError(f"Invalid {key}={value!r}. Choose one of: {sorted(allowed)}")

    positive_values = {
        "data.batch_size": config.data.batch_size,
        "model.num_classes": config.model.num_classes,
        "model.mlp_hidden_features": config.model.mlp_hidden_features,
        "training.epochs": config.training.epochs,
        "training.backbone_learning_rate": config.training.backbone_learning_rate,
        "training.classifier_learning_rate": config.training.classifier_learning_rate,
        "logging.batch_interval": config.logging.batch_interval,
    }
    for key, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{key} must be positive.")

    if not 0.0 < config.data.validation_size < 1.0:
        raise ValueError("data.validation_size must be between 0 and 1.")
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers cannot be negative.")
    if not 0.0 <= config.model.mlp_dropout < 1.0:
        raise ValueError("model.mlp_dropout must be in [0, 1).")
    if config.training.weight_decay < 0:
        raise ValueError("training.weight_decay cannot be negative.")
    if config.experiment.smoke_test_batches < 0:
        raise ValueError("experiment.smoke_test_batches cannot be negative.")

    expected_mode = "min" if config.checkpoint.monitor == "validation_loss" else "max"
    if config.checkpoint.mode != expected_mode:
        raise ValueError(
            f"{config.checkpoint.monitor} must use checkpoint.mode={expected_mode}."
        )

    device = str(config.experiment.device)
    if not (device in {"auto", "cpu", "cuda"} or device.startswith("cuda:")):
        raise ValueError("experiment.device must be auto, cpu, cuda or cuda:<index>.")


def config_to_yaml(config: DictConfig) -> str:
    return OmegaConf.to_yaml(config, resolve=True)
